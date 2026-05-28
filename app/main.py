import logging
import os
import json
import asyncio
import base64
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx
from httpx_sse import aconnect_sse

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types as genai_types
from opentelemetry import trace
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
from opentelemetry.sdk.trace import TracerProvider, export
from pydantic import BaseModel

from authenticated_httpx import create_authenticated_client

class Feedback(BaseModel):
    score: float
    text: str | None = None
    run_id: str | None = None
    user_id: str | None = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

provider = TracerProvider()
processor = export.BatchSpanProcessor(
    CloudTraceSpanExporter(),
)
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent_name = os.getenv("AGENT_NAME", None)
agent_server_url = os.getenv("AGENT_SERVER_URL")
if not agent_server_url:
    raise ValueError("AGENT_SERVER_URL environment variable not set")
else:
    agent_server_url = agent_server_url.rstrip("/")

clients: Dict[str, httpx.AsyncClient] = {}

async def get_client(agent_server_origin: str) -> httpx.AsyncClient:
    global clients
    if agent_server_origin not in clients:
        clients[agent_server_origin] = create_authenticated_client(agent_server_origin)
    return clients[agent_server_origin]

async def create_session(agent_server_origin: str, agent_name: str, user_id: str) -> Dict[str, Any]:
    httpx_client = await get_client(agent_server_origin)
    headers=[
        ("Content-Type", "application/json")
    ]
    session_request_url = f"{agent_server_origin}/apps/{agent_name}/users/{user_id}/sessions"
    session_response = await httpx_client.post(
        session_request_url,
        headers=headers
    )
    session_response.raise_for_status()
    return session_response.json()

async def get_session(agent_server_origin: str, agent_name: str, user_id: str, session_id: str) -> Optional[Dict[str, Any]]:
    httpx_client = await get_client(agent_server_origin)
    headers=[
        ("Content-Type", "application/json")
    ]
    session_request_url = f"{agent_server_origin}/apps/{agent_name}/users/{user_id}/sessions/{session_id}"
    session_response = await httpx_client.get(
        session_request_url,
        headers=headers
    )
    if session_response.status_code == 404:
        return None
    session_response.raise_for_status()
    return session_response.json()


async def list_agents(agent_server_origin: str) -> List[str]:
    httpx_client = await get_client(agent_server_origin)
    headers=[
        ("Content-Type", "application/json")
    ]
    list_url = f"{agent_server_origin}/list-apps"
    list_response = await httpx_client.get(
        list_url,
        headers=headers
    )
    list_response.raise_for_status()
    agent_list = list_response.json()
    if not agent_list:
        agent_list = ["agent"]
    return agent_list


async def query_adk_sever(
        agent_server_origin: str, agent_name: str, user_id: str, message: str, session_id
) -> AsyncGenerator[Dict[str, Any], None]:
    httpx_client = await get_client(agent_server_origin)
    request = {
        "appName": agent_name,
        "userId": user_id,
        "sessionId": session_id,
        "newMessage": {
            "role": "user",
            "parts": [{"text": message}]
        },
        "streaming": False
    }
    async with aconnect_sse(
        httpx_client,
        "POST",
        f"{agent_server_origin}/run_sse",
        json=request
    ) as event_source:
        if event_source.response.is_error:
            event = {
                "author": agent_name,
                "content":{
                    "parts": [
                        {
                            "text": f"Error {event_source.response.text}"
                        }
                    ]
                }
            }
            yield event
        else:
            async for server_event in event_source.aiter_sse():
                event = server_event.json()
                yield event

class SimpleChatRequest(BaseModel):
    message: str
    user_id: str = "test_user"
    session_id: Optional[str] = None

class VideoGenerationRequest(BaseModel):
    course_title: str | None = None
    section_title: str
    section_text: str
    duration_seconds: int = 8

@app.post("/api/chat_stream")
async def chat_stream(request: SimpleChatRequest):
    """Streaming chat endpoint."""
    global agent_name, agent_server_url
    if not agent_name:
        agent_name = (await list_agents(agent_server_url))[0] # type: ignore

    session = None
    if request.session_id:
        session = await get_session(
            agent_server_url, # type: ignore
            agent_name,
            request.user_id,
            request.session_id
        )
    if session is None:
        session = await create_session(
            agent_server_url, # type: ignore
            agent_name,
            request.user_id
        )

    events = query_adk_sever(
        agent_server_url, # type: ignore
        agent_name,
        request.user_id,
        request.message,
        session["id"]
    )

    async def event_generator():
        final_text = ""
        async for event in events:
            # Send progress updates based on which agent is active
            if event["author"] == "researcher":
                 yield json.dumps({"type": "progress", "text": "🔍 Researcher is gathering information..."}) + "\n"
            elif event["author"] == "judge":
                 yield json.dumps({"type": "progress", "text": "⚖️ Judge is evaluating findings..."}) + "\n"
            elif event["author"] == "content_builder":
                 yield json.dumps({"type": "progress", "text": "✍️ Content Builder is writing the course..."}) + "\n"
            # Accumulate final text
            if "content" in event and event["content"]:
                content = genai_types.Content.model_validate(event["content"])
                for part in content.parts: # type: ignore
                    if part.text:
                        final_text += part.text
        # Send final result
        yield json.dumps({"type": "result", "text": final_text.strip()}) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")

def create_video_client():
    use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "false").lower() == "true"
    if not use_vertex:
        return genai.Client()

    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise ValueError("GOOGLE_CLOUD_PROJECT environment variable not set")

    location = os.getenv("VIDEO_GOOGLE_CLOUD_LOCATION")
    if not location:
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        if location == "global":
            location = "us-central1"

    return genai.Client(
        vertexai=True,
        project=project,
        location=location,
    )

def build_video_prompt(request: VideoGenerationRequest) -> str:
    course_title = request.course_title or "Untitled course"
    section_text = request.section_text.strip()[:5000]

    return f"""
Create a high-quality educational explainer video for one section of an online course.

Course title: {course_title}
Section title: {request.section_title}
Section content:
{section_text}

Requirements:
- Focus only on this section, not the entire course.
- Use clear instructional visuals, smooth pacing, and a polished professional style.
- Show concepts with diagrams, examples, animations, and realistic educational scenes where useful.
- Keep text overlays short and readable.
- Make the video useful as a standalone learning aid for this module.
"""

def refresh_video_operation(client, operation):
    try:
        return client.operations.get(operation)
    except TypeError:
        operation_name = getattr(operation, "name", operation)
        return client.operations.get(operation_name)

def generate_section_video(request: VideoGenerationRequest) -> Dict[str, str]:
    prompt = build_video_prompt(request)
    duration_seconds = min(max(request.duration_seconds, 4), 8)
    model = os.getenv("VIDEO_MODEL", "veo-3.1-generate-preview")
    poll_interval = int(os.getenv("VIDEO_POLL_INTERVAL_SECONDS", "10"))
    max_wait = int(os.getenv("VIDEO_MAX_WAIT_SECONDS", "300"))

    client = create_video_client()
    operation = client.models.generate_videos(
        model=model,
        prompt=prompt,
        config=genai_types.GenerateVideosConfig(
            number_of_videos=1,
            duration_seconds=duration_seconds,
            aspect_ratio="16:9",
            enhance_prompt=True,
        ),
    )

    deadline = time.monotonic() + max_wait
    while not operation.done:
        if time.monotonic() >= deadline:
            raise TimeoutError("Video generation timed out. Please try again.")
        time.sleep(poll_interval)
        operation = refresh_video_operation(client, operation)

    if operation.error:
        raise RuntimeError(f"Video generation failed: {operation.error}")

    response = getattr(operation, "response", None) or getattr(operation, "result", None)
    if not response or not response.generated_videos:
        raise RuntimeError("Video generation completed without a video.")

    generated_video = response.generated_videos[0]
    video = generated_video.video
    video_uri = getattr(video, "uri", None)
    mime_type = getattr(video, "mime_type", None) or "video/mp4"

    try:
        downloaded = client.files.download(file=video)
        video_bytes = getattr(video, "video_bytes", None) or downloaded
    except Exception as exc:
        if video_uri:
            logger.warning("Returning generated video URI because download failed: %s", exc)
            return {
                "video_uri": video_uri,
                "mime_type": mime_type,
                "prompt": prompt,
            }
        raise

    if not video_bytes:
        if video_uri:
            return {
                "video_uri": video_uri,
                "mime_type": mime_type,
                "prompt": prompt,
            }
        raise RuntimeError("Generated video did not include downloadable bytes.")

    encoded_video = base64.b64encode(video_bytes).decode("ascii")
    return {
        "video_data_url": f"data:{mime_type};base64,{encoded_video}",
        "mime_type": mime_type,
        "prompt": prompt,
    }

@app.post("/api/generate_video")
async def generate_video(request: VideoGenerationRequest):
    """Generate a short video for a single course section."""
    if not request.section_title.strip():
        raise HTTPException(status_code=400, detail="section_title is required")
    if not request.section_text.strip():
        raise HTTPException(status_code=400, detail="section_text is required")

    try:
        return await asyncio.to_thread(generate_section_video, request)
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Video generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

# Mount frontend from the copied location
frontend_path = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
