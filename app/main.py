import asyncio
import json
import logging
import os
import re
import subprocess
import textwrap
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx
from httpx_sse import aconnect_sse

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.cloud import texttospeech
from google.genai import types as genai_types
from opentelemetry import trace
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
from opentelemetry.sdk.trace import TracerProvider, export
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel

from authenticated_httpx import create_authenticated_client


class Feedback(BaseModel):
    score: float
    text: str | None = None
    run_id: str | None = None
    user_id: str | None = None


class SimpleChatRequest(BaseModel):
    message: str
    user_id: str = "test_user"
    session_id: Optional[str] = None


class VideoGenerationRequest(BaseModel):
    course_title: str | None = None
    section_title: str
    section_text: str


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

BASE_DIR = Path(__file__).resolve().parent
GENERATED_VIDEOS_DIR = BASE_DIR / "generated_videos"
GENERATED_VIDEOS_DIR.mkdir(exist_ok=True)


async def get_client(agent_server_origin: str) -> httpx.AsyncClient:
    global clients
    if agent_server_origin not in clients:
        clients[agent_server_origin] = create_authenticated_client(agent_server_origin)
    return clients[agent_server_origin]


async def create_session(agent_server_origin: str, agent_name: str, user_id: str) -> Dict[str, Any]:
    httpx_client = await get_client(agent_server_origin)
    headers = [
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
    headers = [
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
    headers = [
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
                "content": {
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


@app.post("/api/chat_stream")
async def chat_stream(request: SimpleChatRequest):
    """Streaming chat endpoint."""
    global agent_name, agent_server_url
    if not agent_name:
        agent_name = (await list_agents(agent_server_url))[0]  # type: ignore

    session = None
    if request.session_id:
        session = await get_session(
            agent_server_url,  # type: ignore
            agent_name,
            request.user_id,
            request.session_id
        )
    if session is None:
        session = await create_session(
            agent_server_url,  # type: ignore
            agent_name,
            request.user_id
        )

    events = query_adk_sever(
        agent_server_url,  # type: ignore
        agent_name,
        request.user_id,
        request.message,
        session["id"]
    )

    async def event_generator():
        final_text = ""
        async for event in events:
            if event["author"] == "researcher":
                yield json.dumps({"type": "progress", "text": "🔍 Researcher is gathering information..."}) + "\n"
            elif event["author"] == "judge":
                yield json.dumps({"type": "progress", "text": "⚖️ Judge is evaluating findings..."}) + "\n"
            elif event["author"] == "content_builder":
                yield json.dumps({"type": "progress", "text": "✍️ Content Builder is writing the course..."}) + "\n"

            if "content" in event and event["content"]:
                content = genai_types.Content.model_validate(event["content"])
                for part in content.parts:  # type: ignore
                    if part.text:
                        final_text += part.text

        yield json.dumps({"type": "result", "text": final_text.strip()}) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


def create_gemini_client():
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "global")

    if project:
        return genai.Client(
            vertexai=True,
            project=project,
            location=location,
        )

    return genai.Client()


def choose_target_duration_minutes(section_text: str) -> int:
    word_count = len(section_text.split())

    if word_count < 250:
        return 3
    if word_count < 600:
        return 4
    if word_count < 1000:
        return 5

    return 6


def build_lesson_video_prompt(request: VideoGenerationRequest, target_minutes: int) -> str:
    course_title = request.course_title or "Generated Course"
    target_seconds = target_minutes * 60
    target_words = target_minutes * 130

    return f"""
Create a structured scene plan for a narrated slide-based educational video lesson.

Target video length: {target_minutes} minutes, about {target_seconds} seconds.
Target narration length: about {target_words} spoken words total.

Course title: {course_title}
Section title: {request.section_title}

Section content:
{request.section_text}

Return ONLY valid JSON. Do not include markdown fences.

Use this exact structure:
{{
  "title": "Video lesson title",
  "target_duration_minutes": {target_minutes},
  "summary": "Short summary of the lesson",
  "scenes": [
    {{
      "title": "Scene title",
      "duration_seconds": 30,
      "narration": "Detailed spoken narration for this scene. Write this as a complete voiceover script.",
      "visual": "Description of what the slide should show.",
      "onscreen_text": ["Short bullet 1", "Short bullet 2", "Short bullet 3"]
    }}
  ]
}}

Rules:
- Create enough scenes to fill about {target_minutes} minutes.
- Each scene should be 20 to 45 seconds.
- The total scene durations should be close to {target_seconds} seconds.
- The total narration should be close to {target_words} words.
- Focus only on this section, not the entire course.
- Make it feel like a real course lesson, not a short summary.
- Keep onscreen_text short and readable.
- Write narration in a natural teacher voice.
"""


def extract_json(text: str) -> Dict[str, Any]:
    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def normalize_scene_durations(scenes: List[Dict[str, Any]], target_seconds: int) -> List[Dict[str, Any]]:
    if not scenes:
        raise ValueError("Gemini did not return any scenes.")

    current_total = sum(max(10, int(scene.get("duration_seconds", 30))) for scene in scenes)

    if current_total <= 0:
        per_scene = max(10, target_seconds // len(scenes))
        for scene in scenes:
            scene["duration_seconds"] = per_scene
        return scenes

    scale = target_seconds / current_total

    for scene in scenes:
        original = max(10, int(scene.get("duration_seconds", 30)))
        scene["duration_seconds"] = max(10, int(original * scale))

    return scenes


def get_font(size: int):
    font_candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]

    for font_path in font_candidates:
        if os.path.exists(font_path):
            return ImageFont.truetype(font_path, size=size)

    return ImageFont.load_default()


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    position: tuple[int, int],
    font,
    fill: str,
    max_width_chars: int,
    line_spacing: int = 8,
):
    x, y = position
    lines = []

    for paragraph in text.split("\n"):
        wrapped = textwrap.wrap(paragraph, width=max_width_chars)
        lines.extend(wrapped or [""])

    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        bbox = draw.textbbox((x, y), line, font=font)
        y += (bbox[3] - bbox[1]) + line_spacing

    return y


def create_slide(scene: Dict[str, Any], index: int, total: int, output_path: Path):
    width, height = 1280, 720
    image = Image.new("RGB", (width, height), color="#0f172a")
    draw = ImageDraw.Draw(image)

    title_font = get_font(44)
    body_font = get_font(30)
    small_font = get_font(22)

    draw.rectangle((0, 0, width, 90), fill="#2563eb")
    draw.text((48, 24), f"Scene {index + 1} of {total}", font=small_font, fill="#dbeafe")

    title = str(scene.get("title", "Lesson Scene"))
    draw_wrapped_text(draw, title, (48, 120), title_font, "#ffffff", max_width_chars=34)

    onscreen_text = scene.get("onscreen_text", [])
    if not isinstance(onscreen_text, list):
        onscreen_text = [str(onscreen_text)]

    y = 245
    for item in onscreen_text[:5]:
        draw.text((72, y), "•", font=body_font, fill="#60a5fa")
        y = draw_wrapped_text(
            draw,
            str(item),
            (112, y),
            body_font,
            "#e2e8f0",
            max_width_chars=44,
            line_spacing=10,
        )
        y += 10

    visual = str(scene.get("visual", ""))
    if visual:
        draw.rectangle((48, 560, 1232, 670), outline="#334155", width=2)
        draw_wrapped_text(
            draw,
            f"Visual direction: {visual}",
            (72, 585),
            small_font,
            "#cbd5e1",
            max_width_chars=95,
            line_spacing=6,
        )

    image.save(output_path)


def synthesize_narration(text: str, output_path: Path):
    client = texttospeech.TextToSpeechClient()

    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Neural2-F",
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=1.0,
    )

    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config,
    )

    output_path.write_bytes(response.audio_content)


def create_video_from_slides(scenes: List[Dict[str, Any]], work_dir: Path, output_path: Path):
    scene_video_paths = []

    for index, scene in enumerate(scenes):
        slide_path = work_dir / f"slide_{index:03d}.png"
        audio_path = work_dir / f"audio_{index:03d}.mp3"
        scene_video_path = work_dir / f"scene_{index:03d}.mp4"

        create_slide(scene, index, len(scenes), slide_path)

        narration = str(scene.get("narration", "")).strip()
        if not narration:
            narration = str(scene.get("title", "Lesson scene"))

        synthesize_narration(narration, audio_path)

        command = [
            "ffmpeg",
            "-y",
            "-loop",
            "1",
            "-i",
            str(slide_path),
            "-i",
            str(audio_path),
            "-c:v",
            "libx264",
            "-tune",
            "stillimage",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-pix_fmt",
            "yuv420p",
            "-shortest",
            str(scene_video_path),
        ]

        subprocess.run(command, check=True, capture_output=True, text=True)
        scene_video_paths.append(scene_video_path)

    concat_file = work_dir / "scene_videos.txt"

    with concat_file.open("w", encoding="utf-8") as file:
        for scene_video_path in scene_video_paths:
            file.write(f"file '{scene_video_path}'\n")

    concat_command = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-c",
        "copy",
        str(output_path),
    ]

    subprocess.run(concat_command, check=True, capture_output=True, text=True)


def generate_lesson_video_file(request: VideoGenerationRequest) -> Dict[str, Any]:
    target_minutes = choose_target_duration_minutes(request.section_text)
    target_seconds = target_minutes * 60

    client = create_gemini_client()
    response = client.models.generate_content(
        model=os.getenv("SCRIPT_MODEL", "gemini-3-flash-preview"),
        contents=build_lesson_video_prompt(request, target_minutes),
    )

    plan = extract_json(response.text or "{}")
    scenes = normalize_scene_durations(plan.get("scenes", []), target_seconds)

    video_id = str(uuid.uuid4())
    work_dir = GENERATED_VIDEOS_DIR / video_id
    work_dir.mkdir(parents=True, exist_ok=True)

    output_path = GENERATED_VIDEOS_DIR / f"{video_id}.mp4"
    create_video_from_slides(scenes, work_dir, output_path)

    script_markdown = f"# {plan.get('title', request.section_title)}\n\n"
    script_markdown += f"Target length: {target_minutes} minutes\n\n"
    script_markdown += f"## Summary\n\n{plan.get('summary', '')}\n\n"
    script_markdown += "## Scene Plan\n\n"

    for index, scene in enumerate(scenes, start=1):
        script_markdown += f"### Scene {index}: {scene.get('title', 'Untitled')}\n\n"
        script_markdown += f"Duration: {scene.get('duration_seconds', 0)} seconds\n\n"
        script_markdown += f"**Narration:** {scene.get('narration', '')}\n\n"
        script_markdown += f"**Visual:** {scene.get('visual', '')}\n\n"

    return {
        "video_url": f"/generated_videos/{output_path.name}",
        "target_duration_minutes": target_minutes,
        "script_markdown": script_markdown,
    }


@app.post("/api/generate_lesson_video")
async def generate_lesson_video(request: VideoGenerationRequest):
    if not request.section_title.strip():
        raise HTTPException(status_code=400, detail="section_title is required")

    if not request.section_text.strip():
        raise HTTPException(status_code=400, detail="section_text is required")

    try:
        return await asyncio.to_thread(generate_lesson_video_file, request)
    except subprocess.CalledProcessError as error:
        logger.exception("ffmpeg failed")
        raise HTTPException(
            status_code=500,
            detail=f"ffmpeg failed: {error.stderr or error.stdout or error}",
        ) from error
    except Exception as error:
        logger.exception("Lesson video generation failed")
        raise HTTPException(status_code=500, detail=str(error)) from error


# Serve generated lesson videos before mounting the frontend.
app.mount(
    "/generated_videos",
    StaticFiles(directory=GENERATED_VIDEOS_DIR),
    name="generated_videos",
)

# Mount frontend from the copied location
frontend_path = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))