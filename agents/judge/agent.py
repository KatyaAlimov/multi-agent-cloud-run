from typing import AsyncGenerator, Literal
from google.adk.agents import Agent
from google.adk.events import Event
from google.adk.agents.invocation_context import InvocationContext
from pydantic import BaseModel, Field

MODEL = "gemini-3-flash-preview"

# 1. Define the Schema
class JudgeFeedback(BaseModel):
    """Structured feedback from the Judge agent."""
    status: Literal["pass", "fail"] = Field(
        description="Whether the research is sufficient ('pass') or needs more work ('fail')."
    )
    feedback: str = Field(
        description="Detailed feedback on what is missing. If 'pass', a brief confirmation."
    )

# 2. Define the Multilingual Judge Agent Wrapper
class MultilingualJudge(Agent):
    """A customized Judge Agent that dynamically enforces evaluation output language."""
    
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        # Extract the target language code from the session state (default to English)
        language_code = ctx.session.state.get("language", "en")
        
        # Map language codes to human-readable names for the LLM
        language_map = {
            "en": "English",
            "es": "Spanish (Español)",
            "fr": "French (Français)"
        }
        target_language = language_map.get(language_code, "English")
        
        # Dynamically append the language constraint to the judge's rule book
        language_directive = f"\n\nCRITICAL LANGUAGE ENFORCEMENT: You must write your final evaluation evaluation details and the text inside the 'feedback' property strictly in {target_language}. Do not translate the JSON keys ('status', 'feedback'), but ensure all natural language sentences are completely in {target_language}."
        
        self.instruction = """
        You are a strict editor.
        Evaluate the 'research_findings' against the user's original request.
        If the findings are missing key info, return status='fail'.
        If they are comprehensive, return status='pass'.
        """ + language_directive
        
        print(f"[{self.name}] Running evaluation loops dynamically locked to language: {target_language}")
        
        # Run the standard agent process with updated schema rules
        async for event in super()._run_async_impl(ctx):
            yield event

# Instantiate our specialized multilingual judge variant
judge = MultilingualJudge(
    name="judge",
    model=MODEL,
    description="Evaluates research findings for completeness and accuracy in the chosen language.",
    output_schema=JudgeFeedback,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)

root_agent = judge