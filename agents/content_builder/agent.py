from typing import AsyncGenerator
from google.adk.agents import Agent
from google.adk.events import Event
from google.adk.agents.invocation_context import InvocationContext

MODEL = "gemini-3-flash-preview"

# Define the Multilingual Content Builder Agent Wrapper
class MultilingualContentBuilder(Agent):
    """A customized Content Builder Agent that dynamically enforces course output language."""
    
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
        
        # Dynamically append the strict language constraint to the formatting rules
        language_directive = f"\n\nCRITICAL LANGUAGE ENFORCEMENT: You must write the entire course module content, all headings, summaries, explanations, and video narrative scripts strictly in {target_language}. Do not write any part of the course materials in English unless explicitly requested by the topic."
        
        self.instruction = """
        You are an expert course creator.
        Take the approved 'research_findings' and transform them into a well-structured, engaging course module.

        **Formatting Rules:**
        1. Start with a main title using a single `#` (H1).
        2. Use `##` (H2) for main section headings.
        3. Use bullet points and clear paragraphs.
        4. Maintain a professional but engaging tone.

        Ensure the content directly addresses the user's original request.
        """ + language_directive
        
        print(f"[{self.name}] Building course modules dynamically locked to language: {target_language}")
        
        # Run the standard agent generation process with updated language context instructions
        async for event in super()._run_async_impl(ctx):
            yield event

# Instantiate our specialized multilingual content builder variant
content_builder = MultilingualContentBuilder(
    name="content_builder",
    model=MODEL,
    description="Transforms research findings into a structured course in the chosen language.",
)

root_agent = content_builder