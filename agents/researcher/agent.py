from google.adk.agents import Agent
from google.adk.tools.google_search_tool import google_search
from google.adk.agents.invocation_context import InvocationContext
from typing import AsyncGenerator
from google.adk.events import Event

MODEL = "gemini-3-flash-preview"

class MultilingualResearcher(Agent):
    """A customized Researcher Agent that dynamically enforces the target language context."""
    
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
        
        # Dynamically append the critical language directive to the base instructions
        language_directive = f"\n\nCRITICAL LANGUAGE ENFORCEMENT: You must conduct all internal reasoning, search processing, synthesis, and your final summary output strictly in {target_language}. Do not translate structural formatting keys, but all content must be natively written in {target_language}."
        
        # Inject the modified instructions directly into the running instance execution context
        self.instruction = """
        You are an expert researcher. Your goal is to find comprehensive and accurate information on the user's topic.
        Use Google Search when you need current or factual information.
        Summarize your findings clearly.
        If you receive feedback that your research is insufficient, use the feedback to refine your next search.
        Provide your research directly as text.
        """ + language_directive
        
        print(f"[{self.name}] Running search tasks dynamically locked to language: {target_language}")
        
        # Execute the core agent process with the updated instructions
        async for event in super()._run_async_impl(ctx):
            yield event

# Instantiate our specialized multilingual agent variant
researcher = MultilingualResearcher(
    name="researcher",
    model=MODEL,
    description="Gathers information on a topic using Google Search in the chosen language.",
    tools=[google_search],
)

root_agent = researcher