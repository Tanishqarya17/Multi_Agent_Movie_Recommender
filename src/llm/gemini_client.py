"""Reusable Gemini wrapper for all agents: text + structured (Pydantic) output, with retry."""
import time
from google import genai
from google.genai import types


class GeminiLLM:
    def __init__(self, api_key, model="gemini-3.6-flash", max_retries=5):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.max_retries = max_retries

    def _call(self, contents, config=None):
        for attempt in range(self.max_retries):
            try:
                return self.client.models.generate_content(
                    model=self.model, contents=contents, config=config)
            except Exception as e:
                if "429" in str(e) and attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise
        raise RuntimeError("exhausted retries")

    def complete(self, prompt, system=None, temperature=0.0):
        cfg = types.GenerateContentConfig(system_instruction=system, temperature=temperature)
        return self._call(prompt, cfg).text

    def structured(self, prompt, schema, system=None, temperature=0.0):
        cfg = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            system_instruction=system,
            temperature=temperature,
        )
        return self._call(prompt, cfg).parsed
