"""Model-rotating Gemini wrapper for evaluation under tight per-model RPD caps.
Tries the pinned model first; on rate-limit exhaustion, rotates through the overflow pool. Logs every call's model."""
import time
from google import genai
from google.genai import types


class RotatingGeminiLLM:
    def __init__(self, api_key, pinned_model, overflow_pool=None, per_model_retries=2):
        self.client = genai.Client(api_key=api_key)
        self.pinned = pinned_model
        # ordered attempt sequence: pinned first, then overflow (dedup, keep order)
        self.pool = [pinned_model] + [m for m in (overflow_pool or []) if m != pinned_model]
        self.per_model_retries = per_model_retries
        self.call_log = []                                  # [(model_used, ok/err)] for transparency

    def _generate(self, contents, config):
        last_err = None
        for model in self.pool:                             # walk pinned -> overflow
            for attempt in range(self.per_model_retries):
                try:
                    resp = self.client.models.generate_content(model=model, contents=contents, config=config)
                    self.call_log.append((model, "ok"))
                    return resp, model
                except Exception as e:
                    last_err = e
                    msg = str(e)
                    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                        if "PerMinute" in msg or "per minute" in msg.lower():
                            time.sleep(2 ** attempt); continue     # RPM: short wait, retry SAME model
                        break                                       # RPD/daily: rotate to next model
                    time.sleep(2 ** attempt)                        # other transient error: brief backoff, retry
            # exhausted this model -> next in pool
        self.call_log.append(("NONE", "exhausted"))
        raise RuntimeError(f"all models exhausted. last error: {last_err}")

    def complete(self, prompt, system=None, temperature=0.0):
        cfg = types.GenerateContentConfig(system_instruction=system, temperature=temperature)
        resp, _ = self._generate(prompt, cfg)
        return resp.text

    def structured(self, prompt, schema, system=None, temperature=0.0):
        cfg = types.GenerateContentConfig(response_mime_type="application/json",
                                          response_schema=schema, system_instruction=system, temperature=temperature)
        resp, _ = self._generate(prompt, cfg)
        return resp.parsed

    def model_usage(self):                                   # summary: how many calls each model served
        from collections import Counter
        return dict(Counter(m for m, _ in self.call_log))
