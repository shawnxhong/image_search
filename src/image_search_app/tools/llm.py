"""OpenVINO GenAI LLMPipeline wrapper for local LLM inference."""

from __future__ import annotations

import gc
import json
import logging
import re
import threading
from pathlib import Path

from image_search_app.config import settings

logger = logging.getLogger(__name__)

# Project root (where pyproject.toml lives)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def build_tool_system_prompt(base_prompt: str, tools: list[dict]) -> str:
    """Append tool descriptions to the system prompt in a format
    the LLM understands for ReAct-style tool calling.
    """
    tool_lines = [
        base_prompt,
        "",
        "You can call tools using the following format:",
        "Action: tool_name",
        "Action Input: {json}",
        "",
        "Available tools:",
    ]
    for tool in tools:
        func = tool.get("function", tool)
        name = func.get("name", "")
        description = func.get("description", "")
        params = func.get("parameters", {})
        props_desc = params.get("properties", {})
        args_info = {k: v.get("description", "") for k, v in props_desc.items()}
        tool_lines.append(f"- {name}: {description} Args: {json.dumps(args_info)}")
    tool_lines.append("")
    tool_lines.append(
        "When you have gathered enough results and are ready to finish, "
        "respond to the user WITHOUT Action/Action Input tags. "
        "Just say DONE."
    )
    return "\n".join(tool_lines)


def parse_tool_requests(text: str) -> list[dict]:
    """Parse all tool calls from LLM output.

    Supports both Qwen3 <tool_call> XML format and ReAct Action/Action Input format.
    Returns a list of {"name": ..., "args": ...} dicts.
    """
    calls: list[dict] = []

    # Qwen3 <tool_call> XML format — find ALL matches
    tc_matches = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
    if tc_matches:
        for match in tc_matches:
            try:
                parsed = json.loads(match)
                calls.append({"name": parsed.get("name", ""), "args": parsed.get("arguments", {})})
            except json.JSONDecodeError:
                pass
        if calls:
            return calls

    # ReAct format: Action: name\nAction Input: {json}
    match = re.search(
        r"Action\s*:\s*(?P<name>[\w\-]+)\s*Action Input\s*:\s*(?P<input>.+)",
        text,
        re.S,
    )
    if not match:
        return []
    name = match.group("name").strip()
    raw_input = match.group("input").strip()
    # Clean up trailing text
    for sep in ["\\nFinal:", "\\nObservation:", "\nFinal:", "\nObservation:", "\nAction:"]:
        raw_input = raw_input.split(sep, 1)[0].strip()
    if raw_input.startswith("```"):
        raw_input = raw_input.strip("`").strip()
    try:
        args = json.loads(raw_input)
    except json.JSONDecodeError:
        match_json = re.search(r"(\{.*\})", raw_input, re.S)
        if match_json:
            try:
                args = json.loads(match_json.group(1))
            except json.JSONDecodeError:
                args = {"input": raw_input}
        else:
            args = {"input": raw_input}
    return [{"name": name, "args": args}]


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from Qwen3 output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def scan_available_models(models_dir: str | None = None) -> list[str]:
    """Scan a directory for OpenVINO model subdirectories.

    A valid model directory contains an .xml file (the IR model).
    Returns a list of directory names (not full paths).
    """
    base = Path(models_dir or settings.llm_models_dir)
    if not base.is_absolute():
        base = _PROJECT_ROOT / base
    if not base.is_dir():
        return []
    models = []
    for child in sorted(base.iterdir()):
        if child.is_dir():
            # Check if it looks like an OV GenAI model (has .xml or .bin files)
            has_model_files = any(
                child.glob("*.xml")
            ) or any(
                child.glob("*.bin")
            ) or (child / "openvino_model.xml").exists()
            if has_model_files:
                models.append(child.name)
    return models


class LLMService:
    """Thread-safe wrapper around openvino_genai.LLMPipeline."""

    def __init__(self) -> None:
        self._pipeline = None  # ov_genai.LLMPipeline, lazy loaded
        self._lock = threading.Lock()
        self._model_path: str | None = None
        self._device: str | None = None

    def load(self, model_path: str | None = None, device: str | None = None) -> None:
        """Load (or reload) the LLM pipeline with the given model and device."""
        with self._lock:
            # Unload existing pipeline first
            if self._pipeline is not None:
                self._release_pipeline()

            import openvino_genai as ov_genai

            raw_path = Path(model_path or settings.llm_model_path)
            if not raw_path.is_absolute():
                raw_path = _PROJECT_ROOT / raw_path
            resolved_path = str(raw_path)
            resolved_device = device or settings.llm_device

            logger.info("Loading LLM from %s on %s", resolved_path, resolved_device)
            self._pipeline = ov_genai.LLMPipeline(resolved_path, resolved_device)
            self._model_path = resolved_path
            self._device = resolved_device
            logger.info("LLM loaded successfully")

    def unload(self) -> None:
        """Release the LLM pipeline and free memory."""
        with self._lock:
            if self._pipeline is not None:
                self._release_pipeline()
                logger.info("LLM unloaded")

    def _release_pipeline(self) -> None:
        """Internal: delete pipeline and run GC. Caller must hold _lock."""
        del self._pipeline
        self._pipeline = None
        self._model_path = None
        self._device = None
        gc.collect()

    def status(self) -> dict:
        """Return current LLM status."""
        if self._pipeline is not None:
            return {
                "loaded": True,
                "model_path": self._model_path,
                "model_name": Path(self._model_path).name if self._model_path else None,
                "device": self._device,
            }
        return {"loaded": False, "model_path": None, "model_name": None, "device": None}

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> str:
        """Generate a response given chat messages.

        Uses the tokenizer's apply_chat_template to format messages
        with proper Qwen3 tool-calling support.

        Raises RuntimeError if the LLM is not loaded.
        Returns the raw generated text.
        """
        if self._pipeline is None:
            raise RuntimeError(
                "LLM is not loaded. Please load a model first via the LLM Control panel."
            )

        import openvino_genai as ov_genai

        if tools:
            # Use the native Qwen3 tool-calling chat template
            prompt = self._pipeline.get_tokenizer().apply_chat_template(
                messages, add_generation_prompt=True, tools=tools,
            )
        else:
            # Standard chat template without tools
            prompt = self._pipeline.get_tokenizer().apply_chat_template(
                messages, add_generation_prompt=True,
            )

        config = ov_genai.GenerationConfig()
        config.max_new_tokens = 2048
        config.do_sample = False

        result = self._pipeline.generate(prompt, config)
        if isinstance(result, str):
            return result
        return getattr(result, "text", None) or str(result)


# Module-level singleton
_llm_service: LLMService | None = None
_llm_lock = threading.Lock()


def get_llm_service() -> LLMService:
    global _llm_service
    if _llm_service is None:
        with _llm_lock:
            if _llm_service is None:
                _llm_service = LLMService()
    return _llm_service
