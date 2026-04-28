"""Loaders for real agent trace formats.

Supported formats:
  - AutoGPT (JSON log files)
  - Open Interpreter (JSON session files)
  - SWE-bench (JSONL trajectory files)

Each loader extracts a list of action-type strings from raw log files,
suitable for feeding into run_method_on_trace().
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator

# ── AutoGPT ────────────────────────────────────────────────────────────────────

_AUTOGPT_COMMAND_MAP: dict[str, str] = {
    "write_to_file": "write_file",
    "read_file": "read_file",
    "append_to_file": "write_file",
    "search": "web_search",
    "google": "web_search",
    "browse_website": "web_browse",
    "execute_python_file": "run_code",
    "execute_shell": "run_shell",
    "execute_shell_popen": "run_shell",
    "send_tweet": "post_message",
    "send_email": "send_email",
    "task_complete": "task_complete",
    "finish": "task_complete",
    "do_nothing": "no_op",
    "list_files": "list_files",
    "delete_file": "delete_file",
    "clone_repository": "git_clone",
    "memory_add": "memory_write",
    "memory_delete": "memory_write",
    "memory_clear": "memory_write",
    "memory_retrieve": "memory_read",
    "analyze_code": "analyze_code",
    "improve_code": "write_code",
    "write_tests": "write_code",
    "ask_user": "ask_user",
    "create_file_in_folder": "write_file",
    "web_search": "web_search",
    "run_script": "run_code",
}


class AutoGPTLoader:
    """Load action traces from AutoGPT execution log files.

    AutoGPT logs each step as a JSON object with a "command" field:
    {"thoughts": {...}, "command": {"name": str, "args": {...}}}

    Args:
        path: Path to log file or directory containing .json log files.
        normalize: If True, map raw command names to canonical action types.
    """

    def __init__(self, path: str | Path, normalize: bool = True) -> None:
        self.path = Path(path)
        self.normalize = normalize

    def load(self) -> list[str]:
        """Load and return a flat list of action type strings."""
        actions: list[str] = []
        for record in self._iter_records():
            cmd = record.get("command", {})
            name = cmd.get("name", "") if isinstance(cmd, dict) else str(cmd)
            if name:
                actions.append(self._map(name))
        return actions

    def _map(self, name: str) -> str:
        if not self.normalize:
            return name
        return _AUTOGPT_COMMAND_MAP.get(name.lower(), name.lower())

    def _iter_records(self) -> Iterator[dict]:
        files = [self.path] if self.path.is_file() else sorted(self.path.glob("**/*.json"))
        for f in files:
            try:
                with open(f) as fh:
                    data = json.load(fh)
                if isinstance(data, list):
                    yield from data
                elif isinstance(data, dict):
                    yield data
            except (json.JSONDecodeError, OSError):
                continue

    @staticmethod
    def from_string(json_text: str, normalize: bool = True) -> list[str]:
        """Load from a JSON string (for testing without a file)."""
        loader = AutoGPTLoader.__new__(AutoGPTLoader)
        loader.normalize = normalize
        records = json.loads(json_text)
        if isinstance(records, dict):
            records = [records]
        actions: list[str] = []
        for record in records:
            cmd = record.get("command", {})
            name = cmd.get("name", "") if isinstance(cmd, dict) else str(cmd)
            if name:
                actions.append(loader._map(name))
        return actions


# ── Open Interpreter ───────────────────────────────────────────────────────────

_OI_TYPE_MAP: dict[str, str] = {
    "code": "run_code",
    "shell": "run_shell",
    "python": "run_code",
    "javascript": "run_code",
    "applescript": "run_code",
    "r": "run_code",
    "html": "run_code",
    "console": "run_code",
    "file": "read_file",
    "browser": "web_browse",
    "search": "web_search",
    "message": "send_message",
    "computer": "computer_control",
}


class OpenInterpreterLoader:
    """Load action traces from Open Interpreter session files.

    Open Interpreter sessions are JSON arrays of messages. Each assistant message
    may include a "code" block indicating an execution action.

    Args:
        path: Path to .json session file or directory.
        normalize: If True, map code language/type to canonical action name.
    """

    def __init__(self, path: str | Path, normalize: bool = True) -> None:
        self.path = Path(path)
        self.normalize = normalize

    def load(self) -> list[str]:
        actions: list[str] = []
        for record in self._iter_records():
            if record.get("role") != "assistant":
                continue
            content = record.get("content", [])
            if isinstance(content, str):
                actions.append("send_message")
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type", "")
                lang = block.get("language", block.get("format", ""))
                raw = lang.lower() if lang else btype.lower()
                actions.append(_OI_TYPE_MAP.get(raw, raw) if self.normalize else raw)
        return actions

    def _iter_records(self) -> Iterator[dict]:
        files = [self.path] if self.path.is_file() else sorted(self.path.glob("**/*.json"))
        for f in files:
            try:
                with open(f) as fh:
                    data = json.load(fh)
                if isinstance(data, list):
                    yield from data
                elif isinstance(data, dict):
                    yield data
            except (json.JSONDecodeError, OSError):
                continue


# ── SWE-bench ──────────────────────────────────────────────────────────────────

_SWEB_TOOL_MAP: dict[str, str] = {
    "str_replace_editor": "edit_file",
    "str_replace_based_edit_tool": "edit_file",
    "view": "read_file",
    "create": "write_file",
    "edit": "edit_file",
    "execute_bash": "run_shell",
    "bash": "run_shell",
    "python": "run_code",
    "search_files": "search_code",
    "find_file": "search_code",
    "finish": "task_complete",
    "submit": "task_complete",
}


class SWEBenchLoader:
    """Load action traces from SWE-bench trajectory files (JSONL).

    SWE-bench trajectories are JSONL files where each line is a JSON object
    with a "messages" field containing agent messages with tool_calls.

    Args:
        path: Path to .jsonl trajectory file or directory.
        normalize: If True, map tool names to canonical action types.
    """

    def __init__(self, path: str | Path, normalize: bool = True) -> None:
        self.path = Path(path)
        self.normalize = normalize

    def load(self) -> list[str]:
        actions: list[str] = []
        for step in self._iter_steps():
            tool_name = step.get("tool", step.get("action", step.get("name", "")))
            if tool_name:
                actions.append(_SWEB_TOOL_MAP.get(tool_name.lower(), tool_name.lower())
                                if self.normalize else tool_name)
        return actions

    def _iter_steps(self) -> Iterator[dict]:
        files = [self.path] if self.path.is_file() else sorted(self.path.glob("**/*.jsonl"))
        for f in files:
            try:
                with open(f) as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        msgs = record.get("messages", [record])
                        for msg in msgs:
                            for tc in msg.get("tool_calls", []):
                                fn = tc.get("function", tc)
                                name = fn.get("name", "")
                                if name:
                                    yield {"tool": name}
            except OSError:
                continue
