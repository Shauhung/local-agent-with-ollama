from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = (PROJECT_ROOT / "agent_workspace").resolve()
GENERATED_TOOLS_ROOT = (PROJECT_ROOT / "generated_tools").resolve()

ALLOWED_COMMANDS = {
    "ls",
    "pwd",
    "python",
    "python3",
    "pytest",
}

mcp = FastMCP("local-agent-tools")


def ensure_directories() -> None:
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    GENERATED_TOOLS_ROOT.mkdir(parents=True, exist_ok=True)
    (GENERATED_TOOLS_ROOT / "tests").mkdir(parents=True, exist_ok=True)


def resolve_workspace_path(relative_path: str) -> Path:
    path = (WORKSPACE_ROOT / relative_path).resolve()
    if not path.is_relative_to(WORKSPACE_ROOT):
        raise ValueError(f"Path is outside agent workspace: {relative_path}")
    return path


def resolve_generated_tool_path(relative_path: str) -> Path:
    path = (GENERATED_TOOLS_ROOT / relative_path).resolve()
    if not path.is_relative_to(GENERATED_TOOLS_ROOT):
        raise ValueError(f"Path is outside generated tools workspace: {relative_path}")
    return path


def compact_result(stdout: str, stderr: str, returncode: int) -> dict[str, Any]:
    return {
        "returncode": returncode,
        "stdout": stdout[-8000:],
        "stderr": stderr[-8000:],
    }


@mcp.tool()
def get_time() -> str:
    """Return the current local date and time."""
    return datetime.now().isoformat(timespec="seconds")


@mcp.tool()
def list_files() -> list[str]:
    """List files inside the agent workspace."""
    ensure_directories()
    files: list[str] = []
    for path in WORKSPACE_ROOT.rglob("*"):
        if path.is_file():
            files.append(str(path.relative_to(WORKSPACE_ROOT)))
    return sorted(files)


@mcp.tool()
def read_file(path: str) -> str:
    """Read a UTF-8 text file from the agent workspace."""
    ensure_directories()
    target = resolve_workspace_path(path)
    return target.read_text(encoding="utf-8")


@mcp.tool()
def write_file(path: str, content: str) -> str:
    """Write a UTF-8 text file into the agent workspace."""
    ensure_directories()
    target = resolve_workspace_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Wrote {target.relative_to(WORKSPACE_ROOT)}"


@mcp.tool()
def search_files(query: str) -> list[dict[str, Any]]:
    """Search for text inside files in the agent workspace."""
    ensure_directories()
    matches: list[dict[str, Any]] = []
    for path in WORKSPACE_ROOT.rglob("*"):
        if not path.is_file():
            continue
        try:
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if query in line:
                    matches.append(
                        {
                            "path": str(path.relative_to(WORKSPACE_ROOT)),
                            "line": line_number,
                            "text": line,
                        }
                    )
        except UnicodeDecodeError:
            continue
    return matches[:100]


@mcp.tool()
def run_command(command: list[str], cwd: str = ".") -> dict[str, Any]:
    """Run a whitelisted command inside the agent workspace."""
    ensure_directories()
    if not command:
        raise ValueError("command must not be empty")
    executable = Path(command[0]).name
    if executable not in ALLOWED_COMMANDS:
        raise ValueError(f"Command is not allowed: {executable}")
    if executable in {"python", "python3"}:
        command = [sys.executable, *command[1:]]
    elif executable == "pytest":
        command = [sys.executable, "-m", "pytest", *command[1:]]

    workdir = resolve_workspace_path(cwd)
    workdir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        command,
        cwd=workdir,
        text=True,
        capture_output=True,
        timeout=20,
        env={
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": str(WORKSPACE_ROOT),
        },
    )
    return compact_result(completed.stdout, completed.stderr, completed.returncode)


@mcp.tool()
def create_tool_file(path: str, content: str) -> str:
    """Create or update a Python file in generated_tools for advanced self-tooling practice."""
    ensure_directories()
    if not path.endswith(".py"):
        raise ValueError("Only Python files are allowed in generated_tools")
    target = resolve_generated_tool_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Wrote generated tool file {target.relative_to(GENERATED_TOOLS_ROOT)}"


@mcp.tool()
def read_generated_tool_file(path: str) -> str:
    """Read a file from generated_tools."""
    ensure_directories()
    target = resolve_generated_tool_path(path)
    return target.read_text(encoding="utf-8")


@mcp.tool()
def run_generated_tool_tests() -> dict[str, Any]:
    """Run pytest for files under generated_tools."""
    ensure_directories()
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(GENERATED_TOOLS_ROOT)],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        env={
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": f"{GENERATED_TOOLS_ROOT}{os.pathsep}{PROJECT_ROOT}",
        },
    )
    return compact_result(completed.stdout, completed.stderr, completed.returncode)


if __name__ == "__main__":
    ensure_directories()
    mcp.run()
