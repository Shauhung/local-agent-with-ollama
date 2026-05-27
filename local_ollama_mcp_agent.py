from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


PROJECT_ROOT = Path(__file__).resolve().parent
OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen2.5-coder:32b"
MAX_TOOL_ROUNDS = 30
ACTION_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "object",
    "oneOf": [
        {
            "properties": {
                "type": {"const": "tool_call"},
                "reason": {"type": "string"},
                "tool": {"type": "string", "minLength": 1},
                "arguments": {"type": "object"},
            },
            "required": ["type", "reason", "tool", "arguments"],
            "additionalProperties": False,
        },
        {
            "properties": {
                "type": {"const": "final"},
                "reason": {"type": "string"},
                "answer": {"type": "string"},
            },
            "required": ["type", "reason", "answer"],
            "additionalProperties": False,
        },
    ],
}


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class AgentAction:
    type: str
    tool: str | None = None
    arguments: dict[str, Any] | None = None
    answer: str | None = None
    reason: str | None = None


class ActionValidationError(ValueError):
    """Raised when the model returns JSON that is valid but not a valid action."""


def json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)


def compact_text(value: str, limit: int = 1000) -> str:
    if len(value) <= limit:
        return value
    return f"{value[:limit]}... <truncated {len(value) - limit} chars>"


def trace(message: str, enabled: bool) -> None:
    if enabled:
        print(message, file=sys.stderr)


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            break
        else:
            raise

    if not isinstance(value, dict):
        raise ActionValidationError("JSON root must be an object.")
    return value


def parse_agent_action(raw: str) -> AgentAction:
    action = extract_json_object(raw)
    action_type = action.get("type")

    if action_type == "final":
        answer = action.get("answer")
        if not isinstance(answer, str):
            raise ActionValidationError("final action requires string field 'answer'.")
        reason = action.get("reason")
        if reason is not None and not isinstance(reason, str):
            raise ActionValidationError("action field 'reason' must be a string.")
        return AgentAction(type="final", answer=answer, reason=reason)

    if action_type == "tool_call":
        tool = action.get("tool")
        arguments = action.get("arguments")
        reason = action.get("reason")
        if not isinstance(tool, str) or not tool:
            raise ActionValidationError("tool_call requires non-empty string field 'tool'.")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise ActionValidationError("tool_call field 'arguments' must be an object.")
        if reason is not None and not isinstance(reason, str):
            raise ActionValidationError("action field 'reason' must be a string.")
        return AgentAction(type="tool_call", tool=tool, arguments=arguments, reason=reason)

    raise ActionValidationError("Action type must be 'tool_call' or 'final'.")


def build_system_prompt(tools: list[ToolSpec]) -> str:
    tool_text = json.dumps(
        [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in tools
        ],
        ensure_ascii=False,
        indent=2,
    )
    return f"""
你是一個純 local coding agent。你可以透過 MCP 工具讀寫 agent workspace、搜尋檔案、執行白名單指令、查詢公開網頁，以及在 generated_tools 裡練習產生工具與測試。

可用工具如下：
{tool_text}

你每次只能回覆一個 JSON object，不要加 markdown，不要加額外說明。

如果你需要呼叫工具，格式必須是：
{{
  "type": "tool_call",
  "reason": "簡短說明為什麼下一步要呼叫這個工具",
  "tool": "工具名稱",
  "arguments": {{ "參數": "值" }}
}}

如果任務已完成，格式必須是：
{{
  "type": "final",
  "reason": "簡短說明為什麼任務已完成",
  "answer": "給使用者看的最後回答"
}}

重要規則：
- 只有需要工具時才呼叫工具。
- reason 必須簡短，只說明可觀察的下一步決策依據，不要展開完整推理過程。
- 讀寫檔案時使用相對於 agent_workspace 的路徑。
- 進階自製工具練習只能寫到 generated_tools。
- 需要最新或外部資訊時，先用 web_search 找公開來源，再用 fetch_url 讀取需要的頁面。
- 如果工具執行失敗，根據錯誤修正下一步。
""".strip()


def build_ollama_payload(model: str, messages: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": ACTION_RESPONSE_FORMAT,
        "options": {
            "temperature": 0.1,
        },
    }


def ask_ollama(model: str, messages: list[dict[str, str]]) -> str:
    response = requests.post(
        OLLAMA_URL,
        json=build_ollama_payload(model, messages),
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    return data["message"]["content"]


async def list_mcp_tools(session: ClientSession) -> list[ToolSpec]:
    result = await session.list_tools()
    specs: list[ToolSpec] = []
    for tool in result.tools:
        specs.append(
            ToolSpec(
                name=tool.name,
                description=tool.description or "",
                input_schema=tool.inputSchema or {},
            )
        )
    return specs


async def run_agent(user_input: str, model: str, trace_enabled: bool = False) -> str:
    server = StdioServerParameters(
        command=sys.executable,
        args=[str(PROJECT_ROOT / "mcp_local_server.py")],
        cwd=str(PROJECT_ROOT),
    )

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await list_mcp_tools(session)
            messages = [
                {"role": "system", "content": build_system_prompt(tools)},
                {"role": "user", "content": user_input},
            ]

            for round_index in range(1, MAX_TOOL_ROUNDS + 1):
                raw = ask_ollama(model, messages)
                messages.append({"role": "assistant", "content": raw})

                try:
                    action = parse_agent_action(raw)
                except Exception as exc:
                    trace(f"[round {round_index}] invalid_action: {exc}", trace_enabled)
                    trace(
                        f"[round {round_index}] raw: {compact_text(raw)}",
                        trace_enabled,
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": f"你的回覆無法解析成合法 action。錯誤：{exc}。請重新只回覆一個符合格式的 JSON object。",
                        }
                    )
                    continue

                if action.type == "final":
                    trace(
                        f"[round {round_index}] final: reason={action.reason or '(none)'}",
                        trace_enabled,
                    )
                    return action.answer or ""

                tool_name = action.tool
                arguments = action.arguments or {}
                if tool_name is None:
                    messages.append(
                        {
                            "role": "user",
                            "content": "tool_call action 缺少 tool。請重新輸出合法 JSON object。",
                        }
                    )
                    continue

                trace(
                    f"[round {round_index}] tool_call: tool={tool_name} reason={action.reason or '(none)'}",
                    trace_enabled,
                )
                trace(
                    f"[round {round_index}] arguments: {json.dumps(arguments, ensure_ascii=False, default=json_default)}",
                    trace_enabled,
                )

                try:
                    result = await session.call_tool(tool_name, arguments)
                    tool_payload = json.dumps(result.content, ensure_ascii=False, default=json_default)
                except Exception as exc:
                    tool_payload = json.dumps(
                        {"error": str(exc)},
                        ensure_ascii=False,
                    )

                trace(
                    f"[round {round_index}] tool_result: {compact_text(tool_payload)}",
                    trace_enabled,
                )

                messages.append(
                    {
                        "role": "user",
                        "content": f"工具 {tool_name} 的執行結果：{tool_payload}",
                    }
                )

            return "已達最大工具呼叫輪數，任務可能尚未完成。"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local Ollama agent that calls MCP tools.")
    parser.add_argument("prompt", nargs="*", help="Prompt for the agent. If empty, start interactive mode.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name.")
    parser.add_argument("--trace", action="store_true", help="Print each agent round to stderr.")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    if args.prompt:
        answer = await run_agent(" ".join(args.prompt), args.model, trace_enabled=args.trace)
        print(answer)
        return

    print("Local Ollama MCP agent started. Type exit or quit to stop.")
    while True:
        user_input = input("\nUser: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        answer = await run_agent(user_input, args.model, trace_enabled=args.trace)
        print(f"Agent: {answer}")


if __name__ == "__main__":
    asyncio.run(main())
