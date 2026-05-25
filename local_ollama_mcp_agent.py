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
MAX_TOOL_ROUNDS = 8


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


def json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


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
你是一個純 local coding agent。你可以透過 MCP 工具讀寫 agent workspace、搜尋檔案、執行白名單指令，以及在 generated_tools 裡練習產生工具與測試。

可用工具如下：
{tool_text}

你每次只能回覆一個 JSON object，不要加 markdown，不要加額外說明。

如果你需要呼叫工具，格式必須是：
{{
  "type": "tool_call",
  "tool": "工具名稱",
  "arguments": {{ "參數": "值" }}
}}

如果任務已完成，格式必須是：
{{
  "type": "final",
  "answer": "給使用者看的最後回答"
}}

重要規則：
- 只有需要工具時才呼叫工具。
- 讀寫檔案時使用相對於 agent_workspace 的路徑。
- 進階自製工具練習只能寫到 generated_tools。
- 如果工具執行失敗，根據錯誤修正下一步。
""".strip()


def ask_ollama(model: str, messages: list[dict[str, str]]) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.1,
            },
        },
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


async def run_agent(user_input: str, model: str) -> str:
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

            for _ in range(MAX_TOOL_ROUNDS):
                raw = ask_ollama(model, messages)
                messages.append({"role": "assistant", "content": raw})

                try:
                    action = extract_json_object(raw)
                except Exception as exc:
                    messages.append(
                        {
                            "role": "user",
                            "content": f"你的回覆不是合法 JSON。錯誤：{exc}。請重新只回覆一個 JSON object。",
                        }
                    )
                    continue

                if action.get("type") == "final":
                    return str(action.get("answer", ""))

                if action.get("type") != "tool_call":
                    messages.append(
                        {
                            "role": "user",
                            "content": "JSON 的 type 必須是 tool_call 或 final。請修正。",
                        }
                    )
                    continue

                tool_name = action.get("tool")
                arguments = action.get("arguments") or {}
                if not isinstance(tool_name, str) or not isinstance(arguments, dict):
                    messages.append(
                        {
                            "role": "user",
                            "content": "tool 必須是字串，arguments 必須是 object。請修正。",
                        }
                    )
                    continue

                try:
                    result = await session.call_tool(tool_name, arguments)
                    tool_payload = json.dumps(result.content, ensure_ascii=False, default=json_default)
                except Exception as exc:
                    tool_payload = json.dumps(
                        {"error": str(exc)},
                        ensure_ascii=False,
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
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    if args.prompt:
        answer = await run_agent(" ".join(args.prompt), args.model)
        print(answer)
        return

    print("Local Ollama MCP agent started. Type exit or quit to stop.")
    while True:
        user_input = input("\nUser: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        answer = await run_agent(user_input, args.model)
        print(f"Agent: {answer}")


if __name__ == "__main__":
    asyncio.run(main())
