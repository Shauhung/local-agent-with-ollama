from __future__ import annotations

import argparse
import asyncio
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from local_ollama_mcp_agent import DEFAULT_MODEL, run_agent


def parse_agent_request(payload: dict[str, Any]) -> tuple[str, str, bool]:
    message = payload.get("message")
    if not isinstance(message, str) or not message.strip():
        raise ValueError("message must be a non-empty string")

    model = payload.get("model", DEFAULT_MODEL)
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty string")

    trace_enabled = payload.get("trace", False)
    if not isinstance(trace_enabled, bool):
        raise ValueError("trace must be a boolean")

    return message.strip(), model.strip(), trace_enabled


class AgentRequestHandler(BaseHTTPRequestHandler):
    server_version = "LocalAgentHTTP/0.1"

    def do_OPTIONS(self) -> None:
        self.send_json({"ok": True})

    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_json({"ok": True})
            return
        self.send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.path != "/agent/message":
            self.send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
            return

        try:
            payload = self.read_json()
            message, model, trace_enabled = parse_agent_request(payload)
        except ValueError as exc:
            self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        try:
            answer = asyncio.run(run_agent(message, model, trace_enabled=trace_enabled))
        except Exception as exc:
            self.send_json(
                {"error": f"agent failed: {exc}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return

        self.send_json({"answer": answer, "model": model})

    def read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("content-length", "0"))
        if content_length <= 0:
            raise ValueError("request body must not be empty")

        raw = self.rfile.read(content_length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON body: {exc}") from exc

        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        return payload

    def send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(raw)))
        self.send_header("access-control-allow-origin", "*")
        self.send_header("access-control-allow-methods", "GET, POST, OPTIONS")
        self.send_header("access-control-allow-headers", "content-type")
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HTTP server for the local Ollama MCP agent.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), AgentRequestHandler)
    print(f"Local agent server listening on http://{args.host}:{args.port}")
    print("POST /agent/message with JSON: {\"message\": \"...\", \"trace\": true}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down local agent server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
