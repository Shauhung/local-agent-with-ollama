# Local Ollama MCP Agent

這是一個純 local 的 agent 練習專案：

- `mcp_local_server.py`：MCP server，提供安全受限的本機工具。
- `local_ollama_mcp_agent.py`：Ollama agent loop，讓模型決定何時呼叫 MCP 工具。
- `agent_workspace/`：模型可讀寫的工作區。
- `generated_tools/`：第二階段自製工具與自動偵錯練習區。

## 先決條件

確認 Ollama 已啟動，並準備一個模型：

```bash
ollama pull qwen2.5-coder:32b
```

如果你的機器跑不動 32B，可以改用：

```bash
ollama pull qwen2.5-coder:7b
```

## 安裝依賴

```bash
venv/bin/python -m pip install -r requirements.txt
```

## 最小測試

```bash
venv/bin/python local_ollama_mcp_agent.py "建立 hello.txt，內容是 Hello local agent"
```

指定模型：

```bash
venv/bin/python local_ollama_mcp_agent.py --model qwen2.5-coder:7b "列出目前 workspace 的檔案"
```

互動模式：

```bash
venv/bin/python local_ollama_mcp_agent.py --model qwen2.5-coder:7b
```

HTTP server：

```bash
venv/bin/python local_agent_server.py --host 127.0.0.1 --port 8765
```

呼叫 agent：

```bash
curl -s http://127.0.0.1:8765/agent/message \
  -H 'content-type: application/json' \
  -d '{"session_id":"phone","message":"列出 workspace 的檔案","trace":true}'
```

server 會用 SQLite 保存每個 `session_id` 的對話歷史，預設每次帶入最近 20 則訊息。

## 第一階段工具

目前 MCP server 提供：

- `get_time`
- `list_files`
- `read_file`
- `write_file`
- `search_files`
- `run_command`
- `uv_add`
- `web_search`
- `fetch_url`
- `get_stock_quote`

`read_file`、`write_file`、`search_files`、`run_command` 都限制在 `agent_workspace/` 裡。

`run_command` 只允許白名單指令：

- `ls`
- `pwd`
- `python`
- `python3`
- `pytest`

`uv_add` 是專用依賴安裝工具，會在專案根目錄執行 `uv add <package>`，並只接受單一保守格式的 package spec，例如 `pydantic`、`pytest==9.0.3`、`requests[socks]`。

`web_search` 和 `fetch_url` 提供連網能力，只允許公開 `http`/`https` URL，會拒絕 localhost、內網 IP、`.local` host 和帶帳密的 URL。

`get_stock_quote` 會透過 Finnhub 查詢最新股價，需先設定：

```bash
export FINNHUB_API_KEY="你的 Finnhub API key"
```

## 第二階段：自製工具與偵錯

MCP server 也提供：

- `create_tool_file`
- `read_generated_tool_file`
- `run_generated_tool_tests`

範例 prompt：

```text
請在 generated_tools 建立一個 calculator.py，提供 add(a, b)，再寫 pytest 測試並執行，錯了就修到通過。
```

這個階段的重點是讓模型學會：

1. 產生工具程式。
2. 產生測試。
3. 執行測試。
4. 讀錯誤。
5. 修正程式。

測試通過後，再人工決定是否把新工具正式加入 `mcp_local_server.py`。
