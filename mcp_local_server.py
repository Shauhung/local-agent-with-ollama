from __future__ import annotations

import json
import os
import ipaddress
import re
import subprocess
import sys
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
import sys

load_dotenv()

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

PACKAGE_SPEC_PATTERN = re.compile(
    r"[A-Za-z0-9_.-]+"
    r"(?:\[[A-Za-z0-9_,.-]+\])?"
    r"(?:(?:==|>=|<=|~=|!=|>|<)[A-Za-z0-9_.!*+:-]+)?"
)
STOCK_SYMBOL_PATTERN = re.compile(r"^[A-Z0-9]{1,5}(\.[A-Z]{2,3})?$")
PRIVATE_HOSTNAMES = {"localhost", "localhost.localdomain"}
USER_AGENT = "local-ollama-mcp-agent/0.1"

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


def validate_package_spec(package: str) -> str:
    package = package.strip()
    if not PACKAGE_SPEC_PATTERN.fullmatch(package):
        raise ValueError(f"Invalid package spec: {package}")
    return package


def validate_stock_symbol(symbol: str) -> str:
    symbol = symbol.strip().upper()
    if not STOCK_SYMBOL_PATTERN.fullmatch(symbol):
        raise ValueError(f"Invalid stock symbol: {symbol}")
    return symbol


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        data = data.strip()
        if data:
            self.parts.append(data)

    def text(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self.parts)).strip()


class DuckDuckGoResultParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._href: str | None = None
        self._text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        attr_map = dict(attrs)
        classes = (attr_map.get("class") or "").split()
        if "result__a" in classes:
            self._href = attr_map.get("href")
            self._text_parts = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or self._href is None:
            return
        title = re.sub(r"\s+", " ", " ".join(self._text_parts)).strip()
        if title:
            self.results.append({"title": title, "url": self._href})
        self._href = None
        self._text_parts = []


def validate_public_url(url: str) -> str:
    url = url.strip()
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http and https URLs are allowed")
    if not parsed.hostname:
        raise ValueError("URL must include a hostname")
    if parsed.username or parsed.password:
        raise ValueError("URLs with credentials are not allowed")

    hostname = parsed.hostname.rstrip(".").lower()
    if hostname in PRIVATE_HOSTNAMES or hostname.endswith(".local"):
        raise ValueError(f"Private hostnames are not allowed: {hostname}")

    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return url

    if not address.is_global:
        raise ValueError(f"Private or non-global IP addresses are not allowed: {hostname}")
    return url


def strip_html(text: str) -> str:
    parser = TextExtractor()
    parser.feed(text)
    return parser.text()


class SafeRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req: Request, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Request | None:
        validate_public_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def normalize_duckduckgo_url(href: str) -> str:
    absolute = urljoin("https://duckduckgo.com", href)
    parsed = urlparse(absolute)
    params = parse_qs(parsed.query)
    if parsed.netloc.endswith("duckduckgo.com") and "uddg" in params:
        return params["uddg"][0]
    return absolute


def fetch_public_url_text(url: str, max_chars: int, readable_html: bool) -> dict[str, Any]:
    url = validate_public_url(url)
    max_chars = max(500, min(max_chars, 30000))
    request = Request(url, headers={"User-Agent": USER_AGENT})
    opener = build_opener(SafeRedirectHandler)

    with opener.open(request, timeout=15) as response:
        final_url = validate_public_url(response.geturl())
        content_type = response.headers.get("content-type", "")
        charset = response.headers.get_content_charset() or "utf-8"
        raw = response.read(max_chars * 4)

    text = raw.decode(charset, errors="replace")
    if readable_html and "html" in content_type:
        text = strip_html(text)

    return {
        "url": final_url,
        "content_type": content_type,
        "text": text[:max_chars],
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


# @mcp.tool()
# def uv_add(package: str) -> dict[str, Any]:
#     """Add one Python package dependency to this project using uv."""
#     package = validate_package_spec(package)
#     uv_path = shutil.which("uv")
#     if uv_path is None:
#         raise ValueError("uv executable was not found in PATH")

#     completed = subprocess.run(
#         [uv_path, "add", package],
#         cwd=PROJECT_ROOT,
#         text=True,
#         capture_output=True,
#         timeout=120,
#         env={
#             "PATH": os.environ.get("PATH", ""),
#         },
#     )
#     return compact_result(completed.stdout, completed.stderr, completed.returncode)


@mcp.tool()
def fetch_url(url: str, max_chars: int = 8000) -> dict[str, Any]:
    """Fetch a public HTTP(S) URL and return readable text content."""
    return fetch_public_url_text(url, max_chars=max_chars, readable_html=True)


@mcp.tool()
def web_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Search the web and return public result titles and URLs."""
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")
    max_results = max(1, min(max_results, 10))
    search_url = f"https://duckduckgo.com/html/?{urlencode({'q': query})}"
    payload = fetch_public_url_text(search_url, max_chars=30000, readable_html=False)

    parser = DuckDuckGoResultParser()
    parser.feed(payload["text"])

    results: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for result in parser.results:
        try:
            url = validate_public_url(normalize_duckduckgo_url(result["url"]))
        except ValueError:
            continue
        if url in seen_urls:
            continue
        seen_urls.add(url)
        results.append({"title": result["title"], "url": url})
        if len(results) >= max_results:
            break
    return results


@mcp.tool()
def get_stock_quote_us(symbol: str) -> dict[str, Any]:
    """Get the latest stock quote for a symbol using Finnhub.
    
    Only supports US stock. For example: AAPL, TSLA, MSFT, QQQ.
    """
    symbol = validate_stock_symbol(symbol)
    api_key = os.environ.get("FINNHUB_API_KEY", "").strip()
    if not api_key:
        raise ValueError("FINNHUB_API_KEY is not set")

    url = f"https://finnhub.io/api/v1/quote?{urlencode({'symbol': symbol, 'token': api_key})}"
    request = Request(url, headers={"User-Agent": USER_AGENT})
    opener = build_opener(SafeRedirectHandler)

    with opener.open(request, timeout=15) as response:
        payload = json.loads(response.read(20000).decode("utf-8"))

    timestamp = payload.get("t")
    timestamp_utc = None
    if isinstance(timestamp, int | float) and timestamp > 0:
        timestamp_utc = datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()

    return {
        "symbol": symbol,
        "price": payload.get("c"),
        "change": payload.get("d"),
        "percent_change": payload.get("dp"),
        "high": payload.get("h"),
        "low": payload.get("l"),
        "open": payload.get("o"),
        "previous_close": payload.get("pc"),
        "timestamp_utc": timestamp_utc,
        "source": "Finnhub",
    }


@mcp.tool()
def get_stock_quote_tw(symbol: str) -> dict[str, Any]:
        """Get the latest stock quote for a symbol using twse.
    
        Only supports TW stock. For example: 0050, 2330, 6669.
        """
        # 判斷是上市還是上櫃（台灣常見上櫃為4碼，有些特定權證或ETF是5碼）
        # 這裡預設先查上市(tse)，若使用者有帶 .TWO 則切換為上櫃(otc)
        market = "tse"
        pure_symbol = symbol

        if "." in symbol:
            pure_symbol, suffix = symbol.split(".")
            if suffix == "TWO":
                market = "otc"
        # https://mis.twse.com.tw/stock/api/getStockInfo.jsp?ex_ch=tse_0050.tw
        channel = f"{market}_{pure_symbol}.tw"
        url = f"https://mis.twse.com.tw/stock/api/getStockInfo.jsp?ex_ch={channel}"
        print(f"DEBUG URL: {url}", file=sys.stderr)
        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        opener = build_opener()
        
        with opener.open(request, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
            
        if not data.get("msgArray"):
            # 如果上市查不到，自動嘗試切換到上櫃查一次（防呆）
            if market == "tse":
                channel = f"otc_{pure_symbol}.tw"
                url = f"https://mis.twse.com.tw/stock/api/getStockInfo.jsp?ex_ch={channel}"
                with opener.open(request, timeout=10) as response:
                    data = json.loads(response.read().decode("utf-8"))
                
        if not data.get("msgArray"):
            return {"error": f"Taiwan stock symbol '{pure_symbol}' not found on TWSE/TPEx."}
            
        info = data["msgArray"][0]
        
        return {
            "symbol": pure_symbol,
            "name": info.get("n"),                      # 公司簡稱
            "full_name": info.get("nf"),                 # 全名
            "current_price_raw": info.get("z"),          # 當前成交價 (可能是 '-')
            "yesterday_close_raw": info.get("y"),        # 昨收價
            "open_raw": info.get("o"),                   # 開盤價
            "high_raw": info.get("h"),                   # 最高價
            "low_raw": info.get("l"),                    # 最低價
            "volume_raw": info.get("v"),                 # 成交量
            "trade_time": info.get("t"),                 # 台灣時間
            "trade_date": info.get("d"),                 # 台灣日期
            "source": "TWSE MIS (Raw snapshot for Agent analysis)",
            "raw_msg": info                              # 甚至把整包塞進去讓 Agent 自己挖寶
        }

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
