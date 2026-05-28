import pytest

from mcp_local_server import (
    normalize_duckduckgo_url,
    validate_package_spec,
    validate_public_url,
    validate_stock_symbol,
)


@pytest.mark.parametrize(
    "package",
    [
        "pydantic",
        "pydantic>=2",
        "pytest==9.0.3",
        "requests[socks]",
        "my-package~=1.2",
    ],
)
def test_validate_package_spec_accepts_safe_specs(package: str) -> None:
    assert validate_package_spec(package) == package


@pytest.mark.parametrize(
    "package",
    [
        "",
        "../pydantic",
        "git+https://example.com/pkg.git",
        "pydantic requests",
        "pydantic;python_version>'3.12'",
        "pydantic @ https://example.com/pydantic.whl",
    ],
)
def test_validate_package_spec_rejects_risky_specs(package: str) -> None:
    with pytest.raises(ValueError, match="Invalid package spec"):
        validate_package_spec(package)


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com",
        "http://example.com/path?q=local+agent",
        "https://8.8.8.8/dns-query",
    ],
)
def test_validate_public_url_accepts_public_urls(url: str) -> None:
    assert validate_public_url(url) == url


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://10.0.0.1",
        "http://192.168.1.1",
        "http://example.local",
        "https://user:pass@example.com",
    ],
)
def test_validate_public_url_rejects_private_or_risky_urls(url: str) -> None:
    with pytest.raises(ValueError):
        validate_public_url(url)


def test_normalize_duckduckgo_url_extracts_target() -> None:
    url = normalize_duckduckgo_url("/l/?uddg=https%3A%2F%2Fexample.com%2Fdocs")

    assert url == "https://example.com/docs"


@pytest.mark.parametrize("symbol", ["aapl", "MSFT", "BRK.B", "7203.T"])
def test_validate_stock_symbol_accepts_common_symbols(symbol: str) -> None:
    assert validate_stock_symbol(symbol) == symbol.upper()


@pytest.mark.parametrize("symbol", ["", "../AAPL", "AAPL MSFT", "$AAPL", "A" * 30])
def test_validate_stock_symbol_rejects_invalid_symbols(symbol: str) -> None:
    with pytest.raises(ValueError, match="Invalid stock symbol"):
        validate_stock_symbol(symbol)
