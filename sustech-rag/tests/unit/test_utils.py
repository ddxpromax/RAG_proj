from sustech_rag.common.utils import normalize_url, sha256_text


def test_normalize_url_removes_tracking() -> None:
    url = normalize_url("https://example.com/a?utm_source=x&b=1#frag")
    assert url == "https://example.com/a?b=1"


def test_sha256_text_stable() -> None:
    assert sha256_text("南科大") == sha256_text("南科大")

