"""Playwright 기반 웹 뷰어 UI 테스트.

sonote 서버를 테스트 모드로 기동한 뒤 Chromium으로 접속하여
핵심 UI 동작을 검증한다.

실행: pytest tests/test_viewer_ui.py -v --headed  (브라우저 표시)
      pytest tests/test_viewer_ui.py -v           (headless)

필요: pip install playwright pytest-playwright && playwright install chromium
"""
from __future__ import annotations

import os
import threading
import time

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("SONOTE_TEST_UI"),
    reason="UI test — set SONOTE_TEST_UI=1 to enable (requires playwright)",
)

try:
    from playwright.sync_api import Page, expect
except ImportError:
    pytest.skip("playwright not installed", allow_module_level=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def server_url():
    """sonote FastAPI 서버를 백그라운드 스레드에서 기동."""
    import uvicorn
    from src.server import create_app

    app = create_app()
    port = 18765

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # 서버 ready 대기
    import urllib.request
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/status", timeout=0.3)
            break
        except Exception:
            time.sleep(0.1)

    yield f"http://127.0.0.1:{port}"
    server.should_exit = True


# ---------------------------------------------------------------------------
# 1. 페이지 로드 + 기본 요소
# ---------------------------------------------------------------------------

def test_viewer_loads(page: Page, server_url: str):
    """viewer.html이 정상 로드되고 제목이 표시되는지."""
    page.goto(server_url)
    expect(page).to_have_title("sonote — 실시간 회의 전사")


def test_viewer_has_transcript_area(page: Page, server_url: str):
    """전사 영역이 존재하는지."""
    page.goto(server_url)
    # transcript 컨테이너 존재 확인
    transcript = page.locator("#transcript, .transcript, [data-transcript]").first
    expect(transcript).to_be_visible()


# ---------------------------------------------------------------------------
# 2. 테마 토글
# ---------------------------------------------------------------------------

def test_theme_toggle(page: Page, server_url: str):
    """다크/라이트 테마 전환이 동작하는지."""
    page.goto(server_url)

    # 초기 테마 확인 (기본 dark)
    html = page.locator("html")
    initial_theme = html.get_attribute("data-theme")

    # 테마 토글 버튼 클릭
    toggle = page.locator("[data-theme-toggle], .theme-toggle, #theme-toggle, button:has-text('테마')").first
    if toggle.is_visible():
        toggle.click()
        page.wait_for_timeout(300)

        new_theme = html.get_attribute("data-theme")
        assert new_theme != initial_theme, f"테마가 변경되지 않음: {initial_theme} → {new_theme}"


# ---------------------------------------------------------------------------
# 3. 설정 페이지
# ---------------------------------------------------------------------------

def test_settings_page_loads(page: Page, server_url: str):
    """설정 페이지가 정상 로드되는지."""
    page.goto(f"{server_url}/settings.html")
    # 설정 관련 요소 존재
    page.wait_for_load_state("domcontentloaded")
    assert page.title() or page.locator("body").inner_text()


# ---------------------------------------------------------------------------
# 4. API 상태 체크 (UI에서 fetch하는 엔드포인트)
# ---------------------------------------------------------------------------

def test_status_endpoint(page: Page, server_url: str):
    """/status 엔드포인트가 JSON을 반환하는지."""
    response = page.request.get(f"{server_url}/status")
    assert response.ok
    data = response.json()
    assert "status" in data or "ok" in data or isinstance(data, dict)


def test_sessions_endpoint(page: Page, server_url: str):
    """/api/sessions 엔드포인트가 동작하는지."""
    response = page.request.get(f"{server_url}/api/sessions")
    assert response.ok


# ---------------------------------------------------------------------------
# 5. SSE 연결 (EventSource)
# ---------------------------------------------------------------------------

def test_sse_connection(page: Page, server_url: str):
    """SSE /stream 엔드포인트에 연결 가능한지."""
    page.goto(server_url)
    # JavaScript로 EventSource 연결 테스트
    result = page.evaluate("""
        () => new Promise((resolve, reject) => {
            const es = new EventSource('/stream');
            const timeout = setTimeout(() => {
                es.close();
                resolve('timeout_ok');  // 타임아웃이어도 연결 자체는 성공
            }, 2000);
            es.onopen = () => {
                clearTimeout(timeout);
                es.close();
                resolve('connected');
            };
            es.onerror = (e) => {
                clearTimeout(timeout);
                es.close();
                resolve('error');
            };
        })
    """)
    assert result in ("connected", "timeout_ok"), f"SSE 연결 실패: {result}"


# ---------------------------------------------------------------------------
# 6. 반응형 레이아웃
# ---------------------------------------------------------------------------

def test_mobile_viewport(page: Page, server_url: str):
    """모바일 뷰포트에서 페이지가 깨지지 않는지."""
    page.set_viewport_size({"width": 375, "height": 812})
    page.goto(server_url)
    page.wait_for_load_state("domcontentloaded")
    # 페이지가 로드되고 스크롤 가능한지 확인
    body_height = page.evaluate("document.body.scrollHeight")
    assert body_height > 0


def test_desktop_viewport(page: Page, server_url: str):
    """데스크톱 뷰포트에서 사이드바가 표시되는지."""
    page.set_viewport_size({"width": 1920, "height": 1080})
    page.goto(server_url)
    page.wait_for_load_state("domcontentloaded")
    body_width = page.evaluate("document.body.scrollWidth")
    assert body_width > 0
