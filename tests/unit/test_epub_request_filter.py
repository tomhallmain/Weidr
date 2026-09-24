"""ePub render request filter: CDP Fetch domain, with the legacy Network
interception only as a fallback."""

import asyncio

from image.frame_cache import _block_requests_outside


class _FakeSession:
    def __init__(self, fetch_available=True):
        self.fetch_available = fetch_available
        self.sent = []
        self.handlers = {}

    def on(self, event, handler):
        self.handlers[event] = handler

    async def send(self, method, params=None):
        if method == "Fetch.enable" and not self.fetch_available:
            raise RuntimeError("'Fetch.enable' wasn't found")
        self.sent.append((method, params))


class _FakeTarget:
    def __init__(self, session):
        self._session = session

    async def createCDPSession(self):
        return self._session


class _FakePage:
    def __init__(self, session):
        self.target = _FakeTarget(session)
        self.interception = None
        self.request_handler = None

    async def setRequestInterception(self, value):
        self.interception = value

    def on(self, event, handler):
        if event == "request":
            self.request_handler = handler


def _paused(request_id, url):
    return {"requestId": request_id, "request": {"url": url}}


def test_fetch_domain_continues_inside_and_fails_outside(tmp_path):
    inside = tmp_path / "book" / "ch1.xhtml"
    inside.parent.mkdir()
    inside.write_text("")
    session = _FakeSession()
    page = _FakePage(session)

    async def _run():
        await _block_requests_outside(page, str(tmp_path))
        handler = session.handlers["Fetch.requestPaused"]
        handler(_paused("1", inside.as_uri()))
        handler(_paused("2", "https://tracker.example/pixel.gif"))
        for _ in range(5):
            await asyncio.sleep(0)

    asyncio.run(_run())

    assert ("Fetch.enable", {"patterns": [{"urlPattern": "*"}]}) in session.sent
    assert ("Fetch.continueRequest", {"requestId": "1"}) in session.sent
    assert ("Fetch.failRequest", {"requestId": "2", "errorReason": "BlockedByClient"}) in session.sent
    assert page.interception is None


def test_falls_back_to_network_interception_without_fetch(tmp_path):
    session = _FakeSession(fetch_available=False)
    page = _FakePage(session)
    asyncio.run(_block_requests_outside(page, str(tmp_path)))
    assert page.interception is True
    assert page.request_handler is not None
