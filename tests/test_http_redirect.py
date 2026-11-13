#
# Copyright (C) 2026 Radim Rehurek <me@radimrehurek.com>
#
# This code is distributed under the terms and conditions
# from the MIT License (MIT).
#
"""Tests for the manual, policy-driven HTTP redirect handling in :mod:`smart_open.http`."""

import io

import pytest
import requests
from requests.adapters import BaseAdapter
from urllib3 import HTTPResponse

import smart_open.http

BYTES = b"i tried so hard and got so far but in the end it doesn't even matter"


class TrackedRaw(HTTPResponse):
    """An ``HTTPResponse`` that records whether smart_open closed it."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.release_count = 0

    def release_conn(self):
        self.release_count += 1
        super().release_conn()


def make_response(request, status_code, headers, body=b""):
    """Build a ``requests.Response`` from a plain status/headers/body triple."""
    raw = TrackedRaw(
        body=io.BytesIO(body),
        headers=dict(headers),
        status=status_code,
        preload_content=False,
        reason="OK",
    )
    response = requests.Response()
    response.status_code = status_code
    response.headers = requests.structures.CaseInsensitiveDict(headers)
    response.raw = raw
    response.request = request
    return response


class RecordingAdapter(BaseAdapter):
    """An adapter that only records requests and plays scripted responses.

    No real network is touched. Every request is recorded with the headers and
    auth it was sent with, and every response tracks whether it was closed.
    """

    def __init__(self, routes):
        self.routes = routes
        self.requests = []
        self.responses = []

    def send(self, request, stream=False, timeout=None, verify=True, cert=None, proxies=None):  # noqa: ARG002
        self.requests.append(request)
        status_code, headers, body = self.routes[request.url](request)
        response = make_response(request, status_code, headers, body)
        self.responses.append(response)
        return response

    def close(self):
        pass

    @property
    def urls(self):
        return [request.url for request in self.requests]

    @property
    def closed_flags(self):
        return [response.raw.release_count > 0 for response in self.responses]


def make_session(routes):
    session = requests.Session()
    adapter = RecordingAdapter(routes)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session, adapter


def redirect(target, status_code=302):
    return status_code, {"Location": target, "Content-Length": "0"}, b""


def final_headers(**extra):
    headers = {"Content-Length": str(len(BYTES)), "Accept-Ranges": "bytes"}
    headers.update(extra)
    return headers


def test_three_hop_chain_with_relative_location():
    """The policy sees every hop, relative Locations resolve, and intermediates close."""
    routes = {
        "http://example.com/a": lambda request: redirect("/b", 301),
        "http://example.com/b": lambda request: redirect("http://example.com/c", 302),
        "http://example.com/c": lambda request: redirect("/d", 307),
        "http://example.com/d": lambda request: (200, final_headers(), BYTES),
    }
    session, adapter = make_session(routes)
    seen = []

    def policy(source, target, status_code, chain):
        seen.append((source, target, status_code, tuple(chain)))
        return True

    with smart_open.http.open(
        "http://example.com/a", "rb", session=session, redirect_policy=policy
    ) as reader:
        assert reader.read() == BYTES

    assert adapter.urls == [
        "http://example.com/a",
        "http://example.com/b",
        "http://example.com/c",
        "http://example.com/d",
    ]
    assert seen == [
        (
            "http://example.com/a",
            "http://example.com/b",
            301,
            ("http://example.com/a",),
        ),
        (
            "http://example.com/b",
            "http://example.com/c",
            302,
            ("http://example.com/a", "http://example.com/b"),
        ),
        (
            "http://example.com/c",
            "http://example.com/d",
            307,
            ("http://example.com/a", "http://example.com/b", "http://example.com/c"),
        ),
    ]
    # the three intermediate responses were closed promptly, the final one on reader.close()
    assert adapter.closed_flags == [True, True, True, True]


def test_policy_denies_cross_origin_before_request():
    """A denied cross-origin target never receives a request, and its redirect response closes."""
    routes = {
        "http://example.com/a": lambda request: redirect("http://evil.test/steal"),
        "http://evil.test/steal": lambda request: pytest.fail("the rejected target must never be requested"),
    }
    session, adapter = make_session(routes)

    def policy(source, target, status_code, chain):
        return target.startswith("http://example.com")

    with pytest.raises(smart_open.http.RedirectRejected, match="redirect policy rejected"):
        smart_open.http.open("http://example.com/a", "rb", session=session, redirect_policy=policy)

    assert adapter.urls == ["http://example.com/a"]
    assert adapter.closed_flags == [True]
    # the external session still works for ordinary requests
    routes["http://example.com/a"] = lambda request: (200, final_headers(), BYTES)
    with smart_open.http.open("http://example.com/a", "rb", session=session) as reader:
        assert reader.read() == BYTES


def test_same_host_different_port_is_cross_origin():
    """A port change is a different origin, so credentials and sensitive headers are dropped."""
    routes = {
        "http://example.com:8000/a": lambda request: redirect("http://example.com:9000/b"),
        "http://example.com:9000/b": lambda request: (200, final_headers(), BYTES),
    }
    session, adapter = make_session(routes)

    def policy(source, target, status_code, chain):
        return True

    headers = {
        "Accept-Encoding": "identity",
        "Authorization": "Bearer secret",
        "Cookie": "session=1",
        "X-Api-Key": "key",
        "X-Plain": "kept",
    }
    with smart_open.http.open(
        "http://example.com:8000/a",
        "rb",
        session=session,
        user="user",
        password="pass",
        headers=headers,
        redirect_policy=policy,
    ) as reader:
        assert reader.read() == BYTES

    first, second = adapter.requests
    assert first.headers["Authorization"] == "Basic dXNlcjpwYXNz"
    assert first.headers["Cookie"] == "session=1"
    assert first.headers["X-Api-Key"] == "key"

    assert "Authorization" not in second.headers
    assert "Cookie" not in second.headers
    assert "X-Api-Key" not in second.headers
    assert second.headers["X-Plain"] == "kept"


def test_credentials_stay_dropped_across_later_hops():
    """Once a chain crosses an origin, credentials are not restored by a later same-origin hop."""
    routes = {
        "http://a.example/x": lambda request: redirect("http://b.example/y"),
        "http://b.example/y": lambda request: redirect("/z"),
        "http://b.example/z": lambda request: (200, final_headers(), BYTES),
    }
    session, adapter = make_session(routes)
    headers = {"Accept-Encoding": "identity", "Authorization": "Bearer secret", "X-Plain": "kept"}
    with smart_open.http.open(
        "http://a.example/x",
        "rb",
        session=session,
        user="user",
        password="pass",
        headers=headers,
        redirect_policy=lambda *args: True,
    ) as reader:
        assert reader.read() == BYTES

    assert adapter.requests[0].headers["Authorization"] == "Basic dXNlcjpwYXNz"
    for request in adapter.requests[1:]:
        assert "Authorization" not in request.headers
        assert request.headers["X-Plain"] == "kept"
    assert adapter.urls == ["http://a.example/x", "http://b.example/y", "http://b.example/z"]


def test_same_origin_keeps_credentials_on_redirect():
    """Auth, cookies and custom headers follow same-origin redirects unchanged."""
    routes = {
        "http://example.com/a": lambda request: redirect("/b"),
        "http://example.com/b": lambda request: (200, final_headers(), BYTES),
    }
    session, adapter = make_session(routes)
    headers = {"Accept-Encoding": "identity", "Cookie": "session=1", "X-Api-Key": "key"}
    with smart_open.http.open(
        "http://example.com/a",
        "rb",
        session=session,
        user="user",
        password="pass",
        headers=headers,
    ) as reader:
        assert reader.read() == BYTES

    second = adapter.requests[1]
    assert second.headers["Authorization"] == "Basic dXNlcjpwYXNz"
    assert second.headers["Cookie"] == "session=1"
    assert second.headers["X-Api-Key"] == "key"


def test_seek_range_request_goes_through_policy():
    """A same-URL redirect on a seek is offered to the policy and still pins the version."""

    def partial(request):
        start = int(request.headers["range"].removeprefix("bytes=").partition("-")[0])
        last = min(start + 9, len(BYTES) - 1)
        return (
            206,
            {"Content-Range": f"bytes {start}-{last}/{len(BYTES)}", "ETag": '"v1"'},
            BYTES[start : last + 1],
        )

    def handler(request):
        if "range" not in request.headers:
            return redirect("/canonical")  # the initial open is pinned to the canonical URL
        assert request.headers["If-Match"] == '"v1"'
        if request.url == "http://example.com/seek":
            return redirect("/canonical")  # same origin, lands back on the pinned URL
        return partial(request)

    def canonical(request):
        if "range" not in request.headers:
            return 200, final_headers(ETag='"v1"'), BYTES
        return partial(request)

    routes = {
        "http://example.com/seek": handler,
        "http://example.com/canonical": canonical,
    }
    session, adapter = make_session(routes)
    decisions = []

    def policy(source, target, status_code, chain):
        decisions.append((source, target, status_code))
        return target.endswith("/canonical")

    decisions_before_open = len(decisions)
    with smart_open.http.open(
        "http://example.com/seek", "rb", session=session, redirect_policy=policy
    ) as reader:
        reader.seek(10)
        assert reader.read(10) == BYTES[10:20]

    # one decision for the open's redirect, one for the seek's range-redirect
    assert len(decisions) == decisions_before_open + 2
    assert decisions[-1] == ("http://example.com/seek", "http://example.com/canonical", 302)
    assert adapter.requests[-1].url == "http://example.com/canonical"
    assert adapter.closed_flags[:-1]  # every response except the one still being read is closed


def test_redirect_after_open_refuses_to_mix_versions():
    """A range request redirecting to a new URL fails instead of splicing another body in."""

    def handler(request):
        if "range" not in request.headers:
            return 200, final_headers(ETag='"v1"'), BYTES
        if request.url == "http://example.com/v1" and request.headers["range"] == "bytes=20-":
            return redirect("http://example.com/v2")
        if request.url == "http://example.com/v1":
            start = int(request.headers["range"].removeprefix("bytes=").partition("-")[0])
            last = min(start + 9, len(BYTES) - 1)
            return (
                206,
                {"Content-Range": f"bytes {start}-{last}/{len(BYTES)}", "ETag": '"v1"'},
                BYTES[start : last + 1],
            )
        return 200, final_headers(ETag='"v2"'), b"totally different replacement body"

    routes = {
        "http://example.com/v1": handler,
        "http://example.com/v2": handler,
    }
    session, adapter = make_session(routes)
    with smart_open.http.open("http://example.com/v1", "rb", session=session, buffer_size=8) as reader:
        assert reader.read(5) == BYTES[:5]
        seek_target = 20  # outside the 8-byte buffered prefix, forces a range GET
        with pytest.raises(smart_open.http.RedirectError, match="refusing to mix versions"):
            reader.seek(seek_target)
        # like any failed seek, reads now raise until a seek succeeds again
        with pytest.raises(OSError, match="seek to retry"):
            reader.read()
        assert reader.seek(0) == 0
        assert reader.read(5) == BYTES[:5]  # the original version, never the v2 body

    # the rejected replacement response was closed and never delivered
    assert adapter.closed_flags[-1]


def test_capped_range_continuation_goes_through_policy():
    """The follow-up GET for a server-truncated range is also run through the policy."""

    def partial(request):
        first = int(request.headers["range"].removeprefix("bytes=").partition("-")[0])
        last = min(first + 9, len(BYTES) - 1)  # the server caps every range at 10 bytes
        return (
            206,
            {"Content-Range": f"bytes {first}-{last}/{len(BYTES)}", "ETag": '"v1"'},
            BYTES[first : last + 1],
        )

    def cap(request):
        if "range" not in request.headers:
            return redirect("/cont")  # the initial open is pinned to the canonical URL
        return redirect("/cont")  # every range request, including the capped-range continuation

    def cont(request):
        if "range" not in request.headers:
            return 200, final_headers(ETag='"v1"'), BYTES
        return partial(request)  # the pinned URL always answers range requests directly

    routes = {
        "http://example.com/cap": cap,
        "http://example.com/cont": cont,
    }
    session, adapter = make_session(routes)
    decisions = []

    def policy(source, target, status_code, chain):
        decisions.append(target)
        return target.endswith("/cont")

    with smart_open.http.open(
        "http://example.com/cap", "rb", session=session, redirect_policy=policy
    ) as reader:
        reader.seek(5)
        assert reader.read() == BYTES[5:]

    # the open plus every capped-range continuation were approved through the policy
    assert all(target == "http://example.com/cont" for target in decisions)
    assert len(decisions) >= 2  # at least the open and one continuation
    assert adapter.requests[-2].url == "http://example.com/cap"
    assert adapter.requests[-1].url == "http://example.com/cont"


def test_redirect_loop_and_hop_limit():
    """Loops and over-long chains are refused regardless of what the policy returns."""
    loop_routes = {
        "http://example.com/a": lambda request: redirect("/b"),
        "http://example.com/b": lambda request: redirect("/a"),
    }
    session, _ = make_session(loop_routes)
    with pytest.raises(smart_open.http.RedirectLoop, match="redirect loop"):
        smart_open.http.open(
            "http://example.com/a", "rb", session=session, redirect_policy=lambda *args: True
        )

    def hop_redirect(index):
        def route(request):
            return redirect(f"/hop{index + 1}")

        return route

    long_routes = {f"http://example.com/hop{i}": hop_redirect(i) for i in range(31)}
    long_routes["http://example.com/hop31"] = lambda request: (200, final_headers(), BYTES)
    session, adapter = make_session(long_routes)
    with pytest.raises(smart_open.http.TooManyRedirects, match="exceeded 30 redirects"):
        smart_open.http.open(
            "http://example.com/hop0", "rb", session=session, redirect_policy=lambda *args: True
        )
    assert len(adapter.requests) == smart_open.http.DEFAULT_MAX_REDIRECTS


def test_non_redirect_error_still_raises_and_closes():
    """A non-3xx failure is returned to the caller and its response is released on open failure."""
    routes = {"http://example.com/missing": lambda request: (404, {"Content-Length": "9"}, b"not found")}
    session, adapter = make_session(routes)
    with pytest.raises(requests.HTTPError, match="404"):
        smart_open.http.open("http://example.com/missing", "rb", session=session)
    assert adapter.closed_flags == [True]


def test_default_policy_follows_redirect_without_callback():
    """Without a policy, redirects are followed automatically up to the hop limit."""
    routes = {
        "http://example.com/a": lambda request: redirect("/b"),
        "http://example.com/b": lambda request: (200, final_headers(), BYTES),
    }
    session, adapter = make_session(routes)
    with smart_open.http.open("http://example.com/a", "rb", session=session) as reader:
        assert reader.read() == BYTES
    assert adapter.urls == ["http://example.com/a", "http://example.com/b"]
