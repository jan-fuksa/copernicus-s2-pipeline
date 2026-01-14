from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from tqdm.auto import tqdm

from .auth import CDSEAuth, get_access_token


def _is_auth_error(status_code: int) -> bool:
    return status_code in (401, 403)


def _is_retryable(status_code: int) -> bool:
    return status_code in (429, 500, 502, 503, 504)


@dataclass
class TokenManager:
    auth: CDSEAuth
    access_token: str

    def refresh(self) -> str:
        self.access_token = get_access_token(self.auth)
        return self.access_token


class CDSEHttpClient:
    def __init__(
        self,
        token_mgr: TokenManager,
        timeout_s: int = 60,
        max_retries: int = 6,
        max_auth_refresh: int = 1,
    ):
        self.token_mgr = token_mgr
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.max_auth_refresh = max_auth_refresh

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token_mgr.access_token}"}

    def request(
        self, method: str, url: str, *, stream: bool = False
    ) -> requests.Response:
        last_exc: Exception | None = None
        auth_refreshes = 0

        for attempt in range(self.max_retries + 1):
            try:
                r = requests.request(
                    method,
                    url,
                    headers=self._headers(),
                    timeout=self.timeout_s,
                    stream=stream,
                )

                if (
                    _is_auth_error(r.status_code)
                    and auth_refreshes < self.max_auth_refresh
                ):
                    auth_refreshes += 1
                    self.token_mgr.refresh()
                    continue

                if _is_retryable(r.status_code):
                    if attempt >= self.max_retries:
                        r.raise_for_status()
                    retry_after = r.headers.get("Retry-After")
                    delay = None
                    if retry_after:
                        try:
                            delay = float(retry_after)
                        except Exception:
                            delay = None
                    if delay is None:
                        delay = min(0.75 * (2**attempt), 20.0)
                    delay *= 1.0 + random.uniform(-0.25, 0.25)
                    time.sleep(max(0.0, delay))
                    continue

                r.raise_for_status()
                return r

            except (requests.Timeout, requests.ConnectionError) as e:
                last_exc = e
                if attempt >= self.max_retries:
                    raise
                delay = min(0.75 * (2**attempt), 20.0)
                delay *= 1.0 + random.uniform(-0.25, 0.25)
                time.sleep(max(0.0, delay))

        raise (
            last_exc
            if last_exc is not None
            else RuntimeError("HTTP request failed unexpectedly")
        )

    def get_json(self, url: str) -> dict[str, Any]:
        r = self.request("GET", url, stream=False)
        return r.json()

    def stream_download(
        self,
        url: str,
        dst: Path,
        *,
        overwrite: bool = False,
        chunk_size: int = 8 * 1024 * 1024,
    ) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and not overwrite:
            return

        r = self.request("GET", url, stream=True)
        total = r.headers.get("Content-Length")
        total_i = int(total) if total and total.isdigit() else None

        tmp = dst.with_suffix(dst.suffix + ".part")
        with (
            open(tmp, "wb") as f,
            tqdm(
                total=total_i,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=dst.name,
                leave=False,
            ) as pbar,
        ):
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                if total_i is not None:
                    pbar.update(len(chunk))
        tmp.replace(dst)
