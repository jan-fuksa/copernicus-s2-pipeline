from __future__ import annotations

from dataclasses import dataclass
from getpass import getpass
from typing import Optional

import requests


IDENTITY_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"


@dataclass(frozen=True)
class CDSEAuth:
    username: str
    password: str
    client_id: str = "cdse-public"
    totp: Optional[str] = None  # optional 2FA code


def prompt_auth() -> CDSEAuth:
    username = getpass("CDSE username (email): ")
    password = getpass("CDSE password: ")
    totp = getpass("CDSE TOTP (leave empty if not enabled): ").strip() or None
    return CDSEAuth(username=username, password=password, totp=totp)


def get_access_token(auth: CDSEAuth, timeout_s: int = 60) -> str:
    data = {
        "client_id": auth.client_id,
        "grant_type": "password",
        "username": auth.username,
        "password": auth.password,
    }
    if auth.totp:
        data["totp"] = auth.totp

    r = requests.post(
        IDENTITY_TOKEN_URL,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=timeout_s,
    )
    r.raise_for_status()
    js = r.json()
    if "access_token" not in js:
        raise RuntimeError(f"OAuth response missing access_token: keys={list(js.keys())}")
    return str(js["access_token"])
