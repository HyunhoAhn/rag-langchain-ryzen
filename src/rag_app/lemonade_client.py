"""Small Lemonade Server connectivity helper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

from rag_app.config import AppConfig


@dataclass(frozen=True)
class LemonadeCheckResult:
    base_url: str
    model_name: str
    models_url: str
    reachable: bool
    model_ids: list[str] | None = None
    error: str | None = None

    @property
    def model_found(self) -> bool | None:
        if self.model_ids is None:
            return None
        return self.model_name in self.model_ids


def _models_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/models"


def _extract_model_ids(payload: Any) -> list[str] | None:
    if not isinstance(payload, dict):
        return None

    data = payload.get("data")
    if not isinstance(data, list):
        return None

    model_ids: list[str] = []
    for item in data:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            model_ids.append(item["id"])

    return model_ids


def check_lemonade_models(config: AppConfig, timeout: float = 5.0) -> LemonadeCheckResult:
    models_url = _models_url(config.lemonade_base_url)

    try:
        response = requests.get(models_url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        return LemonadeCheckResult(
            base_url=config.lemonade_base_url,
            model_name=config.lemonade_chat_model,
            models_url=models_url,
            reachable=False,
            error=str(exc),
        )

    try:
        payload = response.json()
    except ValueError:
        return LemonadeCheckResult(
            base_url=config.lemonade_base_url,
            model_name=config.lemonade_chat_model,
            models_url=models_url,
            reachable=True,
            error="The /models endpoint returned a non-JSON response.",
        )

    model_ids = _extract_model_ids(payload)
    error = None if model_ids is not None else "The /models response did not contain a usable model list."

    return LemonadeCheckResult(
        base_url=config.lemonade_base_url,
        model_name=config.lemonade_chat_model,
        models_url=models_url,
        reachable=True,
        model_ids=model_ids,
        error=error,
    )
