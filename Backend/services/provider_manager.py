"""
provider_manager.py — Centralized LLM provider configuration
────────────────────────────────────────────────────────────
Loads providers.json, resolves API keys from env vars, and provides
lookup methods for the LLM client and frontend.

Usage:
    from provider_manager import provider_manager

    provider = provider_manager.get_provider("cf_primary")
    api_key = provider_manager.get_api_key(provider)
    all_available = provider_manager.get_available_providers()
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ProviderConfig:
    id: str
    name: str
    api_url: str
    model: str
    description: str
    api_key_env: str
    color: str
    is_cf_worker: bool
    fallback_order: int
    fast_model: Optional[str] = None
    fallback_api_key_env: Optional[str] = None


class ProviderManager:
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "providers.json",
            )
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._providers: Dict[str, ProviderConfig] = {}
        for p in data["providers"]:
            api_url = p.get("api_url", "")
            # Resolve api_url from env if api_url_env is set
            if p.get("api_url_env"):
                api_url = os.getenv(p["api_url_env"], api_url)

            config = ProviderConfig(
                id=p["id"],
                name=p["name"],
                api_url=api_url,
                model=p["model"],
                description=p["description"],
                api_key_env=p["api_key_env"],
                color=p["color"],
                is_cf_worker=p.get("is_cf_worker", False),
                fallback_order=p.get("fallback_order", 99),
                fast_model=p.get("fast_model"),
                fallback_api_key_env=p.get("fallback_api_key_env"),
            )
            self._providers[config.id] = config

    def get_provider(self, provider_id: str) -> Optional[ProviderConfig]:
        return self._providers.get(provider_id)

    def get_api_key(self, provider: ProviderConfig) -> str:
        key = os.getenv(provider.api_key_env, "")
        if not key and provider.fallback_api_key_env:
            key = os.getenv(provider.fallback_api_key_env, "")
        return key

    def get_all_providers(self) -> List[ProviderConfig]:
        return sorted(self._providers.values(), key=lambda p: p.fallback_order)

    def get_available_providers(self) -> List[ProviderConfig]:
        return [
            p for p in self.get_all_providers()
            if self.get_api_key(p)
        ]

    def get_frontend_providers(self) -> List[Dict]:
        """Return provider list safe for frontend (no secrets)."""
        return [
            {
                "id": p.id,
                "name": p.name,
                "description": p.description,
                "color": p.color,
            }
            for p in self.get_all_providers()
        ]


# Module-level singleton
provider_manager = ProviderManager()
