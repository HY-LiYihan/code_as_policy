"""Create the runtime OpenAI-compatible client without storing credentials."""

from __future__ import annotations

import os


def create_openai_client():
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY before contacting the language model")
    options = {"api_key": api_key}
    base_url = os.environ.get("OPENAI_BASE_URL")
    if base_url:
        options["base_url"] = base_url
    return OpenAI(**options)
