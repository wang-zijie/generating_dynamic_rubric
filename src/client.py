"""Bedrock API client with retry logic."""

import logging
import os
import time

import requests

logger = logging.getLogger(__name__)

JUDGES = {
    "claude-sonnet-4": {
        "model_id": "us.anthropic.claude-sonnet-4-20250514-v1:0",
        "max_tokens": 256,
        "temperature": 0.0,
    },
    "llama-3.1-8b": {
        "model_id": "us.meta.llama3-1-8b-instruct-v1:0",
        "max_tokens": 256,
        "temperature": 0.0,
    },
    "llama-3.1-70b": {
        "model_id": "us.meta.llama3-1-70b-instruct-v1:0",
        "max_tokens": 256,
        "temperature": 0.0,
    },
    "qwen3-14b": {
        "model_id": "qwen3-14b",
        "max_tokens": 256,
        "temperature": 0.0,
    },
}


class BedrockClient:
    """Bedrock Runtime Converse API with bearer-token authentication."""

    def __init__(self, region: str | None = None, bearer_token: str | None = None):
        self.region = region or os.environ.get("AWS_REGION", "us-east-1")
        self.bearer_token = bearer_token or os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        if not self.bearer_token:
            raise ValueError(
                "Bearer token not found. Set AWS_BEARER_TOKEN_BEDROCK."
            )
        self.base_url = f"https://bedrock-runtime.{self.region}.amazonaws.com"
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.bearer_token}",
                "Content-Type": "application/json",
            }
        )

    def converse(
        self,
        model_id: str,
        *,
        system: str,
        user_message: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> dict:
        url = f"{self.base_url}/model/{model_id}/converse"
        payload = {
            "system": [{"text": system}],
            "messages": [
                {"role": "user", "content": [{"text": user_message}]},
            ],
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": temperature,
            },
        }
        resp = self.session.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()


def call_judge(
    client: BedrockClient,
    model_id: str,
    *,
    system: str,
    user_message: str,
    max_tokens: int = 256,
    temperature: float = 0.0,
    max_retries: int = 5,
) -> str:
    """Call a Bedrock model and return text, with retry + backoff."""
    for attempt in range(max_retries):
        try:
            resp = client.converse(
                model_id,
                system=system,
                user_message=user_message,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp["output"]["message"]["content"][0]["text"]
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            wait = 2 ** attempt
            if status == 429:
                logger.warning(
                    "Throttled -- retry in %ds (%d/%d)", wait, attempt + 1, max_retries
                )
            else:
                logger.error(
                    "HTTP %s (%d/%d): %s", status, attempt + 1, max_retries, e
                )
                if attempt == max_retries - 1:
                    raise
            time.sleep(wait)
        except Exception as e:
            logger.error("Error (%d/%d): %s", attempt + 1, max_retries, e)
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("Max retries exceeded")
