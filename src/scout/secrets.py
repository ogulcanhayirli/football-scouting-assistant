"""Secret lookup: environment variable first, then AWS SSM Parameter Store.

Keys are never read from config files, so a config can be committed safely.
"""

from __future__ import annotations

import os
from collections.abc import Callable


class MissingSecretError(RuntimeError):
    pass


def _ssm_fetch(name: str) -> str:
    import boto3  # optional dependency, only needed in AWS

    client = boto3.client("ssm")
    response = client.get_parameter(Name=name, WithDecryption=True)
    return response["Parameter"]["Value"]


def get_secret(
    env_var: str,
    ssm_param_env: str | None = None,
    *,
    ssm_fetch: Callable[[str], str] = _ssm_fetch,
) -> str:
    """Return the secret from ``env_var``, or from the SSM parameter named in ``ssm_param_env``."""
    value = os.environ.get(env_var, "").strip()
    if value:
        return value
    if ssm_param_env:
        param_name = os.environ.get(ssm_param_env, "").strip()
        if param_name:
            return ssm_fetch(param_name)
    hint = f" or set {ssm_param_env} to an SSM parameter name" if ssm_param_env else ""
    raise MissingSecretError(f"Set {env_var}{hint}. See .env.example.")


def api_football_key() -> str:
    return get_secret("API_FOOTBALL_KEY", "SCOUT_SSM_API_FOOTBALL_KEY")
