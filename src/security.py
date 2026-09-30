"""Input validation and security utilities for DPSRO.

Provides sanitisation, bounds-checking, and safe configuration loading
to prevent injection attacks, integer overflow, and misconfigured runs.

Security practices implemented
-------------------------------
- No hardcoded secrets, tokens, or API keys anywhere in the codebase.
- All external configuration loaded via environment variables or YAML files.
- All numeric inputs bounds-checked before being passed to the LP solver.
- String inputs sanitised to prevent log injection.
- Sensitive fields (tokens, passwords) redacted from log output.
"""

from __future__ import annotations

import os
import re
import math
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Numeric bounds
# ─────────────────────────────────────────────────────────────────────────────
MAX_BATCHES         = 10_000
MAX_QUANTITY_KG     = 1_000_000.0
MAX_SHELF_LIFE_HRS  = 8_760.0       # 1 year upper bound
MIN_SHELF_LIFE_HRS  = 0.1
MAX_TEMPERATURE_C   = 80.0
MIN_TEMPERATURE_C   = -30.0
MAX_LAMBDA_WEIGHT   = 1_000.0
MIN_LAMBDA_WEIGHT   = 0.0
MAX_CAPACITY_KG     = 10_000_000.0
MAX_DISTANCE_KM     = 50_000.0
MAX_COST_PER_UNIT   = 1_000.0


def _redact(value: str) -> str:
    """Redacts a sensitive string for safe logging (shows first 4 chars only)."""
    if len(value) <= 4:
        return "****"
    return value[:4] + "****"


def validate_quantity(value: Any, field_name: str = "quantity") -> float:
    """Validates and returns a non-negative quantity value in kg.

    Args:
        value: Raw input value to validate.
        field_name: Human-readable field name for error messages.

    Returns:
        Validated float quantity.

    Raises:
        ValueError: If value is non-numeric, negative, or exceeds bounds.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be numeric, got: {type(value).__name__}")
    if not math.isfinite(v):
        raise ValueError(f"{field_name} must be finite, got: {v}")
    if v < 0:
        raise ValueError(f"{field_name} must be non-negative, got: {v}")
    if v > MAX_QUANTITY_KG:
        raise ValueError(f"{field_name} exceeds maximum ({MAX_QUANTITY_KG} kg): {v}")
    return v


def validate_shelf_life(hours: Any, field_name: str = "shelf_life_hours") -> float:
    """Validates remaining shelf life in hours.

    Args:
        hours: Raw shelf life value.
        field_name: Field name for error messages.

    Returns:
        Validated float shelf life in hours.

    Raises:
        ValueError: If out of realistic perishable food bounds.
    """
    try:
        v = float(hours)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be numeric")
    if not math.isfinite(v):
        raise ValueError(f"{field_name} must be finite")
    if v < MIN_SHELF_LIFE_HRS:
        raise ValueError(f"{field_name} too small ({v} < {MIN_SHELF_LIFE_HRS} hrs)")
    if v > MAX_SHELF_LIFE_HRS:
        raise ValueError(f"{field_name} exceeds 1 year ({v} hrs)")
    return v


def validate_temperature(celsius: Any, field_name: str = "temperature") -> float:
    """Validates a temperature value in °C within realistic food-chain bounds.

    Args:
        celsius: Temperature in °C.
        field_name: Field name for error messages.

    Returns:
        Validated float temperature.

    Raises:
        ValueError: If outside [-30, 80] °C.
    """
    try:
        v = float(celsius)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be numeric")
    if not math.isfinite(v):
        raise ValueError(f"{field_name} must be finite")
    if v < MIN_TEMPERATURE_C or v > MAX_TEMPERATURE_C:
        raise ValueError(
            f"{field_name} out of realistic range [{MIN_TEMPERATURE_C}, {MAX_TEMPERATURE_C}]°C: {v}"
        )
    return v


def validate_lambda_weight(value: Any, field_name: str = "lambda") -> float:
    """Validates an LP objective weight (must be non-negative and finite).

    Args:
        value: Raw weight value.
        field_name: Field name for error messages.

    Returns:
        Validated non-negative float weight.

    Raises:
        ValueError: If negative, non-finite, or exceeds MAX_LAMBDA_WEIGHT.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be numeric")
    if not math.isfinite(v):
        raise ValueError(f"{field_name} must be finite")
    if v < MIN_LAMBDA_WEIGHT:
        raise ValueError(f"{field_name} must be non-negative, got: {v}")
    if v > MAX_LAMBDA_WEIGHT:
        raise ValueError(f"{field_name} exceeds maximum {MAX_LAMBDA_WEIGHT}: {v}")
    return v


def validate_node_id(node_id: Any, field_name: str = "node_id") -> str:
    """Validates and sanitises a supply-chain node ID string.

    Accepts only alphanumeric characters, hyphens, and underscores to
    prevent log injection and downstream string-formatting attacks.

    Args:
        node_id: Raw node identifier.
        field_name: Field name for error messages.

    Returns:
        Sanitised string node ID.

    Raises:
        ValueError: If node_id contains disallowed characters or is too long.
    """
    if not isinstance(node_id, str):
        node_id = str(node_id)
    if len(node_id) > 64:
        raise ValueError(f"{field_name} too long (max 64 chars): {len(node_id)}")
    if not re.match(r'^[A-Za-z0-9_\-]+$', node_id):
        raise ValueError(
            f"{field_name} contains invalid characters (alphanumeric, _, - only): {node_id!r}"
        )
    return node_id


def safe_get_env(var_name: str, default: Optional[str] = None) -> Optional[str]:
    """Safely reads an environment variable, never logging its value.

    Args:
        var_name: Environment variable name.
        default: Default value if variable is not set.

    Returns:
        Variable value or default. Logs presence (not value) at DEBUG level.
    """
    value = os.environ.get(var_name, default)
    if value is not None and value != default:
        logger.debug("Environment variable %s is set (%s)", var_name, _redact(str(value)))
    else:
        logger.debug("Environment variable %s not set, using default", var_name)
    return value


def validate_optimizer_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validates a full optimizer configuration dictionary.

    Checks all weight parameters and solver settings, returning a cleaned
    config dict with all values coerced to valid types.

    Args:
        config: Raw configuration dict (e.g. loaded from YAML).

    Returns:
        Validated and cleaned configuration dict.

    Raises:
        ValueError: If any field fails validation.
    """
    validated: Dict[str, Any] = {}
    weights = config.get("weights", {})
    validated["weights"] = {
        "lambda_spoilage":    validate_lambda_weight(weights.get("lambda_spoilage", 10.0), "lambda_spoilage"),
        "lambda_cost":        validate_lambda_weight(weights.get("lambda_cost", 0.15), "lambda_cost"),
        "lambda_unmet_demand": validate_lambda_weight(weights.get("lambda_unmet_demand", 15.0), "lambda_unmet_demand"),
        "lambda_risk":        validate_lambda_weight(weights.get("lambda_risk", 5.0), "lambda_risk"),
    }
    solver = config.get("solver", {})
    tl = float(solver.get("time_limit_seconds", 30.0))
    if tl <= 0 or tl > 3600:
        raise ValueError(f"time_limit_seconds must be in (0, 3600], got: {tl}")
    validated["solver"] = {"time_limit_seconds": tl}
    return validated
