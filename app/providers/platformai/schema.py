"""Tool JSON-Schema rewriting for the PlatformAI gateway.

The gateway rejects parts of JSON Schema that tool definitions routinely use
(confirmed on v3: object-form ``additionalProperties`` and fractional numeric
bounds fail its own validation, ``type`` arrays are rejected by Anthropic
models behind it; keywords outside an allow-list are dropped as well),
so each is rewritten to an equivalent or strictly more permissive form: the
relaxed schema never forbids a value the original allowed.
"""

from __future__ import annotations

import math
from typing import Any, Dict

_SUBSCHEMA_KEYS = {
    "items", "contains", "not", "if", "then", "else", "propertyNames",
    "unevaluatedItems", "unevaluatedProperties", "contentSchema",
}
_SUBSCHEMA_LIST_KEYS = {"anyOf", "oneOf", "allOf", "prefixItems"}
_SUBSCHEMA_MAP_KEYS = {"properties", "patternProperties", "$defs", "definitions"}
_PASSTHROUGH_KEYS = {
    "type", "enum", "const", "default", "examples", "example", "required",
    "dependentRequired", "format", "pattern", "title", "description",
    "readOnly", "writeOnly", "contentEncoding", "contentMediaType",
    "multipleOf", "uniqueItems", "discriminator", "$schema", "$id", "$ref",
    "$anchor", "$dynamicRef", "$dynamicAnchor",
}
_LOWER_BOUNDS = {"minimum", "exclusiveMinimum", "minLength", "minItems", "minProperties", "minContains"}
_UPPER_BOUNDS = {"maximum", "exclusiveMaximum", "maxLength", "maxItems", "maxProperties", "maxContains"}

_MAX_SAFE_INT = 2 ** 53 - 1


def _is_obj(value: Any) -> bool:
    return isinstance(value, dict)


def _bound(key: str, value: float):
    """Round a bound outward to a whole number; drop it when out of range."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    rounded = math.floor(value) if key in _LOWER_BOUNDS else math.ceil(value)
    return int(rounded) if abs(rounded) <= _MAX_SAFE_INT else None


def _boolean_exclusive_bounds(schema: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(schema)
    for exclusive, inclusive in (("exclusiveMinimum", "minimum"), ("exclusiveMaximum", "maximum")):
        flag = out.get(exclusive)
        if not isinstance(flag, bool):
            continue
        out.pop(exclusive)
        if flag and isinstance(out.get(inclusive), (int, float)) and not isinstance(out.get(inclusive), bool):
            out[exclusive] = out.pop(inclusive)
    return out


def _legacy_spellings(schema: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(schema)
    if "nullable" in out:
        nullable = out.pop("nullable")
        type_ = out.get("type")
        if nullable is True and isinstance(type_, str):
            out["type"] = [type_, "null"]
        elif nullable is True and isinstance(type_, list):
            out["type"] = [*type_, "null"]
    if isinstance(out.get("items"), list):
        items = out.pop("items")
        additional = out.pop("additionalItems", None)
        out["prefixItems"] = items
        if additional is not None:
            out["items"] = additional
    else:
        out.pop("additionalItems", None)
    if "dependencies" in out:
        deps = out.pop("dependencies")
        required = {k: v for k, v in deps.items() if isinstance(v, list)} if _is_obj(deps) else {}
        if required and "dependentRequired" not in out:
            out["dependentRequired"] = required
    return out


def _type_array(schema: Dict[str, Any]) -> Dict[str, Any]:
    type_ = schema.get("type")
    if not isinstance(type_, list):
        return schema
    out = {k: v for k, v in schema.items() if k != "type"}
    names = list(dict.fromkeys(t for t in type_ if isinstance(t, str)))
    if not names:
        return out
    if len(names) == 1:
        out["type"] = names[0]
        return out
    any_of = [{"type": name} for name in names]
    if "anyOf" not in out:
        out["anyOf"] = any_of
        return out
    all_of = out.get("allOf") if isinstance(out.get("allOf"), list) else []
    out["allOf"] = [*all_of, {"anyOf": any_of}]
    return out


def _subschema(value: Any) -> Any:
    if value is True:
        return {}
    if value is False:
        return {"not": {}}
    return _sanitize(value)


def _sanitize(schema: Any) -> Any:
    if isinstance(schema, list):
        return [_sanitize(item) for item in schema]
    if not _is_obj(schema):
        return schema
    normalized = _type_array(_legacy_spellings(_boolean_exclusive_bounds(schema)))
    out: Dict[str, Any] = {}
    for key, value in normalized.items():
        if key == "additionalProperties":
            out[key] = value if isinstance(value, bool) else True
        elif key in _LOWER_BOUNDS or key in _UPPER_BOUNDS:
            bound = _bound(key, value)
            if bound is not None:
                out[key] = bound
        elif key in _SUBSCHEMA_KEYS:
            out[key] = _subschema(value)
        elif key in _SUBSCHEMA_LIST_KEYS and isinstance(value, list):
            out[key] = [_subschema(item) for item in value]
        elif key in _SUBSCHEMA_MAP_KEYS and _is_obj(value):
            out[key] = {name: _subschema(sub) for name, sub in value.items() if name != ""}
        elif key == "dependentRequired" and _is_obj(value):
            out[key] = {name: req for name, req in value.items() if name != ""}
        elif key in _PASSTHROUGH_KEYS:
            out[key] = value
    return out


def sanitize_tool_schema(schema: Any) -> Dict[str, Any]:
    """Rewrite a tool's ``parameters`` schema into the form the gateway accepts."""
    if schema is None:
        return {"type": "object", "properties": {}}
    result = _sanitize(schema)
    if not _is_obj(result):
        return {"type": "object", "properties": {}}
    if not isinstance(result.get("type"), str):
        return {"type": "object", **result}
    return result
