"""
PaceAI Customize UI -- versioned, validated presentation configuration.

This module is pure Python (no Streamlit) so the load/save/validate logic is
unit-testable. It stores ONLY presentation choices (brand text, hero copy,
theme tokens, section spacing) in a JSON file under ``data/``. It never touches
analysis results, model parameters, detection, pose or biomechanical values.

The saved design is applied to the real dashboard by the page code via
``css_overrides()`` (a CSS-only override block) and by substituting the brand /
hero strings at render time. Config is treated as untrusted input: it is
validated on read and on import, and CSS overrides interpolate only validated
values.
"""
import hashlib
import json
import os
import re
import shutil

from . import config

SCHEMA_VERSION = 1

# Overridable for tests and alternate installs.
CONFIG_PATH = os.environ.get(
    "PACEAI_UI_DESIGN_FILE",
    os.path.join(config.DATA_DIR, "ui_design.json"),
)

_HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")

# Allow-listed keys and per-key validation. Anything not in this schema is a
# validation error. Structure is fixed (no user-chosen keys / components).
_PALETTE_KEYS = ("accent", "background", "surface", "muted", "line")
_BRAND_KEYS = ("name", "subtitle")
_HERO_KEYS = ("enabled", "headline", "subtitle")

_STRING_LIMITS = {
    ("brand", "name"): 24,
    ("brand", "subtitle"): 60,
    ("hero", "headline"): 80,
    ("hero", "subtitle"): 160,
}


def default_config() -> dict:
    """Factory for the pristine dashboard design -- one canonical source so a
    'reset' and a fresh install always produce the identical object."""
    return {
        "schema_version": SCHEMA_VERSION,
        "brand": {"name": "PACEAI", "subtitle": "Cricket bowling biomechanics"},
        "hero": {
            "enabled": True,
            "headline": "Fast-Bowling Biomechanics AI",
            "subtitle": "Kinematic Chain Profiling - ICC Elbow Screening - "
                        "Literature-Informed Biomechanical Risk",
        },
        "theme": {
            "accent": "#20D9FF",
            "background": "#212833",
            "surface": "#232b38",
            "muted": "#aeb9c8",
            "line": "#3d4859",
        },
        "section_gap_px": 66,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(cfg: dict) -> tuple[bool, str]:
    """Return (ok, error). Rejects unknown keys, malformed values, oversized
    text and out-of-range integers so a hand-edited or imported file can never
    inject something the CSS renderer does not expect."""
    if not isinstance(cfg, dict):
        return False, "top level must be an object"
    defaults = default_config()
    if cfg.get("schema_version") != SCHEMA_VERSION:
        return False, f"unsupported schema_version {cfg.get('schema_version')!r}"

    allowed_keys = {"schema_version", "brand", "hero", "theme", "section_gap_px"}
    unknown = set(cfg) - allowed_keys
    if unknown:
        return False, f"unknown keys {sorted(unknown)}"

    brand = cfg.get("brand")
    if not isinstance(brand, dict) or set(brand) != set(_BRAND_KEYS):
        return False, "'brand' must be an object with exactly {name, subtitle}"
    hero = cfg.get("hero")
    if not isinstance(hero, dict) or set(hero) != set(_HERO_KEYS):
        return False, "'hero' must be an object with exactly {enabled, headline, subtitle}"
    if not isinstance(hero["enabled"], bool):
        return False, "'hero.enabled' must be a boolean"
    theme = cfg.get("theme")
    if not isinstance(theme, dict) or set(theme) != set(_PALETTE_KEYS):
        return False, f"'theme' must be an object with exactly {sorted(_PALETTE_KEYS)}"

    for key, value in list(brand.items()) + list(hero.items())[1:] + list(theme.items()):
        if not isinstance(value, str):
            return False, f"{key!r} must be a string"
    for hex_key in _PALETTE_KEYS:
        if not _HEX_COLOR.match(theme[hex_key]):
            return False, f"'theme.{hex_key}' must be a #RRGGBB colour"
    for (group, key), limit in _STRING_LIMITS.items():
        value = cfg[group][key]
        if len(value) > limit:
            return False, f"'{group}.{key}' exceeds {limit} characters"
        if not value.strip():
            return False, f"'{group}.{key}' cannot be empty"

    gap = cfg.get("section_gap_px")
    if not isinstance(gap, int) or isinstance(gap, bool) or not 0 <= gap <= 160:
        return False, "'section_gap_px' must be an integer in 0..160"
    return True, ""


def _validated_or_default(raw) -> dict:
    ok, _ = validate(raw)
    return raw if ok else default_config()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def load() -> dict:
    """Read the saved design. Missing file -> defaults (never writes, so a
    routine rerun can't clobber a user design). Corrupt file -> defaults kept,
    with the bad copy parked aside so no work is lost."""
    if not os.path.exists(CONFIG_PATH):
        return default_config()
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if validate(raw)[0]:
            return raw
    except (OSError, ValueError, TypeError):
        pass
    # Malformed: keep a copy for inspection, then fall back to defaults.
    try:
        shutil.copyfile(CONFIG_PATH, CONFIG_PATH + ".corrupt")
    except OSError:
        pass
    return default_config()


def save(cfg: dict) -> tuple[bool, str]:
    """Validate then write atomically (temp file + os.replace) with a backup
    of the previous good design. Returns (ok, error)."""
    ok, err = validate(cfg)
    if not ok:
        return False, err
    os.makedirs(os.path.dirname(CONFIG_PATH) or ".", exist_ok=True)
    # Backup the last good design before replacing it.
    if os.path.exists(CONFIG_PATH):
        try:
            shutil.copyfile(CONFIG_PATH, CONFIG_PATH + ".bak")
        except OSError:
            pass
    tmp = CONFIG_PATH + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)
        os.replace(tmp, CONFIG_PATH)
        return True, ""
    except OSError as exc:
        return False, f"could not write design file: {exc}"


def reset_to_default() -> tuple[bool, str]:
    return save(default_config())


def export(cfg: dict) -> str:
    """Serialise the validated design for export."""
    return json.dumps(_validated_or_default(cfg), indent=2)


def import_config(text: str) -> tuple[dict, str]:
    """Parse and validate an imported design. Returns (config, error)."""
    try:
        raw = json.loads(text)
    except (ValueError, TypeError) as exc:
        return {}, f"not valid JSON: {exc}"
    ok, err = validate(raw)
    if not ok:
        return {}, err
    return raw, ""


# ---------------------------------------------------------------------------
# Renderer helpers (pure string builders -- no Streamlit imports)
# ---------------------------------------------------------------------------

def css_overrides(cfg: dict) -> str:
    """CSS-only override block injected after the app's own stylesheet.

    Interpolates ONLY validated values (config was checked by parse/validate),
    so this never evaluates user-supplied code -- colours have a fixed format
    and the gap is an integer. Sets the shared ``--app-*`` token palette (used
    by every shell rule) and the result page's ``--pai-*`` tokens so one saved
    design re-skins the whole dashboard."""
    t = cfg["theme"]
    return (
        "<style>\n"
        "  :root {"
        f" --app-bg:{t['background']}; --app-surface:{t['surface']};"
        f" --app-line:{t['line']}; --app-accent:{t['accent']};"
        f" --app-accent-text:{t['accent']}; --app-muted:{t['muted']};"
        " }\n"
        "  .pai-root {"
        f" --pai-accent:{t['accent']}; --pai-bg:{t['background']};"
        f" --pai-surface:{t['surface']}; --pai-muted:{t['muted']};"
        f" --pai-line:{t['line']};"
        " }\n"
        f"  .pai-sec {{ margin-top:{cfg['section_gap_px']}px !important; }}\n"
        "</style>\n"
    )


def fingerprint(cfg: dict) -> str:
    """Stable id for the loaded design (used to detect an unsaved draft)."""
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest()[:12]