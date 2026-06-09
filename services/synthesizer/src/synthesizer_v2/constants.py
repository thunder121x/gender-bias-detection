from pathlib import Path

# Package root: services/synthesizer_v2/src/synthesizer_v2/
_PKG_ROOT = Path(__file__).resolve().parents[2]  # services/synthesizer_v2/

ASSETS_DIR = _PKG_ROOT / "assets"
OUTPUT_DIR = _PKG_ROOT / "output"

ANNOTATION_GUIDELINE = ASSETS_DIR / "annotation-guideline.md"

DEFAULT_MODEL = "gemini-2.5-flash-lite"
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"  # kept for reference
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_ENV_FILE = _PKG_ROOT / ".env"

BATCH_SIZE = 25  # items per API call

# ---------------------------------------------------------------------------
# GB label values
# ---------------------------------------------------------------------------
GB_LABELS = ("GB-ATTACK", "GB-NORMATIVE", "GB-SEX")
NON_GB_SUBTYPES = ("neutral", "meta_counter", "gendered_insult")

# ---------------------------------------------------------------------------
# Modes exposed via CLI
# ---------------------------------------------------------------------------
MODES = ("gb-attack", "gb-normative", "gb-sex", "non-gb-neutral", "non-gb-meta", "non-gb-insult")
