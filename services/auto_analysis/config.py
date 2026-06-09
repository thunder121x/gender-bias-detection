"""Configuration for auto-analysis service."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
ENV_FILE = Path(__file__).parent / ".env"
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
# Using auto_analysis directory for assets (where data actually is)
ASSETS_DIR = Path(__file__).parent / "assets"
GUIDELINE_PATH = ASSETS_DIR / "prompt" / "annotation" / "annotation-guideline.md"
SCRAPED_DATA_PATH = ASSETS_DIR / "scraped_data.yaml"
OUTPUT_DIR = Path(__file__).parent / "output"

# API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-3.1-flash-lite-preview"  # Fastest, lightweight model

# Processing Configuration
BATCH_SIZE = 100  # Items per batch
MAX_CONCURRENT_REQUESTS = 10  # Max concurrent Gemini API calls
TIMEOUT_SECONDS = 120  # Increased from 60s

# Labels for validation
VALID_LABELS = {"neutral", "GB-ATTACK", "GB-NORMATIVE", "GB-SEX", "meta_counter"}
