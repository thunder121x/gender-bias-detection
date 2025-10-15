from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parent.parent.parent

OUTPUT_DIR = Path(ROOT_PATH, "output")
OUTPUT_DIR_TIKTOK = Path(OUTPUT_DIR, "tiktok")
OUTPUT_DIR_YOUTUBE = Path(OUTPUT_DIR, "youtube")

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"