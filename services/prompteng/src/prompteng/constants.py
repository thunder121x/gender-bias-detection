from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parent.parent.parent

OUTPUT_DIR = Path(ROOT_PATH, "output")
ASSETS_DIR = Path(ROOT_PATH, "assets")
PROMPT_DIR = Path("services/prompteng/src/prompteng/prompt")