from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parent.parent.parent

OUTPUT_DIR = ROOT_PATH / "output"
ASSETS_DIR = ROOT_PATH / "assets"
# Use an absolute path so the script works no matter the current working directory
PROMPT_DIR = ROOT_PATH / "src" / "prompteng" / "prompt"
