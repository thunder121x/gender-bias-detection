# 📘 Prompt Engineering & Annotation Pipeline (annotate.py)

This README explains how to install and run the local LLM annotation pipeline that powers **Thai gender-bias span-level annotation**.  
The system uses **Ollama** + **Qwen 2.5 models** to generate pre-annotations for your dataset.

Supported OS:
- **macOS** (Homebrew or official installer)
- **Windows** (Ollama Desktop)

---

## 🚀 1. Install Ollama

### **macOS — install via Homebrew (recommended)**

```bash
brew install ollama

Start the service:

ollama serve

Alternatively, download the installer:

👉 https://ollama.com/download

⸻

Windows — Ollama Desktop App
	1.	Download from:
👉 https://ollama.com/download
	2.	Install the app — Ollama service runs automatically in the background.
	3.	Verify:

ollama --version

No need to run ollama serve on Windows.

⸻

📥 2. Download Required LLM Model

We use Qwen 2.5 (7B) for Thai token-level annotation.

ollama pull qwen2.5:7b

Test the model:

ollama run qwen2.5:7b

If you see a prompt → you’re good.

⸻

📁 3. Project Structure

Your repo should contain the following:

project/
│
├── annotate.py
├── constants.py
├── prompts/
│   └── draft_1/
│       ├── system_prompt.txt
│       └── user_prompt.txt
├── assets/
│   └── testset.csv
└── output/
    └── (generated CSV files)

Make sure constants.py points to the correct paths:

PROMPT_DIR = Path("prompts")
OUTPUT_DIR = Path("output")
ASSETS_DIR = Path("assets")


⸻

🧩 4. Writing Your Prompts

Each prompt folder contains:

system_prompt.txt
user_prompt.txt

In user_prompt.txt, include the placeholder:

<PASTE_TOKEN_LIST_HERE>

This placeholder will be dynamically replaced by the script.

Example CSV (assets/testset.csv):

tokens
[“ฉัน”,“ชอบ”,“แมว”]
[“ไป”,“กิน”,“ข้าว”]


⸻

🏃‍♂️ 5. Running the Annotation Script

Same command for macOS / Linux / Windows:

python annotate.py \
    --model qwen2.5:7b \
    --prompt draft_1 \
    --input_csv assets/testset.csv \
    --output_csv output/testset_qwen2.5_7b.csv \
    --autosave

Expected console output:

--- Starting Annotation ---
Model: qwen2.5:7b
Prompt folder: prompts/draft_1
Input CSV: assets/testset.csv
Output CSV: output/testset_qwen2.5_7b.csv
Autosave: ON

[✓] Done!

Your annotated file will appear in output/.

⸻

✨ 6. Troubleshooting

❗ Missing system_prompt.txt

Check your prompt directory paths in constants.py.

❗ Python cannot find Ollama

macOS: restart your terminal after brew install.

❗ Annotation is too slow

Try a smaller model:

ollama pull qwen2.5:3b

Then:

python annotate.py --model qwen2.5:3b ...
