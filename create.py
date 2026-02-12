from pathlib import Path

# You are already inside "emitr project"
BASE_DIR = Path.cwd()

folders = [
    "data",
    "src",
    "outputs",
]

files = [
    "data/sample_transcript.txt",

    "src/preprocess.py",
    "src/ner.py",
    "src/summarizer.py",
    "src/keywords.py",
    "src/sentiment_intent.py",
    "src/soap.py",
    "src/pipeline.py",

    "run_pipeline.py",
    "README.md",
]

# requirements.txt already exists, so we skip it

def create_structure():
    # Create folders
    for folder in folders:
        (BASE_DIR / folder).mkdir(parents=True, exist_ok=True)

    # Create files (only if they don't exist)
    for file in files:
        path = BASE_DIR / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)

    print(f"✅ Folder structure created in: {BASE_DIR.resolve()}")

if __name__ == "__main__":
    create_structure()
