from typing import Dict, Any
from transformers import pipeline


def load_summarizer(model_name: str = "google/flan-t5-base"):
    return pipeline("text2text-generation", model=model_name)


def medical_summary_structured(transcript: str) -> Dict[str, Any]:
    summarizer = load_summarizer()

    prompt = f"""
Write a short clinical summary (5-7 lines) of this physician-patient transcript.
Include:
- Accident details
- Symptoms
- Diagnosis
- Treatment
- Current status
- Prognosis

Transcript:
{transcript}
"""

    out = summarizer(prompt, max_new_tokens=220, do_sample=False)[0]["generated_text"]

    return {"Model_Summary_Text": out}
