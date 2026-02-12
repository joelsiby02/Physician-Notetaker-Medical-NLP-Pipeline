import re
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class Turn:
    speaker: str
    text: str


def normalize_text(text: str) -> str:
    text = text.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_turns(transcript: str) -> List[Turn]:
    """
    Splits text like:
    Physician: ...
    Patient: ...
    into a list of Turn(s).

    Handles bracket sections like [Physical Examination Conducted] as "SYSTEM".
    """
    transcript = normalize_text(transcript)

    # Convert bracket blocks into explicit lines
    transcript = re.sub(r"\[(.*?)\]", r"\nSYSTEM: \1\n", transcript)

    pattern = r"(Physician|Doctor|Patient|SYSTEM)\s*:\s*"
    parts = re.split(pattern, transcript)

    # re.split gives: [prefix, speaker, text, speaker, text, ...]
    turns = []
    i = 1
    while i < len(parts) - 1:
        speaker = parts[i].strip()
        text = parts[i + 1].strip()
        if text:
            turns.append(Turn(speaker=speaker, text=text))
        i += 2

    return turns


def group_by_speaker(turns: List[Turn]) -> Dict[str, str]:
    grouped = {}
    for t in turns:
        grouped.setdefault(t.speaker, [])
        grouped[t.speaker].append(t.text)

    return {k: " ".join(v) for k, v in grouped.items()}
