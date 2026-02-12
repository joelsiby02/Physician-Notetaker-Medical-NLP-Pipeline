from typing import List
from keybert import KeyBERT


def extract_keywords(text: str, top_n: int = 12) -> List[str]:
    kw_model = KeyBERT("all-MiniLM-L6-v2")
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 3),
        stop_words="english",
        top_n=top_n
    )
    return [k for k, score in keywords]
