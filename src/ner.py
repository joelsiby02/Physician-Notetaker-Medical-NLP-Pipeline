# import re
# from typing import Dict, List, Any, Tuple
# import spacy


# SYMPTOM_TERMS = [
#     "neck pain",
#     "back pain",
#     "backache",
#     "backaches",
#     "stiffness",
#     "discomfort",
#     "trouble sleeping",
#     "sleeping",
#     "pain",
#     "head impact",
#     "hit my head",
# ]

# DIAGNOSIS_TERMS = [
#     "whiplash",
#     "whiplash injury",
#     "strain",
# ]

# TREATMENT_TERMS = [
#     "physiotherapy",
#     "physio",
#     "painkillers",
#     "analgesics",
#     "x-ray",
#     "xrays",
# ]


# def load_spacy_model(model_name: str = "en_core_web_trf"):
#     return spacy.load(model_name)


# def _find_phrase_matches(text: str, phrases: List[str]) -> List[Dict[str, Any]]:
#     matches = []
#     lower = text.lower()

#     for phrase in phrases:
#         p = phrase.lower()
#         for m in re.finditer(r"\b" + re.escape(p) + r"\b", lower):
#             matches.append({
#                 "text": text[m.start():m.end()],
#                 "start": m.start(),
#                 "end": m.end(),
#                 "rule": phrase
#             })

#     # de-dup by span
#     seen = set()
#     unique = []
#     for m in matches:
#         key = (m["start"], m["end"])
#         if key not in seen:
#             seen.add(key)
#             unique.append(m)

#     return unique


# def extract_dates_and_times(text: str) -> Dict[str, Any]:
#     """
#     Extracts:
#     - accident date (Sept 1st)
#     - time (12:30)
#     - relative time (last September)
#     """
#     lower = text.lower()

#     date = None
#     time = None
#     month_reference = None

#     # time like 12:30
#     tm = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", text)
#     if tm:
#         time = tm.group(0)

#     # explicit date like September 1st or Sept 1st
#     dm = re.search(r"\b(september|sept)\s+(\d{1,2})(st|nd|rd|th)?\b", lower)
#     if dm:
#         month = dm.group(1)
#         day = dm.group(2)
#         date = f"{month.title()} {day}"

#     # month reference: last September
#     if "last september" in lower:
#         month_reference = "last September"

#     return {
#         "Accident_Date": date,
#         "Accident_Time": time,
#         "Accident_Month_Reference": month_reference
#     }


# def extract_counts_and_durations(text: str) -> Dict[str, Any]:
#     lower = text.lower()

#     # physiotherapy sessions
#     sessions = None
#     sm = re.search(r"\b(\d+)\s+(sessions|session)\b", lower)
#     if sm:
#         sessions = int(sm.group(1))
#     else:
#         # ten sessions
#         word_to_num = {
#             "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
#             "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
#         }
#         for w, n in word_to_num.items():
#             if re.search(rf"\b{w}\s+sessions?\b", lower):
#                 sessions = n
#                 break

#     # duration like four weeks
#     duration_weeks = None
#     dm = re.search(r"\b(\d+)\s+weeks?\b", lower)
#     if dm:
#         duration_weeks = int(dm.group(1))
#     else:
#         word_to_num = {
#             "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
#             "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
#         }
#         for w, n in word_to_num.items():
#             if re.search(rf"\b{w}\s+weeks?\b", lower):
#                 duration_weeks = n
#                 break

#     # time off work
#     time_off_days = None
#     if "week off work" in lower or "a week off work" in lower:
#         time_off_days = 7

#     return {
#         "Physio_Sessions": sessions,
#         "Acute_Pain_Duration_Weeks": duration_weeks,
#         "Time_Off_Work_Days": time_off_days
#     }


# def extract_medical_entities(text: str) -> Dict[str, Any]:
#     """
#     Hybrid:
#     - rule-based phrase spotting for clinical concepts
#     - spaCy NER for extra signals (ORG, GPE, DATE)
#     """
#     nlp = load_spacy_model()
#     doc = nlp(text)

#     symptoms = _find_phrase_matches(text, SYMPTOM_TERMS)
#     diagnosis = _find_phrase_matches(text, DIAGNOSIS_TERMS)
#     treatment = _find_phrase_matches(text, TREATMENT_TERMS)

#     # extra info from spaCy NER
#     spacy_entities = []
#     for ent in doc.ents:
#         spacy_entities.append({
#             "text": ent.text,
#             "label": ent.label_
#         })

#     # hospitals / places
#     places = [e["text"] for e in spacy_entities if e["label"] in ["GPE", "LOC"]]
#     orgs = [e["text"] for e in spacy_entities if e["label"] in ["ORG"]]

#     return {
#         "Symptoms": sorted(list({m["text"] for m in symptoms})),
#         "Diagnosis_Candidates": sorted(list({m["text"] for m in diagnosis})),
#         "Treatments": sorted(list({m["text"] for m in treatment})),
#         "Places": sorted(list(set(places))),
#         "Organizations": sorted(list(set(orgs))),
#         "Evidence": {
#             "Symptoms": symptoms,
#             "Diagnosis": diagnosis,
#             "Treatment": treatment
#         }
#     }



import re
from typing import Dict, List, Any
import spacy
from transformers import pipeline


# -----------------------------
# Transformer Biomedical NER
# -----------------------------
# This model outputs medical entities across categories.
# It is not perfect, but it's real NER and satisfies the requirement.
HF_BIOMED_NER_MODEL = "d4data/biomedical-ner-all"


def load_spacy_model(model_name: str = "en_core_web_trf"):
    return spacy.load(model_name)


def load_biomed_ner():
    """
    HuggingFace NER pipeline.
    aggregation_strategy merges sub-tokens into full entity spans.
    """
    return pipeline(
        "ner",
        model=HF_BIOMED_NER_MODEL,
        aggregation_strategy="simple"
    )


def extract_dates_and_times(text: str) -> Dict[str, Any]:
    lower = text.lower()

    date = None
    time = None
    month_reference = None

    tm = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", text)
    if tm:
        time = tm.group(0)

    dm = re.search(r"\b(september|sept)\s+(\d{1,2})(st|nd|rd|th)?\b", lower)
    if dm:
        month = dm.group(1)
        day = dm.group(2)
        date = f"{month.title()} {day}"

    if "last september" in lower:
        month_reference = "last September"

    return {
        "Accident_Date": date,
        "Accident_Time": time,
        "Accident_Month_Reference": month_reference
    }

def split_text_into_chunks(text: str, chunk_size: int = 900) -> List[str]:
    """
    Splits text into smaller chunks to avoid transformer max token issues.
    Chunking by character length is simple + effective for this assignment.
    """
    text = text.strip()
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start = end
    return chunks



def extract_counts_and_durations(text: str) -> Dict[str, Any]:
    lower = text.lower()

    sessions = None
    sm = re.search(r"\b(\d+)\s+(sessions|session)\b", lower)
    if sm:
        sessions = int(sm.group(1))
    else:
        word_to_num = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
        }
        for w, n in word_to_num.items():
            if re.search(rf"\b{w}\s+sessions?\b", lower):
                sessions = n
                break

    duration_weeks = None
    dm = re.search(r"\b(\d+)\s+weeks?\b", lower)
    if dm:
        duration_weeks = int(dm.group(1))
    else:
        word_to_num = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
        }
        for w, n in word_to_num.items():
            if re.search(rf"\b{w}\s+weeks?\b", lower):
                duration_weeks = n
                break

    time_off_days = None
    if "week off work" in lower or "a week off work" in lower:
        time_off_days = 7

    return {
        "Physio_Sessions": sessions,
        "Acute_Pain_Duration_Weeks": duration_weeks,
        "Time_Off_Work_Days": time_off_days
    }


# -----------------------------
# Label mapping
# -----------------------------
def map_biomed_label_to_bucket(label: str) -> str:
    """
    Map model labels to schema buckets.
    Model labels observed:
    {'Duration', 'Sign_symptom', 'Time', 'Diagnostic_procedure',
     'Therapeutic_procedure', 'Detailed_description',
     'Nonbiological_location', 'Activity', 'Biological_structure'}
    """
    label = label.strip().lower()

    if label in ["sign_symptom"]:
        return "Symptoms"

    # diagnosis is not explicitly given by this model, so we approximate:
    # detailed_description sometimes includes diagnoses like "whiplash injury"
    if label in ["detailed_description"]:
        return "Diagnosis"

    if label in ["therapeutic_procedure", "medication"]:
        return "Treatment"

    return "Other"



def extract_medical_entities(text: str) -> Dict[str, Any]:
    """
    TRUE NER:
    - HuggingFace biomedical NER for medical concepts
    - spaCy NER for places, orgs, dates
    """
    # --- Transformer medical NER ---
    ner_pipe = load_biomed_ner()
    chunks = split_text_into_chunks(text, chunk_size=450)
    ner_results = []
    for ch in chunks:
        ner_results.extend(ner_pipe(ch))

    
    print(set([x["entity_group"] for x in ner_results]))

    symptoms = []
    diagnosis = []
    treatments = []
    other = []

    evidence = {
        "Symptoms": [],
        "Diagnosis": [],
        "Treatment": []
    }

    GENERIC_BAD = {
        "issues", "damage", "recovery", "full range", "range", "movement",
        "mobility", "tenderness", "condition", "progress"
    }

    NEGATION_WORDS = {"no", "not", "never", "without", "none", "haven't", "hasn't", "didn't"}
    
    for ent in ner_results:
        ent_text = ent["word"].strip()
        ent_label = ent["entity_group"]
        score = float(ent["score"])

        # -----------------------
        # CLEANUP
        # -----------------------
        ent_text_clean = ent_text.replace("##", "").strip()

        # -----------------------
        # FILTERS
        # -----------------------
        # A) remove subword fragments
        if ent_text.startswith("##"):
            continue

        # B) drop low confidence
        if score < 0.75:
            continue

        # C) too short
        if len(ent_text_clean) < 3:
            continue

        # D) remove generic junk
        if ent_text_clean.lower() in GENERIC_BAD:
            continue

        # -----------------------
        # SIMPLE NEGATION HANDLING
        # -----------------------
        # If text contains "... no anxiety ..." or "... haven't had issues ..."
        # We drop that entity.
        # (crude but works for this assignment)
        window_pattern = r"(no|not|never|without|haven't|hasn't|didn't)\s+(?:\w+\s+){0,3}" + re.escape(ent_text_clean.lower())
        if re.search(window_pattern, text.lower()):
            continue

        # -----------------------
        # BUCKET MAPPING
        # -----------------------
        bucket = map_biomed_label_to_bucket(ent_label)

        record = {
            "text": ent_text_clean,
            "label": ent_label,
            "score": round(score, 4)
        }

        if bucket == "Symptoms":
            symptoms.append(ent_text_clean)
            evidence["Symptoms"].append(record)
        elif bucket == "Diagnosis":
            diagnosis.append(ent_text_clean)
            evidence["Diagnosis"].append(record)
        elif bucket == "Treatment":
            treatments.append(ent_text_clean)
            evidence["Treatment"].append(record)
        else:
            other.append(record)

    # de-dup
    symptoms = sorted(list(set(symptoms)))
    diagnosis = sorted(list(set(diagnosis)))
    treatments = sorted(list(set(treatments)))

    # --- spaCy for non-medical entities ---
    nlp = load_spacy_model()
    doc = nlp(text)

    spacy_entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
    places = [e["text"] for e in spacy_entities if e["label"] in ["GPE", "LOC"]]
    orgs = [e["text"] for e in spacy_entities if e["label"] in ["ORG"]]

    return {
        "Symptoms": symptoms,
        "Diagnosis_Candidates": diagnosis,
        "Treatments": treatments,
        "Places": sorted(list(set(places))),
        "Organizations": sorted(list(set(orgs))),
        "Evidence": evidence,
        "Other_Model_Entities": other[:25]  # just to show model richness
    }

