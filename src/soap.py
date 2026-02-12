from typing import Dict, Any


def build_soap_note(extracted: Dict[str, Any]) -> Dict[str, Any]:
    """
    Uses extracted medical summary fields to populate SOAP.
    """
    symptoms = extracted.get("Symptoms", [])
    diagnosis = extracted.get("Diagnosis", None)
    treatment = extracted.get("Treatment", [])
    prognosis = extracted.get("Prognosis", None)

    hpi = extracted.get("HPI", None)
    objective = extracted.get("Physical_Exam", None)

    return {
        "Subjective": {
            "Chief_Complaint": ", ".join(symptoms) if symptoms else None,
            "History_of_Present_Illness": hpi
        },
        "Objective": {
            "Physical_Exam": objective,
            "Observations": extracted.get("Observations", None)
        },
        "Assessment": {
            "Diagnosis": diagnosis,
            "Severity": extracted.get("Severity", "Mild, improving")
        },
        "Plan": {
            "Treatment": treatment if treatment else None,
            "Follow-Up": extracted.get("Follow_Up", "Return if symptoms worsen or persist.")
        },
        "Prognosis": prognosis
    }
