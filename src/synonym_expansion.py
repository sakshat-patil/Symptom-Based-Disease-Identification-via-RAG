"""
Symptom synonym expansion for bridging the vocabulary gap between Kaggle
snake_case tokens and MedQuAD clinical prose.

Maps normalised Kaggle symptoms (e.g. ``muscle_pain``) to their clinical
equivalents (e.g. ``myalgia``, ``myositis``). Built from a condensed UMLS /
MeSH subset for the 41 diseases covered by the Kaggle training set.

Why this exists
---------------
Check-in 3 error analysis showed dense retrieval collapsed to 7.5% Recall@1
because the general-purpose encoder did not connect ``high_fever`` with
``pyrexia``. Expanding the query with clinical synonyms before encoding
should let even a non-biomedical encoder match MedQuAD passages.

Usage
-----
    from src.synonym_expansion import expand_query

    expand_query(["fever", "muscle_pain"])
    # -> ["fever", "pyrexia", "muscle_pain", "myalgia"]
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Curated mapping: Kaggle snake_case  →  clinical synonyms
# ---------------------------------------------------------------------------
SYMPTOM_SYNONYMS: dict[str, list[str]] = {
    "abdominal_pain":           ["stomach pain", "epigastric pain", "abdominalgia"],
    "acid_regurgitation":       ["regurgitation", "reflux", "water brash"],
    "acidity":                  ["hyperacidity", "acid dyspepsia"],
    "altered_sensorium":        ["altered mental status", "confusion", "delirium"],
    "anxiety":                  ["nervousness", "apprehension"],
    "back_pain":                ["dorsalgia", "lumbago", "lower back pain"],
    "belly_pain":               ["abdominal pain", "abdominalgia"],
    "blister":                  ["vesicle", "bulla"],
    "bloody_stool":             ["hematochezia", "melena"],
    "blurred_vision":           ["visual blurring", "decreased visual acuity"],
    "breathlessness":           ["dyspnea", "shortness of breath", "air hunger"],
    "burning_urination":        ["dysuria", "urinary burning"],
    "chest_pain":               ["angina", "precordial pain", "thoracic pain"],
    "chills":                   ["rigors", "shivering"],
    "cold_hands_and_feets":     ["peripheral cyanosis", "acrocyanosis"],
    "constipation":             ["obstipation", "dyschezia"],
    "continuous_sneezing":      ["rhinorrhea", "sternutation"],
    "cough":                    ["tussis"],
    "cramps":                   ["muscle spasm", "myalgia"],
    "dark_urine":               ["bilirubinuria", "choluria"],
    "dehydration":              ["hypovolemia", "volume depletion"],
    "depression":               ["low mood", "anhedonia", "dysphoria"],
    "diarrhoea":                ["diarrhea", "loose stools"],
    "dischromic_patches":       ["hypopigmentation", "hyperpigmentation"],
    "dizziness":                ["vertigo", "lightheadedness"],
    "enlarged_thyroid":         ["goiter"],
    "excessive_hunger":         ["polyphagia", "hyperphagia"],
    "extra_marital_contacts":   [],
    "family_history":           [],
    "fast_heart_rate":          ["tachycardia"],
    "fatigue":                  ["asthenia", "lassitude", "malaise"],
    "fluid_overload":           ["edema", "volume overload"],
    "foul_smell_of urine":      ["urine malodor"],
    "headache":                 ["cephalalgia", "cephalgia"],
    "high_fever":               ["pyrexia", "hyperthermia", "febrile"],
    "hip_joint_pain":           ["coxalgia", "coxitis"],
    "indigestion":              ["dyspepsia"],
    "increased_appetite":       ["polyphagia"],
    "irregular_sugar_level":    ["dysglycemia", "glycemic lability"],
    "irritability":             ["agitation"],
    "itching":                  ["pruritus"],
    "joint_pain":               ["arthralgia", "polyarthralgia"],
    "knee_pain":                ["gonalgia", "knee arthralgia"],
    "lack_of_concentration":    ["inattention", "cognitive dysfunction"],
    "lethargy":                 ["hypersomnolence", "listlessness"],
    "light_sensitivity":        ["photophobia"],
    "loss_of_appetite":         ["anorexia", "hyporexia"],
    "loss_of_balance":          ["ataxia", "disequilibrium"],
    "loss_of_smell":            ["anosmia", "hyposmia"],
    "malaise":                  ["general ill feeling"],
    "mild_fever":                ["low-grade pyrexia"],
    "mood_swings":              ["affective lability"],
    "movement_stiffness":       ["rigidity", "muscle rigidity"],
    "muscle_pain":              ["myalgia", "myositis"],
    "muscle_weakness":          ["asthenia", "paresis"],
    "nausea":                   ["emesis precursor", "queasiness"],
    "obesity":                  ["adiposity", "overweight"],
    "pain_behind_the_eyes":     ["retro-orbital pain"],
    "pain_during_bowel_movements":["dyschezia"],
    "pain_in_anal_region":      ["proctalgia"],
    "palpitations":             ["cardiac palpitations", "irregular heartbeat"],
    "passage_of_gases":         ["flatulence", "meteorism"],
    "phlegm":                   ["sputum", "mucus"],
    "polyuria":                 ["frequent urination", "excessive urination"],
    "pus_filled_pimples":       ["pustules"],
    "rash":                     ["eruption", "exanthema"],
    "red_spots_over_body":      ["petechiae", "purpura"],
    "redness_of_eyes":          ["conjunctival injection", "conjunctivitis"],
    "restlessness":             ["akathisia", "psychomotor agitation"],
    "runny_nose":               ["rhinorrhea"],
    "rusty_sputum":             ["hemoptysis"],
    "shivering":                ["rigors", "tremulousness"],
    "skin_peeling":             ["desquamation"],
    "skin_rash":                ["exanthema", "cutaneous eruption"],
    "slurred_speech":           ["dysarthria"],
    "small_dents_in_nails":     ["nail pitting"],
    "spinning_movements":       ["vertigo"],
    "stiff_neck":               ["nuchal rigidity"],
    "stomach_bleeding":         ["upper gastrointestinal bleeding", "hematemesis"],
    "stomach_pain":             ["abdominal pain", "gastralgia"],
    "sunken_eyes":              ["enophthalmos"],
    "sweating":                 ["diaphoresis", "hyperhidrosis"],
    "swelled_lymph_nodes":      ["lymphadenopathy"],
    "swelling_joints":          ["joint effusion", "arthritis"],
    "swelling_of_stomach":      ["abdominal distension", "ascites"],
    "swollen_extremeties":      ["peripheral edema"],
    "throat_irritation":        ["pharyngeal irritation", "sore throat"],
    "toxic_look_(typhos)":      ["toxic appearance", "septic appearance"],
    "ulcers_on_tongue":         ["glossal ulcers", "aphthous ulcers"],
    "unsteadiness":             ["ataxia"],
    "visual_disturbances":      ["scotoma", "photopsia"],
    "vomiting":                 ["emesis"],
    "watering_from_eyes":       ["epiphora", "lacrimation"],
    "weakness_in_limbs":        ["paresis"],
    "weakness_of_one_body_side":["hemiparesis"],
    "weight_gain":              ["adiposity"],
    "weight_loss":              ["cachexia", "emaciation"],
    "yellow_crust_ooze":        ["impetiginous crust"],
    "yellow_urine":              ["bilirubinuria"],
    "yellowing_of_eyes":        ["scleral icterus"],
    "yellowish_skin":           ["jaundice", "icterus"],
}


def normalise(symptom: str) -> str:
    """Lowercase + underscore-normalise a symptom token."""
    return symptom.strip().lower().replace("-", "_").replace(" ", "_")


def synonyms_for(symptom: str) -> list[str]:
    """Return the clinical synonyms for a single symptom token.

    Returns an empty list if the symptom has no registered synonyms.
    """
    return list(SYMPTOM_SYNONYMS.get(normalise(symptom), []))


def expand_query(symptoms: list[str] | str) -> list[str]:
    """Expand a symptom list with registered clinical synonyms.

    Parameters
    ----------
    symptoms:
        Either a list of symptoms or a comma-separated string.

    Returns
    -------
    list[str]
        The original symptoms (lower-cased, underscore-normalised) followed
        by all known synonyms. Duplicates are removed while preserving
        insertion order.
    """
    if isinstance(symptoms, str):
        symptoms = [s for s in symptoms.split(",") if s.strip()]

    expanded: list[str] = []
    seen: set[str] = set()
    for s in symptoms:
        s_norm = normalise(s)
        if s_norm and s_norm not in seen:
            expanded.append(s_norm.replace("_", " "))
            seen.add(s_norm)
        for syn in synonyms_for(s):
            if syn not in seen:
                expanded.append(syn)
                seen.add(syn)

    return expanded


def expand_query_string(symptoms: list[str] | str) -> str:
    """Expand a symptom list and join with spaces — ready for the encoder."""
    return " ".join(expand_query(symptoms))


def coverage_report(symptoms: list[str]) -> dict:
    """Count how many of the given symptoms have registered synonyms."""
    total = len(symptoms)
    covered = sum(1 for s in symptoms if synonyms_for(s))
    return {
        "total":    total,
        "covered":  covered,
        "coverage": round(covered / total, 3) if total else 0.0,
    }
