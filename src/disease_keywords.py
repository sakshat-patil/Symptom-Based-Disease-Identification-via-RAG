"""Map our 41 Kaggle disease labels to MedQuAD-friendly keyword lists.

A retrieved passage is attributed to a disease if any of that disease's
keywords appear in the passage's focus, question, or text. This bridges the
"common Kaggle name <-> formal MedQuAD article title" gap.
"""
from __future__ import annotations

DISEASE_KEYWORDS: dict[str, list[str]] = {
    "fungal_infection": ["fungal infection", "tinea", "candidiasis", "ringworm", "mycosis", "dermatophyt"],
    "allergy": ["allergy", "allergic", "allergies", "hay fever", "hypersensitivity"],
    "gerd": ["gerd", "gastroesophageal reflux", "acid reflux", "heartburn"],
    "chronic_cholestasis": ["cholestasis", "biliary", "bile duct", "primary biliary"],
    "drug_reaction": ["drug reaction", "drug allergy", "adverse drug", "drug-induced", "hypersensitivity reaction"],
    "peptic_ulcer_diseae": ["peptic ulcer", "gastric ulcer", "duodenal ulcer", "stomach ulcer"],
    "aids": ["hiv", "aids", "acquired immunodeficiency", "human immunodeficiency"],
    "diabetes": ["diabetes", "diabetic", "hyperglycemia", "type 1 diabetes", "type 2 diabetes"],
    "gastroenteritis": ["gastroenteritis", "stomach flu", "viral gastroenteritis"],
    "bronchial_asthma": ["asthma", "bronchial asthma"],
    "hypertension": ["hypertension", "high blood pressure", "elevated blood pressure"],
    "migraine": ["migraine", "migraine headache"],
    "cervical_spondylosis": ["cervical spondylosis", "neck arthritis", "cervical osteoarthritis"],
    "paralysis_brain_hemorrhage": ["stroke", "brain hemorrhage", "intracerebral hemorrhage", "cerebrovascular accident", "paralysis"],
    "jaundice": ["jaundice", "icterus", "hyperbilirubinemia"],
    "malaria": ["malaria", "plasmodium"],
    "chicken_pox": ["chickenpox", "chicken pox", "varicella"],
    "dengue": ["dengue", "dengue fever", "dengue hemorrhagic"],
    "typhoid": ["typhoid", "typhoid fever", "salmonella typhi"],
    "hepatitis_a": ["hepatitis a", "hepatitis-a"],
    "hepatitis_b": ["hepatitis b", "hepatitis-b"],
    "hepatitis_c": ["hepatitis c", "hepatitis-c"],
    "hepatitis_d": ["hepatitis d", "hepatitis-d"],
    "hepatitis_e": ["hepatitis e", "hepatitis-e"],
    "alcoholic_hepatitis": ["alcoholic hepatitis", "alcoholic liver"],
    "tuberculosis": ["tuberculosis", "mycobacterium tuberculosis"],
    "common_cold": ["common cold", "upper respiratory infection", "viral rhinitis"],
    "pneumonia": ["pneumonia", "pneumonitis"],
    "dimorphic_hemmorhoids": ["hemorrhoids", "haemorrhoids", "piles", "dimorphic hemorrhoids"],
    "heart_attack": ["heart attack", "myocardial infarction", "acute coronary"],
    "varicose_veins": ["varicose veins", "venous insufficiency"],
    "hypothyroidism": ["hypothyroidism", "underactive thyroid"],
    "hyperthyroidism": ["hyperthyroidism", "overactive thyroid", "graves disease"],
    "hypoglycemia": ["hypoglycemia", "low blood sugar"],
    "osteoarthristis": ["osteoarthritis", "degenerative joint disease"],
    "arthritis": ["arthritis", "rheumatoid arthritis", "joint inflammation"],
    "paroymsal_positional_vertigo": ["benign paroxysmal positional vertigo", "vertigo"],
    "acne": ["acne", "acne vulgaris"],
    "urinary_tract_infection": ["urinary tract infection", "cystitis", "bladder infection"],
    "psoriasis": ["psoriasis", "psoriatic"],
    "impetigo": ["impetigo"],
}


import re as _re

# Compile a word-boundary-aware matcher per keyword to avoid false positives
# like 'hav' (Hepatitis A virus abbrev) matching inside 'have'.
def _kw_matches(kw: str, text: str) -> bool:
    # If the keyword has spaces, do plain substring -- multi-word phrases
    # rarely appear inside another word.
    if " " in kw or "-" in kw:
        return kw in text
    # Otherwise require word boundaries.
    return bool(_re.search(rf"\b{_re.escape(kw)}\b", text))


def diseases_matching(text: str, universe: set[str]) -> list[str]:
    """Return all diseases whose keywords appear (case-insensitive) in text.

    Multi-word keywords use plain substring matching; single-token keywords
    require word boundaries so 'hav' doesn't fire on 'have'.
    """
    if not text:
        return []
    low = text.lower()
    out = []
    for d in universe:
        for kw in DISEASE_KEYWORDS.get(d, []):
            if _kw_matches(kw, low):
                out.append(d)
                break
    return out
