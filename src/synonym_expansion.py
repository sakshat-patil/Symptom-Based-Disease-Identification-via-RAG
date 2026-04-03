"""Hand-curated Kaggle-token -> clinical-synonym dictionary.

Check-in 3 surfaced the biggest single failure mode in our retrieval
pipeline: MedQuAD writes "myalgia" while our training symptoms are
"muscle_pain". A general sentence encoder doesn't know they refer to the
same concept, so retrieval recall cratered.

This module fixes that by appending the clinical synonyms to the query
string before we encode. It's cheap (no corpus re-embedding), it's
auditable (we maintain the dictionary by hand for now), and it gave us a
clean +0.05 R@10 lift in the ablation.

The 130-entry dictionary covers every Kaggle symptom token we saw during
ETL. A future pass should swap this for an automatic UMLS / MeSH lookup.

Owner: Aishwarya, with Vineet medically reviewing the entries.
"""
from __future__ import annotations

SYMPTOM_SYNONYMS: dict[str, list[str]] = {
    "muscle_pain": ["myalgia", "muscle ache"],
    "joint_pain": ["arthralgia", "polyarthralgia"],
    "high_fever": ["pyrexia", "hyperthermia"],
    "mild_fever": ["low-grade fever"],
    "breathlessness": ["dyspnea", "shortness of breath"],
    "yellowish_skin": ["jaundice", "icterus"],
    "yellowing_of_eyes": ["scleral icterus", "ocular jaundice"],
    "dark_urine": ["bilirubinuria"],
    "vomiting": ["emesis"],
    "headache": ["cephalalgia"],
    "nausea": ["queasiness"],
    "diarrhoea": ["diarrhea", "loose stools"],
    "constipation": ["obstipation"],
    "abdominal_pain": ["abdominalgia", "stomach pain"],
    "chest_pain": ["thoracic pain", "angina"],
    "fast_heart_rate": ["tachycardia", "palpitations"],
    "lightheadedness": ["presyncope"],
    "loss_of_appetite": ["anorexia"],
    "weight_loss": ["cachexia"],
    "fatigue": ["asthenia", "lethargy"],
    "skin_rash": ["dermatitis", "exanthem"],
    "itching": ["pruritus"],
    "swelling_joints": ["arthritis", "joint swelling"],
    "stiff_neck": ["nuchal rigidity"],
    "blurred_and_distorted_vision": ["blurred vision", "diplopia"],
    "burning_micturition": ["dysuria"],
    "polyuria": ["frequent urination"],
    "spinning_movements": ["vertigo"],
    "loss_of_balance": ["ataxia"],
    "irregular_sugar_level": ["hyperglycemia", "hypoglycemia"],
    "excessive_hunger": ["polyphagia"],
    "weight_gain": ["obesity"],
    "cough": ["expectoration"],
    "phlegm": ["sputum"],
    "blood_in_sputum": ["hemoptysis"],
    "watering_from_eyes": ["lacrimation", "epiphora"],
    "continuous_sneezing": ["paroxysmal sneezing", "rhinitis"],
    "runny_nose": ["rhinorrhea"],
    "congestion": ["nasal congestion"],
    "swelled_lymph_nodes": ["lymphadenopathy"],
    "red_spots_over_body": ["petechiae", "exanthema"],
    "pain_behind_the_eyes": ["retro-orbital pain"],
    "back_pain": ["lumbago"],
    "knee_pain": ["gonalgia"],
    "neck_pain": ["cervicalgia"],
    "swelling_of_stomach": ["abdominal distension"],
    "muscle_weakness": ["myasthenia"],
    "puffy_face_and_eyes": ["periorbital edema"],
    "swollen_legs": ["pedal edema"],
    "depression": ["depressive disorder"],
    "anxiety": ["anxiety disorder"],
    "irritability": ["agitation"],
    "altered_sensorium": ["altered mental status"],
    "weakness_of_one_body_side": ["hemiparesis"],
    "slurred_speech": ["dysarthria"],
    "abnormal_menstruation": ["menstrual irregularity", "dysmenorrhea"],
    "extra_marital_contacts": ["unprotected intercourse"],
    "receiving_blood_transfusion": ["transfusion exposure"],
    "receiving_unsterile_injections": ["needle exposure"],
    "history_of_alcohol_consumption": ["alcohol use disorder"],
    "acute_liver_failure": ["fulminant hepatic failure"],
    "stomach_bleeding": ["gastrointestinal hemorrhage"],
    "fluid_overload": ["volume overload"],
    "swollen_extremeties": ["peripheral edema"],
    "brittle_nails": ["onychoschizia"],
    "enlarged_thyroid": ["goiter"],
    "skin_peeling": ["desquamation"],
    "silver_like_dusting": ["psoriatic plaques"],
    "small_dents_in_nails": ["nail pitting"],
    "inflammatory_nails": ["paronychia"],
    "pus_filled_pimples": ["pustules"],
    "blackheads": ["comedones"],
    "scurring": ["scarring"],
    "blister": ["bulla", "vesicle"],
    "red_sore_around_nose": ["impetigo lesions"],
    "yellow_crust_ooze": ["honey-coloured crust"],
    "bladder_discomfort": ["bladder pain"],
    "foul_smell_of_urine": ["malodorous urine"],
    "continuous_feel_of_urine": ["urinary urgency"],
    "rusty_sputum": ["rust-coloured sputum"],
    "toxic_look_typhos": ["typhoid facies"],
    "belly_pain": ["abdominal pain"],
    "pain_during_bowel_movements": ["dyschezia"],
    "pain_in_anal_region": ["proctalgia"],
    "bloody_stool": ["hematochezia", "melena"],
    "irritation_in_anus": ["anal pruritus"],
    "swollen_blood_vessels": ["varicosities"],
    "prominent_veins_on_calf": ["varicose veins"],
    "cold_hands_and_feet": ["acrocyanosis"],
    "mood_swings": ["affective lability"],
    "restlessness": ["psychomotor agitation"],
    "sweating": ["diaphoresis"],
    "shivering": ["rigors"],
    "chills": ["rigors"],
    "dehydration": ["volume depletion"],
    "sunken_eyes": ["enophthalmos"],
    "ulcers_on_tongue": ["aphthous ulcer"],
    "patches_in_throat": ["pharyngitis", "thrush"],
    "muscle_wasting": ["sarcopenia"],
    "throat_irritation": ["pharyngeal irritation"],
    "loss_of_smell": ["anosmia"],
    "redness_of_eyes": ["conjunctival injection"],
    "sinus_pressure": ["sinusitis"],
    "visual_disturbances": ["scotoma"],
    "indigestion": ["dyspepsia"],
    "acidity": ["acid reflux", "heartburn"],
    "stomach_pain": ["gastralgia"],
    "movement_stiffness": ["rigidity"],
    "painful_walking": ["antalgic gait"],
    "hip_joint_pain": ["coxalgia"],
    "passage_of_gases": ["flatulence"],
    "internal_itching": ["visceral pruritus"],
    "spotting_urination": ["hematuria"],
    "yellow_urine": ["bilirubinuria"],
    "lack_of_concentration": ["cognitive impairment"],
    "dizziness": ["vertigo"],
    "weakness_in_limbs": ["limb paresis"],
    "obesity": ["adiposity"],
    "cramps": ["muscle cramps"],
    "bruising": ["ecchymoses"],
    "mucoid_sputum": ["mucopurulent sputum"],
    "family_history": ["familial predisposition"],
    "malaise": ["general unwellness"],
    "increased_appetite": ["hyperphagia"],
    "coma": ["comatose state"],
    "distention_of_abdomen": ["abdominal distension"],
    "unsteadiness": ["disequilibrium"],
    "palpitations": ["heart palpitations"],
    "dischromic_patches": ["hypopigmented patches"],
    "nodal_skin_eruptions": ["nodular skin lesions"],
    "swelling_joints": ["polyarthritis"],
}


def expand_tokens(symptoms: list[str]) -> list[str]:
    """Return the union of symptoms and their clinical synonyms."""
    out = list(symptoms)
    for s in symptoms:
        out.extend(SYMPTOM_SYNONYMS.get(s, []))
    return out


def expand_query_string(base_query: str, symptoms: list[str]) -> str:
    """Append clinical synonyms to a base query string."""
    extras = []
    for s in symptoms:
        extras.extend(SYMPTOM_SYNONYMS.get(s, []))
    if not extras:
        return base_query
    return f"{base_query} (clinical terms: {', '.join(extras)})"
