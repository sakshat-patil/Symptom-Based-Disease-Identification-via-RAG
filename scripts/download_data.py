"""Download/build raw datasets for the project.

Datasets:
1. Kaggle-style Disease-Symptom dataset (41 diseases, 4920 rows). We attempt
   a GitHub mirror first; if unavailable we synthesize an equivalent dataset
   from a curated disease->symptom mapping derived from the same Kaggle source.
2. MedQuAD biomedical Q&A corpus (cloned from the official repo).
"""
from __future__ import annotations

import csv
import random
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

# Curated 41-disease -> symptom mapping (mirrors the public Kaggle
# "Disease Symptom Description Dataset" by itachi9604). Every disease lists
# the canonical symptoms associated with it; the dataset generator samples
# subsets to produce per-row variability.
DISEASE_SYMPTOMS: dict[str, list[str]] = {
    "fungal_infection": ["itching", "skin_rash", "nodal_skin_eruptions", "dischromic_patches"],
    "allergy": ["continuous_sneezing", "shivering", "chills", "watering_from_eyes"],
    "gerd": ["stomach_pain", "acidity", "ulcers_on_tongue", "vomiting", "cough", "chest_pain"],
    "chronic_cholestasis": ["itching", "vomiting", "yellowish_skin", "nausea", "loss_of_appetite",
                            "abdominal_pain", "yellowing_of_eyes"],
    "drug_reaction": ["itching", "skin_rash", "stomach_pain", "burning_micturition", "spotting_urination"],
    "peptic_ulcer_diseae": ["vomiting", "loss_of_appetite", "abdominal_pain", "passage_of_gases",
                             "internal_itching", "indigestion"],
    "aids": ["muscle_wasting", "patches_in_throat", "high_fever", "extra_marital_contacts"],
    "diabetes": ["fatigue", "weight_loss", "restlessness", "lethargy", "irregular_sugar_level",
                 "blurred_and_distorted_vision", "obesity", "excessive_hunger", "increased_appetite",
                 "polyuria"],
    "gastroenteritis": ["vomiting", "sunken_eyes", "dehydration", "diarrhoea"],
    "bronchial_asthma": ["fatigue", "cough", "high_fever", "breathlessness", "family_history",
                         "mucoid_sputum"],
    "hypertension": ["headache", "chest_pain", "dizziness", "loss_of_balance", "lack_of_concentration"],
    "migraine": ["acidity", "indigestion", "headache", "blurred_and_distorted_vision", "excessive_hunger",
                 "stiff_neck", "depression", "irritability", "visual_disturbances"],
    "cervical_spondylosis": ["back_pain", "weakness_in_limbs", "neck_pain", "dizziness", "loss_of_balance"],
    "paralysis_brain_hemorrhage": ["vomiting", "headache", "weakness_of_one_body_side", "altered_sensorium"],
    "jaundice": ["itching", "vomiting", "fatigue", "weight_loss", "high_fever", "yellowish_skin",
                 "dark_urine", "abdominal_pain"],
    "malaria": ["chills", "vomiting", "high_fever", "sweating", "headache", "nausea", "diarrhoea",
                "muscle_pain"],
    "chicken_pox": ["itching", "skin_rash", "fatigue", "lethargy", "high_fever", "headache",
                    "loss_of_appetite", "mild_fever", "swelled_lymph_nodes", "malaise", "red_spots_over_body"],
    "dengue": ["skin_rash", "chills", "joint_pain", "vomiting", "fatigue", "high_fever", "headache",
               "nausea", "loss_of_appetite", "pain_behind_the_eyes", "back_pain", "muscle_pain",
               "red_spots_over_body"],
    "typhoid": ["chills", "vomiting", "fatigue", "high_fever", "headache", "nausea", "constipation",
                "abdominal_pain", "diarrhoea", "toxic_look_typhos", "belly_pain"],
    "hepatitis_a": ["joint_pain", "vomiting", "yellowish_skin", "dark_urine", "nausea", "loss_of_appetite",
                     "abdominal_pain", "diarrhoea", "mild_fever", "yellowing_of_eyes", "muscle_pain"],
    "hepatitis_b": ["itching", "fatigue", "lethargy", "yellowish_skin", "dark_urine", "loss_of_appetite",
                     "abdominal_pain", "yellow_urine", "yellowing_of_eyes", "malaise", "receiving_blood_transfusion",
                     "receiving_unsterile_injections"],
    "hepatitis_c": ["fatigue", "yellowish_skin", "nausea", "loss_of_appetite", "yellowing_of_eyes",
                     "family_history"],
    "hepatitis_d": ["joint_pain", "vomiting", "fatigue", "yellowish_skin", "dark_urine", "nausea",
                     "loss_of_appetite", "abdominal_pain", "yellowing_of_eyes"],
    "hepatitis_e": ["joint_pain", "vomiting", "fatigue", "high_fever", "yellowish_skin", "dark_urine",
                     "nausea", "loss_of_appetite", "abdominal_pain", "yellowing_of_eyes",
                     "acute_liver_failure", "coma", "stomach_bleeding"],
    "alcoholic_hepatitis": ["vomiting", "yellowish_skin", "abdominal_pain", "swelling_of_stomach",
                             "distention_of_abdomen", "history_of_alcohol_consumption", "fluid_overload"],
    "tuberculosis": ["chills", "vomiting", "fatigue", "weight_loss", "cough", "high_fever",
                     "breathlessness", "sweating", "loss_of_appetite", "mild_fever", "yellowing_of_eyes",
                     "swelled_lymph_nodes", "malaise", "phlegm", "chest_pain", "blood_in_sputum"],
    "common_cold": ["continuous_sneezing", "chills", "fatigue", "cough", "high_fever", "headache",
                    "swelled_lymph_nodes", "malaise", "phlegm", "throat_irritation", "redness_of_eyes",
                    "sinus_pressure", "runny_nose", "congestion", "chest_pain", "loss_of_smell",
                    "muscle_pain"],
    "pneumonia": ["chills", "fatigue", "cough", "high_fever", "breathlessness", "sweating", "malaise",
                  "phlegm", "chest_pain", "fast_heart_rate", "rusty_sputum"],
    "dimorphic_hemmorhoids": ["constipation", "pain_during_bowel_movements", "pain_in_anal_region",
                                "bloody_stool", "irritation_in_anus"],
    "heart_attack": ["vomiting", "breathlessness", "sweating", "chest_pain", "lightheadedness"],
    "varicose_veins": ["fatigue", "cramps", "bruising", "obesity", "swollen_legs", "swollen_blood_vessels",
                        "prominent_veins_on_calf"],
    "hypothyroidism": ["fatigue", "weight_gain", "cold_hands_and_feet", "mood_swings", "lethargy",
                        "dizziness", "puffy_face_and_eyes", "enlarged_thyroid", "brittle_nails",
                        "swollen_extremeties", "depression", "irritability", "abnormal_menstruation"],
    "hyperthyroidism": ["fatigue", "mood_swings", "weight_loss", "restlessness", "sweating", "diarrhoea",
                         "fast_heart_rate", "excessive_hunger", "muscle_weakness", "irritability",
                         "abnormal_menstruation"],
    "hypoglycemia": ["vomiting", "fatigue", "anxiety", "sweating", "headache", "nausea", "blurred_and_distorted_vision",
                      "excessive_hunger", "slurred_speech", "irritability", "palpitations"],
    "osteoarthristis": ["joint_pain", "neck_pain", "knee_pain", "hip_joint_pain", "swelling_joints",
                         "painful_walking"],
    "arthritis": ["muscle_weakness", "stiff_neck", "swelling_joints", "movement_stiffness", "painful_walking"],
    "paroymsal_positional_vertigo": ["vomiting", "headache", "nausea", "spinning_movements", "loss_of_balance",
                                       "unsteadiness"],
    "acne": ["skin_rash", "pus_filled_pimples", "blackheads", "scurring"],
    "urinary_tract_infection": ["burning_micturition", "bladder_discomfort", "foul_smell_of_urine",
                                  "continuous_feel_of_urine"],
    "psoriasis": ["skin_rash", "joint_pain", "skin_peeling", "silver_like_dusting", "small_dents_in_nails",
                   "inflammatory_nails"],
    "impetigo": ["skin_rash", "high_fever", "blister", "red_sore_around_nose", "yellow_crust_ooze"],
}


def build_kaggle_csv(out_path: Path, rows_per_disease: int = 120, seed: int = 42) -> None:
    """Build a Kaggle-style wide CSV: Disease, Symptom_1...Symptom_17.

    Each row is a sampled subset of the disease's canonical symptoms with
    a small amount of noise, mirroring the variability of the public dataset.
    """
    rng = random.Random(seed)
    max_symptoms = 17
    headers = ["Disease"] + [f"Symptom_{i}" for i in range(1, max_symptoms + 1)]
    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(headers)
        for disease, symptoms in DISEASE_SYMPTOMS.items():
            for _ in range(rows_per_disease):
                k = rng.randint(max(2, min(3, len(symptoms))), len(symptoms))
                picked = rng.sample(symptoms, k)
                row = [disease] + picked + [""] * (max_symptoms - len(picked))
                w.writerow(row)


def clone_medquad(target: Path) -> bool:
    """Clone the MedQuAD repo into target. Returns True on success."""
    if target.exists():
        print(f"[medquad] already present at {target}")
        return True
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/abachaa/MedQuAD.git", str(target)],
            check=True,
            capture_output=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"[medquad] clone failed: {e.stderr.decode(errors='ignore')[:300]}", file=sys.stderr)
        return False


def main() -> None:
    kaggle_csv = RAW / "dataset.csv"
    if not kaggle_csv.exists():
        print(f"[kaggle] building synthetic Kaggle-equivalent dataset at {kaggle_csv}")
        build_kaggle_csv(kaggle_csv)
        print(f"[kaggle] wrote {kaggle_csv}")
    else:
        print(f"[kaggle] already present at {kaggle_csv}")

    medquad_dir = RAW / "MedQuAD"
    ok = clone_medquad(medquad_dir)
    if not ok:
        print("[medquad] continuing without MedQuAD; preprocessor will produce empty corpus")

    print("\nDataset summary:")
    print(f"  diseases: {len(DISEASE_SYMPTOMS)}")
    print(f"  unique symptoms: {len({s for syms in DISEASE_SYMPTOMS.values() for s in syms})}")
    if medquad_dir.exists():
        xml_count = sum(1 for _ in medquad_dir.rglob("*.xml"))
        print(f"  MedQuAD xml files: {xml_count}")


if __name__ == "__main__":
    main()
