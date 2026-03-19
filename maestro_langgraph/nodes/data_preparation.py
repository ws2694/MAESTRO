"""
Stage 1: Data Preparation Node
Converts raw EHR data into structured clinical narratives with CCI computation
"""

from typing import Dict, Any, List
from models.schemas import PatientNarrative, PatientFeatures, MaestroState


# ============================================================================
# CCI COMPUTATION (Charlson Comorbidity Index)
# ============================================================================

# Charlson et al. (1987, PMID 3558716) weights
CCI_WEIGHTS = {
    "myocardial_infarction": 1,
    "congestive_heart_failure": 1,
    "peripheral_vascular_disease": 1,
    "cerebrovascular_disease": 1,
    "dementia": 1,
    "copd": 1,
    "rheumatic_disease": 1,
    "peptic_ulcer": 1,
    "mild_liver_disease": 1,
    "diabetes_without_complication": 1,
    "hemiplegia_paraplegia": 2,
    "renal_disease": 2,
    "malignancy": 2,
    "diabetes_with_complication": 2,
    "moderate_severe_liver_disease": 3,
    "aids_hiv": 6,
}


def compute_cci(comorbidities: Dict[str, bool], age: float) -> tuple[int, int]:
    """
    Compute Charlson Comorbidity Index with age adjustment.
    
    Mutual Exclusivity Rules:
        - If both mild and moderate/severe liver: score only 3 (not 1+3)
        - If both DM without and DM with complications: score only 2 (not 1+2)
    
    Age adjustment:
        - +1 for age 50-59
        - +2 for age 60-69
        - +3 for age 70-79
        - +4 for age 80+
    
    Returns:
        (base_cci, age_adjusted_cci)
    """
    score = 0
    
    # Apply mutual exclusivity rules
    effective_comorbidities = comorbidities.copy()
    
    if comorbidities.get("moderate_severe_liver_disease", False):
        effective_comorbidities["mild_liver_disease"] = False
    
    if comorbidities.get("diabetes_with_complication", False):
        effective_comorbidities["diabetes_without_complication"] = False
    
    # Sum weights
    for condition, present in effective_comorbidities.items():
        if present and condition in CCI_WEIGHTS:
            score += CCI_WEIGHTS[condition]
    
    # Age adjustment
    age_adjustment = 0
    if age >= 80:
        age_adjustment = 4
    elif age >= 70:
        age_adjustment = 3
    elif age >= 60:
        age_adjustment = 2
    elif age >= 50:
        age_adjustment = 1
    
    return score, score + age_adjustment


# ============================================================================
# NARRATIVE SERIALIZATION
# ============================================================================

def serialize_to_narrative(patient_data: Dict[str, Any]) -> PatientNarrative:
    """
    Convert raw patient data to structured clinical narrative.
    
    Key Design Principle:
        Data absence is EXPLICITLY narrated ('No osteoporosis diagnosis was recorded'),
        not silently omitted. This prevents agent from confusing 'not tested' with 'negative'.
    
    Narrative sequence:
        1. Demographics and tumor location
        2. Osteoporosis status and medications
        3. Comorbidities with CCI calculation
        4. Fracture events with timing
        5. Outcome label (hidden during prediction, revealed for CECF)
    """
    patient_id = patient_data["patient_id"]
    
    # Extract demographics
    gender = patient_data["gender"]
    age = patient_data["age"]
    diagnosis_date = patient_data.get("diagnosis_date", "unknown")
    
    # Tumor location
    tumor_location = patient_data.get("tumor_location", "unspecified")
    
    # Comorbidities
    comorbidities = patient_data.get("comorbidities", {})
    base_cci, age_adjusted_cci = compute_cci(comorbidities, age)
    
    # Build comorbidities text
    present_conditions = [name.replace("_", " ").title() for name, present in comorbidities.items() if present]
    if present_conditions:
        comorbidities_text = ", ".join(present_conditions)
    else:
        comorbidities_text = "No significant comorbidities recorded"
    
    # Osteoporosis
    osteo_diagnosed = patient_data.get("osteoporosis_diagnosed", False)
    osteo_visit_count = patient_data.get("osteoporosis_visit_count", 0)
    osteo_treatment = patient_data.get("osteoporosis_treatment_received", False)
    
    if osteo_diagnosed:
        osteo_text = f"Osteoporosis diagnosed ({osteo_visit_count} related visits). "
        osteo_text += "Treatment received." if osteo_treatment else "No documented treatment."
    else:
        osteo_text = "No osteoporosis diagnosis was recorded."
    
    # Bone medications
    possible_meds = [
        "alendronate", "calcitonin", "denosumab", "etidronate",
        "estrogen", "ibandronic", "pamidronate", "raloxifene",
        "risedronate", "teriparatide", "zoledronic"
    ]
    medications = [med for med in possible_meds if patient_data.get(f"medication_{med}", False)]
    
    if medications:
        meds_text = f"Bone-modifying medications: {', '.join(medications)}."
    else:
        meds_text = "No bone-modifying medications recorded."
    
    # Fracture events
    fracture_types = {
        "vertebral": patient_data.get("fracture_vertebral", False),
        "hip": patient_data.get("fracture_hip", False),
        "wrist": patient_data.get("fracture_wrist", False),
    }
    
    fracture_events = []
    for fx_type, present in fracture_types.items():
        if present:
            timing = patient_data.get(f"fracture_{fx_type}_timing", "unknown timing")
            surgery = patient_data.get(f"fracture_{fx_type}_surgery", False)
            fx_text = f"{fx_type.capitalize()} fracture ({timing})"
            if surgery:
                fx_text += ", surgical intervention"
            fracture_events.append(fx_text)
    
    if fracture_events:
        fracture_text = "; ".join(fracture_events) + "."
    else:
        fracture_text = "No fracture events recorded."
    
    # Construct full narrative
    narrative = f"""
PATIENT DEMOGRAPHICS:
{gender.capitalize()}, {age:.0f} years old. Lung cancer diagnosis date: {diagnosis_date}.
Tumor location: {tumor_location}.

OSTEOPOROSIS AND BONE HEALTH:
{osteo_text}
{meds_text}

COMORBIDITY PROFILE:
{comorbidities_text}
Charlson Comorbidity Index (CCI): {base_cci} (age-adjusted: {age_adjusted_cci})

FRACTURE HISTORY:
{fracture_text}
""".strip()
    
    # Extract structured features for ML
    raw_features = extract_ml_features(patient_data)
    
    return PatientNarrative(
        patient_id=patient_id,
        demographics=f"{gender}, age {age:.0f}",
        tumor_location=tumor_location,
        comorbidities=comorbidities_text,
        cci_score=base_cci,
        cci_age_adjusted=age_adjusted_cci,
        osteoporosis_status=osteo_text,
        medications=medications,
        fracture_events=fracture_text,
        raw_features=raw_features,
        ground_truth=patient_data.get("bone_metastasis_outcome")  # Hidden initially
    )


def extract_ml_features(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract structured feature vector for ML models.
    
    This is separate from narrative generation because ML models need
    numerical/categorical features, not natural language.
    """
    comorbidities = patient_data.get("comorbidities", {})
    base_cci, age_adj_cci = compute_cci(comorbidities, patient_data["age"])
    
    return {
        "gender": patient_data["gender"],
        "age": patient_data["age"],
        "cci": age_adj_cci,
        "tumor_location": patient_data.get("tumor_location", "unspecified"),
        "has_vertebral_fx": patient_data.get("fracture_vertebral", False),
        "has_hip_fx": patient_data.get("fracture_hip", False),
        "has_wrist_fx": patient_data.get("fracture_wrist", False),
        "osteoporosis_diagnosed": patient_data.get("osteoporosis_diagnosed", False),
        "osteo_treatment": patient_data.get("osteoporosis_treatment_received", False),
        "bone_medication_count": sum([
            patient_data.get(f"medication_{med}", False)
            for med in ["alendronate", "calcitonin", "denosumab", "zoledronic"]
        ]),
        "comorbidity_flags": comorbidities,
    }


# ============================================================================
# LANGGRAPH NODE
# ============================================================================

async def data_preparation_node(state: MaestroState) -> Dict[str, Any]:
    """
    LangGraph node for Stage 1: Data Preparation.
    
    Converts current patient's raw data into structured narrative.
    This narrative will be consumed by Stage 2 (Agent Reasoning).
    """
    if state.current_patient is None:
        return {
            "error_message": "No patient data loaded for preparation",
            "should_terminate": True
        }
    
    # Patient data already loaded as PatientNarrative by pipeline
    # This node primarily validates and formats the narrative text
    patient = state.current_patient
    
    # Generate the full narrative text for LLM consumption
    narrative_text = f"""
PATIENT DEMOGRAPHICS:
{patient.demographics}. Tumor location: {patient.tumor_location}.

OSTEOPOROSIS AND BONE HEALTH:
{patient.osteoporosis_status}
Medications: {', '.join(patient.medications) if patient.medications else 'None'}.

COMORBIDITY PROFILE:
{patient.comorbidities}
Charlson Comorbidity Index (CCI): {patient.cci_score} (age-adjusted: {patient.cci_age_adjusted})

FRACTURE HISTORY:
{patient.fracture_events}
""".strip()
    
    return {
        "narrative_text": narrative_text,
        "case_number": state.case_number + 1,
    }
