"""
Stage 3: ML Oracle Node
Agent-driven ML model selection with uncertainty-weighted consensus
"""

from typing import Dict, Any, List
import numpy as np
from models.schemas import (
    MaestroState, MLModelPrediction, MLConsensus
)


async def ml_oracle_node(state: MaestroState) -> Dict[str, Any]:
    """
    LangGraph node for Stage 3: ML Oracle.
    
    Four-step protocol:
        S1: Generate data profile (completeness, feature count, temporal data)
        S2: Model eligibility reasoning (exclude models requiring missing data)
        S3: Parallel execution of eligible models
        S4: Uncertainty-weighted consensus (narrower CI = higher weight)
    
    Returns ML consensus prediction + three-way comparison readiness.
    """
    
    if state.current_patient is None or state.agent_reasoning is None:
        return {
            "error_message": "Missing patient data or agent reasoning for ML Oracle",
            "should_terminate": True
        }
    
    # S1: Data Profile Generation
    patient_features = state.current_patient.raw_features
    data_profile = generate_data_profile(patient_features)
    
    # S2: Model Eligibility Reasoning
    # Get ML models from config (passed via thread context)
    # For now, simulate with placeholder models
    all_models = get_ml_models()
    eligible_models = select_eligible_models(all_models, data_profile)
    
    # S3: Candidate Execution (parallel)
    predictions = []
    for model in eligible_models:
        pred = await execute_ml_model(model, patient_features)
        predictions.append(pred)
    
    # S4: Uncertainty-Weighted Consensus
    consensus = compute_consensus(predictions)
    
    return {
        "ml_consensus": consensus
    }


# ============================================================================
# S1: DATA PROFILE GENERATION
# ============================================================================

def generate_data_profile(patient_features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze patient data to determine ML model suitability.
    
    Returns:
        - data_completeness_score: % of features present (0-1)
        - feature_count: Number of non-null features
        - has_temporal_data: Whether longitudinal data exists
        - data_quality_flags: Specific indicators
    """
    # Count non-null features
    total_features = len(patient_features)
    non_null_features = sum(1 for v in patient_features.values() if v not in [None, "", []])
    
    completeness = non_null_features / total_features if total_features > 0 else 0
    
    # Check for temporal data (would require special handling)
    has_temporal = "visit_dates" in patient_features or "medication_timeline" in patient_features
    
    return {
        "data_completeness_score": completeness,
        "feature_count": non_null_features,
        "has_temporal_data": has_temporal,
        "has_fracture_data": any([
            patient_features.get("has_vertebral_fx", False),
            patient_features.get("has_hip_fx", False),
            patient_features.get("has_wrist_fx", False),
        ]),
        "has_medication_data": patient_features.get("bone_medication_count", 0) > 0,
        "comorbidity_count": sum(patient_features.get("comorbidity_flags", {}).values())
    }


# ============================================================================
# S2: MODEL ELIGIBILITY REASONING
# ============================================================================

def get_ml_models() -> List[Dict[str, Any]]:
    """
    Get available ML models from model pool.
    
    In production, this would load from:
        - MLflow model registry
        - Saved model files
        - API endpoints
    
    For now, return placeholder model specs.
    """
    return [
        {
            "name": "XGBoost_Baseline",
            "type": "tree_based",
            "requires_temporal": False,
            "min_features": 10,
            "handles_missing": True,
        },
        {
            "name": "LogisticRegression_Simple",
            "type": "linear",
            "requires_temporal": False,
            "min_features": 5,
            "handles_missing": False,
        },
        {
            "name": "RandomForest_Ensemble",
            "type": "tree_based",
            "requires_temporal": False,
            "min_features": 15,
            "handles_missing": True,
        },
        {
            "name": "LSTM_Temporal",
            "type": "deep_learning",
            "requires_temporal": True,  # Excluded if no temporal data
            "min_features": 20,
            "handles_missing": False,
        },
        {
            "name": "TabNet_Deep",
            "type": "deep_learning",
            "requires_temporal": False,
            "min_features": 12,
            "handles_missing": True,
        }
    ]


def select_eligible_models(
    all_models: List[Dict[str, Any]],
    data_profile: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Agent-driven model selection based on data characteristics.
    
    Exclusion criteria:
        - Model requires temporal data but none available
        - Data completeness too low for model requirements
        - Feature count below model's minimum
    """
    eligible = []
    
    for model in all_models:
        # Check temporal requirement
        if model.get("requires_temporal", False) and not data_profile["has_temporal_data"]:
            continue
        
        # Check feature count
        if data_profile["feature_count"] < model.get("min_features", 0):
            continue
        
        # Check missing data handling
        if not model.get("handles_missing", True) and data_profile["data_completeness_score"] < 0.95:
            continue
        
        eligible.append(model)
    
    return eligible


# ============================================================================
# S3: MODEL EXECUTION
# ============================================================================

async def execute_ml_model(
    model: Dict[str, Any],
    patient_features: Dict[str, Any]
) -> MLModelPrediction:
    """
    Execute a single ML model on patient features.
    
    In production, this would:
        - Load model from disk/registry
        - Preprocess features
        - Run inference
        - Compute confidence intervals (bootstrap, quantile regression, etc.)
    
    For now, simulate with placeholder predictions.
    """
    # Simulate prediction (in production, replace with real model inference)
    np.random.seed(hash(patient_features.get("age", 0)) % 2**32)
    
    # Base probability (simplified from real model)
    base_prob = 0.15  # Base rate from dataset (17.4%)
    
    # Adjust based on risk factors
    if patient_features.get("has_vertebral_fx", False):
        base_prob += 0.25
    if patient_features.get("cci", 0) > 3:
        base_prob -= 0.05  # Competing mortality effect
    if patient_features.get("osteoporosis_diagnosed", False):
        base_prob += 0.10
    
    # Add model-specific noise
    noise = np.random.normal(0, 0.05)
    probability = np.clip(base_prob + noise, 0.01, 0.99)
    
    # Simulate confidence interval
    ci_width = np.random.uniform(0.10, 0.30)
    ci_lower = max(0.0, probability - ci_width / 2)
    ci_upper = min(1.0, probability + ci_width / 2)
    
    return MLModelPrediction(
        model_name=model["name"],
        probability=float(probability),
        confidence_interval_lower=float(ci_lower),
        confidence_interval_upper=float(ci_upper),
        ci_width=float(ci_width),
        weight=1.0 / ci_width  # Narrower CI = higher weight
    )


# ============================================================================
# S4: UNCERTAINTY-WEIGHTED CONSENSUS
# ============================================================================

def compute_consensus(predictions: List[MLModelPrediction]) -> MLConsensus:
    """
    Compute uncertainty-weighted consensus from ML predictions.
    
    Weighting scheme:
        weight_i = 1 / CI_width_i
        Narrower confidence interval → more certain → higher weight
    
    Consensus patterns:
        - High: All models agree (all probs within 0.1 range)
        - Moderate: Majority agree (≥75% within 0.2 range)
        - Low: Near decision boundary (consensus prob 0.4-0.6)
        - Split: Bimodal disagreement
    """
    if not predictions:
        # Fallback if no models eligible
        return MLConsensus(
            eligible_models=[],
            predictions=[],
            consensus_probability=0.5,
            consensus_direction=False,
            pattern="Low"
        )
    
    # Compute weights
    total_weight = sum(p.weight for p in predictions)
    for pred in predictions:
        pred.weight = pred.weight / total_weight  # Normalize
    
    # Weighted average
    consensus_prob = sum(p.probability * p.weight for p in predictions)
    consensus_direction = consensus_prob >= 0.5
    
    # Determine consensus pattern
    probs = [p.probability for p in predictions]
    prob_range = max(probs) - min(probs)
    
    if prob_range < 0.1:
        pattern = "High"  # All agree
    elif prob_range < 0.2:
        pattern = "Moderate"  # Mostly agree
    elif 0.4 <= consensus_prob <= 0.6:
        pattern = "Low"  # Near boundary, uncertain
    else:
        # Check for bimodal (some say <0.3, some say >0.7)
        low_preds = sum(1 for p in probs if p < 0.3)
        high_preds = sum(1 for p in probs if p > 0.7)
        if low_preds > 0 and high_preds > 0:
            pattern = "Split"
        else:
            pattern = "Moderate"
    
    return MLConsensus(
        eligible_models=[p.model_name for p in predictions],
        predictions=predictions,
        consensus_probability=float(consensus_prob),
        consensus_direction=consensus_direction,
        pattern=pattern
    )
