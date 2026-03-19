"""
MAESTRO V4 Data Models and State Schema
Uses Pydantic for structured outputs and type safety
"""

from typing import Literal, Optional, Dict, List, Any
from pydantic import BaseModel, Field
from enum import Enum


# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class ConfidenceZone(str, Enum):
    """Zone classification for agent predictions"""
    A = "A"  # High confidence
    B = "B"  # Moderate confidence
    C = "C"  # Low confidence


class CECFTier(str, Enum):
    """CECF tier classification"""
    PROVISIONAL = "Provisional"  # < 0.50 or n < 20
    DEVELOPING = "Developing"    # 0.50 - 0.80
    RELIABLE = "Reliable"        # 0.80 - 0.95
    EXPERT = "Expert"            # >= 0.95


class RuleDirection(str, Enum):
    """Rule directional signal"""
    YES = "YES"
    NO = "NO"
    NEUTRAL = "NEUTRAL"


class MilestoneLevel(str, Enum):
    """Milestone achievement levels"""
    M1_INTERN = "Intern"
    M2_RESIDENT = "Resident"
    M3_FELLOW = "Fellow"
    M4_SENIOR_FELLOW = "Senior Fellow"
    M5_ASSOCIATE_PROFESSOR = "Associate Professor"
    M6_PROFESSOR = "Professor"
    M7_KEY_OPINION_LEADER = "Key Opinion Leader"


# ============================================================================
# PATIENT DATA MODELS
# ============================================================================

class PatientNarrative(BaseModel):
    """Structured patient narrative"""
    patient_id: str
    demographics: str
    tumor_location: str
    comorbidities: str
    cci_score: int
    cci_age_adjusted: int
    osteoporosis_status: str
    medications: List[str]
    fracture_events: str
    raw_features: Dict[str, Any]
    ground_truth: Optional[bool] = None  # Hidden during prediction


class PatientFeatures(BaseModel):
    """Raw structured patient features for ML models"""
    gender: str
    age: float
    cci: int
    tumor_location_code: int
    has_vertebral_fx: bool
    has_hip_fx: bool
    has_wrist_fx: bool
    osteoporosis_diagnosed: bool
    bone_medications: List[str]
    comorbidity_flags: Dict[str, bool]


# ============================================================================
# KNOWLEDGE RULE MODELS
# ============================================================================

class KnowledgeRule(BaseModel):
    """Knowledge Rule with CECF tracking"""
    kr_id: str  # e.g., "KR-31"
    type: Literal["established", "empirical_association", "emerging"]
    topic: str  # Thematic group
    content: str  # Rule statement
    clinical_implication: str
    confidence_label: Literal["Established", "Supported", "Emerging"]
    applicability_condition: str  # Python expression or description
    
    # CECF tracking
    n: int = 0  # Total activations
    k: float = 0.0  # Quality-weighted successes
    cecf: float = 0.12  # Starts at Provisional
    tier: CECFTier = CECFTier.PROVISIONAL
    
    # Metadata
    provenance: Optional[str] = None
    created_at: Optional[str] = None
    is_agent_discovered: bool = False


# ============================================================================
# REASONING OUTPUT MODELS
# ============================================================================

class RuleActivation(BaseModel):
    """Record of a rule activation during reasoning"""
    kr_id: str
    direction: RuleDirection
    influence_weight: float = Field(ge=0.0, le=1.0)
    rationale: str


class AgentReasoning(BaseModel):
    """Structured output from JDIP reasoning"""
    patient_id: str
    baseline_summary: str
    activated_rules: List[RuleActivation]
    pr_routes_triggered: List[str]
    specialized_analyses: str
    synthesis: str
    zone: ConfidenceZone
    prediction: bool  # True = YES bone met, False = NO
    confidence_rationale: str


# ============================================================================
# ML ORACLE MODELS
# ============================================================================

class MLModelPrediction(BaseModel):
    """Single ML model prediction"""
    model_name: str
    probability: float = Field(ge=0.0, le=1.0)
    confidence_interval_lower: float = Field(ge=0.0, le=1.0)
    confidence_interval_upper: float = Field(ge=0.0, le=1.0)
    ci_width: float
    weight: float = 0.0  # Computed: 1 / ci_width


class MLConsensus(BaseModel):
    """Consensus from ML Oracle"""
    eligible_models: List[str]
    predictions: List[MLModelPrediction]
    consensus_probability: float
    consensus_direction: bool  # True if >= 0.5
    pattern: Literal["High", "Moderate", "Low", "Split"]


# ============================================================================
# EXPERIENCE MEMORY MODELS
# ============================================================================

class ExperienceNote(BaseModel):
    """Agent-discovered pattern/observation"""
    note_id: str
    patient_id: str
    case_number: int
    trigger: Literal["error", "zone_c_correct", "novel_observation"]
    content: str
    embedding: Optional[List[float]] = None
    
    # CECF tracking (like KRs)
    n: int = 1
    k: float = 0.0
    cecf: float = 0.12
    status: Literal["observation", "candidate_kr", "promoted_kr", "deprecated"] = "observation"
    
    created_at: str
    related_rules: List[str] = []


# ============================================================================
# LANGGRAPH STATE SCHEMA
# ============================================================================

class MaestroState(BaseModel):
    """Complete state for LangGraph workflow"""
    
    # Current patient
    current_patient: Optional[PatientNarrative] = None
    case_number: int = 0
    
    # Stage 1: Data Preparation
    narrative_text: str = ""
    
    # Stage 2: Agent Reasoning
    agent_reasoning: Optional[AgentReasoning] = None
    
    # Stage 3: ML Oracle
    ml_consensus: Optional[MLConsensus] = None
    
    # Stage 4: Experience Update
    three_way_comparison: Optional[Dict[str, Any]] = None
    updated_rules: List[str] = []
    
    # Stage 5: Milestone Gate
    current_milestone: Optional[MilestoneLevel] = None
    milestone_passed: bool = True
    validation_auc: float = 0.0
    
    # Stage 6: CECE Memory
    experience_notes: List[ExperienceNote] = []
    memory_consolidation_triggered: bool = False
    
    # Persistent knowledge state (shared across all patients)
    knowledge_rules: Dict[str, KnowledgeRule] = {}
    memory_store: List[ExperienceNote] = []
    
    # Workflow control
    should_terminate: bool = False
    error_message: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True


# ============================================================================
# CECF COMPUTATION HELPERS
# ============================================================================

class CECFUpdate(BaseModel):
    """Record of CECF update for audit trail"""
    kr_id: str
    case_number: int
    direction: RuleDirection
    influence_weight: float
    ground_truth: bool
    k_increment: float
    n_increment: int = 1
    new_k: float
    new_n: int
    new_cecf: float
    new_tier: CECFTier
    layer3_counterfactual: bool = False
    timestamp: str
