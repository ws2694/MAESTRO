"""
Stage 2: Agent Reasoning Node (JDIP)
Implements three-layer cognitive structure with structured outputs
"""

import json
from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser

from models.schemas import (
    MaestroState, AgentReasoning, RuleActivation,
    RuleDirection, ConfidenceZone
)
from prompts.jdip_prompts import build_jdip_system_prompt


async def agent_reasoning_node(state: MaestroState, llm) -> Dict[str, Any]:
    """
    LangGraph node for Stage 2: Agent Reasoning (JDIP).
    
    Executes three-layer cognitive protocol:
        Layer 1: Activate relevant Knowledge Rules based on applicability
        Layer 2: Execute PR-routed specialized analyses
        Layer 3: Classify confidence zone
    
    Returns structured JSON with rule activations and influence weights.
    """
    
    if not state.narrative_text:
        return {
            "error_message": "No patient narrative available for reasoning",
            "should_terminate": True
        }
    
    # Build JDIP system prompt with current CECF-weighted knowledge
    knowledge_rules = list(state.knowledge_rules.values())
    system_prompt = build_jdip_system_prompt(knowledge_rules)
    
    # Build user message with patient narrative
    user_message = f"""Analyze this patient and predict bone metastasis risk:

{state.narrative_text}

Follow the 6-step reasoning template. Return structured JSON as specified.
"""
    
    # Call LLM with structured output
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ]
    
    try:
        # Get structured output (Pydantic-enforced)
        response = await llm.ainvoke(messages)
        
        # Parse JSON response
        reasoning_data = json.loads(response.content)
        
        # Validate and construct AgentReasoning
        activated_rules = [
            RuleActivation(**rule_data)
            for rule_data in reasoning_data["activated_rules"]
        ]
        
        # Validate influence weights sum to 1.0
        total_weight = sum(r.influence_weight for r in activated_rules)
        if abs(total_weight - 1.0) > 0.01:
            # Normalize if close
            for rule in activated_rules:
                rule.influence_weight /= total_weight
        
        agent_reasoning = AgentReasoning(
            patient_id=state.current_patient.patient_id,
            baseline_summary=reasoning_data["baseline_summary"],
            activated_rules=activated_rules,
            pr_routes_triggered=reasoning_data.get("pr_routes_triggered", []),
            specialized_analyses=reasoning_data.get("specialized_analyses", ""),
            synthesis=reasoning_data["synthesis"],
            zone=ConfidenceZone(reasoning_data["zone"]),
            prediction=reasoning_data["prediction"],
            confidence_rationale=reasoning_data["confidence_rationale"]
        )
        
        return {
            "agent_reasoning": agent_reasoning
        }
        
    except Exception as e:
        return {
            "error_message": f"Agent reasoning failed: {str(e)}",
            "should_terminate": True
        }


# ============================================================================
# PR ROUTER LOGIC (Layer 2 Helper)
# ============================================================================

def activate_pr_routes(narrative_text: str, patient_data: Dict[str, Any]) -> list[str]:
    """
    Determine which Procedural Routes should be activated.
    
    This is a deterministic pre-filter based on data availability.
    The LLM still performs the actual specialized analysis.
    """
    routes = []
    
    # Extract flags from patient data
    cci = patient_data.get("cci_age_adjusted", 0)
    has_bone_meds = len(patient_data.get("medications", [])) > 0
    has_fracture = any([
        patient_data.get("fracture_vertebral", False),
        patient_data.get("fracture_hip", False),
        patient_data.get("fracture_wrist", False),
    ])
    has_osteo = patient_data.get("osteoporosis_diagnosed", False)
    
    # PR-ROUTE-1: Multi-comorbidity interaction
    if cci >= 3:
        routes.append("PR-ROUTE-1")
    
    # PR-ROUTE-2: Bisphosphonate paradox
    if has_bone_meds:
        routes.append("PR-ROUTE-2")
    
    # PR-ROUTE-3: Sentinel fracture signal
    if has_fracture:
        routes.append("PR-ROUTE-3")
    
    # PR-ROUTE-4: Bone health baseline + detection bias
    if has_osteo:
        routes.append("PR-ROUTE-4")
    
    # PR-ROUTE-5: Joint bone health assessment
    if "PR-ROUTE-2" in routes and "PR-ROUTE-3" in routes:
        routes.append("PR-ROUTE-5")
    
    # PR-ROUTE-6: Conservative mode (default if nothing triggered)
    if not routes:
        routes.append("PR-ROUTE-6")
    
    return routes


# ============================================================================
# RULE APPLICABILITY CHECKING (Layer 1 Helper)
# ============================================================================

def check_rule_applicability(rule, patient_data: Dict[str, Any]) -> bool:
    """
    Evaluate if a Knowledge Rule's applicability condition is met.
    
    This is a simple heuristic filter. The LLM makes the final decision
    on whether to activate the rule during reasoning.
    
    In production, this could be:
        - Python eval() with sandboxing
        - Rule engine (Drools, PyRules)
        - LLM-based applicability check
    
    For now, we use keyword matching as a lightweight filter.
    """
    condition = rule.applicability_condition.lower()
    
    # Example conditions from the paper:
    # "patient.fracture_events.vertebral == True"
    # "CCI >= 3"
    # "osteoporosis_diagnosed AND bone_medication"
    
    if "vertebral" in condition and "fracture" in condition:
        return patient_data.get("fracture_vertebral", False)
    
    if "hip" in condition and "fracture" in condition:
        return patient_data.get("fracture_hip", False)
    
    if "wrist" in condition and "fracture" in condition:
        return patient_data.get("fracture_wrist", False)
    
    if "osteoporosis" in condition or "osteo" in condition:
        return patient_data.get("osteoporosis_diagnosed", False)
    
    if "cci" in condition:
        cci = patient_data.get("cci_age_adjusted", 0)
        if ">=" in condition:
            threshold = int([s for s in condition.split() if s.isdigit()][0])
            return cci >= threshold
        elif "<=" in condition:
            threshold = int([s for s in condition.split() if s.isdigit()][0])
            return cci <= threshold
    
    if "medication" in condition or "bisphosphonate" in condition:
        return len(patient_data.get("medications", [])) > 0
    
    # Default: assume applicable (let LLM decide)
    return True
