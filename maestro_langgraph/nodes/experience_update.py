"""
Stage 4: Experience Update Node (CECF)
Three-layer Bayesian credit assignment system
"""

from typing import Dict, Any
from datetime import datetime
from models.schemas import (
    MaestroState, RuleDirection, CECFUpdate
)
from utils.cecf import (
    update_rule_cecf, apply_layer3_penalty, get_cecf_tier
)


async def experience_update_node(state: MaestroState) -> Dict[str, Any]:
    """
    LangGraph node for Stage 4: Experience Update (CECF).
    
    Three-layer credit assignment:
        Layer 1: Direction-aware attribution (correct/neutral/wrong)
        Layer 2: Influence-weighted attribution (blame proportional to reliance)
        Layer 3: Counterfactual probing (Zone A errors only)
    
    Updates k, n, CECF for each activated rule.
    """
    
    if state.agent_reasoning is None or state.ml_consensus is None:
        return {
            "error_message": "Missing reasoning or ML consensus for experience update",
            "should_terminate": True
        }
    
    # Get ground truth (now revealed)
    ground_truth = state.current_patient.ground_truth
    if ground_truth is None:
        return {
            "error_message": "Ground truth not available for CECF update",
            "should_terminate": True
        }
    
    # Three-way comparison
    agent_pred = state.agent_reasoning.prediction
    ml_pred = state.ml_consensus.consensus_direction
    
    three_way = {
        "agent_prediction": agent_pred,
        "ml_consensus": ml_pred,
        "ground_truth": ground_truth,
        "agent_correct": agent_pred == ground_truth,
        "ml_correct": ml_pred == ground_truth,
        "pattern": classify_three_way_pattern(agent_pred, ml_pred, ground_truth)
    }
    
    # Layer 1 + Layer 2: Update all activated rules
    updated_rules = []
    cecf_updates = []
    
    knowledge_rules = state.knowledge_rules.copy()
    
    for rule_activation in state.agent_reasoning.activated_rules:
        kr_id = rule_activation.kr_id
        
        if kr_id not in knowledge_rules:
            continue  # Skip if rule not found
        
        rule = knowledge_rules[kr_id]
        
        # Update CECF
        new_k, new_n, k_increment, new_cecf, new_tier = update_rule_cecf(
            current_k=rule.k,
            current_n=rule.n,
            rule_direction=rule_activation.direction,
            ground_truth=ground_truth,
            influence_weight=rule_activation.influence_weight
        )
        
        # Record update for audit trail
        cecf_update = CECFUpdate(
            kr_id=kr_id,
            case_number=state.case_number,
            direction=rule_activation.direction,
            influence_weight=rule_activation.influence_weight,
            ground_truth=ground_truth,
            k_increment=k_increment,
            n_increment=1,
            new_k=new_k,
            new_n=new_n,
            new_cecf=new_cecf,
            new_tier=new_tier,
            layer3_counterfactual=False,
            timestamp=datetime.utcnow().isoformat()
        )
        cecf_updates.append(cecf_update)
        
        # Apply update
        rule.k = new_k
        rule.n = new_n
        rule.cecf = new_cecf
        rule.tier = new_tier
        
        knowledge_rules[kr_id] = rule
        updated_rules.append(kr_id)
    
    # Layer 3: Counterfactual probing (Zone A errors only)
    layer3_triggered = False
    
    if state.agent_reasoning.zone == "A" and not three_way["agent_correct"]:
        layer3_triggered = True
        
        # Run counterfactual for each activated rule
        for rule_activation in state.agent_reasoning.activated_rules:
            kr_id = rule_activation.kr_id
            
            # Simulate counterfactual (in production, re-run LLM)
            would_fix_error = await simulate_counterfactual(
                state, kr_id, ground_truth
            )
            
            if would_fix_error:
                # This rule was the PRIMARY CAUSAL DRIVER of error
                rule = knowledge_rules[kr_id]
                
                new_k, new_cecf, new_tier = apply_layer3_penalty(
                    current_k=rule.k,
                    current_n=rule.n
                )
                
                # Record Layer 3 penalty
                cecf_update = CECFUpdate(
                    kr_id=kr_id,
                    case_number=state.case_number,
                    direction=rule_activation.direction,
                    influence_weight=rule_activation.influence_weight,
                    ground_truth=ground_truth,
                    k_increment=-0.5,  # Penalty
                    n_increment=0,
                    new_k=new_k,
                    new_n=rule.n,
                    new_cecf=new_cecf,
                    new_tier=new_tier,
                    layer3_counterfactual=True,
                    timestamp=datetime.utcnow().isoformat()
                )
                cecf_updates.append(cecf_update)
                
                rule.k = new_k
                rule.cecf = new_cecf
                rule.tier = new_tier
                knowledge_rules[kr_id] = rule
    
    return {
        "knowledge_rules": knowledge_rules,
        "updated_rules": updated_rules,
        "three_way_comparison": three_way,
        # Store CECF updates for audit/logging (not in state schema, handled by DB)
    }


# ============================================================================
# THREE-WAY COMPARISON HELPERS
# ============================================================================

def classify_three_way_pattern(
    agent_pred: bool,
    ml_pred: bool,
    ground_truth: bool
) -> str:
    """
    Classify the three-way comparison pattern.
    
    Returns diagnostic meaning (matches Table in Section 3.2 of paper).
    """
    if agent_pred and ml_pred and ground_truth:
        return "Both correct, concordant"
    elif agent_pred and not ml_pred and ground_truth:
        return "Agent outperformed ML (clinical reasoning added value)"
    elif not agent_pred and ml_pred and ground_truth:
        return "ML outperformed agent (agent missed statistical signal)"
    elif agent_pred and not ml_pred and not ground_truth:
        return "Agent overrode ML incorrectly (overconfidence)"
    elif agent_pred and ml_pred and not ground_truth:
        return "Both wrong (systematic difficulty)"
    elif not agent_pred and not ml_pred and ground_truth:
        return "Both missed (hard case, likely Zone C)"
    elif not agent_pred and not ml_pred and not ground_truth:
        return "Both correct, concordant (negative)"
    else:  # agent disagrees with ML, both disagree with GT in opposite ways
        return "Complex disagreement pattern"


# ============================================================================
# LAYER 3: COUNTERFACTUAL ANALYSIS
# ============================================================================

async def simulate_counterfactual(
    state: MaestroState,
    rule_to_remove: str,
    ground_truth: bool
) -> bool:
    """
    Simulate counterfactual: Would removing this rule fix the prediction error?
    
    In production, this would:
        1. Re-run LLM with rule_to_remove excluded from knowledge base
        2. Check if new prediction matches ground truth
    
    For efficiency, we use heuristics here:
        - If rule had highest influence weight AND pointed wrong direction, likely causal
        - Otherwise, less likely to be primary driver
    
    Cost consideration:
        Each counterfactual = 1 LLM call
        If 5 rules activated, that's 5 extra calls per Zone A error
        Estimated: 50-100 Zone A errors in 5000 cases = 250-500 extra calls total
    
    Args:
        state: Current MaestroState
        rule_to_remove: KR ID to test
        ground_truth: Actual outcome
    
    Returns:
        True if removing rule would fix prediction, False otherwise
    """
    
    # Find the rule's activation
    rule_activation = None
    for ra in state.agent_reasoning.activated_rules:
        if ra.kr_id == rule_to_remove:
            rule_activation = ra
            break
    
    if rule_activation is None:
        return False
    
    # Heuristic: Rule is likely causal if:
    # 1. It had high influence weight (>0.25)
    # 2. It pointed in the WRONG direction
    
    agent_pred = state.agent_reasoning.prediction
    rule_direction = rule_activation.direction
    
    # Convert rule direction to boolean
    rule_predicts_positive = (rule_direction == RuleDirection.YES)
    
    # If rule pointed WRONG direction with high influence, removing it might fix
    if rule_predicts_positive != ground_truth and rule_activation.influence_weight > 0.25:
        return True
    
    return False


# ============================================================================
# CECF STATISTICS TRACKING
# ============================================================================

def compute_cecf_statistics(knowledge_rules: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute aggregate statistics for monitoring learning progress.
    
    Useful for:
        - Tracking how many rules reach Expert tier
        - Monitoring rule divergence over training
        - Identifying deprecated rules
    """
    rules = list(knowledge_rules.values())
    
    tier_counts = {
        "Provisional": sum(1 for r in rules if r.tier.value == "Provisional"),
        "Developing": sum(1 for r in rules if r.tier.value == "Developing"),
        "Reliable": sum(1 for r in rules if r.tier.value == "Reliable"),
        "Expert": sum(1 for r in rules if r.tier.value == "Expert"),
    }
    
    # Top 10 most reliable rules
    sorted_rules = sorted(rules, key=lambda r: r.cecf, reverse=True)
    top_rules = [
        {"kr_id": r.kr_id, "cecf": r.cecf, "n": r.n}
        for r in sorted_rules[:10]
    ]
    
    # Bottom 10 (candidates for deprecation)
    bottom_rules = [
        {"kr_id": r.kr_id, "cecf": r.cecf, "n": r.n}
        for r in sorted_rules[-10:]
        if r.n >= 30  # Only consider rules with sufficient evidence
    ]
    
    return {
        "total_rules": len(rules),
        "tier_distribution": tier_counts,
        "avg_cecf": sum(r.cecf for r in rules) / len(rules) if rules else 0,
        "top_10_rules": top_rules,
        "bottom_10_rules": bottom_rules,
    }
