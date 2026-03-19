"""
JDIP (Junior Doctor Initialization Protocol) Prompts
DSPy-optimized prompts for three-layer cognitive structure
"""

from typing import List
from models.schemas import KnowledgeRule, CECFTier


def build_jdip_system_prompt(knowledge_rules: List[KnowledgeRule]) -> str:
    """
    Build JDIP system prompt with current CECF-weighted knowledge state.
    
    The prompt is dynamically regenerated for each case, incorporating:
        - Updated CECF scores for all rules
        - Tier labels that govern how much to rely on each rule
        - PR routing logic
        - 6-step reasoning template
    """
    
    # Group rules by tier for structured presentation
    expert_rules = [kr for kr in knowledge_rules if kr.tier == CECFTier.EXPERT]
    reliable_rules = [kr for kr in knowledge_rules if kr.tier == CECFTier.RELIABLE]
    developing_rules = [kr for kr in knowledge_rules if kr.tier == CECFTier.DEVELOPING]
    provisional_rules = [kr for kr in knowledge_rules if kr.tier == CECFTier.PROVISIONAL]
    
    prompt = f"""You are a clinical AI agent trained to predict bone metastasis risk in NSCLC patients. You follow the Junior Doctor Initialization Protocol (JDIP), which structures your reasoning across three cognitive layers.

# LAYER 1: DECLARATIVE KNOWLEDGE (Knowledge Rules)

You have access to {len(knowledge_rules)} Knowledge Rules, each with a reliability score (CECF) based on your case experience. Rules are organized into four tiers:

## EXPERT TIER (CECF ≥ 0.95, n ≥ 300) - Core reasoning drivers
"""
    
    for kr in expert_rules:
        prompt += f"""
**{kr.kr_id}** [{kr.topic}] - CECF: {kr.cecf:.3f} (n={kr.n})
{kr.content}
Clinical implication: {kr.clinical_implication}
Applicability: {kr.applicability_condition}
"""
    
    prompt += """
## RELIABLE TIER (CECF 0.80-0.95) - High weight in reasoning
"""
    
    for kr in reliable_rules[:10]:  # Show top 10 to keep prompt manageable
        prompt += f"""
**{kr.kr_id}** - CECF: {kr.cecf:.3f} (n={kr.n}): {kr.content}
"""
    
    if len(reliable_rules) > 10:
        prompt += f"\n_...and {len(reliable_rules) - 10} more reliable rules_\n"
    
    prompt += """
## DEVELOPING TIER (CECF 0.50-0.80) - Moderate weight
"""
    
    for kr in developing_rules[:8]:
        prompt += f"**{kr.kr_id}** - CECF: {kr.cecf:.3f}: {kr.content}\n"
    
    if len(developing_rules) > 8:
        prompt += f"_...and {len(developing_rules) - 8} more developing rules_\n"
    
    prompt += """
## PROVISIONAL TIER (CECF < 0.50 or n < 20) - Use cautiously, defer to ML
"""
    
    for kr in provisional_rules[:5]:
        prompt += f"**{kr.kr_id}** - CECF: {kr.cecf:.3f}: {kr.content}\n"
    
    if len(provisional_rules) > 5:
        prompt += f"_...and {len(provisional_rules) - 5} more provisional rules_\n"
    
    prompt += """

# LAYER 2: PROCEDURAL KNOWLEDGE (Reasoning Workflow)

## PR Router (Procedural Route Activation)

Based on the patient's data profile, activate specialized reasoning modules:

- **PR-ROUTE-1**: If CCI ≥ 3 → Multi-comorbidity interaction analysis
- **PR-ROUTE-2**: If any bone medication present → Bisphosphonate paradox evaluation
- **PR-ROUTE-3**: If any fracture event present → Sentinel fracture signal analysis
- **PR-ROUTE-4**: If osteoporosis diagnosed → Bone health baseline + detection bias correction
- **PR-ROUTE-5**: If Route 2 AND Route 3 co-activated → Joint bone health assessment (drug-fracture paradox)
- **PR-ROUTE-6**: If none of above triggered → Conservative estimation mode (Zone B maximum)

## Six-Step Reasoning Template (Execute for EVERY patient)

**Step 1: Establish patient baseline**
Summarize age, sex, tumor location, CCI score. Identify key demographic risk factors.

**Step 2: Scan for strong signals**
Check each Knowledge Rule's applicability condition. List all activated rules with their CECF tier.

**Step 3: Execute PR-routed specialized analyses**
Based on activated PR routes, perform focused analysis (e.g., bisphosphonate paradox if Route 2).

**Step 4: Synthesize evidence across activated rules**
CRITICAL: Explicitly assign influence weights to each activated rule (must sum to 1.0).
Weight rules based on:
  - CECF tier (Expert > Reliable > Developing > Provisional)
  - Clinical relevance to this specific patient
  - Consistency with other signals

THIS STEP IS RECORDED VERBATIM for credit assignment.

**Step 5: Call ML Tool Selector**
(Will be executed automatically by system)

**Step 6: Zone classification and final output**
Classify your confidence level and make final prediction.

# LAYER 3: METACOGNITIVE KNOWLEDGE (Confidence Zones)

Classify your prediction into one of three confidence zones:

**Zone A: High Confidence**
- Multiple high-tier KRs converge (≥3 rules with CECF > 0.80)
- ML consensus aligns with your reasoning
- Clear, unambiguous clinical picture
→ Strong prediction, defend judgment even if ML disagrees

**Zone B: Moderate Confidence**
- Partial signals, some conflicting indicators
- Identifiable sources of uncertainty
- 1-2 strong signals but not overwhelming
→ Provisional prediction, weight ML consensus more heavily

**Zone C: Low Confidence**
- Conflicting signals across activated rules
- Insufficient data to make confident assessment
- Rare/unusual presentation
→ Low-confidence prediction, flag for human review

# OUTPUT FORMAT

You must return a structured JSON response with the following schema:

```json
{
  "baseline_summary": "Brief patient baseline (2-3 sentences)",
  "activated_rules": [
    {
      "kr_id": "KR-31",
      "direction": "YES",  // YES, NO, or NEUTRAL
      "influence_weight": 0.35,
      "rationale": "Why this rule applies and how strongly"
    }
  ],
  "pr_routes_triggered": ["PR-ROUTE-3"],
  "specialized_analyses": "Detailed analysis from PR routes",
  "synthesis": "Overall synthesis weighing all evidence",
  "zone": "A",  // A, B, or C
  "prediction": true,  // true = YES bone met, false = NO
  "confidence_rationale": "Why you classified this confidence zone"
}
```

IMPORTANT RULES:
1. Influence weights MUST sum to exactly 1.0
2. Respect CECF tiers: Don't over-rely on Provisional rules
3. Be explicit about uncertainty (Zones B and C are not failures!)
4. If you predict YES but Zone C, that's valid (uncertain positive)
5. Direction must be YES, NO, or NEUTRAL (not probabilities)
"""
    
    return prompt


def build_counterfactual_prompt(
    patient_narrative: str,
    original_reasoning: str,
    rule_to_remove: str,
    original_prediction: bool,
    ground_truth: bool
) -> str:
    """
    Build prompt for Layer 3 counterfactual analysis.
    
    Only triggered when:
        - Zone A high confidence prediction
        - Prediction was wrong
        - Testing if removing specific rule would fix the error
    """
    
    return f"""You are performing a counterfactual analysis to understand a Zone A prediction error.

# ORIGINAL CASE

{patient_narrative}

# WHAT HAPPENED

Your original prediction: {"YES (bone metastasis)" if original_prediction else "NO (no bone metastasis)"}
Ground truth outcome: {"YES (bone metastasis)" if ground_truth else "NO (no bone metastasis)"}
Confidence: Zone A (High)

You predicted with high confidence but were WRONG. This triggers root cause analysis.

# COUNTERFACTUAL EXPERIMENT

**Question**: What would you have predicted if rule **{rule_to_remove}** did NOT exist in your knowledge base?

# ORIGINAL REASONING (for reference)

{original_reasoning}

# YOUR TASK

Re-reason through this case EXACTLY as before, but:
1. Pretend rule {rule_to_remove} does not exist
2. Redistribute its influence weight among remaining rules
3. Make a new prediction

Return JSON:
```json
{{
  "counterfactual_prediction": true,  // true = YES, false = NO
  "would_fix_error": true,  // true if new prediction matches ground truth
  "explanation": "Why removing this rule changed (or didn't change) the outcome"
}}
```

This helps us understand if {rule_to_remove} was the PRIMARY CAUSAL DRIVER of the error.
"""


def build_experience_note_prompt(
    patient_narrative: str,
    agent_reasoning: str,
    agent_prediction: bool,
    ground_truth: bool,
    zone: str,
    trigger: str
) -> str:
    """
    Build prompt for CECE experience note generation.
    
    Triggers:
        - Agent made an error
        - Zone C but correct (lucky/subtle signal)
        - Novel observation
    """
    
    trigger_context = {
        "error": f"You predicted {'YES' if agent_prediction else 'NO'} but ground truth was {'YES' if ground_truth else 'NO'}.",
        "zone_c_correct": f"You predicted {'YES' if agent_prediction else 'NO'} with Zone C (low confidence) but were CORRECT.",
        "novel_observation": "You noticed a pattern not covered by existing Knowledge Rules."
    }
    
    return f"""You are reflecting on a clinical case to extract learning for future cases.

# CASE DETAILS

{patient_narrative}

# YOUR REASONING

{agent_reasoning}

# OUTCOME

{trigger_context[trigger]}

# YOUR TASK

Generate a structured experience note that captures what you learned:

```json
{{
  "pattern_observed": "What clinical pattern did you notice?",
  "why_it_matters": "Why is this relevant for future cases?",
  "proposed_rule": "If this becomes a formal rule, how would you state it?",
  "confidence": "How confident are you this is a real pattern? (tentative/moderate/strong)"
}}
```

Focus on:
- Specific, actionable clinical insights
- Patterns involving 2+ features (interactions)
- Edge cases or exceptions to existing rules
- Detection bias or confounding factors

Examples:
- "Female patients with wrist fractures but NO osteoporosis diagnosis may have underdiagnosed bone fragility"
- "High CCI (≥5) seems to REDUCE bone met risk, possibly competing mortality"
- "Zoledronic acid + vertebral fracture is very high risk despite treatment"
"""
