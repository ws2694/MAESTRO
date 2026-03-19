"""
CECF (Clinically-weighted Evidence Confidence Function) Implementation
Bayesian Beta-Binomial credit assignment for Knowledge Rules
"""

from scipy.stats import beta as beta_dist
from models.schemas import CECFTier, RuleDirection
from typing import Tuple


def compute_cecf(k: float, n: int, tau: float = 0.65) -> float:
    """
    Compute CECF score using Bayesian Beta distribution.
    
    CECF = P(θ > τ | k, n) where θ is the true reliability of the rule.
    
    Args:
        k: Quality-weighted successes (can be non-integer)
        n: Total activations
        tau: Clinical validity threshold (default 0.65)
    
    Returns:
        CECF score between 0 and 1
        
    Mathematical Foundation:
        Prior: θ ~ Beta(1, 1) [uniform, non-informative]
        Posterior: θ | k,n ~ Beta(α=k+1, β=n-k+1)
        CECF = 1 - F_Beta(τ; α, β)
        
    Hard Cap Rule:
        When n < 20, CECF is capped to prevent novice overconfidence.
        Even 100% success on 15 cases cannot exceed ~0.12 (Provisional tier).
    """
    if n == 0:
        return 0.12  # Provisional default
    
    # Beta distribution parameters
    alpha = k + 1
    beta_param = n - k + 1
    
    # P(theta > tau) = 1 - CDF(tau)
    cecf = float(1 - beta_dist.cdf(tau, alpha, beta_param))
    
    # Hard cap for small n (apprenticeship period)
    if n < 20:
        # Linear ramp from 0.12 at n=1 to uncapped at n=20
        max_cecf = 0.12 + (cecf - 0.12) * ((n - 1) / 19)
        cecf = min(cecf, max_cecf)
    
    return cecf


def get_cecf_tier(cecf: float, n: int) -> CECFTier:
    """
    Classify CECF into tier based on score and sample size.
    
    Tier system ensures both high success rate AND sufficient evidence.
    """
    if cecf < 0.50 or n < 20:
        return CECFTier.PROVISIONAL
    elif cecf < 0.80:
        return CECFTier.DEVELOPING
    elif cecf < 0.95:
        return CECFTier.RELIABLE
    else:  # cecf >= 0.95
        return CECFTier.EXPERT


def update_rule_cecf(
    current_k: float,
    current_n: int,
    rule_direction: RuleDirection,
    ground_truth: bool,
    influence_weight: float,
    tau: float = 0.65
) -> Tuple[float, int, float, float, CECFTier]:
    """
    Update rule's k, n, CECF after a single case (Layer 1 + Layer 2).
    
    Layer 1: Direction-Aware Attribution
        - Correct direction: k += 1.0
        - Neutral direction: k += 0.5
        - Wrong direction: k += (1 - influence_weight) [Layer 2]
    
    Layer 2: Influence-Weighted Attribution
        - Rules with high influence_weight get heavier penalty
        - Rules barely considered get minimal penalty
    
    Args:
        current_k: Current quality-weighted successes
        current_n: Current total activations
        rule_direction: Rule's directional signal (YES/NO/NEUTRAL)
        ground_truth: Actual outcome (True = bone met, False = no bone met)
        influence_weight: How much agent relied on this rule (0-1)
        tau: Clinical validity threshold
    
    Returns:
        (new_k, new_n, k_increment, new_cecf, new_tier)
    """
    new_n = current_n + 1
    
    # Determine rule's correctness
    # Rule direction YES/NO maps to True/False for comparison
    rule_predicts_positive = (rule_direction == RuleDirection.YES)
    
    if rule_direction == RuleDirection.NEUTRAL:
        # Layer 1: Neutral direction gets half credit
        k_increment = 0.5
    elif rule_predicts_positive == ground_truth:
        # Layer 1: Correct direction gets full credit
        k_increment = 1.0
    else:
        # Layer 1 + Layer 2: Wrong direction, influence-weighted penalty
        k_increment = 1 - influence_weight
    
    new_k = current_k + k_increment
    new_cecf = compute_cecf(new_k, new_n, tau)
    new_tier = get_cecf_tier(new_cecf, new_n)
    
    return new_k, new_n, k_increment, new_cecf, new_tier


def apply_layer3_penalty(current_k: float, current_n: int, tau: float = 0.65) -> Tuple[float, float, CECFTier]:
    """
    Apply Layer 3 counterfactual penalty.
    
    Triggered only when:
        1. Agent predicted with Zone A high confidence
        2. Prediction was wrong
        3. Counterfactual analysis shows removing this rule fixes prediction
    
    Penalty: k -= 0.5 (causal driver penalty)
    
    Returns:
        (new_k, new_cecf, new_tier)
    """
    new_k = max(0.0, current_k - 0.5)  # Can't go below 0
    new_cecf = compute_cecf(new_k, current_n, tau)
    new_tier = get_cecf_tier(new_cecf, current_n)
    
    return new_k, new_cecf, new_tier


def is_ckip_eligible(cecf: float, n: int, is_agent_discovered: bool) -> bool:
    """
    Check if rule is eligible for Cross-Run Knowledge Inheritance.
    
    CKIP Eligibility (all three required):
        1. CECF > 0.97 (Expert tier, near certainty)
        2. n >= 300 (extensive validation)
        3. Clinical domain overlap (handled externally)
    
    Note: Inherited rules have CECF RESET to 0.60 in new run to prevent overfitting.
    """
    return cecf > 0.97 and n >= 300


# ============================================================================
# CECF NUMERICAL BEHAVIOR EXAMPLES (for validation/testing)
# ============================================================================

def print_cecf_table():
    """
    Print CECF behavior table matching Section 4.4 of the paper.
    Useful for validation and debugging.
    """
    scenarios = [
        ("First case, correct", 1, 1.0),
        ("5 cases, 4 correct", 5, 4.0),
        ("20 cases, 16 correct", 20, 16.0),
        ("50 cases, 41 correct", 50, 41.0),
        ("100 cases, 82 correct", 100, 82.0),
        ("300 cases, 246 correct", 300, 246.0),
        ("100 cases, 70 correct", 100, 70.0),
        ("500 cases, 260 correct", 500, 260.0),
    ]
    
    print(f"{'Scenario':<30} {'n':<5} {'k':<8} {'k/n':<8} {'CECF':<8} {'Tier':<15}")
    print("-" * 85)
    
    for scenario, n, k in scenarios:
        cecf = compute_cecf(k, n)
        tier = get_cecf_tier(cecf, n)
        success_rate = k / n if n > 0 else 0
        print(f"{scenario:<30} {n:<5} {k:<8.1f} {success_rate:<8.1%} {cecf:<8.3f} {tier.value:<15}")


if __name__ == "__main__":
    print("CECF Bayesian Credit Assignment - Validation Table")
    print("=" * 85)
    print_cecf_table()
    print("\nHard cap demonstration (n < 20):")
    for n in [1, 5, 10, 15, 19, 20]:
        k = n  # 100% success rate
        cecf = compute_cecf(k, n)
        tier = get_cecf_tier(cecf, n)
        print(f"  n={n:2d}, k={k:2d} (100%) → CECF={cecf:.3f} ({tier.value})")
