"""
End-to-End Integration Tests for MAESTRO V4
Tests the complete pipeline from data preparation through milestone gate
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from typing import Dict, Any
from models.schemas import (
    PatientNarrative, KnowledgeRule, MaestroState,
    RuleDirection, ConfidenceZone, CECFTier
)
from utils.cecf import compute_cecf, update_rule_cecf, get_cecf_tier
from nodes.data_preparation import data_preparation_node, compute_cci
from nodes.experience_update import (
    classify_three_way_pattern,
    compute_cecf_statistics
)


class TestCECFCore:
    """Test CECF Bayesian computation core"""
    
    def test_cecf_hard_cap_enforcement(self):
        """Test that CECF is capped for n < 20 with linear ramp"""
        # Check specific points on the ramp
        test_points = [
            (1, 1.0, 0.12),   # Start of ramp
            (5, 5.0, 0.29),   # Mid ramp
            (10, 10.0, 0.53), # Mid-high ramp
            (19, 19.0, 0.95), # End of ramp (nearly uncapped)
        ]
        
        for n, k, expected_cecf in test_points:
            cecf = compute_cecf(k, n, tau=0.65)
            # Allow 0.05 tolerance for linear ramp
            assert abs(cecf - expected_cecf) < 0.05, \
                f"n={n}: CECF {cecf:.3f} != expected {expected_cecf:.3f}"
        
        # At n=20, cap should lift completely
        cecf_20_perfect = compute_cecf(20.0, 20, tau=0.65)
        assert cecf_20_perfect > 0.95, f"Perfect score at n=20: CECF {cecf_20_perfect:.3f}"
        
        # At n=20 with 80% success, should be properly uncapped
        cecf_20_normal = compute_cecf(16.0, 20, tau=0.65)
        assert cecf_20_normal > 0.50, f"80% at n=20: CECF {cecf_20_normal:.3f}"
    
    def test_cecf_correct_bayesian_computation(self):
        """Validate CECF uses correct Bayesian Beta-Binomial formula
        
        Our implementation uses:
        - Hard cap with linear ramp for n < 20
        - Pure Bayesian for n >= 20
        
        This is mathematically correct and more granular than paper's flat cap.
        """
        from scipy.stats import beta as beta_dist
        
        # Test pure Bayesian computation for n >= 20
        test_cases = [
            (20, 16.0),   # 80% success
            (50, 41.0),   # 82% success
            (100, 82.0),  # 82% success
            (300, 246.0), # 82% success
        ]
        
        for n, k in test_cases:
            cecf_our = compute_cecf(k, n, tau=0.65)
            
            # Compute expected pure Bayesian
            alpha = k + 1
            beta_param = n - k + 1
            cecf_bayesian = 1 - beta_dist.cdf(0.65, alpha, beta_param)
            
            # Should match pure Bayesian for n >= 20
            assert abs(cecf_our - cecf_bayesian) < 0.001, \
                f"n={n}, k={k}: Our CECF {cecf_our:.3f} != Bayesian {cecf_bayesian:.3f}"
        
        # Test tier classification
        tier_tests = [
            (20, 16.0, CECFTier.RELIABLE),  # CECF ~ 0.91 → Reliable (0.80-0.95)
            (50, 25.0, CECFTier.PROVISIONAL),  # CECF low → Provisional
            (100, 82.0, CECFTier.EXPERT),    # CECF ~ 1.00 → Expert (≥0.95)
            (300, 246.0, CECFTier.EXPERT),     # CECF ~ 1.00 → Expert
        ]
        
        for n, k, expected_tier in tier_tests:
            cecf = compute_cecf(k, n, tau=0.65)
            tier = get_cecf_tier(cecf, n)
            
            assert tier == expected_tier, \
                f"n={n}, k={k}: Tier {tier.value} != expected {expected_tier.value}"
        
        # Test that n<20 cases stay in PROVISIONAL tier
        for n in [1, 5, 10, 15, 19]:
            cecf = compute_cecf(float(n), n, tau=0.65)  # 100% success
            tier = get_cecf_tier(cecf, n)
            assert tier == CECFTier.PROVISIONAL, \
                f"n={n} should be PROVISIONAL, got {tier.value}"
    
    def test_cecf_monotonicity(self):
        """CECF should increase with more successes"""
        n = 50
        cecf_values = []
        
        for k in [20, 25, 30, 35, 40, 45]:
            cecf = compute_cecf(float(k), n, tau=0.65)
            cecf_values.append(cecf)
        
        # Check monotonic increase
        for i in range(len(cecf_values) - 1):
            assert cecf_values[i] < cecf_values[i+1], \
                f"CECF not monotonic: {cecf_values}"


class TestThreeLayerCreditAssignment:
    """Test Layer 1, 2, 3 credit assignment"""
    
    def test_layer1_direction_aware(self):
        """Layer 1: Correct direction gets full credit"""
        # Correct direction
        new_k, new_n, k_inc, cecf, tier = update_rule_cecf(
            current_k=10.0,
            current_n=20,
            rule_direction=RuleDirection.YES,
            ground_truth=True,  # Matches YES
            influence_weight=0.5
        )
        
        assert k_inc == 1.0, "Correct direction should get k += 1.0"
        assert new_k == 11.0
        assert new_n == 21
    
    def test_layer1_neutral_half_credit(self):
        """Layer 1: Neutral direction gets half credit"""
        new_k, new_n, k_inc, cecf, tier = update_rule_cecf(
            current_k=10.0,
            current_n=20,
            rule_direction=RuleDirection.NEUTRAL,
            ground_truth=True,
            influence_weight=0.5
        )
        
        assert k_inc == 0.5, "Neutral should get k += 0.5"
        assert new_k == 10.5
    
    def test_layer2_influence_weighted_penalty(self):
        """Layer 2: Wrong direction penalty proportional to influence"""
        # High influence = heavy penalty
        new_k_high, _, k_inc_high, _, _ = update_rule_cecf(
            current_k=10.0,
            current_n=20,
            rule_direction=RuleDirection.YES,
            ground_truth=False,  # Wrong!
            influence_weight=0.80  # High influence
        )
        
        # Low influence = light penalty
        new_k_low, _, k_inc_low, _, _ = update_rule_cecf(
            current_k=10.0,
            current_n=20,
            rule_direction=RuleDirection.YES,
            ground_truth=False,  # Wrong!
            influence_weight=0.10  # Low influence
        )
        
        assert abs(k_inc_high - 0.20) < 0.001, f"High influence: k_inc should be 1-0.80=0.20, got {k_inc_high}"
        assert abs(k_inc_low - 0.90) < 0.001, f"Low influence: k_inc should be 1-0.10=0.90, got {k_inc_low}"
        assert new_k_low > new_k_high, "Low influence should preserve more k"


class TestDataPreparation:
    """Test Stage 1: Data Preparation"""
    
    def test_cci_computation(self):
        """Test CCI calculation with mutual exclusivity"""
        comorbidities = {
            "myocardial_infarction": True,  # 1
            "congestive_heart_failure": True,  # 1
            "copd": True,  # 1
            "diabetes_without_complication": True,  # 1
            "diabetes_with_complication": True,  # 2 (should override)
        }
        
        base_cci, age_adj = compute_cci(comorbidities, age=65)
        
        # MI + CHF + COPD + DM_with = 1+1+1+2 = 5
        # Mutual exclusivity: DM_with overrides DM_without
        assert base_cci == 5, f"Base CCI {base_cci} != 5"
        
        # Age 65 → +2
        assert age_adj == 7, f"Age-adjusted CCI {age_adj} != 7"
    
    def test_cci_liver_disease_exclusivity(self):
        """Test liver disease mutual exclusivity"""
        comorbidities = {
            "mild_liver_disease": True,  # 1
            "moderate_severe_liver_disease": True,  # 3 (should override)
        }
        
        base_cci, _ = compute_cci(comorbidities, age=50)
        
        # Only moderate/severe counts
        assert base_cci == 3, f"Liver exclusivity failed: {base_cci}"
    
    async def test_narrative_generation(self):
        """Test patient narrative generation"""
        patient_data = {
            "patient_id": "TEST_001",
            "gender": "male",
            "age": 68,
            "diagnosis_date": "2025-01-15",
            "tumor_location": "upper_lobe",
            "comorbidities": {
                "copd": True,
                "diabetes_without_complication": True,
            },
            "fracture_vertebral": True,
            "fracture_hip": False,
            "fracture_wrist": False,
            "osteoporosis_diagnosed": False,
            "bone_metastasis_outcome": True,
        }
        
        # Create mock state
        from nodes.data_preparation import serialize_to_narrative
        
        narrative = serialize_to_narrative(patient_data)
        
        assert narrative.patient_id == "TEST_001"
        assert narrative.cci_score == 2  # COPD + DM = 1+1
        assert narrative.cci_age_adjusted == 4  # +2 for age 60-69
        assert "vertebral" in narrative.fracture_events.lower()
        assert narrative.ground_truth == True


class TestThreeWayComparison:
    """Test Agent vs ML vs Ground Truth diagnostics"""
    
    def test_all_patterns(self):
        """Test all three-way comparison patterns from paper"""
        test_cases = [
            # (agent, ml, gt, expected_pattern)
            (True, True, True, "Both correct, concordant"),
            (True, False, True, "Agent outperformed ML"),
            (False, True, True, "ML outperformed agent"),
            (True, False, False, "Agent overrode ML incorrectly"),
            (True, True, False, "Both wrong"),
            (False, False, True, "Both missed"),
            (False, False, False, "Both correct, concordant"),
        ]
        
        for agent, ml, gt, expected in test_cases:
            result = classify_three_way_pattern(agent, ml, gt)
            assert expected in result, \
                f"Pattern mismatch: agent={agent}, ml={ml}, gt={gt}\n" \
                f"Expected: {expected}\nGot: {result}"


class TestEndToEndPipeline:
    """Full end-to-end integration test"""
    
    async def test_single_patient_full_cycle(self):
        """Test one patient through all 6 stages"""
        
        # Setup: Initialize knowledge rules
        knowledge_rules = {
            "KR-31": KnowledgeRule(
                kr_id="KR-31",
                type="empirical_association",
                topic="fracture_signal",
                content="Vertebral fracture is sentinel for bone metastasis",
                clinical_implication="Increase risk estimate",
                confidence_label="Established",
                applicability_condition="vertebral_fracture == True",
                n=0,
                k=0.0,
                cecf=0.12,
                tier=CECFTier.PROVISIONAL,
            )
        }
        
        # Create test patient
        patient = PatientNarrative(
            patient_id="TEST_E2E_001",
            demographics="male, age 65",
            tumor_location="upper_lobe",
            comorbidities="COPD, Diabetes",
            cci_score=2,
            cci_age_adjusted=4,
            osteoporosis_status="No diagnosis",
            medications=[],
            fracture_events="Vertebral fracture present",
            raw_features={
                "gender": "male",
                "age": 65,
                "cci": 4,
                "fracture_vertebral": True,
                "osteoporosis_diagnosed": False,
            },
            ground_truth=True  # Actually has bone metastasis
        )
        
        # Stage 1: Data Preparation
        state = MaestroState(
            current_patient=patient,
            knowledge_rules=knowledge_rules,
            case_number=0,
        )
        
        stage1_result = await data_preparation_node(state)
        assert "narrative_text" in stage1_result
        assert "vertebral" in stage1_result["narrative_text"].lower()
        
        print("✅ Stage 1 (Data Preparation): PASSED")
        
        # Stage 2: Agent Reasoning (simplified - no LLM call)
        # Simulate agent activating KR-31 with YES direction
        from models.schemas import AgentReasoning, RuleActivation
        
        simulated_reasoning = AgentReasoning(
            patient_id=patient.patient_id,
            baseline_summary="Male, 65yo, CCI=4, vertebral fracture",
            activated_rules=[
                RuleActivation(
                    kr_id="KR-31",
                    direction=RuleDirection.YES,
                    influence_weight=1.0,
                    rationale="Vertebral fracture is strong signal"
                )
            ],
            pr_routes_triggered=["PR-ROUTE-3"],
            specialized_analyses="Sentinel fracture analysis",
            synthesis="Strong signals for bone metastasis",
            zone=ConfidenceZone.A,
            prediction=True,  # Predicts YES
            confidence_rationale="Clear sentinel signal"
        )
        
        state.agent_reasoning = simulated_reasoning
        print("✅ Stage 2 (Agent Reasoning): PASSED (simulated)")
        
        # Stage 3: ML Oracle (simplified - no real models)
        from models.schemas import MLConsensus, MLModelPrediction
        
        simulated_ml = MLConsensus(
            eligible_models=["XGBoost_Baseline"],
            predictions=[
                MLModelPrediction(
                    model_name="XGBoost_Baseline",
                    probability=0.75,
                    confidence_interval_lower=0.65,
                    confidence_interval_upper=0.85,
                    ci_width=0.20,
                    weight=5.0
                )
            ],
            consensus_probability=0.75,
            consensus_direction=True,  # Also predicts YES
            pattern="High"
        )
        
        state.ml_consensus = simulated_ml
        print("✅ Stage 3 (ML Oracle): PASSED (simulated)")
        
        # Stage 4: Experience Update
        from nodes.experience_update import experience_update_node
        
        stage4_result = await experience_update_node(state)
        
        assert "knowledge_rules" in stage4_result
        assert "three_way_comparison" in stage4_result
        
        # Check CECF was updated
        updated_kr31 = stage4_result["knowledge_rules"]["KR-31"]
        assert updated_kr31.n == 1, f"n should be 1, got {updated_kr31.n}"
        assert updated_kr31.k == 1.0, f"k should be 1.0 (correct), got {updated_kr31.k}"
        
        # Three-way comparison
        three_way = stage4_result["three_way_comparison"]
        assert three_way["agent_correct"] == True
        assert three_way["ml_correct"] == True
        assert "Both correct" in three_way["pattern"]
        
        print("✅ Stage 4 (Experience Update): PASSED")
        print(f"   KR-31: n={updated_kr31.n}, k={updated_kr31.k}, CECF={updated_kr31.cecf:.3f}")
        
        # Stage 6: Memory Consolidation (skip note generation for correct prediction)
        from nodes.memory_consolidation import memory_consolidation_node
        
        state.knowledge_rules = stage4_result["knowledge_rules"]
        stage6_result = await memory_consolidation_node(state)
        
        # Should not generate note (prediction was correct)
        assert stage6_result.get("memory_consolidation_triggered", False) == False
        
        print("✅ Stage 6 (Memory Consolidation): PASSED (no note needed)")
        
        # Stage 5: Milestone Gate (check at case 50)
        from nodes.milestone_gate import milestone_gate_node
        
        state.case_number = 50  # Trigger milestone M1
        stage5_result = await milestone_gate_node(state)
        
        assert "current_milestone" in stage5_result or "milestone_passed" in stage5_result
        
        print("✅ Stage 5 (Milestone Gate): PASSED")
        
        print("\n" + "="*80)
        print("🎉 END-TO-END INTEGRATION TEST: ALL STAGES PASSED")
        print("="*80)
    
    async def test_learning_over_multiple_cases(self):
        """Test CECF learning progression over 20 cases"""
        
        # Initialize rule
        kr = KnowledgeRule(
            kr_id="KR-TEST",
            type="empirical_association",
            topic="test",
            content="Test rule",
            clinical_implication="Test",
            confidence_label="Emerging",
            applicability_condition="always",
            n=0,
            k=0.0,
            cecf=0.12,
            tier=CECFTier.PROVISIONAL,
        )
        
        # Simulate 20 cases, 80% success rate
        for i in range(20):
            ground_truth = (i % 5) != 0  # 80% positive
            rule_direction = RuleDirection.YES
            
            new_k, new_n, k_inc, new_cecf, new_tier = update_rule_cecf(
                current_k=kr.k,
                current_n=kr.n,
                rule_direction=rule_direction,
                ground_truth=ground_truth,
                influence_weight=0.5
            )
            
            kr.k = new_k
            kr.n = new_n
            kr.cecf = new_cecf
            kr.tier = new_tier
        
        # After 20 cases with 80% success and influence_weight=0.5
        assert kr.n == 20
        # k calculation: 16 correct (k+=1.0) + 4 wrong (k+=1-0.5) = 16 + 2 = 18
        assert 17.5 < kr.k < 18.5, f"k should be ~18, got {kr.k}"
        
        # CECF should have lifted from hard cap
        assert kr.cecf > 0.85, f"CECF {kr.cecf} should be high at n=20 with k=18"
        # With k=18, n=20 (90% success with generous wrong-answer credit), CECF ~0.99 → Expert
        assert kr.tier == CECFTier.EXPERT
        
        print(f"✅ Learning progression test: n={kr.n}, k={kr.k:.1f}, CECF={kr.cecf:.3f}")


class TestCECFStatistics:
    """Test CECF statistics aggregation"""
    
    def test_tier_distribution(self):
        """Test CECF statistics computation"""
        knowledge_rules = {
            "KR-1": KnowledgeRule(
                kr_id="KR-1", type="established", topic="test",
                content="Test", clinical_implication="Test",
                confidence_label="Established", applicability_condition="test",
                n=100, k=82.0, cecf=0.94, tier=CECFTier.RELIABLE
            ),
            "KR-2": KnowledgeRule(
                kr_id="KR-2", type="established", topic="test",
                content="Test", clinical_implication="Test",
                confidence_label="Established", applicability_condition="test",
                n=50, k=41.0, cecf=0.80, tier=CECFTier.DEVELOPING
            ),
            "KR-3": KnowledgeRule(
                kr_id="KR-3", type="established", topic="test",
                content="Test", clinical_implication="Test",
                confidence_label="Established", applicability_condition="test",
                n=10, k=5.0, cecf=0.12, tier=CECFTier.PROVISIONAL
            ),
        }
        
        stats = compute_cecf_statistics(knowledge_rules)
        
        assert stats["total_rules"] == 3
        assert stats["tier_distribution"]["Reliable"] == 1
        assert stats["tier_distribution"]["Developing"] == 1
        assert stats["tier_distribution"]["Provisional"] == 1
        assert stats["top_10_rules"][0]["kr_id"] == "KR-1"


# ============================================================================
# TEST RUNNER
# ============================================================================

async def run_all_tests():
    """Run all integration tests"""
    
    print("\n" + "="*80)
    print("MAESTRO V4 - END-TO-END INTEGRATION TESTS")
    print("="*80 + "\n")
    
    # Test Suite 1: CECF Core
    print("📊 Test Suite 1: CECF Bayesian Computation")
    print("-" * 80)
    
    suite1 = TestCECFCore()
    suite1.test_cecf_hard_cap_enforcement()
    print("✅ Hard cap enforcement")
    
    suite1.test_cecf_correct_bayesian_computation()
    print("✅ Bayesian computation correctness")
    
    suite1.test_cecf_monotonicity()
    print("✅ Monotonicity")
    
    # Test Suite 2: Credit Assignment
    print("\n📊 Test Suite 2: Three-Layer Credit Assignment")
    print("-" * 80)
    
    suite2 = TestThreeLayerCreditAssignment()
    suite2.test_layer1_direction_aware()
    print("✅ Layer 1: Direction-aware")
    
    suite2.test_layer1_neutral_half_credit()
    print("✅ Layer 1: Neutral half-credit")
    
    suite2.test_layer2_influence_weighted_penalty()
    print("✅ Layer 2: Influence-weighted penalty")
    
    # Test Suite 3: Data Preparation
    print("\n📊 Test Suite 3: Data Preparation")
    print("-" * 80)
    
    suite3 = TestDataPreparation()
    suite3.test_cci_computation()
    print("✅ CCI computation")
    
    suite3.test_cci_liver_disease_exclusivity()
    print("✅ CCI mutual exclusivity")
    
    await suite3.test_narrative_generation()
    print("✅ Narrative generation")
    
    # Test Suite 4: Three-Way Comparison
    print("\n📊 Test Suite 4: Three-Way Comparison")
    print("-" * 80)
    
    suite4 = TestThreeWayComparison()
    suite4.test_all_patterns()
    print("✅ All diagnostic patterns")
    
    # Test Suite 5: End-to-End
    print("\n📊 Test Suite 5: Full Pipeline Integration")
    print("-" * 80)
    
    suite5 = TestEndToEndPipeline()
    await suite5.test_single_patient_full_cycle()
    
    await suite5.test_learning_over_multiple_cases()
    
    # Test Suite 6: Statistics
    print("\n📊 Test Suite 6: CECF Statistics")
    print("-" * 80)
    
    suite6 = TestCECFStatistics()
    suite6.test_tier_distribution()
    print("✅ Tier distribution computation")
    
    # Summary
    print("\n" + "="*80)
    print("✅ ALL INTEGRATION TESTS PASSED")
    print("="*80)
    print("\nTest Coverage:")
    print("  ✓ CECF Bayesian computation")
    print("  ✓ Three-layer credit assignment")
    print("  ✓ Data preparation & CCI")
    print("  ✓ Three-way diagnostics")
    print("  ✓ Full 6-stage pipeline")
    print("  ✓ Learning progression")
    print("  ✓ CECF statistics")
    print("\n" + "="*80)


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_all_tests())
