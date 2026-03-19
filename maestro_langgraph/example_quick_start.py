"""
MAESTRO V4 - Quick Start Example
Demonstrates minimal setup to run MAESTRO training
"""

import asyncio
from typing import Dict, List, Any
from models.schemas import KnowledgeRule, PatientNarrative
from utils.cecf import print_cecf_table


# ============================================================================
# EXAMPLE: LOAD 48 BASELINE KNOWLEDGE RULES
# ============================================================================

def load_example_knowledge_rules() -> Dict[str, KnowledgeRule]:
    """
    Load example Knowledge Rules.
    
    In production, load from config/baseline_krs.json with all 48 rules.
    Here we show 5 representative rules for demonstration.
    """
    
    example_rules = [
        {
            "kr_id": "KR-31",
            "type": "empirical_association",
            "topic": "fracture_signal",
            "content": "Vertebral fracture in NSCLC patient is a strong sentinel signal for bone metastasis",
            "clinical_implication": "Vertebral fracture → increase bone metastasis risk estimate",
            "confidence_label": "Established",
            "applicability_condition": "patient.fracture_events.vertebral == True",
        },
        {
            "kr_id": "KR-10",
            "type": "empirical_association",
            "topic": "comorbidity_pattern",
            "content": "High Charlson Comorbidity Index (CCI ≥ 5) may indicate competing mortality risk",
            "clinical_implication": "High CCI → possible DECREASE in bone met risk (competing risk)",
            "confidence_label": "Emerging",
            "applicability_condition": "patient.cci_age_adjusted >= 5",
        },
        {
            "kr_id": "KR-23",
            "type": "established",
            "topic": "bone_medication",
            "content": "Zoledronic acid prescription suggests clinician concern for bone involvement",
            "clinical_implication": "Zoledronic acid → increase bone metastasis suspicion",
            "confidence_label": "Established",
            "applicability_condition": "patient.medications contains 'zoledronic'",
        },
        {
            "kr_id": "KR-42",
            "type": "established",
            "topic": "osteoporosis_status",
            "content": "Osteoporosis diagnosis increases bone health monitoring, detection bias possible",
            "clinical_implication": "Osteoporosis → potential detection bias correction needed",
            "confidence_label": "Supported",
            "applicability_condition": "patient.osteoporosis_diagnosed == True",
        },
        {
            "kr_id": "KR-15",
            "type": "empirical_association",
            "topic": "demographics",
            "content": "Male gender shows slightly higher bone metastasis rate in NSCLC cohorts",
            "clinical_implication": "Male → marginal increase in bone met risk",
            "confidence_label": "Emerging",
            "applicability_condition": "patient.gender == 'male'",
        },
    ]
    
    # Initialize with CECF starting values
    knowledge_rules = {}
    for rule_data in example_rules:
        kr = KnowledgeRule(**rule_data)
        kr.n = 0
        kr.k = 0.0
        kr.cecf = 0.12  # Provisional starting point
        knowledge_rules[kr.kr_id] = kr
    
    return knowledge_rules


# ============================================================================
# EXAMPLE: GENERATE SYNTHETIC PATIENT DATA
# ============================================================================

def generate_example_patients(n: int = 100) -> List[PatientNarrative]:
    """
    Generate synthetic patient data for demonstration.
    
    In production, load from CSV with real EHR data.
    """
    import random
    
    patients = []
    
    for i in range(n):
        # Simulate diverse patient profiles
        age = random.randint(45, 85)
        gender = random.choice(["male", "female"])
        cci = random.randint(0, 8)
        
        has_vertebral_fx = random.random() < 0.05
        has_osteo = random.random() < 0.15
        has_meds = random.random() < 0.05
        
        # Outcome (simplified risk model for demo)
        base_risk = 0.15
        if has_vertebral_fx:
            base_risk += 0.30
        if cci > 5:
            base_risk -= 0.08  # Competing mortality
        if has_osteo:
            base_risk += 0.08
        
        outcome = random.random() < base_risk
        
        raw_features = {
            "gender": gender,
            "age": age,
            "cci": cci,
            "cci_age_adjusted": cci + (1 if age >= 60 else 0),
            "fracture_vertebral": has_vertebral_fx,
            "fracture_hip": False,
            "fracture_wrist": False,
            "osteoporosis_diagnosed": has_osteo,
            "medications": ["zoledronic"] if has_meds else [],
            "tumor_location": random.choice(["upper_lobe", "lower_lobe", "main_bronchus"]),
        }
        
        patient = PatientNarrative(
            patient_id=f"DEMO_{i:04d}",
            demographics=f"{gender}, age {age}",
            tumor_location=raw_features["tumor_location"],
            comorbidities=f"CCI={cci}",
            cci_score=cci,
            cci_age_adjusted=raw_features["cci_age_adjusted"],
            osteoporosis_status="Diagnosed" if has_osteo else "No diagnosis",
            medications=raw_features["medications"],
            fracture_events="Vertebral fracture" if has_vertebral_fx else "None",
            raw_features=raw_features,
            ground_truth=outcome
        )
        patients.append(patient)
    
    return patients


# ============================================================================
# EXAMPLE: SIMPLIFIED TRAINING LOOP (WITHOUT LANGGRAPH)
# ============================================================================

def run_simple_training_demo():
    """
    Demonstrate MAESTRO concepts without LangGraph dependencies.
    
    This shows the core CECF learning loop in isolation.
    """
    print("=" * 80)
    print("MAESTRO V4 - Simplified Training Demo")
    print("=" * 80)
    print()
    
    # Load knowledge rules
    print("Loading 5 example Knowledge Rules...")
    knowledge_rules = load_example_knowledge_rules()
    
    for kr_id, kr in knowledge_rules.items():
        print(f"  {kr_id}: {kr.content[:60]}... [CECF={kr.cecf:.3f}]")
    print()
    
    # Generate synthetic patients
    print("Generating 100 synthetic patients...")
    patients = generate_example_patients(100)
    print(f"  Positive cases: {sum(1 for p in patients if p.ground_truth)}")
    print(f"  Negative cases: {sum(1 for p in patients if not p.ground_truth)}")
    print()
    
    # Training loop
    print("Starting training loop...")
    print("-" * 80)
    
    from utils.cecf import update_rule_cecf, compute_cecf
    from models.schemas import RuleDirection
    
    for i, patient in enumerate(patients):
        # Simulate rule activation (simplified)
        activated = []
        
        if patient.raw_features.get("fracture_vertebral"):
            activated.append(("KR-31", RuleDirection.YES, 0.40))
        
        if patient.raw_features.get("cci_age_adjusted", 0) >= 5:
            activated.append(("KR-10", RuleDirection.NO, 0.25))  # Competing mortality
        
        if "zoledronic" in patient.raw_features.get("medications", []):
            activated.append(("KR-23", RuleDirection.YES, 0.20))
        
        if patient.raw_features.get("osteoporosis_diagnosed"):
            activated.append(("KR-42", RuleDirection.NEUTRAL, 0.10))
        
        if patient.raw_features.get("gender") == "male":
            activated.append(("KR-15", RuleDirection.YES, 0.05))
        
        # Normalize weights
        if activated:
            total_weight = sum(w for _, _, w in activated)
            activated = [(kr, d, w/total_weight) for kr, d, w in activated]
        
        # Update CECF for each activated rule
        for kr_id, direction, weight in activated:
            if kr_id not in knowledge_rules:
                continue
            
            kr = knowledge_rules[kr_id]
            
            new_k, new_n, k_inc, new_cecf, new_tier = update_rule_cecf(
                current_k=kr.k,
                current_n=kr.n,
                rule_direction=direction,
                ground_truth=patient.ground_truth,
                influence_weight=weight
            )
            
            kr.k = new_k
            kr.n = new_n
            kr.cecf = new_cecf
            kr.tier = new_tier
        
        # Print progress every 20 cases
        if (i + 1) % 20 == 0:
            print(f"\nCase {i+1}/100:")
            for kr_id, kr in knowledge_rules.items():
                if kr.n > 0:
                    print(f"  {kr_id}: n={kr.n:3d}, k={kr.k:5.1f}, CECF={kr.cecf:.3f} [{kr.tier.value}]")
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print("\nFinal Knowledge State:")
    
    for kr_id, kr in sorted(knowledge_rules.items(), key=lambda x: x[1].cecf, reverse=True):
        if kr.n > 0:
            success_rate = (kr.k / kr.n) if kr.n > 0 else 0
            print(f"  {kr_id}: CECF={kr.cecf:.3f} ({kr.tier.value:12s}) | n={kr.n:3d} | k/n={success_rate:.1%}")
    
    print("\n" + "=" * 80)
    print("Key Observations:")
    print("-" * 80)
    
    # Identify top rule
    top_rule = max(knowledge_rules.values(), key=lambda kr: kr.cecf if kr.n > 0 else 0)
    print(f"✅ Most reliable rule: {top_rule.kr_id}")
    print(f"   {top_rule.content}")
    print(f"   CECF={top_rule.cecf:.3f}, n={top_rule.n}")
    
    # Identify deprecated
    deprecated = [kr for kr in knowledge_rules.values() if kr.n >= 30 and kr.cecf < 0.15]
    if deprecated:
        print(f"\n❌ Deprecated rules (n≥30, CECF<0.15):")
        for kr in deprecated:
            print(f"   {kr.kr_id}: CECF={kr.cecf:.3f}")
    
    print("\n" + "=" * 80)
    print("\nCECF Validation Table (from paper Section 4.4):")
    print("-" * 80)
    print_cecf_table()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    run_simple_training_demo()
    
    print("\n" + "=" * 80)
    print("Next Steps:")
    print("=" * 80)
    print("1. Run full LangGraph version:")
    print("   python main.py --config configs/default.yaml")
    print()
    print("2. Visualize in LangGraph Studio:")
    print("   langgraph studio graph.py")
    print()
    print("3. Load real NSCLC dataset:")
    print("   python data_loader.py --csv data/nsclc_7315.csv")
    print()
    print("4. Export CKIP rules:")
    print("   python scripts/export_ckip.py --run_id maestro_run_2026_01")
    print("=" * 80)
