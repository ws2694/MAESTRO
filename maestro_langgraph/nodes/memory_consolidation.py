"""
Stage 6: Memory Consolidation Node (CECE)
Clinical Experience Consolidation Engine
"""

from typing import Dict, Any, List
import hashlib
from datetime import datetime
from models.schemas import (
    MaestroState, ExperienceNote
)


async def memory_consolidation_node(state: MaestroState) -> Dict[str, Any]:
    """
    LangGraph node for Stage 6: CECE Memory Consolidation.
    
    Manages agent-discovered patterns:
        1. Generate experience notes (on errors, Zone C correct, novel observations)
        2. Similarity retrieval (embedding + LLM classification)
        3. Observation → Candidate KR → Promoted KR progression
        4. Periodic consolidation (cluster/merge, deprecation)
    
    Returns updated memory store.
    """
    
    # Check if experience note should be generated
    should_generate = check_note_trigger(state)
    
    if not should_generate:
        # No note needed, just pass through
        return {}
    
    # Generate experience note
    note = await generate_experience_note(state)
    
    if note is None:
        return {}
    
    # Similarity retrieval: Is this pattern new?
    memory_store = state.memory_store.copy()
    similar_notes = find_similar_notes(note, memory_store)
    
    if similar_notes:
        # Classify relationship (DUPLICATE/REFINEMENT/CONTRADICTION/NOVEL)
        relationship = await classify_note_relationship(note, similar_notes[0])
        
        if relationship == "DUPLICATE":
            # Just update counter, don't add new note
            similar_notes[0].n += 1
        elif relationship == "REFINEMENT":
            # Merge content, preserve CECF history
            similar_notes[0] = merge_notes(similar_notes[0], note)
        elif relationship == "CONTRADICTION":
            # Flag for rule split analysis
            similar_notes[0].status = "candidate_for_split"
        elif relationship == "NOVEL":
            # Add as new observation
            memory_store.append(note)
    else:
        # No similar notes, add as new observation
        memory_store.append(note)
    
    # Periodic consolidation (every 50 cases)
    if state.case_number % 50 == 0:
        memory_store = periodic_consolidation(memory_store)
    
    return {
        "memory_store": memory_store,
        "memory_consolidation_triggered": should_generate
    }


# ============================================================================
# EXPERIENCE NOTE GENERATION
# ============================================================================

def check_note_trigger(state: MaestroState) -> bool:
    """
    Check if experience note should be generated.
    
    Triggers:
        1. Agent made an error (prediction != ground truth)
        2. Zone C but correct (low confidence but lucky)
        3. Novel observation (agent noticed unusual pattern)
    """
    if state.agent_reasoning is None or state.current_patient is None:
        return False
    
    agent_pred = state.agent_reasoning.prediction
    ground_truth = state.current_patient.ground_truth
    zone = state.agent_reasoning.zone
    
    # Trigger 1: Error
    if agent_pred != ground_truth:
        return True
    
    # Trigger 2: Zone C but correct
    if zone == "C" and agent_pred == ground_truth:
        return True
    
    # Trigger 3: Novel observation (heuristic: unusual feature combinations)
    # In production, the LLM would flag this during reasoning
    # For now, skip this trigger
    
    return False


async def generate_experience_note(state: MaestroState) -> ExperienceNote:
    """
    Generate structured experience note.
    
    In production, this would call LLM with experience_note_prompt.
    For now, create structured note from state.
    """
    agent_pred = state.agent_reasoning.prediction
    ground_truth = state.current_patient.ground_truth
    zone = state.agent_reasoning.zone
    
    # Determine trigger
    if agent_pred != ground_truth:
        trigger = "error"
    elif zone == "C":
        trigger = "zone_c_correct"
    else:
        trigger = "novel_observation"
    
    # Generate note content (in production, LLM-generated)
    pattern = extract_pattern_from_reasoning(state)
    
    note_id = hashlib.md5(
        f"{state.case_number}_{trigger}".encode()
    ).hexdigest()[:12]
    
    return ExperienceNote(
        note_id=note_id,
        patient_id=state.current_patient.patient_id,
        case_number=state.case_number,
        trigger=trigger,
        content=pattern,
        n=1,
        k=1.0 if trigger == "zone_c_correct" else 0.0,
        cecf=0.12,  # Starts provisional
        status="observation",
        created_at=datetime.utcnow().isoformat(),
        related_rules=[ra.kr_id for ra in state.agent_reasoning.activated_rules]
    )


def extract_pattern_from_reasoning(state: MaestroState) -> str:
    """
    Extract key pattern from agent's reasoning.
    
    In production, LLM would generate this with reflection prompt.
    For now, create from synthesis.
    """
    synthesis = state.agent_reasoning.synthesis
    activated_rules = state.agent_reasoning.activated_rules
    
    # Simple heuristic: mention key activated rules
    rule_ids = [ra.kr_id for ra in activated_rules if ra.influence_weight > 0.2]
    
    return f"Pattern involving: {', '.join(rule_ids)}. {synthesis[:200]}..."


# ============================================================================
# SIMILARITY RETRIEVAL
# ============================================================================

def find_similar_notes(
    new_note: ExperienceNote,
    memory_store: List[ExperienceNote],
    threshold: float = 0.75
) -> List[ExperienceNote]:
    """
    Find similar notes using embedding similarity.
    
    In production:
        - Compute embeddings with text-embedding-3-large
        - Store in ChromaDB vector store
        - Cosine similarity search
    
    For now, use simple heuristic (related rules overlap).
    """
    similar = []
    
    for existing_note in memory_store:
        # Heuristic similarity: shared related rules
        shared_rules = set(new_note.related_rules) & set(existing_note.related_rules)
        similarity = len(shared_rules) / max(len(new_note.related_rules), 1)
        
        if similarity >= threshold:
            similar.append(existing_note)
    
    return similar


async def classify_note_relationship(
    new_note: ExperienceNote,
    existing_note: ExperienceNote
) -> str:
    """
    Classify relationship between notes.
    
    In production, LLM would classify with structured prompt.
    
    Returns: DUPLICATE / REFINEMENT / CONTRADICTION / NOVEL
    """
    # Simple heuristic
    if new_note.trigger == existing_note.trigger:
        if new_note.content[:50] == existing_note.content[:50]:
            return "DUPLICATE"
        else:
            return "REFINEMENT"
    else:
        return "NOVEL"


def merge_notes(existing: ExperienceNote, new: ExperienceNote) -> ExperienceNote:
    """
    Merge refinement into existing note.
    
    CECF histories are summed (k, n accumulate).
    """
    existing.content += f"\n\nREFINEMENT (Case {new.case_number}): {new.content}"
    existing.n += new.n
    existing.k += new.k
    
    from utils.cecf import compute_cecf, get_cecf_tier
    existing.cecf = compute_cecf(existing.k, existing.n)
    
    return existing


# ============================================================================
# PERIODIC CONSOLIDATION
# ============================================================================

def periodic_consolidation(memory_store: List[ExperienceNote]) -> List[ExperienceNote]:
    """
    Run four maintenance operations every 50 cases.
    
    1. Cluster and merge: High embedding similarity (>0.85)
    2. Promotion check: n≥15, CECF≥0.50 → candidate_kr
    3. Deprecation check: n>30, CECF<0.15 → deprecated
    4. Rule split check: Bimodal success patterns
    """
    from utils.cecf import compute_cecf, get_cecf_tier
    
    # 1. Cluster and merge (simplified)
    # In production: Use clustering algorithm on embeddings
    
    # 2. Promotion check
    for note in memory_store:
        if note.status == "observation" and note.n >= 15 and note.cecf >= 0.50:
            note.status = "candidate_kr"
    
    # 3. Deprecation check
    for note in memory_store:
        if note.n > 30 and note.cecf < 0.15:
            note.status = "deprecated"
    
    # 4. Rule split check (complex, skip for now)
    
    return memory_store
