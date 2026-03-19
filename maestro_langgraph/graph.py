"""
MAESTRO V4 LangGraph Implementation
Main StateGraph with 6 nodes forming a closed-loop training system
"""

import os
from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from models.schemas import MaestroState, MilestoneLevel
from nodes.data_preparation import data_preparation_node
from nodes.agent_reasoning import agent_reasoning_node
from nodes.ml_oracle import ml_oracle_node
from nodes.experience_update import experience_update_node
from nodes.milestone_gate import milestone_gate_node
from nodes.memory_consolidation import memory_consolidation_node


# ============================================================================
# CONDITIONAL EDGES (Routing Logic)
# ============================================================================

def should_continue_training(state: MaestroState) -> Literal["terminate", "next_patient"]:
    """
    After milestone gate, decide whether to continue or terminate.
    
    Terminate if:
        - Milestone gate failed
        - Error occurred
        - Reached max cases (5000)
    """
    if state.should_terminate or not state.milestone_passed:
        return "terminate"
    
    if state.case_number >= 5000:
        return "terminate"
    
    return "next_patient"


def route_after_reasoning(state: MaestroState) -> Literal["ml_oracle", "error"]:
    """
    After agent reasoning, route to ML Oracle or error handling.
    """
    if state.agent_reasoning is None or state.error_message:
        return "error"
    return "ml_oracle"


# ============================================================================
# MILESTONE CHECKPOINTS
# ============================================================================

MILESTONE_CHECKPOINTS = {
    50: (MilestoneLevel.M1_INTERN, 0.60),
    200: (MilestoneLevel.M2_RESIDENT, 0.68),
    500: (MilestoneLevel.M3_FELLOW, 0.72),
    1500: (MilestoneLevel.M4_SENIOR_FELLOW, 0.74),
    2000: (MilestoneLevel.M5_ASSOCIATE_PROFESSOR, 0.76),
    3500: (MilestoneLevel.M6_PROFESSOR, 0.80),
    5000: (MilestoneLevel.M7_KEY_OPINION_LEADER, 0.82),
}


def should_check_milestone(state: MaestroState) -> Literal["check_milestone", "skip_milestone"]:
    """
    Determine if current case number triggers milestone evaluation.
    """
    if state.case_number in MILESTONE_CHECKPOINTS:
        return "check_milestone"
    return "skip_milestone"


# ============================================================================
# BUILD GRAPH
# ============================================================================

def build_maestro_graph(
    llm_model: str = "gpt-4o",
    checkpoint_dir: str = "./checkpoints",
    enable_persistence: bool = True
) -> StateGraph:
    """
    Build the complete MAESTRO LangGraph workflow.
    
    Graph Structure (6 nodes, closed loop):
    
        START → data_preparation → agent_reasoning → ml_oracle 
          ↑                                             ↓
          |                                       experience_update
          |                                             ↓
          |                                    memory_consolidation
          |                                             ↓
          |                                       milestone_gate
          |                                       /           \\
          |                                  [pass]         [fail]
          |                                    /               \\
          +---[next_patient]------------------              [END/CKIP]
    
    Args:
        llm_model: Primary LLM for reasoning (gpt-4o, claude-sonnet-4, etc.)
        checkpoint_dir: Directory for persistent checkpoints
        enable_persistence: Enable PostgreSQL checkpointing for resume/audit
    
    Returns:
        Compiled LangGraph workflow
    """
    
    # Initialize LLM
    if "gpt" in llm_model.lower():
        llm = ChatOpenAI(model=llm_model, temperature=0.0)
    else:
        llm = ChatAnthropic(model=llm_model, temperature=0.0)
    
    # Build StateGraph
    workflow = StateGraph(MaestroState)
    
    # Add nodes (6 stages)
    workflow.add_node("data_preparation", data_preparation_node)
    workflow.add_node("agent_reasoning", lambda state: agent_reasoning_node(state, llm))
    workflow.add_node("ml_oracle", ml_oracle_node)
    workflow.add_node("experience_update", experience_update_node)
    workflow.add_node("memory_consolidation", memory_consolidation_node)
    workflow.add_node("milestone_gate", milestone_gate_node)
    
    # Add edges (flow control)
    # Stage 1 → Stage 2
    workflow.add_edge("data_preparation", "agent_reasoning")
    
    # Stage 2 → Stage 3 (with error handling)
    workflow.add_conditional_edges(
        "agent_reasoning",
        route_after_reasoning,
        {
            "ml_oracle": "ml_oracle",
            "error": END
        }
    )
    
    # Stage 3 → Stage 4
    workflow.add_edge("ml_oracle", "experience_update")
    
    # Stage 4 → Stage 6 (CECE consolidation)
    workflow.add_edge("experience_update", "memory_consolidation")
    
    # Stage 6 → Stage 5 (Milestone check conditional)
    workflow.add_conditional_edges(
        "memory_consolidation",
        should_check_milestone,
        {
            "check_milestone": "milestone_gate",
            "skip_milestone": "data_preparation"  # Loop back to next patient
        }
    )
    
    # Stage 5 → Continue or Terminate
    workflow.add_conditional_edges(
        "milestone_gate",
        should_continue_training,
        {
            "next_patient": "data_preparation",  # CLOSED LOOP
            "terminate": END
        }
    )
    
    # Set entry point
    workflow.set_entry_point("data_preparation")
    
    # Compile with checkpointing
    if enable_persistence:
        # PostgreSQL checkpointer for production
        checkpointer = PostgresSaver.from_conn_string(
            os.getenv("POSTGRES_CONN_STRING", "postgresql://localhost/maestro")
        )
        compiled_graph = workflow.compile(checkpointer=checkpointer)
    else:
        # In-memory for development/testing
        compiled_graph = workflow.compile()
    
    return compiled_graph


# ============================================================================
# GRAPH EXECUTION
# ============================================================================

async def run_maestro_training(
    patients_dataset: list,
    validation_dataset: list,
    initial_knowledge_rules: Dict[str, Any],
    ml_models: Dict[str, Any],
    config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Execute MAESTRO training loop on dataset.
    
    Args:
        patients_dataset: List of patient records (up to 5000)
        validation_dataset: Held-out set for milestone evaluation (315 cases)
        initial_knowledge_rules: 48 baseline KRs + any inherited CKIP rules
        ml_models: Pre-trained ML model pool
        config: Runtime configuration
    
    Returns:
        Final knowledge state + training metrics
    """
    
    config = config or {}
    
    # Build graph
    graph = build_maestro_graph(
        llm_model=config.get("llm_model", "gpt-4o"),
        enable_persistence=config.get("enable_persistence", True)
    )
    
    # Initialize state
    initial_state = MaestroState(
        knowledge_rules=initial_knowledge_rules,
        case_number=0,
        memory_store=[],
        should_terminate=False,
    )
    
    # Store validation set and ML models in config for nodes to access
    thread_config = {
        "configurable": {
            "thread_id": config.get("run_id", "maestro_run_1"),
            "validation_dataset": validation_dataset,
            "ml_models": ml_models,
        }
    }
    
    # Run training loop
    results = []
    
    for i, patient in enumerate(patients_dataset):
        # Update state with current patient
        initial_state.current_patient = patient
        
        # Execute one iteration (one patient through 6 stages)
        async for output in graph.astream(initial_state, thread_config):
            results.append(output)
            
            # Check termination
            if output.get("should_terminate", False):
                print(f"Training terminated at case {i+1}")
                break
        
        # Check if we should stop
        if results[-1].get("should_terminate", False):
            break
    
    # Extract final knowledge state
    final_state = results[-1] if results else initial_state
    
    return {
        "final_knowledge_rules": final_state.get("knowledge_rules", {}),
        "final_memory_store": final_state.get("memory_store", []),
        "total_cases_processed": final_state.get("case_number", 0),
        "final_milestone": final_state.get("current_milestone"),
        "final_validation_auc": final_state.get("validation_auc", 0.0),
        "milestone_passed": final_state.get("milestone_passed", False),
        "ckip_eligible_rules": [
            kr for kr in final_state.get("knowledge_rules", {}).values()
            if kr.cecf > 0.97 and kr.n >= 300
        ]
    }


# ============================================================================
# VISUALIZATION (LangGraph Studio)
# ============================================================================

def visualize_graph():
    """
    Generate graph visualization for LangGraph Studio.
    Run: langgraph studio graph.py
    """
    graph = build_maestro_graph(enable_persistence=False)
    return graph


if __name__ == "__main__":
    # For LangGraph Studio visualization
    print("MAESTRO V4 LangGraph Implementation")
    print("Run: langgraph studio graph.py")
    print("\nGraph structure:")
    graph = build_maestro_graph(enable_persistence=False)
    print(graph.get_graph().draw_ascii())
