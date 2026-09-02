import time
from typing import Dict, Any, List, Optional
from .tools import ToolRegistry, default_registry
from .memory import MemoryManager, AgentMemory
from .planner import (
    BasePlanner,
    AutonomousHeuristicPlanner,
    ExternalLLMPlanner,
    GGUFLocalLLMPlanner
)

class AgenticExecutor:
    """
    Core ReAct (Reasoning + Action + Observation) Orchestrator for the Agentic AI.
    Seamlessly supports Qwen3-4B-Q4_K_M.gguf on Apple Silicon Metal GPU,
    Autonomous Cognitive Planner, and External LLM providers.
    """
    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        default_max_iterations: int = 6
    ):
        self.registry = registry or default_registry
        self.default_max_iterations = default_max_iterations

    def run(
        self,
        prompt: str,
        session_id: Optional[str] = None,
        max_iterations: Optional[int] = None,
        planner_mode: str = "gguf",
        model_path: Optional[str] = None,
        llm_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes an autonomous agentic task based on the user's objective.
        """
        start_time = time.perf_counter()
        iterations_limit = max_iterations or self.default_max_iterations
        
        # 1. Retrieve session memory
        memory: AgentMemory = MemoryManager.get_session(session_id)
        memory.add_history_entry(role="user", content=prompt)

        # 2. Select appropriate planner
        if planner_mode == "external_llm" or (llm_config and llm_config.get("api_key")):
            planner: BasePlanner = ExternalLLMPlanner(
                api_key=(llm_config or {}).get("api_key"),
                endpoint_url=(llm_config or {}).get("endpoint_url"),
                model=(llm_config or {}).get("model", "gpt-4o-mini")
            )
            engine_name = "External LLM Adapter"
        elif planner_mode == "heuristic":
            planner = AutonomousHeuristicPlanner()
            engine_name = "Autonomous Cognitive (Rule/Heuristic)"
        else:
            # Default to GGUF (Qwen3-4B on Metal GPU)
            custom_path = model_path or (llm_config or {}).get("model_path")
            planner = GGUFLocalLLMPlanner(model_path=custom_path)
            engine_name = "Qwen3-4B GGUF (Apple Silicon Metal GPU)"

        tools_info = self.registry.list_tools()
        trace: List[Dict[str, Any]] = []
        final_answer: Optional[str] = None
        status = "in_progress"

        # 3. ReAct Execution Loop
        for iteration in range(1, iterations_limit + 1):
            step_start = time.perf_counter()
            
            # Step planning
            step_plan = planner.plan_next_step(
                prompt=prompt,
                tools_info=tools_info,
                trace=trace,
                memory_vars=memory.get_all_vars()
            )

            thought = step_plan.get("thought", "Analyzing current state...")
            is_final = step_plan.get("is_final", False)

            if is_final:
                final_answer = step_plan.get("final_answer", "Objective completed.")
                status = "completed"
                break

            action = step_plan.get("action")
            action_input = step_plan.get("action_input") or {}

            # Tool Execution
            if action:
                tool_result = self.registry.execute(action, **action_input)
            else:
                tool_result = {"error": "No action specified by planner."}

            step_duration_ms = round((time.perf_counter() - step_start) * 1000, 2)

            trace_entry = {
                "step": iteration,
                "thought": thought,
                "action": action,
                "action_input": action_input,
                "observation": tool_result,
                "duration_ms": step_duration_ms
            }
            trace.append(trace_entry)

        # 4. Handle recursion / max iterations limit
        if status != "completed":
            status = "max_iterations_reached"
            final_answer = f"Agent reached the maximum iteration limit ({iterations_limit}). Completed {len(trace)} steps."

        total_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
        
        # Store in session memory history
        memory.add_history_entry(role="agent", content={
            "final_answer": final_answer,
            "steps_count": len(trace)
        })

        return {
            "status": status,
            "engine": engine_name,
            "prompt": prompt,
            "final_answer": final_answer,
            "total_steps": len(trace),
            "total_execution_time_ms": total_time_ms,
            "session_id": memory.session_id,
            "trace": trace
        }

# Global singleton executor
agent_executor = AgenticExecutor()
