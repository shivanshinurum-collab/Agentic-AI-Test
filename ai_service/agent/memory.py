import time
from typing import Dict, Any, List, Optional

class AgentMemory:
    """
    Session-aware memory manager for holding scratchpad traces,
    variables, and historical context during task execution.
    """
    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self.created_at = time.time()
        self.variables: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []

    def set_var(self, key: str, value: Any) -> None:
        self.variables[key] = value

    def get_var(self, key: str, default: Any = None) -> Any:
        return self.variables.get(key, default)

    def get_all_vars(self) -> Dict[str, Any]:
        return dict(self.variables)

    def add_history_entry(self, role: str, content: Any) -> None:
        self.history.append({
            "timestamp": time.time(),
            "role": role,
            "content": content
        })

    def get_history(self) -> List[Dict[str, Any]]:
        return list(self.history)

    def clear(self) -> None:
        self.variables.clear()
        self.history.clear()


class MemoryManager:
    """
    Multi-session memory repository.
    """
    _sessions: Dict[str, AgentMemory] = {}

    @classmethod
    def get_session(cls, session_id: Optional[str] = None) -> AgentMemory:
        sid = session_id or "default"
        if sid not in cls._sessions:
            cls._sessions[sid] = AgentMemory(session_id=sid)
        return cls._sessions[sid]

    @classmethod
    def clear_session(cls, session_id: str) -> bool:
        if session_id in cls._sessions:
            del cls._sessions[session_id]
            return True
        return False

    @classmethod
    def list_sessions(cls) -> List[str]:
        return list(cls._sessions.keys())
