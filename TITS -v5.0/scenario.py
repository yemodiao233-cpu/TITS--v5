# scenario.py
import numpy as np
from typing import Dict, Any, List

class ScenarioLoader:
    """
    Minimal scenario generator ensuring each timeslot returns valid tasks.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg or {}
        self.num_vehicles = int(cfg.get("num_vehicles", 3))

    def new_tasks(self, Nv: int = None) -> List[Dict[str, Any]]:
        """
        ALWAYS return a valid list of tasks.
        Each task is a dict with keys: Din, Cv, kv.
        """
        if Nv is None:
            Nv = self.num_vehicles

        tasks = []
        for _ in range(Nv):
            tasks.append({
                "Din": float(max(0.1, np.random.rand() * 0.5)),
                "Cv": float(2e5),
                "kv": 0
            })
        return tasks