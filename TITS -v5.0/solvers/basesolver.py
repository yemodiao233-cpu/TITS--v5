# solvers/basesolver.py
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseSolver(ABC):
    @abstractmethod
    def __init__(self, env_config: Dict[str, Any], cfg: Dict[str, Any]):
        self.env_config = env_config
        self.cfg = cfg

    @abstractmethod
    def solve(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Given system_state from environment, return decisions dict.
        Decisions format is flexible but environment.step must accept it.
        """
        pass