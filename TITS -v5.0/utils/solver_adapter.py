import importlib
from typing import Dict, Any, List
import time

# 默认安全决策（防崩溃）
def _default_decision(Nv: int, Jn: int):
    return {
        "assignment": [0] * Nv,
        "power": [0.0] * Nv,
        "bandwidth": [[0.0] * Jn for _ in range(Nv)],
        "freq": [[0.0] * Jn for _ in range(Nv)],
        "debug": {}
    }


def load_solver(class_path: str, env_cfg: Dict[str, Any], solver_cfg: Dict[str, Any]):
    """
    输入格式例：
        solvers.OLMA_Solver_perfect.OLMA_Solver
        solvers.MyNewSolver.MyNewSolver
    """
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls(env_cfg, solver_cfg)


def _normalize_list(x, length, default=0.0):
    try:
        x = list(x)
        if len(x) < length:
            x = x + [default] * (length - len(x))
        return x[:length]
    except:
        return [default] * length


def _normalize_matrix(mat, Nv, Jn, default=0.0):
    try:
        result = []
        for i in range(Nv):
            row = mat[i] if i < len(mat) else []
            row = _normalize_list(row, Jn, default)
            result.append(row)
        return result
    except:
        return [[default] * Jn for _ in range(Nv)]


def validate_and_normalize_decision(decision: Dict[str, Any], Nv: int, Jn: int):
    """
    修正 solver 输出的格式，保证：
    - assignment 长度 Nv
    - power 长度 Nv
    - bandwidth Nv×Jn
    - freq Nv×Jn
    """
    safe = _default_decision(Nv, Jn)

    if not isinstance(decision, dict):
        return safe

    assignment = decision.get("assignment", safe["assignment"])
    power = decision.get("power", safe["power"])
    bw = decision.get("bandwidth", safe["bandwidth"])
    fq = decision.get("freq", safe["freq"])

    assignment = _normalize_list(assignment, Nv, default=0)
    power = _normalize_list(power, Nv, default=0.0)
    bw = _normalize_matrix(bw, Nv, Jn, default=0.0)
    fq = _normalize_matrix(fq, Nv, Jn, default=0.0)

    return {
        "assignment": assignment,
        "power": power,
        "bandwidth": bw,
        "freq": fq,
        "debug": decision.get("debug", {}),
    }


def timed_solve(solver_obj, system_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    调 solver.solve() + 自动格式修复
    """
    Nv = len(system_state.get("V_set", []))
    Jn = len(system_state.get("J_set", []))

    t0 = time.time()
    try:
        out = solver_obj.solve(system_state)
    except Exception as e:
        out = {}
    t1 = time.time()

    norm = validate_and_normalize_decision(out, Nv, Jn)
    norm["_solve_time_s"] = t1 - t0
    return norm