# run_experiment.py
"""
Run experiments end-to-end without touching main.py.

Usage:
    python run_experiment.py --solver solvers.OLMA_Solver_perfect.OLMA_Solver --slots 100

Notes:
 - solver: Python path to solver class (module.ClassName)
 - slots: number of time slots to run
 - cfg: optional JSON config file with structure {"env": {...}, "solver": {...}, "out_dir": "logs"}
"""
import argparse
import json
import time
import os
from typing import Dict, Any

# imports rely on files we added earlier
from utils.solver_adapter import load_solver, timed_solve
from utils.metrics_logger import MetricsLogger
from solvers.environment import VEC_Environment

# optional pretty printer: if you have a pretty_print_metrics in utils.print_utils use it,
# else fallback to a simple printer below.
try:
    from utils.print_utils import pretty_print_metrics
    PRETTY = True
except Exception:
    PRETTY = False

def simple_print_metrics(solver_name: str, summary: Dict[str, Any]):
    print("\n" + "="*70)
    print(f"Results for Solver: {solver_name}")
    print("="*70)
    for k,v in summary.items():
        print(f"{k}: {v}")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--solver', type=str, required=True,
                        help='Solver class path e.g. solvers.OLMA_Solver_perfect.OLMA_Solver')
    parser.add_argument('--slots', type=int, default=100)
    parser.add_argument('--cfg', type=str, default=None,
                        help='Optional JSON config file (env/solver/out_dir)')
    args = parser.parse_args()

    cfg = {}
    if args.cfg:
        with open(args.cfg, 'r') as f:
            cfg = json.load(f)

    env_cfg = cfg.get('env', {})
    solver_cfg = cfg.get('solver', {})
    out_dir = cfg.get('out_dir', 'logs')
    os.makedirs(out_dir, exist_ok=True)

    # instantiate env and solver
    env = VEC_Environment(env_cfg)
    env.reset()

    solver_obj = load_solver(args.solver, env_cfg, solver_cfg)

    logger = MetricsLogger(out_dir=out_dir)

    # run loop
    for t in range(args.slots):
        state = env.get_state()
        t0 = time.time()
        decision = timed_solve(solver_obj, state)
        t1 = time.time()
        # environment step returns diagnostics
        diagnostics = env.step(decision, state)
        # optionally enrich diagnostics with timing / debug
        diagnostics['_slot_solve_time'] = decision.get('_solve_time_s', t1 - t0)
        # log all
        logger.log_slot(t, state, decision, diagnostics)

    summary = logger.summarize()
    csv_path = logger.save_csv("overall_slots.csv")
    if csv_path:
        print(f"Saved per-slot CSV to: {csv_path}")

    # print summary
    if PRETTY:
        pretty_print_metrics(args.solver, summary)
    else:
        simple_print_metrics(args.solver, summary)

    # Save summary json
    try:
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary.json to {out_dir}")
    except Exception:
        pass


if __name__ == "__main__":
    main()