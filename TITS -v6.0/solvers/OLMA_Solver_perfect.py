# -*- coding: utf-8 -*-
"""
solvers/OLMA_Solver_perfect.py

Paper-accurate OLMA solver (strict implementation of Algorithm 1 & 2 in TITS2)
- Constructor: OLMA_Solver(env_config:dict, cfg:dict)
- Usage: solver = OLMA_Solver(env_cfg, solver_cfg)
         out = solver.solve(system_state)

Notes:
- Units (defaults & compatibility with provided VEC_Environment):
    - Power p: Watts (W)
    - Bandwidth B: MHz (environment usually uses MHz)
    - Frequency F: Hz
    - Din in tasks: MB (environment uses MB)
- This implementation uses CVXPY for SP2. If CVXPY isn't installed, it falls back to a safe heuristic.
"""
import math
import numpy as np
from typing import Dict, Any, List
from scipy.optimize import linear_sum_assignment

# try cvxpy
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except Exception:
    CVXPY_AVAILABLE = False

# constants
EPS = 1e-12
BITS_PER_MB = 8e6  # MB -> bits
MIN_POWER = 1e-12
MIN_BW_MHZ = 1e-6
MIN_FREQ_HZ = 1e5


def _safe_log2(x: float) -> float:
    return math.log(x, 2) if x > 0 else 0.0


def _np_log2(x):
    # numpy-safe log2
    return np.log2(np.maximum(x, EPS))


class OLMA_Solver:
    def __init__(self, env_config: Dict[str, Any], cfg: Dict[str, Any]):
        """
        env_config: optional environment-level defaults (not required but accepted)
        cfg: solver config (V, bcd_max_iter, sca_max_iter, sca_eps, warm-start, etc.)
        """
        self.env_config = env_config or {}
        self.cfg = cfg or {}

        # Lyapunov parameter
        self.V = float(self.cfg.get("V", 50.0))

        # BCD / SCA iteration limits
        self.bcd_max_iter = int(self.cfg.get("bcd_max_iter", self.cfg.get("I_bcd", 6)))
        self.sca_max_iter = int(self.cfg.get("sca_max_iter", self.cfg.get("I_sca", 6)))
        self.sca_eps = float(self.cfg.get("sca_eps", self.cfg.get("epsilon", 1e-4)))

        # fallback heuristic toggle (if CVXPY missing)
        self.allow_heuristic = bool(self.cfg.get("allow_heuristic_fallback", True))

        # warm-start storage (X_bar)
        self.prev_p = None
        self.prev_B = None
        self.prev_F = None
        self.prev_gamma = None

    # -----------------------
    # Utility: partial derivatives of R(B,gamma) = B * log2(1+gamma)
    # -----------------------
    def _partial_deriv_R(self, B: float, gamma: float):
        # returns (dR/dB, dR/dgamma)
        B_safe = max(B, MIN_BW_MHZ * 1e6)  # B in Hz if used; careful
        gamma_safe = max(gamma, EPS)
        dR_dB = math.log2(1.0 + gamma_safe)
        dR_dgamma = B_safe / (math.log(2.0) * (1.0 + gamma_safe))
        return dR_dB, dR_dgamma

    # -----------------------
    # Eq.(4): interference computation exact (using assignment)
    # -----------------------
    def compute_interference(self, Nv: int, Jn: int, assignment: np.ndarray, p: np.ndarray, g: np.ndarray) -> np.ndarray:
        I = np.zeros(Jn, dtype=float)
        # each vehicle v assigned to a_v; contributes p[v]*g[v,m] to other servers m != a_v
        for v in range(Nv):
            a_v = int(assignment[v]) if v < len(assignment) else -1
            if a_v < 0:
                continue
            for m in range(Jn):
                if m != a_v:
                    I[m] += float(p[v]) * float(g[v, m])
        return I

    # -----------------------
    # Eq.(5): achievable rate and gamma
    # B in MHz internally expected; convert to Hz when computing rate
    # -----------------------
    def compute_rate_matrix(self, Nv: int, Jn: int, p: np.ndarray, B_mhz: np.ndarray, g: np.ndarray, I: np.ndarray, sigma2: float):
        rate = np.zeros((Nv, Jn), dtype=float)
        gamma = np.zeros((Nv, Jn), dtype=float)
        for v in range(Nv):
            for j in range(Jn):
                denom = float(I[j]) + float(sigma2) + EPS
                gamma_val = (float(p[v]) * float(g[v, j])) / denom
                gamma_val = max(gamma_val, 0.0)
                gamma[v, j] = gamma_val
                Bj_hz = max(float(B_mhz[v, j]), MIN_BW_MHZ) * 1e6
                rate[v, j] = Bj_hz * math.log2(1.0 + max(gamma_val, EPS))
        return rate, gamma

    # -----------------------
    # Transmission energy (Eq.(13) style)
    # E_tx = p * (Din_bits / R)
    # -----------------------
    def compute_tx_energy(self, Din_bits: float, p_val: float, R_bps: float) -> float:
        if R_bps <= EPS:
            return 0.0
        return float(p_val) * (float(Din_bits) / (float(R_bps) + EPS))

    # -----------------------
    # Server energy (Eq.(14))
    # -----------------------
    def server_energy(self, kappa_j: float, f: float, Delta_t: float) -> float:
        return float(kappa_j) * (float(f) ** 2) * float(Delta_t)

    # -----------------------
    # Compute ω (Eq.(33)-style) using X_bar (use_p, use_B, use_F, use_gamma)
    # Returns Nv x Jn cost matrix for Hungarian.
    # This function is robust: uses X_bar values and reasonable infeasibility penalties.
    # -----------------------
    def compute_omega(self, Nv: int, Jn: int, tasks: List[Dict[str, Any]], assignment_prev: np.ndarray,
                      Qjk_func, use_p: np.ndarray, use_B: np.ndarray, use_F: np.ndarray, use_gamma: np.ndarray,
                      kappa_j: np.ndarray, phi: List[List[float]], Delta_t: float, wE: float, wC: float, wH: float,
                      cloud_index):
        omega = np.zeros((Nv, Jn), dtype=float)
        # Rbar from X_bar
        Rbar = np.zeros((Nv, Jn), dtype=float)
        for v in range(Nv):
            for j in range(Jn):
                Bj_hz = max(float(use_B[v, j]), MIN_BW_MHZ) * 1e6
                gammabar = max(float(use_gamma[v, j]), EPS)
                Rbar[v, j] = Bj_hz * math.log2(1.0 + gammabar)

        fbar = np.array(use_F, dtype=float)

        for v in range(Nv):
            Din_bits = float(tasks[v].get('Din', 0.0)) * BITS_PER_MB
            Cv = float(tasks[v].get('Cv', 0.0))
            kv = int(tasks[v].get('kv', 0))
            prev_j = int(assignment_prev[v]) if assignment_prev is not None and len(assignment_prev) > v else -1

            for j in range(Jn):
                Rvj_bar = max(Rbar[v, j], EPS)
                p_bar_v = float(use_p[v]) if use_p is not None and v < len(use_p) else 0.0
                # Etx based on X_bar
                Etx_bar = self.compute_tx_energy(Din_bits, p_bar_v, Rvj_bar)

                # server dynamic energy
                if j == cloud_index:
                    Esrv_bar = 0.0
                    cloud_pen = float(wC)
                else:
                    Esrv_bar = float(kappa_j[j]) * (float(fbar[v, j]) ** 2) * float(Delta_t)
                    cloud_pen = 0.0

                Cbar = float(wE) * (Etx_bar + Esrv_bar) + cloud_pen

                # queue arrival
                Q_jkv = float(Qjk_func(j, kv))
                term_arrival = Q_jkv * Cv

                # service reduction: phi_jk * sum_v f * Delta_t (using fbar)
                service_sum = 0.0
                fbar_vj = float(fbar[v, j])
                for kk in range(len(phi[j])):
                    Q_jk = float(Qjk_func(j, kk))
                    service_sum += Q_jk * float(phi[j][kk]) * fbar_vj * float(Delta_t)

                hv = 1 if (prev_j != -1 and prev_j != j) else 0

                omega[v, j] = float(self.V) * Cbar + term_arrival - service_sum + float(wH) * float(hv)

                # infeasible if Rbar insufficient to upload Din within Delta_t
                if Rvj_bar * float(Delta_t) + 1e-12 < Din_bits:
                    omega[v, j] += 1e9  # very large penalty

        return omega

    # -----------------------
    # SP1: Hungarian assignment wrapper
    # -----------------------
    def solve_assignment(self, omega: np.ndarray) -> np.ndarray:
        try:
            row, col = linear_sum_assignment(omega)
            Nv = omega.shape[0]
            assignment = np.full(Nv, -1, dtype=int)
            assignment[row] = col
        except Exception:
            # fallback: greedy min per row
            assignment = np.argmin(omega, axis=1)
        return assignment

    # -----------------------
    # SP2: SCA implemented with CVXPY (Algorithm 2) - robust strict version
    # Returns p_final (Nv,), B_final (Nv x Jn in MHz), F_full (Nv x Jn in Hz), gamma_final (Nv x Jn)
    # -----------------------
    def solve_SP2_SCA(self, Nv, Jn, assignment, tasks, params,
                             p_init, B_init, F_init, g, sigma2):
        """
        Strict-paper SP2 solved by SCA (convex subproblems via CVXPY),
        faithful to TITS2: implements (29a)-(29d), Eq.(33), Eq.(36) and Algorithm 2.

        Inputs:
          - Nv (int): number of vehicles
          - Jn (int): number of servers
          - assignment (array-like, len Nv): a(v) for each vehicle v (int)
          - tasks (list of dict): tasks[v] must contain 'Din' in MB and 'Cv' in cycles
          - params (dict): parameters: Delta_t, Pmax_v (len Nv), Bup_max (len Jn),
                           Fmax_j (len Jn), kappa_j (len Jn), wE (scalar),
                           sca_iters (opt), sca_tol (opt)
          - p_init (array-like length Nv): initial power guess (W)
          - B_init (Nv x Jn): initial bandwidth matrix (MHz)
          - F_init (Nv x Jn) or other: initial cpu assignment (Hz)
          - g (Nv x Jn): channel gains
          - sigma2 (float): noise power

        Returns:
          p_opt (Nv,), B_opt_full (Nv x Jn), F_opt_full (Nv x Jn), gamma_opt_full (Nv x Jn)
        """

        import numpy as np
        import math
        import cvxpy as cp

        # ----------------------------
        # Minimal numeric safeguards
        # ----------------------------
        EPS = 1e-12  # tiny to avoid divide-by-zero in formula evaluation
        MIN_B_MHZ = 1e-9  # minimal bandwidth (MHz) to avoid zero in logs
        MIN_RATE = 1e-12

        # ----------------------------
        # Read parameters (paper units)
        # ----------------------------
        Delta_t = float(params.get("Delta_t", 1.0))
        Pmax_v = np.asarray(params.get("Pmax_v", [1.0] * Nv), dtype=float)  # W
        Bmax_j = np.asarray(params.get("Bup_max", [10.0] * Jn), dtype=float)  # MHz
        Fmax_j = np.asarray(params.get("Fmax_j", [2e9] * Jn), dtype=float)  # Hz
        kappa_j = np.asarray(params.get("kappa_j", [1e-27] * Jn), dtype=float)  # CPU energy coeff
        wE = float(params.get("wE", params.get("w_E", 1.0)))  # energy weight

        sca_iters = int(params.get("sca_iters", getattr(self, "sca_max_iter", 8)))
        sca_tol = float(params.get("sca_tol", getattr(self, "sca_eps", 1e-4)))

        # ----------------------------
        # Validate/prepare inputs
        # ----------------------------
        assignment = np.asarray(assignment, dtype=int)
        if assignment.shape[0] != Nv:
            raise ValueError("assignment length must equal Nv")

        # ensure tasks list length >= Nv
        if len(tasks) < Nv:
            raise ValueError("tasks must contain at least Nv entries")

        # convert Din MB -> bits (paper uses bits)
        Din_bits = np.array([max(1e-12, float(tasks[v].get("Din", 0.0))) * 8e6 for v in range(Nv)], dtype=float)
        Cv_vec = np.array([float(tasks[v].get("Cv", 0.0)) for v in range(Nv)], dtype=float)  # may be unused here

        # sanitize initial iterates (Algorithm 2 requires starting x^(0))
        p_prev = np.nan_to_num(np.asarray(p_init, dtype=float).flatten(), nan=1e-6)
        # B_prev must be (Nv, Jn)
        B_prev = np.asarray(B_init, dtype=float)
        if B_prev.shape != (Nv, Jn):
            # if user gave different shape, create small feasible B_prev concentrated on assigned server
            B_prev = np.zeros((Nv, Jn), dtype=float)
            for v in range(Nv):
                a = int(assignment[v])
                if 0 <= a < Jn:
                    B_prev[v, a] = min(0.1 * float(Bmax_j[a]), float(Bmax_j[a]))

        # F_prev: per-vehicle-per-server matrix; if F_init is vector or other, map into matrix
        F_prev_full = np.zeros((Nv, Jn), dtype=float)
        F_init_arr = np.asarray(F_init, dtype=float)
        if F_init_arr.shape == (Nv, Jn):
            F_prev_full = F_init_arr.copy()
        else:
            # map provided per-vehicle scalar to assigned server
            flat = F_init_arr.flatten()
            for v in range(Nv):
                a = int(assignment[v])
                if 0 <= a < Jn:
                    F_prev_full[v, a] = flat[v % flat.size] if flat.size > 0 else min(1e6, Fmax_j[a] * 0.01)

        # ----------------------------
        # Helper: compute interference I_m given p vector
        # paper: I_m = sum_{v: a(v) != m} p_v * g_{v,m}
        # ----------------------------
        def compute_interference(p_vec):
            I = np.zeros(Jn, dtype=float)
            for vv in range(Nv):
                av = int(assignment[vv])
                for m in range(Jn):
                    if m != av:
                        I[m] += float(p_vec[vv]) * float(g[vv, m])
            return I

        # Helper: compute true R and gamma (for re-evaluation; not used inside CVXPY)
        def compute_rate_and_gamma(p_vec, B_mat, sigma2_local):
            Ivec = compute_interference(p_vec)
            R = np.zeros((Nv, Jn), dtype=float)
            gamma = np.zeros((Nv, Jn), dtype=float)
            for v in range(Nv):
                for j in range(Jn):
                    Bj_MHz = max(MIN_B_MHZ, float(B_mat[v, j]))
                    Bj_Hz = Bj_MHz * 1e6
                    denom = Ivec[j] + sigma2_local + EPS
                    gamma_val = max((float(p_vec[v]) * float(g[v, j])) / denom, MIN_RATE)
                    gamma[v, j] = gamma_val
                    R[v, j] = Bj_Hz * math.log2(1.0 + gamma_val)
            return R, gamma

        # initial gamma_prev from p_prev, B_prev (used as linearization point)
        _, gamma_prev = compute_rate_and_gamma(p_prev, B_prev, sigma2)

        # ----------------------------
        # SCA outer loop (Algorithm 2 in paper)
        # each iteration we form convex subproblem via first-order Taylor of R = B*log2(1+gamma)
        # and linearization of 1/R for u surrogate
        # ----------------------------
        for s_it in range(max(1, sca_iters)):
            # compute interference fixed at previous iterate I^{s-1}
            I_prev = compute_interference(p_prev)

            # CVXPY variables for convex subproblem (per SCA iteration)
            p_var = cp.Variable(Nv, nonneg=True)  # p_v
            B_var = cp.Variable((Nv, Jn), nonneg=True)  # B_{v,j} (MHz)
            F_var = cp.Variable((Nv, Jn), nonneg=True)  # f_{v,j} (Hz)
            gamma_var = cp.Variable((Nv, Jn), nonneg=True)  # gamma_{v,j} (we will enforce zeros for j != a(v))
            u_var = cp.Variable(Nv, nonneg=True)  # convex surrogate for 1/R_v (upper bound)
            # No regularizer per strict paper; EPS used only in numeric checks outside CVX expressions if needed

            constraints = []

            # (29b) 0 <= p_v <= Pmax_v
            constraints += [p_var <= Pmax_v]

            # (29c) sum_v B_{v,j} <= Bmax_j  (per-server uplink bandwidth)
            for j in range(Jn):
                constraints += [cp.sum(B_var[:, j]) <= float(Bmax_j[j])]

            # (29d) sum_v F_{v,j} <= Fmax_j  (server CPU capacity)
            for j in range(Jn):
                constraints += [cp.sum(F_var[:, j]) <= float(Fmax_j[j])]

            # Enforce F_{v,j} == 0 for j != a(v) to match paper's mapping where v only uses a(v)
            for v in range(Nv):
                a_v = int(assignment[v])
                for j in range(Jn):
                    if j != a_v:
                        constraints += [F_var[v, j] == 0.0]

            # SINR linearization Eq.(33) (using fixed I_prev)
            # p_v * g_vj >= gamma_vj * (I_prev[j] + sigma2)
            Ivec = np.array(I_prev, dtype=float)
            for v in range(Nv):
                for j in range(Jn):
                    # For strict paper mapping it suffices to enforce gamma only for assigned server
                    # but here we keep gamma_var defined for all j and will set constraints accordingly
                    if j == int(assignment[v]):
                        constraints += [cp.multiply(p_var[v], float(g[v, j])) >= gamma_var[v, j] * (
                                    float(Ivec[j]) + float(sigma2) + EPS)]
                    else:
                        # enforce gamma zero at non-assigned servers (paper's focus on assigned link)
                        constraints += [gamma_var[v, j] == 0.0]

            # Rate linearization Eq.(36) and 1/R convex surrogate for each vehicle at its assigned server
            ln2 = math.log(2.0)
            for v in range(Nv):
                a_v = int(assignment[v])
                # use previous B_prev and gamma_prev at (v, a_v) as linearization point
                B0 = max(float(B_prev[v, a_v]), MIN_B_MHZ)
                gamma0 = max(float(gamma_prev[v, a_v]), MIN_RATE)
                # R0 in bits/s: B0(MHz->Hz) * log2(1 + gamma0)
                R0 = max(B0 * 1e6 * math.log2(1.0 + gamma0), MIN_RATE)

                # derivatives:
                dR_dB = 1e6 * math.log2(1.0 + gamma0)  # dR/d(B in MHz)
                dR_dgamma = (B0 * 1e6) / (ln2 * (1.0 + gamma0))  # dR/d(gamma)

                # affine linearization R_lin(B_var[v,a_v], gamma_var[v,a_v])
                R_lin = R0 + dR_dB * (B_var[v, a_v] - B0) + dR_dgamma * (gamma_var[v, a_v] - gamma0)

                # upload feasibility: R_lin * Delta_t >= Din_bits[v]   (linearized form of upload constraint)
                constraints += [R_lin * float(Delta_t) >= float(Din_bits[v])]

                # convex upper bound (linearized) on u = 1/R:
                u0 = 1.0 / (R0 + EPS)
                # first-order Taylor upper bound at R0: u >= u0 - (1/R0^2) * (R_lin - R0)
                constraints += [u_var[v] >= u0 - (1.0 / ((R0 + EPS) ** 2)) * (R_lin - R0)]
                # keep u_var bounded for numeric sanity (not a modeling change in paper; small safeguard)
                constraints += [u_var[v] <= 1e12, u_var[v] >= 0.0]

            # Objective (29a): minimize wE * (E_tx + E_srv)
            # E_tx approximated as sum_v p_v * u_v * Din_bits[v]
            Din_vec = Din_bits.astype(float)
            E_tx_expr = cp.sum(cp.multiply(p_var, cp.multiply(u_var, Din_vec)))

            # E_srv: sum_j kappa_j * (sum_v F_{v,j})^2 * Delta_t   (server-level quadratic)
            E_srv_terms = []
            for j in range(Jn):
                sum_fj = cp.sum(F_var[:, j])
                E_srv_terms.append(float(kappa_j[j]) * cp.square(sum_fj) * float(Delta_t))
            E_srv_expr = cp.sum(cp.hstack(E_srv_terms))

            objective = cp.Minimize(wE * (E_tx_expr + E_srv_expr))

            # Solve convex subproblem
            prob = cp.Problem(objective, constraints)
            # Use ECOS (recommended for SOCP / QP-ish problems). Per strict paper we don't add engineering relaxations.
            try:
                prob.solve(solver=cp.ECOS, verbose=False, abstol=1e-7, reltol=1e-7, feastol=1e-7)
            except Exception:
                # If ECOS throws, try SCS once (still no engineered fallbacks)
                prob.solve(solver=cp.SCS, verbose=False, eps=1e-5)

            # Check solver status: if not optimal/inaccurate, still extract values (user may decide how to treat)
            status = prob.status

            # Extract numerical solution (may be None if solver failed completely)
            p_sol = np.nan_to_num(np.asarray(p_var.value).flatten() if p_var.value is not None else p_prev, nan=1e-12)
            B_sol = np.nan_to_num(np.asarray(B_var.value) if B_var.value is not None else B_prev, nan=0.0)
            F_sol = np.nan_to_num(np.asarray(F_var.value) if F_var.value is not None else F_prev_full, nan=0.0)
            gamma_sol = np.nan_to_num(
                np.asarray(gamma_var.value) if gamma_var.value is not None else np.zeros((Nv, Jn)), nan=0.0)
            u_sol = np.nan_to_num(
                np.asarray(u_var.value).flatten() if u_var.value is not None else np.ones(Nv) * (1.0 / (MIN_RATE)),
                nan=1.0)

            # update iterates for next SCA round (Algorithm 2)
            # Note: in strict paper they compare x^{s} and x^{s-1} and stop when norm small.
            # Here we update and compute delta below.
            prev_concat = np.hstack([p_prev.flatten(), B_prev.flatten()])
            new_concat = np.hstack([p_sol.flatten(), B_sol.flatten()])

            p_prev = p_sol.copy()
            B_prev = B_sol.copy()
            F_prev_full = F_sol.copy()
            gamma_prev = gamma_sol.copy()

            # SCA convergence check (Frobenius-style): ||x^{new} - x^{old}||_2
            delta = np.linalg.norm(new_concat - prev_concat)
            if delta <= sca_tol:
                # reached tolerance; break SCA loop
                break
            # else continue to next SCA iter

        # End SCA loop

        # Final recompute true gamma (consistent with nonconvex definition) for returning
        _, gamma_final = compute_rate_and_gamma(p_prev, B_prev, sigma2)

        # Ensure F matrix shape (Nv, Jn)
        F_final = np.asarray(F_prev_full, dtype=float)
        if F_final.shape != (Nv, Jn):
            tmpF = np.zeros((Nv, Jn), dtype=float)
            # if per-vehicle scalar present in diagonal-like mapping, place at assigned server
            for v in range(Nv):
                a_v = int(assignment[v])
                if 0 <= a_v < Jn:
                    tmpF[v, a_v] = float(F_prev_full[v] if getattr(F_prev_full, "shape", ()) == (Nv,) else F_prev_full[
                        v, a_v] if F_prev_full.size > 0 else 0.0)
            F_final = tmpF

        # Return in the interface expected by rest of code
        return p_prev, B_prev, F_final, gamma_final

    # -----------------------
    # Main solve() - Algorithm 1 (BCD outer loop)
    # -----------------------
    def solve(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        # parse system_state
        V_set = system_state.get("V_set", [])
        J_set = system_state.get("J_set", [])
        Nv = max(1, len(V_set))
        Jn = max(1, len(J_set))
        cloud_index = system_state.get("cloud_index", None)
        params = system_state.get("params", {})

        # extract tasks and Qjk
        tasks = system_state.get("tasks", [{"Din": 0.0, "Cv": 0.0, "kv": 0} for _ in range(Nv)])
        Qjk_func = system_state.get("Qjk_func", lambda j, k: 0.0)

        # channel matrix
        g = np.array(system_state.get("g", np.ones((Nv, Jn))), dtype=float)
        if g.shape != (Nv, Jn):
            g = np.ones((Nv, Jn), dtype=float)

        # resource caps/params (ensure shapes)
        Pmax_v = np.array(params.get("Pmax_v", [1.0] * Nv), dtype=float)
        Bup_max = np.array(params.get("Bup_max", [10.0] * Jn), dtype=float)
        Fmax_j = np.array(params.get("Fmax_j", [2e9] * Jn), dtype=float)
        kappa_j = np.array(params.get("kappa_j", [1e-27] * Jn), dtype=float)

        wE = float(params.get("wE", 1.0))
        wC = float(params.get("wC", 0.0))
        wH = float(params.get("wH", 0.0))
        phi = params.get("phi", [[1.0] for _ in range(Jn)])
        Delta_t = float(params.get("Delta_t", 1.0))
        sigma2 = float(params.get("sigma2", 1e-9))

        # warm-start defaults
        if self.prev_p is None or len(self.prev_p) != Nv:
            self.prev_p = np.minimum(np.array(Pmax_v, dtype=float) * 0.5, np.array(Pmax_v, dtype=float))
            self.prev_B = np.ones((Nv, Jn), dtype=float) * (np.maximum(1.0, np.mean(Bup_max)) / max(1, Nv))
            # F in Hz matrix
            self.prev_F = np.ones((Nv, Jn), dtype=float) * np.maximum(MIN_FREQ_HZ, np.minimum(np.mean(Fmax_j) * 0.01, np.mean(Fmax_j)))
            I_init = self.compute_interference(Nv, Jn, np.argmax(g, axis=1), self.prev_p, g)
            _, self.prev_gamma = self.compute_rate_matrix(Nv, Jn, self.prev_p, self.prev_B, g, I_init, sigma2)

        prev_assignment = np.array(system_state.get("prev_assignment", np.argmax(g, axis=1)), dtype=int)

        # current iterate
        p_curr = self.prev_p.copy()
        B_curr = self.prev_B.copy()
        F_curr = self.prev_F.copy()
        gamma_curr = self.prev_gamma.copy()
        assignment = prev_assignment.copy()

        metrics = {"omega_iters": [], "p_iters": [], "B_iters": [], "F_iters": [], "gamma_iters": []}

        # BCD outer loop
        for iter_bcd in range(max(1, self.bcd_max_iter)):
            # 1) compute omega using X_bar = self.prev_*
            omega = self.compute_omega(Nv, Jn, tasks, prev_assignment, Qjk_func,
                                       use_p=self.prev_p, use_B=self.prev_B, use_F=self.prev_F,
                                       use_gamma=self.prev_gamma,
                                       kappa_j=kappa_j, phi=phi, Delta_t=Delta_t,
                                       wE=wE, wC=wC, wH=wH, cloud_index=cloud_index)

            metrics.setdefault("omega_iters", []).append(omega.copy())

            # 2) SP1: Hungarian assignment
            assignment_new = self.solve_assignment(omega)

            # 3) SP2: continuous resource allocation for assignment_new via SCA
            try:
                p_new, B_new, F_new, gamma_new = self.solve_SP2_SCA(Nv, Jn, assignment_new, tasks, params,
                                                                  p_curr, B_curr, F_curr, g, sigma2)
            except Exception as e:
                # fallback simple heuristic allocation to keep pipeline alive
                p_new = np.minimum(p_curr, Pmax_v)
                B_new = np.zeros((Nv, Jn), dtype=float)
                F_new = np.zeros((Nv, Jn), dtype=float)
                for v in range(Nv):
                    j = int(assignment_new[v])
                    if 0 <= j < Jn:
                        B_new[v, j] = max(MIN_BW_MHZ, float(Bup_max[j]) / max(1, Nv))
                        F_new[v, j] = max(MIN_FREQ_HZ, float(Fmax_j[j]) / max(1, Nv))
                I_final = self.compute_interference(Nv, Jn, assignment_new, p_new, g)
                _, gamma_new = self.compute_rate_matrix(Nv, Jn, p_new, B_new, g, I_final, sigma2)

            # 4) update warm-start X_bar <- X_new
            self.prev_p = p_new.copy()
            self.prev_B = B_new.copy()
            self.prev_F = F_new.copy()
            self.prev_gamma = gamma_new.copy()

            metrics.setdefault("p_iters", []).append(p_new.copy())
            metrics.setdefault("B_iters", []).append(B_new.copy())
            metrics.setdefault("F_iters", []).append(F_new.copy())
            metrics.setdefault("gamma_iters", []).append(gamma_new.copy())

            # check convergence: assignment stabilized and p converged
            if np.array_equal(assignment_new, assignment) and np.linalg.norm(p_new - p_curr) < 1e-6:
                assignment = assignment_new.copy()
                p_curr, B_curr, F_curr, gamma_curr = p_new.copy(), B_new.copy(), F_new.copy(), gamma_new.copy()
                break

            assignment = assignment_new.copy()
            p_curr, B_curr, F_curr, gamma_curr = p_new.copy(), B_new.copy(), F_new.copy(), gamma_new.copy()
            prev_assignment = assignment.copy()

        # End BCD: compute final metrics per paper
        I_final = self.compute_interference(Nv, Jn, assignment, p_curr, g)
        rate_final, gamma_final = self.compute_rate_matrix(Nv, Jn, p_curr, B_curr, g, I_final, sigma2)

        E_tx = np.zeros((Nv, Jn), dtype=float)
        E_srv = np.zeros((Nv, Jn), dtype=float)
        E_sys = 0.0
        for v in range(Nv):
            j = int(assignment[v])
            Din_bits = float(tasks[v].get('Din', 0.0)) * BITS_PER_MB
            Rvj = float(rate_final[v, j])
            # transmission energy: p * (Din_bits / R)
            if Rvj > EPS:
                E_tx[v, j] = float(p_curr[v]) * (Din_bits / (Rvj + EPS))
            else:
                E_tx[v, j] = 0.0
            Cv = float(tasks[v].get('Cv', 0.0))
            E_srv[v, j] = self.server_energy(float(kappa_j[j]), float(F_curr[v, j]), Delta_t)
            E_sys += E_tx[v, j] + E_srv[v, j]

        cloud_usage = 0.0
        if cloud_index is not None:
            for v in range(Nv):
                if int(assignment[v]) == int(cloud_index):
                    cloud_usage += float(tasks[v].get('Din', 0.0)) * float(params.get('wC', 0.0))

        handover = 0
        prev_assignment_safe = system_state.get("prev_assignment", np.argmax(g, axis=1))
        for v in range(Nv):
            prev_j = int(prev_assignment_safe[v])
            if prev_j != -1 and prev_j != int(assignment[v]):
                handover += 1

        wE = float(params.get('wE', 1.0))
        wH = float(params.get('wH', 0.0))
        C_sys = float(wE) * float(E_sys) + float(cloud_usage) + float(wH) * float(handover)

        # Lyapunov term J (Eq.26)
        Jval = self.compute_J(C_sys, Qjk_func, assignment, tasks, F_curr, phi, Delta_t)

        out = {
            "assignment": assignment.tolist(),
            "power": p_curr.tolist(),
            "bandwidth": B_curr.tolist(),
            "freq": F_curr.tolist(),
            "rate": rate_final.tolist(),
            "gamma": gamma_final.tolist(),
            "E_tx": E_tx.tolist(),
            "E_srv": E_srv.tolist(),
            "E_sys": float(E_sys),
            "C_sys": float(C_sys),
            "J": float(Jval),
            "handover_count": int(handover),
            "cloud_usage": float(cloud_usage),
            "interference": I_final.tolist(),
            "metrics": metrics
        }
        return out

    # -----------------------
    # Compute J(t) and Lyapunov term (Eq.26) as helper used by solve()
    # -----------------------
    def compute_J(self, C_sys: float, Qjk_func, assignment: np.ndarray, tasks: List[Dict[str, Any]],
                  F: np.ndarray, phi: List[List[float]], Delta_t: float) -> float:
        Nv = len(assignment)
        Jn = F.shape[1] if F is not None and F.ndim == 2 else 1
        # build A_jk
        Ajk = {}
        for v in range(Nv):
            j = int(assignment[v])
            kv = int(tasks[v].get('kv', 0))
            Ajk[(j, kv)] = Ajk.get((j, kv), 0.0) + float(tasks[v].get('Cv', 0.0))
        # build S_jk
        Sjk = {}
        for j in range(Jn):
            f_sum = float(np.sum(F[:, j]))
            for k in range(len(phi[j])):
                Sjk[(j, k)] = float(phi[j][k]) * f_sum * Delta_t
        Lyap = 0.0
        for (j, k), A_val in Ajk.items():
            Qval = Qjk_func(j, k)
            Sval = Sjk.get((j, k), 0.0)
            Lyap += float(Qval) * (float(A_val) - float(Sval))
        return float(self.V) * float(C_sys) + float(Lyap)

# End of file
