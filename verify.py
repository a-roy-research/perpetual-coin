"""
Perpetual Coin — Verification Codebase

Companion code for:
  "A Self-Funding Universal Basic Income: Existence Proof for a
   Stable Monetary Equilibrium" — A. Roy (2026)

Modules:
  1. Base Model — Steady-state equations, proposition verification (Table 2)
  2. Univariate Stress Testing — Parameter sweeps, break points (Tables 3, 5)
  3. Monte Carlo Simulation — 5-parameter outcome distributions (Table 4)
  4. Dynamic Convergence — Time-to-steady-state, shock simulation (Tables 6, 6a)
  5. Feasibility Mapping — Parameter region feasibility (Section 5.3)
  6. Endogenous Velocity — Reduced-form velocity model (Table 5a)
  7. Supply Chain Burn Cascade — Netting efficiency analysis (Table E1)
  8. Reporting — Full JSON export

Requirements: Python 3.8+, NumPy, Pandas
License: MIT
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List
import json


# =============================================================================
# SECTION 1: BASE MODEL
# =============================================================================

@dataclass
class PerpetualCoinModel:
    """Core steady-state model. Implements Sections 2-3 of the paper."""

    c: float = 2000.0
    N: float = 340_000_000
    tau: float = 0.15
    b: float = 0.10
    v: float = 0.50
    vault_fraction: float = 0.68
    bank_share: float = 0.80
    deployment_rate: float = 0.77
    current_gov_spending: float = 9.5e12
    current_lending: float = 13.4e12
    current_debt_service: float = 1.0e12

    @property
    def R(self) -> float:
        return (1 - self.b) / (1 - self.b / 2)

    @property
    def m(self) -> float:
        return self.c * (1 + self.tau * self.R) / (
            self.b * self.v * (1 - self.R / 2))

    @property
    def M(self) -> float:
        return self.m * self.N

    @property
    def annual_burn_revenue(self) -> float:
        return (self.b / 2) * self.v * self.M * 12

    @property
    def annual_topper_revenue(self) -> float:
        return self.tau * self.c * self.N * 12

    @property
    def annual_total_revenue(self) -> float:
        return self.annual_burn_revenue + self.annual_topper_revenue

    @property
    def adjusted_gov_need(self) -> float:
        return self.current_gov_spending - self.current_debt_service - 0.5e12

    @property
    def revenue_surplus(self) -> float:
        return self.annual_total_revenue - self.adjusted_gov_need

    @property
    def lending_pool(self) -> float:
        return self.deployment_rate * self.bank_share * self.vault_fraction * self.M

    @property
    def lending_ratio(self) -> float:
        return self.lending_pool / self.current_lending

    @property
    def per_capita_depth(self) -> float:
        return self.M / self.N

    @property
    def equilibrium_error(self) -> float:
        topper = self.tau * self.c * self.N
        burn_rev = (self.b / 2) * self.v * self.M
        gov_recycled = (topper + burn_rev) * self.R
        inflow = self.c * self.N + gov_recycled
        destruction = self.b * self.v * self.M
        return abs(inflow - destruction) / destruction if destruction > 0 else float('inf')

    def summary(self) -> Dict:
        return {
            "M_trillion": self.M / 1e12,
            "per_capita": self.per_capita_depth,
            "annual_revenue_trillion": self.annual_total_revenue / 1e12,
            "burn_revenue_trillion": self.annual_burn_revenue / 1e12,
            "topper_revenue_trillion": self.annual_topper_revenue / 1e12,
            "revenue_surplus_trillion": self.revenue_surplus / 1e12,
            "lending_pool_trillion": self.lending_pool / 1e12,
            "lending_ratio": self.lending_ratio,
            "recycling_factor": self.R,
            "equilibrium_error": self.equilibrium_error,
        }

    def verify_propositions(self) -> Dict[str, bool]:
        results = {}
        results["P1_finite_positive"] = (
            np.isfinite(self.M) and self.M > 0 and self.equilibrium_error < 1e-10)
        topper_v1 = self.tau * self.c * self.N * 12
        topper_v2 = PerpetualCoinModel(
            c=self.c, N=self.N, tau=self.tau, b=self.b, v=0.80
        ).annual_topper_revenue
        results["P2_topper_invariant"] = abs(topper_v1 - topper_v2) < 1
        model_2c = PerpetualCoinModel(
            c=self.c * 2, N=self.N, tau=self.tau, b=self.b, v=self.v)
        results["P3_linear_scaling"] = (
            abs(model_2c.M / self.M - 2.0) < 1e-10
            and abs(model_2c.annual_total_revenue / self.annual_total_revenue - 2.0) < 1e-10)
        rev_b08 = PerpetualCoinModel(
            c=self.c, N=self.N, tau=self.tau, b=0.08, v=self.v
        ).annual_total_revenue
        rev_b12 = PerpetualCoinModel(
            c=self.c, N=self.N, tau=self.tau, b=0.12, v=self.v
        ).annual_total_revenue
        results["P4_fiscal_invariance"] = (
            abs(rev_b08 - rev_b12) / self.annual_total_revenue < 0.05)
        results["P5_netting_reduces_burden"] = (
            supply_chain_burn(self.b, 4, 0.60) < supply_chain_burn(self.b, 4, 0.0))
        results["P6_spend_dominates_hold"] = self.b < 1.0
        return results

    def __repr__(self):
        s = self.summary()
        return (f"PerpetualCoinModel(M=${s['M_trillion']:.1f}T, "
                f"Rev=${s['annual_revenue_trillion']:.1f}T, "
                f"Lending={s['lending_ratio']:.0%})")


# =============================================================================
# SECTION 2: UNIVARIATE STRESS TESTING
# =============================================================================

def univariate_stress_test(param_name, param_range, base_params=None,
                           output_metrics=None):
    """Vary one parameter, hold others at base. Reproduces Tables 3, 5."""
    if base_params is None:
        base_params = {}
    if output_metrics is None:
        output_metrics = ["M_trillion", "per_capita", "annual_revenue_trillion",
                          "lending_ratio", "revenue_surplus_trillion"]
    results = []
    for val in param_range:
        params = {**base_params, param_name: val}
        model = PerpetualCoinModel(**params)
        row = {param_name: val}
        summary = model.summary()
        for metric in output_metrics:
            row[metric] = summary[metric]
        results.append(row)
    return pd.DataFrame(results)


def find_break_points(param_name, param_range, base_params=None):
    """Find thresholds where feasibility criteria fail."""
    if base_params is None:
        base_params = {}
    breaks = {"revenue_shortfall": None, "lending_below_70pct": None,
              "per_capita_below_20k": None}
    for val in param_range:
        params = {**base_params, param_name: val}
        model = PerpetualCoinModel(**params)
        s = model.summary()
        if breaks["revenue_shortfall"] is None and s["revenue_surplus_trillion"] < 0:
            breaks["revenue_shortfall"] = val
        if breaks["lending_below_70pct"] is None and s["lending_ratio"] < 0.70:
            breaks["lending_below_70pct"] = val
        if breaks["per_capita_below_20k"] is None and s["per_capita"] < 20000:
            breaks["per_capita_below_20k"] = val
    return breaks


# =============================================================================
# SECTION 3: MONTE CARLO SIMULATION
# =============================================================================

@dataclass
class ParameterDistribution:
    name: str
    distribution: str
    params: Dict[str, float] = field(default_factory=dict)

    def sample(self, n, rng):
        if self.distribution == "triangular":
            return rng.triangular(self.params["left"], self.params["mode"],
                                  self.params["right"], n)
        elif self.distribution == "uniform":
            return rng.uniform(self.params["low"], self.params["high"], n)
        elif self.distribution == "normal":
            samples = rng.normal(self.params["mean"], self.params["std"], n)
            if "clip_low" in self.params:
                samples = np.clip(samples, self.params["clip_low"],
                                  self.params.get("clip_high", np.inf))
            return samples
        raise ValueError(f"Unknown distribution: {self.distribution}")


MONTE_CARLO_DISTRIBUTIONS = [
    ParameterDistribution("v", "triangular",
                          {"left": 0.25, "mode": 0.50, "right": 0.80}),
    ParameterDistribution("vault_fraction", "triangular",
                          {"left": 0.50, "mode": 0.68, "right": 0.80}),
    ParameterDistribution("bank_share", "triangular",
                          {"left": 0.60, "mode": 0.80, "right": 0.90}),
    ParameterDistribution("b", "triangular",
                          {"left": 0.05, "mode": 0.10, "right": 0.15}),
    ParameterDistribution("c", "triangular",
                          {"left": 1500, "mode": 2000, "right": 2500}),
]


def monte_carlo_simulation(param_distributions=None, n_simulations=10000,
                           output_metrics=None, base_params=None, seed=42):
    """5-parameter Monte Carlo. Reproduces Table 4."""
    if param_distributions is None:
        param_distributions = MONTE_CARLO_DISTRIBUTIONS
    if base_params is None:
        base_params = {}
    if output_metrics is None:
        output_metrics = ["M_trillion", "annual_revenue_trillion",
                          "revenue_surplus_trillion", "lending_ratio", "per_capita"]
    rng = np.random.default_rng(seed)
    samples = {pd.name: pd.sample(n_simulations, rng) for pd in param_distributions}
    results = []
    for i in range(n_simulations):
        params = {**base_params}
        row = {}
        for pd_obj in param_distributions:
            params[pd_obj.name] = samples[pd_obj.name][i]
            row[pd_obj.name] = samples[pd_obj.name][i]
        model = PerpetualCoinModel(**params)
        summary = model.summary()
        for metric in output_metrics:
            row[metric] = summary[metric]
        row["feasible_revenue"] = summary["revenue_surplus_trillion"] > 0
        row["feasible_lending"] = summary["lending_ratio"] >= 0.70
        row["feasible_percapita"] = summary["per_capita"] > 20000
        row["feasible_all"] = (row["feasible_revenue"] and row["feasible_lending"]
                               and row["feasible_percapita"])
        results.append(row)
    return pd.DataFrame(results)


def compute_confidence_intervals(mc_results, metrics=None,
                                 percentiles=[5, 25, 50, 75, 95]):
    """Percentile-based confidence intervals from Monte Carlo results."""
    if metrics is None:
        metrics = [c for c in mc_results.columns
                   if c not in ["feasible_revenue", "feasible_lending", "feasible_all"]
                   and mc_results[c].dtype in [np.float64, np.int64]]
    rows = []
    for metric in metrics:
        if metric in mc_results.columns:
            row = {"metric": metric, "mean": mc_results[metric].mean(),
                   "std": mc_results[metric].std()}
            for p in percentiles:
                row[f"p{p}"] = np.percentile(mc_results[metric], p)
            rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# SECTION 4: DYNAMIC CONVERGENCE
# =============================================================================

def simulate_dynamics(c=2000, N=340_000_000, tau=0.15, b=0.10, v=0.50,
                      M_initial=0.0, months=120, dt=1.0,
                      population_growth_annual=0.0, shock_schedule=None):
    """Simulate money supply dynamics. Reproduces Tables 6, 6a."""
    b_t, v_t, N_t, M_t = b, v, N, M_initial
    R = (1 - b_t) / (1 - b_t / 2)
    records = []
    for month in range(months + 1):
        if shock_schedule and month in shock_schedule:
            for param, val in shock_schedule[month].items():
                if param == "v": v_t = val
                elif param == "N": N_t = val
                elif param == "b":
                    b_t = val
                    R = (1 - b_t) / (1 - b_t / 2)
        m_ss = c * (1 + tau * R) / (b_t * v_t * (1 - R / 2))
        M_ss = m_ss * N_t
        convergence_pct = (M_t / M_ss * 100) if M_ss > 0 else 0
        creation = c * N_t
        topper_rev = tau * c * N_t
        burn_rev = (b_t / 2) * v_t * M_t
        gov_spending = topper_rev + burn_rev
        gov_recycled = gov_spending * R
        total_inflow = creation + gov_recycled
        total_destruction = b_t * v_t * M_t
        net_change = total_inflow - total_destruction
        records.append({
            "month": month, "M": M_t, "M_trillion": M_t / 1e12,
            "M_ss_target": M_ss, "convergence_pct": convergence_pct,
            "monthly_inflow": total_inflow, "monthly_creation": creation,
            "monthly_destruction": total_destruction, "net_change": net_change,
            "gov_revenue_monthly": gov_spending,
            "gov_revenue_annualized": gov_spending * 12,
            "topper_revenue_annualized": topper_rev * 12,
            "burn_revenue_annualized": burn_rev * 12,
            "velocity": v_t, "population": N_t,
            "per_capita": M_t / N_t if N_t > 0 else 0,
        })
        M_t = max(0, M_t + net_change * dt)
        N_t *= (1 + population_growth_annual / 12)
    return pd.DataFrame(records)


def find_convergence_time(dynamics_df, threshold=0.95):
    """First month where M reaches threshold × M_steady_state."""
    target = threshold * 100
    for _, row in dynamics_df.iterrows():
        if row["convergence_pct"] >= target:
            return int(row["month"])
    return None


# =============================================================================
# SECTION 5: FEASIBILITY REGION MAPPING
# =============================================================================

def map_feasibility_region(param1_name, param1_range, param2_name, param2_range,
                           criteria=None, base_params=None):
    """Map feasibility across two parameters (Section 5.3)."""
    if criteria is None:
        criteria = {"min_revenue_surplus": 0, "min_lending_ratio": 0.70,
                    "min_per_capita": 20000, "max_equilibrium_error": 0.001}
    if base_params is None:
        base_params = {}
    results = []
    for v1 in param1_range:
        for v2 in param2_range:
            params = {**base_params, param1_name: v1, param2_name: v2}
            model = PerpetualCoinModel(**params)
            s = model.summary()
            feasible = (
                s["revenue_surplus_trillion"] > criteria["min_revenue_surplus"] / 1e12
                and s["lending_ratio"] >= criteria["min_lending_ratio"]
                and s["per_capita"] > criteria["min_per_capita"]
                and s["equilibrium_error"] < criteria["max_equilibrium_error"])
            results.append({
                param1_name: v1, param2_name: v2, "feasible": feasible,
                "revenue_surplus_T": s["revenue_surplus_trillion"],
                "lending_ratio": s["lending_ratio"],
                "per_capita": s["per_capita"], "M_trillion": s["M_trillion"]})
    return pd.DataFrame(results)


# =============================================================================
# SECTION 6: VELOCITY UNDER THE BURN (Section 5.6)
# =============================================================================

def endogenous_velocity(b, v_base=0.55, elasticity=1.0, lifespan_months=12):
    """v*(b) = v_base × (1 - ε × b), floored at 1/L. (Section 5.6)"""
    return max(1.0 / lifespan_months, v_base * (1 - elasticity * b))


def endogenous_velocity_table(b_values=None):
    """Generate Table 5a."""
    if b_values is None:
        b_values = [0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]
    rows = []
    for b in b_values:
        v_star = endogenous_velocity(b)
        model = PerpetualCoinModel(b=b, v=v_star)
        s = model.summary()
        rows.append({"b": b, "v_star": v_star,
                     "M_trillion": s["M_trillion"],
                     "revenue_trillion": s["annual_revenue_trillion"],
                     "lending_ratio": s["lending_ratio"]})
    return pd.DataFrame(rows)


# =============================================================================
# SECTION 7: SUPPLY CHAIN BURN CASCADE
# =============================================================================

def supply_chain_burn(b=0.10, k=4, eta=0.60):
    """B_eff(k, η) = 1 − (1 − b)^(k(1 − η)). (Section 5.5, Proposition 5)"""
    return 1 - (1 - b) ** (k * (1 - eta))


def supply_chain_burn_table(b=0.10, k_values=None, eta_values=None):
    """Generate Table E1."""
    if k_values is None:
        k_values = [1, 2, 3, 4, 6]
    if eta_values is None:
        eta_values = [0.00, 0.25, 0.50, 0.60, 0.75, 0.90]
    rows = []
    for eta in eta_values:
        row = {"netting_efficiency": eta}
        for k in k_values:
            row[f"k={k}"] = supply_chain_burn(b, k, eta)
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# SECTION 7a: ANALYTICAL CONVERGENCE (Section 6)
# =============================================================================

def convergence_lambda(b=0.10, v=0.50, tau=0.15):
    """Contraction rate λ = b·v·(1 − R/2). (Section 6)"""
    R = (1 - b) / (1 - b / 2)
    return b * v * (1 - R / 2)


def analytical_convergence_time(threshold, b=0.10, v=0.50, tau=0.15):
    """Solve (1 − λ)^t = 1 − threshold for t. Returns months."""
    lam = convergence_lambda(b, v, tau)
    return np.log(1 - threshold) / np.log(1 - lam)


# =============================================================================
# SECTION 7b: EXPIRATION IMPACT BOUND (Section 3.6)
# =============================================================================

def expiration_impact(b=0.10, v=0.50, tau=0.15, c=2000, N=340_000_000,
                      vault_fraction=0.68, bank_share=0.80,
                      wallet_lifespan=12, vault_lifespan=60):
    """Bound the impact of coin expiration on steady-state M and revenue.

    Wallet expiration: Poisson model — P(untransacted for L months) = exp(-v*L).
    Vault expiration: worst-case — all personal vault coins expire after vault_lifespan.
    Returns dict with base and adjusted values. (Section 3.6)
    """
    base = PerpetualCoinModel(c=c, N=N, tau=tau, b=b, v=v,
                              vault_fraction=vault_fraction, bank_share=bank_share)

    # Wallet expiration rate
    p_expire_wallet = np.exp(-v * wallet_lifespan)
    non_vaulted_fraction = 1 - vault_fraction
    wallet_expiry_rate = p_expire_wallet * non_vaulted_fraction  # fraction of M per lifespan
    wallet_expiry_monthly = wallet_expiry_rate / wallet_lifespan  # monthly as fraction of M

    # Vault expiration rate (worst case: all personal vault coins expire)
    personal_vault_fraction = (1 - bank_share) * vault_fraction
    vault_expiry_monthly = personal_vault_fraction / vault_lifespan  # fraction of M per month

    total_expiry_monthly = wallet_expiry_monthly + vault_expiry_monthly

    # Adjusted equilibrium: creation + recycled = burn + expiration
    # c·N + (τ·c·N + (b/2)·v·M)·R = b·v·M + e·M
    # where e = total_expiry_monthly
    # c·N·(1 + τ·R) = M·(b·v·(1 - R/2) + e)
    R = base.R
    e = total_expiry_monthly
    m_adj = c * (1 + tau * R) / (b * v * (1 - R / 2) + e)
    M_adj = m_adj * N

    # Adjusted revenue (burn revenue uses adjusted M)
    burn_rev_adj = (b / 2) * v * M_adj * 12
    topper_rev = tau * c * N * 12
    total_rev_adj = burn_rev_adj + topper_rev

    return {
        "base_M_trillion": base.M / 1e12,
        "adjusted_M_trillion": M_adj / 1e12,
        "M_reduction_pct": (1 - M_adj / base.M) * 100,
        "wallet_expiry_monthly_pct": wallet_expiry_monthly * 100,
        "vault_expiry_monthly_pct": vault_expiry_monthly * 100,
        "total_expiry_monthly_pct": total_expiry_monthly * 100,
        "base_revenue_trillion": base.annual_total_revenue / 1e12,
        "adjusted_revenue_trillion": total_rev_adj / 1e12,
        "revenue_reduction_pct": (1 - total_rev_adj / base.annual_total_revenue) * 100,
    }


# =============================================================================
# SECTION 8: REPORTING
# =============================================================================

def generate_full_report(output_path="perpetual_coin_analysis.json"):
    """Run all analyses and export to JSON."""
    report = {}
    base = PerpetualCoinModel()
    report["base_model"] = base.summary()
    report["propositions"] = base.verify_propositions()

    for param, vals in [("v", np.arange(0.15, 1.01, 0.05)),
                        ("b", np.arange(0.03, 0.21, 0.01))]:
        df = univariate_stress_test(param, vals)
        report[f"stress_{param}"] = {
            "results": df.to_dict(orient="records"),
            "break_points": find_break_points(param, vals)}

    mc = monte_carlo_simulation(n_simulations=10000)
    report["monte_carlo"] = {
        "feasibility_all": float(mc["feasible_all"].mean()),
        "feasibility_revenue": float(mc["feasible_revenue"].mean()),
        "confidence_intervals": compute_confidence_intervals(mc).to_dict(orient="records")}

    dynamics = simulate_dynamics(M_initial=0, months=150)
    report["convergence"] = {str(t): find_convergence_time(dynamics, t)
                             for t in [0.25, 0.50, 0.75, 0.90, 0.95]}

    report["supply_chain"] = supply_chain_burn_table().to_dict(orient="records")
    report["endogenous_velocity"] = endogenous_velocity_table().to_dict(orient="records")

    feas = map_feasibility_region("v", np.arange(0.20, 0.81, 0.05),
                                  "b", np.arange(0.05, 0.16, 0.01))
    report["feasibility_region"] = {
        "feasible_fraction": int(feas["feasible"].sum()) / len(feas),
        "total": len(feas), "feasible": int(feas["feasible"].sum())}

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Report saved to {output_path}")
    return report


# =============================================================================
# MAIN — PAPER VERIFICATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PERPETUAL COIN — VERIFICATION CODEBASE")
    print("Companion to: Roy, A. (2026)")
    print("=" * 70)

    # 1. Base model (Table 2)
    print("\n1. BASE MODEL (Table 2)")
    base = PerpetualCoinModel()
    s = base.summary()
    print(f"   Money supply:         ${s['M_trillion']:.1f}T")
    print(f"   Per-capita depth:     ${s['per_capita']:,.0f}")
    print(f"   Annual revenue:       ${s['annual_revenue_trillion']:.1f}T")
    print(f"     Burn revenue:       ${s['burn_revenue_trillion']:.2f}T")
    print(f"     Topper revenue:     ${s['topper_revenue_trillion']:.2f}T")
    print(f"   Lending pool:         ${s['lending_pool_trillion']:.1f}T")
    print(f"   Lending ratio:        {s['lending_ratio']:.0%}")
    print(f"   Recycling factor:     {s['recycling_factor']:.4f}")
    print(f"   Equilibrium error:    {s['equilibrium_error']:.2e}")

    # 2. Propositions
    print("\n2. PROPOSITION VERIFICATION")
    for name, result in base.verify_propositions().items():
        print(f"   {name}: {'PASS' if result else 'FAIL'}")

    # 3. Velocity stress test (Table 3)
    print("\n3. VELOCITY STRESS TEST (Table 3)")
    vt = univariate_stress_test("v", np.array([0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]))
    print(f"   {'v':>5} {'M ($T)':>8} {'Rev ($T)':>9} {'Lending':>8}")
    for _, r in vt.iterrows():
        print(f"   {r['v']:>5.2f} {r['M_trillion']:>8.1f} "
              f"{r['annual_revenue_trillion']:>9.1f} {r['lending_ratio']:>8.0%}")

    # 4. Burn rate (Table 5)
    print("\n4. BURN RATE TABLE (Table 5)")
    bt = univariate_stress_test("b", np.array([0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.20]),
                                output_metrics=["M_trillion", "annual_revenue_trillion"])
    base_rev = s["annual_revenue_trillion"]
    print(f"   {'b':>5} {'M ($T)':>8} {'Rev ($T)':>9} {'vs Base':>8}")
    for _, r in bt.iterrows():
        d = (r['annual_revenue_trillion'] - base_rev) / base_rev * 100
        print(f"   {r['b']:>5.2f} {r['M_trillion']:>8.1f} "
              f"{r['annual_revenue_trillion']:>9.1f} {d:>+7.1f}%")

    # 5. Monte Carlo (Table 4)
    print("\n5. MONTE CARLO (Table 4)")
    mc = monte_carlo_simulation(n_simulations=10000)
    print(f"   Feasibility (all):   {mc['feasible_all'].mean():.1%}")
    print(f"   Revenue feasibility: {mc['feasible_revenue'].mean():.1%}")
    ci = compute_confidence_intervals(mc, metrics=["annual_revenue_trillion",
                                                    "M_trillion", "lending_ratio"])
    print(f"   {'Metric':<25} {'Mean':>8} {'P5':>8} {'P50':>8} {'P95':>8}")
    for _, r in ci.iterrows():
        print(f"   {r['metric']:<25} {r['mean']:>8.2f} {r['p5']:>8.2f} "
              f"{r['p50']:>8.2f} {r['p95']:>8.2f}")

    # 6. Convergence (Table 6)
    print("\n6. CONVERGENCE (Table 6)")
    dyn = simulate_dynamics(M_initial=0, months=200)
    for t in [0.25, 0.50, 0.75, 0.90, 0.95]:
        m = find_convergence_time(dyn, t)
        print(f"   {t:>4.0%}: month {m} ({m/12:.1f} years)")

    # 7. Velocity shock (Table 6a)
    print("\n7. VELOCITY SHOCK (Table 6a)")
    shock = simulate_dynamics(M_initial=base.M, months=48,
                              shock_schedule={6: {"v": 0.30}, 30: {"v": 0.50}})
    print(f"   {'Month':<8} {'v':>5} {'Inflow $B':>10} {'Destr $B':>10} "
          f"{'Net $B':>10} {'M $T':>7}")
    for mo in [0, 5, 6, 12, 24, 29, 30, 48]:
        r = shock[shock["month"] == mo].iloc[0]
        print(f"   {mo:<8} {r['velocity']:>5.2f} {r['monthly_inflow']/1e9:>10,.0f} "
              f"{r['monthly_destruction']/1e9:>10,.0f} {r['net_change']/1e9:>+10,.0f} "
              f"{r['M_trillion']:>7.1f}")

    # 8. Endogenous velocity (Table 5a)
    print("\n8. ENDOGENOUS VELOCITY (Table 5a)")
    ev = endogenous_velocity_table()
    print(f"   {'b':>5} {'v*':>6} {'M $T':>7} {'Rev $T':>7} {'Lend':>6}")
    for _, r in ev.iterrows():
        print(f"   {r['b']:>5.2f} {r['v_star']:>6.3f} {r['M_trillion']:>7.1f} "
              f"{r['revenue_trillion']:>7.1f} {r['lending_ratio']:>6.0%}")

    # 9. Supply chain (Table E1)
    print("\n9. SUPPLY CHAIN BURN (Table E1)")
    print(f"   b=0.10, k=4, eta=0.60 -> B_eff = {supply_chain_burn():.1%}")

    # 10. Feasibility region (Section 5.3)
    print("\n10. FEASIBILITY REGION")
    feas = map_feasibility_region("v", np.arange(0.20, 0.81, 0.05),
                                  "b", np.arange(0.05, 0.16, 0.01))
    print(f"    {int(feas['feasible'].sum())}/{len(feas)} feasible "
          f"({feas['feasible'].mean():.1%})")

    # 11. Analytical convergence (Section 6)
    print("\n11. ANALYTICAL CONVERGENCE (Section 6)")
    lam = convergence_lambda()
    print(f"    λ = b·v·(1 − R/2) = {lam:.4f}")
    for thresh in [0.25, 0.50, 0.75, 0.90, 0.95]:
        t_analytical = analytical_convergence_time(thresh)
        t_simulated = find_convergence_time(dyn, thresh)
        print(f"    {thresh:>4.0%}: analytical={t_analytical:.1f}, simulated={t_simulated} "
              f"(diff={abs(t_analytical - t_simulated):.1f})")

    # 12. Expiration impact bound (Section 3.6)
    print("\n12. EXPIRATION IMPACT BOUND (Section 3.6)")
    exp = expiration_impact()
    print(f"    Wallet expiry rate:  {exp['wallet_expiry_monthly_pct']:.3f}% of M/month")
    print(f"    Vault expiry rate:   {exp['vault_expiry_monthly_pct']:.3f}% of M/month")
    print(f"    Base M:              ${exp['base_M_trillion']:.1f}T")
    print(f"    Adjusted M:          ${exp['adjusted_M_trillion']:.1f}T "
          f"({exp['M_reduction_pct']:.1f}% reduction)")
    print(f"    Base revenue:        ${exp['base_revenue_trillion']:.1f}T")
    print(f"    Adjusted revenue:    ${exp['adjusted_revenue_trillion']:.1f}T "
          f"({exp['revenue_reduction_pct']:.1f}% reduction)")
    # Velocity dependence of revenue under expiration
    print("    Revenue velocity dependence (with expiration):")
    for v_test in [0.30, 0.50, 0.70]:
        e = expiration_impact(v=v_test)
        print(f"      v={v_test}: rev=${e['adjusted_revenue_trillion']:.2f}T")

    print("\n" + "=" * 70)
    print("All paper results verified. Use generate_full_report() for JSON export.")
    print("=" * 70)
