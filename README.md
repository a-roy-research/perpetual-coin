# Perpetual Coin

Companion code for:

**"A Self-Funding Universal Basic Income: Existence Proof for a Stable Monetary Equilibrium"**
A. Roy (2026)

Paper: [arXiv link to be added] | [SSRN link to be added] 

Zenodo: https://zenodo.org/records/18830075

## What This Is

A six-rule monetary architecture in which money creation, destruction, government revenue, and universal income are unified into a single closed-form system. The paper derives the steady-state equilibrium, proves uniqueness and global stability, and calibrates to US scale.

## What the Code Does

`verify.py` reproduces every numerical result in the paper:

- Steady-state money supply, revenue, and lending pool (Table 2)
- Velocity and burn rate sensitivity (Tables 3, 5)
- Monte Carlo feasibility across 10,000 parameter draws (Table 4)
- Dynamic convergence from zero and from above (Table 6)
- Counter-cyclical velocity shock simulation (Table 6a)
- Endogenous velocity model (Table 5a)
- Supply chain burn cascade (Table E1)
- Tax comparison across income levels and household types

## Requirements

```
Python 3.8+
NumPy
Pandas
```

## Usage

```bash
python verify.py
```

Runs all verifications and prints results to stdout.

For a complete JSON export of all analyses:

```python
from verify import generate_full_report
report = generate_full_report()
```

## Parameters

| Symbol | Description | Base Value |
|--------|-------------|------------|
| c | Monthly claim per person | 2,000 |
| N | Population | 340,000,000 |
| τ | Topper rate | 0.15 |
| b | Burn rate | 0.10 |
| v | Velocity | 0.50 |

## Core Result

At base parameters: M = $29.5T, annual government revenue = $10.1T, lending pool = $12.4T (92% of current US), equilibrium error < 10⁻¹⁰.

Revenue is algebraically invariant to velocity (Proposition 2). The system is globally stable from any initial condition (Section 6).

## License

MIT
Creative Commons Attribution 4.0 International 

## Contact

a.roy.research@proton.me
