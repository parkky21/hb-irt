# hb-irt

Python package implementing the Bayesian ability estimation and Item Response Theory
(IRT) core of the Candidate Skill Assessment Model (spec v2.0). Source of truth for
all formulas/algorithms is `Candidate Skill Assessment Model Specification v2.pdf`.

## Scope

This package covers **only** the following, and anything directly required to support
them:

1. **Bayesian updating**
   - EAP / MAP ability estimation via Gauss-Hermite quadrature (spec §4.1-4.2, A.1)
   - Posterior variance, standard error, 95% credible intervals (spec §4.3)
   - Sequential Bayesian updating across test modules — posterior of module *t-1*
     becomes the prior for module *t* (spec §4.4)

2. **IRT**
   - **2.1 MCQ** — Three-Parameter Logistic (3PL) model: probability of correct
     response, Fisher item/test information, MMLE/EM item calibration (spec §3)
   - **2.2 QA (graded, non-binary answers)**
     - Score 0-10 (11 ordinal categories) → Samejima Graded Response Model (GRM)
     - Score 0-100 (continuous) → Continuous Response Model (CRM)
     - Not defined in the spec (which only covers binary MCQ items); added to satisfy
       "IRT for QA where answer is a 0-100/0-10 score." Follows the same latent-trait
       (θ) and information-function conventions as the 3PL model so it composes with
       the rest of the package.

3. **Everything else in the spec that these two topics require to function**, i.e.:
   - Bloom level (L1-L6) → IRT difficulty anchor mapping with shrinkage (spec §3.3)
   - Precision-weighted level aggregation into a sub-skill score, 0-100 rescaling,
     margin of error / confidence bands (spec §5.1-5.3)
   - MSAT module selection (information-maximizing / EIG, exposure control) and
     stopping rules (spec §2.2-2.5), since these are the mechanism that decides what
     responses feed the Bayesian updater next

**The package's output boundary is the sub-skill score** (`θ, variance, 0-100 score ±
margin`, spec Table 7). It does not compute anything above that.

## Out of scope

- **DAG-based hierarchical skill aggregation** (spec §6 in full): skill DAG structure,
  weighted sub-skill→skill contribution, DAG topological propagation, non-compensatory
  minimum-competency gates, skill-level scores/CIs.
- Anything that consumes sub-skill scores to produce skill-level scores.
- MSAT/CAT/Fixed-form comparison content (spec §2.6) — descriptive only, not modeled.
- Candidate-facing UI/presentation layer (spec §7.2).

If a change would require importing or computing a skill-level (DAG) score, it does
not belong in this package.

## Package layout

```
src/hb_irt/
  types.py          # Item, Response, Posterior, SubskillScore, TestModule dataclasses
  quadrature.py      # Gauss-Hermite nodes/weights, EAP integration helper
  models/
    base.py          # ItemModel interface: prob / loglik / info
    threepl.py        # 3PL MCQ model
    grm.py            # Graded Response Model (0-10 ordinal QA)
    crm.py             # Continuous Response Model (0-100 QA)
  bayes/
    estimation.py      # EAP, MAP, posterior variance, credible interval
    sequential.py       # sequential Bayesian update across modules
  information.py     # Fisher item/test information, SEM
  calibration.py     # MMLE/EM item parameter calibration
  bloom.py           # Bloom level -> difficulty anchor mapping
  scoring.py          # 0-100 rescale, margin of error, level aggregation
  msat/
    module_bank.py     # TestModule container
    selection.py         # information-maximizing module selection
    stopping.py           # stopping rules
```

## Conventions

- Ability (θ) is represented on the logit scale internally; rescale to 0-100 only at
  the reporting boundary (`50 + 10*θ`, margin `19.6*sqrt(var)`, per spec §5.1 eq 11-13).
- All numerics use `numpy`/`scipy`; Gauss-Hermite quadrature uses 21-40 points per
  spec Appendix A.1.
- Every module ships with tests; **project-wide test coverage must stay at 90-100%**
  (enforced via `pytest-cov`, `--cov-fail-under=90` in `pyproject.toml`).

## Package manager

This project uses **uv**. Do not use bare `pip`/`venv`.

- Install/sync: `uv sync`
- Add a dependency: `uv add <package>`
- Add a dev dependency: `uv add --dev <package>`
- Run tests with coverage: `uv run pytest`
