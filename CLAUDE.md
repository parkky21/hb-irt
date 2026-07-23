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
    factory.py           # dispatch item data type -> its ItemModel
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

### Module map: what maps to what in the spec

| Module | Spec section | Purpose |
|---|---|---|
| `types.py` | Table 5, Table 7, eq 1 | Core data model: item params, responses, posteriors, modules, sub-skill scores |
| `quadrature.py` | Appendix A.1 | Gauss-Hermite nodes/weights and posterior-moment computation used by every Bayesian update |
| `models/threepl.py` | §3.1, eq 4; §A.2 | 3PL probability + Fisher information for MCQ items |
| `models/grm.py` | not in spec (see docstring) | Samejima GRM for 0-10 graded QA responses |
| `models/crm.py` | not in spec (see docstring) | Samejima continuous response model for 0-100 QA responses |
| `models/factory.py` | not in spec (support code) | `build_model(item) -> ItemModel` dispatch, so `TestModule`/MSAT selection aren't restricted to a single item type |
| `bayes/estimation.py` | §4.1-4.2, eq 7 | EAP (quadrature) and MAP (optimization) ability estimates |
| `bayes/sequential.py` | §4.4, eq 10 | Chains module posteriors: posterior_{t-1} becomes prior_t |
| `information.py` | eq 2, A.2 | Additive test information, SEM = 1/sqrt(I(theta)) |
| `calibration.py` | §3.2, eq 5 (3PL); not in spec (CRM/GRM, see docstrings) | MMLE via EM (Bock & Aitkin) to fit 3PL, CRM, and GRM item parameters from response data |
| `bloom.py` | §3.3, Table 6, eq 6 | Bloom level difficulty anchors + empirical-Bayes shrinkage |
| `scoring.py` | §5.1-5.3, eq 11-15 | 0-100 rescaling (`50 + 10*theta`), margin of error, per-level precision-weighted aggregation |
| `msat/module_bank.py` | §2.1-2.2, Table 1-2 | Module repository, target ability ranges by type |
| `msat/selection.py` | §2.3-2.4, eq 2-3 | Information-maximizing module selection with exposure control |
| `msat/stopping.py` | §2.5, Table 3 | Precision / max-modules / min-items / saturation stopping rules |

There is **no re-exporting package `__init__.py`** — import directly from the
submodule that defines what you need, e.g. `from hb_irt.models.threepl import
ThreePLModel`, not `from hb_irt import ThreePLModel`. This keeps import paths
traceable to a single file and avoids a growing "god module."

Tests mirror the source layout 1:1 under `tests/` (e.g.
`src/hb_irt/models/grm.py` <-> `tests/models/test_grm.py`).

## Conventions

- Ability (θ) is represented on the logit scale internally; rescale to 0-100 only at
  the reporting boundary (`50 + 10*θ`, margin `19.6*sqrt(var)`, per spec §5.1 eq 11-13).
  Never do this rescaling inline elsewhere — always call into `scoring.py`.
- All numerics use `numpy`/`scipy`; Gauss-Hermite quadrature uses 21-40 points per
  spec Appendix A.1 (`quadrature.DEFAULT_N_POINTS = 21`). Reuse
  `quadrature.quadrature_grid`/`quadrature.posterior_moments` for any new EAP-style
  integration rather than hand-rolling another Gauss-Hermite implementation.
- **Frozen dataclasses everywhere** for value types (`Item`, `Response`,
  `Posterior`, `TestModule`, `SubskillScore`, `GRMItem`, `CRMItem`). Don't make
  these mutable — callers should construct a new instance rather than mutate.
- **Validation happens in `__post_init__`.** Raise `ValueError` with a message
  that includes the invalid value, e.g. `f"discrimination a must be > 0, got
  {self.a}"`. Follow this pattern for new item/data types.
- **No global re-exports.** Each new module should be imported by its own path.
  Do not add symbols to `hb_irt/__init__.py` beyond the version string.
- **Docstrings cite the spec.** Any formula-bearing function's docstring
  should reference the section/equation number it implements (e.g. `(spec eq
  14-15)`), or explicitly say "not part of the spec" plus a citation for the
  model used (see `models/grm.py`, `models/crm.py` for the pattern), if the
  spec doesn't define it directly (as is the case for the QA graded/continuous
  models, which the spec doesn't cover — only binary MCQ items).
- **Keep the `ItemModel` interface minimal.** Every item model (3PL, GRM, CRM)
  implements only `loglik(value, theta)` and `info(theta)` — exactly what
  `bayes/estimation.py` and `msat/selection.py` need, nothing else. Don't widen
  the interface unless every existing model can implement the new method too.
- Every module ships with tests; **project-wide test coverage must stay at 90-100%**
  (enforced via `pytest-cov`, `--cov-fail-under=90` in `pyproject.toml`).

## Package manager

This project uses **uv**. Do not use bare `pip`/`venv`.

- Install/sync: `uv sync`
- Add a dependency: `uv add <package>`
- Add a dev dependency: `uv add --dev <package>`
- Run tests with coverage: `uv run pytest`

## Distribution

This package is published to PyPI as `hb-irt` (import name `hb_irt`), built via
the `uv_build` backend. `pyproject.toml`'s `[project].version` is the single
source of truth for the version; `hb_irt.__version__` reads it at runtime via
`importlib.metadata`. PyPI file uploads are immutable per version — bump the
version in `pyproject.toml` before every `uv build` + `uv publish`, even for a
docs-only change, since a previously published version's files can never be
overwritten or reused.

## Guidance for AI agents

If you are an agent extending this repository, read this section fully before
writing code.

### Ground rules

1. **Respect the scope boundary.** Anything that requires a *skill-level*
   score (DAG propagation, weighted sub-skill→skill contribution,
   non-compensatory gates, skill-level CIs — spec §6) is explicitly out of
   scope. If a task description asks you to compute or import anything
   downstream of a `SubskillScore`, stop and flag it rather than
   implementing it — it belongs in a different package.
2. **Every formula needs a citation.** Before implementing new math, find the
   corresponding section/equation in the spec PDF (or, if it's a QA-specific
   extension the spec doesn't cover, use a citable IRT technique — e.g.
   Samejima's GRM/CRM — and say so explicitly in the docstring, the way
   `models/grm.py` and `models/crm.py` do). Do not invent ad hoc formulas
   without documenting the reasoning.
3. **Maintain 90-100% test coverage.** `pyproject.toml` enforces
   `--cov-fail-under=90`; the project currently sits at 100%. Any new module
   needs tests covering: the happy path, at least one boundary/edge case
   (extreme theta, boundary parameter values like `c` near 0 or 1, category
   index 0 and the max category, empty/degenerate inputs), and every
   `raise ValueError`/`raise KeyError` branch. Run `uv run pytest` before
   declaring a task done — it will fail the whole suite if coverage drops
   below 90%.
4. **Don't reimplement the quadrature primitive.** If new code needs EAP-style
   integration over a normal posterior, call
   `quadrature.quadrature_grid`/`quadrature.posterior_moments`, don't hand-roll
   another Gauss-Hermite implementation.
5. **Keep the `ItemModel` interface minimal** (see Conventions above).

### A known pytest gotcha in this repo

Two names in this codebase intentionally collide with pytest's default
collection prefixes:

- `types.TestModule` is a dataclass, not a test class. It sets
  `__test__ = False` in its class body so pytest skips it. If you add another
  class whose *domain name* happens to start with `Test`, do the same.
- `information.test_information` is a function, not a test. If you import it
  into a test file, **alias the import** (see
  `tests/test_information.py`: `from hb_irt.information import test_information
  as calc_test_information`) — otherwise pytest tries to collect and call it
  as a zero-argument test function and errors out.

### Where to make changes

- Adding a new IRT response model → new file in `models/`, implementing
  `ItemModel`; mirror test file in `tests/models/`.
- Adding a new Bayesian estimation technique → `bayes/estimation.py` or a new
  file in `bayes/`, reusing `quadrature.py`.
- Adding a new stopping/selection rule → `msat/stopping.py` /
  `msat/selection.py`; extend `StoppingConfig`/`StoppingDecision` rather than
  inventing a parallel structure.
- Anything touching score presentation/rescaling → `scoring.py` only; don't
  inline `50 + 10*theta` elsewhere.
- Public-facing usage docs live in `README.md` and should stay spec-agnostic
  (no "spec §X" citations there — that document is internal). Keep spec
  citations and contribution conventions in this file instead.

### Before finishing a task

Run, in order:

```bash
uv run pytest              # full suite + coverage gate
```

If coverage fails, the terminal output (`term-missing`) tells you exactly
which lines are uncovered — add a test for that branch rather than lowering
the threshold. Do not lower `--cov-fail-under` in `pyproject.toml` to make a
task pass.
