# hb-irt

Bayesian ability estimation and Item Response Theory (IRT) for candidate skill
assessment. Implements the MCQ 3PL model, graded/continuous QA response models,
sequential Bayesian updating across test modules, and MSAT (Multi-Stage Adaptive
Testing) module selection and stopping rules.

This README is written for both **human developers** and **AI coding agents**
working in this repository. If you are an agent picking up a task here, read
[Guidance for AI agents](#guidance-for-ai-agents) before making changes.

Source of truth for the underlying math: `Candidate Skill Assessment Model
Specification v2.pdf`. Every non-obvious formula in this codebase cites a
section/equation number from that spec in its docstring.

---

## Scope

See [CLAUDE.md](CLAUDE.md) for the authoritative scope statement. Summary:

**In scope:** Bayesian updating (EAP/MAP, sequential updating across modules),
IRT for MCQs (3PL), IRT for QA scored 0-10 (GRM) or 0-100 (CRM), and everything
those require to function (Fisher information, Bloom-level difficulty mapping,
item calibration, sub-skill score rescaling, MSAT module selection/stopping).

**Out of scope:** DAG-based hierarchical skill aggregation (spec §6) and
anything downstream of it. This package's output boundary is the **sub-skill
score** — it never computes a skill-level (DAG-aggregated) score.

---

## Requirements

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) as the package manager — do not use bare `pip`/`venv` in this repo.

## Install

```bash
uv sync
```

This creates `.venv/` and installs `numpy`/`scipy` plus the `pytest`/`pytest-cov`/`coverage` dev group.

## Test

```bash
uv run pytest
```

- Coverage is enforced at **90%+** via `--cov-fail-under=90` in `pyproject.toml`
  (current actual coverage is 100%). A `term-missing` report prints uncovered
  lines directly in the terminal; `coverage.xml` is also written.
- Run a single file: `uv run pytest tests/models/test_threepl.py`
- Run with verbose test names: `uv run pytest -v`

## Add a dependency

```bash
uv add <package>          # runtime dependency
uv add --dev <package>    # dev-only dependency (linters, test tools, etc.)
```

---

## Publishing to PyPI

This package is set up to publish via `uv`, using the `uv_build` backend
(`[build-system]` in `pyproject.toml`). `hb-irt` is confirmed available (not
yet registered) on PyPI as of this writing.

### One-time setup

1. Create accounts on both [TestPyPI](https://test.pypi.org) and
   [PyPI](https://pypi.org) if you don't have them.
2. Generate an API token for each (Account Settings -> API tokens). Scope the
   token to the project once it exists there, or use an account-wide token for
   the very first upload.
3. Do not commit tokens. Export them as an environment variable per upload, or
   use `uv publish --token <token>` interactively (avoid putting the token in
   shell history — prefer the environment variable form below).

### Build

```bash
uv build
```

Produces `dist/hb_irt-<version>-py3-none-any.whl` and
`dist/hb_irt-<version>.tar.gz`. Inspect the sdist before publishing if you've
changed packaging config:

```bash
tar -tzf dist/hb_irt-*.tar.gz    # confirm only src/, LICENSE, README, pyproject.toml are included
```

### Publish to TestPyPI first (recommended)

Always do a dry run on TestPyPI before the real index — a version number can
never be reused on PyPI once published, even if deleted.

```bash
UV_PUBLISH_TOKEN=<test-pypi-token> uv publish --publish-url https://test.pypi.org/legacy/
```

Then verify installability from TestPyPI in a scratch venv:

```bash
uv venv /tmp/hb-irt-smoke && uv pip install --python /tmp/hb-irt-smoke/bin/python \
  --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ hb-irt
```

(`--extra-index-url` is needed because `numpy`/`scipy` aren't on TestPyPI.)

### Publish to PyPI

```bash
UV_PUBLISH_TOKEN=<pypi-token> uv publish
```

### Versioning

`pyproject.toml`'s `[project].version` is the **single source of truth**.
`hb_irt.__version__` reads it at runtime via `importlib.metadata.version("hb-irt")`
(see `src/hb_irt/__init__.py`) — there is nothing else to keep in sync.

Before each release:

1. Bump `version` in `pyproject.toml` (follow [SemVer](https://semver.org/):
   this is pre-1.0, so breaking changes can land in `0.x` minor bumps — bump
   the patch version for fixes, minor for additions/breaking changes while
   pre-1.0).
2. Run `uv run pytest` (100% coverage gate must pass).
3. `uv build`, then publish to TestPyPI, verify, then publish to PyPI.
4. Tag the release in git (`git tag v<version>`) once published — ask before
   pushing tags to a shared remote.

### Before the first publish, also consider

- Re-reading `Development Status :: 3 - Alpha` in the classifiers list in
  `pyproject.toml` and bumping it (e.g. to `4 - Beta` / `5 - Production/Stable`)
  as the package matures.

---

## Architecture

```
src/hb_irt/
  types.py            Item, Response, Posterior, TestModule, SubskillScore
  quadrature.py        Gauss-Hermite quadrature (EAP integration primitive)
  models/
    base.py             ItemModel interface: loglik(value, theta), info(theta)
    threepl.py           3PL model for MCQ items
    grm.py                Graded Response Model for 0-10 ordinal QA
    crm.py                 Continuous Response Model for 0-100 QA
  bayes/
    estimation.py         EAP / MAP ability estimation
    sequential.py          Sequential Bayesian updating across modules
  information.py       Fisher test information, standard error of measurement
  calibration.py       MMLE/EM calibration of 3PL item parameters
  bloom.py             Bloom level (L1-L6) -> difficulty anchor + shrinkage
  scoring.py           0-100 rescaling, precision-weighted level aggregation
  msat/
    module_bank.py       TestModule repository, queryable by type/history
    selection.py           Information-maximizing module selection (EIG)
    stopping.py             Stopping rules (precision / max modules / saturation)
```

There is **no re-exporting package `__init__.py`** — import directly from the
submodule that defines what you need, e.g. `from hb_irt.models.threepl import
ThreePLModel`, not `from hb_irt import ThreePLModel`. This keeps import paths
traceable to a single file and avoids a growing "god module."

Tests mirror the source layout 1:1 under `tests/` (e.g.
`src/hb_irt/models/grm.py` <-> `tests/models/test_grm.py`).

### Module map: what maps to what in the spec

| Module | Spec section | Purpose |
|---|---|---|
| `types.py` | Table 5, Table 7, eq 1 | Core data model: item params, responses, posteriors, modules, sub-skill scores |
| `quadrature.py` | Appendix A.1 | Gauss-Hermite nodes/weights and posterior-moment computation used by every Bayesian update |
| `models/threepl.py` | §3.1, eq 4; §A.2 | 3PL probability + Fisher information for MCQ items |
| `models/grm.py` | not in spec (see docstring) | Samejima GRM for 0-10 graded QA responses |
| `models/crm.py` | not in spec (see docstring) | Samejima continuous response model for 0-100 QA responses |
| `bayes/estimation.py` | §4.1-4.2, eq 7 | EAP (quadrature) and MAP (optimization) ability estimates |
| `bayes/sequential.py` | §4.4, eq 10 | Chains module posteriors: posterior_{t-1} becomes prior_t |
| `information.py` | eq 2, A.2 | Additive test information, SEM = 1/sqrt(I(theta)) |
| `calibration.py` | §3.2, eq 5 | MMLE via EM (Bock & Aitkin) to fit 3PL item parameters from response data |
| `bloom.py` | §3.3, Table 6, eq 6 | Bloom level difficulty anchors + empirical-Bayes shrinkage |
| `scoring.py` | §5.1-5.3, eq 11-15 | 0-100 rescaling (`50 + 10*theta`), margin of error, per-level precision-weighted aggregation |
| `msat/module_bank.py` | §2.1-2.2, Table 1-2 | Module repository, target ability ranges by type |
| `msat/selection.py` | §2.3-2.4, eq 2-3 | Information-maximizing module selection with exposure control |
| `msat/stopping.py` | §2.5, Table 3 | Precision / max-modules / min-items / saturation stopping rules |

---

## Core concepts

- **Ability scale**: candidate ability (theta, `θ`) is represented on the
  **logit scale** internally everywhere (typically in `[-4, 4]`). It is
  rescaled to a 0-100 "score" only at the reporting boundary, via
  `scoring.rescale_0_100` (`score = 50 + 10*theta`, `margin = 19.6 * SE`).
  Never do this rescaling inline elsewhere — always call into `scoring.py`.
- **Posterior**: `types.Posterior(mu, variance)` is the universal
  representation of a belief about theta. It exposes `.sem` (standard error)
  and `.credible_interval(level)`.
- **ItemModel interface**: every item model (3PL, GRM, CRM) implements two
  methods only — `loglik(value, theta)` and `info(theta)`. This is
  deliberately minimal: it's exactly what Bayesian estimation
  (`bayes/estimation.py`) and MSAT selection (`msat/selection.py`) need, and
  nothing else. `value` means different things per model (0/1 for 3PL, an
  integer category for GRM, a raw 0-100 score for CRM) — the interface doesn't
  care, since it never inspects `value` itself.
- **Gauss-Hermite quadrature**: `quadrature.quadrature_grid(mu, sigma,
  n_points)` returns `(theta_nodes, weights)` approximating `N(mu, sigma^2)`.
  `quadrature.posterior_moments(log_likelihood, theta, prior_weight)` combines
  a log-likelihood evaluated at those nodes with the prior weights to get a
  normalized posterior mean/variance. This pair of functions is the single
  numerical engine behind `eap_estimate`, `expected_information_gain`, and
  (indirectly) `calibrate_3pl`. If you need EAP-style integration anywhere
  else in this codebase, reuse these — don't reimplement quadrature.

---

## Usage examples

### MCQ item (3PL) — probability, log-likelihood, information

```python
from hb_irt.types import Item
from hb_irt.models.threepl import ThreePLModel

item = ThreePLModel(Item(item_id="q1", a=1.2, b=0.3, c=0.2))
item.probability(theta=0.3)   # -> 0.6  (== (1+c)/2 at theta == b, per Table 5)
item.loglik(value=1, theta=0.3)
item.info(theta=0.3)          # Fisher information at theta=0.3
```

### Graded QA response (0-10 scale) — GRM

```python
from hb_irt.models.grm import GRMItem, GRMModel

# 10 boundaries -> 11 ordered categories (raw scores 0..10)
item = GRMModel(GRMItem(item_id="qa1", a=1.0, boundaries=(-2, -1, 0, 1, 2, 3, 4, 5, 6, 7)))
item.category_probabilities(theta=0.5)   # -> array of 11 probabilities, sums to 1
item.loglik(value=7, theta=0.5)          # value is the observed 0-10 category
item.info(theta=0.5)
```

### Continuous QA response (0-100 scale) — CRM

```python
from hb_irt.models.crm import CRMItem, CRMModel

item = CRMModel(CRMItem(item_id="qa2", a=1.0, b=0.0, max_score=100.0))
item.loglik(value=72.0, theta=0.4)
item.info(theta=0.4)   # constant == a^2 for this model (see crm.py docstring)
```

### EAP / MAP ability estimation from mixed item types

```python
from hb_irt.bayes.estimation import eap_estimate, map_estimate

# Any mix of ThreePLModel / GRMModel / CRMModel instances works — each just
# contributes a scalar loglik(value, theta).
responses = [(mcq_model, 1), (grm_model, 8), (crm_model, 80.0)]

posterior = eap_estimate(responses, prior_mu=0.0, prior_sigma=1.0)
# posterior.mu, posterior.variance, posterior.sem, posterior.credible_interval(0.95)

theta_map = map_estimate(responses, prior_mu=0.0, prior_sigma=1.0)
```

### Sequential updating across test modules

```python
from hb_irt.types import Posterior
from hb_irt.bayes.sequential import sequential_update, sequential_update_all

prior = Posterior(mu=0.0, variance=1.0)
module_1_posterior = sequential_update(prior, module_1_responses)
module_2_posterior = sequential_update(module_1_posterior, module_2_responses)

# or in one call, given an ordered list of modules' responses:
history = sequential_update_all(prior, [module_1_responses, module_2_responses])
```

Posterior variance is guaranteed to be non-increasing across calls (assuming
non-degenerate item information) — this is asserted by
`tests/bayes/test_sequential.py`.

### Item calibration (MMLE/EM)

```python
import numpy as np
from hb_irt.calibration import calibrate_3pl

responses = np.array(...)  # shape (n_examinees, n_items), binary 0/1
result = calibrate_3pl(responses, item_ids=["q1", "q2", "q3"], fixed_c=0.2)
result.items          # tuple[Item, ...] with fitted a, b, (fixed) c
result.converged       # bool
result.n_iterations    # int
```

Use `fixed_c=<value>` when calibrating with fewer than ~500 responses per item
(spec §3.2); otherwise omit it to freely estimate guessing per item.

### Bloom level difficulty mapping + shrinkage

```python
from hb_irt.bloom import difficulty_anchor, shrink_difficulty

difficulty_anchor("L4")  # -> 1.2  (Analysis, Table 6)

# Pull a noisy raw calibration estimate toward its Bloom-level anchor,
# precision-weighted by how confident each estimate is (spec eq 6).
shrunk_b = shrink_difficulty(raw_difficulty=1.9, raw_variance=0.3, level="L4", sigma_b=0.4)
```

### Score rescaling and level aggregation

```python
from hb_irt.types import Posterior
from hb_irt.scoring import aggregate_levels, rescale_0_100, build_subskill_score

# Combine per-Bloom-level EAP estimates into one sub-skill posterior (eq 14-15)
level_posterior = aggregate_levels(
    level_thetas={"L1": 0.4, "L2": 0.6, "L3": 0.5},
    level_variances={"L1": 0.05, "L2": 0.08, "L3": 0.06},
)

score, margin, ci_lower, ci_upper = rescale_0_100(level_posterior)

subskill_score = build_subskill_score(
    subskill_id="sk_python_debugging",
    posterior=level_posterior,
    items_administered=42,
    modules_completed=4,
    level_thetas={"L1": 0.4, "L2": 0.6, "L3": 0.5},
)
# subskill_score.score_0_100, .margin_error_95, .ci_lower_95, .ci_upper_95, ...
```

### MSAT: module bank, selection, stopping

```python
from hb_irt.types import Posterior, TestModule
from hb_irt.msat.module_bank import ModuleBank
from hb_irt.msat.selection import select_next_module
from hb_irt.msat.stopping import StoppingConfig, evaluate_stopping

bank = ModuleBank(modules=(easy_module, medium_module, hard_module, challenge_module))

current_posterior = Posterior(mu=0.2, variance=0.6)
administered = ["easy_1"]

next_module = select_next_module(bank, current_posterior, administered_ids=administered)

decision = evaluate_stopping(
    posterior=current_posterior,
    previous_posterior=prior_posterior,   # or None on the first module
    n_modules=2,
    n_items=15,
    config=StoppingConfig(),  # sigma_min=0.3, max_modules=8, min_items=20, delta_saturation=0.01
)
if decision.should_stop:
    print("stopping:", decision.reasons)  # e.g. ("precision_threshold",)
```

`ModuleBank.available(administered_ids)` implements the spec's `B \ H` set
(available minus already-administered). `select_next_module` picks the
module maximizing `S(Q) = I_Q(mu_curr) + alpha * exp(-N_Q / beta)` (spec
Algorithm §2.3), where `N_Q` is `TestModule.n_exposures` (exposure control).

---

## Conventions

- **Frozen dataclasses everywhere** for value types (`Item`, `Response`,
  `Posterior`, `TestModule`, `SubskillScore`, `GRMItem`, `CRMItem`). Don't make
  these mutable — callers should construct a new instance rather than mutate.
- **Validation happens in `__post_init__`.** Raise `ValueError` with a message
  that includes the invalid value, e.g. `f"discrimination a must be > 0, got
  {self.a}"`. Follow this pattern for new item/data types.
- **21-40 point Gauss-Hermite quadrature** (spec Appendix A.1) for all EAP-style
  integration; `quadrature.DEFAULT_N_POINTS = 21` is the default everywhere
  and should stay the default unless a caller has a specific reason to change it.
- **No global re-exports.** Each new module should be imported by its own path.
  Do not add symbols to `hb_irt/__init__.py` beyond the version string.
- **Docstrings cite the spec.** Any formula-bearing function's docstring
  should reference the section/equation number it implements (e.g. `(spec eq
  14-15)`), or explicitly say "not part of the spec" plus a citation for the
  model used (see `models/grm.py`, `models/crm.py` for the pattern), if the
  spec doesn't define it directly (as is the case for the QA graded/continuous
  models, which the spec doesn't cover — only binary MCQ items).

---

## Guidance for AI agents

If you are an agent extending this repository, read this section fully before
writing code.

### Ground rules

1. **Respect the scope boundary.** Anything that requires a *skill-level*
   score (DAG propagation, weighted sub-skill→skill contribution,
   non-compensatory gates, skill-level CIs — spec §6) is explicitly out of
   scope. If a task description asks you to compute or import anything
   downstream of a `SubskillScore`, stop and flag it rather than
   implementing it — it belongs in a different package. See
   [CLAUDE.md](CLAUDE.md).
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
5. **Keep the `ItemModel` interface minimal.** New item/response models
   (should another response type ever be needed) must implement only
   `loglik(value, theta)` and `info(theta)` to compose with
   `bayes/estimation.py` and `msat/selection.py` unmodified. Don't widen the
   interface unless every existing model can implement the new method too.

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

### Before finishing a task

Run, in order:

```bash
uv run pytest              # full suite + coverage gate
```

If coverage fails, the terminal output (`term-missing`) tells you exactly
which lines are uncovered — add a test for that branch rather than lowering
the threshold. Do not lower `--cov-fail-under` in `pyproject.toml` to make a
task pass.
