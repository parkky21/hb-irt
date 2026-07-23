<div align="center">

# hb-irt

**Bayesian ability estimation and Item Response Theory (IRT) for skill assessment**

Same math as the GRE, with uncertainty we actually show — and the score gets more trustworthy every time someone takes the test."

[![PyPI version](https://img.shields.io/pypi/v/hb-irt.svg?color=blue)](https://pypi.org/project/hb-irt/)
[![Python versions](https://img.shields.io/pypi/pyversions/hb-irt.svg)](https://pypi.org/project/hb-irt/)
[![License: MIT](https://img.shields.io/pypi/l/hb-irt.svg)](LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](CLAUDE.md)

</div>

`hb-irt` scores candidate responses — multiple-choice questions and open-ended
questions graded on a 0-10 or 0-100 scale — into a single latent ability
estimate with a calibrated uncertainty range, and supports adaptive,
multi-stage testing on top of it.

---

### Contents

- [Features](#features)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Concepts](#concepts)
- [Usage guide](#usage-guide)
- [Package layout](#package-layout)
- [Development](#development)
- [License](#license)

---

## Features

| | |
|---|---|
| **Item models** | 3PL (multiple-choice), GRM (0-10 graded), CRM (0-100 continuous) |
| **Ability estimation** | EAP (Gauss-Hermite quadrature) and MAP, with posterior variance and 95% credible intervals |
| **Sequential updating** | Each test stage's posterior becomes the next stage's prior — no re-scoring from scratch |
| **Item calibration** | Marginal Maximum Likelihood (MMLE/EM) fit from raw response data |
| **Score aggregation** | Precision-weighted combination of per-topic estimates into a 0-100 score ± margin of error |
| **Adaptive testing** | Information-maximizing module selection with exposure control, and configurable stopping rules |

Responses from any mix of item types combine into a single ability estimate:

| Response type | Model | `value` represents |
|---|---|---|
| Multiple-choice | **3PL** | `0` / `1` |
| Open-ended, scored 0-10 | **GRM** | integer category `0`–`10` |
| Open-ended, scored 0-100 | **CRM** | continuous score `0`–`100` |

## Installation

```bash
pip install hb-irt
```

or, with [uv](https://docs.astral.sh/uv/):

```bash
uv add hb-irt
```

> Requires Python 3.12+. Depends on `numpy` and `scipy`.

## Quickstart

```python
from hb_irt.types import Item
from hb_irt.models.threepl import ThreePLModel
from hb_irt.bayes.estimation import eap_estimate
from hb_irt.bayes.sequential import sequential_update
from hb_irt.scoring import build_subskill_score

# Define a small item bank (discrimination, difficulty, guessing)
items = [Item(item_id=f"q{i}", a=1.2, b=b, c=0.2)
         for i, b in enumerate([-1.0, -0.3, 0.4, 1.0, 1.6])]
models = [ThreePLModel(item) for item in items]

# Stage 1: estimate ability from a prior N(0, 1)
posterior = eap_estimate(list(zip(models, [1, 1, 0, 1, 0])), prior_mu=0.0, prior_sigma=1.0)

# Stage 2: the previous posterior becomes the new prior
posterior = sequential_update(posterior, list(zip(models, [1, 1, 1, 0, 1])))

# Rescale to a 0-100 score with a 95% margin of error
score = build_subskill_score(
    "python_debugging", posterior, items_administered=10, modules_completed=2
)
print(f"{score.score_0_100:.1f} ± {score.margin_error_95:.1f}")
```

## Concepts

- **Ability (θ)** is represented on a **logit scale** internally, typically in
  the range `[-4, 4]`. Use `hb_irt.scoring.rescale_0_100` to convert a
  posterior to a 0-100 score with a margin of error whenever you need to
  display it.
- **`Posterior(mu, variance)`** is the shared representation of a belief about
  a candidate's ability throughout the library. It exposes `.sem` (standard
  error) and `.credible_interval(level)`.
- **Item models** (3PL, GRM, CRM) share a minimal common interface:
  `loglik(value, theta)` and `info(theta)` — which is what lets responses of
  every type combine into a single ability estimate.

## Usage guide

<details>
<summary><strong>Multiple-choice items (3PL)</strong></summary>

```python
from hb_irt.types import Item
from hb_irt.models.threepl import ThreePLModel

item = ThreePLModel(Item(item_id="q1", a=1.2, b=0.3, c=0.2))
item.probability(theta=0.3)   # probability of a correct response at theta=0.3
item.loglik(value=1, theta=0.3)
item.info(theta=0.3)          # Fisher information at theta=0.3
```

</details>

<details>
<summary><strong>Open-ended answers scored 0-10 (Graded Response Model)</strong></summary>

```python
from hb_irt.models.grm import GRMItem, GRMModel

# 10 boundaries define 11 ordered categories (scores 0..10)
item = GRMModel(GRMItem(item_id="qa1", a=1.0, boundaries=(-2, -1, 0, 1, 2, 3, 4, 5, 6, 7)))
item.category_probabilities(theta=0.5)   # probability of each of the 11 scores
item.loglik(value=7, theta=0.5)          # value is the observed 0-10 score
item.info(theta=0.5)
```

</details>

<details>
<summary><strong>Open-ended answers scored 0-100 (Continuous Response Model)</strong></summary>

```python
from hb_irt.models.crm import CRMItem, CRMModel

item = CRMModel(CRMItem(item_id="qa2", a=1.0, b=0.0, max_score=100.0))
item.loglik(value=72.0, theta=0.4)
item.info(theta=0.4)
```

</details>

<details>
<summary><strong>Ability estimation (EAP / MAP) across mixed item types</strong></summary>

```python
from hb_irt.bayes.estimation import eap_estimate, map_estimate

# Any mix of item models works, since each just contributes a scalar
# loglik(value, theta) — MCQ, graded, and continuous responses combine freely.
responses = [(mcq_model, 1), (grm_model, 8), (crm_model, 80.0)]

posterior = eap_estimate(responses, prior_mu=0.0, prior_sigma=1.0)
# posterior.mu, posterior.variance, posterior.sem, posterior.credible_interval(0.95)

theta_map = map_estimate(responses, prior_mu=0.0, prior_sigma=1.0)
```

</details>

<details>
<summary><strong>Sequential updating across test stages</strong></summary>

```python
from hb_irt.types import Posterior
from hb_irt.bayes.sequential import sequential_update, sequential_update_all

prior = Posterior(mu=0.0, variance=1.0)
stage_1_posterior = sequential_update(prior, stage_1_responses)
stage_2_posterior = sequential_update(stage_1_posterior, stage_2_responses)

# or in one call, given an ordered list of each stage's responses:
history = sequential_update_all(prior, [stage_1_responses, stage_2_responses])
```

> Posterior variance is guaranteed to never increase across stages (assuming
> non-degenerate item information), so estimates only get more precise as a
> candidate answers more items.

</details>

<details>
<summary><strong>Item calibration (MMLE/EM)</strong></summary>

Fit 3PL item parameters from a batch of raw response data:

```python
import numpy as np
from hb_irt.calibration import calibrate_3pl

responses = np.array(...)  # shape (n_examinees, n_items), binary 0/1
result = calibrate_3pl(responses, item_ids=["q1", "q2", "q3"], fixed_c=0.2)
result.items          # tuple of Item, with fitted discrimination/difficulty (and fixed guessing)
result.converged      # bool
result.n_iterations   # int
```

> Pass `fixed_c=<value>` when calibrating with fewer than ~500 responses per
> item; otherwise omit it to freely estimate a guessing parameter per item.

</details>

<details>
<summary><strong>Difficulty mapping by cognitive level</strong></summary>

```python
from hb_irt.bloom import difficulty_anchor, shrink_difficulty

difficulty_anchor("L4")  # -> 1.2  (an "Analysis"-level item's typical difficulty)

# Pull a noisy raw calibration estimate toward its level's typical difficulty,
# weighted by how confident each estimate is.
shrunk_b = shrink_difficulty(raw_difficulty=1.9, raw_variance=0.3, level="L4", sigma_b=0.4)
```

</details>

<details>
<summary><strong>Score rescaling and topic aggregation</strong></summary>

```python
from hb_irt.scoring import aggregate_levels, rescale_0_100, build_subskill_score

# Combine estimates from several cognitive levels into one topic-level posterior
level_posterior = aggregate_levels(
    level_thetas={"L1": 0.4, "L2": 0.6, "L3": 0.5},
    level_variances={"L1": 0.05, "L2": 0.08, "L3": 0.06},
)

score, margin, ci_lower, ci_upper = rescale_0_100(level_posterior)

subskill_score = build_subskill_score(
    subskill_id="python_debugging",
    posterior=level_posterior,
    items_administered=42,
    modules_completed=4,
    level_thetas={"L1": 0.4, "L2": 0.6, "L3": 0.5},
)
# subskill_score.score_0_100, .margin_error_95, .ci_lower_95, .ci_upper_95, ...
```

</details>

<details>
<summary><strong>Adaptive testing (MSAT): module bank, selection, stopping</strong></summary>

```python
from hb_irt.types import Posterior
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

`ModuleBank.available(administered_ids)` returns modules not yet given to a
candidate. `select_next_module` picks the module that maximizes expected
information gain at the candidate's current ability estimate, with an
exposure-control bonus that favors less-used modules.

</details>

## Package layout

| Module | Provides |
|---|---|
| `hb_irt.types` | Core data types: `Item`, `Response`, `Posterior`, `TestModule`, `SubskillScore` |
| `hb_irt.models.threepl` | 3PL model for multiple-choice items |
| `hb_irt.models.grm` | Graded Response Model for 0-10 scored answers |
| `hb_irt.models.crm` | Continuous Response Model for 0-100 scored answers |
| `hb_irt.bayes.estimation` | EAP and MAP ability estimation |
| `hb_irt.bayes.sequential` | Sequential Bayesian updating across test stages |
| `hb_irt.information` | Fisher test information and standard error of measurement |
| `hb_irt.calibration` | MMLE/EM calibration of 3PL item parameters |
| `hb_irt.bloom` | Cognitive-level difficulty anchors and shrinkage |
| `hb_irt.scoring` | 0-100 rescaling and precision-weighted level aggregation |
| `hb_irt.msat` | Adaptive module bank, selection, and stopping rules |

Import directly from the submodule you need, e.g.
`from hb_irt.models.threepl import ThreePLModel`.

## Development

For contribution guidelines, architecture notes, and project conventions, see
[CLAUDE.md](CLAUDE.md).

```bash
uv sync
uv run pytest
```

## License

MIT — see [LICENSE](LICENSE).
