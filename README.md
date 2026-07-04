# hb-irt

Bayesian ability estimation and Item Response Theory (IRT) for candidate skill
assessment: 3PL for MCQs, Graded/Continuous Response Models for QA scored 0-10 or
0-100, sequential Bayesian updating across test modules, and MSAT module
selection/stopping rules.

See [CLAUDE.md](CLAUDE.md) for the full scope (what is and is not covered by this
package) and package layout.

## Install

```bash
uv sync
```

## Test

```bash
uv run pytest
```

Coverage is enforced at 90%+ (`--cov-fail-under=90`); a `term-missing` + `xml` report
is produced on every run.

## Usage

```python
from hb_irt.types import Item
from hb_irt.models.threepl import ThreePLModel
from hb_irt.bayes.estimation import eap_estimate
from hb_irt.bayes.sequential import sequential_update
from hb_irt.scoring import build_subskill_score

items = [Item(item_id=f"q{i}", a=1.2, b=b, c=0.2)
         for i, b in enumerate([-1.0, -0.3, 0.4, 1.0, 1.6])]
models = [ThreePLModel(item) for item in items]

# Module 1: EAP estimate from a prior N(0, 1)
posterior = eap_estimate(list(zip(models, [1, 1, 0, 1, 0])), prior_mu=0.0, prior_sigma=1.0)

# Module 2: previous posterior becomes the new prior (spec §4.4)
posterior = sequential_update(posterior, list(zip(models, [1, 1, 1, 0, 1])))

# Rescale to a 0-100 sub-skill score with a 95% margin of error (spec §5.1)
score = build_subskill_score(
    "sk_python_debugging", posterior, items_administered=10, modules_completed=2
)
print(score.score_0_100, "+/-", score.margin_error_95)
```
