# mlbag

A personal toolkit for DS/ML/DL/NLP projects. It exists so that project code can be
class/function/script-heavy from day one — reproducible seeding, run tracking,
checkpointing, hyperparameter search, and evaluation/plotting live here once, instead of
being copy-pasted and re-invented per notebook or per project.

Notebooks in projects that depend on `mlbag` should be a companion to the implementation
(demonstrating a proven result), not the place development happens. `mlbag` is what makes
that possible: the reusable logic lives in importable modules, not notebook cells.

## Install

From another project on the same machine, add `mlbag` as an editable path dependency:

```bash
uv add --editable /path/to/mlbag
```

Pick the extras your project needs — the base install is deliberately light (numpy,
pandas, matplotlib, pyyaml, pydantic, joblib) so you never pull in torch/tensorflow/optuna
unless you actually use them:

```bash
uv add --editable /path/to/mlbag --extra torch --extra bayesian
```

Available extras:

| Extra | Adds | Use for |
|---|---|---|
| `sklearn` | scikit-learn, scipy | classical ML pipelines, grid/random search |
| `torch` | torch | PyTorch models and training loops |
| `tensorflow` | tensorflow | Keras models and training loops |
| `nlp` | nltk, gensim, transformers | text preprocessing, embeddings, HF fine-tuning |
| `viz` | seaborn, umap-learn | extra plotting backends (embedding scatter) |
| `bayesian` | optuna | Bayesian hyperparameter search |
| `all` | everything above | quick local experimentation |

For local development on `mlbag` itself:

```bash
cd mlbag
uv sync --dev
uv run ruff check . && uv run mypy src/ && uv run pytest
```

## Modules

### `mlbag.seeding` — reproducibility

```python
from mlbag.seeding import seed_everything

seed_everything(42)  # seeds random, numpy, and (if installed) torch/tensorflow
```

### `mlbag.device` — device selection

```python
from mlbag.device import get_torch_device, device_summary

device = get_torch_device()   # cuda > mps > cpu, or pass prefer="cpu"
print(device_summary())       # logs what was detected, for run metadata
```

### `mlbag.config` — typed, YAML-backed config

Subclass `MLBagConfig` per project instead of scattering hyperparameters across globals
or notebook cells:

```python
from mlbag.config import MLBagConfig

class TrainConfig(MLBagConfig):
    learning_rate: float = 1e-3
    batch_size: int = 32

config = TrainConfig.from_yaml("configs/default.yaml")
config.to_yaml("artifacts/run_001/config.yaml")
```

### `mlbag.tracking.RunManager` — runs, checkpoints, metrics

```python
from mlbag.tracking import RunManager

run = RunManager(base_dir="artifacts", run_id="baseline")
run.save_config(config)
run.save_metrics({"epoch": 1, "val_loss": 0.42}, filename="metrics.jsonl", append=True)
run.save_checkpoint(model, name="model", epoch=1, optimizer=optimizer)
...
model = run.load_checkpoint(model, optimizer=optimizer)  # resumes from latest checkpoint
```

Checkpoint save/load dispatches automatically by object type — sklearn estimators go
through `joblib`, `torch.nn.Module` through `torch.save`/`load_state_dict`, and Keras
models through `model.save`/`load_model`. No `framework=` flag required.

### `mlbag.tuning` — grid, random, and Bayesian search behind one API

```python
from mlbag.tuning import optimize

result = optimize(
    estimator, param_space={"C": ("float", 0.01, 10, "log")},
    X=X_train, y=y_train, method="bayesian", n_trials=30, run=run,
)
result.best_params, result.best_score, result.trials  # trials is a DataFrame
```

`method` is one of `"grid"`, `"random"`, `"bayesian"` — same call shape regardless of
strategy, and results are persisted to the given `RunManager` when one is passed.

### `mlbag.evaluation` — classification/regression metrics + artifacts

```python
from mlbag.evaluation import evaluate_classifier

metrics = evaluate_classifier(y_test, y_pred, labels=class_names, run=run, split_name="test")
```

### `mlbag.plotting` — confusion matrices, training curves, decision regions, embeddings

Each function returns a `matplotlib.figure.Figure` so it composes with `RunManager.save_plot`
or your own subplot layout.

### `mlbag.data` — splitting, pipelines, text utilities

```python
from mlbag.data import stratified_split, make_model_pipeline, get_top_ngram

X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(X, y, val_size=0.1)
pipeline = make_model_pipeline(model, numeric_features=[...], categorical_features=[...])
top_bigrams = get_top_ngram(corpus, n=2, top_k=10)
```

### `mlbag.training` — generic supervised training loops

`mlbag.training.torch_loop.run_training` and `mlbag.training.keras_loop.run_training`
provide a matching API for a plain single-model supervised train/val loop: epoch looping,
per-epoch metric logging, checkpointing, and early stopping, wired to a `RunManager` when
given one. GAN-style multi-optimizer loops are out of scope — write those by hand and use
`RunManager` directly for checkpointing/metrics.

## Development

- `uv run pytest` runs the fast unit suite by default (`slow` and `live` tests are
  excluded — see `pyproject.toml`'s pytest markers).
- `uv run pytest -m slow` additionally runs framework-dependent integration smoke tests
  (requires the relevant extras installed).
- Keep `ruff check .`, `mypy src/`, and `pytest` green before every commit.

## License

MIT — see [LICENSE](LICENSE).
