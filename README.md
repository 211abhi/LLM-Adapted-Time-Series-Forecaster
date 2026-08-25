# LLM-Adapted Time Series Forecaster

Forecasting time series with a **completely frozen** language model. Nothing inside
GPT-2 is fine-tuned — the model is steered entirely from its input side by a learned
soft prompt, and read out by a small MLP head.

```
raw series ──► patch tokenizer ──► TS encoder ──┐
                                                ├──► [soft prompt ‖ ts embedding] ──► frozen GPT-2 ──► pred head ──► forecast
regime label ──► text template ──► frozen LLM ──┘        (CMA-ES, 500-dim subspace)      (no grads)      (2-layer MLP)
                        ▲
                 InfoNCE alignment
```

## How it works

| Phase | Script | What trains |
|---|---|---|
| 1 | `01_prepare_data.py` | — (feature engineering, indicators, rolling-minmax normalization, splits) |
| 1b | `02_label_regimes.py` | — (rule-based regime labels: `TRENDING_UP/DOWN`, `RANGING`, `VOLATILE`, `BREAKOUT`) |
| 2 | `03_train_encoder.py` | **TS encoder** — 1D-Conv + Transformer, contrastively aligned (InfoNCE) to frozen LLM text embeddings of each regime description |
| 3 | `04_optimize_soft_prompt.py` | **Soft prompt** — a `(32, 1024)` prefix found by CMA-ES in a 500-dim random subspace (BBTv2 trick). Gradient-free; the LLM stays frozen |
| 4 | `05_train_pred_head.py` | **Prediction head** — 2-layer MLP on the frozen encoder + frozen LLM + frozen soft prompt |
| 5 | `06_evaluate.py` | — (MAE / RMSE / directional accuracy, overall and per regime) |

The encoder is trained **once** and shared frozen across every dataset. Only the soft
prompt and prediction head are re-fit per dataset.

## Dataset

[ETT — Electricity Transformer Temperature](https://github.com/zhouhaoyi/ETDataset)
(`ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`), downloaded automatically in Phase 1. No manual
upload required.

ETT rows are point-in-time sensor readings rather than candles, so `src/utils/data_loader.py`
derives a synthetic OHLCV frame from them: `close = OT` (oil temperature), `open` = the
prior reading, `high`/`low` bound the pair, and `volume = HUFL` as a load-intensity proxy.
That lets the same indicator and regime-labeling stack run over any OHLCV-shaped series.

## Running it

The notebook `pipeline.ipynb` detects its environment and runs unchanged on all three:

| Environment | Setup |
|---|---|
| **Kaggle** | Settings → Accelerator → **GPU T4 x2**, **Internet On**. Code is cloned from GitHub |
| **Colab** | Runtime → Change runtime type → **GPU**. Code is cloned from GitHub |
| **Local** | Your own CUDA GPU. The notebook finds the repo you launched it from and never clones or pulls over your working tree |

Local:

```bash
git clone https://github.com/211abhi/LLM-Adapted-Time-Series-Forecaster.git
cd LLM-Adapted-Time-Series-Forecaster
# install torch for YOUR CUDA version first: https://pytorch.org/get-started/locally/
pip install -r requirements.txt
jupyter lab pipeline.ipynb
```

Or drive the scripts directly, once a config exists at `configs/base_config.yaml`
(copy `configs/base_config.example.yaml` and edit the paths):

```bash
python scripts/01_prepare_data.py       --config configs/base_config.yaml
python scripts/02_label_regimes.py      --config configs/base_config.yaml
python scripts/03_train_encoder.py      --config configs/base_config.yaml
python scripts/04_optimize_soft_prompt.py --config configs/base_config.yaml
python scripts/05_train_pred_head.py    --config configs/base_config.yaml
python scripts/06_evaluate.py           --config configs/base_config.yaml
```

Every phase takes `--force` to warm-start from an existing checkpoint instead of
refusing to overwrite it.

### Non-interactive runs

Kaggle's *Save & Run All* and `jupyter nbconvert --execute` have no stdin. The notebook's
prompts fall back to defaults automatically, and you can pin them up front:

```bash
export OUTPUT_STEPS=24     # forecast horizon
export FORCE_RETRAIN=y     # warm-start every phase from its existing checkpoint
```

### Multi-GPU

When more than one GPU is visible (e.g. Kaggle's T4 x2), training wraps the model in
`DataParallel` automatically. Nothing to configure.

## Secrets (optional)

Only needed to push trained checkpoints back to GitHub from inside the notebook — skip
those cells otherwise.

| | Kaggle | Colab | Local |
|---|---|---|---|
| `GITHUB_PAT` | Add-ons → Secrets | 🔑 sidebar → Secrets | env var, or a `.env` file |
| `HF_TOKEN` | Add-ons → Secrets | 🔑 sidebar → Secrets | env var, or a `.env` file |

`.env` is gitignored. Pushes go to a one-off tokenized URL rather than rewriting your
`origin`, so a local checkout never ends up with a PAT baked into its git config.

## Configuration

`configs/base_config.example.yaml` is the single source of truth for every hyperparameter
— no values are hardcoded in source files. The notebook regenerates a runtime
`configs/base_config.yaml` each session with the detected device and your chosen horizon,
which is why only the `.example` file is tracked.

Key knobs:

| | |
|---|---|
| `tokenizer.window_size` | 96 steps of context, split into 6 patches of 16 |
| `llm.model_name` | `gpt2-medium` (1024 hidden). Swap for any HF causal LM — match `hidden_dim` |
| `soft_prompt.n_tokens` | 32 prefix tokens |
| `cmaes.maxiter` | 100 generations × 20 candidates. **Drop to ~50 for a quick test run** |
| `prediction_head.output_steps` | Forecast horizon, set interactively by the notebook |

## Layout

```
src/
  tokenizer/    patch tokenizer + rule-based regime labeler
  encoder/      1D-Conv + Transformer TS encoder and its trainer
  alignment/    InfoNCE contrastive loss, text encoder, pair dataset
  llm/          frozen LLM wrapper
  soft_prompt/  soft prompt + CMA-ES optimizer
  prediction/   prediction head + end-to-end Forecaster
  utils/        data loading, metrics, checkpointing, regime templates
scripts/        the six numbered pipeline phases
configs/        base_config.example.yaml
checkpoints/    encoder (shared) + per-variant soft prompts and pred heads
pipeline.ipynb  Kaggle / Colab / local notebook
```

## Costs

Rough T4 timings: encoder ~20–40 min, CMA-ES soft prompt ~1–3 h (the bottleneck —
lower `cmaes.maxiter` to trade quality for time), prediction head ~5–15 min.
