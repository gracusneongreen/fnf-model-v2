# FNF Model V2

A small, FNF-focused language model prototype built **from random initialization** with PyTorch. This repository is dedicated to Friday Night Funkin' mod-development workflows.

> Status: Phase 1 prototype. The starter dataset is a tiny, original demonstration corpus for smoke-testing the training pipeline. It is not enough to create a capable FNF assistant yet.

## Hardware target

- Windows 11
- Intel Core i5
- 8 GB RAM
- Intel Iris Xe integrated graphics
- CPU training for the first prototype (the Iris Xe is not used as a CUDA GPU)

## Architecture

- Decoder-only causal Transformer
- UTF-8 byte tokenizer (256 byte values + BOS/EOS/PAD)
- Small default configuration: 128-dimensional embeddings, 4 attention heads, 4 layers, context length 128
- Random initialization; no pretrained LLM, pretrained weights, or external model checkpoint
- JSONL training data with one `{"text": "..."}` object per line

The byte tokenizer is intentionally simple and dependency-free. A learned subword tokenizer can be evaluated in a later phase.

## Requirements

Use Python 3.12 on Windows for the initial setup. Python 3.14 is not the target for this first PyTorch environment.

## Setup (PowerShell)

From the repository root:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
.\.venv\Scripts\Activate.ps1
```

If the setup script cannot find Python 3.12, install Python 3.12 with the Python Launcher enabled, then run it again.

## Run a short smoke-training session

This small configuration is intended to verify the complete pipeline on a modest CPU:

```powershell
python -m fnf_model.train --steps 20 --batch-size 2 --block-size 64 --d-model 64 --heads 4 --layers 2
```

The checkpoint is written to `checkpoints/fnf_model_v2.pt`.

For a longer experiment, try:

```powershell
python -m fnf_model.train --steps 500 --batch-size 4 --block-size 128 --d-model 128 --heads 4 --layers 4
```

Training time depends on the CPU. Start with the smoke test and monitor memory and temperature before increasing settings.

## Generate text

After a checkpoint has been created:

```powershell
python -m fnf_model.generate --prompt "FNF mod idea:" --tokens 120
```

The first output may be incoherent. A model trained from scratch needs a much larger, carefully prepared dataset and substantially more training than the included demo corpus.

## Tests

```powershell
python -m unittest discover -s tests -v
```

## Project structure

```text
fnf-model-v2/
├── data/
│   └── fnf_train.jsonl
├── fnf_model/
│   ├── config.py
│   ├── data.py
│   ├── generate.py
│   ├── model.py
│   ├── tokenizer.py
│   └── train.py
├── tests/
│   └── test_model.py
├── requirements.txt
└── setup.ps1
```

## Roadmap

1. Validate setup, unit tests, smoke training, and checkpoint loading.
2. Build a larger original FNF knowledge dataset with clear source and license tracking.
3. Add validation split, loss curves, resumable checkpoints, and dataset quality checks.
4. Add FNF-specific structured tasks for Psych Engine, charts, characters, stages, and mod manifests.
5. Evaluate generations against held-out FNF tasks before increasing model size.

## Important limitations

This is a research prototype, not yet a production-quality FNF mod generator. The starter data is intentionally small. Do not expect accurate Lua/Haxe code, complete mods, or reliable engine-specific answers until the dataset, evaluation, and training process have been expanded.

## Author

GracusNeonGreen

Repository: https://github.com/gracusneongreen/fnf-model-v2
