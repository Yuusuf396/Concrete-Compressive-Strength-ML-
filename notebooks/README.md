# Notebooks

Use this folder strictly for exploratory data analysis (EDA) or quick prototyping. All training/evaluation runs must be scripted via `src/` modules to keep experiments reproducible.

Suggested workflow:

1. Run `python -m src.data` to download + preprocess the dataset.
2. Create a new notebook that loads `data/processed/concrete_with_targets.parquet`.
3. Keep notebook outputs lightweight so the repo stays clean.
