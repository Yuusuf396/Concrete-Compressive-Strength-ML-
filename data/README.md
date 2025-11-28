# Data directory

Raw data is intentionally excluded from git. To fetch the canonical UCI Concrete dataset and regenerate processed artifacts:

1. Create/activate your virtual environment.
2. Install dependencies via `pip install -r requirements.txt`.
3. Execute:

	```bash
	python -m src.data
	```

The script downloads the dataset from the UCI repository (via `ucimlrepo`), renames the columns, and saves artifacts to:

- `data/raw/concrete_data.csv` – raw download (gitignored)
- `data/processed/concrete_with_targets.parquet` – processed copy with `strength_class`
- `data/processed/summary_statistics.csv` – quick descriptive stats

### Folder conventions

- `raw/` – never checked in; contains source CSVs once downloaded.
- `interim/` – scratchpad space for notebooks/experiments.
- `processed/` – curated artifacts consumed by training scripts.
- `external/` – optional spot for third-party enrichments.

Each folder includes a `.gitkeep` so the structure is preserved even when empty.