import os
from pathlib import Path

project_name = "AutoCatalogAI"
list_of_files = [
    f"{project_name}/app/app.py",
    f"{project_name}/app/inference.py",
    f"{project_name}/app/templates/index.html",

    f"{project_name}/autocatalog/__init__.py",

    f"{project_name}/autocatalog/data/__init__.py",
    f"{project_name}/autocatalog/data/dataset.py",
    f"{project_name}/autocatalog/data/preprocessing.py",

    f"{project_name}/autocatalog/models/__init__.py",
    f"{project_name}/autocatalog/models/multitask_clip.py",
    f"{project_name}/autocatalog/models/heads.py",

    f"{project_name}/autocatalog/training/__init__.py",
    f"{project_name}/autocatalog/training/train.py",
    f"{project_name}/autocatalog/training/losses.py",

    f"{project_name}/autocatalog/evaluation/__init__.py",
    f"{project_name}/autocatalog/evaluation/evaluate.py",
    f"{project_name}/autocatalog/evaluation/metrics.py",
    f"{project_name}/autocatalog/evaluation/error_analysis.py",

    f"{project_name}/autocatalog/inference/__init__.py",
    f"{project_name}/autocatalog/inference/predictor.py",
    f"{project_name}/autocatalog/inference/catalog_generator.py",

    f"{project_name}/autocatalog/utils/__init__.py",
    f"{project_name}/autocatalog/utils/config.py",
    f"{project_name}/autocatalog/utils/logger.py",
    f"{project_name}/autocatalog/utils/seed.py",

    f"{project_name}/configs/config.yaml",

    f"{project_name}/data/processed/train.csv",
    f"{project_name}/data/processed/val.csv",
    f"{project_name}/data/processed/test.csv",

    f"{project_name}/artifacts/models/.gitkeep",
    f"{project_name}/artifacts/evaluation/.gitkeep",
    f"{project_name}/artifacts/plots/.gitkeep",
    f"{project_name}/artifacts/examples/.gitkeep",

    f"{project_name}/notebooks/01_dataset_experiment.ipynb",

    f"{project_name}/scripts/prepare_dataset.py",
    f"{project_name}/scripts/train_baseline.py",
    f"{project_name}/scripts/train_multitask_clip.py",
    f"{project_name}/scripts/evaluate_model.py",
    f"{project_name}/scripts/predict_image.py",

    f"{project_name}/tests/test_dataset.py",
    f"{project_name}/tests/test_model.py",
    f"{project_name}/tests/test_inference.py",

    f"{project_name}/.gitignore",
    f"{project_name}/README.md",
    f"{project_name}/model_card.md",
    f"{project_name}/requirements.txt",
    f"{project_name}/Dockerfile",
    f"{project_name}/setup.py",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir:
        os.makedirs(filedir, exist_ok=True)

    if not filepath.exists():
        filepath.touch()
        print(f"Created: {filepath}")
    else:
        print(f"Already exists: {filepath}")