import argparse
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0,str(ROOT_DIR),)

from autocatalog.training.pipeline import run_training
from autocatalog.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train AutoCatalogAI V2"
        )
    )

    parser.add_argument(
        "--config",
        default="configs/config.yaml",
    )

    arguments = parser.parse_args()
    config = load_config(ROOT_DIR / arguments.config)
    run_training(config,ROOT_DIR)


if __name__ == "__main__":
    main()