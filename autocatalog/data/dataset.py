import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from autocatalog.data.preprocessing import extract_color_features
from autocatalog.utils.logger import get_logger
from datasets import load_dataset
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
logger = get_logger(__name__)

def load_clean_dataset(dataset_name, tasks, label_maps):
    dataset = load_dataset(dataset_name, split="train")
    if "image" not in dataset.column_names:
        raise ValueError("Dataset must contain an image column")

    missing = [task for task in tasks if task not in dataset.column_names]
    if missing:
        raise ValueError(f"Missing task columns: {missing}")

    def valid(row):
        if row.get("image") is None:
            return False

        for task in tasks:
            value = str(row.get(task, "")).strip()
            if not value:
                return False
            if value not in label_maps[task]["label2id"]:
                return False

        return True

    clean_dataset = dataset.filter(valid)
    logger.info("Dataset loaded | raw=%d | clean=%d",len(dataset),len(clean_dataset),)
    return clean_dataset

def _stratify_labels(series):
    counts = series.value_counts()
    return series.apply(lambda value: value if counts[value] >= 2 else "__rare__")

def create_splits(dataset, tasks, output_dir, seed, train_ratio, validation_ratio, test_ratio,):
    total_ratio = train_ratio + validation_ratio + test_ratio
    if round(total_ratio, 6) != 1.0:
        raise ValueError("Split ratios must sum to 1.0")

    metadata = {
        task: [str(value).strip() for value in dataset[task]]
        for task in tasks
    }

    if "id" in dataset.column_names:
        metadata["id"] = dataset["id"]

    if "productDisplayName" in dataset.column_names:
        metadata["productDisplayName"] = dataset["productDisplayName"]

    dataframe = pd.DataFrame(metadata)
    dataframe["dataset_idx"] = np.arange(len(dataset))

    all_indices = dataframe.index.to_numpy()
    temporary_ratio = validation_ratio + test_ratio

    stratify_task = "articleType" if "articleType" in tasks else tasks[-1]
    try:
        train_idx, temporary_idx = train_test_split(
            all_indices,
            test_size=temporary_ratio,
            random_state=seed,
            stratify=_stratify_labels(dataframe[stratify_task]),
        )
    except ValueError:
        train_idx, temporary_idx = train_test_split(
            all_indices,
            test_size=temporary_ratio,
            random_state=seed,
        )

    temporary_dataframe = dataframe.loc[temporary_idx]
    test_share = test_ratio / temporary_ratio

    try:
        validation_idx, test_idx = train_test_split(
            temporary_idx,
            test_size=test_share,
            random_state=seed,
            stratify=_stratify_labels(temporary_dataframe[stratify_task]),
        )
    except ValueError:
        validation_idx, test_idx = train_test_split(
            temporary_idx,
            test_size=test_share,
            random_state=seed,
        )

    train_df = dataframe.loc[train_idx].copy()
    validation_df = dataframe.loc[validation_idx].copy()
    test_df = dataframe.loc[test_idx].copy()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_dir / "train.csv", index=False)
    validation_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    logger.info("Dataset split | train=%d | validation=%d | test=%d",len(train_df),len(validation_df),len(test_df),)
    return train_df, validation_df, test_df

def load_or_create_color_cache(dataset, cache_path, image_size, feature_dim):
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        features = np.load(cache_path)
        expected_shape = (len(dataset), feature_dim)

        if features.shape == expected_shape:
            logger.info("Color cache loaded | shape=%s", features.shape)
            return features

        logger.warning("Color cache shape mismatch; rebuilding")

    features = np.zeros((len(dataset), feature_dim), dtype=np.float32)
    for index in tqdm(range(len(dataset)), desc="Extracting color features"):
        features[index] = extract_color_features(
            dataset[index]["image"],
            image_size=image_size,
        )

    np.save(cache_path, features)
    logger.info("Color cache saved | path=%s", cache_path)
    return features


class FashionMultiTaskDataset(Dataset):
    def __init__(self, source_dataset, indices, color_features, processor, label_maps, tasks, training=False):
        self.source_dataset = source_dataset
        self.indices = list(map(int, indices))
        self.color_features = color_features
        self.processor = processor
        self.label_maps = label_maps
        self.tasks = tasks
        self.training = training

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        global_index = self.indices[index]
        item = self.source_dataset[global_index]
        image = item["image"]

        if not isinstance(image, Image.Image):
            image = Image.open(image)

        image = image.convert("RGB")
        if self.training and random.random() < 0.5:
            image = ImageOps.mirror(image)

        pixel_values = self.processor(
            images=image,
            return_tensors="pt",
        )["pixel_values"].squeeze(0)

        labels = {
            task: torch.tensor(
                self.label_maps[task]["label2id"][str(item[task]).strip()],
                dtype=torch.long,
            )
            for task in self.tasks
        }

        color_features = torch.tensor(
            self.color_features[global_index],
            dtype=torch.float32,
        )

        return {
            "pixel_values": pixel_values,
            "color_features": color_features,
            "labels": labels,
            "global_index": global_index,
        }


class MultiTaskCollator:
    def __init__(self, tasks):
        self.tasks = tasks

    def __call__(self, batch):
        return {
            "pixel_values": torch.stack(
                [item["pixel_values"] for item in batch]
            ),
            "color_features": torch.stack(
                [item["color_features"] for item in batch]
            ),
            "labels": {
                task: torch.stack([item["labels"][task] for item in batch])
                for task in self.tasks
            },
            "global_indices": [item["global_index"] for item in batch],
        }


def build_dataloaders( dataset, train_df, validation_df, test_df, color_features, processor, label_maps, tasks, batch_size, num_workers):
    collator = MultiTaskCollator(tasks)
    train_dataset = FashionMultiTaskDataset(
        dataset,
        train_df["dataset_idx"],
        color_features,
        processor,
        label_maps,
        tasks,
        training=True,
    )

    validation_dataset = FashionMultiTaskDataset(
        dataset,
        validation_df["dataset_idx"],
        color_features,
        processor,
        label_maps,
        tasks,
    )

    test_dataset = FashionMultiTaskDataset(
        dataset,
        test_df["dataset_idx"],
        color_features,
        processor,
        label_maps,
        tasks,
    )

    loader_arguments = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collator,
    }

    return {
        "train_dataset": train_dataset,
        "validation_dataset": validation_dataset,
        "test_dataset": test_dataset,
        "train_loader": DataLoader(
            train_dataset,
            shuffle=True,
            **loader_arguments,
        ),
        "validation_loader": DataLoader(
            validation_dataset,
            shuffle=False,
            **loader_arguments,
        ),
        "test_loader": DataLoader(
            test_dataset,
            shuffle=False,
            **loader_arguments,
        ),
    }