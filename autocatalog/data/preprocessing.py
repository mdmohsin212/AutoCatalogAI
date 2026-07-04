import numpy as np
from PIL import Image

COLOR_IMAGE_SIZE = 128
COLOR_FEATURE_DIM = 37

def extract_color_features(image, image_size=COLOR_IMAGE_SIZE):
    image = image.convert("RGB").resize((image_size, image_size))
    margin = int(image_size * 0.10)
    image = image.crop(
        (
            margin,
            margin,
            image_size - margin,
            image_size - margin,
        )
    )

    rgb = np.asarray(image, dtype=np.float32) / 255.0
    hsv = np.asarray(image.convert("HSV"), dtype=np.float32) / 255.0

    rgb_flat = rgb.reshape(-1, 3)
    hsv_flat = hsv.reshape(-1, 3)

    saturation = hsv_flat[:, 1]
    value = hsv_flat[:, 2]
    foreground_mask = (saturation > 0.08) | (value < 0.92)

    if foreground_mask.sum() < 256:
        foreground_mask = np.ones(len(hsv_flat), dtype=bool)

    selected_rgb = rgb_flat[foreground_mask]
    selected_hsv = hsv_flat[foreground_mask]

    hue_hist, _ = np.histogram(
        selected_hsv[:, 0],
        bins=12,
        range=(0.0, 1.0),
    )

    saturation_hist, _ = np.histogram(
        selected_hsv[:, 1],
        bins=8,
        range=(0.0, 1.0),
    )

    value_hist, _ = np.histogram(
        selected_hsv[:, 2],
        bins=8,
        range=(0.0, 1.0),
    )

    hue_hist = hue_hist.astype(np.float32)
    saturation_hist = saturation_hist.astype(np.float32)
    value_hist = value_hist.astype(np.float32)

    hue_hist /= max(hue_hist.sum(), 1.0)
    saturation_hist /= max(saturation_hist.sum(), 1.0)
    value_hist /= max(value_hist.sum(), 1.0)

    rgb_mean = selected_rgb.mean(axis=0).astype(np.float32)
    rgb_std = selected_rgb.std(axis=0).astype(np.float32)
    rgb_median = np.median(selected_rgb, axis=0).astype(np.float32)

    features = np.concatenate(
        [
            hue_hist,
            saturation_hist,
            value_hist,
            rgb_mean,
            rgb_std,
            rgb_median,
        ]
    ).astype(np.float32)

    if features.shape[0] != COLOR_FEATURE_DIM:
        raise ValueError(
            f"Expected {COLOR_FEATURE_DIM} color features, "
            f"got {features.shape[0]}"
        )

    return features