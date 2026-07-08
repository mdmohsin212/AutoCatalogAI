import numpy as np
from PIL import Image

from autocatalog.data.preprocessing import (
    COLOR_FEATURE_DIM,
    extract_color_features,
)


def test_color_feature_shape_and_values():
    image = Image.new(
        "RGB",
        (128, 128),
        color=(255, 0, 0),
    )

    features = extract_color_features(image)

    assert features.shape == (COLOR_FEATURE_DIM,)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()