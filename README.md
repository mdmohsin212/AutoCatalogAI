# AutoCatalogAI

AutoCatalogAI is a CLIP-based multi-task computer vision project for automatic fashion product cataloging. Given a product image, it predicts seven product attributes and generates catalog-ready metadata such as a suggested title, search tags, confidence scores, and JSON output.

The project includes reproducible training, two-stage fine-tuning, evaluation, consistency-aware predictions, latency benchmarking, Hugging Face model artifacts, and a Streamlit interface.

## Predicted Attributes

The model predicts:

1. Gender
2. Master category
3. Subcategory
4. Article type
5. Base colour
6. Season
7. Usage

## Model Architecture

AutoCatalogAI uses `openai/clip-vit-base-patch32` as the shared image encoder.

```text
Product Image
      |
      v
CLIP Image Encoder
      |
      v
Shared Image Embedding
      |
      +--> Gender Head
      +--> Master Category Head
      +--> Subcategory Head
      +--> Article Type Head
      +--> Base Colour Head
      +--> Season Head
      +--> Usage Head
```

The V2 model also includes:

- A lightweight 37-dimensional HSV/RGB colour-feature branch
- Hierarchical residual connections between related tasks
- Mild class balancing for selected imbalanced attributes
- Validation-based checkpoint selection
- Optional consistency correction during inference

## Dataset

```text
ashraq/fashion-product-images-small
```

| Split | Ratio |
|---|---:|
| Training | 70% |
| Validation | 15% |
| Test | 15% |

## Test Performance

Production V2 results on 6,611 held-out test images:

| Metric | Result |
|---|---:|
| Average Accuracy | 87.52% |
| Average Macro F1 | 67.44% |
| Average Weighted F1 | 87.10% |
| Average Top-3 Accuracy | 98.15% |
| Exact-Match Accuracy | 40.63% |
| Base Colour Accuracy | 69.72% |

Exact-match accuracy requires all seven attributes to be correct for the same image.

## Project Structure

```text
AutoCatalogAI/
├── app/
│   ├── app.py
│   └── inference.py
├── autocatalog/
│   ├── data/
│   │   ├── dataset.py
│   │   └── preprocessing.py
│   ├── models/
│   │   ├── heads.py
│   │   └── multitask_clip.py
│   ├── training/
│   │   ├── checkpoint.py
│   │   ├── losses.py
│   │   ├── pipeline.py
│   │   └── train.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── evaluate.py
│   │   └── error_analysis.py
│   ├── inference/
│   │   ├── predictor.py
│   │   └── catalog_generator.py
│   └── utils/
│       ├── config.py
│       ├── logger.py
│       └── seed.py
├── configs/
│   └── config.yaml
├── scripts/
│   ├── train_multitask_clip.py
│   ├── predict_image.py
│   └── upload_model.py
├── notebooks/
├── artifacts/
├── requirements.txt
└── README.md
```

## Installation

Create and activate a virtual environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

## Train the Model

Run the full training pipeline:

```bash
python scripts/train_multitask_clip.py --config configs/config.yaml
```

This command will:

1. Download the source checkpoint
2. Download and clean the dataset
3. Create train, validation, and test splits
4. Extract and cache colour features
5. Run two-stage fine-tuning
6. Select the best checkpoint using validation metrics
7. Evaluate raw and corrected predictions
8. Benchmark inference latency
9. Save model and evaluation artifacts

## Run the Streamlit App

```bash
streamlit run app/app.py
```

The interface supports product-image upload, seven attribute predictions, confidence scores, Top-K predictions, suggested product titles, search tags, JSON export, and runtime display.

## Generated Artifacts

```text
artifacts/models/autocatalogai_v2/
├── model.pt
├── config.json
├── label_maps.json
├── consistency_rules.json
├── metrics.json
├── history.json
└── README.md
```

```text
artifacts/evaluation/v2/
├── final_metrics.json
├── test_predictions.csv
├── classification reports
└── confusion matrices
```

## Model Repositories

Production model:

```text
mohsin416/autocatalogai-clip-multitask-v2
```

Experimental optimization checkpoint:

```text
mohsin416/fashion-attribute-lab
```

The production V2 model is used in the application because it provides a better balance between predictive performance, inference latency, deployment simplicity, and maintainability.

## Limitations

- Rare colour and usage classes remain difficult
- Some fashion colours are visually ambiguous
- Season and usage are not always directly visible from an image
- Performance may decrease on images outside the training distribution
- The `gender` label represents the dataset's product-target category, not the gender of a person

## Main Technologies

- Python
- PyTorch
- Hugging Face Transformers
- CLIP
- Hugging Face Datasets
- Scikit-learn
- Streamlit
- Pandas
- NumPy