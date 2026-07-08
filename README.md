# AutoCatalogAI

AutoCatalogAI is a CLIP-based multi-task computer vision system for automatic fashion product cataloging. Given a product image, it predicts seven product attributes and generates catalog-ready metadata such as a suggested title, search tags, confidence scores, and JSON output.

The project includes reproducible training, two-stage fine-tuning, multi-task evaluation, consistency-aware predictions, latency benchmarking, Hugging Face model artifacts, and a Streamlit interface.

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

AutoCatalogAI V2 also includes:

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

## Production Test Performance

Corrected V2 results on 6,611 held-out test images:

| Metric | Result |
|---|---:|
| Average Accuracy | 87.52% |
| Average Macro F1 | 67.44% |
| Average Weighted F1 | 87.10% |
| Average Top-3 Accuracy | 98.15% |
| Exact-Match Accuracy | 40.63% |
| Base Colour Accuracy | 69.72% |

Exact-match accuracy requires all seven attributes to be correct for the same image.

## Model Comparison

All models were evaluated on the same 6,611-image held-out test split using raw predictions and the same metric implementation.

| Model | Avg Accuracy | Avg Macro F1 | Top-3 Accuracy | Exact Match | Latency |
|---|---:|---:|---:|---:|---:|
| Majority Baseline | 42.63% | 7.93% | 73.44% | 0.95% | 0.001 ms |
| Frozen CLIP + Heads (V1) | 83.35% | 65.68% | 97.11% | 27.94% | 6.227 ms |
| **AutoCatalogAI V2** | **87.48%** | **67.41%** | **98.15%** | **40.46%** | **7.474 ms** |

The full reproducible comparison is available in:

```text
notebooks/03_autocatalogai_model_comparison.ipynb
artifacts/evaluation/model_comparison/
```

## Project Structure

```text
AutoCatalogAI/
├── src/
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
├── notebooks/
├── artifacts/
├── app.py
└── requirements.txt
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

The interface supports:

- Product-image upload
- Seven attribute predictions
- Confidence scores
- Top-K predictions
- Suggested product titles
- Search tags
- JSON export
- Inference runtime display

## Generated Artifacts

Model artifacts:

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

Evaluation artifacts:

```text
artifacts/evaluation/v2/
├── final_metrics.json
├── test_predictions.csv
├── classification reports
└── confusion matrices
```

Comparison artifacts:

```text
artifacts/evaluation/model_comparison/
├── model_comparison.csv
├── model_comparison.json
├── model_comparison_per_task.csv
├── benchmark_metadata.json
└── README_model_comparison.md
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

The production V2 model is used in the application because it provides a strong balance between predictive performance, inference latency, deployment simplicity, and maintainability.

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