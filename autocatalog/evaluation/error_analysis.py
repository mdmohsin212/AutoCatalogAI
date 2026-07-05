import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def save_evaluation_artifacts(
    output_dir,
    tasks,
    label_maps,
    y_true,
    raw_predictions,
    corrected_predictions,
    probabilities,
    global_indices,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_variants = {
        "raw": raw_predictions,
        "corrected": corrected_predictions,
    }

    for variant, predictions in prediction_variants.items():
        for task in tasks:
            id2label = label_maps[task]["id2label"]
            label_ids = list(range(len(id2label)))

            label_names = [
                id2label[str(label_id)]
                for label_id in label_ids
            ]

            report = classification_report(
                y_true[task],
                predictions[task],
                labels=label_ids,
                target_names=label_names,
                zero_division=0,
                output_dict=True,
            )

            report_path = (output_dir / f"{variant}_{task}_classification_report.json")
            with open(report_path, "w", encoding="utf-8") as file:
                json.dump(
                    report,
                    file,
                    indent=2,
                    ensure_ascii=False,
                )

            matrix = confusion_matrix(
                y_true[task],
                predictions[task],
                labels=label_ids,
            )

            matrix_path = (output_dir / f"{variant}_{task}_confusion_matrix.csv")
            pd.DataFrame(
                matrix,
                index=label_names,
                columns=label_names,
            ).to_csv(
                matrix_path,
                encoding="utf-8",
            )

    rows = []
    for row_index, global_index in enumerate(global_indices):
        row = {"global_index": int(global_index)}

        raw_exact = True
        corrected_exact = True

        for task in tasks:
            id2label = label_maps[task]["id2label"]

            true_id = int(y_true[task][row_index])
            raw_id = int(raw_predictions[task][row_index])
            corrected_id = int(corrected_predictions[task][row_index])

            row[f"{task}_true"] = id2label[str(true_id)]
            row[f"{task}_raw"] = id2label[str(raw_id)]
            row[f"{task}_corrected"] = id2label[str(corrected_id)]
            row[f"{task}_confidence"] = float(probabilities[task][row_index, raw_id])

            raw_exact &= true_id == raw_id
            corrected_exact &= true_id == corrected_id

        row["raw_exact_match"] = bool(raw_exact)
        row["corrected_exact_match"] = bool(corrected_exact)
        rows.append(row)

    pd.DataFrame(rows).to_csv(
        output_dir / "test_predictions.csv",
        index=False,
        encoding="utf-8",
    )