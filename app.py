import json
import sys
from pathlib import Path
import streamlit as st
from PIL import Image
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from src.inference import load_predictor,render_metrics,render_prediction_card,render_top_predictions
from autocatalog.utils.config import load_config

def main():
    st.set_page_config(
        page_title="AutoCatalogAI V2",
        page_icon="🛍️",
        layout="wide",
    )
    
    st.markdown(
        """
        <style>
        .main-title {
            font-size: 2.5rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            color: #666;
            font-size: 1.05rem;
            margin-bottom: 2rem;
        }
        .prediction-card {
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 16px;
            margin-bottom: 12px;
            background: #ffffff !important;
            color: #111827 !important;
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04);
        }
        .prediction-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .task-name {
            font-size: 0.9rem;
            font-weight: 700;
            color: #374151 !important;
        }
        .confidence {
            font-size: 0.9rem;
            font-weight: 700;
            color: #111827 !important;
        }
        .label {
            font-size: 1.25rem;
            font-weight: 800;
            color: #111827 !important;
            margin-top: 8px;
            margin-bottom: 10px;
        }
        .bar-bg {
            width: 100%;
            height: 8px;
            background: #e5e7eb;
            border-radius: 999px;
            overflow: hidden;
        }
        .bar-fill {
            height: 100%;
            background: #111827;
            border-radius: 999px;
        }
        .catalog-box {
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 18px;
            background: #fafafa !important;
            color: #111827 !important;
            margin-bottom: 16px;
        }
        .catalog-box strong {
            color: #4b5563 !important;
        }
        .catalog-box h3 {
            color: #111827 !important;
            margin-top: 4px;
            margin-bottom: 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    config = load_config(ROOT_DIR / "configs" / "config.yaml")
    repo_id = config.get("model", {}).get(
        "repo_id",
        "mohsin416/autocatalogai-clip-multitask-v2",
    )

    inference_config = config.get("inference", {})
    top_k = int(inference_config.get("top_k", 3))
    device = inference_config.get("device")
    default_consistency = bool(
        inference_config.get(
            "apply_consistency_rules",
            True,
        )
    )

    st.markdown(
        '<div class="main-title">AutoCatalogAI V2</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="subtitle">
            Fashion product attribute extraction and catalog metadata
            generation using CLIP, colour features, and hierarchical learning.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Settings")
        st.write("Model repository")
        st.code(repo_id)

        selected_top_k = st.slider(
            "Top-K predictions",
            min_value=1,
            max_value=5,
            value=top_k,
        )

        consistency_enabled = st.toggle(
            "Apply consistency correction",
            value=default_consistency,
        )

        st.divider()
        st.caption(
            "The model loads from Hugging Face Hub "
            "and performs inference only."
        )

    with st.spinner("Loading AutoCatalogAI V2 model..."):
        predictor = load_predictor(
                repo_id=repo_id,
                device=device,
        )

    metrics = predictor.get_model_metrics()
    corrected_metrics = metrics.get("corrected", metrics)
    render_metrics(corrected_metrics)
    st.divider()
    left_col, right_col = st.columns([0.9, 1.1])
    image = None

    with left_col:
        st.subheader("Upload Product Image")
        uploaded_file = st.file_uploader(
            "Choose a product image",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(
                image,
                caption="Uploaded Image",
                width="stretch",
            )

    with right_col:
        st.subheader("Prediction Result")
        if image is None:
            st.info(
                "Upload a fashion product image "
                "to generate catalog attributes."
            )
            return

        if st.button(
            "Generate Catalog",
            type="primary",
            width="stretch",
        ):
            with st.spinner("Predicting product attributes..."):
                result = predictor.predict(
                    image=image,
                    top_k=selected_top_k,
                    apply_consistency_rules=consistency_enabled,
                )

            prediction = result["prediction"]
            catalog_output = result["catalog_output"]
            runtime = result["runtime"]
            st.markdown(
                f"""
                <div class="catalog-box">
                    <strong>Suggested Title</strong>
                    <h3>{catalog_output["suggested_title"]}</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("**Search Tags**")
            st.write(
                ", ".join(
                    catalog_output["search_tags"]
                )
            )

            st.markdown("**Predicted Attributes**")
            for task, task_result in prediction.items():
                render_prediction_card(task,task_result,)
                if task_result.get("corrected"):
                    st.caption(
                        f"Corrected from: "
                        f"{task_result['raw_label']}"
                    )

            render_top_predictions(prediction)
            st.markdown("**Runtime**")
            st.write(f"Device: `{runtime['device']}`")
            st.write(f"Inference time: "f"`{runtime['inference_time_ms']:.2f} ms`")
            json_output = json.dumps(
                catalog_output["json_export"],
                indent=2,
                ensure_ascii=False,
            )

            st.download_button(
                label="Download JSON",
                data=json_output,
                file_name="autocatalogai_v2_prediction.json",
                mime="application/json",
                width="stretch",
            )

            with st.expander("Raw JSON Output"):
                st.json(catalog_output["json_export"])

if __name__ == "__main__":
    main()