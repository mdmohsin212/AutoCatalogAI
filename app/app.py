import json
import sys
from pathlib import Path
import streamlit as st
from PIL import Image
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
    
from inference import load_predictor, render_prediction_card, render_top_predictions, render_metrics
from autocatalog.utils.config import load_config    

def main():
    st.set_page_config(
        page_title="AutoCatalogAI",
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
            box-shadow: 0 1px 4px rgba(0,0,0,0.04);
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
            text-shadow: none !important;
        }
        .confidence {
            font-size: 0.9rem;
            font-weight: 700;
            color: #111827 !important;
            text-shadow: none !important;
        }
        .label {
            font-size: 1.25rem;
            font-weight: 800;
            color: #111827 !important;
            margin-top: 8px;
            margin-bottom: 10px;
            text-shadow: none !important;
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
            background: #fafafa;
            margin-bottom: 16px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    config = load_config("configs/config.yaml")
    repo_id = config.get("model", {}).get(
        "repo_id",
        "mohsin416/autocatalogai-clip-multitask",
    )
    
    top_k = int(config.get("inference", {}).get("top_k", 3))
    device = config.get("inference", {}).get("device", None)
    
    st.markdown('<div class="main-title">AutoCatalogAI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Fashion product attribute extraction and catalog metadata generation using CLIP multi-task learning.</div>',
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
        
        st.divider()
        st.caption("Model loads from Hugging Face Hub and runs inference only.")
    
    with st.spinner("Loading AutoCatalogAI model..."):
        predictor = load_predictor(
            repo_id=repo_id,
            device=device,
            top_k=selected_top_k
        )
    
    render_metrics(predictor.get_model_metrics())
    st.divider()
    
    left_col, right_col = st.columns([0.9, 1.1])
    
    with left_col:
        st.subheader("Upload Product Image")
        uploaded_file = st.file_uploader(
            "Choose a product image",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)

    with right_col:
        st.subheader("Prediction Result")

        if uploaded_file is None:
            st.info("Upload a fashion product image to generate catalog attributes.")
            return

        if st.button("Generate Catalog", type="primary", use_container_width=True):
            with st.spinner("Predicting product attributes..."):
                result = predictor.predict(
                    image=image,
                    top_k=selected_top_k,
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
            st.write(", ".join(catalog_output["search_tags"]))
            st.markdown("**Predicted Attributes**")

            for task, task_result in prediction.items():
                render_prediction_card(task, task_result)

            render_top_predictions(prediction)
            st.markdown("**Runtime**")
            st.write(f"Device: `{runtime['device']}`")
            st.write(f"Inference time: `{runtime['inference_time_ms']:.2f} ms`")

            json_output = json.dumps(
                catalog_output["json_export"],
                indent=2,
                ensure_ascii=False,
            )

            st.download_button(
                label="Download JSON",
                data=json_output,
                file_name="autocatalogai_prediction.json",
                mime="application/json",
                use_container_width=True,
            )

            with st.expander("Raw JSON Output"):
                st.json(catalog_output["json_export"])


if __name__ == "__main__":
    main()