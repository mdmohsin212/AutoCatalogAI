import streamlit as st
from autocatalog.inference.predictor import AutoCatalogPredictor

@st.cache_resource(show_spinner=False)
def load_predictor(repo_id, device=None):
    return AutoCatalogPredictor(
        repo_id=repo_id,
        device=device,
    )


def format_percent(value):
    return f"{value * 100:.2f}%"

def render_prediction_card(task_name, task_result):
    label = task_result["label"]
    confidence = task_result["confidence"]

    st.markdown(
        f"""
        <div class="prediction-card">
            <div class="prediction-header">
                <span class="task-name">{task_name}</span>
                <span class="confidence">
                    {format_percent(confidence)}
                </span>
            </div>
            <div class="label">{label}</div>
            <div class="bar-bg">
                <div
                    class="bar-fill"
                    style="width: {confidence * 100:.2f}%"
                ></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_top_predictions(prediction):
    with st.expander("View Top-3 Predictions"):
        for task, result in prediction.items():
            st.markdown(f"**{task}**")
            for item in result["top_3"]:
                st.write(
                    f"{item['label']} — "
                    f"{format_percent(item['confidence'])}"
                )

            st.divider()


def render_metrics(metrics):
    if not metrics:
        return

    overall = metrics.get("overall_metrics",{},)
    if not overall:
        return

    st.subheader("Model Evaluation")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Average Accuracy",
        format_percent(
            overall.get(
                "average_accuracy",
                0,
            )
        ),
    )
    col2.metric(
        "Weighted F1",
        format_percent(
            overall.get(
                "average_weighted_f1",
                0,
            )
        ),
    )
    col3.metric(
        "Top-3 Accuracy",
        format_percent(
            overall.get(
                "average_top3_accuracy",
                0,
            )
        ),
    )
    col4.metric(
        "Test Samples",
        f"{overall.get('samples', 0):,}",
    )