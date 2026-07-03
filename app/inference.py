import streamlit as st
from autocatalog.inference.predictor import AutoCatalogPredictor

@st.cache_resource(show_spinner=False)
def load_predictor(repo_id, device, top_k):
    return AutoCatalogPredictor(
        repo_id=repo_id,
        device=device,
        top_k=top_k
    )