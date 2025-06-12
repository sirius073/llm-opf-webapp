import os

os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

import torch
torch.classes.__path__ = []

import streamlit as st
from config.prompts import code_template, summary_template
from core.model import load_model
from core.executor import run_pipeline
from langchain.chains import LLMChain

st.set_page_config(page_title="Power Grid LLM Interface", layout="wide")
st.title("🔌 Power Grid Code Assistant with LLM")

# Session setup
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

with st.sidebar:
    st.header("Configuration")
    model_id = st.text_input("Model ID", value="deepseek-ai/deepseek-coder-6.7b-instruct")
    uploaded_file = st.file_uploader("Upload JSON file", type="json")
    if st.button("Load Model and Data"):
        if uploaded_file:
            st.session_state.data = json.load(uploaded_file)
            st.session_state.llm = load_model(model_id)
            st.session_state.code_chain = LLMChain(llm=st.session_state.llm, prompt=code_template)
            st.session_state.summary_chain = LLMChain(llm=st.session_state.llm, prompt=summary_template)
            st.session_state.model_loaded = True
            st.success("Model and data loaded!")

if st.session_state.model_loaded:
    st.subheader("💬 Ask a Question")
    query = st.text_area("Enter your prompt", height=150)
    if st.button("Run Query"):
        with st.spinner("Running..."):
            summary, code, result_dict = run_pipeline(
                query,
                st.session_state.code_chain,
                st.session_state.summary_chain,
                st.session_state.data,
            )
        st.code(code, language="python")
        st.json(result_dict)
        st.success(summary)
else:
    st.info("Upload a file and load model to start.")
