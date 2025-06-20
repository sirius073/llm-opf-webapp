import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

import torch
torch.classes.__path__ = []

import streamlit as st
import json
from config.prompts import code_template, summary_template
from core.model import load_model
from core.executor import run_pipeline
from langchain.chains import LLMChain
from torch_geometric.datasets import OPFDataset

st.set_page_config(page_title="Power Grid LLM Interface", layout="wide")
st.title("🔌 Power Grid Code Assistant with LLM")

# Session setup
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

# Sidebar for model + data loading
with st.sidebar:
    st.header("Configuration")
    model_id = st.text_input("Model ID", value="deepseek-ai/deepseek-coder-6.7b-instruct")
    
    dataset_options = [
        "pglib_opf_case14_ieee",
        "pglib_opf_case118_ieee"
    ]
    selected_case = st.selectbox("Select OPF Dataset Case", dataset_options)

    if st.button("Load Model and Data"):
        try:
            dataset = OPFDataset(root='data', case_name=selected_case)
            st.session_state.data = dataset # Load the first (and usually only) graph
            st.session_state.llm = load_model(model_id)
            
            st.write("Code template input vars:", code_template.input_variables)
            st.write("Summary template input vars:", summary_template.input_variables)
            st.session_state.code_chain = LLMChain(llm=st.session_state.llm, prompt=code_template)
            st.write("✅ Code chain input keys:", st.session_state.code_chain.input_keys)
            st.session_state.summary_chain = LLMChain(llm=st.session_state.llm, prompt=summary_template)
            st.session_state.model_loaded = True
            st.success(f"✅ Loaded model and {selected_case}!")
        except Exception as e:
            st.error(f"❌ Error loading dataset: {e}")

# Main logic after loading
if st.session_state.model_loaded:
    st.subheader("💬 Ask a Question")
    query = st.text_area("Enter your prompt", height=150)

    if st.button("Run Query"):
        with st.spinner("⚙️ Running..."):
            summary, code, result_dict = run_pipeline(
                query,
                st.session_state.code_chain,
                st.session_state.summary_chain,
                st.session_state.data,
            )

        st.subheader("🧠 Generated Code")
        st.code(code, language="python")

        
        st.subheader("📦 Result Dictionary")
        st.json(result_dict)

        # 🔹 Handle multiple plots
        if "plots" in result_dict:
            all_plots = result_dict["plots"]
            if all_plots:
                st.subheader("📊 Plots")
                # Make sure it's a list, even if it's a single Figure
                if not isinstance(all_plots, list):
                    all_plots = [all_plots]
                for i, fig in enumerate(all_plots):
                    st.markdown(f"**Plot {i+1}**")
                    st.pyplot(fig)


        # 🔸 Handle single plot
        elif "plot" in result_dict:
            try:
                st.pyplot(result_dict["plot"])
            except:
                st.plotly_chart(result_dict["plot"])
        
        # Final summary
        st.success(f"✅ {summary}")
else:
    st.info("📂 Upload a file and load the model to begin.")
