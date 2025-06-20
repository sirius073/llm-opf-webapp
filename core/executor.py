import re
import streamlit as st
import json
import torch
from torch_geometric.data import HeteroData

def run_pipeline(query, code_chain, summary_chain, dataset: HeteroData):
    result = {}  # This will collect all outputs from the executed code
    torch.cuda.empty_cache()

    # 🧠 Call the LLM code chain safely
    raw_code_output = code_chain.invoke({"query": query})

    # 🔍 Extract text if wrapped in a dict or LangChain object
    if isinstance(raw_code_output, dict):
        llm_code_output = raw_code_output.get("text", "")
    else:
        llm_code_output = str(raw_code_output)

    # 🔍 Extract code between <code>...</code>
    code_match = re.search(r"<code>(.*?)</code>", llm_code_output, re.DOTALL)

    # Fallback: try extracting from triple backticks
    if not code_match:
        code_match = re.search(r"```(?:python)?\n?(.*?)\n?```", llm_code_output, re.DOTALL)

    if not code_match:
        return "Code not found", "", {}

    # ✅ Clean extracted code
    code_block = code_match.group(1).strip()
    if code_block.startswith("```"):
        code_block = re.sub(r"^```(?:python)?\n?", "", code_block)
        code_block = re.sub(r"\n?```$", "", code_block)
    code_block = code_block.strip()

    # 🔄 Execute the generated code safely
    try:
        exec_scope = {
            "dataset": dataset,
            "result": result,
            "torch": torch,
            "st": st,
        }
        exec(code_block, exec_scope)
        result = exec_scope.get("result", {})
    except Exception as e:
        return f"Execution error: {e}", code_block, {}

    # 📄 Convert tensors and objects into JSON-serializable form
    def make_serializable(v):
        if isinstance(v, torch.Tensor):
            return v.tolist()
        if isinstance(v, dict):
            return {k: make_serializable(vv) for k, vv in v.items()}
        if isinstance(v, list):
            return [make_serializable(vv) for vv in v]
        return v

    serializable_result = {
        k: make_serializable(v)
        for k, v in result.items()
        if k not in ["plot", "plots"]  # skip non-serializable visual objects
    }
    torch.cuda.empty_cache()

    # 🧠 Call the summary LLM
    raw_summary_output = summary_chain.invoke({
        "query": query,
        "result": json.dumps(serializable_result, indent=2)
    })

    if isinstance(raw_summary_output, dict):
        summary_output = raw_summary_output.get("text", "")
    else:
        summary_output = str(raw_summary_output)

    # 📝 Extract summary from <one-line-summary> tags
    summary_match = re.search(r"<one-line-summary>(.*?)</one-line-summary>", summary_output, re.DOTALL)
    summary = summary_match.group(1).strip() if summary_match else "Summary not found."

    return summary, code_block, result
