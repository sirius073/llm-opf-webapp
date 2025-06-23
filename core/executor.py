import re
import streamlit as st
import json
import torch
from torch_geometric.data import HeteroData

def run_pipeline(query, code_chain, summary_chain, dataset: HeteroData):
    result = {}
    torch.cuda.empty_cache()
    max_attempts = 2  # 🔁 Retry up to N times on failure

    raw_code_output = code_chain.invoke({"query": query})
    llm_code_output = raw_code_output.get("text", "") if isinstance(raw_code_output, dict) else str(raw_code_output)

    code_match = re.search(r"<code>(.*?)</code>", llm_code_output, re.DOTALL) or \
                 re.search(r"```(?:python)?\n?(.*?)\n?```", llm_code_output, re.DOTALL)

    if not code_match:
        return "Code not found", "", {}

    code_block = code_match.group(1).strip()
    if code_block.startswith("```"):
        code_block = re.sub(r"^```(?:python)?\n?", "", code_block)
        code_block = re.sub(r"\n?```$", "", code_block)
    code_block = code_block.strip()

    attempt = 0
    error_message = ""
    while attempt < max_attempts:
        try:
            exec_scope = {
                "dataset": dataset,
                "result": result,
                "torch": torch,
                "st": st,
            }
            exec(code_block, exec_scope)
            result = exec_scope.get("result", {})
            break  # ✅ Success, break the loop
        except Exception as e:
            error_message = str(e)
            attempt += 1
            if attempt >= max_attempts:
                return f"Execution error after {max_attempts} attempts: {error_message}", code_block, {}

            # 🔧 Ask LLM to fix the broken code
            fix_prompt = f"""
The following code failed with an error. Please fix the code based on the error message and return only the corrected Python code.

--- Original Code ---
{code_block}

--- Error ---
{error_message}
"""
            fixed_output = code_chain.invoke({"query": fix_prompt})
            code_block = fixed_output.get("text", "") if isinstance(fixed_output, dict) else str(fixed_output)

            # Optional: Re-extract clean code if wrapped in ```
            fixed_match = re.search(r"```(?:python)?\n?(.*?)\n?```", code_block, re.DOTALL)
            if fixed_match:
                code_block = fixed_match.group(1).strip()

    # 🧼 Post-processing for JSON
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
        if k not in ["plot", "plots"]
    }
    torch.cuda.empty_cache()

    # 🧠 Summarize output
    raw_summary_output = summary_chain.invoke({
        "query": query,
        "result": json.dumps(serializable_result, indent=2)
    })
    summary_output = raw_summary_output.get("text", "") if isinstance(raw_summary_output, dict) else str(raw_summary_output)
    summary_match = re.search(r"<one-line-summary>(.*?)</one-line-summary>", summary_output, re.DOTALL)
    summary = summary_match.group(1).strip() if summary_match else "Summary not found."

    return summary, code_block, result
