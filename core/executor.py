import re
import streamlit as st

def run_pipeline(query, code_chain, summary_chain, data):
    result = {}  # This will collect all outputs from the code
    llm_code_output = code_chain.run(query=query)

    # 🔍 Extract code between <code>...</code>
    code_match = re.search(r"<code>(.*?)</code>", llm_code_output, re.DOTALL)
    if not code_match:
        return "Code not found", "", {}

    code_block = code_match.group(1).strip()

    try:
        exec_scope = {
            "data": data,
            "result": result,
            "st": st,  # optionally allow st access
        }
        exec(code_block, exec_scope)
        result = exec_scope.get("result", {})
    except Exception as e:
        return f"Execution error: {e}", code_block, {}

    # 📄 Extract summary
    summary_output = summary_chain.run(query=query, result_json=json.dumps(result, indent=2))
    summary_match = re.search(r"<one-line-summary>(.*?)</one-line-summary>", summary_output, re.DOTALL)
    summary = summary_match.group(1).strip() if summary_match else "Summary not found."

    return summary, code_block, result
