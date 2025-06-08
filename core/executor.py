import re

def run_pipeline(query, code_chain, summary_chain, data):
    result = {}
    llm_code_output = code_chain.run(query=query)
    code_match = re.search(r"<code>(.*?)</code>", llm_code_output, re.DOTALL)
    if not code_match:
        return "Code not found", "", ""
    code_block = code_match.group(1).strip()
    try:
        exec(code_block, {"data": data, "result": result})
    except Exception as e:
        return f"Execution error: {e}", code_block, ""
    summary_output = summary_chain.run(query=query, result=result)
    summary_match = re.search(r"<one-line-summary>(.*?)</one-line-summary>", summary_output, re.DOTALL)
    summary = summary_match.group(1).strip() if summary_match else "Summary not found."
    return summary, code_block, result
