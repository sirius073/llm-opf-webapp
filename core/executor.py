import re
import streamlit as st
import json
import torch
from torch_geometric.data import HeteroData

code_template_raw2 = """
<instruction>
You are a Python data analyst and power systems expert with experience using torch geometric datasets.

Your task is to write clean, valid Python code that:
- Analyzes a power grid dataset stored in the variable `dataset`, a list of PyTorch Geometric `HeteroData` objects.
- Computes values or generates plots based on the user's request.
- Respects the exact data schema below — do not assume any additional fields.

# DATA SCHEMA (based on OPFData, with clear names and short forms)

## Global Fields:
- `data.x`: shape [1]
  - Column 0 → global_index_or_timestep

- `data.objective`: shape [1]
  - Column 0 → optimization_objective_value ($/h)

## NODE TYPES:

### `data['bus']` (Node):
- `data['bus'].x`: shape [num_buses, 4]
  - Column 0 → base_voltage_kv (base_kv)
  - Column 1 → bus_type (PQ=1, PV=2, ref=3, inactive=4)
  - Column 2 → minimum_voltage_magnitude_limit (vmin)
  - Column 3 → maximum_voltage_magnitude_limit (vmax)

- `data['bus'].y`: shape [num_buses, 2]
  - Column 0 → voltage_angle_solution (va)
  - Column 1 → voltage_magnitude_solution (vm)

### `data['generator']` (Node):
- `data['generator'].x`: shape [num_generators, 11]
  - Column 0 → machine_base_mva (mbase)
  - Column 1 → active_power_output (pg)
  - Column 2 → minimum_active_power (pmin)
  - Column 3 → maximum_active_power (pmax)
  - Column 4 → reactive_power_output (qg)
  - Column 5 → minimum_reactive_power (qmin)
  - Column 6 → maximum_reactive_power (qmax)
  - Column 7 → voltage_setpoint (vg)
  - Column 8 → cost_quadratic (c2)
  - Column 9 → cost_linear (c1)
  - Column 10 → cost_constant (c0)

- `data['generator'].y`: shape [num_generators, 2]
  - Column 0 → active_power_solution (pg)
  - Column 1 → reactive_power_solution (qg)

### `data['load']` (Node):
- `data['load'].x`: shape [num_loads, 2]
  - Column 0 → active_power_demand (pd)
  - Column 1 → reactive_power_demand (qd)

### `data['shunt']` (Node):
- `data['shunt'].x`: shape [num_shunts, 2]
  - Column 0 → susceptance (bs)
  - Column 1 → conductance (gs)

## EDGE TYPES (Heterogeneous):

### AC Line: `('bus', 'ac_line', 'bus')`
- `edge_index`: [2, num_ac_lines] — [source_bus_index, target_bus_index]
- `edge_attr`: [num_ac_lines, 9]
  - angle_min (θ_l), angle_max (θ_u), b_from, b_to,
    resistance (br_r), reactance (br_x),
    thermal_rating_a (rate_a), b (rate_b), c (rate_c)
- `edge_label`: [num_ac_lines, 4]
  - active_power_to (pt), reactive_power_to (qt),
    active_power_from (pf), reactive_power_from (qf)

### Transformer: `('bus', 'transformer', 'bus')`
- `edge_index`: [2, num_transformers]
- `edge_attr`: [num_transformers, 11]
  - angle_min (θ_l), angle_max (θ_u), resistance (br_r), reactance (br_x),
    thermal_rating_a (rate_a), b (rate_b), c (rate_c),
    tap_ratio (tap), phase_shift (shift), b_from, b_to
- `edge_label`: [num_transformers, 4] → [pt, qt, pf, qf]

### Generator/Load/Shunt Links:
- `('generator', 'generator_link', 'bus')` and `('bus', 'generator_link', 'generator')`
- `('load', 'load_link', 'bus')` and `('bus', 'load_link', 'load')`
- `('shunt', 'shunt_link', 'bus')` and `('bus', 'shunt_link', 'shunt')`
  • edge_index: [2, N] — connectivity only

# CODING RULES:
- You must iterate through all `data` in `dataset`.
- Use `matplotlib.pyplot` with `fig, ax = plt.subplots()` for plots.
- No markdown, comments, triple backticks, or explanations.
- Store all results in `result` dictionary.
- If any plots are generated, store them in `result["plots"] = [fig1, fig2, ...]`, or an empty list if none.
</instruction>
""" 
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
            st.warning(f"❌ Attempt {attempt} failed: {error_message}")
            st.info("🛠️ The LLM is attempting to fix the code and retry...")
            fix_prompt = f"""{code_template_raw2}
<user>
The following code failed. Fix it. Return only clean Python code completely inside 'correct-code' tag.
</user>
<broken-code>
{code_block}
</broken-code>
<error-message>
{error_message}
</error-message>
<correct-code>
"""
            fixed_output = code_chain.invoke({"query": fix_prompt})
            code_block = fixed_output.get("text", "") if isinstance(fixed_output, dict) else str(fixed_output)

            fixed_match = re.search(r"<correct-code>(.*?)</correct-code>", code_block, re.DOTALL) or \
                          re.search(r"```(?:python)?\n?(.*?)\n?```", code_block, re.DOTALL)
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
