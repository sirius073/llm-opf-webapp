from langchain.prompts import PromptTemplate

code_template = PromptTemplate(
    input_variables=["query"],
    template=
"""
<instruction>
You are an expert in:
- Electrical Power Systems
- Data Analysis using PyTorch Geometric
- Writing Python code for data analysis and visualization

The dataset has already been loaded into a variable named `dataset`. It is a list of `HeteroData` objects (from torch_geometric).
Each `HeteroData` object (let's call it `data`) represents a power grid with multiple node and edge types. 

Here's the exact structure and dimensional meanings:

DATA SCHEMA: -

- Node Types:

  • `data['bus'].x`: [14, 4] → [base_kv, type, vmin, vmax]  
  • `data['bus'].y`: [14, 2] → [voltage_angle, voltage_magnitude]  

  • `data['generator'].x`: [5, 11] → [machine_base_mva, p_out, p_min, p_max, q_out, q_min, q_max, v_setpoint, cost_quad, cost_lin, cost_const]  
  • `data['generator'].y`: [5, 2] → [active_power_output, reactive_power_output]  

  • `data['load'].x`: [11, 2] → [active_power_demand, reactive_power_demand]  
  • `data['shunt'].x`: [1, 2] → [susceptance, conductance]  

# Edge types (heterogeneous):

- ('bus', 'ac_line', 'bus'):
  • edge_index: [2, 17] → [source_bus_idx, target_bus_idx] for each AC line
  • edge_attr: [17, 9] → [angle_min, angle_max, b_from, b_to, resistance, reactance, thermal_a, thermal_b, thermal_c]
  • edge_label: [17, 4] → [p_to, q_to, p_from, q_from]

- ('bus', 'transformer', 'bus'):
  • edge_index: [2, 3] → [source_bus_idx, target_bus_idx] for each transformer
  • edge_attr: [3, 11] → [angle_min, angle_max, resistance, reactance, thermal_a, thermal_b, thermal_c, tap_ratio, phase_shift, b_from, b_to]
  • edge_label: [3, 4] → [p_to, q_to, p_from, q_from]

- ('generator', 'generator_link', 'bus') and ('bus', 'generator_link', 'generator'):
  • edge_index: [2, 5] → [src, tgt] mapping of generator and its linked bus

- ('load', 'load_link', 'bus') and ('bus', 'load_link', 'load'):
  • edge_index: [2, 11] → connectivity between loads and buses

- ('shunt', 'shunt_link', 'bus') and ('bus', 'shunt_link', 'shunt'):
  • edge_index: [2, 1] → connectivity between shunt elements and buses

 INSTRUCTIONS: -
- You MUST loop over the entire dataset
- Always take base_mva as 100
- The `dataset` variable is preloaded — do not load files or JSON
- Do **not** include markdown (no triple backticks like ```python)
- Loop through each `data in dataset` and sum results if needed
- Use only standard Python libraries + matplotlib
- Use `fig, ax = plt.subplots()` for plots
- Store all results in a Python dictionary: `result`
- If plots are created, store them in a list: `result["plots"] = [fig1, fig2, ...]`
- `result["plots"]` must always be a list (even if empty)
- Do not include any explanation or return statements

</instruction>

<user>
{query}
</user>

<code>
"""
)

summary_template = PromptTemplate(
    input_variables=["query", "result"],
    template=
"""
<instruction>
You are a data analyst. You are given:
- A user query on a power grid dataset.
- The Python execution result for that query.

Based on this, write a one-line summary of the output.
</instruction>

<user>
{query}
</user>

<result>
{result}
</result>

<one-line-summary>
"""
)
