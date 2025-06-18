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

Each `HeteroData` object (let's call it `data`) represents a power grid with multiple node and edge types. Here's the exact structure and dimensional meanings:

DATA SCHEMA: -

# Scalar fields:
- `data.x`: Shape = [1], global features (e.g., index or time)
- `data.objective`: Shape = [1], optimization objective (float)

# Node types:

- `data['bus'].x`: Shape = [14, 4], columns = [base_kv, type, vmin, vmax]
- `data['bus'].y`: Shape = [14, 2], columns = [voltage_angle, voltage_magnitude] (solution)

- `data['generator'].x`: Shape = [5, 11], columns = [machine_base_mva, p_out, p_min, p_max, q_out, q_min, q_max, v_setpoint, cost_quad, cost_lin, cost_const]
- `data['generator'].y`: Shape = [5, 2], columns = [p_output, q_output] (solution)

- `data['load'].x`: Shape = [11, 2], columns = [p_demand, q_demand]

- `data['shunt'].x`: Shape = [1, 2], columns = [susceptance, conductance]

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
- take base_mva as 100 always.
- Write only valid, clean Python code.
- The `data` object is preloaded—**do not load JSON** or import it.
- You have to sum the outputs from all the data objects and then give the results and plots.
- Use `matplotlib.pyplot` for any plotting with `fig, ax = plt.subplots()`.
- Store all outputs in a dictionary named `result`.
- If any plots are generated, store them in a list: `result["plots"] = [fig1, fig2, ...]`
- Do **not** store `plt` itself or include explanations.
- Do not return anything in the code
- The code should be enclosed inside <code>...</code> 
- Ensure `result["plots"]` exists and is a list (even if empty or only one plot).

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
