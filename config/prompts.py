from langchain.prompts import PromptTemplate

code_template = PromptTemplate(
    input_variables=["query"],
    template=
"""
<instruction>
You are a Python data analyst and power systems expert and have expertise in torch geometric datasets.

Your task is to write clean, valid Python code that:
- Analyzes a power grid dataset stored in the variable `dataset`, a list of PyTorch Geometric `HeteroData` objects.
- Computes values or generates plots based on the user's request.
- Respects the exact data schema below — do not assume any additional fields.

# DATA SCHEMA (with index mapping and clear names)

## Global Fields:
- `data.x`: shape [1]
  - Column 0 → global_index_or_timestep

- `data.objective`: shape [1]
  - Column 0 → optimization_objective_value

## NODE TYPES:

- `data['bus'].x`: shape [num_buses, 4]
  - Column 0 → base_voltage_kv
  - Column 1 → bus_type
  - Column 2 → minimum_voltage_magnitude_limit
  - Column 3 → maximum_voltage_magnitude_limit

- `data['bus'].y`: shape [num_buses, 2]
  - Column 0 → voltage_angle_solution
  - Column 1 → voltage_magnitude_solution

- `data['generator'].x`: shape [num_generators, 11]
  - Column 0 → machine_base_mva
  - Column 1 → active_power_output (p_out)
  - Column 2 → minimum_active_power (p_min)
  - Column 3 → maximum_active_power (p_max)
  - Column 4 → reactive_power_output (q_out)
  - Column 5 → minimum_reactive_power (q_min)
  - Column 6 → maximum_reactive_power (q_max)
  - Column 7 → voltage_setpoint
  - Column 8 → cost_quadratic
  - Column 9 → cost_linear
  - Column 10 → cost_constant

- `data['generator'].y`: shape [num_generators, 2]
  - Column 0 → active_power_solution
  - Column 1 → reactive_power_solution

- `data['load'].x`: shape [num_loads, 2]
  - Column 0 → active_power_demand
  - Column 1 → reactive_power_demand

- `data['shunt'].x`: shape [num_shunts, 2]
  - Column 0 → susceptance
  - Column 1 → conductance

## EDGE TYPES (heterogeneous):

- ('bus', 'ac_line', 'bus'):
  • `edge_index`: [2, num_ac_lines]
    - Row 0 → source_bus_index
    - Row 1 → target_bus_index
  • `edge_attr`: [num_ac_lines, 9]
    - Column 0 → angle_min
    - Column 1 → angle_max
    - Column 2 → b_from
    - Column 3 → b_to
    - Column 4 → resistance
    - Column 5 → reactance
    - Column 6 → thermal_rating_a
    - Column 7 → thermal_rating_b
    - Column 8 → thermal_rating_c
  • `edge_label`: [num_ac_lines, 4]
    - Column 0 → active_power_to
    - Column 1 → reactive_power_to
    - Column 2 → active_power_from
    - Column 3 → reactive_power_from

- ('bus', 'transformer', 'bus'):
  • `edge_index`: [2, num_transformers]
    - Row 0 → source_bus_index
    - Row 1 → target_bus_index
  • `edge_attr`: [num_transformers, 11]
    - Column 0 → angle_min
    - Column 1 → angle_max
    - Column 2 → resistance
    - Column 3 → reactance
    - Column 4 → thermal_rating_a
    - Column 5 → thermal_rating_b
    - Column 6 → thermal_rating_c
    - Column 7 → tap_ratio
    - Column 8 → phase_shift
    - Column 9 → b_from
    - Column 10 → b_to
  • `edge_label`: [num_transformers, 4]
    - Column 0 → active_power_to
    - Column 1 → reactive_power_to
    - Column 2 → active_power_from
    - Column 3 → reactive_power_from

- ('generator', 'generator_link', 'bus') and ('bus', 'generator_link', 'generator'):
  • `edge_index`: [2, num_generators]

- ('load', 'load_link', 'bus') and ('bus', 'load_link', 'load'):
  • `edge_index`: [2, num_loads]

- ('shunt', 'shunt_link', 'bus') and ('bus', 'shunt_link', 'shunt'):
  • `edge_index`: [2, num_shunts]

# CODING RULES
- You must loop through all `data` in `dataset` — never use just one `data` object.
- Use `base_mva = 100` in all calculations.
- Do not import or load any files — `dataset` is preloaded.
- Use `matplotlib.pyplot` with `fig, ax = plt.subplots()` to create plots.
- Do not add markdown, triple backticks, or explanations.
- Store results in a dictionary called `result`.
- All plots must be stored in `result["plots"] = [fig1, fig2, ...]`, or an empty list if no plots are generated.

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
You are a concise data analyst.

You are given:
- A user query related to electrical power grid data analysis.
- The result of executing Python code on the dataset.

Write **exactly one line** summarizing the result of that code execution.
Do not speculate — summarize only what the result dictionary contains.
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
