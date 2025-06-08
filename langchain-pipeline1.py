import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load dataset
with open("your_json_file.json", "r") as f:
    data = json.load(f)

# Load model once
def load_model():
    model_id = "deepseek-ai/deepseek-coder-6.7b-instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto"
    )
    hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=500)
    return HuggingFacePipeline(pipeline=hf_pipeline)

# Code generation prompt
code_template = PromptTemplate(
    input_variables=["query"],
    template= """
<instruction>
You are a Python expert. I have already loaded a dataset from a JSON file into a Python variable called `data`.

The dataset contains power grid information structured using arrays only. Here's how the data is organized:

#============================#
#       DATA STRUCTURE       #
#============================#

data = {{
  "grid": {{
    "nodes": {{
      "bus": [[...], [...], ...],         # = [base_voltage_kv (0), bus_type (1), voltage_minimum (2), voltage_maximum (3)]
      "generator": [[...], [...], ...],   # = [machine_base_mva (0), active_power_output (1), active_power_minimum (2), active_power_maximum (3), reactive_power_output (4), reactive_power_minimum (5), reactive_power_maximum (6), voltage_setpoint (7), cost_quadratic (8), cost_linear (9), cost_constant (10)]
      "load": [[...], [...], ...],        # = [active_power_demand (0), reactive_power_demand (1)]
      "shunt": [[...], [...], ...]        # = [susceptance (0), conductance (1)]
    }},
    "edges": {{
      "ac_line": {{
        "senders": [...],
        "receivers": [...],
        "features": [[...], [...], ...]   # = [angle_minimum (0), angle_maximum (1), from_end_susceptance (2), to_end_susceptance (3), resistance (4), reactance (5), thermal_rating_a (6), thermal_rating_b (7), thermal_rating_c (8)]
      }},
      "transformer": {{
        "senders": [...],
        "receivers": [...],
        "features": [[...], [...], ...]   # = [angle_minimum (0), angle_maximum (1), resistance (2), reactance (3), thermal_rating_a (4), thermal_rating_b (5), thermal_rating_c (6), tap_ratio (7), phase_shift (8), from_end_susceptance (9), to_end_susceptance (10)]
      }},
      "generator_link": {{ "senders": [...], "receivers": [...] }},
      "load_link": {{ "senders": [...], "receivers": [...] }},
      "shunt_link": {{ "senders": [...], "receivers": [...] }}
    }},
    "context": {{
      "baseMVA": float
    }}
  }},
  "solution": {{
    "nodes": {{
      "bus": [[...], [...], ...],         # = [voltage_angle (0), voltage_magnitude (1)]
      "generator": [[...], [...], ...]    # = [active_power_output (0), reactive_power_output (1)]
    }},
    "edges": {{
      "ac_line": {{
        "senders": [...],
        "receivers": [...],
        "features": [[...], [...], ...]   # = [power_to_bus (0), reactive_power_to_bus (1), power_from_bus (2), reactive_power_from_bus (3)]
      }},
      "transformer": {{
        "senders": [...],
        "receivers": [...],
        "features": [[...], [...], ...]   # = [power_to_bus (0), reactive_power_to_bus (1), power_from_bus (2), reactive_power_from_bus (3)]
      }}
    }}
  }},
  "metadata": {{
    "objective": float
  }}
}}

The values inside all node/edge lists are arrays with fixed positional meanings as shown above.

You must respond only with valid, clean Python code using this `data` structure.
Do not explain anything.
Do not output text outside code.
Generate only concise, correct Python code.
Do not print the numerical outputs. Store all numerical outputs inside the `result` dictionary with appropriate keys.
Only display plots using plt.show().
</instruction>
<user>
{query}
</user>
<code>"""
)

# Summary prompt
summary_template = PromptTemplate(
    input_variables=["query", "result"],
    template="""<instruction>
You are a data analyst. You are given:

- A user query related to a power grid dataset.
- The result of a Python code execution for that query.

Based on the result and query, output the result in one line.
</instruction>

<user>
{query}
</user>
<result>
{result}
</result>
<one-line-summary>"""
)

def run_pipeline(user_query, code_chain, summary_chain):
    global result
    result = {}

    llm_code_output = code_chain.run(query=user_query)
    code_match = re.search(r"<code>(.*?)</code>", llm_code_output, re.DOTALL)

    if not code_match:
        print("Code not found.")
        return

    code_block = code_match.group(1).strip()
    print("\n--- Generated Code ---\n", code_block)

    try:
        exec(code_block, globals())
    except Exception as e:
        print(f"Execution error: {e}")
        return

    print("\n--- Execution Result ---\n", result)

    summary_output = summary_chain.run(query=user_query, result=result)
    summary_match = re.search(r"<one-line-summary>(.*?)</one-line-summary>", summary_output, re.DOTALL)

    if summary_match:
        print("\n--- Summary ---\n", summary_match.group(1).strip())
    else:
        print("Summary not found.")

def main():
    print("Loading model... Please wait.")
    llm = load_model()
    print("Model loaded.")

    code_chain = LLMChain(llm=llm, prompt=code_template)
    summary_chain = LLMChain(llm=llm, prompt=summary_template)

    while True:
        user_query = input("\nEnter your query (or type 'stop' to exit): ").strip()
        if user_query.lower() == "stop":
            print("Exiting...")
            break
        run_pipeline(user_query, code_chain, summary_chain)

if __name__ == "__main__":
    main()
