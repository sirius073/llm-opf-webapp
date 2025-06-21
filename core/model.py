from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_community.llms import HuggingFacePipeline
from transformers import TextGenerationPipeline

def load_model(model_id="deepseek-ai/deepseek-coder-6.7b-instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,  # ✅ Use float32 on CPU
        trust_remote_code=True,
        device_map={"": "cpu"}      # ✅ Force CPU execution
    )

    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
        device=-1  # ✅ -1 tells HuggingFace to use CPU
    )

    return HuggingFacePipeline(pipeline=hf_pipeline)
