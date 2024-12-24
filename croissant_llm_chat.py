
from huggingface_hub import login
from pathlib import Path
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline

current_file_directory = Path(__file__).parent
model_name = "croissantllm/CroissantLLMChat-v0.1"
model_save_directory = Path(current_file_directory / "downloaded_models" / model_name)

def load_model():
    model_fp = model_save_directory

    model = AutoModelForCausalLM.from_pretrained(model_fp, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_fp, local_files_only=True)

    return model, tokenizer

def test_llm_from_hugging_face(input: str, model, tokenizer):
    text_generation_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)

    llm_output = text_generation_pipeline(input, max_new_tokens=50)

    return llm_output[0]["generated_text"]

if __name__ == "__main__":
    model, tokenizer = load_model()

    
    print("Interactive Mode: Type your input and press Enter. Type 'exit' to quit.")

    while True:
        input_text = input(">> ")  # Prompt for user input
        if input_text.lower() == "exit":
            print("Exiting...")
            break

        try:
            # Call the function with the user input
            output = test_llm_from_hugging_face(input_text, model, tokenizer)
            print(f"Output: {output}")
        except Exception as e:
            print(f"An error occurred: {e}")