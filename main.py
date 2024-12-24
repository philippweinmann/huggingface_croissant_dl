from huggingface_hub import login
from pathlib import Path
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

current_file_directory = Path(__file__).parent
model_name = "croissantllm/CroissantLLMChat-v0.1"
model_save_directory = Path(current_file_directory / "downloaded_models" / model_name)

def download_model_locally():
    if not os.path.isdir(model_save_directory):
        # Replace 'your_huggingface_token' with your actual Hugging Face token
        login(token="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")

        model = AutoModelForCausalLM.from_pretrained(model_name, token=True)
        model.save_pretrained(model_save_directory)

        tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)

        os.makedirs(model_save_directory, exist_ok=True)
        tokenizer.save_pretrained(model_save_directory)
    else:
        raise Exception(f"""Model Folder at path {model_save_directory} already exists. 
                        Not doing anything. If you do want to download the model again, 
                        delete it before running this method.""") 

if __name__ == "__main__":
    download_model_locally()