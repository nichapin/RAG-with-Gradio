from dataclasses import dataclass
import transformers
from transformers import AutoTokenizer
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

@dataclass
class ModelConfig:
    def __init__(self):
        self.model_name = "mistralai/Mistral-7B-Instruct-v0.1"
        self.get_token = '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'

    def pretrained_model_config(self):
        # HuggingFace login
        login(self.get_token)
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = self.get_token

        # Load config and tokenizer
        model_config = transformers.AutoConfig.from_pretrained(self.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # Use MPS (Mac) or CPU
        device = "mps" if torch.backends.mps.is_available() else "cpu"

        # Load model with correct device
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",              # Let it choose the best option (MPS/CPU/GPU)
            torch_dtype=torch.float16 if device == "mps" else torch.float32,
            # load_in_4bit=True,
        )
        return model, tokenizer