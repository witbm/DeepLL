import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def extract(input       = "../results/manual/dataset",
            output      = "../results/manual/internal_states",
            mode        = "training",
            llm_name    = "meta-llama/Llama-3.1-8B",
            layer       = -1, # Layer index (-1 for last)
            token       = -1, # Token position (-1 for last)
            model       = None, # Optional: Preloaded model
            tokenizer   = None): # Optional: Preloaded tokenizer
    
    # Configuration
    input = Path(input) / f"{mode}.csv"
    output = Path(output) / mode
    df = pd.read_csv(input)
    labels = sorted(df["label"].astype(str).unique())

    # Load model if not already preloaded
    if model is None or tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(llm_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(llm_name, device_map="auto")
        model.config.output_hidden_states = True
        model.eval()

    # Inference
    for lab in labels:
        out_dir = output / lab
        out_dir.mkdir(parents=True, exist_ok=True) # Create A1,... folders

        sub = df[df["label"].astype(str) == lab] # Filter for current label
        print(f"extract {lab}\t {len(sub)}")
        for i, text in enumerate(sub["text"].astype(str).tolist()):
            enc = tokenizer(text, return_tensors="pt").to(model.device) # Tokenize text
            with torch.no_grad(): # Disable gradient calculation
                out = model(**enc) # Unpack dictionary "enc" and run with model (save output to "out")
            vec = out.hidden_states[layer][0, token, :].cpu().numpy() # Extract vector from output
            np.save(out_dir / f"{lab}_{i:06d}.npy", vec) # Save vector, for example "A1_000001.npy"

if __name__ == "__main__":
    if len(sys.argv) > 1:
        extract(mode=sys.argv[1])
    else: print("Add \"training\" or \"evaluation\" argument")