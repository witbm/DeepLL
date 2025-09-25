import torch
import joblib
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

if __name__ == "__main__":
    
    # Load MLP
    model_dir = Path("../results/english/mlp/best_mlp8761")
    llm_name = "meta-llama/Llama-3.1-8B"
    pipeline = joblib.load(model_dir / "mlp.joblib")
    levels = json.loads((model_dir / "meta.json").read_text())["classes"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load LLM
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    llm_model = AutoModelForCausalLM.from_pretrained(llm_name, device_map="auto")
    llm_model.config.output_hidden_states = True
    llm_model.eval()

    # Classify user input (loop)
    while True:
        text = input("\n>>> ")
        if text.lower() in ["exit", "quit"]:
            break
        if not text.strip():
            continue

        # Extract CEV
        enc = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = llm_model(**enc)
        vec = out.hidden_states[-1][0, -1, :].cpu().numpy()

        # Calculate probabilities for CEFR levels
        prob = pipeline.predict_proba(vec.reshape(1, -1))[0]
        distribution = {level: prob for level, prob in zip(levels, prob)}

        print("\n--- CEFR prediction ---")
        for level, prob in sorted(distribution.items(), key=lambda item: item[1], reverse=True):
            print(f"{level}:\t{prob:.2%}")
        print("-" * 23)