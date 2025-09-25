import shutil
import argparse
import random
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from split import split
from extract import extract
from train import train
from evaluate import evaluate

def pipeline(split_seed         = 1,
             train_seed         = 1,
             dir                = "../results/current/",
             train_test_split   = 0.8,
             model_name         = "meta-llama/Llama-3.1-8B",
             layer              = -1,
             token              = -1,
             train_only         = False):

    dir = Path(dir)
    print(f"\n{'='*15} Split Seed: {split_seed} Train Seed: {train_seed} {'='*15}")

    if not train_only: # Optionally skipt dataset creation
        # Delete previous results when creating a new dataset
        if dir.exists():
            shutil.rmtree(dir)
        
        # split.py
        print("\n--- Splitting corpus ---\n")
        split(input="../corpus",
                output=dir / "dataset",
                split=train_test_split,
                seed=split_seed)

        # Load LLM once
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        llm = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        llm.config.output_hidden_states = True
        llm.eval()

        # extract.py
        print("\n--- Extraction (train) ---\n")
        extract(input=dir / "dataset",
                output=dir / "internal_states",
                mode="training",
                model=llm,
                tokenizer=tokenizer,
                layer=layer,
                token=token)

        print("\n--- Extraction (eval) ---\n")
        extract(input=dir / "dataset",
                output=dir / "internal_states",
                mode="evaluation",
                model=llm,
                tokenizer=tokenizer,
                layer=layer,
                token=token)
        
        # Free GPU
        del llm
        del tokenizer
        torch.cuda.empty_cache()
        
    print("\n--- Training MLP ---\n")
    train(input=dir / "internal_states/training",
          output=dir / "mlp" / f"mlp{train_seed}",
          seed=train_seed)

    print("\n--- Evaluating MLP ---\n")
    evaluate(model=dir / "mlp" / f"mlp{train_seed}",
            input=dir / "internal_states/evaluation")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int, nargs="?", default=None) # number of train/test splits
    parser.add_argument("m", type=int, nargs="?", default=None) # number of trainings per split
    args = parser.parse_args() # "python pipeline.py 2 3" generates 2 * 3 = 6 models

    if args.n is None and args.m is None: pipeline() # No arguments, no batch processing
    else:
        for i in range(args.n): # generate n random splits
            split_seed = random.randint(1, 10000)
            split_dir = Path(f"../results/split{split_seed}") # New directory for each split
            for j in range(args.m): # Perform m random trainings for each split
                train_seed = random.randint(1, 10000)
                pipeline(split_seed=split_seed,
                        train_seed=train_seed,
                        dir=split_dir,
                        train_only=(j > 0)) # Only generate the dataset once per split