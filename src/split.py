import pandas as pd
from pathlib import Path

def split(input     = "../corpus",
          output    = "../results/manual/dataset",
          split     = 0.8,
          seed      = 1):
    
    # Preparation
    input = Path(input)
    output = Path(output)
    datasets = [f for f in input.iterdir() if f.suffix == ".csv"]
    combined = pd.concat([pd.read_csv(d) for d in datasets], ignore_index=True)
    train, eval = [], []

    # Split per CEFR-label
    for label, group in combined.groupby("label", dropna=False):
        train_sample = group.sample(frac=split, random_state=seed)
        eval_sample = group.drop(train_sample.index)
        train.append(train_sample)
        eval.append(eval_sample)

    # Create training and 
    train_df = pd.concat(train, ignore_index=True)
    eval_df = pd.concat(eval, ignore_index=True)

    # Save to .csv
    output.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output / "training.csv", index=False)
    eval_df.to_csv(output / "evaluation.csv", index=False)

    print("combined\t", len(combined))
    print("training\t", len(train_df))
    print("evaluation\t", len(eval_df))

    # Create README.md file
    readme_path = output.parent / "README.md"
    source_files = "\n".join([f"- `{d.name}`" for d in datasets])

    readme_content = f"""
# Dataset Generation Report

## Parameters

- **Split Seed:** `{seed}`
- **Split Ratio (Training):** `{split}`

## Source Files

The following files from the `{input.name}/` directory were combined to create the dataset:

{source_files}
"""
    # Write the content to README.md file
    readme_path.write_text(readme_content.strip())

if __name__ == "__main__":
    split()