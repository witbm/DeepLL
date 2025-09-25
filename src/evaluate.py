import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def evaluate(model  = "../results/manual/mlp",
             input  = "../results/manual/internal_states/evaluation",
             ):
    
    # Configuration
    model = Path(model)
    input = Path(input)
    pipeline = joblib.load(model / "mlp.joblib")
    meta = json.loads((model / "meta.json").read_text())
    levels = meta["classes"]
    level_map = {level: i for i, level in enumerate(levels)}
    all_files = list(input.glob("*/*.npy"))
    X_test = np.stack([np.load(f) for f in all_files])
    y_test_true = np.array([level_map[f.parent.name] for f in all_files])
    
    # Evaluate on evaluation dataset
    y_test_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test_true, y_test_pred)
    within1_accuracy = np.mean(np.abs(y_test_true - y_test_pred) <= 1)
    report = classification_report(y_test_true, y_test_pred, target_names=levels)
    report_dict = classification_report(y_test_true, y_test_pred, target_names=levels, output_dict=True)
    report_dict['within1_accuracy'] = within1_accuracy
    pd.DataFrame(report_dict).round(4).transpose().to_csv(model / "evaluation_report.csv", index=True)
    
    # Print metrics
    print(f"Accuracy:\t {accuracy:.2%}")
    print(f"±1 Level:\t {within1_accuracy:.2%}")
    print("\nReport:")
    print(report)

    # Print confusion matrix
    cm = confusion_matrix(y_test_true, y_test_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=levels, yticklabels=levels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    output_figure_path = model / "confusion_matrix.pdf"
    plt.savefig(output_figure_path)

if __name__ == "__main__":
    evaluate()