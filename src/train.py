import numpy as np
import json
import joblib
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def train(
    input       = "../results/manual/internal_states/training",
    output      = "../results/manual/mlp",
    seed        = 1,
    crossval    = 5):

    # Configuration
    input = Path(input)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    levels = sorted([d.name for d in input.iterdir() if d.is_dir()]) # = ["A1","A2","B1","B2","C1","C2"]
    level_map = {level: i for i, level in enumerate(levels)} # Map Levels to numbers (0-5)

    # Load X (internal states) and Y (CEFR labels)
    all_files = list(input.glob("*/*.npy"))
    X_train = np.stack([np.load(f) for f in all_files])
    y_train = np.array([level_map[f.parent.name] for f in all_files])

    # Create Pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(random_state=seed, early_stopping=True, max_iter=100))
    ])

    # Grid search
    param_grid = {
    'mlp__hidden_layer_sizes': [(512, 256, 128),
                                (1024, 512, 256),
                                (512, 256, 128, 64),
                                (1024, 512, 256, 128, 64)],
    'mlp__learning_rate_init': [ 0.001, 0.0005, 0.0001],
    'mlp__alpha': [0.0001, 0.001, 0.01]}

    search = GridSearchCV(pipeline, param_grid, cv=crossval, n_jobs=-1, verbose=0)
    search.fit(X_train, y_train)
    
    # Results
    print(f"parameters:\t{search.best_params_}")
    print(f"accuracy:\t{search.best_score_:.4f}")
    best_model=search.best_estimator_
    joblib.dump(best_model, output / f"mlp.joblib") # Save best performing model
    json.dump({"classes": levels}, (output / "meta.json").open("w")) # Save Classes

    # Plot training loss curve
    loss_curve = best_model.named_steps['mlp'].loss_curve_
    plt.figure(figsize=(10, 6))
    plt.plot(loss_curve)
    plt.title("Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(output / "training_loss.pdf")
    plt.close()

    # Plot hyperparameter performance
    hp_results = pd.DataFrame(search.cv_results_).sort_values(by='mean_test_score').tail(10)
    param_labels = [str(p) for p in hp_results['params']] # Hyperparameter combinations
    
    min, max = hp_results['mean_test_score'].min(), hp_results['mean_test_score'].max()
    x_min, x_max = min - ((max - min) * 0.2), max + ((max - min) * 0.2) # For scaling
    
    plt.figure(figsize=(12, 8))
    plt.barh(param_labels, hp_results['mean_test_score'])
    plt.xlim(left=x_min, right=x_max) # Zoom in to make differences visible
    plt.xlabel("Mean Accuracy")
    plt.ylabel("Hyperparameters")
    plt.grid(axis='x')
    plt.tight_layout()
    plt.savefig(output / "hyperparameters.pdf")
    plt.close()

    # 3D hyperparameter visualization
    results_df = pd.DataFrame(search.cv_results_)
    params_df = pd.json_normalize(results_df['params'])
    results_df = pd.concat([results_df, params_df], axis=1)

    # Create integer mappings for all hyperparameters
    hls_str = results_df['mlp__hidden_layer_sizes'].astype(str) # Hidden layer sizes
    hls_map = {val: i for i, val in enumerate(sorted(hls_str.unique()))}
    results_df['hls_numeric'] = hls_str.map(hls_map)
    lr_map = {val: i for i, val in enumerate(sorted(results_df['mlp__learning_rate_init'].unique()))} # Learning rate
    results_df['lr_numeric'] = results_df['mlp__learning_rate_init'].map(lr_map)
    alpha_map = {val: i for i, val in enumerate(sorted(results_df['mlp__alpha'].unique()))} # Alpha
    results_df['alpha_numeric'] = results_df['mlp__alpha'].map(alpha_map)

    # Scale point size proportional to accuracy
    min_acc = results_df['mean_test_score'].min()
    max_acc = results_df['mean_test_score'].max()
    min_size = 150
    max_size = 3000
    if (max_acc - min_acc) > 0:
        results_df['point_size'] = min_size + ((results_df['mean_test_score'] - min_acc) / (max_acc - min_acc)) * (max_size - min_size)
    else:
        results_df['point_size'] = (min_size + max_size) / 2

    # Create 3D plot
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(
        results_df['hls_numeric'],
        results_df['lr_numeric'],
        results_df['alpha_numeric'],
        c=results_df['mean_test_score'],
        s=results_df['point_size'],
        cmap='viridis',
        alpha=0.8,
        edgecolor='k'
    )

    # Set label and scale for all three axes
    ax.set_xticks(list(hls_map.values()))
    ax.set_xticklabels(list(hls_map.keys()), rotation=20, ha='right', fontsize=9)

    ax.set_ylabel('Learning Rate', labelpad=15)
    ax.set_yticks(list(lr_map.values()))
    ax.set_yticklabels(list(lr_map.keys()))

    ax.set_zlabel('Alpha', labelpad=10)
    ax.set_zticks(list(alpha_map.values()))
    ax.set_zticklabels(list(alpha_map.keys()))

    ax.set_title('Grid Search Hyperparameter Performance', pad=20, fontsize=16)
    cbar = fig.colorbar(scatter, shrink=0.6, aspect=20)
    cbar.set_label('Mean Test Score (Accuracy)', labelpad = 5)

    plt.savefig(output / "hyperparameter_3d_visualization.pdf")
    plt.close()

if __name__ == "__main__":
    train()