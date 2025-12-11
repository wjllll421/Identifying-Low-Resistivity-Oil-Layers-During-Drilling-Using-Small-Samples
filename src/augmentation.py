import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from config.settings import RANDOM_STATE, TARGET_SAMPLES_PER_CLASS

def improved_smote(X, y, target_class, n_target=None, k=5):
    """
    Improved SMOTE with noise filtering for lithology data.
    """
    if n_target is None:
        n_target = TARGET_SAMPLES_PER_CLASS

    X = X.copy()
    y = y.copy()
    
    # Encode labels temporarily
    unique_classes = sorted(y.unique())
    label_to_int = {cls: i for i, cls in enumerate(unique_classes)}
    int_to_label = {v: k for k, v in label_to_int.items()}
    y_int = y.map(label_to_int)
    target_int = label_to_int[target_class]

    X_np = X.values
    y_np = y_int.values

    # Get minority samples
    minority_mask = (y_np == target_int)
    X_minority = X_np[minority_mask]
    n_minority = len(X_minority)

    if n_minority >= n_target:
        return X[y == target_class], y[y == target_class]

    n_synthetic = n_target - n_minority

    # Fit NN on full dataset
    nn = NearestNeighbors(n_neighbors=k+1, metric='euclidean')
    nn.fit(X_np)
    distances, indices = nn.kneighbors(X_minority)

    synthetic_samples = []
    np.random.seed(RANDOM_STATE)

    for _ in range(n_synthetic):
        idx = np.random.randint(0, n_minority)
        x_i = X_minority[idx]
        neighbors = indices[idx][1:]  # exclude self
        neighbor_labels = y_np[neighbors]

        # Only use neighbors of same class
        same_class_neighbors = neighbors[neighbor_labels == target_int]
        if len(same_class_neighbors) == 0:
            continue

        nn_idx = np.random.choice(same_class_neighbors)
        x_nn = X_np[nn_idx]

        # Generate synthetic sample
        diff = x_nn - x_i
        gap = np.random.rand()
        synthetic = x_i + gap * diff
        synthetic_samples.append(synthetic)

    if len(synthetic_samples) == 0:
        synthetic_df = pd.DataFrame(columns=X.columns)
        synthetic_y = pd.Series([], dtype=y.dtype)
    else:
        synthetic_df = pd.DataFrame(synthetic_samples, columns=X.columns)
        synthetic_y = pd.Series([target_class] * len(synthetic_samples), dtype=y.dtype)

    # Combine original + synthetic
    original_minority = X[y == target_class]
    combined_X = pd.concat([original_minority, synthetic_df], ignore_index=True)
    combined_y = pd.concat([y[y == target_class], synthetic_y], ignore_index=True)

    return combined_X, combined_y