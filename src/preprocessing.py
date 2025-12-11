import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import train_test_split
from config.settings import KPCA_N_COMPONENTS, KPCA_GAMMA, RANDOM_STATE

def prepare_data(X, y, use_kpca=True):
    """Standardize, optionally apply KPCA, and split data."""
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = le.classes_

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if use_kpca:
        kpca = KernelPCA(
            n_components=KPCA_N_COMPONENTS,
            kernel='rbf',
            gamma=KPCA_GAMMA,
            random_state=RANDOM_STATE
        )
        X_processed = kpca.fit_transform(X_scaled)
    else:
        X_processed = X_scaled

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_encoded, test_size=0.3,
        random_state=RANDOM_STATE, stratify=y_encoded
    )

    return X_train, X_test, y_train, y_test, class_names