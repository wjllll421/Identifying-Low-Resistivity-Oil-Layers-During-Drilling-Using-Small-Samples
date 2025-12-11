import numpy as np
import pandas as pd
from config.settings import FEATURES, CLASS_NUM2NAME

def generate_synthetic_well_logging_data(n_samples_per_class=100, random_state=42):
    """
    Generate synthetic well logging data for reproducibility.
    Mimics typical GR, TG, hydrocarbon features for three classes.
    """
    np.random.seed(random_state)
    classes = list(CLASS_NUM2NAME.keys())
    data_frames = []

    for cls in classes:
        n = n_samples_per_class
        if cls == 0:  # Low-resistivity oil
            GR = np.clip(np.random.normal(65, 8, n), 40, 90)
            TG = np.clip(np.random.normal(0.85, 0.1, n), 0.6, 1.2)
            C1 = np.clip(np.random.normal(75, 6, n), 50, 100)
        elif cls == 1:  # Normal oil
            GR = np.clip(np.random.normal(45, 7, n), 30, 70)
            TG = np.clip(np.random.normal(0.45, 0.06, n), 0.3, 0.7)
            C1 = np.clip(np.random.normal(88, 4, n), 70, 100)
        else:  # Water layer
            GR = np.clip(np.random.normal(85, 10, n), 60, 110)
            TG = np.clip(np.random.normal(1.15, 0.12, n), 0.9, 1.5)
            C1 = np.clip(np.random.normal(25, 8, n), 5, 50)

        # Derived gas components (physically plausible ratios)
        C2 = C1 * np.random.uniform(0.04, 0.08, n)
        C3 = C2 * np.random.uniform(0.15, 0.35, n)
        iC4 = C3 * np.random.uniform(0.25, 0.45, n)
        nC4 = iC4 * np.random.uniform(0.9, 1.3, n)
        iC5 = C3 * np.random.uniform(0.08, 0.18, n)
        nC5 = iC5 * np.random.uniform(0.85, 1.15, n)

        df_cls = pd.DataFrame({
            'GR': GR, 'TG': TG, 'C1': C1, 'C2': C2, 'C3': C3,
            'iC4': iC4, 'nC4': nC4, 'iC5': iC5, 'nC5': nC5,
            '备注': CLASS_NUM2NAME[cls]
        })
        data_frames.append(df_cls)

    full_df = pd.concat(data_frames, ignore_index=True)
    return full_df

if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    df = generate_synthetic_well_logging_data(n_samples_per_class=100)
    output_path = "data/synthetic_data.xlsx"
    df.to_excel(output_path, index=False)
    print(f"✅ Synthetic dataset saved to {output_path} (shape: {df.shape})")