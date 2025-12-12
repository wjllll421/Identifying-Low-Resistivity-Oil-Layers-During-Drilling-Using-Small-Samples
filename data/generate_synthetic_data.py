import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
import os
import numpy as np
import pandas as pd
from config.settings import FEATURES, CLASS_NUM2NAME

def generate_synthetic_well_logging_data(n_samples_per_class=100, random_state=42):
    """
    Generate synthetic well logging data with:
      - '备注': Chinese class name (for readability)
      - '结论': numeric label (for model training: 0, 1, 2)
    """
    np.random.seed(random_state)
    classes = list(CLASS_NUM2NAME.keys())  # [0, 1, 2]
    data_frames = []

    class_params = {
        0: {  # 低阻油
            'GR': (69.327, 10.0),
            'TG': (34.731, 25.0),
            'C1': (10.584, 12.0),
            'C2': (0.423, 0.6),
            'C3': (0.065, 0.08),
            'iC4': (0.024, 0.05),
            'nC4': (0.045, 0.15), 
            'iC5': (0.074, 0.08),
            'nC5': (0.016, 0.02),
        },
        1: {  # 正常油
            'GR': (67.318, 6.0),
            'TG': (34.731, 20.0),
            'C1': (19.710, 15.0),
            'C2': (1.086, 0.8),
            'C3': (0.169, 0.1),
            'iC4': (0.062, 0.05),
            'nC4': (0.058, 0.05),
            'iC5': (0.044, 0.03),
            'nC5': (0.012, 0.01),
        },
        2: {  # 水层
            'GR': (62.129, 5.5),
            'TG': (4.259, 1.5),
            'C1': (3.338, 1.2),
            'C2': (0.040, 0.04),
            'C3': (0.006, 0.005),
            'iC4': (0.003, 0.004),
            'nC4': (0.002, 0.004),
            'iC5': (0.005, 0.005),
            'nC5': (0.0004, 0.0005),
        }
    }

    for cls in classes:
        n = n_samples_per_class
        df_cls = pd.DataFrame()


        for feat in FEATURES:
            if feat in class_params[cls]:
                mean, std = class_params[cls][feat]
                values = np.random.normal(mean, std, n)

                values = np.clip(values, 0, None)
                df_cls[feat] = values
            else:
                
                df_cls[feat] = np.zeros(n)

        df_cls['备注'] = CLASS_NUM2NAME[cls]      
        df_cls['结论'] = cls                       

        data_frames.append(df_cls)

    full_df = pd.concat(data_frames, ignore_index=True)
    return full_df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_synthetic_well_logging_data(n_samples_per_class=100)
    output_path = "data/synthetic_data.xlsx"
    df.to_excel(output_path, index=False)
    print(f"✅ Synthetic dataset saved to {output_path} (shape: {df.shape})")