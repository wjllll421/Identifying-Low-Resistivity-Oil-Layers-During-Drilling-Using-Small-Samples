import os
import pandas as pd
from data.generate_synthetic_data import generate_synthetic_well_logging_data
from src.augmentation import improved_smote
from src.preprocessing import prepare_data
from src.experiment import run_single_experiment, save_results_to_excel, plot_results
from config.settings import FEATURES, LABEL_COLUMN, TARGET_SAMPLES_PER_CLASS

def main():
    # Step 1: Load or generate data
    data_path = "data/synthetic_data.xlsx"
    if not os.path.exists(data_path):
        print("Generating synthetic data...")
        df = generate_synthetic_well_logging_data(n_samples_per_class=100)
        os.makedirs("data", exist_ok=True)
        df.to_excel(data_path, index=False)
    else:
        df = pd.read_excel(data_path)

    print(f"Original data shape: {df.shape}")

    # Step 2: Augment each class to 300 samples
    X_list, y_list = [], []
    for cls in df[LABEL_COLUMN].unique():
        X_cls, y_cls = improved_smote(df[FEATURES], df[LABEL_COLUMN], cls)
        X_list.append(X_cls)
        y_list.append(y_cls)
    df_balanced = pd.concat(X_list + y_list, axis=1)
    print(f"Balanced data shape: {df_balanced.shape}")

    # Step 3: Prepare datasets
    X_no_kpca, y_no_kpca = df_balanced[FEATURES], df_balanced[LABEL_COLUMN]
    X_with_kpca, y_with_kpca = df_balanced[FEATURES], df_balanced[LABEL_COLUMN]

    # Without KPCA
    X_train_nok, X_test_nok, y_train_nok, y_test_nok, _ = prepare_data(X_no_kpca, y_no_kpca, use_kpca=False)
    # With KPCA
    X_train_kpca, X_test_kpca, y_train_kpca, y_test_kpca, _ = prepare_data(X_with_kpca, y_with_kpca, use_kpca=True)

    # PSO bounds
    lb = [2, 0.01, 10, 1]
    ub = [10, 1.0, 200, 10]

    # Step 4: Run ablation groups
    results = []
    results.append({
        "group": "A",
        "use_kpca": False,
        "use_pso": False,
        "result": run_single_experiment(X_train_nok, X_test_nok, y_train_nok, y_test_nok, use_pso=False)
    })
    results.append({
        "group": "B",
        "use_kpca": False,
        "use_pso": True,
        "result": run_single_experiment(X_train_nok, X_test_nok, y_train_nok, y_test_nok, use_pso=True, lb=lb, ub=ub)
    })
    results.append({
        "group": "C",
        "use_kpca": True,
        "use_pso": False,
        "result": run_single_experiment(X_train_kpca, X_test_kpca, y_train_kpca, y_test_kpca, use_pso=False)
    })
    results.append({
        "group": "D",
        "use_kpca": True,
        "use_pso": True,
        "result": run_single_experiment(X_train_kpca, X_test_kpca, y_train_kpca, y_test_kpca, use_pso=True, lb=lb, ub=ub)
    })

    # Step 5: Save and visualize
    save_results_to_excel(results, "results")
    plot_results(results, "results")
    print("✅ Ablation study completed. Results saved in 'results/' folder.")

if __name__ == "__main__":
    main()