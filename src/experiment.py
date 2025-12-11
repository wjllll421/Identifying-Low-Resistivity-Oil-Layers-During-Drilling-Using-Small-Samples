import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from catboost import CatBoostClassifier
from config.settings import CLASS_NUM2NAME, CATBOOST_DEFAULTS, RANDOM_STATE

CLASS_NUMS = list(CLASS_NUM2NAME.keys())
CLASS_LABELS = [CLASS_NUM2NAME[i] for i in CLASS_NUMS]

def run_single_experiment(X_train, X_test, y_train, y_test, use_pso=False, lb=None, ub=None):
    if use_pso and lb is not None and ub is not None:
        from src.optimizer import pso_optimize_catboost
        # Simple validation split for PSO
        from sklearn.model_selection import train_test_split
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=RANDOM_STATE)
        best_params_arr, _ = pso_optimize_catboost(X_tr, y_tr, X_val, y_val, lb, ub)
        params = {
            "depth": int(round(best_params_arr[0])),
            "learning_rate": best_params_arr[1],
            "iterations": int(round(best_params_arr[2])),
            "l2_leaf_reg": best_params_arr[3]
        }
    else:
        params = CATBOOST_DEFAULTS

    model = CatBoostClassifier(
        **{k: v for k, v in params.items()},
        loss_function='MultiClass',
        verbose=0,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test).flatten()
    y_true = y_test.flatten()

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')

    # Class-wise metrics
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_NUMS)
    precisions = precision_score(y_true, y_pred, labels=CLASS_NUMS, average=None, zero_division=0)
    recalls = recall_score(y_true, y_pred, labels=CLASS_NUMS, average=None, zero_division=0)
    f1s = f1_score(y_true, y_pred, labels=CLASS_NUMS, average=None, zero_division=0)

    class_metrics = {}
    for i, cls in enumerate(CLASS_NUMS):
        cls_acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
        class_metrics[cls] = {
            "类别名称": CLASS_NUM2NAME[cls],
            "准确率": round(cls_acc, 4),
            "精确率": round(precisions[i], 4),
            "召回率": round(recalls[i], 4),
            "F1值": round(f1s[i], 4)
        }

    return {
        "整体准确率": round(acc, 4),
        "Macro-F1": round(f1_macro, 4),
        "模型参数": str(params),
        "类别级指标": class_metrics,
        "y_test": y_true,
        "y_pred": y_pred
    }

def save_results_to_excel(results, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    df_main = pd.DataFrame([
        {
            "消融组": r["group"],
            "是否使用KPCA": r["use_kpca"],
            "是否使用PSO": r["use_pso"],
            "整体准确率": r["result"]["整体准确率"],
            "Macro-F1": r["result"]["Macro-F1"],
            "模型参数": r["result"]["模型参数"]
        }
        for r in results
    ])

    # Expand class metrics
    for cls in CLASS_NUMS:
        cls_name = CLASS_NUM2NAME[cls]
        df_main[f"{cls_name}_准确率"] = [r["result"]["类别级指标"][cls]["准确率"] for r in results]
        df_main[f"{cls_name}_精确率"] = [r["result"]["类别级指标"][cls]["精确率"] for r in results]
        df_main[f"{cls_name}_召回率"] = [r["result"]["类别级指标"][cls]["召回率"] for r in results]
        df_main[f"{cls_name}_F1值"] = [r["result"]["类别级指标"][cls]["F1值"] for r in results]

    df_main.to_excel(os.path.join(save_dir, "Ablation_Results.xlsx"), index=False)

def plot_results(results, save_dir="results"):
    groups = [r["group"] for r in results]
    accs = [r["result"]["整体准确率"] for r in results]
    f1s = [r["result"]["Macro-F1"] for r in results]

    x = np.arange(len(groups))
    plt.figure(figsize=(8, 5))
    plt.bar(x - 0.2, accs, 0.4, label='Accuracy', color='#1f77b4')
    plt.bar(x + 0.2, f1s, 0.4, label='Macro-F1', color='#ff7f0e')
    plt.xticks(x, groups)
    plt.ylim(0.7, 1.0)
    plt.ylabel('Score')
    plt.legend()
    plt.title('Ablation Study Results')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Ablation_Results.png"), dpi=300)
    plt.close()