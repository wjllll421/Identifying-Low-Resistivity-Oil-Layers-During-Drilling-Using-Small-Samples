import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
from config.settings import RANDOM_STATE

def pso_optimize_catboost(X_train, y_train, X_val, y_val, lb, ub, pop_size=20, max_iter=30):
    dim = len(lb)
    np.random.seed(RANDOM_STATE)
    pop_pos = np.random.uniform(lb, ub, (pop_size, dim))
    pop_vel = np.random.uniform(-1, 1, (pop_size, dim))
    p_best_pos = pop_pos.copy()
    p_best_score = np.full(pop_size, -np.inf)
    g_best_pos = pop_pos[0].copy()
    g_best_score = -np.inf

    w_start, w_end = 0.9, 0.4

    for iter in range(max_iter):
        w = w_start - (w_start - w_end) * (iter / max_iter)
        for i in range(pop_size):
            params = {
                "depth": int(round(pop_pos[i, 0])),
                "learning_rate": pop_pos[i, 1],
                "iterations": int(round(pop_pos[i, 2])),
                "l2_leaf_reg": pop_pos[i, 3]
            }
            model = CatBoostClassifier(
                **params,
                loss_function='MultiClass',
                verbose=0,
                random_state=RANDOM_STATE
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            score = f1_score(y_val, y_pred, average='macro')

            if score > p_best_score[i]:
                p_best_score[i] = score
                p_best_pos[i] = pop_pos[i].copy()
            if score > g_best_score:
                g_best_score = score
                g_best_pos = pop_pos[i].copy()

        r1, r2 = np.random.rand(dim), np.random.rand(dim)
        pop_vel = w * pop_vel + 2.0 * r1 * (p_best_pos - pop_pos) + 2.0 * r2 * (g_best_pos - pop_pos)
        pop_pos = pop_pos + pop_vel
        pop_pos = np.clip(pop_pos, lb, ub)

    return g_best_pos, g_best_score