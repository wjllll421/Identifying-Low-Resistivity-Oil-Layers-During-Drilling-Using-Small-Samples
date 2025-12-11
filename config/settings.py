# Reproducibility
RANDOM_STATE = 42

# Data
FEATURES = ['GR', 'TG', 'C1', 'C2', 'C3', 'iC4', 'nC4', 'iC5', 'nC5']
LABEL_COLUMN = '备注'
CLASS_NUM2NAME = {
    0: "Low-Resistivity Reservoir",
    1: "Normal Reservoir",
    2: "Water Layer"
}
TARGET_SAMPLES_PER_CLASS = 300

# KPCA
KPCA_N_COMPONENTS = 3
KPCA_GAMMA = 0.01

# PSO
PSO_POP_SIZE = 20      # Reduced for demo speed
PSO_MAX_ITER = 30      # Reduced for demo speed

# CatBoost
CATBOOST_DEFAULTS = {
    "depth": 6,
    "learning_rate": 0.1,
    "iterations": 50,
    "l2_leaf_reg": 3.0
}