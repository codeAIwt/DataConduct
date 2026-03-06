#!/usr/bin/env python3
# coding: utf-8
"""
最终稳定版 XGBoost.py（无 early_stopping_rounds，随机搜索 n_iter=2）
适用于 Allstate Claims Severity Kaggle 数据集
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import joblib
import time

# ---------------------------
# 配置路径
# ---------------------------
DATA_DIR = r"D:\WorkSpace\work_develop\AI_Project\DataBase"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
OUT_DIR = Path(__file__).resolve().parent

RANDOM_STATE = 42

# ---------------------------
# 工具函数
# ---------------------------
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def load_data():
    print(f"[{time.ctime()}] 读取数据...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    print(f"  train shape: {train.shape}, test shape: {test.shape}")
    return train, test

def basic_feature_engineering(train, test):
    print(f"[{time.ctime()}] 做基础特征工程...")
    id_col = "id"
    y = train["loss"].values
    y_trans = np.log1p(y)

    X_train = train.drop(columns=[id_col, "loss"])
    X_test = test.drop(columns=[id_col])

    cat_cols = [c for c in X_train.columns if c.startswith("cat")]
    cont_cols = [c for c in X_train.columns if c not in cat_cols]
    print(f"  categorical cols: {len(cat_cols)}, continuous cols: {len(cont_cols)}")

    for c in cont_cols:
        med = X_train[c].median()
        X_train[c].fillna(med, inplace=True)
        X_test[c].fillna(med, inplace=True)

    for c in cat_cols:
        X_train[c] = X_train[c].fillna("missing").astype(str)
        X_test[c] = X_test[c].fillna("missing").astype(str)
        codes, uniques = pd.factorize(X_train[c])
        X_train[c] = codes
        mapping = {val: i for i, val in enumerate(uniques)}
        X_test[c] = X_test[c].map(mapping).fillna(-1).astype(int)

    print(f"  total features: {len(X_train.columns)}")
    return X_train, y_trans, X_test, test[id_col]

def train_xgb_with_random_search(X, y):
    print(f"[{time.ctime()}] 开始超参数随机搜索 (n_iter=2)...")
    model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=RANDOM_STATE,
        verbosity=0,
        n_jobs=1
    )
    param_dist = {
        "n_estimators": [100, 150, 200],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0]
    }

    rand = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=2,
        scoring="neg_mean_squared_error",
        cv=2,
        verbose=0,
        random_state=RANDOM_STATE,
        n_jobs=1
    )

    rand.fit(X, y)
    print("随机搜索完成。最佳参数：")
    print(rand.best_params_)
    return rand.best_estimator_, rand

def evaluate_and_predict(model, X_train, y_train, X_test, id_test):
    print(f"[{time.ctime()}] 训练最终模型并评估...")

    # 手动划分验证集，旧版兼容
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=RANDOM_STATE
    )

    model.fit(X_tr, y_tr, verbose=False)

    pred_val = model.predict(X_val)
    val_rmse = rmse(np.expm1(y_val), np.expm1(pred_val))
    print(f"Validation RMSE (orig scale): {val_rmse:.4f}")

    preds_test = np.expm1(model.predict(X_test))
    submission = pd.DataFrame({id_test.name: id_test.values, "loss": preds_test})
    return submission, val_rmse

# ---------------------------
# 主程序
# ---------------------------
def main():
    train_df, test_df = load_data()
    X_train, y_train_log, X_test, id_test = basic_feature_engineering(train_df, test_df)

    print(f"[{time.ctime()}] 训练 baseline 模型...")
    baseline = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        tree_method="hist",
        n_jobs=1,
        verbosity=0
    )
    baseline.fit(X_train, y_train_log)
    base_rmse = rmse(np.expm1(y_train_log), np.expm1(baseline.predict(X_train)))
    print(f"Baseline train RMSE (orig scale): {base_rmse:.4f}")

    best_model, rand_search = train_xgb_with_random_search(X_train, y_train_log)
    submission_df, val_rmse = evaluate_and_predict(best_model, X_train, y_train_log, X_test, id_test)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    submission_path = OUT_DIR / f"submission_{timestamp}.csv"
    submission_df.to_csv(submission_path, index=False)
    joblib.dump(best_model, OUT_DIR / f"xgb_model_{timestamp}.joblib")
    joblib.dump(rand_search, OUT_DIR / f"xgb_randsearch_{timestamp}.joblib")

    print(f"提交文件已保存 -> {submission_path}")
    print(f"最终 Validation RMSE (orig scale): {val_rmse:.4f}")
    print("运行结束。请将提交文件上传至 Kaggle 进行正式评分。")

if __name__ == "__main__":
    main()
