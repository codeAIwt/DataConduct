# 集成学习模型训练与应用实战 — XGBoost（Allstate Claims Severity）

## 一、任务说明
- 使用 Kaggle 的 **Allstate - Claims Severity** 数据集（train.csv / test.csv）
- 构建 XGBoost 回归模型，进行特征工程与超参数调优
- 在 test.csv 上生成预测并导出提交文件（submission.csv）
- **注意**：该问题为回归问题（目标列 `loss` 为连续值）。因此使用 **RMSE** 作为评估指标（不是分类准确率）。

## 二、文件与路径
- 数据：`D:\WorkSpace\work_develop\py_人工智能\DataBase\train.csv` 和 `... \test.csv`
- 代码：`D:\WorkSpace\work_develop\py_人工智能\xgboost.py`
- 依赖列表：`requirements.txt`
- 说明文档：`README.md`
- 输出（模型与提交文件）会被保存到与脚本相同目录（例如 `submission_YYYYMMDD_HHMMSS.csv`）

## 三、主要步骤（脚本实现概述）
1. **读取数据**：载入 train / test（脚本开头可以修改路径）
2. **目标变换**：对 `loss` 使用 `log1p` 变换（减小偏度，常用技巧）
3. **特征工程**：
   - 将以 `cat` 为前缀的列视为类别特征
   - 类别缺失值填 `'missing'` 并用 `factorize`（label 编码）
   - 连续特征缺失用中位数填补
4. **模型**：
   - 使用 `XGBRegressor(objective='reg:squarederror')`
   - 先训练一个 baseline 模型（固定参数，快速得到基准）
   - 使用 `RandomizedSearchCV` 对若干超参数（`n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`）进行随机搜索（默认 20 次）
5. **评估**：
   - 在训练集上做 10% 的 holdout 验证，计算 **原始尺度**下的 RMSE（对预测结果做 `expm1` 反变换）
6. **输出**：
   - 将对 test 的预测保存为 `submission_YYYYMMDD_HHMMSS.csv`（包含 `id` 与 `loss` 列）
   - 保存训练好的模型与随机搜索对象为 joblib 文件

## 四、如何运行
1. 安装依赖：
   ```bash
   pip install -r requirements.txt
