# functions.py
import os
import re
import ast
import json
import textwrap
import tempfile
import subprocess
from typing import List, Dict, Tuple, Optional

import pandas as pd
import openai

# ---- OpenAI client (use env var for security) ----
# Set an environment variable: OPENAI_API_KEY=...
openai_client = openai.OpenAI(api_key="")

# ==================================================
# Prompt builders (your existing ones, lightly kept)
# ==================================================

def _kvpairs(params: dict, keys: List[str]) -> str:
    out = []
    for k in keys:
        v = params.get(k)
        if v is None:
            continue
        if isinstance(v, str):
            out.append(f"{k}='{v}'")
        else:
            out.append(f"{k}={v}")
    return ", ".join(out)

def get_model_info(model_name: str) -> str:
    prompt = f"Explain what the {model_name} model is in 3-4 sentences. Keep it simple and practical."
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert ML tutor."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    try:
        return response.choices[0].message.content.strip()
    except Exception:
        return response.choices[0].message.get("content", "").strip()

def analyze_csv(file_path: str):
    df = pd.read_csv(file_path)
    return df.columns.tolist()

def generate_prompt(model_name: str, params: dict, dep_vars: List[str], indep_vars: List[str], csv_path: str) -> str:
    if model_name == "XGBoost":
        return generate_xgboost_prompt(params, dep_vars, indep_vars, csv_path)
    elif model_name == "LinearRegression":
        return generate_linear_regression_prompt(params, dep_vars, indep_vars, csv_path)
    elif model_name == "Ridge":
        return generate_ridge_prompt(params, dep_vars, indep_vars, csv_path)
    elif model_name == "Lasso":
        return generate_lasso_prompt(params, dep_vars, indep_vars, csv_path)
    elif model_name == "ElasticNet":
        return generate_elasticnet_prompt(params, dep_vars, indep_vars, csv_path)
    elif model_name == "XGBClassifier":
        return generate_xgb_classifier_prompt(params, dep_vars, indep_vars, csv_path)
    elif model_name == "RandomForestClassifier":
        return generate_random_forest_classifier_prompt(params, dep_vars, indep_vars, csv_path)
    elif model_name == "LogisticRegression":
        return generate_logreg_classifier_prompt(params, dep_vars, indep_vars, csv_path)
    elif model_name == "SVC":
        return generate_svc_classifier_prompt(params, dep_vars, indep_vars, csv_path)
    else:
        raise ValueError("Unsupported model selected")

# -------------------------
# Regression prompt builders
# -------------------------
def generate_xgboost_prompt(params, dep_vars, indep_vars, csv_path):
    dep_str = ", ".join(dep_vars)
    indep_str = ", ".join(indep_vars)
    if params.get("n_estimators") is None and params.get("epochs") is not None:
        params["n_estimators"] = params.get("epochs")
    param_str = _kvpairs(
        params,
        ["learning_rate", "n_estimators", "max_depth", "min_child_weight", "subsample", "colsample_bytree", "gamma"],
    )
    prompt = f"""
Write complete Python code to:
- Load data from '{csv_path}'
- Use columns [{indep_str}] as features and [{dep_str}] as target(s)
- For any missing (NaN) values in the dataset:
    - Replace missing values in numeric columns with the column's mean using df[col] = df[col].fillna(df[col].mean())
- Randomly split the cleaned dataset into 80% training and 20% testing using scikit-learn
- Use the XGBRegressor from the xgboost library for regression
- Initialize the model with: {param_str}
- Train the model on the training data
- After testing, calculate and print:
    - RMSE using root_mean_squared_error from sklearn.metrics
    - R2 Score
    - NRMSE (Normalized RMSE = RMSE divided by the range of actual target values)
- Save the trained model using joblib (not sklearn.externals)

Only return a single Python code block without extra explanation.
"""
    return prompt

def generate_linear_regression_prompt(params, dep_vars, indep_vars, csv_path):
    dep_str = ", ".join(dep_vars)
    indep_str = ", ".join(indep_vars)
    param_str = _kvpairs(params, ["fit_intercept", "copy_X", "positive", "n_jobs"]) or " "
    prompt = f"""
Write complete Python code to:
- Load data from '{csv_path}'
- Use columns [{indep_str}] as features and [{dep_str}] as target(s)
- For any missing (NaN) values in the dataset:
    - Replace missing values in numeric columns with the column's mean using df[col] = df[col].fillna(df[col].mean())
- Randomly split the cleaned dataset into 80% training and 20% testing using scikit-learn
- Use LinearRegression from sklearn.linear_model
- Initialize the model with: {param_str}
- Train the model on the training data
- After testing, calculate and print:
    - RMSE using root_mean_squared_error from sklearn.metrics
    - R2 Score
    - NRMSE (Normalized RMSE = RMSE divided by the range of actual target values)
- Save the trained model using joblib (not sklearn.externals)

Only return a single Python code block without extra explanation.
"""
    return prompt

def generate_ridge_prompt(params, dep_vars, indep_vars, csv_path):
    dep_str = ", ".join(dep_vars)
    indep_str = ", ".join(indep_vars)
    param_str = _kvpairs(params, ["alpha", "fit_intercept", "solver", "max_iter", "tol", "positive", "random_state"])
    prompt = f"""
Write complete Python code to:
- Load data from '{csv_path}'
- Use columns [{indep_str}] as features and [{dep_str}] as target(s)
- For any missing (NaN) values in the dataset:
    - Replace missing values in numeric columns with the column's mean using df[col] = df[col].fillna(df[col].mean())
- Randomly split the cleaned dataset into 80% training and 20% testing using scikit-learn
- Use Ridge from sklearn.linear_model
- Initialize the model with: {param_str}
- Train the model on the training data
- After testing, calculate and print:
    - RMSE using root_mean_squared_error from sklearn.metrics
    - R2 Score
    - NRMSE (Normalized RMSE = RMSE divided by the range of actual target values)
- Save the trained model using joblib (not sklearn.externals)

Only return a single Python code block without extra explanation.
"""
    return prompt

def generate_lasso_prompt(params, dep_vars, indep_vars, csv_path):
    dep_str = ", ".join(dep_vars)
    indep_str = ", ".join(indep_vars)
    param_str = _kvpairs(params, ["alpha", "fit_intercept", "max_iter", "tol", "selection", "positive", "random_state"])
    prompt = f"""
Write complete Python code to:
- Load data from '{csv_path}'
- Use columns [{indep_str}] as features and [{dep_str}] as target(s)
- For any missing (NaN) values in the dataset:
    - Replace missing values in numeric columns with the column's mean using df[col] = df[col].fillna(df[col].mean())
- Randomly split the cleaned dataset into 80% training and 20% testing using scikit-learn
- Use Lasso from sklearn.linear_model
- Initialize the model with: {param_str}
- Train the model on the training data
- After testing, calculate and print:
    - RMSE using root_mean_squared_error from sklearn.metrics
    - R2 Score
    - NRMSE (Normalized RMSE = RMSE divided by the range of actual target values)
- Save the trained model using joblib (not sklearn.externals)

Only return a single Python code block without extra explanation.
"""
    return prompt

def generate_elasticnet_prompt(params, dep_vars, indep_vars, csv_path):
    dep_str = ", ".join(dep_vars)
    indep_str = ", ".join(indep_vars)
    param_str = _kvpairs(params, ["alpha", "l1_ratio", "fit_intercept", "max_iter", "tol", "selection", "positive", "random_state"])
    prompt = f"""
Write complete Python code to:
- Load data from '{csv_path}'
- Use columns [{indep_str}] as features and [{dep_str}] as target(s)
- For any missing (NaN) values in the dataset:
    - Replace missing values in numeric columns with the column's mean using df[col] = df[col].fillna(df[col].mean())
- Randomly split the cleaned dataset into 80% training and 20% testing using scikit-learn
- Use ElasticNet from sklearn.linear_model
- Initialize the model with: {param_str}
- Train the model on the training data
- After testing, calculate and print:
    - RMSE using root_mean_squared_error from sklearn.metrics
    - R2 Score
    - NRMSE (Normalized RMSE = RMSE divided by the range of actual target values)
- Save the trained model using joblib (not sklearn.externals)

Only return a single Python code block without extra explanation.
"""
    return prompt

# -----------------------------
# Classification prompt builders
# -----------------------------
def generate_xgb_classifier_prompt(params, dep_vars, indep_vars, csv_path):
    dep_str = ", ".join(dep_vars)
    indep_str = ", ".join(indep_vars)
    param_str = _kvpairs(
        params,
        [
            "learning_rate", "n_estimators", "max_depth", "min_child_weight",
            "subsample", "colsample_bytree", "gamma", "reg_alpha", "reg_lambda"
        ],
    )
    prompt = f"""
Write complete Python code to:
- Import: numpy as np, pandas as pd, joblib; from sklearn.model_selection import train_test_split; from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score; from xgboost import XGBClassifier
- Load data from '{csv_path}'
- Use columns [{indep_str}] as features (X) and [{dep_str}] as the classification target (y)
- For any missing (NaN) values in numeric columns, fill with column mean (df[col] = df[col].fillna(df[col].mean()))
- Perform an 80/20 **stratified** train/test split with scikit-learn
- Initialize XGBClassifier with: {param_str}
- Fit the model
- Evaluate on the test set:
    * y_pred = model.predict(X_test)
    * Print Accuracy, Precision (macro), Recall (macro), F1-score (macro), and Confusion Matrix
    * If the model supports predict_proba:
        - proba = model.predict_proba(X_test)
        - classes = model.classes_
        - If len(classes) == 2:
            - pos_label = 1 if 1 in classes else max(classes)
            - pos_index = list(classes).index(pos_label)
            - roc = roc_auc_score(y_test, proba[:, pos_index])
          Else:
            - roc = roc_auc_score(y_test, proba, multi_class='ovr', average='macro')
        - Print ROC-AUC = roc
      Otherwise, print that ROC-AUC is unavailable.
- Save the trained model to disk with joblib (e.g., 'xgb_classifier_trained_model.joblib')
- Also print a short summary of metrics in a clean format

Only return a single Python code block without extra explanation.
"""
    return prompt

def generate_random_forest_classifier_prompt(params, dep_vars, indep_vars, csv_path):
    dep_str = ", ".join(dep_vars)
    indep_str = ", ".join(indep_vars)
    param_str = _kvpairs(
        params,
        ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_features", "bootstrap", "class_weight", "random_state"],
    )
    prompt = f"""
Write complete Python code to:
- Import: numpy as np, pandas as pd, joblib; from sklearn.model_selection import train_test_split; from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score; from sklearn.ensemble import RandomForestClassifier
- Load data from '{csv_path}'
- Use columns [{indep_str}] as features and [{dep_str}] as the classification target
- Impute numeric NaNs with the column mean
- Perform an 80/20 stratified train/test split
- Build a RandomForestClassifier with: {param_str}
- Train on the training set
- Evaluate on the test set and print:
    - Accuracy
    - Precision (macro), Recall (macro), F1-score (macro)
    - Confusion matrix
    - ROC-AUC:
        * If predict_proba is available:
            - proba = model.predict_proba(X_test)
            - classes = model.classes_
            - If len(classes) == 2:
                - pos_label = 1 if 1 in classes else max(classes)
                - pos_index = list(classes).index(pos_label)
                - roc = roc_auc_score(y_test, proba[:, pos_index])
              Else:
                - roc = roc_auc_score(y_test, proba, multi_class='ovr', average='macro')
            - Print ROC-AUC = roc
          Else: print that ROC-AUC is unavailable.
- Save the trained model with joblib (e.g., 'random_forest_trained_model.joblib')

Only return a single Python code block without extra explanation.
"""
    return prompt

def generate_logreg_classifier_prompt(params, dep_vars, indep_vars, csv_path):
    dep_str = ", ".join(dep_vars)
    indep_str = ", ".join(indep_vars)
    param_str = _kvpairs(
        params,
        ["penalty", "C", "solver", "max_iter", "tol", "class_weight", "fit_intercept", "n_jobs", "l1_ratio", "random_state"],
    )
    prompt = f"""
Write complete Python code to:
- Import: numpy as np, pandas as pd, joblib; from sklearn.model_selection import train_test_split; from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score; from sklearn.pipeline import Pipeline; from sklearn.preprocessing import StandardScaler; from sklearn.linear_model import LogisticRegression
- Load data from '{csv_path}'
- Use columns [{indep_str}] as features and [{dep_str}] as the classification target
- Impute numeric NaNs with column mean
- Perform an 80/20 stratified train/test split
- Build a scikit-learn Pipeline with StandardScaler() followed by LogisticRegression initialized with: {param_str}
- Train the pipeline
- Evaluate and print:
    - Accuracy
    - Precision (macro), Recall (macro), F1-score (macro)
    - Confusion matrix
    - ROC-AUC:
        * If the final estimator supports predict_proba:
            - proba = pipeline.predict_proba(X_test)
            - classes = pipeline.classes_
            - If len(classes) == 2:
                - pos_label = 1 if 1 in classes else max(classes)
                - pos_index = list(classes).index(pos_label)
                - roc = roc_auc_score(y_test, proba[:, pos_index])
              Else:
                - roc = roc_auc_score(y_test, proba, multi_class='ovr', average='macro')
          Else if decision_function is available:
            - scores = pipeline.decision_function(X_test)
            - Compute ROC-AUC using scores (for multiclass, pass the (n_samples, n_classes) score matrix with multi_class='ovr', average='macro')
          Else: print that ROC-AUC is unavailable.
- Save the trained pipeline with joblib (e.g., 'logistic_regression_trained_model.joblib')

Only return a single Python code block without extra explanation.
"""
    return prompt

def generate_svc_classifier_prompt(params, dep_vars, indep_vars, csv_path):
    dep_str = ", ".join(dep_vars)
    indep_str = ", ".join(indep_vars)
    param_str = _kvpairs(
        params,
        ["C", "kernel", "gamma", "degree", "coef0", "class_weight", "probability", "shrinking", "tol", "max_iter", "random_state"],
    )
    prompt = f"""
Write complete Python code to:
- Import: numpy as np, pandas as pd, joblib; from sklearn.model_selection import train_test_split; from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score; from sklearn.pipeline import Pipeline; from sklearn.preprocessing import StandardScaler; from sklearn.svm import SVC
- Load data from '{csv_path}'
- Use columns [{indep_str}] as features and [{dep_str}] as the classification target
- Impute numeric NaNs with column mean
- Perform an 80/20 stratified train/test split
- Build a Pipeline with StandardScaler() then SVC initialized with: {param_str}
- Fit the pipeline
- Evaluate and print:
    - Accuracy
    - Precision (macro), Recall (macro), F1-score (macro)
    - Confusion matrix
    - ROC-AUC:
        * If probability=True:
            - proba = pipeline.predict_proba(X_test)
            - classes = pipeline.classes_
            - If len(classes) == 2:
                - pos_label = 1 if 1 in classes else max(classes)
                - pos_index = list(classes).index(pos_label)
                - roc = roc_auc_score(y_test, proba[:, pos_index])
              Else:
                - roc = roc_auc_score(y_test, proba, multi_class='ovr', average='macro')
        * Else (use decision_function if available):
            - scores = pipeline.decision_function(X_test)
            - If binary: roc = roc_auc_score(y_test, scores)
              Else: roc = roc_auc_score(y_test, scores, multi_class='ovr', average='macro')
        * Print ROC-AUC (or that it is unavailable if neither API exists)
- Save the trained pipeline with joblib (e.g., 'svc_trained_model.joblib')

Only return a single Python code block without extra explanation.
"""
    return prompt

# =========================================
# LLM contract, extraction, validation, run
# =========================================

DEVELOPER_CONTRACT = textwrap.dedent("""
Output contract:
1) Return **one** fenced code block only, with language tag `python`, nothing before/after.
2) No prose. Comments are allowed **inside** code.
3) Use only standard Python and the libraries explicitly requested by the user prompt.
4) If multiple regression targets are provided and the chosen regressor is not natively multi-output, wrap it with `sklearn.multioutput.MultiOutputRegressor`.
5) Use `joblib` for persistence (not `sklearn.externals`).
6) For NaNs in numeric columns, use `df[col] = df[col].fillna(df[col].mean())`.
7) Compute metrics **on the test set** and, for multi-output regression, print metrics per target (RMSE, R2, NRMSE).
8) Do not invent column names—use exactly those provided. If a provided name is missing, raise a clear `ValueError` in code.
""").strip()

CODE_BLOCK_RE = re.compile(r"```python\s+([\s\S]*?)\s*```", re.IGNORECASE)

ALLOWED_IMPORT_PREFIXES = (
    "numpy", "pandas", "sklearn", "xgboost", "joblib", "os", "sys", "math", "json"
)

def _build_messages(user_prompt: str):
    return [
        {"role": "system", "content": "You are a meticulous Python engineer. Always return a single runnable solution. Never include explanations. Respect the output contract exactly."},
        {"role": "developer", "content": DEVELOPER_CONTRACT},
        {"role": "user", "content": user_prompt},
        {"role": "user", "content": textwrap.dedent("""
        Before returning, internally verify:
        - I used exactly the provided column names.
        - I handled NaNs with column means for numeric columns.
        - I performed the specified train/test split.
        - I used the specified model and hyperparameters.
        - For multiple regression targets, I used MultiOutputRegressor (if needed).
        - I computed the requested metrics on the test set (per target for multi-output).
        - I saved the trained model with joblib.
        Return only a single fenced python code block.
        """).strip()},
    ]

def _extract_code_block(response_text: str) -> str:
    m = CODE_BLOCK_RE.search(response_text or "")
    if not m:
        raise ValueError("No fenced python code block found.")
    return m.group(1).strip()

def _check_allowed_imports(code_str: str):
    # Parse imports via AST and ensure they start with allowed prefixes
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        raise ValueError(f"Generated code has syntax errors: {e}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = (alias.name or "").split(".")[0]
                if root and root not in [p.split(".")[0] for p in ALLOWED_IMPORT_PREFIXES]:
                    raise ValueError(f"Disallowed import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = module.split(".")[0]
            if root and root not in [p.split(".")[0] for p in ALLOWED_IMPORT_PREFIXES]:
                raise ValueError(f"Disallowed import from: {module}")

def _static_validate(code_str: str, feature_cols: List[str], target_cols: List[str]):
    # 1) Ensure column names referenced are a subset of allowed names
    allowed = set(feature_cols) | set(target_cols)
    referenced = set(re.findall(r'["\']([^"\']+)["\']', code_str))
    suspects = [s for s in referenced if (" " in s or s in allowed)]
    for s in suspects:
        if s not in allowed and (" " in s):  # looks like a column-like string but not allowed
            raise ValueError(f"Unapproved/unknown column referenced: {s}")

    # 2) MultiOutput check for multi-target regression use cases
    if len(target_cols) > 1:
        needs_multi_output = True
        # If user chose XGBRegressor/Linear etc., they are not natively multi-output; enforce wrapper presence
        if "MultiOutputRegressor" not in code_str:
            # If the code uses two separate models or pipelines, it might still be valid.
            # We require explicit wrapper for simplicity/consistency in your app:
            raise ValueError("MultiOutputRegressor required for multiple targets but missing.")

    # 3) joblib presence
    if "joblib.dump" not in code_str:
        raise ValueError("joblib.dump missing.")

    # 4) metrics presence
    must_have = ["root_mean_squared_error", "r2_score"]
    for token in must_have:
        if token not in code_str:
            raise ValueError(f"Metric function missing: {token}")

def _request_code_from_llm(user_prompt: str) -> str:
    resp = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=_build_messages(user_prompt),
        temperature=0.1,
        top_p=1.0,
        max_tokens=2800,
    )
    return resp.choices[0].message.content

def get_code_from_chatgpt(prompt: str, feature_cols: List[str] = None, target_cols: List[str] = None) -> Optional[str]:
    """
    Main entry: asks the LLM for code, extracts one fenced python block,
    validates it, retries once if needed.
    """
    feature_cols = feature_cols or []
    target_cols = target_cols or []

    def _attempt(prompt_text: str) -> str:
        text = _request_code_from_llm(prompt_text)
        code = _extract_code_block(text)
        _check_allowed_imports(code)
        if feature_cols or target_cols:
            _static_validate(code, feature_cols, target_cols)
        return code

    try:
        return _attempt(prompt)
    except Exception as e1:
        # Retry with a violation notice prepended to the original prompt
        retry_notice = f"You violated the output contract: {str(e1)}. Fix it. Return one fenced python code block only."
        retry_prompt = retry_notice + "\n\n" + prompt
        try:
            return _attempt(retry_prompt)
        except Exception as e2:
            # Surface the second error to the UI/logs
            raise RuntimeError(f"Code generation failed after retry. First error: {e1}. Second error: {e2}")

def execute_code(code_str: str) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as temp:
        temp.write(code_str)
        script_path = temp.name
    try:
        result = subprocess.run(
            ["python", script_path],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return e.stderr
