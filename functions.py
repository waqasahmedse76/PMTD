# functions.py
import openai
import pandas as pd
import tempfile
import subprocess
import re
import os


openai_client = openai.OpenAI(
    api_key="" #Enter API Key
)


def _kvpairs(params: dict, keys: list[str]) -> str:
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
    )
    return response.choices[0].message.content.strip()

def analyze_csv(file_path: str):
    df = pd.read_csv(file_path)
    return df.columns.tolist()


def generate_prompt(model_name: str, params: dict, dep_vars: list[str], indep_vars: list[str], csv_path: str) -> str:
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
    else:
        raise ValueError("Unsupported model selected")

def generate_xgboost_prompt(params, dep_vars, indep_vars, csv_path):
    dep_str = ", ".join(dep_vars)
    indep_str = ", ".join(indep_vars)

    # Fallback: use epochs as n_estimators if not provided
    if params.get("n_estimators") is None and params.get("epochs") is not None:
        params["n_estimators"] = params.get("epochs")

    param_str = _kvpairs(
        params,
        [
            "learning_rate",
            "n_estimators",
            "max_depth",
            "min_child_weight",
            "subsample",
            "colsample_bytree",
            "gamma",
        ],
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


    param_str = _kvpairs(
        params,
        [
            "fit_intercept",
            "copy_X",
            "positive",
            "n_jobs",  
        ],
    )

    if not param_str:
        param_str = " "  

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


    param_str = _kvpairs(
        params,
        [
            "alpha",
            "fit_intercept",
            "solver",
            "max_iter",
            "tol",
            "positive",
            "random_state",
        ],
    )

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


    param_str = _kvpairs(
        params,
        [
            "alpha",
            "fit_intercept",
            "max_iter",
            "tol",
            "selection",
            "positive",
            "random_state",
        ],
    )

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


    param_str = _kvpairs(
        params,
        [
            "alpha",
            "l1_ratio",
            "fit_intercept",
            "max_iter",
            "tol",
            "selection",
            "positive",
            "random_state",
        ],
    )

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


def get_code_from_chatgpt(prompt: str):
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You generate clean and executable Python ML code."},
            {"role": "user", "content": prompt},
        ],
    )
    code_response = response.choices[0].message.content
    match = re.search(r"```python(.*?)```", code_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def execute_code(code_str: str):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as temp:
        temp.write(code_str)
        script_path = temp.name
    try:
        result = subprocess.run(["python", script_path], capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return e.stderr
