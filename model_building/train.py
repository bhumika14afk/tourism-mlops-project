import pandas as pd
import os
import joblib
from huggingface_hub import HfApi
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import create_repo
import urllib.request

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Get HF Token from environment (NO HARDCODED TOKEN)
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set!")

HF_USERNAME = "bhumikam14"

if MLFLOW_AVAILABLE:
    try:
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("Tourism-Package-Prediction-Experiment")
    except:
        MLFLOW_AVAILABLE = False

api = HfApi(token=HF_TOKEN)

base_path = f"hf://datasets/{HF_USERNAME}/tourism-dataset/splits"

print("\nLoading datasets from Hugging Face...")
X_train = pd.read_csv(f"{base_path}/Xtrain.csv")
X_test = pd.read_csv(f"{base_path}/Xtest.csv")
y_train = pd.read_csv(f"{base_path}/ytrain.csv").values.ravel()
y_test = pd.read_csv(f"{base_path}/ytest.csv").values.ravel()

cat_url = f"https://huggingface.co/datasets/{HF_USERNAME}/tourism-dataset/resolve/main/splits/categorical_cols.txt"
num_url = f"https://huggingface.co/datasets/{HF_USERNAME}/tourism-dataset/resolve/main/splits/numerical_cols.txt"

with urllib.request.urlopen(cat_url) as f:
    categorical_cols = f.read().decode('utf-8').strip().split(",")

with urllib.request.urlopen(num_url) as f:
    numerical_cols = f.read().decode('utf-8').strip().split(",")

preprocessor = make_column_transformer(
    (StandardScaler(), numerical_cols),
    (OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
    remainder='passthrough'
)

base_model = XGBClassifier(random_state=42, eval_metric='logloss')

model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', base_model)
])

param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [3, 5, 7],
    'classifier__learning_rate': [0.01, 0.1],
    'classifier__subsample': [0.8, 1.0]
}

print(f"\nStarting Grid Search...")

if MLFLOW_AVAILABLE:
    mlflow.start_run()

grid_search = GridSearchCV(
    model_pipeline,
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=0
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

if MLFLOW_AVAILABLE:
    mlflow.log_params(grid_search.best_params_)

y_test_pred = best_model.predict(X_test)

test_accuracy = accuracy_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

print(f"\nTest Accuracy: {test_accuracy:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")

if MLFLOW_AVAILABLE:
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("test_f1", test_f1)
    mlflow.sklearn.log_model(best_model, "model")
    mlflow.end_run()

os.makedirs("model_artifacts", exist_ok=True)
joblib.dump(best_model, "model_artifacts/tourism_model.pkl")

with open("model_artifacts/categorical_cols.txt", "w") as f:
    f.write(",".join(categorical_cols))

with open("model_artifacts/numerical_cols.txt", "w") as f:
    f.write(",".join(numerical_cols))

repo_id = f"{HF_USERNAME}/tourism-package-model"
repo_type = "model"

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
except RepositoryNotFoundError:
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False, token=HF_TOKEN)

api.upload_folder(
    folder_path="model_artifacts",
    repo_id=repo_id,
    repo_type=repo_type
)

print(f"\n✓ Model uploaded to: https://huggingface.co/{repo_id}")
