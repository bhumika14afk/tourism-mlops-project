import pandas as pd
import os
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split

# Get HF Token from environment (NO HARDCODED TOKEN)
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set!")

HF_USERNAME = "bhumikam14"

api = HfApi(token=HF_TOKEN)

DATASET_PATH = f"hf://datasets/{HF_USERNAME}/tourism-dataset/tourism.csv"
repo_id = f"{HF_USERNAME}/tourism-dataset"
repo_type = "dataset"

print("Loading dataset from Hugging Face...")
df = pd.read_csv(DATASET_PATH)

print("Cleaning data...")
df = df.drop(['CustomerID'], axis=1, errors='ignore')
if 'Unnamed: 0' in df.columns:
    df = df.drop(['Unnamed: 0'], axis=1)

df['Gender'] = df['Gender'].replace('Fe Male', 'Female')
df = df.dropna()

print(f"Cleaned data shape: {df.shape}")

X = df.drop('ProdTaken', axis=1)
y = df['ProdTaken']

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"Categorical columns: {categorical_cols}")
print(f"Numerical columns: {numerical_cols}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

os.makedirs("temp_data", exist_ok=True)
X_train.to_csv("temp_data/Xtrain.csv", index=False)
X_test.to_csv("temp_data/Xtest.csv", index=False)
y_train.to_csv("temp_data/ytrain.csv", index=False)
y_test.to_csv("temp_data/ytest.csv", index=False)

with open("temp_data/categorical_cols.txt", "w") as f:
    f.write(",".join(categorical_cols))

with open("temp_data/numerical_cols.txt", "w") as f:
    f.write(",".join(numerical_cols))

print("Uploading split datasets to Hugging Face...")
api.upload_folder(
    folder_path="temp_data",
    repo_id=repo_id,
    repo_type=repo_type,
    path_in_repo="splits"
)

print("✓ Data preparation completed successfully!")
