from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os

# Get HF Token from environment (NO HARDCODED TOKEN)
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set!")

HF_USERNAME = "bhumikam14"

api = HfApi(token=HF_TOKEN)

repo_id = f"{HF_USERNAME}/tourism-dataset"
repo_type = "dataset"

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False, token=HF_TOKEN)
    print(f"Dataset space '{repo_id}' created.")

print("Uploading data to Hugging Face...")
api.upload_folder(
    folder_path="data",
    repo_id=repo_id,
    repo_type=repo_type,
)

print(f"✓ Data uploaded successfully to {repo_id}")
print(f"✓ View at: https://huggingface.co/datasets/{repo_id}")
