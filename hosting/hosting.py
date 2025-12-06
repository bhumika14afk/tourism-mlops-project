from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import create_repo
import os

# Get HF Token from environment variable (NO HARDCODED TOKEN)
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set!")

HF_USERNAME = "bhumikam14"

api = HfApi(token=HF_TOKEN)

repo_id = f"{HF_USERNAME}/tourism-package-app"
repo_type = "space"

print("="*60)
print("DEPLOYING TO HUGGING FACE SPACES")
print("="*60)

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"\nSpace '{repo_id}' already exists. Updating it.")
except RepositoryNotFoundError:
    print(f"\nSpace '{repo_id}' not found. Creating new space...")
    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        space_sdk="docker",
        private=False,
        token=HF_TOKEN
    )
    print(f"Space '{repo_id}' created.")

print("\nUploading files to Hugging Face Space...")
api.upload_folder(
    folder_path="deployment",
    repo_id=repo_id,
    repo_type=repo_type
)

print("\n" + "="*60)
print("✓ DEPLOYMENT SUCCESSFUL!")
print("="*60)
print(f"\n✓ Your app: https://huggingface.co/spaces/{repo_id}")
print("="*60)
