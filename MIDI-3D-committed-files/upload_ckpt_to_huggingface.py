from huggingface_hub import HfApi, create_repo

# Update these values before uploading a checkpoint.
username = "your-huggingface-username"

repo_name = "scone"
checkpoint_path = "path/to/your/scone.ckpt"

repo_id = f"{username}/{repo_name}"

create_repo(repo_id, exist_ok=True, repo_type="model")

api = HfApi()
api.upload_file(
    path_or_fileobj=checkpoint_path,
    path_in_repo="scone.ckpt",
    repo_id=repo_id,
    repo_type="model",
)