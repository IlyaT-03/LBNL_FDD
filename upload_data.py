from huggingface_hub import HfApi, upload_folder

repo_id = "Ilya-huggingface/lbnl-fdd-data"

api = HfApi()

upload_folder(
    folder_path="data/processed",
    repo_id=repo_id,
    repo_type="dataset",
)