from huggingface_hub import HfApi, upload_folder

repo_id = "Ilya-huggingface/lbnl-fdd-data-resampled-3"

api = HfApi()

upload_folder(
    folder_path="data/processed/SDAHU/preprocessed_resampled_1",
    repo_id=repo_id,
    repo_type="dataset",
)