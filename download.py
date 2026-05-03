from huggingface_hub import snapshot_download

repo_id = "Ilya-huggingface/lbnl-fdd-data-resampled-3"
out_dir = "/workspace/LBNL_FDD/data/preprocessed_data"

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",  # use "model" for models
    local_dir=out_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
    max_workers=8,
)