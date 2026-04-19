from huggingface_hub import HfApi, CommitOperationCopy, CommitOperationDelete

api = HfApi()

repo_id = "Ilya-huggingface/lbnl-fdd-data-resampled-3"

operations = [
    CommitOperationCopy(
        src_path_in_repo="train_target.csv",
        path_in_repo="SDAHU/train_target.csv",
    ),
    CommitOperationDelete(
        path_in_repo="train_target.csv",
    ),
]

api.create_commit(
    repo_id=repo_id,
    operations=operations,
    repo_type="dataset", 
    commit_message="Move file into new_folder",
)