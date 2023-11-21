import os
from huggingface_hub import hf_hub_download


def download_mpt_quant(destination_folder: str, repo_id: str, model_filename: str):
    local_path = os.path.abspath(destination_folder)
    return hf_hub_download(
        repo_id=repo_id,
        filename=model_filename,
        local_dir=local_path,
        local_dir_use_symlinks=True
    )


if __name__ == "__main__":
    """full url: https://https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/blob/main/llama-2-7b-chat.ggmlv3.q2_K.bin"""

    repo_id = "TheBloke/Llama-2-7B-Chat-GGML"
    model_filename = "llama-2-7b-chat.ggmlv3.q2_K.bin"
    destination_folder = "models"
    download_mpt_quant(destination_folder, repo_id, model_filename)
