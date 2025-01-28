import os

import modal

app = modal.App("yue-lyrics-to-song")

volume = modal.Volume.from_name("yue-models-cache", create_if_missing=True)
VOLUME_PATH = "/models"


def download_models():
    import shutil

    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM

    os.makedirs(f"{VOLUME_PATH}/models", exist_ok=True)
    os.makedirs(f"{VOLUME_PATH}/xcodec_mini_infer", exist_ok=True)

    if not os.path.exists(f"{VOLUME_PATH}/models/YuE-s1-7B-anneal-en-cot"):
        print("Downloading stage1 model...")
        stage1_model = AutoModelForCausalLM.from_pretrained(
            "m-a-p/YuE-s1-7B-anneal-en-cot"
        )
        stage1_model.save_pretrained(f"{VOLUME_PATH}/models/YuE-s1-7B-anneal-en-cot")
    else:
        print("Stage1 model already exists in volume")

    if not os.path.exists(f"{VOLUME_PATH}/models/YuE-s2-1B-general"):
        print("Downloading stage2 model...")
        stage2_model = AutoModelForCausalLM.from_pretrained("m-a-p/YuE-s2-1B-general")
        stage2_model.save_pretrained(f"{VOLUME_PATH}/models/YuE-s2-1B-general")
    else:
        print("Stage2 model already exists in volume")

    if not os.path.exists(f"{VOLUME_PATH}/xcodec_mini_infer/final_ckpt"):
        print("Downloading xcodec files...")
        repo_path = snapshot_download(
            repo_id="m-a-p/xcodec_mini_infer",
            local_dir="/tmp/xcodec_mini_infer",
            local_dir_use_symlinks=False,
        )

        if os.path.exists(repo_path):
            shutil.copytree(
                repo_path, f"{VOLUME_PATH}/xcodec_mini_infer", dirs_exist_ok=True
            )
            os.makedirs(f"{VOLUME_PATH}/mm_tokenizer_v0.2_hf", exist_ok=True)

            tokenizer_src = os.path.join(
                repo_path, "mm_tokenizer_v0.2_hf", "tokenizer.model"
            )
            tokenizer_dst = f"{VOLUME_PATH}/mm_tokenizer_v0.2_hf/tokenizer.model"

            if os.path.exists(tokenizer_src):
                shutil.copy(tokenizer_src, tokenizer_dst)
            else:
                raise FileNotFoundError(f"Tokenizer model not found at {tokenizer_src}")
    else:
        print("Xcodec files already exist in volume")


cuda_version = "12.4.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "ninja",
        "packaging",
        "wheel",
        "torch",
        "torchaudio",
    )
    .pip_install(
        "transformers",
        "huggingface-hub",
        "einops",
        "omegaconf",
        "soundfile",
        "numpy",
        "tqdm",
    )
    .run_commands("CXX=g++ pip install flash-attn --no-build-isolation")
    .run_function(download_models)
)


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    gpu="H100",
)
def init_volume():
    download_models()


@app.local_entrypoint()
def main():
    init_volume.remote()
