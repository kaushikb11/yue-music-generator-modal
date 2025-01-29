import copy
import os
import re
import uuid
from collections import Counter
from dataclasses import dataclass

import modal


@dataclass
class GenerationConfig:
    """Configuration for music generation."""

    max_new_tokens: int = 3000
    run_n_segments: int = 2
    stage2_batch_size: int = 4
    top_p: float = 0.93
    temperature: float = 1.0
    repetition_penalty: float = 1.2
    stage2_max_new_tokens: int = 512
    stage2_min_new_tokens: int = 100
    max_context: int = 16384


app = modal.App("yue-lyrics-to-song")

volume = modal.Volume.from_name("yue-models-cache", create_if_missing=True)
VOLUME_PATH = "/models"

cuda_version = "12.4.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"


def download_models():
    import shutil

    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM

    os.makedirs(f"{VOLUME_PATH}/xcodec_mini_infer", exist_ok=True)

    if not os.path.exists(f"{VOLUME_PATH}/YuE-s1-7B-anneal-en-cot"):
        print("Downloading stage1 model...")
        stage1_model = AutoModelForCausalLM.from_pretrained(
            "m-a-p/YuE-s1-7B-anneal-en-cot"
        )
        stage1_model.save_pretrained(f"{VOLUME_PATH}/YuE-s1-7B-anneal-en-cot")
    else:
        print("Stage1 model already exists in volume")

    if not os.path.exists(f"{VOLUME_PATH}/YuE-s2-1B-general"):
        print("Downloading stage2 model...")
        stage2_model = AutoModelForCausalLM.from_pretrained("m-a-p/YuE-s2-1B-general")
        stage2_model.save_pretrained(f"{VOLUME_PATH}/YuE-s2-1B-general")
    else:
        print("Stage2 model already exists in volume")

    if not os.path.exists(f"{VOLUME_PATH}/xcodec_mini_infer/final_ckpt"):
        print("Downloading xcodec files...")
        repo_path = snapshot_download(
            repo_id="m-a-p/xcodec_mini_infer",
            local_dir="/tmp/xcodec_mini_infer",
        )

        if os.path.exists(repo_path):
            shutil.copytree(
                repo_path, f"{VOLUME_PATH}/xcodec_mini_infer", dirs_exist_ok=True
            )
    else:
        print("Xcodec files already exist in volume")


def setup_working_directory():
    import shutil

    if os.path.exists(f"{VOLUME_PATH}/xcodec_mini_infer"):
        shutil.copytree(
            f"{VOLUME_PATH}/xcodec_mini_infer",
            "./xcodec_mini_infer",
            dirs_exist_ok=True,
        )
        print("Copied xcodec_mini_infer to working directory")


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
        "sentencepiece",
        "tensorboard",
        "tqdm",
        "descript-audiotools>=0.7.2",
        "descript-audio-codec",
        "scipy==1.10.1",
    )
    .run_commands("CXX=g++ pip install flash-attn --no-build-isolation")
    .run_function(download_models, volumes={VOLUME_PATH: volume})
    .run_function(setup_working_directory, volumes={VOLUME_PATH: volume})
    .add_local_file("mmtokenizer.py", remote_path="/root/mmtokenizer.py")
    .add_local_file("codecmanipulator.py", remote_path="/root/codecmanipulator.py")
    .add_local_dir("mm_tokenizer_v0.2_hf", remote_path="/root/mm_tokenizer_v0.2_hf")
)


@app.cls(
    image=image,
    gpu="A100",
    volumes={VOLUME_PATH: volume},
    timeout=1800,
)
class YuEGenerator:
    @modal.enter()
    def init(self):
        import sys
        from pathlib import Path

        import torch
        from codecmanipulator import CodecManipulator
        from mmtokenizer import _MMSentencePieceTokenizer
        from omegaconf import OmegaConf
        from transformers import (
            AutoModelForCausalLM,
            LogitsProcessor,
            LogitsProcessorList,
        )

        sys.path.append("xcodec_mini_infer")

        from models.soundstream_hubert_new import SoundStream

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize tokenizer
        self.mmtokenizer = _MMSentencePieceTokenizer(
            f"mm_tokenizer_v0.2_hf/tokenizer.model"
        )

        # Load stage 1 model
        self.stage1_model = AutoModelForCausalLM.from_pretrained(
            os.path.join(VOLUME_PATH, "YuE-s1-7B-anneal-en-cot"),
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to(self.device)
        self.stage1_model.eval()

        # Load stage 2 model
        self.stage2_model = AutoModelForCausalLM.from_pretrained(
            os.path.join(VOLUME_PATH, "YuE-s2-1B-general"),
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to(self.device)
        self.stage2_model.eval()

        # Initialize codec tools
        self.codectool = CodecManipulator("xcodec", 0, 1)
        self.codectool_stage2 = CodecManipulator("xcodec", 0, 8)

        # Load codec model
        model_config = OmegaConf.load(
            f"{VOLUME_PATH}/xcodec_mini_infer/final_ckpt/config.yaml"
        )
        self.codec_model = eval(model_config.generator.name)(
            **model_config.generator.config
        ).to(self.device)
        parameter_dict = torch.load(
            f"{VOLUME_PATH}/xcodec_mini_infer/final_ckpt/ckpt_00360000.pth",
            map_location="cpu",
        )
        self.codec_model.load_state_dict(parameter_dict["codec_model"])
        self.codec_model.eval()

    def _split_lyrics(self, lyrics):
        pattern = r"\[(\w+)\](.*?)\n(?=\[|\Z)"
        segments = re.findall(pattern, lyrics, re.DOTALL)
        return [f"[{seg[0]}]\n{seg[1].strip()}\n\n" for seg in segments]

    def _stage2_inference(self, tokens, batch_size=4):
        """Run stage 2 upsampling inference using the same approach as infer.py"""
        import numpy as np
        import torch
        from einops import rearrange
        from transformers import LogitsProcessor, LogitsProcessorList

        class BlockTokenRangeProcessor(LogitsProcessor):
            def __init__(self, start_id, end_id):
                self.blocked_token_ids = list(range(start_id, end_id))

            def __call__(self, input_ids, scores):
                scores[:, self.blocked_token_ids] = -float("inf")
                return scores

        # Convert to numpy if it's a tensor
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().numpy()

        def stage2_generate(model, prompt, batch_size=4):
            codec_ids = self.codectool.unflatten(prompt, n_quantizer=1)
            codec_ids = self.codectool.offset_tok_ids(
                codec_ids,
                global_offset=self.codectool.global_offset,
                codebook_size=self.codectool.codebook_size,
                num_codebooks=self.codectool.num_codebooks,
            ).astype(np.int32)

            # Handle batching
            if batch_size > 1:
                codec_list = []
                for i in range(batch_size):
                    idx_begin = i * 300
                    idx_end = (i + 1) * 300
                    codec_list.append(codec_ids[:, idx_begin:idx_end])

                codec_ids = np.concatenate(codec_list, axis=0)
                prompt_ids = np.concatenate(
                    [
                        np.tile(
                            [self.mmtokenizer.soa, self.mmtokenizer.stage_1],
                            (batch_size, 1),
                        ),
                        codec_ids,
                        np.tile([self.mmtokenizer.stage_2], (batch_size, 1)),
                    ],
                    axis=1,
                )
            else:
                prompt_ids = np.concatenate(
                    [
                        np.array([self.mmtokenizer.soa, self.mmtokenizer.stage_1]),
                        codec_ids.flatten(),
                        np.array([self.mmtokenizer.stage_2]),
                    ]
                ).astype(np.int32)
                prompt_ids = prompt_ids[np.newaxis, ...]

            codec_ids = torch.as_tensor(codec_ids).to(self.device)
            prompt_ids = torch.as_tensor(prompt_ids).to(self.device)
            len_prompt = prompt_ids.shape[-1]

            block_list = LogitsProcessorList(
                [
                    BlockTokenRangeProcessor(0, 46358),
                    BlockTokenRangeProcessor(53526, self.mmtokenizer.vocab_size),
                ]
            )

            # Teacher forcing generate loop
            for frames_idx in range(codec_ids.shape[1]):
                cb0 = codec_ids[:, frames_idx : frames_idx + 1]
                prompt_ids = torch.cat([prompt_ids, cb0], dim=1)

                with torch.no_grad():
                    stage2_output = self.stage2_model.generate(
                        input_ids=prompt_ids,
                        min_new_tokens=7,
                        max_new_tokens=7,
                        eos_token_id=self.mmtokenizer.eoa,
                        pad_token_id=self.mmtokenizer.eoa,
                        logits_processor=block_list,
                    )

                assert (
                    stage2_output.shape[1] - prompt_ids.shape[1] == 7
                ), f"output new tokens={stage2_output.shape[1]-prompt_ids.shape[1]}"
                prompt_ids = stage2_output

            # Return output based on batch size
            if batch_size > 1:
                output = prompt_ids.cpu().numpy()[:, len_prompt:]
                output_list = [output[i] for i in range(batch_size)]
                output = np.concatenate(output_list, axis=0)
            else:
                output = prompt_ids[0].cpu().numpy()[len_prompt:]

            return output

        # Process in segments like infer.py
        output_duration = tokens.shape[-1] // 50 // 6 * 6
        num_batch = output_duration // 6

        if num_batch <= batch_size:
            output = stage2_generate(
                self.stage2_model,
                tokens[:, : output_duration * 50],
                batch_size=num_batch,
            )
        else:
            segments = []
            num_segments = (num_batch // batch_size) + (
                1 if num_batch % batch_size != 0 else 0
            )

            for seg in range(num_segments):
                start_idx = seg * batch_size * 300
                end_idx = min((seg + 1) * batch_size * 300, output_duration * 50)
                current_batch_size = (
                    batch_size
                    if seg != num_segments - 1 or num_batch % batch_size == 0
                    else num_batch % batch_size
                )
                segment = stage2_generate(
                    self.stage2_model,
                    tokens[:, start_idx:end_idx],
                    batch_size=current_batch_size,
                )
                segments.append(segment)

            output = np.concatenate(segments, axis=0)

        # Handle remaining tokens
        if output_duration * 50 != tokens.shape[-1]:
            ending = stage2_generate(
                self.stage2_model, tokens[:, output_duration * 50 :], batch_size=1
            )
            output = np.concatenate([output, ending], axis=0)

        output = self.codectool_stage2.ids2npy(output)

        # Fix invalid codes as in infer.py
        fixed_output = copy.deepcopy(output)
        for i, line in enumerate(output):
            for j, element in enumerate(line):
                if element < 0 or element > 1023:
                    counter = Counter(line)
                    most_frequent = sorted(
                        counter.items(), key=lambda x: x[1], reverse=True
                    )[0][0]
                    fixed_output[i, j] = most_frequent

        return fixed_output

    @modal.method()
    def generate(
        self,
        genre_txt: str,
        lyrics_txt: str,
        max_new_tokens: int = 3000,
        run_n_segments: int = 2,
        stage2_batch_size: int = 4,
    ):
        import numpy as np
        import torch
        from einops import rearrange
        from transformers import LogitsProcessor, LogitsProcessorList

        genres = genre_txt.strip()
        lyrics = self._split_lyrics(lyrics_txt)

        # Format prompts
        full_lyrics = "\n".join(lyrics)
        prompt_texts = [
            f"Generate music from the given lyrics segment by segment.\n[Genre] {genres}\n{full_lyrics}"
        ]
        prompt_texts += lyrics

        # Stage 1: Generate initial tokens
        output_seq = None
        raw_output = None

        # Decoding config
        top_p = 0.93
        temperature = 1.0
        repetition_penalty = 1.2

        # Special tokens
        start_of_segment = self.mmtokenizer.tokenize("[start_of_segment]")
        end_of_segment = self.mmtokenizer.tokenize("[end_of_segment]")

        # Run generation for each segment
        run_n_segments = min(run_n_segments + 1, len(lyrics))
        stage1_outputs = []

        class BlockTokenRangeProcessor(LogitsProcessor):
            def __init__(self, start_id, end_id):
                self.blocked_token_ids = list(range(start_id, end_id))

            def __call__(self, input_ids, scores):
                scores[:, self.blocked_token_ids] = -float("inf")
                return scores

        for i, p in enumerate(prompt_texts[:run_n_segments]):
            if i == 0:
                continue

            section_text = p.replace("[start_of_segment]", "").replace(
                "[end_of_segment]", ""
            )
            guidance_scale = 1.5 if i <= 1 else 1.2

            # Format prompt ids
            if i == 1:
                head_id = self.mmtokenizer.tokenize(prompt_texts[0])
                prompt_ids = (
                    head_id
                    + start_of_segment
                    + self.mmtokenizer.tokenize(section_text)
                    + [self.mmtokenizer.soa]
                    + self.codectool.sep_ids
                )
            else:
                prompt_ids = (
                    end_of_segment
                    + start_of_segment
                    + self.mmtokenizer.tokenize(section_text)
                    + [self.mmtokenizer.soa]
                    + self.codectool.sep_ids
                )

            prompt_ids = torch.as_tensor(prompt_ids).unsqueeze(0).to(self.device)
            input_ids = (
                torch.cat([raw_output, prompt_ids], dim=1) if i > 1 else prompt_ids
            )

            # Window slicing for long sequences
            max_context = 16384 - max_new_tokens - 1
            if input_ids.shape[-1] > max_context:
                input_ids = input_ids[:, -max_context:]

            max_context = 16384 - max_new_tokens - 1
            if input_ids.shape[-1] > max_context:
                input_ids = input_ids[:, -max_context:]

            # Generate
            with torch.no_grad():
                output_seq = self.stage1_model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=100,
                    do_sample=True,
                    top_p=top_p,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    eos_token_id=self.mmtokenizer.eoa,
                    pad_token_id=self.mmtokenizer.eoa,
                    logits_processor=LogitsProcessorList(
                        [
                            BlockTokenRangeProcessor(0, 32002),
                            BlockTokenRangeProcessor(32016, 32016),
                        ]
                    ),
                    guidance_scale=guidance_scale,
                )

                if output_seq[0][-1].item() != self.mmtokenizer.eoa:
                    tensor_eoa = torch.as_tensor([[self.mmtokenizer.eoa]]).to(
                        self.device
                    )
                    output_seq = torch.cat((output_seq, tensor_eoa), dim=1)

            if i > 1:
                raw_output = torch.cat(
                    [raw_output, prompt_ids, output_seq[:, input_ids.shape[-1] :]],
                    dim=1,
                )
            else:
                raw_output = output_seq

        print("Stage 1 generation complete")
        # Process stage 1 outputs
        ids = raw_output[0].cpu().numpy()
        soa_idx = np.where(ids == self.mmtokenizer.soa)[0].tolist()
        eoa_idx = np.where(ids == self.mmtokenizer.eoa)[0].tolist()

        if len(soa_idx) != len(eoa_idx):
            raise ValueError(
                f"Invalid pairs of soa and eoa, Num of soa: {len(soa_idx)}, Num of eoa: {len(eoa_idx)}"
            )

        # Extract vocals and instrumentals
        vocals = []
        instrumentals = []

        for i in range(len(soa_idx)):
            codec_ids = ids[soa_idx[i] + 1 : eoa_idx[i]]
            if codec_ids[0] == 32016:  # Skip separator token if present
                codec_ids = codec_ids[1:]
            codec_ids = codec_ids[: 2 * (codec_ids.shape[0] // 2)]

            # Split into vocals and instrumentals
            vocals_ids = self.codectool.ids2npy(
                rearrange(codec_ids, "(n b) -> b n", b=2)[0]
            )
            vocals.append(vocals_ids)
            instrumentals_ids = self.codectool.ids2npy(
                rearrange(codec_ids, "(n b) -> b n", b=2)[1]
            )
            instrumentals.append(instrumentals_ids)

        vocals = np.concatenate(vocals, axis=1)
        instrumentals = np.concatenate(instrumentals, axis=1)

        print("Stage 2 upsampling starting...")

        # Stage 2: Upsampling
        vocals_stage2 = self._stage2_inference(vocals, stage2_batch_size)
        instrumentals_stage2 = self._stage2_inference(instrumentals, stage2_batch_size)

        print("Generating final audio...")

        # Generate final audio
        with torch.no_grad():
            # Generate vocals
            vocals_wav = (
                self.codec_model.decode(
                    torch.as_tensor(vocals_stage2, dtype=torch.long)
                    .unsqueeze(0)
                    .permute(1, 0, 2)
                    .to(self.device)
                )
                .cpu()
                .numpy()
            )

            # Generate instrumentals
            instrumentals_wav = (
                self.codec_model.decode(
                    torch.as_tensor(instrumentals_stage2, dtype=torch.long)
                    .unsqueeze(0)
                    .permute(1, 0, 2)
                    .to(self.device)
                )
                .cpu()
                .numpy()
            )

        # Mix tracks
        mixed_wav = vocals_wav + instrumentals_wav

        print("Generation complete!")

        return {
            "vocals": vocals_wav,
            "instrumentals": instrumentals_wav,
            "mixed": mixed_wav,
        }


@app.local_entrypoint()
def main():
    generator = YuEGenerator()

    with open("inference/prompt_examples/genre.txt") as f:
        genre_txt = f.read()
    with open("inference/prompt_examples/lyrics.txt") as f:
        lyrics_txt = f.read()

    outputs = generator.generate.remote(
        genre_txt=genre_txt,
        lyrics_txt=lyrics_txt,
        max_new_tokens=1000,
        run_n_segments=1,
        stage2_batch_size=4,
    )

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    import soundfile as sf

    sf.write(f"{output_dir}/vocals.wav", outputs["vocals"], 16000)
    sf.write(f"{output_dir}/instrumentals.wav", outputs["instrumentals"], 16000)
    sf.write(f"{output_dir}/mixed.wav", outputs["mixed"], 16000)
