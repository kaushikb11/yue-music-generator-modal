# YuE Music Generator Modal

A Modal deployment of [YuE](https://github.com/multimodal-art-projection/YuE), an open-source foundation model for full-song music generation. This implementation provides an easy-to-use web interface for generating music from lyrics.

## Features

- 🎵 Generate complete songs from lyrics with vocal and instrumental tracks
- 🎤 Support for multiple languages including English, Mandarin, Japanese, and Korean
- 🎸 Various music genres and styles
- 🌐 Web interface powered by Gradio
- ⚡ High-performance inference using Modal's cloud infrastructure

## Getting Started

### Prerequisites

1. Install Modal CLI:

```bash
pip install modal
```

2. Configure Modal:

```bash
modal token new
```

### Usage Options

#### Deploy Web Interface

To deploy the Gradio web interface:

```bash
modal deploy yue_modal.py
```

The web interface will be available at the URL provided in the deployment output.

#### Run Locally

To generate audio files locally and save them to disk:

```bash
modal run yue_modal.py
```

This will:

1. Generate audio using the example prompts
1. Save the output files to the `output` directory:
   - `vocals.wav`
   - `instrumentals.wav`
   - `mixed.wav`

## Usage

### Web Interface

1. Enter your desired genre and style tags in the "Genre & Style" field
1. Input your lyrics with appropriate section markers (\[verse\], \[chorus\], etc.)
1. Adjust generation parameters if needed:
   - Max Tokens: Controls the length of generated audio (default: 3000)
   - Number of Segments: Number of sections to generate (default: 2)
1. Click "Generate Music" and wait for the results

### Prompt Engineering Guide

#### Genre Tagging

- Include a combination of genre, instrument, mood, gender, and timbre tags
- Example: `inspiring female uplifting pop airy vocal electronic bright vocal`

#### Lyrics Format

```
[verse]
Your verse lyrics here

[chorus]
Your chorus lyrics here
```

## Technical Details

This implementation uses:

- Modal for deployment and GPU acceleration
- Gradio for the web interface
- YuE's two-stage generation process:
  - Stage 1: Initial token generation
  - Stage 2: Audio upsampling
- Flash Attention 2 for efficient transformer inference

## Hardware Requirements

The model requires significant GPU memory:

- Recommended: 80GB GPU memory (H100/A100) for full song generation

## License

This project follows the same licensing as the original YuE model:

- Model weights are licensed under Creative Commons Attribution Non Commercial 4.0
- Generated outputs can be used and monetized with attribution to "YuE by M-A-P"

## Acknowledgments

This project is a Modal deployment of [YuE](https://github.com/multimodal-art-projection/YuE) by the Multimodal Art Projection team. Please refer to their repository for more detailed information about the underlying model.

## Citation

If you use this project in your research, please cite the original YuE repository.
