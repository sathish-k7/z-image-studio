# Z-Image CLI

A command-line interface for the **Z-Image-Turbo** text-to-image generation model (`Tongyi-MAI/Z-Image-Turbo`).

This tool is designed to run efficiently on local machines, with specific optimizations for **Apple Silicon (MPS)**, falling back to CPU if unavailable.

## Features

*   **Z-Image-Turbo Model**: Utilizes the high-quality `Tongyi-MAI/Z-Image-Turbo` model via `diffusers`.
*   **MPS Acceleration**: Optimized for Mac users with Apple Silicon.
*   **Automatic Dimension Adjustment**: Ensures image dimensions are compatible (multiples of 8).
*   **Simple CLI**: Easy-to-use command line arguments.

## Requirements

*   Python >= 3.11
*   `uv` (recommended for dependency management)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd zimage-cli
    ```

2.  **Install dependencies:**
    Using `uv` (recommended):
    ```bash
    uv sync
    ```

    Or using standard pip:
    ```bash
    pip install -r requirements.txt
    # Note: You may need to install the specific git version of diffusers manually if pip fails
    ```

## Usage

Run the script using `uv run` or directly with python if your environment is active.

### Basic Generation
```bash
uv run main.py "A futuristic city with neon lights"
```
This will save the generated image to `outputs/Afuturisticcity.png` (filename derived from prompt).

### Custom Output Path
```bash
uv run main.py "A cute cat" --output "my_cat.png"
```

### Adjusting Resolution and Quality
```bash
uv run main.py "Landscape view of mountains" --width 1920 --height 1080 --steps 20
```

## Command Line Arguments

| Argument | Short | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `prompt` | | `str` | Required | The text prompt for image generation. |
| `--output` | `-o` | `str` | `None` | Custom output filename. Defaults to `outputs/<prompt-slug>.png`. |
| `--steps` | | `int` | `9` | Number of inference steps. Higher usually means better quality (try 15-25). |
| `--width` | `-w` | `int` | `1280` | Image width (automatically adjusted to be a multiple of 8). |
| `--height` | `-H` | `int` | `720` | Image height (automatically adjusted to be a multiple of 8). |

## Notes

*   **Guidance Scale**: The script hardcodes `guidance_scale=0.0` as required by the Turbo model distillation process.
*   **Safety Checker**: Disabled by default to prevent false positives and potential black image outputs during local testing.
