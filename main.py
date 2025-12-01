import argparse
from pathlib import Path
import warnings
import traceback

import torch
from diffusers import ZImagePipeline

# Silence the noisy CUDA autocast warning on Mac
warnings.filterwarnings(
    "ignore",
    message="User provided device_type of 'cuda', but CUDA is not available. Disabling warnings.warn",
    category=UserWarning,
)

warnings.filterwarnings(
    "ignore",
    message="`torch_dtype` is deprecated! Use `dtype` instead!",
    category=UserWarning,
)


def load_pipeline(device: str) -> ZImagePipeline:
    model_id = "Tongyi-MAI/Z-Image-Turbo"

    if device == "cpu":
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.bfloat16

    print(f"[info] try to load model with torch_dtype={torch_dtype} ...")

    pipe = ZImagePipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,   # ZImagePipeline still expects torch_dtype
        low_cpu_mem_usage=False,
    )
    pipe = pipe.to(device)

    # Disable safety checker while debugging (it can output black images)
    if hasattr(pipe, "safety_checker") and pipe.safety_checker is not None:
        print("[info] disable safety_checker")
        pipe.safety_checker = None

    # Debug actual dtypes
    if hasattr(pipe, "unet"):
        print("[info] UNet dtype:", pipe.unet.dtype)
    if hasattr(pipe, "vae"):
        print("[info] VAE  dtype:", pipe.vae.dtype)
    if hasattr(pipe, "text_encoder"):
        print("[info] Text encoder dtype:", pipe.text_encoder.dtype)

    return pipe


def parse_args():
    parser = argparse.ArgumentParser(
        description="Z-Image Turbo CLI (prompt, resolution, optional save path)"
    )
    parser.add_argument(
        "prompt",
        type=str,
        help="Prompt for image generation",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output image path (optional, default: outputs/<prompt>.png)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=9,
        help="Sampling steps (default 9, try 15–25 for better quality)",
    )
    parser.add_argument(
        "--width",
        "-w",
        type=int,
        default=1280,
        help="Image width (must be multiple of 8), default 768",
    )
    parser.add_argument(
        "--height",
        "-H",
        type=int,
        default=720,
        help="Image height (must be multiple of 8), default 768",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"[debug] cwd: {Path.cwd().resolve()}")

    # Ensure width/height are multiples of 16
    for name in ["width", "height"]:
        v = getattr(args, name)
        if v % 8 != 0:
            fixed = (v // 16) * 16
            print(f"[warn] {name}={v} is not a multiple of 8, adjust to {fixed}")
            setattr(args, name, fixed)

    # Choose device
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        print("[warn] MPS not available, using CPU (slow).")
        device = "cpu"

    print(f"[info] using device: {device}")
    pipe = load_pipeline(device)

    # Determine output path
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if args.output is None:
        safe_prompt = "".join(
            c for c in args.prompt[:30] if c.isalnum() or c in "-_"
        )
        if not safe_prompt:
            safe_prompt = "image"
        filename = f"{safe_prompt}.png"
        output_path = outputs_dir / filename
    else:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            # If user gives relative path, put it under outputs/ for clarity
            output_path = outputs_dir / output_path

    print(f"[debug] final output path will be: {output_path.resolve()}")

    # Generate image with strong logging & error handling
    try:
        print(f"[info] generating image for prompt: {args.prompt!r}")
        print(
            f"[debug] steps={args.steps}, width={args.width}, "
            f"height={args.height}, guidance_scale=0.0"
        )

        image = pipe(
            args.prompt,
            num_inference_steps=args.steps,
            height=args.height,
            width=args.width,
            guidance_scale=0.0,  # Turbo model: CFG must be 0
        ).images[0]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        print(f"[info] image saved to: {output_path.resolve()}")

    except Exception as e:
        print("[error] exception during generation or saving:")
        print(e)
        traceback.print_exc()


if __name__ == "__main__":
    main()

