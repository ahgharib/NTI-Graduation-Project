#!/usr/bin/env python3
"""
paged_handwritten_notes_openai_flux.py

- OpenAI generates paged student notes
- Prints LLM output in terminal
- One FLUX call per page
- Notebook-style handwritten output
"""

import sys
import os
import random
from typing import List

import torch
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI

# =========================
# LOD ENV
# =========================
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY not found")

client = OpenAI()

# =========================
# FLUX
# =========================
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline

# =========================
# CONFIG
# =========================
FLUX_MODEL_ID = "black-forest-labs/FLUX.1-dev"
LORA_HANDWRITE_ID = "fofr/flux-handwriting"

OPENAI_MODEL = "gpt-4.1-mini"

RESOLUTION = (1536, 2048)
CFG_SCALE = 5.0
STEPS = 35

MAX_WORDS_PER_LINE = 6
MAX_LINES_PER_PAGE = 8
MAX_PAGES = 2

LORA_SCALE = 1.0

NEGATIVE_PROMPT = (
    "typed text, digital font, calligraphy, diagram, "
    "symbols, icons, perfect handwriting, watermark"
)

PROMPT_TEMPLATE = (
    "<lora:flux-handwriting:{scale}> "
    "messy student handwriting saying \"{text}\", "
    "uneven spacing, blue ballpoint pen, "
    "old lined notebook paper, text only"
)

# =========================
# IMAGE EXTRACTION
# =========================
def extract_image(output) -> Image.Image:
    image = output[0] if isinstance(output, (list, tuple)) else output.images[0]
    if not isinstance(image, Image.Image):
        raise RuntimeError("Invalid FLUX output")
    return image

# =========================
# OPENAI → PAGED NOTES
# =========================
def generate_pages(topic: str) -> List[List[str]]:
    system_prompt = (
        "You are a university student writing handwritten revision notes. "
        "Notes must look like notebook bullets. "
        "Introductory and general only."
    )

    user_prompt = f"""
Topic: {topic}

RULES:
- Short bullet fragments
- No full sentences
- Max {MAX_WORDS_PER_LINE} words per line
- Max {MAX_LINES_PER_PAGE} lines per page
- No numbering
- No markdown
- No emojis

STRUCTURE:
Title
What it is
Why it matters
Main ideas
Simple examples
Quick summary

FORMAT:
Separate pages using:
===PAGE===

Return text only.
"""

    response = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_output_tokens=1200,
    )

    text = response.output_text
    raw_pages = text.split("===PAGE===")

    pages: List[List[str]] = []

    for page in raw_pages[:MAX_PAGES]:
        lines = []
        for line in page.splitlines():
            line = line.strip()
            if not line:
                continue

            words = line.split()
            if len(words) > MAX_WORDS_PER_LINE:
                line = " ".join(words[:MAX_WORDS_PER_LINE])

            lines.append(line)

        if lines:
            pages.append(lines[:MAX_LINES_PER_PAGE])

    return pages

# =========================
# PRINT LLM OUTPUT
# =========================
def print_llm_output(pages: List[List[str]]):
    print("\n" + "=" * 50)
    print("LLM OUTPUT (PAGED NOTES)")
    print("=" * 50 + "\n")

    for i, page in enumerate(pages, start=1):
        print(f"--- PAGE {i} ---")
        for line in page:
            print(line)
        print()

# =========================
# FLUX PAGE RENDER
# =========================
def render_page(pipe, lines, seed, output_name):
    handwritten_text = " | ".join(lines)

    prompt = PROMPT_TEMPLATE.format(
        scale=LORA_SCALE,
        text=""handwritten_text
    )

    generator = torch.Generator(device="cpu").manual_seed(seed)

    result = pipe(
        prompt,
        negative_prompt=NEGATIVE_PROMPT,
        width=RESOLUTION[0],
        height=RESOLUTION[1],
        guidance_scale=CFG_SCALE,
        num_inference_steps=STEPS,
        generator=generator
    )

    image = extract_image(result)
    image.save(f"{output_name}.png")

# =========================
# MAIN
# =========================
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Study topic")
    parser.add_argument("-o", "--out", default="notes")
    parser.add_argument("-s", "--seed", type=int)

    args = parser.parse_args()

    # 1. GENERATE NOTES
    pages = generate_pages(args.input)

    # 2. PRINT LLM OUTPUT
    print_llm_output(pages)

    print(f"Generated {len(pages)} pages")

    # 3. LOAD FLUX
    pipe = FluxPipeline.from_pretrained(
        FLUX_MODEL_ID,
        torch_dtype=torch.bfloat16
    )
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()

    pipe.load_lora_weights(LORA_HANDWRITE_ID)
    pipe.fuse_lora(lora_scale=LORA_SCALE)

    base_seed = args.seed or random.randint(1, 10_000_000)

    # 4. RENDER EACH PAGE
    for i, page_lines in enumerate(pages, start=1):
        print(f"Rendering page {i}")
        render_page(
            pipe,
            page_lines,
            seed=base_seed + i,
            output_name=f"{args.out}_page_{i:02d}"
        )

    print("Done.")

if __name__ == "__main__":
    main()