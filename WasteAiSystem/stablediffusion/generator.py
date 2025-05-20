import openai
from diffusers import StableDiffusionXLPipeline
import torch
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration API Mistral (using environment variables)
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = "https://api.mistral.ai/v1"

# Génération de description
def generate_description_with_mistral(pred_class: str) -> str:
    # Check if API key is available
    if not openai.api_key:
        return f"A sustainable product made from recycled {pred_class}, with eco-friendly design elements, shown on a clean white studio background."
    
    prompt = (
        f"You are a sustainable product designer. Propose one specific, useful and realistic object made only from recycled '{pred_class}'. "
        "Keep the description short and precise, suitable for image generation. The object must be 100% made from that material and usable in daily life. "
        "Include the object's name, what it is used for, key visual features (like size, shape, texture, and color), and how it looks on a studio background. "
        "Output in one single sentence formatted like: "
        "\"a [object] made from recycled [material], [key visual details], shown on a clean white studio background\"."
    )

    try:
        response = openai.ChatCompletion.create(
            model="mistral-medium",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Error generating description: {e}")
        return f"A sustainable product made from recycled {pred_class}, with eco-friendly design elements, shown on a clean white studio background."

# Chargement du pipeline une seule fois
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    variant="fp16" if torch.cuda.is_available() else None
)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Génération d'image
def generate_image(description: str):
    prompt = (
        f"High-resolution studio photo of {description}. "
        "Ultra realistic materials, DSLR depth of field, soft shadows, subtle reflections, "
        "high-quality product photography, Canon 85mm lens, smooth background, shot on white."
    )
    negative_prompt = (
        "blurry, distorted, out of frame, watermark, overexposed, unnatural, cartoon, low resolution"
    )

    image = pipe(prompt=prompt, negative_prompt=negative_prompt, guidance_scale=9).images[0]
    return image
