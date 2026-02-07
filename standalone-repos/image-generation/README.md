# Project 5: AI Image Generation Service

![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)

A cloud-native image generation service with **multi-provider support** (OpenAI DALL-E, Replicate/Stable Diffusion/FLUX), **style presets**, **LLM-powered prompt enhancement**, and **educational content** explaining the generative model architectures behind modern text-to-image systems.

---

## What You'll Learn

- **Generative Model Architectures** -- How VAEs, GANs, and Diffusion Models work under the hood, including mathematical foundations, training procedures, and architectural diagrams
- **Text-to-Image Pipeline Design** -- Building a production pipeline: prompt enhancement, provider dispatch, style presets, and post-processing
- **Multi-Provider Architecture** -- Abstract base class pattern enabling hot-swappable image generation backends (OpenAI DALL-E 2/3, Replicate SD 3.5, FLUX)
- **Prompt Engineering for Images** -- How LLM-based prompt enhancement rewrites terse descriptions into detailed, high-quality prompts with composition, lighting, and style keywords
- **Style Presets** -- Configurable parameter profiles (photorealistic, anime, cinematic, watercolor, pixel art, etc.) with per-style guidance scale, step counts, and negative prompts
- **Storage Abstraction** -- Pluggable backend (local filesystem or S3-compatible) for persisting generated images with gallery metadata tracking
- **Batch Generation** -- Concurrent multi-seed generation with asyncio for exploring prompt variations efficiently

---

## Architecture

```
                        +-------------------+
                        |   FastAPI Server   |
                        |     (Port 8005)    |
                        +--------+----------+
                                 |
                    +------------+------------+
                    |                         |
           +-------v--------+       +--------v--------+
           | Image Pipeline |       | Concepts Module |
           | (Orchestrator) |       | (Educational)   |
           +-------+--------+       +-----------------+
                   |                  VAE | GAN | Diffusion
          +--------+--------+         | Autoregressive
          |                 |
   +------v------+   +-----v-------+
   |   Prompt    |   |  Provider   |
   |  Enhancer   |   |  Dispatch   |
   |  (LLM)     |   +------+------+
   +-------------+          |
                   +--------+--------+
                   |                 |
            +------v------+  +------v--------+
            |   OpenAI    |  |  Replicate    |
            |  DALL-E 2/3 |  |  SD 3.5/FLUX  |
            +-------------+  +---------------+
                   |
            +------v------+
            |   Storage   |
            | Local | S3  |
            +------+------+
                   |
            +------v------+
            |   Gallery   |
            | (Metadata)  |
            +-------------+
```

---

## Quick Start

### Docker (Recommended)

```bash
# Build the image
docker build -t image-generation -f Dockerfile .

# Run with API keys
docker run -p 8005:8005 \
  -e IMG_GEN_OPENAI_API_KEY=sk-your-key-here \
  -e IMG_GEN_REPLICATE_API_TOKEN=r8_your-token-here \
  image-generation

# Verify it's running
curl http://localhost:8005/health
```

### Local Development

```bash
# Navigate to the project
cd projects/05-image-generation

# Create virtual environment
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Configure environment
cat > .env << 'EOF'
IMG_GEN_OPENAI_API_KEY=sk-your-key-here
IMG_GEN_REPLICATE_API_TOKEN=r8_your-token-here
IMG_GEN_DEFAULT_PROVIDER=openai
IMG_GEN_ENABLE_PROMPT_ENHANCEMENT=true
EOF

# Start the server
python -m image_generation.main

# Open the API docs
open http://localhost:8005/docs
```

---

## API Reference

### Generate a Single Image

```bash
curl -X POST http://localhost:8005/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A serene mountain lake at sunset with reflections",
    "style": "photorealistic",
    "width": 1024,
    "height": 1024,
    "guidance_scale": 8.0,
    "seed": 42
  }'
```

### Batch Generation (Multiple Seeds)

```bash
curl -X POST http://localhost:8005/api/v1/generate/batch \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cyberpunk cityscape at night",
    "count": 4,
    "style": "cinematic",
    "enhance_prompt": true
  }'
```

### Image-to-Image Transformation

```bash
curl -X POST http://localhost:8005/api/v1/img2img \
  -F "image=@input.png" \
  -F "prompt=Transform into a watercolor painting" \
  -F "strength=0.75" \
  -F "provider=replicate"
```

### Enhance a Prompt via LLM

```bash
curl -X POST http://localhost:8005/api/v1/enhance-prompt \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a cat on a beach"}'
```

### Browse the Gallery

```bash
# List all generated images
curl http://localhost:8005/api/v1/gallery?limit=20

# Get a specific image with base64 data
curl http://localhost:8005/api/v1/gallery/abc123def456
```

### Explore Generative Model Concepts

```bash
# List all model architectures
curl http://localhost:8005/api/v1/concepts

# Get detailed explanation of diffusion models
curl http://localhost:8005/api/v1/concepts/diffusion

# Also available: vae, gan, autoregressive
curl http://localhost:8005/api/v1/concepts/vae
```

### List Available Providers and Styles

```bash
curl http://localhost:8005/api/v1/providers
curl http://localhost:8005/api/v1/styles
```

---

## Implementation Deep Dive

### 1. Generative Model Concepts

The `/api/v1/concepts/{model_type}` endpoint serves rich, structured educational content about four generative architectures. Each explanation includes ASCII diagrams, mathematical formulations, training procedures, strengths/weaknesses, and key references.

**Variational Autoencoders (VAE)**
- Encoder maps input `x` to latent distribution parameters `(mu, log_var)`
- Reparameterization trick: `z = mu + sigma * epsilon` where `epsilon ~ N(0, I)`
- Decoder reconstructs from latent code: `p(x|z)`
- Loss: `L = Reconstruction Loss + beta * KL(q(z|x) || p(z))`
- Used as the latent compressor in Stable Diffusion (512x512 image to 64x64x4 latent)

**Generative Adversarial Networks (GAN)**
- Generator `G(z)` maps noise to images; Discriminator `D(x)` classifies real vs. fake
- Minimax game: `min_G max_D E[log D(x)] + E[log(1 - D(G(z)))]`
- Key innovations: spectral normalization, progressive growing, style-based generation
- Fast single-pass inference, but training instability and mode collapse are challenges

**Diffusion Models (DDPM / Stable Diffusion)**
- Forward process gradually adds Gaussian noise over T timesteps
- UNet backbone predicts noise at each step, conditioned on text via CLIP cross-attention
- Latent diffusion operates in compressed space (8x spatial reduction) for efficiency
- Classifier-free guidance: `eps = eps_uncond + scale * (eps_cond - eps_uncond)`
- Multiple schedulers available: DDPM (1000 steps), DDIM (50-100), Euler (20-30), DPM-Solver (20-25)

**Autoregressive Models (PixelCNN, DALL-E, Parti)**
- Generate images token-by-token using VQ-VAE codebooks (8192-16384 entries)
- Transformer decoder predicts image tokens conditioned on text tokens
- Exact log-likelihood training, but slow sequential generation

### 2. Text-to-Image Pipeline

The `ImagePipeline` class orchestrates the full generation flow:

```python
# Simplified pipeline flow
class ImagePipeline:
    async def generate(self, prompt, *, style="default", enhance_prompt=None, ...):
        # 1. Resolve style preset (guidance_scale, steps, negative_prompt)
        preset = get_preset(style)

        # 2. Optionally enhance prompt via LLM (GPT-4o-mini)
        if should_enhance:
            working_prompt = await self._enhancer.enhance(prompt)

        # 3. Apply style suffix and negative prompt
        final_prompt = working_prompt + preset.suffix

        # 4. Dispatch to configured provider
        result = await self._provider.generate(
            prompt=final_prompt,
            guidance_scale=preset.guidance_scale,
            steps=preset.steps,
            ...
        )
        return result
```

**Prompt Enhancement** uses GPT-4o-mini with a specialized system prompt to expand terse descriptions into detailed image prompts with composition, lighting, color palette, style, mood, and camera angle details. Falls back gracefully if no API key is configured.

**Style Presets** are frozen dataclasses mapping style names to generation parameters:

| Style | Suffix Keywords | Guidance Scale | Steps |
|-------|----------------|---------------|-------|
| Photorealistic | 8K UHD, Canon EOS R5, natural lighting | 8.0 | 35 |
| Cinematic | anamorphic lens flare, depth of field, 35mm film | 8.0 | 35 |
| Anime | cel shading, clean linework, trending on pixiv | 8.5 | 30 |
| Watercolor | soft washes, paper texture, wet-on-wet technique | 7.0 | 30 |
| Pixel Art | 16-bit, retro game style, limited color palette | 8.0 | 25 |

### 3. Multi-Provider Architecture

All providers implement the abstract `ImageProvider` base class:

```python
class ImageProvider(ABC):
    @abstractmethod
    async def generate(self, prompt, *, width, height, steps, guidance_scale, seed, model, **kwargs) -> GenerationResult: ...
    async def img2img(self, image, prompt, *, strength, ...) -> GenerationResult: ...
    async def inpaint(self, image, prompt, ...) -> GenerationResult: ...
    async def health_check(self) -> dict[str, Any]: ...
```

**OpenAI Provider** (`OpenAIProvider`):
- Supports DALL-E 2 (256/512/1024px, img2img, inpainting) and DALL-E 3 (1024px, text-to-image only)
- Snaps arbitrary dimensions to supported sizes (1024x1024, 1024x1792, 1792x1024)
- Returns both image URL and base64-encoded bytes
- Captures revised prompts from DALL-E 3

**Replicate Provider** (`ReplicateProvider`):
- Supports Stable Diffusion 3.5 Large/Medium and FLUX 1.1 Pro/Dev/Schnell
- Full parameter control: steps, guidance_scale, seed, negative_prompt
- Handles img2img via data URI encoding and `prompt_strength` parameter
- Downloads generated images from Replicate output URLs

### 4. Storage System

The storage layer uses the Strategy pattern with two backends:

- **`LocalStorageBackend`** -- Writes images to a configurable directory (`generated_images/`)
- **`S3StorageBackend`** -- Uploads to S3-compatible stores (AWS S3, MinIO) with configurable bucket, prefix, and region

The `Gallery` class maintains an in-memory index of image metadata (prompt, provider, model, dimensions, seed, style, timing) with filtering and pagination support.

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Framework | FastAPI 0.115+ | Async REST API with auto-generated OpenAPI docs |
| Image Generation | OpenAI API (DALL-E 2/3) | Cloud-hosted image generation |
| Image Generation | Replicate API (SD 3.5, FLUX) | Open-source model hosting |
| Prompt Enhancement | GPT-4o-mini | LLM-powered prompt rewriting |
| Storage | Local FS / S3 (boto3) | Image persistence |
| Validation | Pydantic 2.6+ | Request/response schemas |
| Image Processing | Pillow 10.4+ | Image format handling |
| HTTP Client | httpx 0.27+ | Async HTTP for downloads |
| Caching | Redis 5.0+ | Optional result caching |
| Logging | structlog 24.1+ | Structured JSON logging |
| Runtime | Python 3.11+ | Async/await, type hints |

---

## Project Structure

```
05-image-generation/
├── Dockerfile                         # Multi-stage production build
├── pyproject.toml                     # Dependencies and build config
├── k8s/
│   └── deployment.yaml                # Kubernetes deployment manifest
├── src/
│   └── image_generation/
│       ├── __init__.py
│       ├── main.py                    # Uvicorn entry point
│       ├── config.py                  # Settings (providers, sizes, storage, S3)
│       ├── api.py                     # FastAPI endpoints (generate, batch, img2img, gallery, concepts)
│       ├── pipeline.py                # ImagePipeline: prompt enhancement, style presets, batch generation
│       ├── storage.py                 # StorageBackend (Local/S3), Gallery metadata tracker
│       ├── models/
│       │   └── concepts.py            # Educational explainers: VAE, GAN, Diffusion, Autoregressive
│       └── providers/
│           ├── __init__.py            # Provider factory (get_provider, list_providers)
│           ├── base.py                # ImageProvider ABC, GenerationResult, GenerationStatus
│           ├── openai_provider.py     # OpenAI DALL-E 2/3 implementation
│           └── replicate_provider.py  # Replicate SD 3.5/FLUX implementation
└── tests/
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `IMG_GEN_OPENAI_API_KEY` | `""` | OpenAI API key for DALL-E |
| `IMG_GEN_REPLICATE_API_TOKEN` | `""` | Replicate API token for SD/FLUX |
| `IMG_GEN_DEFAULT_PROVIDER` | `openai` | Default provider: `openai` or `replicate` |
| `IMG_GEN_OPENAI_MODEL` | `dall-e-3` | Default OpenAI model |
| `IMG_GEN_REPLICATE_MODEL` | `stability-ai/stable-diffusion-3.5-large` | Default Replicate model |
| `IMG_GEN_DEFAULT_SIZE` | `1024x1024` | Default output image size |
| `IMG_GEN_ENABLE_PROMPT_ENHANCEMENT` | `true` | Enable LLM prompt rewriting |
| `IMG_GEN_STORAGE_BACKEND` | `local` | Storage: `local` or `s3` |
| `IMG_GEN_S3_BUCKET` | `""` | S3 bucket name |
| `IMG_GEN_PORT` | `8005` | Server port |

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Write tests for new functionality
4. Ensure all tests pass (`pytest`)
5. Submit a pull request

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
