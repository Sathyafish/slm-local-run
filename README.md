# slm-local-run — Gemma 3n E2B IT on your laptop

Run **[google/gemma-3n-E2B-it](https://huggingface.co/google/gemma-3n-E2B-it)** locally with Hugging Face Transformers.

---

## Prerequisites

- **Python 3.10+**
- **Hugging Face account** — the model is gated; open [google/gemma-3n-E2B-it](https://huggingface.co/google/gemma-3n-E2B-it), sign in, and **accept** Google’s Gemma terms so files can download
- **RAM / GPU** — an **NVIDIA GPU** or **Apple Silicon Mac** is recommended; **CPU-only** works but is slow and memory-heavy
- **`transformers` 4.53.0+** (needed for Gemma 3n)

---

## Environment setup

**1. Create and activate a virtual environment**

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

**2. Install PyTorch** for your machine (pick one command from [pytorch.org](https://pytorch.org/get-started/locally/) — e.g. CUDA for an NVIDIA laptop, default pip install for Mac CPU/MPS).

**3. Install the rest**

```bash
pip install -U pip
pip install -U "transformers>=4.53.0" accelerate safetensors pillow requests
```

**4. Log in to Hugging Face** (so the gated model can download)

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login
```

Paste a **read** token from [Hugging Face → Settings → Access Tokens](https://huggingface.co/settings/tokens).  
Alternatively, set `HF_TOKEN` in your environment (do not commit it to git).

---

## Run the model

Save this as `run_gemma.py` (or paste into a notebook) and run: `python run_gemma.py`

```python
import torch
from transformers import pipeline

# Pick device: CUDA → MPS (Mac) → CPU
if torch.cuda.is_available():
    device, dtype = "cuda", torch.bfloat16
elif torch.backends.mps.is_available():
    device, dtype = "mps", torch.float16
else:
    device, dtype = "cpu", torch.float32

pipe = pipeline(
    "image-text-to-text",
    model="google/gemma-3n-e2b-it",
    device=device,
    torch_dtype=dtype,
)

messages = [
    {
        "role": "user",
        "content": [{"type": "text", "text": "Say hello in one short sentence."}],
    },
]

result = pipe(text=messages, max_new_tokens=128)
print(result[0]["generated_text"][-1]["content"])
```

The first run **downloads the model** (large); wait until it finishes.

---

## If something fails

- **401 / cannot access repo** — Accept the license on the model page and run `huggingface-cli login` again.
- **Out of memory** — Close other apps, lower `max_new_tokens`, or use a machine with more GPU memory / unified memory.
- **Slow on CPU** — Expected; use a GPU or Apple Silicon if you can.

---

## Links

[Model card](https://huggingface.co/google/gemma-3n-E2B-it) · [Gemma terms](https://ai.google.dev/gemma/terms) · [Gemma 3n docs](https://ai.google.dev/gemma/docs/gemma-3n)
