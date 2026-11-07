"""Encoder backends for BiomedCLIP, MedSigLIP, MedImageInsight and Rad-DINO.

Each backend exposes:
  visual            the image encoder (nn.Module)
  encode_image(x)   (B, D) image embedding, before L2 normalization
  eval_transform    PIL -> tensor, the same preprocessing as feature extraction
  text_embeddings() (C, D) L2-normalized concept embeddings (four-template prompt ensemble)
  drop_text()       frees the text encoder

All weights must already be on disk (compute nodes are offline).
"""
import os
import sys

import torch
import torch.nn.functional as F

from common import PROMPT_TEMPLATES, prompts_for

BIOMEDCLIP_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
MEDSIGLIP_ID = "google/medsiglip-448"
RADDINO_ID = "microsoft/rad-dino"


class Backend:
    name = "base"
    amp = True
    has_text = True
    eval_transform = None

    def encode_image(self, x):
        raise NotImplementedError

    @torch.no_grad()
    def _encode_prompts(self, prompts, device):
        raise NotImplementedError

    @torch.no_grad()
    def text_embeddings(self, names, device, batch_size=256):
        per_template = []
        for tpl in PROMPT_TEMPLATES:
            prompts = prompts_for(names, tpl)
            chunks = [self._encode_prompts(prompts[i:i + batch_size], device)
                      for i in range(0, len(prompts), batch_size)]
            per_template.append(F.normalize(torch.cat(chunks).float(), dim=-1))
        return F.normalize(torch.stack(per_template).mean(0), dim=-1).cpu()

    def drop_text(self):
        pass


class _HFImageTransform:
    """Picklable PIL -> pixel_values wrapper around a Hugging Face image processor."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, pil):
        return self.processor(images=pil, return_tensors="pt")["pixel_values"][0]


class BiomedCLIPBackend(Backend):
    name = "biomedclip"

    def __init__(self, cache_dir, **_):
        from open_clip import create_model_from_pretrained, get_tokenizer
        model, preprocess = create_model_from_pretrained(BIOMEDCLIP_ID, cache_dir=cache_dir)
        try:
            self.tokenizer = get_tokenizer(BIOMEDCLIP_ID, cache_dir=cache_dir)
        except TypeError:  # older open_clip without cache_dir
            self.tokenizer = get_tokenizer(BIOMEDCLIP_ID)
        self.model = model
        self.visual = model.visual
        self.eval_transform = preprocess

    def encode_image(self, x):
        return self.visual(x)

    @torch.no_grad()
    def _encode_prompts(self, prompts, device):
        self.model.to(device).eval()
        return self.model.encode_text(self.tokenizer(prompts).to(device))

    def drop_text(self):
        self.model.text = None
        self.model = None


class MedSigLIPBackend(Backend):
    name = "medsiglip"

    def __init__(self, cache_dir, grad_ckpt=True, **_):
        from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
        model = AutoModel.from_pretrained(MEDSIGLIP_ID, cache_dir=cache_dir)
        processor = AutoImageProcessor.from_pretrained(MEDSIGLIP_ID, cache_dir=cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(MEDSIGLIP_ID, cache_dir=cache_dir)
        if grad_ckpt:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model = model
        self.visual = model.vision_model
        self.tokenizer = tokenizer
        self.eval_transform = _HFImageTransform(processor)

    def encode_image(self, x):
        return self.visual(pixel_values=x).pooler_output  # attention-pooling head

    @torch.no_grad()
    def _encode_prompts(self, prompts, device):
        self.model.text_model.to(device).eval()
        tok = self.tokenizer(prompts, padding="max_length", max_length=64,
                             truncation=True, return_tensors="pt").to(device)
        return self.model.text_model(input_ids=tok["input_ids"]).pooler_output

    def drop_text(self):
        self.model.text_model = None
        self.model = None


class _MIITransform:
    def __init__(self, preprocess):
        self.preprocess = preprocess

    def __call__(self, pil):
        return self.preprocess(pil.convert("RGB"))


class MedImageInsightBackend(Backend):
    """Community port of MedImageInsight (https://huggingface.co/lion-ai/MedImageInsights)."""
    name = "medimageinsight"

    def __init__(self, cache_dir, repo_dir=None, model_dir=None,
                 vision_model_name="medimageinsigt-v1.0.0.pt",
                 language_model_name="language_model.pth", **_):
        repo_dir = repo_dir or os.path.join(
            os.environ.get("WORK", ""), "Long-Tailed-Medical-Concept-Detection/third_party/MedImageInsights")
        model_dir = model_dir or os.path.join(repo_dir, "2024.09.27")
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)
        from medimageinsightmodel import MedImageInsight
        clf = MedImageInsight(model_dir=model_dir, vision_model_name=vision_model_name,
                              language_model_name=language_model_name)
        clf.load_model()
        clf.model.eval()  # the port leaves DropPath active otherwise
        self.model = clf.model
        self.model.image_encoder.enable_checkpoint = False
        self.visual = clf.model
        self.tokenizer = clf.tokenize
        self.max_length = clf.max_length
        self.eval_transform = _MIITransform(clf.preprocess)

    def encode_image(self, x):
        return self.visual.encode_image(x, norm=False)

    @torch.no_grad()
    def _encode_prompts(self, prompts, device):
        self.model.to(device).eval()
        tok = self.tokenizer(prompts, padding="max_length", max_length=self.max_length,
                             truncation=True, return_tensors="pt")
        tok = {k: v.to(device) for k, v in tok.items()}
        return self.model.encode_text(tok, norm=False)

    def drop_text(self):
        self.model.lang_encoder = None
        self.model = None


class RadDINOBackend(Backend):
    name = "raddino"
    has_text = False

    def __init__(self, cache_dir, grad_ckpt=True, **_):
        from transformers import AutoImageProcessor, AutoModel
        model = AutoModel.from_pretrained(RADDINO_ID, cache_dir=cache_dir)
        processor = AutoImageProcessor.from_pretrained(RADDINO_ID, cache_dir=cache_dir)
        if grad_ckpt:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model = model
        self.visual = model
        self.eval_transform = _HFImageTransform(processor)

    def encode_image(self, x):
        out = self.visual(pixel_values=x)
        emb = getattr(out, "pooler_output", None)  # CLS token after the final LayerNorm
        return emb if emb is not None else out.last_hidden_state[:, 0]

    def text_embeddings(self, names, device, batch_size=256):
        raise RuntimeError("Rad-DINO has no text encoder")


BACKENDS = {"biomedclip": BiomedCLIPBackend, "medsiglip": MedSigLIPBackend,
            "medimageinsight": MedImageInsightBackend, "raddino": RadDINOBackend}
MODEL_CHOICES = list(BACKENDS)


def build_backend(name, cache_dir, model_cfg=None):
    """Build a backend; model_cfg is the models.<name> block of config.yaml."""
    kwargs = {k: v for k, v in dict(model_cfg or {}).items()
              if k in ("grad_ckpt", "repo_dir", "model_dir",
                       "vision_model_name", "language_model_name") and v is not None}
    return BACKENDS[name](cache_dir, **kwargs)
