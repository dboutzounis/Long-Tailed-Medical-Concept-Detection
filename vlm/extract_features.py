"""Extract and cache L2-normalized image embeddings of the four encoders (Section 4.4.1).

Writes <output-dir>/{train,val,test}_embs.pt and _labels.pt, plus test_keys.pkl. All splits
are read in shard order, so the caches of different encoders share the same row order.
Before writing, each encoder is checked to give identical embeddings for repeated inputs.

Usage (GPU node, weights already in the cache):
    python extract_features.py --config common/base_config.yaml --encoder medsiglip \
        --cache-dir $WORK/Long-Tailed-Medical-Concept-Detection/model_cache \
        --output-dir $WORK/Long-Tailed-Medical-Concept-Detection/artifacts/medsiglip_features
"""
import argparse
import base64
import io
import os
import pickle
import sys

import torch
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "common"))
from loaders import build_biomedclip_dataloader, load_config


def _as_embedding(out):
    """Return a [B, D] tensor from a tensor or a Hugging Face model output."""
    if torch.is_tensor(out):
        return out
    for attr in ("pooler_output", "image_embeds"):
        v = getattr(out, attr, None)
        if v is not None:
            return v
    lhs = getattr(out, "last_hidden_state", None)
    if lhs is not None:
        return lhs.mean(dim=1)
    raise TypeError(f"cannot extract embedding from {type(out)}")


def build_biomedclip(cache_dir, device):
    from open_clip import create_model_from_pretrained
    model, preprocess = create_model_from_pretrained(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        cache_dir=os.path.expandvars(cache_dir),
    )
    model = model.to(device).eval()

    def encode(imgs):
        return model.encode_image(imgs)  # image projection, 512-d

    return preprocess, encode


def build_rad_dino(cache_dir, device):
    from transformers import AutoImageProcessor, AutoModel
    repo = "microsoft/rad-dino"
    cd = os.path.expandvars(cache_dir)
    processor = AutoImageProcessor.from_pretrained(repo, cache_dir=cd)
    model = AutoModel.from_pretrained(repo, cache_dir=cd).to(device).eval()

    def preprocess(pil_image):
        return processor(images=pil_image, return_tensors="pt")["pixel_values"][0]

    def encode(imgs):
        out = model(pixel_values=imgs)
        emb = getattr(out, "pooler_output", None)  # CLS token after the final LayerNorm
        return emb if emb is not None else out.last_hidden_state[:, 0]

    return preprocess, encode


def build_medsiglip(cache_dir, device):
    from transformers import AutoImageProcessor, AutoModel
    repo = "google/medsiglip-448"
    cd = os.path.expandvars(cache_dir)
    processor = AutoImageProcessor.from_pretrained(repo, cache_dir=cd)
    model = AutoModel.from_pretrained(repo, cache_dir=cd).to(device).eval()

    def preprocess(pil_image):
        return processor(images=pil_image, return_tensors="pt")["pixel_values"][0]

    def encode(imgs):
        vm = getattr(model, "vision_model", model)
        return _as_embedding(vm(pixel_values=imgs))  # attention-pooling head

    return preprocess, encode


def build_medimageinsight():
    """Community port of MedImageInsight, expected under $WORK/.../third_party/MedImageInsights."""
    repo = os.path.join(os.environ.get("WORK", ""),
                        "Long-Tailed-Medical-Concept-Detection/third_party/MedImageInsights")
    sys.path.insert(0, repo)
    from medimageinsightmodel import MedImageInsight
    clf = MedImageInsight(
        model_dir=os.path.join(repo, "2024.09.27"),
        vision_model_name="medimageinsigt-v1.0.0.pt",
        language_model_name="language_model.pth",
    )
    clf.load_model()
    clf.model.eval()  # the port never calls eval(), which leaves DropPath active
    return clf


def assert_deterministic(encode_twice, name, min_cos=0.9999):
    """Refuse to extract if encoding the same images twice gives different embeddings."""
    a, b = encode_twice()
    a = torch.nn.functional.normalize(torch.as_tensor(a).float().flatten(1), dim=-1)
    b = torch.nn.functional.normalize(torch.as_tensor(b).float().flatten(1), dim=-1)
    cos = (a * b).sum(1)
    print(f"[determinism] {name}: repeat-encode cos min={cos.min():.6f} mean={cos.mean():.6f}")
    if cos.min() < min_cos:
        raise RuntimeError(f"{name} is not deterministic (min cos {cos.min():.4f}): "
                           "a stochastic layer is active. Refusing to write noisy features.")


def any_training(module):
    return [n for n, m in module.named_modules() if m.training]


def save_split(embs, labels, keys, split, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    embs, labels = torch.cat(embs, dim=0), torch.cat(labels, dim=0)
    torch.save(embs, os.path.join(output_dir, f"{split}_embs.pt"))
    torch.save(labels, os.path.join(output_dir, f"{split}_labels.pt"))
    if keys is not None:
        with open(os.path.join(output_dir, f"{split}_keys.pkl"), "wb") as f:
            pickle.dump(keys, f)
    print(f"Saved {split}: embeddings {tuple(embs.shape)}, labels {tuple(labels.shape)}")


def extract_and_save(encode, loader, device, split, output_dir, return_keys=False):
    all_embs, all_labels, all_keys = [], [], []
    use_amp = device.type == "cuda"
    checked = False
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"  {split}"):
            imgs, labels = batch[0].to(device), batch[1]
            if return_keys:
                all_keys.extend(batch[2])

            if not checked:
                probe = imgs[:8]

                def _twice():
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                        return encode(probe).float().cpu(), encode(probe).float().cpu()

                assert_deterministic(_twice, f"{split} encoder")
                checked = True

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                emb = encode(imgs)
            all_embs.append(torch.nn.functional.normalize(emb.float(), dim=-1).cpu())
            all_labels.append(labels.cpu())
    save_split(all_embs, all_labels, all_keys if return_keys else None, split, output_dir)


def _pil_to_b64_png(pil_image):
    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, format="PNG")
    return base64.encodebytes(buf.getvalue()).decode("utf-8")


def extract_medimageinsight(clf, raw_loader, split, output_dir, return_keys=False, mii_batch=32):
    """The port takes base64-encoded images and does its own preprocessing."""
    still_training = any_training(clf.model)
    if still_training:
        raise RuntimeError(f"MedImageInsight has modules in train mode (e.g. {still_training[:3]})")
    all_embs, all_labels, all_keys = [], [], []
    checked = False
    for batch in tqdm(raw_loader, desc=f"  {split}"):
        pil_list, labels = batch[0], batch[1]
        if return_keys:
            all_keys.extend(batch[2])
        if not checked:
            b64_probe = [_pil_to_b64_png(im) for im in pil_list[:8]]
            assert_deterministic(
                lambda: (clf.encode(images=b64_probe)["image_embeddings"],
                         clf.encode(images=b64_probe)["image_embeddings"]),
                f"{split} MedImageInsight")
            checked = True
        chunk_embs = []
        for s in range(0, len(pil_list), mii_batch):
            b64 = [_pil_to_b64_png(im) for im in pil_list[s:s + mii_batch]]
            chunk_embs.append(torch.as_tensor(clf.encode(images=b64)["image_embeddings"]).float())
        all_embs.append(torch.nn.functional.normalize(torch.cat(chunk_embs), dim=-1).cpu())
        all_labels.append(torch.as_tensor(labels).cpu())
    save_split(all_embs, all_labels, all_keys if return_keys else None, split, output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(HERE, "common", "base_config.yaml"))
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--encoder", default="biomedclip",
                        choices=["biomedclip", "rad-dino", "medsiglip", "medimageinsight"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = [("train", False), ("val", False), ("test", True)]

    if args.encoder == "medimageinsight":
        clf = build_medimageinsight()
        for split, keys in splits:
            loader = build_biomedclip_dataloader(split, cfg, None, return_keys=keys, raw=True)
            extract_medimageinsight(clf, loader, split, args.output_dir, return_keys=keys)
        return

    builders = {"biomedclip": build_biomedclip, "rad-dino": build_rad_dino,
                "medsiglip": build_medsiglip}
    preprocess, encode = builders[args.encoder](args.cache_dir, device)
    for split, keys in splits:
        loader = build_biomedclip_dataloader(split, cfg, preprocess, return_keys=keys)
        extract_and_save(encode, loader, device, split, args.output_dir, return_keys=keys)


if __name__ == "__main__":
    main()
