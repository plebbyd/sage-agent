from __future__ import annotations

"""
tools/detectors.py — Lazy-loaded detection backends for PTZ viewer + MSA tools.

Each detector is optional. Missing packages are detected at import time;
the detector reports itself as unavailable and raises a clear error when called.

Supported backends:
  YOLO       — pip install ultralytics (supports batched + tiled inference)
  BioCLIP 2  — pip install open_clip_torch torch torchvision opencv-python numpy
               (OpenCLIP hub ``imageomics/bioclip-2`` + TreeOfLife-200M text
               embeddings; ``bioclip`` and ``bioclip2`` both resolve here.
               Doubles as an object detector via Grad-CAM localization; the
               taxon filter matches any ranked candidate, not only top-1)
  Gemma 4    — Ollama with a Gemma 4 vision+text tag (default ``gemma4:31b``,
               e.g. ``ollama pull gemma4:31b``); set ``OLLAMA_HOST`` /
               ``GEMMA4_OLLAMA_MODEL`` as needed. The same model is used by
               the MSA agent loop (``msa/model.py`` Ollama backend).

Tiled inference: ``detect(image, model, tile=True)`` (or ``detect_tiled``) slices
a large frame into a grid of model-sized tiles, runs inference on each (batched
for YOLO), and merges boxes back to original coordinates with NMS — preserving
small-object resolution that a single down-scaled pass would lose.

All detectors are singletons: loaded once on first use, then cached.
"""

import base64
import io
import logging
import os
import re
import sys
import time

logger = logging.getLogger(__name__)


def _clear_torchvision_modules() -> None:
    """Remove torchvision from ``sys.modules`` so a later import is clean.

    Use **only** when ``torchvision`` is half-loaded (no ``extension``). Do **not**
    call after a successful import: PyTorch keeps torchvision C++ ops registered;
    re-importing causes duplicate-kernel errors (e.g. ``roi_align`` Meta dispatch).
    """
    for k in list(sys.modules.keys()):
        if k == "torchvision" or k.startswith("torchvision."):
            del sys.modules[k]


def _torchvision_is_partial() -> bool:
    """True if ``torchvision`` is stuck half-loaded (no ``extension`` sub-module)."""
    tv = sys.modules.get("torchvision")
    if tv is None:
        return False
    try:
        return not hasattr(tv, "extension")
    except Exception:
        return True


def _reset_torchvision_if_partial() -> None:
    """Drop broken torchvision so ultralytics / open_clip can import it again."""
    if _torchvision_is_partial():
        _clear_torchvision_modules()


def _normalize_ollama_host(raw: str) -> str:
    raw = raw.strip().rstrip("/")
    if not raw.startswith(("http://", "https://")):
        raw = "http://" + raw
    return raw


def _ollama_probe() -> bool:
    """True if Ollama responds at ``OLLAMA_HOST`` (default ``http://127.0.0.1:11434``)."""
    try:
        import requests

        base = _normalize_ollama_host(os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
        r = requests.get(f"{base}/api/tags", timeout=1.5)
        return r.status_code == 200
    except Exception:
        return False


_GEMMA_THINK_BLOCK = re.compile(
    r"<\|channel>thought\s*\n.*?<channel\|>",
    re.DOTALL | re.IGNORECASE,
)


def _strip_gemma_thinking(text: str) -> str:
    """Remove Gemma 4 ``<|channel>thought`` … ``<channel|>`` blocks when present."""
    if not text:
        return text
    t = _GEMMA_THINK_BLOCK.sub("", text)
    return t.strip()


# ---------------------------------------------------------------------------
# BioCLIP taxon filter (lineage strings from TreeOfLife-style labels)
# ---------------------------------------------------------------------------

def _canonicalize_taxon_token(t: str) -> str:
    """Normalize one rank token (kingdom synonyms like Metazoa ↔ Animalia)."""
    tt = (t or "").strip().lower()
    if not tt:
        return ""
    if tt in ("metazoa", "animalia"):
        return "animalia"
    return tt


def _normalize_taxon_query(s: str) -> list[str]:
    """Split user input on whitespace, commas, semicolons into rank tokens."""
    if not s or not str(s).strip():
        return []
    parts = re.split(r"[,;\s]+", str(s).strip())
    return [_canonicalize_taxon_token(p) for p in parts if p]


def _strip_taxon_parenthetical(label: str) -> str:
    """Remove trailing ``(common name)`` from BioCLIP species labels."""
    return re.sub(r"\s*\([^)]*\)\s*$", "", (label or "").strip())


def _bioclip_debug_requested(kwargs: dict | None) -> bool:
    """True if API ``bioclip_debug`` / ``debug`` or env ``BIOCLIP_DEBUG`` is set."""
    if not kwargs:
        return False
    if kwargs.get("bioclip_debug") or kwargs.get("debug"):
        return True
    return os.environ.get("BIOCLIP_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")


def _taxon_lineage_matches(query_tokens: list[str], lineage: str) -> bool:
    """True if ``lineage`` matches the query: substring, or token-prefix after canonicalization.

    Supports GN/TaxonoPy-style kingdom variants (e.g. Metazoa vs Animalia) for comparison only;
    see https://imageomics.github.io/TaxonoPy/user-guide/quick-reference/
    """
    if not query_tokens:
        return True
    base = _strip_taxon_parenthetical(lineage)
    ln_lower = base.lower()
    qjoin = " ".join(query_tokens)
    if qjoin in ln_lower:
        return True
    lin_tokens = [_canonicalize_taxon_token(x) for x in re.split(r"\s+", ln_lower) if x]
    qtok = list(query_tokens)
    if len(lin_tokens) < len(qtok):
        return False
    return lin_tokens[: len(qtok)] == qtok


# ---------------------------------------------------------------------------
# Dependency probing
# ---------------------------------------------------------------------------

_HAS_TORCH = False
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    pass

_HAS_YOLO = False
try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except ImportError:
    pass

# NumPy + OpenCV for BioCLIP (open_clip/torchvision load lazily in
# _BioCLIPDetector.__init__; do not import torchvision here — a mismatched
# torch/torchvision pair would mark bioclip unavailable even though the package
# is installed; runtime errors are returned from detect() instead).
np = None  # type: ignore[assignment]
cv2 = None  # type: ignore[assignment]
try:
    import numpy as np
except ImportError:
    pass
try:
    import cv2
except ImportError:
    pass

_HAS_BIOCLIP = False
if _HAS_TORCH and np is not None and cv2 is not None:
    try:
        import importlib.util

        _HAS_BIOCLIP = importlib.util.find_spec("open_clip") is not None
    except Exception:
        _HAS_BIOCLIP = False

_HAS_GEMMA4 = _ollama_probe()


def available_models() -> dict:
    """Return ``{model_name: bool}`` for each backend."""
    return {
        "yolo": _HAS_YOLO,
        "bioclip": _HAS_BIOCLIP,
        "bioclip2": _HAS_BIOCLIP,
        "gemma4": _HAS_GEMMA4,
    }


# ``bioclip`` now resolves to BioCLIP 2 weights; keep the alias explicit.
_MODEL_ALIASES = {"bioclip2": "bioclip", "bioclip-2": "bioclip"}


def _canonical_model(model: str) -> str:
    m = str(model or "").strip().lower()
    return _MODEL_ALIASES.get(m, m)


# ---------------------------------------------------------------------------
# YOLO
# ---------------------------------------------------------------------------

class _YOLODetector:
    def __init__(self, model_name: str = "yolo11n"):
        self.model_name = model_name
        if model_name.startswith("yolov8"):
            path = model_name
        elif model_name.startswith("yolo11"):
            path = f"yolo11{model_name[-1]}"
        else:
            path = model_name
        self.model = YOLO(path)
        logger.info("YOLO loaded: %s", model_name)

    @staticmethod
    def _normalize_targets(targets):
        if isinstance(targets, str):
            return [t.strip().lower() for t in targets.split(",")]
        return [t.lower() for t in targets]

    @staticmethod
    def _to_np(image):
        from PIL import Image as _PILImage
        import numpy as _np

        return _np.array(image) if isinstance(image, _PILImage.Image) else image

    @classmethod
    def _parse_result(cls, r, targets):
        dets = []
        for box in r.boxes:
            name = r.names[int(box.cls[0])]
            if "*" in targets or name.lower() in targets:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                dets.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "label": name,
                    "confidence": round(float(box.conf[0]), 3),
                })
        return dets

    def detect(self, image, targets="*"):
        targets = self._normalize_targets(targets)
        img_np = self._to_np(image)
        # NMS lazily imports torchvision; only drop a *partial* stub (never reload
        # after a full torchvision import — torch keeps ops; reload → duplicate kernel).
        _reset_torchvision_if_partial()
        results = self.model(img_np, verbose=False)
        detections = []
        for r in results:
            detections.extend(self._parse_result(r, targets))
        return detections

    def detect_batch(self, images, targets="*"):
        """Run a single batched forward pass over many images/tiles.

        Returns a list (one entry per input image) of detection lists, so
        callers can offset per-tile boxes back into a parent frame.
        """
        targets = self._normalize_targets(targets)
        batch = [self._to_np(im) for im in images]
        if not batch:
            return []
        _reset_torchvision_if_partial()
        results = self.model(batch, verbose=False)
        return [self._parse_result(r, targets) for r in results]


# ---------------------------------------------------------------------------
# BioCLIP
# ---------------------------------------------------------------------------

class _BioCLIPDetector:
    """BioCLIP 2 species classifier + object detector (adapted from ptz-app).

    Uses the BioCLIP 2 weights (``hf-hub:imageomics/bioclip-2``, a ViT-L/14
    backbone) with the matching TreeOfLife-200M text embeddings. Localization
    for the object-detector path is produced with Grad-CAM over a mid/late
    transformer block (auto-selected for the loaded depth).
    """

    RANKS = ("Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species")

    # OpenCLIP hub id + the dataset repo that ships the matching text embeddings.
    MODEL_HUB_ID = "hf-hub:imageomics/bioclip-2"
    TOL_EMB_REPO = "imageomics/TreeOfLife-200M"

    def __init__(self):
        import json as _json

        # Only clear a stuck partial torchvision; do not unload a working import.
        _reset_torchvision_if_partial()
        try:
            import open_clip
            from torchvision import transforms
        except Exception as exc:
            raise RuntimeError(
                "BioCLIP 2 failed to import open_clip/torchvision. "
                "Install matching torch+torchvision wheels (same CUDA/CPU line), then "
                "pip install open_clip_torch opencv-python huggingface_hub. "
                f"Original error: {exc}"
            ) from exc

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # CLIP image side; ViT-L/14 @ 224 -> 16x16 patch grid.
        self.input_size = 224

        cache_dir = os.environ.get("HF_HOME", None)
        self.model, _, _ = open_clip.create_model_and_transforms(
            self.MODEL_HUB_ID, cache_dir=cache_dir,
        )
        self.model = self.model.to(self.device).eval()

        # Standard CLIP preprocessing (BioCLIP 2 trained at 224x224).
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.input_size, self.input_size), antialias=True),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ])

        npy_path, json_path = self._find_embeddings()
        # Embeddings ship as [emb_dim, num_species]; cast to float32 so the
        # image/text matmul never trips on a float16-on-CPU dtype mismatch.
        self.txt_emb = (
            torch.from_numpy(np.load(npy_path, mmap_mode="r"))
            .to(self.device)
            .float()
        )
        with open(json_path) as f:
            self.txt_names = _json.load(f)
        # Auto-pick a Grad-CAM hook layer ~70% deep in the visual transformer.
        self._cam_layer = self._select_cam_layer()
        logger.info(
            "BioCLIP 2 loaded — %d species embeddings (cam layer: %s)",
            self.txt_emb.shape[1], self._cam_layer or "n/a",
        )

    # -- helpers --

    def _select_cam_layer(self) -> str | None:
        """Name of a mid/late ``visual.transformer.resblocks.N`` block.

        Depth differs across BioCLIP versions (ViT-B/16 = 12 blocks, the
        BioCLIP 2 ViT-L/14 = 24); pick ~70% deep for a meaningful heatmap.
        """
        idxs = []
        for name, _ in self.model.named_modules():
            m = re.fullmatch(r"visual\.transformer\.resblocks\.(\d+)", name)
            if m:
                idxs.append(int(m.group(1)))
        if not idxs:
            return None
        n = max(idxs) + 1
        target = min(max(int(round(0.7 * (n - 1))), 0), n - 1)
        return f"visual.transformer.resblocks.{target}"

    def _find_embeddings(self):
        import os
        from huggingface_hub import hf_hub_download as _dl

        # Local override (BioCLIP-2-specific names so v1 files never collide).
        for base in [os.getcwd(), ".", "/app"]:
            np_ = os.path.join(base, "txt_emb_bioclip2.npy")
            js_ = os.path.join(base, "txt_emb_bioclip2.json")
            if os.path.exists(np_) and os.path.exists(js_):
                return np_, js_
        # TreeOfLife-200M ships the BioCLIP 2 text embeddings; txt_emb_species.*
        # are duplicates of txt_emb_bioclip-2.* kept for tooling compatibility.
        repo = self.TOL_EMB_REPO
        return (
            _dl(repo, "embeddings/txt_emb_species.npy", repo_type="dataset"),
            _dl(repo, "embeddings/txt_emb_species.json", repo_type="dataset"),
        )

    @staticmethod
    def _format_name(taxon, common):
        name = " ".join(taxon)
        return f"{name} ({common})" if common else name

    # -- inference --

    def classify(self, image, rank="Class", top_k=5):
        """Top-k classification at a taxonomic rank."""
        import collections
        import heapq
        import torch.nn.functional as F

        rank_idx = self.RANKS.index(rank)
        img_t = self.preprocess(image).to(self.device).unsqueeze(0)

        with torch.no_grad():
            feats = F.normalize(self.model.encode_image(img_t), dim=-1)
            logits = (self.model.logit_scale.exp() * feats @ self.txt_emb).squeeze()
            probs = F.softmax(logits, dim=0)

        if rank_idx + 1 == len(self.RANKS):
            topk = probs.topk(top_k)
            return [
                (self._format_name(*self.txt_names[i]), float(p))
                for i, p in zip(topk.indices, topk.values)
            ]

        agg = collections.defaultdict(float)
        for i in torch.nonzero(probs > 1e-9).squeeze():
            agg[" ".join(self.txt_names[i][0][: rank_idx + 1])] += probs[i]
        topk_names = heapq.nlargest(top_k, agg, key=agg.get)
        return [(n, float(agg[n])) for n in topk_names]

    def detect(
        self,
        image,
        rank="Class",
        target_taxon="",
        min_confidence=0.1,
        *,
        out_debug: dict | None = None,
    ):
        """Classification + Grad-CAM localization → bounding boxes.

        Pass ``out_debug={}`` to receive a filled dict (also returned in JSON when
        ``bioclip_debug`` is set on /detect). Set env ``BIOCLIP_DEBUG=1`` for
        matching lines in ``logs`` / stderr.
        """
        import collections
        import heapq
        import torch.nn.functional as F

        def _dbg(msg: str, **extra: object) -> None:
            if out_debug is not None:
                out_debug.setdefault("log", []).append({"msg": msg, **extra})
            if os.environ.get("BIOCLIP_DEBUG", "").strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            ) or out_debug is not None:
                logger.info("BioCLIP %s %s", msg, extra)

        orig_w, orig_h = image.size
        rank_idx = self.RANKS.index(rank)
        query_tokens = _normalize_taxon_query(target_taxon)

        if out_debug is not None:
            out_debug.update(
                {
                    "openclip_model": self.MODEL_HUB_ID,
                    "note": "BioCLIP 2 (ViT-L/14) weights + TreeOfLife-200M embeddings via OpenCLIP",
                    "rank": rank,
                    "rank_idx": rank_idx,
                    "raw_target_taxon": (target_taxon or "").strip(),
                    "query_tokens": query_tokens,
                    "min_confidence": min_confidence,
                    "image_size": [orig_w, orig_h],
                }
            )

        img_t = self.preprocess(image).to(self.device).unsqueeze(0)
        img_t.requires_grad = True

        feats = F.normalize(self.model.encode_image(img_t), dim=-1)
        logits = (self.model.logit_scale.exp() * feats @ self.txt_emb).squeeze()
        probs = F.softmax(logits, dim=0)

        if out_debug is not None:
            out_debug["num_species_embeddings"] = int(probs.numel())
            out_debug["prob_max"] = float(probs.max().item())
            out_debug["prob_entropy"] = float((-probs * probs.clamp_min(1e-12).log()).sum().item())

        top_label: str | None = None
        top_conf = 0.0
        top_idx = 0

        if rank_idx + 1 == len(self.RANKS):
            k = min(50, int(probs.numel()))
            topk = probs.topk(k)
            if out_debug is not None:
                out_debug["branch"] = "species"
                n = min(5, int(topk.indices.size(0)))
                out_debug["top_species_logits"] = [
                    {
                        "label": self._format_name(*self.txt_names[int(topk.indices[j])]),
                        "prob": float(topk.values[j]),
                    }
                    for j in range(n)
                ]
            if not query_tokens:
                if float(topk.values[0]) < min_confidence:
                    if out_debug is not None:
                        out_debug["exit"] = "top_species_below_min_confidence"
                        out_debug["top1_prob"] = float(topk.values[0])
                    _dbg("exit: top species below min_confidence", top1=float(topk.values[0]))
                    return []
                top_idx = int(topk.indices[0])
                top_label = self._format_name(*self.txt_names[top_idx])
                top_conf = float(topk.values[0])
            else:
                for j in range(topk.indices.size(0)):
                    idx = int(topk.indices[j])
                    conf = float(topk.values[j])
                    if conf < min_confidence:
                        break
                    label = self._format_name(*self.txt_names[idx])
                    if _taxon_lineage_matches(query_tokens, label):
                        top_idx = idx
                        top_label = label
                        top_conf = conf
                        break
                if top_label is None:
                    for j in range(topk.indices.size(0)):
                        idx = int(topk.indices[j])
                        label = self._format_name(*self.txt_names[idx])
                        if _taxon_lineage_matches(query_tokens, label):
                            top_idx = idx
                            top_label = label
                            top_conf = float(topk.values[j])
                            break
                if top_label is None:
                    if out_debug is not None:
                        out_debug["exit"] = "no_species_matches_taxon_filter"
                    _dbg("exit: no species label matched filter")
                    return []
        else:
            agg = collections.defaultdict(float)
            idx_map = collections.defaultdict(list)
            flat = torch.nonzero(probs > 1e-9).flatten()
            if flat.numel() == 0:
                if out_debug is not None:
                    out_debug["exit"] = "no_species_prob_mass"
                _dbg("exit: zero prob mass")
                return []
            for i in flat:
                ii = int(i)
                rn = " ".join(self.txt_names[ii][0][: rank_idx + 1])
                agg[rn] += probs[ii]
                idx_map[rn].append(ii)

            if out_debug is not None:
                out_debug["branch"] = "aggregated_rank"
                cand = sorted(agg.items(), key=lambda x: -x[1])[:15]
                out_debug["top_lineages"] = [
                    {"lineage": rn, "mass": float(m)} for rn, m in cand
                ]
                out_debug["num_distinct_lineages"] = len(agg)

            if not query_tokens:
                topk_names = heapq.nlargest(5, agg, key=agg.get)
                top_label = topk_names[0]
                top_conf = float(agg[top_label])
                top_idx = max(idx_map[top_label], key=lambda ix: probs[ix].item())
            else:
                candidates = sorted(agg.items(), key=lambda x: -x[1])
                chosen = None
                for rn, mass in candidates:
                    if mass < min_confidence:
                        continue
                    if _taxon_lineage_matches(query_tokens, rn):
                        chosen = rn
                        break
                if chosen is None:
                    for rn, mass in candidates:
                        if _taxon_lineage_matches(query_tokens, rn):
                            chosen = rn
                            break
                if chosen is None:
                    if out_debug is not None:
                        out_debug["exit"] = "no_lineage_matches_taxon_filter"
                    _dbg("exit: no aggregated lineage matched filter")
                    return []
                top_label = chosen
                top_conf = float(agg[chosen])
                top_idx = max(idx_map[chosen], key=lambda ix: probs[ix].item())

        if top_label is None:
            if out_debug is not None:
                out_debug["exit"] = "no_label_selected"
            _dbg("exit: top_label is None")
            return []
        if top_conf < min_confidence:
            if out_debug is not None:
                out_debug["exit"] = "chosen_below_min_confidence"
                out_debug["chosen_label"] = top_label
                out_debug["chosen_conf"] = top_conf
            _dbg("exit: below min_confidence", label=top_label, conf=top_conf)
            return []

        if out_debug is not None:
            out_debug["chosen_label"] = top_label
            out_debug["chosen_conf"] = top_conf
            out_debug["chosen_species_idx"] = int(top_idx)

        cam = self._grad_cam(img_t, top_idx)
        if cam is None:
            if out_debug is not None:
                out_debug["grad_cam"] = "unavailable"
                out_debug["exit"] = "fallback_full_frame"
                out_debug["hint"] = "Grad-CAM hook failed (model depth or attention layout); using full-frame box"
            _dbg("grad_cam None, full-frame fallback")
            return [{"bbox": [0, 0, orig_w, orig_h], "label": top_label,
                      "confidence": round(top_conf, 3)}]

        cam_rs = cv2.resize(
            cam, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR
        )
        raw_boxes = self._heatmap_to_bboxes(cam_rs, threshold=0.4)
        if not raw_boxes:
            if out_debug is not None:
                out_debug["grad_cam"] = "ok"
                out_debug["heatmap_boxes"] = 0
                out_debug["exit"] = "no_heatmap_components"
                out_debug["hint"] = "Heatmap produced no connected blobs; using full-frame box"
            _dbg("heatmap empty, full-frame fallback")
            return [{"bbox": [0, 0, orig_w, orig_h], "label": top_label,
                      "confidence": round(top_conf, 3)}]

        if out_debug is not None:
            out_debug["grad_cam"] = "ok"
            out_debug["heatmap_boxes"] = len(raw_boxes)

        sx, sy = orig_w / self.input_size, orig_h / self.input_size
        dets = [
            {"bbox": [int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)],
             "label": top_label, "confidence": round(top_conf, 3)}
            for x1, y1, x2, y2, _, _ in raw_boxes
        ]
        if out_debug is not None:
            out_debug["exit"] = "ok"
            out_debug["num_detections"] = len(dets)
        return dets

    def _grad_cam(self, img_t, target_idx):
        import torch.nn.functional as F

        act, grad = [None], [None]

        def fwd(m, inp, out):
            act[0] = (out[0] if isinstance(out, tuple) else out).detach()

        def bwd(m, gi, go):
            if isinstance(go[0], torch.Tensor):
                grad[0] = go[0].detach()

        handles = []
        cam_layer = getattr(self, "_cam_layer", None) or "visual.transformer.resblocks.9"
        for name, mod in self.model.named_modules():
            if name == cam_layer:
                handles.append(mod.register_forward_hook(fwd))
                handles.append(mod.register_full_backward_hook(bwd))
                break
        try:
            feats = F.normalize(self.model.encode_image(img_t), dim=-1)
            logits = (self.model.logit_scale.exp() * feats @ self.txt_emb).squeeze()
            self.model.zero_grad()
            logits[target_idx].backward(retain_graph=True)

            if act[0] is not None and grad[0] is not None:
                w = grad[0].abs().mean(dim=2)
                B, N = w.shape
                gs = int(np.sqrt(N - 1))
                if gs * gs == N - 1:
                    cam = w[:, 1:].reshape(B, gs, gs).squeeze(0).cpu().numpy()
                    if cam.max() > cam.min():
                        cam = (cam - cam.min()) / (cam.max() - cam.min())
                    return cam
        finally:
            for h in handles:
                h.remove()
        return None

    @staticmethod
    def _heatmap_to_bboxes(heatmap, threshold=0.5, max_boxes=5):
        binary = (heatmap > threshold).astype(np.uint8)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if n_labels <= 1:
            return []
        boxes = []
        for i in range(1, n_labels):
            if stats[i, cv2.CC_STAT_AREA] < 10:
                continue
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            mask = (labels == i).astype(np.uint8)
            intensity = float(np.mean(heatmap[mask == 1]))
            boxes.append((x, y, x + w, y + h, stats[i, cv2.CC_STAT_AREA], intensity))
        boxes.sort(key=lambda b: b[5], reverse=True)
        return boxes[:max_boxes]


# ---------------------------------------------------------------------------
# Gemma 4 (Ollama vision API — JSON boxes + captioning)
# ---------------------------------------------------------------------------


class _Gemma4Detector:
    """Gemma 4 via Ollama ``/api/chat``: detection JSON ``box_2d`` + captioning.

    Boxes use a normalized 1024x1024 grid (y1, x1, y2, x2); we map to pixel bboxes.
    Env: ``OLLAMA_HOST``, ``GEMMA4_OLLAMA_MODEL`` (default ``gemma4:31b``),
    ``GEMMA4_MAX_NEW_TOKENS`` (maps to ``num_predict``), ``GEMMA4_TEMPERATURE`` /
    ``GEMMA4_TOP_P`` / ``GEMMA4_TOP_K``, ``OLLAMA_TIMEOUT`` (seconds).
    """

    NOMINAL_CONFIDENCE = 0.92
    _VISUAL_BUDGETS = (70, 140, 280, 560, 1120)

    def __init__(self, model_id: str | None = None):
        try:
            import requests  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Gemma4 (Ollama) requires the ``requests`` package. pip install requests"
            ) from exc

        self.base_url = _normalize_ollama_host(
            os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        )
        self.model = model_id or os.environ.get("GEMMA4_OLLAMA_MODEL", "gemma4:31b")
        self._default_soft_tokens = int(os.environ.get("GEMMA4_MAX_SOFT_TOKENS", "280"))
        self._num_predict = int(os.environ.get("GEMMA4_MAX_NEW_TOKENS", "512"))
        self._temperature = float(os.environ.get("GEMMA4_TEMPERATURE", "1.0"))
        self._top_p = float(os.environ.get("GEMMA4_TOP_P", "0.95"))
        self._top_k = int(os.environ.get("GEMMA4_TOP_K", "64"))
        self._timeout = float(os.environ.get("OLLAMA_TIMEOUT", "600"))
        logger.info("Gemma4 (Ollama): model=%s base=%s", self.model, self.base_url)

    @classmethod
    def _snap_visual_budget(cls, n: int | None) -> int | None:
        if n is None:
            return None
        if n in cls._VISUAL_BUDGETS:
            return n
        return min(cls._VISUAL_BUDGETS, key=lambda b: abs(b - n))

    def _pil_to_b64_png(self, image) -> str:
        from PIL import Image as PILImage

        if not isinstance(image, PILImage.Image):
            raise TypeError("Gemma4 expects a PIL Image")
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _generate(self, image, text_prompt: str, max_soft_tokens: int | None = None):
        import requests

        budget = self._snap_visual_budget(max_soft_tokens)
        if budget is not None:
            text_prompt = (
                f"{text_prompt}\n\n"
                f"(Preferred visual token budget: {budget}; "
                "supported values are 70, 140, 280, 560, 1120.)"
            )

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": text_prompt,
                    "images": [self._pil_to_b64_png(image)],
                }
            ],
            "stream": False,
            "options": {
                "temperature": self._temperature,
                "top_p": self._top_p,
                "top_k": self._top_k,
                "num_predict": self._num_predict,
            },
        }
        url = f"{self.base_url}/api/chat"
        try:
            r = requests.post(url, json=payload, timeout=self._timeout)
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Gemma4 Ollama request failed ({url}). Is Ollama running? "
                f"Set OLLAMA_HOST if needed. {exc}"
            ) from exc

        try:
            data = r.json()
        except Exception as exc:
            raise RuntimeError(
                f"Gemma4 Ollama returned non-JSON (HTTP {r.status_code}): {r.text[:500]}"
            ) from exc

        if r.status_code != 200:
            err = data.get("error") if isinstance(data, dict) else None
            raise RuntimeError(
                f"Gemma4 Ollama error (HTTP {r.status_code}): {err or r.text[:500]}"
            )

        if isinstance(data, dict) and data.get("error"):
            raise RuntimeError(f"Gemma4 Ollama error: {data['error']}")

        msg = data.get("message") if isinstance(data, dict) else None
        content = ""
        if isinstance(msg, dict):
            content = str(msg.get("content") or "")
        raw = content or (data.get("response") if isinstance(data, dict) else "") or ""
        return _strip_gemma_thinking(str(raw))

    @staticmethod
    def _parse_json_boxes(text: str) -> list[dict]:
        import json

        m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        raw = m.group(1).strip() if m else None
        if not raw:
            m2 = re.search(r"(\[\s*\{.*?\}\s*\])", text, re.DOTALL)
            if m2:
                raw = m2.group(1)
        if not raw:
            return []
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            return []
        return [x for x in data if isinstance(x, dict)]

    @staticmethod
    def _box_2d_to_pixels(box_2d, w: int, h: int) -> list[int] | None:
        if not isinstance(box_2d, (list, tuple)) or len(box_2d) != 4:
            return None
        try:
            y1, x1, y2, x2 = [float(c) / 1024.0 for c in box_2d]
        except (TypeError, ValueError):
            return None
        x1p = int(round(x1 * w))
        y1p = int(round(y1 * h))
        x2p = int(round(x2 * w))
        y2p = int(round(y2 * h))
        x1p = max(0, min(w, x1p))
        x2p = max(0, min(w, x2p))
        y1p = max(0, min(h, y1p))
        y2p = max(0, min(h, y2p))
        if x2p <= x1p or y2p <= y1p:
            return None
        return [x1p, y1p, x2p, y2p]

    def detect(
        self,
        image,
        target: str = "",
        *,
        max_soft_tokens: int | None = None,
    ) -> list[dict]:
        from PIL import Image as PILImage

        if not isinstance(image, PILImage.Image):
            raise TypeError("Gemma4 detect expects a PIL Image")

        w, h = image.size
        hint = (target or "").strip()
        if hint in ("*", ""):
            detect_prompt = (
                "Detect all prominent objects in this image. "
                "Output only a markdown fenced block ```json with a JSON array. "
                "Each item: {\"label\": string, \"box_2d\": [y1,x1,y2,x2]} "
                "with integers 0-1024 on a normalized 1024x1024 grid (Gemma convention)."
            )
        else:
            detect_prompt = (
                f"Detect these categories: {hint}. "
                "Output only ```json with a JSON array of "
                '{{"label": string, "box_2d": [y1,x1,y2,x2]}} with integers 0-1024.'
            )

        raw_text = self._generate(image, detect_prompt, max_soft_tokens)
        items = self._parse_json_boxes(raw_text)
        detections: list[dict] = []
        for item in items:
            lab = item.get("label", "object")
            if isinstance(lab, str):
                label = lab
            else:
                label = str(lab)
            box_2d = item.get("box_2d") or item.get("box")
            bbox = self._box_2d_to_pixels(box_2d, w, h)
            if bbox is None:
                continue
            detections.append(
                {
                    "bbox": bbox,
                    "label": label,
                    "confidence": round(self.NOMINAL_CONFIDENCE, 3),
                }
            )
        return detections

    def describe(self, image, prompt: str | None = None, max_soft_tokens: int | None = None) -> str:
        from PIL import Image as PILImage

        if not isinstance(image, PILImage.Image):
            raise TypeError("Gemma4 describe expects a PIL Image")

        p = prompt or (
            "Describe this image in 2-4 sentences: main objects, setting, and lighting."
        )
        return self._generate(image, p, max_soft_tokens).strip()


# ---------------------------------------------------------------------------
# Tiled / sliced batch inference (SAHI-style)
# ---------------------------------------------------------------------------
#
# Slice a large frame into a grid of model-sized tiles, run inference on every
# tile (batched when the backend supports it), then map each tile's boxes back
# into the original frame and de-duplicate overlaps with NMS. This preserves
# small-object resolution that a single down-scaled forward pass would lose
# (e.g. a 1920x1080 frame at a 640x640 model => a 3x2 grid of 6 tiles).

# Default tile (model input) sizes per backend, as (width, height).
_DEFAULT_TILE_SIZE = {
    "yolo": (640, 640),
    "bioclip": (224, 224),
    "gemma4": (768, 768),
}


def _model_tile_size(model: str, detector) -> tuple[int, int]:
    """Best-effort model input size; YOLO/BioCLIP are introspected when possible."""
    model = _canonical_model(model)
    if model == "bioclip":
        n = int(getattr(detector, "input_size", 224))
        return (n, n)
    if model == "yolo":
        for attr in ("args", "overrides"):
            try:
                cfg = getattr(detector.model, attr, None)
                imgsz = cfg.get("imgsz") if isinstance(cfg, dict) else getattr(cfg, "imgsz", None)
                if imgsz:
                    s = int(imgsz[0] if isinstance(imgsz, (list, tuple)) else imgsz)
                    return (s, s)
            except Exception:
                pass
    return _DEFAULT_TILE_SIZE.get(model, (640, 640))


def _tile_offsets(total: int, tile: int, overlap: int) -> list[int]:
    """Start offsets covering ``[0, total)`` with ``tile``-wide windows.

    The final window is snapped flush to the far edge so the whole axis is
    covered without rescaling. ``overlap`` (px) reduces the stride.
    """
    if tile <= 0 or total <= tile:
        return [0]
    stride = max(1, tile - max(0, overlap))
    offsets = list(range(0, total - tile + 1, stride))
    if not offsets or offsets[-1] != total - tile:
        offsets.append(total - tile)
    return offsets


def _plan_tiles(
    w: int, h: int, tile_w: int, tile_h: int, overlap: int
) -> list[tuple[int, int, int, int]]:
    """Return tile boxes ``(x1, y1, x2, y2)`` covering a ``w x h`` frame."""
    tile_w = min(tile_w, w)
    tile_h = min(tile_h, h)
    xs = _tile_offsets(w, tile_w, overlap)
    ys = _tile_offsets(h, tile_h, overlap)
    boxes = []
    for y in ys:
        for x in xs:
            boxes.append((x, y, x + tile_w, y + tile_h))
    return boxes


def _bbox_iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _nms(dets: list[dict], iou_thr: float) -> list[dict]:
    """Greedy per-label NMS; keeps the highest-confidence box in each cluster."""
    out: list[dict] = []
    for label in {d["label"] for d in dets}:
        group = sorted(
            [d for d in dets if d["label"] == label],
            key=lambda d: d.get("confidence", 0.0),
            reverse=True,
        )
        kept: list[dict] = []
        for d in group:
            if all(_bbox_iou(d["bbox"], k["bbox"]) < iou_thr for k in kept):
                kept.append(d)
        out.extend(kept)
    out.sort(key=lambda d: d.get("confidence", 0.0), reverse=True)
    return out


def detect_tiled(
    image,
    model: str = "yolo",
    *,
    tile_size: tuple[int, int] | int | None = None,
    tile_overlap: int = 0,
    tile_iou: float = 0.45,
    **kwargs,
) -> dict:
    """Slice ``image`` into model-sized tiles, infer on each, merge results.

    Boxes are returned in original-image pixel coordinates. YOLO tiles run as a
    single batched forward pass; BioCLIP / Gemma4 tiles run per-tile. Falls back
    to a normal full-frame ``detect`` when the image already fits one tile.
    """
    from PIL import Image as _PILImage

    t0 = time.time()
    canon = _canonical_model(model)
    try:
        det = get_detector(canon)
    except RuntimeError as exc:
        return {"model": canon, "error": str(exc), "detections": [],
                "elapsed_ms": 0, "image_size": list(image.size), "tiled": False}

    if not isinstance(image, _PILImage.Image):
        return {"model": canon, "error": "tiled detection expects a PIL Image",
                "detections": [], "elapsed_ms": 0,
                "image_size": list(getattr(image, "size", [0, 0])), "tiled": False}

    w, h = image.size
    if tile_size is None:
        tile_w, tile_h = _model_tile_size(canon, det)
    elif isinstance(tile_size, int):
        tile_w = tile_h = int(tile_size)
    else:
        tile_w, tile_h = int(tile_size[0]), int(tile_size[1])

    tiles = _plan_tiles(w, h, tile_w, tile_h, int(tile_overlap))
    # Single tile => no benefit over the standard path.
    if len(tiles) <= 1:
        out = detect(image, model=canon, **kwargs)
        out["tiled"] = False
        out["tiles"] = 1
        return out

    crops = [image.crop(box) for box in tiles]
    merged: list[dict] = []

    try:
        if canon == "yolo":
            targets = kwargs.get("targets", "*")
            per_tile = det.detect_batch(crops, targets)
            for (tx1, ty1, _, _), dets in zip(tiles, per_tile):
                for d in dets:
                    x1, y1, x2, y2 = d["bbox"]
                    merged.append({**d, "bbox": [x1 + tx1, y1 + ty1,
                                                 x2 + tx1, y2 + ty1]})
        else:
            for (tx1, ty1, _, _), crop in zip(tiles, crops):
                sub = detect(crop, model=canon, **kwargs)
                for d in sub.get("detections", []):
                    x1, y1, x2, y2 = d["bbox"]
                    merged.append({**d, "bbox": [x1 + tx1, y1 + ty1,
                                                 x2 + tx1, y2 + ty1]})
    except Exception as exc:
        logger.exception("Tiled detection failed (%s)", canon)
        return {"model": canon, "error": str(exc), "detections": [],
                "elapsed_ms": int((time.time() - t0) * 1000),
                "image_size": [w, h], "tiled": True, "tiles": len(tiles)}

    merged = _nms(merged, float(tile_iou))
    return {
        "model": canon,
        "detections": merged,
        "elapsed_ms": int((time.time() - t0) * 1000),
        "image_size": [w, h],
        "tiled": True,
        "tiles": len(tiles),
        "tile_size": [tile_w, tile_h],
        "tile_overlap": int(tile_overlap),
    }


# ---------------------------------------------------------------------------
# Singleton management
# ---------------------------------------------------------------------------

_instances: dict = {}


def get_detector(model: str):
    """Return a cached detector instance (lazy-loaded)."""
    model = _canonical_model(model)
    if model in _instances:
        return _instances[model]

    if model == "yolo":
        if not _HAS_YOLO:
            raise RuntimeError(
                "YOLO unavailable — pip install ultralytics; "
                "torch and torchvision must be matching builds (same PyTorch install line)."
            )
        _instances[model] = _YOLODetector()
    elif model == "bioclip":
        if not _HAS_BIOCLIP:
            raise RuntimeError(
                "BioCLIP unavailable — pip install open_clip_torch opencv-python "
                "numpy huggingface_hub; torch and torchvision must match."
            )
        _instances[model] = _BioCLIPDetector()
    elif model == "gemma4":
        if not _HAS_GEMMA4:
            raise RuntimeError(
                "Gemma4 unavailable — start Ollama (ollama serve), pull a vision tag "
                "(e.g. ollama pull gemma4:31b), ensure requests is installed. "
                "Set OLLAMA_HOST (default http://127.0.0.1:11434) and GEMMA4_OLLAMA_MODEL if needed."
            )
        _instances[model] = _Gemma4Detector()
    else:
        raise ValueError(f"Unknown model: {model}. Choose from: yolo, bioclip, gemma4")

    return _instances[model]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect(image, model: str = "yolo", **kwargs) -> dict:
    """Run detection and return a standardised result dict.

    Returns::

        {
            "model": str,
            "detections": [{"bbox": [x1,y1,x2,y2], "label": str, "confidence": float}, ...],
            "elapsed_ms": int,
            "image_size": [width, height],
        }
    """
    # Tiled / sliced batch inference: slice large frames into model-sized tiles.
    if kwargs.pop("tile", False) or kwargs.pop("tiled", False):
        tile_kwargs = {
            "tile_size": kwargs.pop("tile_size", None),
            "tile_overlap": int(kwargs.pop("tile_overlap", 0) or 0),
            "tile_iou": float(kwargs.pop("tile_iou", 0.45)),
        }
        return detect_tiled(image, model=model, **tile_kwargs, **kwargs)

    t0 = time.time()
    model = _canonical_model(model)
    try:
        det = get_detector(model)
    except RuntimeError as exc:
        return {"model": model, "error": str(exc), "detections": [],
                "elapsed_ms": 0, "image_size": list(image.size)}

    bio_dbg: dict | None = None
    try:
        if model == "yolo":
            dets = det.detect(image, kwargs.get("targets", "*"))
        elif model == "bioclip":
            bio_dbg = {} if _bioclip_debug_requested(kwargs) else None
            dets = det.detect(
                image,
                rank=kwargs.get("rank", "Class"),
                target_taxon=kwargs.get("target_taxon", ""),
                min_confidence=float(kwargs.get("min_confidence", 0.1)),
                out_debug=bio_dbg,
            )
        elif model == "gemma4":
            tgt = kwargs.get("target")
            if tgt is None:
                tgt = kwargs.get("targets", "")
            if isinstance(tgt, str) and tgt.strip() == "*":
                tgt = ""
            ms = kwargs.get("max_soft_tokens")
            dets = det.detect(
                image,
                target=str(tgt or ""),
                max_soft_tokens=int(ms) if ms is not None else None,
            )
        else:
            dets = []
    except Exception as exc:
        logger.exception("Detection failed (%s)", model)
        return {"model": model, "error": str(exc), "detections": [],
                "elapsed_ms": int((time.time() - t0) * 1000),
                "image_size": list(image.size)}

    out: dict = {
        "model": model,
        "detections": dets,
        "elapsed_ms": int((time.time() - t0) * 1000),
        "image_size": list(image.size),
    }
    if bio_dbg is not None:
        out["bioclip_debug"] = bio_dbg
    return out


def caption(image, model: str = "bioclip", **kwargs) -> dict:
    """Run captioning / classification. Returns ``{caption, elapsed_ms}``."""
    t0 = time.time()
    model = _canonical_model(model)
    try:
        det = get_detector(model)
    except RuntimeError as exc:
        return {"model": model, "error": str(exc), "caption": "",
                "elapsed_ms": 0}

    try:
        if model == "bioclip":
            cls = det.classify(image)
            text = "; ".join(f"{n}: {c:.1%}" for n, c in cls[:5])
        elif model == "gemma4":
            ms = kwargs.get("max_soft_tokens")
            text = det.describe(
                image,
                prompt=kwargs.get("prompt"),
                max_soft_tokens=int(ms) if ms is not None else None,
            )
        else:
            text = "Captioning supports bioclip or gemma4"
    except Exception as exc:
        logger.exception("Caption failed (%s)", model)
        return {"model": model, "error": str(exc), "caption": "",
                "elapsed_ms": int((time.time() - t0) * 1000)}

    return {
        "model": model,
        "caption": text,
        "elapsed_ms": int((time.time() - t0) * 1000),
    }
