import os

from PIL import Image
import torch
import clip
from transformers import AutoModel, AutoProcessor, FlavaProcessor, FlavaModel, AlignProcessor, AlignModel

from image.frame_cache import FrameCache
from utils.config import config
from utils.logging_setup import get_logger
from utils.pillow_plugins import ensure_pillow_plugins_registered

ensure_pillow_plugins_registered()

logger = get_logger("model")

# XVLM may not be loaded if the config.json file is not updated
# or if the model files are not downloaded
xvlm_loaded = False

# EVA CLIP requires the open_clip package (pip install open-clip-torch)
eva_clip_loaded = False
try:
    import open_clip
    eva_clip_loaded = True
    logger.info("open_clip available — EVA CLIP mode enabled")
except ImportError:
    logger.info("open_clip not installed — EVA CLIP mode unavailable (pip install open-clip-torch)")

# InsightFace requires insightface + onnxruntime (or onnxruntime-gpu)
insightface_loaded = False
try:
    import insightface  # noqa: F401
    insightface_loaded = True
    logger.info("insightface available — Face Embedding mode enabled")
except ImportError:
    logger.info("insightface not installed — Face Embedding mode unavailable (pip install insightface onnxruntime-gpu)")

if config.xvlm_loc is not None:
    logger.info(f"Loading XVLM modules from {config.xvlm_loc}")
    try:
        import sys
        from transformers import BertTokenizer
        from torchvision import transforms
        sys.path.insert(0, config.xvlm_loc)
        from models.xvlm import XVLMBase
        logger.info("XVLM modules loaded")
        xvlm_loaded = True
    except Exception as e:
        logger.error(f"Error loading XVLM modules: {e}")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Lazy initialization variables for CLIP
_clip_model = None
_clip_preprocess = None

# Lazy initialization variables for SIGLIP
_siglip_model = None
_siglip_processor = None

# Lazy initialization variables for FLAVA
_flava_model = None
_flava_processor = None

# Lazy initialization variables for ALIGN
_align_model = None
_align_processor = None

# Lazy initialization variables for XVLM
_xvlm_model = None
_xvlm_tokenizer = None
_xvlm_img_transform = None

# Lazy initialization variables for LAION
_laion_model = None
_laion_processor = None

# Lazy initialization variables for MetaCLIP
_metaclip_model = None
_metaclip_processor = None

# Lazy initialization variables for V-JEPA 2
_vjepa2_model = None
_vjepa2_processor = None

# Lazy initialization variables for EVA CLIP
_eva_clip_model = None
_eva_clip_preprocess = None
_eva_clip_tokenizer = None

# open_clip pretrained weights IDs keyed by model name
EVA_CLIP_PRETRAINED = {
    'EVA01-g-14':       'laion400m_s11b_b41k',
    'EVA01-g-14-plus':  'merged2b_s11b_b114k',
    'EVA02-B-16':       'merged2b_s8b_b131k',
    'EVA02-L-14':       'merged2b_s4b_b131k',
    'EVA02-L-14-336':   'merged2b_s6b_b61k',
    'EVA02-E-14':       'laion2b_s4b_b115k',
    'EVA02-E-14-plus':  'laion2b_s9b_b144k',
}

# Model and processor access functions

def _get_clip_model():
    global _clip_model
    if _clip_model is None:
        _clip_model, _ = clip.load(config.clip_model, device=device)
    return _clip_model

def _get_clip_preprocess():
    global _clip_preprocess
    if _clip_preprocess is None:
        _, _clip_preprocess = clip.load(config.clip_model, device=device)
    return _clip_preprocess

def _get_siglip_model():
    global _siglip_model
    if _siglip_model is None:
        if config.siglip_enable_large_model:
            _siglip_model = AutoModel.from_pretrained("google/siglip-large-patch16-384", torch_dtype=torch.float16).to(device)
        else:
            _siglip_model = AutoModel.from_pretrained("google/siglip-base-patch16-224").to(device)
    return _siglip_model

def _get_siglip_processor():
    global _siglip_processor
    if _siglip_processor is None:
        if config.siglip_enable_large_model:
            _siglip_processor = AutoProcessor.from_pretrained("google/siglip-large-patch16-384", torch_dtype=torch.float16)
        else:
            _siglip_processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
    return _siglip_processor

def _get_flava_model():
    global _flava_model
    if _flava_model is None:
        _flava_model = FlavaModel.from_pretrained("facebook/flava-full").to(device)
    return _flava_model

def _get_flava_processor():
    global _flava_processor
    if _flava_processor is None:
        _flava_processor = FlavaProcessor.from_pretrained("facebook/flava-full")
    return _flava_processor

def _get_align_model():
    global _align_model
    if _align_model is None:
        _align_model = AlignModel.from_pretrained("kakaobrain/align-base").to(device)
    return _align_model

def _get_align_processor():
    global _align_processor
    if _align_processor is None:
        _align_processor = AlignProcessor.from_pretrained("kakaobrain/align-base")
    return _align_processor

# Define preset configs for 4m/16m (extracted from YAMLs)
XVLM_CONFIGS = {
    '4m': {
        'vision_encoder': 'swin_base_patch4_window12_384',
        'text_encoder': 'bert-base-uncased',
        'embed_dim': 256,
        'temp': 0.07,
        'multi_grained': True,
        'max_words': 40  # From 4m.yaml
    },
    '16m': {
        'vision_encoder': 'swin_base_patch4_window12_384',
        'text_encoder': 'bert-base-uncased',
        'embed_dim': 256,
        'temp': 0.07,
        'multi_grained': True,
        'max_words': 30  # From 16m.yaml
    }
}

def _get_xvlm_model():
    global _xvlm_model
    if _xvlm_model is None:
        if config.xvlm_model_size not in XVLM_CONFIGS:
            raise ValueError(f"Invalid XVLM model size - update config.json: {config.xvlm_model_size}")
        if config.xvlm_model_loc is None:
            raise ValueError("xvlm_model_loc is not set in config.json — provide the path to the downloaded checkpoint")
        xvlm_cfg = XVLM_CONFIGS[config.xvlm_model_size]
        _xvlm_model = XVLMBase(xvlm_cfg)
        checkpoint = torch.load(config.xvlm_model_loc, map_location='cpu')
        _xvlm_model.load_state_dict(checkpoint['model'], strict=False)
        _xvlm_model.eval()
        _xvlm_model = _xvlm_model.to(device)
    return _xvlm_model

def _get_xvlm_tokenizer():
    global _xvlm_tokenizer
    if _xvlm_tokenizer is None:
        _xvlm_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return _xvlm_tokenizer

def _get_xvlm_img_transform():
    global _xvlm_img_transform
    if _xvlm_img_transform is None:
        _xvlm_img_transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                               (0.26862954, 0.26130258, 0.27577711))
        ])
    return _xvlm_img_transform

def _get_metaclip_model():
    global _metaclip_model
    if _metaclip_model is None:
        if config.metaclip_half_precision:
            _metaclip_model = AutoModel.from_pretrained(config.metaclip_model, torch_dtype=torch.float16).to(device)
        else:
            _metaclip_model = AutoModel.from_pretrained(config.metaclip_model).to(device)
        _metaclip_model.eval()
    return _metaclip_model

def _get_metaclip_processor():
    global _metaclip_processor
    if _metaclip_processor is None:
        _metaclip_processor = AutoProcessor.from_pretrained(config.metaclip_model)
    return _metaclip_processor

def _get_vjepa2_model():
    global _vjepa2_model
    if _vjepa2_model is None:
        from transformers import AutoModel
        dtype = torch.float16 if config.vjepa2_half_precision else torch.float32
        _vjepa2_model = AutoModel.from_pretrained(config.vjepa2_model, torch_dtype=dtype).to(device)
        _vjepa2_model.eval()
    return _vjepa2_model

def _get_vjepa2_processor():
    global _vjepa2_processor
    if _vjepa2_processor is None:
        from transformers import AutoVideoProcessor
        _vjepa2_processor = AutoVideoProcessor.from_pretrained(config.vjepa2_model)
    return _vjepa2_processor

def _get_eva_clip_model():
    global _eva_clip_model, _eva_clip_preprocess
    if _eva_clip_model is None:
        model_name = config.eva_clip_model
        pretrained = EVA_CLIP_PRETRAINED.get(model_name)
        if pretrained is None:
            raise ValueError(
                f"Unknown EVA CLIP model '{model_name}'. "
                f"Valid options: {list(EVA_CLIP_PRETRAINED.keys())}"
            )
        dtype = torch.float16 if config.eva_clip_half_precision else torch.float32
        _eva_clip_model, _, _eva_clip_preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device, precision='fp16' if config.eva_clip_half_precision else 'fp32'
        )
        _eva_clip_model = _eva_clip_model.to(dtype=dtype)
        _eva_clip_model.eval()
    return _eva_clip_model

def _get_eva_clip_preprocess():
    global _eva_clip_preprocess
    if _eva_clip_preprocess is None:
        _get_eva_clip_model()  # preprocess is populated as a side-effect
    return _eva_clip_preprocess

def _get_eva_clip_tokenizer():
    global _eva_clip_tokenizer
    if _eva_clip_tokenizer is None:
        _eva_clip_tokenizer = open_clip.get_tokenizer(config.eva_clip_model)
    return _eva_clip_tokenizer

def _get_laion_model():
    global _laion_model
    if _laion_model is None:
        if config.laion_enable_half_precision:
            _laion_model = AutoModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", torch_dtype=torch.float16).to(device)
        else:
            _laion_model = AutoModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K").to(device)
    return _laion_model

def _get_laion_processor():
    global _laion_processor
    if _laion_processor is None:
        if config.laion_enable_half_precision:
            _laion_processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", torch_dtype=torch.float16)
        else:
            _laion_processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    return _laion_processor


# Embedding similarity
def embedding_similarity(embedding0, embedding1):
    # TODO maybe find out a way to not have to reconvert back to tensor
    # since this might be less efficient then a simple list
    t0 = torch.Tensor([list(embedding0)])
    t1 = torch.Tensor([list(embedding1)])
    # logger.debug(f"[SigLIP] Embedding 1 norm: {embedding0.norm().item():.4f}")
    # logger.debug(f"[SigLIP] Embedding 2 norm: {embedding1.norm().item():.4f}")
    return torch.nn.functional.cosine_similarity(t0, t1)


# CLIP embeddings

def image_embeddings_clip(image_path):
    try:
        with Image.open(image_path) as pil_img:
            image = _get_clip_preprocess()(pil_img).unsqueeze(0).to(device)
    except Exception as e:
        image_path = FrameCache.get_image_path(image_path)
        with Image.open(image_path) as pil_img:
            image = _get_clip_preprocess()(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = _get_clip_model().encode_image(image)
        embedding /= embedding.norm(dim=-1, keepdim=True)
        return embedding.tolist()[0]


def text_embeddings_clip(text):
    tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        embedding = _get_clip_model().encode_text(tokens).float()
        embedding /= embedding.norm(dim=-1, keepdim=True)
        return embedding.tolist()[0]


# SigLIP embeddings

def image_embeddings_siglip(image_path):
    try:
        with Image.open(image_path) as pil_img:
            # Process image with SIGLIP processor
            inputs = _get_siglip_processor()(images=pil_img, return_tensors="pt").to(device)
    except Exception as e:
        image_path = FrameCache.get_image_path(image_path)
        with Image.open(image_path) as pil_img:
            inputs = _get_siglip_processor()(images=pil_img, return_tensors="pt").to(device)
    
    with torch.no_grad():
        # Get image features using SIGLIP model
        outputs = _get_siglip_model().get_image_features(**inputs)
        # Normalize the embeddings
        outputs = outputs / outputs.norm(dim=-1, keepdim=True)
        return outputs.tolist()[0]

def text_embeddings_siglip(text):
    # Process text with SIGLIP processor
    inputs = _get_siglip_processor()(text=[text], padding="max_length", return_tensors="pt").to(device)
    
    with torch.no_grad():
        # Get text features using SIGLIP model
        outputs = _get_siglip_model().get_text_features(**inputs)
        # Normalize the embeddings
        outputs = outputs / outputs.norm(dim=-1, keepdim=True)
        return outputs.tolist()[0]


# FLAVA embeddings

def image_embeddings_flava(image_path):
    try:
        with Image.open(image_path) as img:
            image = img.convert("RGB")
            # Process image with FLAVA processor
            inputs = _get_flava_processor()(images=image, return_tensors="pt").to(device)
            image.close()
    except Exception as e:
        image_path = FrameCache.get_image_path(image_path)
        with Image.open(image_path) as img:
            image = img.convert("RGB")
            inputs = _get_flava_processor()(images=image, return_tensors="pt").to(device)
            image.close()
    
    with torch.no_grad():
        # Get image features using FLAVA model
        outputs = _get_flava_model().get_image_features(**inputs)
        image_embed = outputs.squeeze(0)  # Remove batch dimension [1, 768] → [768]
        # Normalize the embeddings
        image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)
        return image_embed.tolist()[0]

def text_embeddings_flava(text):
    # Process text with FLAVA processor
    inputs = _get_flava_processor()(text=[text], return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        # Get text features using FLAVA model
        outputs = _get_flava_model().get_text_features(**inputs)
        text_embed = outputs.squeeze(0)  # Remove batch dimension [1, 768] → [768]
        # Normalize the embeddings
        text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
        return text_embed.tolist()[0]


# ALIGN embeddings

def image_embeddings_align(image_path):
    try:
        with Image.open(image_path) as img:
            image = img.convert("RGB")
            # Process image with ALIGN processor
            inputs = _get_align_processor()(images=image, return_tensors="pt").to(device)
            image.close()
    except Exception as e:
        image_path = FrameCache.get_image_path(image_path)
        with Image.open(image_path) as img:
            image = img.convert("RGB")
            inputs = _get_align_processor()(images=image, return_tensors="pt").to(device)
            image.close()
    
    with torch.no_grad():
        # Get image features using ALIGN model
        outputs = _get_align_model().get_image_features(**inputs)
        image_embed = outputs.squeeze(0)  # Remove batch dimension [1, 640] → [640]
        # Normalize the embeddings
        image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)
        return image_embed.tolist()


def text_embeddings_align(text):
    # Process text with ALIGN processor
    inputs = _get_align_processor()(text=text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        # Get text features using ALIGN model
        outputs = _get_align_model().get_text_features(**inputs)
        text_embed = outputs.squeeze(0)  # Remove batch dimension [1, 640] → [640]
        # Normalize the embeddings
        text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
        return text_embed.tolist()


# X-VLM embeddings

def image_embeddings_xvlm(image_path):
    try:
        with Image.open(image_path) as img:
            image = img.convert("RGB")
            # Process image with XVLM transform
            image_tensor = _get_xvlm_img_transform()(image).unsqueeze(0).to(device)
            image.close()
    except Exception as e:
        image_path = FrameCache.get_image_path(image_path)
        with Image.open(image_path) as img:
            image = img.convert("RGB")
            image_tensor = _get_xvlm_img_transform()(image).unsqueeze(0).to(device)
            image.close()
    
    with torch.no_grad():
        # Get image features using XVLM model
        image_embeds = _get_xvlm_model().vision_encoder(image_tensor)
        image_feat = _get_xvlm_model().vision_proj(image_embeds[:, 0, :])
        # Normalize the embeddings
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
        return image_feat.tolist()[0]


def text_embeddings_xvlm(text):
    # Process text with XVLM tokenizer
    inputs = _get_xvlm_tokenizer()(text, return_tensors='pt', padding=True, truncation=True).to(device)
    
    with torch.no_grad():
        # Get text features using XVLM model
        text_embeds = _get_xvlm_model().text_encoder(**inputs).last_hidden_state
        text_feat = _get_xvlm_model().text_proj(text_embeds[:, 0, :])
        # Normalize the embeddings
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        return text_feat.tolist()[0]


# LAION embeddings

def image_embeddings_laion(image_path):
    try:
        with Image.open(image_path) as img:
            image = img.convert("RGB")
            # Process image with LAION processor
            inputs = _get_laion_processor()(images=image, return_tensors="pt").to(device)
            image.close()
    except Exception as e:
        image_path = FrameCache.get_image_path(image_path)
        with Image.open(image_path) as img:
            image = img.convert("RGB")
            inputs = _get_laion_processor()(images=image, return_tensors="pt").to(device)
            image.close()
    
    with torch.no_grad():
        # Get image features using LAION model
        outputs = _get_laion_model().get_image_features(**inputs)
        # Normalize the embeddings
        outputs = outputs / outputs.norm(dim=-1, keepdim=True)
        return outputs.tolist()[0]

def text_embeddings_laion(text):
    # Process text with LAION processor
    inputs = _get_laion_processor()(text=[text], padding="max_length", return_tensors="pt").to(device)

    with torch.no_grad():
        # Get text features using LAION model
        outputs = _get_laion_model().get_text_features(**inputs)
        # Normalize the embeddings
        outputs = outputs / outputs.norm(dim=-1, keepdim=True)
        return outputs.tolist()[0]


# EVA CLIP embeddings

def image_embeddings_eva_clip(image_path):
    try:
        with Image.open(image_path) as img:
            image = _get_eva_clip_preprocess()(img.convert("RGB")).unsqueeze(0).to(device)
    except Exception:
        image_path = FrameCache.get_image_path(image_path)
        with Image.open(image_path) as img:
            image = _get_eva_clip_preprocess()(img.convert("RGB")).unsqueeze(0).to(device)
    if config.eva_clip_half_precision:
        image = image.half()
    with torch.no_grad():
        embedding = _get_eva_clip_model().encode_image(image)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.float().tolist()[0]


def text_embeddings_eva_clip(text):
    tokens = _get_eva_clip_tokenizer()([text]).to(device)
    with torch.no_grad():
        embedding = _get_eva_clip_model().encode_text(tokens)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.float().tolist()[0]


# MetaCLIP embeddings

def image_embeddings_metaclip(image_path):
    try:
        with Image.open(image_path) as img:
            inputs = _get_metaclip_processor()(images=img.convert("RGB"), return_tensors="pt").to(device)
    except Exception:
        image_path = FrameCache.get_image_path(image_path)
        with Image.open(image_path) as img:
            inputs = _get_metaclip_processor()(images=img.convert("RGB"), return_tensors="pt").to(device)
    if config.metaclip_half_precision:
        inputs = {k: v.half() if v.is_floating_point() else v for k, v in inputs.items()}
    with torch.no_grad():
        outputs = _get_metaclip_model().get_image_features(**inputs)
        outputs = outputs / outputs.norm(dim=-1, keepdim=True)
        return outputs.float().tolist()[0]


def text_embeddings_metaclip(text):
    inputs = _get_metaclip_processor()(text=[text], return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = _get_metaclip_model().get_text_features(**inputs)
        outputs = outputs / outputs.norm(dim=-1, keepdim=True)
        return outputs.float().tolist()[0]


# V-JEPA 2 embeddings

def _sample_vjepa2_frames(media_path: str, num_frames: int) -> "np.ndarray":
    """Return a (num_frames, H, W, 3) uint8 RGB array for any media path.

    For still images the single frame is repeated.  For videos, num_frames
    frames are sampled at evenly-spaced positions using PyAV seeking.  Falls
    back to the FrameCache thumbnail (repeated) if decoding fails.
    """
    import av
    import numpy as np
    from image.frame_cache import _pyav_video_stats

    ext = os.path.splitext(media_path)[1].lower()
    is_image = ext in Image.registered_extensions()

    if is_image:
        with Image.open(media_path) as img:
            arr = np.array(img.convert("RGB"))
        return np.stack([arr] * num_frames)

    try:
        frames = []
        total, _fps, _dur = _pyav_video_stats(media_path)
        with av.open(media_path, metadata_errors="ignore") as container:
            stream = container.streams.video[0]
            if total > 0 and stream.duration and stream.time_base:
                for i in range(num_frames):
                    target_pts = int(stream.duration * i / num_frames)
                    container.seek(target_pts, stream=stream)
                    for frame in container.decode(stream):
                        frames.append(frame.to_ndarray(format="rgb24"))
                        break
            else:
                # Unknown duration: decode up to a budget then subsample
                raw = []
                for frame in container.decode(stream):
                    raw.append(frame.to_ndarray(format="rgb24"))
                    if len(raw) >= 300:
                        break
                if raw:
                    indices = np.linspace(0, len(raw) - 1, num_frames, dtype=int)
                    frames = [raw[i] for i in indices]
        if not frames:
            raise ValueError("no frames decoded")
    except Exception:
        frame_path = FrameCache.get_image_path(media_path)
        with Image.open(frame_path) as img:
            arr = np.array(img.convert("RGB"))
        return np.stack([arr] * num_frames)

    while len(frames) < num_frames:
        frames.append(frames[-1])
    return np.stack(frames[:num_frames])  # (T, H, W, 3)


def image_embeddings_vjepa2(media_path):
    """Embed any media (image, video, GIF) using the V-JEPA 2 encoder.

    Produces a single normalised vector by mean-pooling the encoder's
    spatial-temporal patch tokens.  No text embedding counterpart exists;
    this mode is used for media-to-media similarity only.
    """
    import numpy as np
    frames = _sample_vjepa2_frames(media_path, config.vjepa2_num_frames)
    inputs = _get_vjepa2_processor()(frames, return_tensors="pt").to(device)
    if config.vjepa2_half_precision:
        inputs = {k: v.half() if v.is_floating_point() else v for k, v in inputs.items()}
    with torch.no_grad():
        outputs = _get_vjepa2_model()(**inputs, skip_predictor=True)
        # last_hidden_state: (B, seq_len, hidden_size) — mean-pool to single vector
        embedding = outputs.last_hidden_state.mean(dim=1)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.float().tolist()[0]


# ---------------------------------------------------------------------------
# InsightFace ArcFace — face identity embeddings
# ---------------------------------------------------------------------------

_insightface_app = None


def _get_insightface_app():
    global _insightface_app
    if _insightface_app is None:
        import insightface
        import onnxruntime
        cuda_available = "CUDAExecutionProvider" in onnxruntime.get_available_providers()
        providers = ["CUDAExecutionProvider"] if cuda_available else ["CPUExecutionProvider"]
        _insightface_app = insightface.app.FaceAnalysis(
            name=config.insightface_model,
            providers=providers,
        )
        _insightface_app.prepare(ctx_id=0 if cuda_available else -1, det_size=(640, 640))
    return _insightface_app


def image_embeddings_face(media_path: str):
    """Return a mean-pooled ArcFace embedding for all detected faces, or None if no face is found.

    Detection confidence must meet config.insightface_det_thresh.  The returned
    vector is L2-normalised and has shape (512,).
    """
    import cv2
    import numpy as np
    img = cv2.imread(media_path)
    if img is None:
        return None
    faces = _get_insightface_app().get(img)
    valid = [f.embedding for f in faces if f.det_score >= config.insightface_det_thresh]
    if not valid:
        return None
    mean_emb = np.mean(valid, axis=0)
    norm = np.linalg.norm(mean_emb)
    if norm == 0:
        return None
    return (mean_emb / norm).tolist()
