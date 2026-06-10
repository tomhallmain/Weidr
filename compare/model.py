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

if config.xvlm_loc is not None:
    logger.info(f"Loading XVLM modules from {config.xvlm_loc}")
    try:
        import sys
        from transformers import BertTokenizer, BertModel
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
        # Initialize model with config
        if config.xvlm_model_size not in XVLM_CONFIGS:
            raise ValueError(f"Invalid XVLM model size - update config.json: {config.xvlm_model_size}")
        config = XVLM_CONFIGS[config.xvlm_model_size]
        _xvlm_model = XVLMBase(config)
        
        # Load pretrained weights
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
