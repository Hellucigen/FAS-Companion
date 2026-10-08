# vision/object_embedding.py — 视觉对象 Embedding
# ============================================================================
# 对每个 Mask 区域裁剪，使用视觉 Embedding 模型生成向量。
# 推荐: SigLIP 或 DINOv2
# 回退: CLIP (OpenAI) 或 ResNet 特征
# ============================================================================

import os
import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


class ObjectEmbedder:
    """视觉对象 Embedding 生成器

    支持多种后端，按优先级自动选择：
    1. SigLIP (推荐，强图像-文本对齐)
    2. DINOv2 (强视觉特征)
    3. CLIP (OpenAI)
    4. ResNet (基础回退)
    """

    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name
        self._model = None
        self._processor = None
        self._device = device or self._resolve_device()
        self._dim = None
        self._backend = None  # "siglip" | "dinov2" | "clip" | "resnet"

    @staticmethod
    def _resolve_device() -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._init_model()
        return self._dim or 768

    def _init_model(self):
        """按优先级自动选择并加载模型"""
        if self._model is not None:
            return

        # 优先尝试 SigLIP
        if self._try_siglip():
            return
        # 其次 DINOv2
        if self._try_dinov2():
            return
        # 再次 CLIP
        if self._try_clip():
            return
        # 最后 ResNet
        self._try_resnet()

    def _try_siglip(self) -> bool:
        try:
            from transformers import AutoModel, AutoProcessor
            import torch

            # 优先使用本地模型
            local_path = "E:/Models/siglip-base-patch16-224"
            if os.path.isdir(local_path):
                model_id = local_path
                logger.info(f"[ObjectEmbedder] 使用本地 SigLIP: {model_id}")
            else:
                model_id = self.model_name or "google/siglip-base-patch16-224"

            self._model = AutoModel.from_pretrained(model_id).to(self._device)
            self._processor = AutoProcessor.from_pretrained(model_id)
            self._dim = self._model.config.vision_config.hidden_size
            self._backend = "siglip"
            logger.info(f"[ObjectEmbedder] SigLIP 加载成功 (dim={self._dim}, device={self._device})")
            return True
        except Exception as e:
            logger.debug(f"[ObjectEmbedder] SigLIP 不可用: {e}")
            return False

    def _try_dinov2(self) -> bool:
        try:
            import io
            import warnings
            import contextlib
            import torch

            model_id = self.model_name or "facebook/dinov2-small"
            # 三条第三方回退提示，都不是我们的状态：
            #   1) torch.hub 直接 sys.stderr.write("Using cache found in ...")
            #   2) dinov2 hub 代码探测 xFormers → 三条 UserWarning（SwiGLU/Attention/Block）
            #   3) dinov2 的 logging.getLogger("dinov2").info("using MLP layer as FFN")
            # 全是"没装 xFormers → 回退标准注意力/MLP"的说明，功能等价（xFormers 是
            # 长序列/大 batch 的显存优化，DINOv2-S 单张小图用不上）。装 xFormers 也能
            # 消掉，但它是按特定 torch 版本编译的，与当前 2.12 的兼容性未验证、收益
            # 可忽略 —— 所以按掉噪音，不引入新依赖。
            # redirect_stderr 不会吞掉我们自己的日志：logging 的 StreamHandler 在
            # 构造时就绑定了原 stderr 对象，不受 sys.stderr 替换影响。
            logging.getLogger("dinov2").setLevel(logging.WARNING)
            with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
                warnings.filterwarnings(
                    "ignore", message="xFormers is not available")
                self._model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
            self._model.to(self._device)
            self._model.eval()
            self._dim = 384  # DINOv2-S 输出 384 维
            self._backend = "dinov2"
            logger.info(f"[ObjectEmbedder] DINOv2 加载成功 (dim={self._dim})")
            return True
        except Exception as e:
            logger.warning(f"[ObjectEmbedder] DINOv2 不可用: {e}")
            return False

    def _try_clip(self) -> bool:
        try:
            import torch
            import clip

            model_id = self.model_name or "ViT-B/32"
            self._model, self._processor = clip.load(model_id, device=self._device)
            self._dim = 512  # CLIP ViT-B/32 输出 512 维
            self._backend = "clip"
            logger.info(f"[ObjectEmbedder] CLIP 加载成功 (dim={self._dim})")
            return True
        except Exception as e:
            logger.debug(f"[ObjectEmbedder] CLIP 不可用: {e}")
            return False

    def _try_resnet(self) -> bool:
        try:
            import torch
            import torchvision.models as models
            import torchvision.transforms as T

            self._model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self._model.fc = torch.nn.Identity()  # 去掉分类头
            self._model.to(self._device)
            self._model.eval()

            self._processor = T.Compose([
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self._dim = 2048  # ResNet50 特征维度
            self._backend = "resnet"
            logger.info(f"[ObjectEmbedder] ResNet50 加载成功 (dim={self._dim})")
            return True
        except Exception as e:
            logger.warning(f"[ObjectEmbedder] ResNet 也不可用: {e}")
            self._dim = 768
            return False

    def encode(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        对图像/区域生成 Embedding

        Args:
            image: RGB 图像 (H, W, 3) numpy array
            mask: 可选，对象 mask (H, W) bool，用于裁剪

        Returns:
            embedding: (dim,) numpy array
        """
        self._init_model()

        # 裁剪
        if mask is not None and mask.any():
            cropped = self._crop_with_mask(image, mask)
        else:
            cropped = image

        try:
            if self._backend == "siglip":
                return self._encode_siglip(cropped)
            elif self._backend == "dinov2":
                return self._encode_dinov2(cropped)
            elif self._backend == "clip":
                return self._encode_clip(cropped)
            elif self._backend == "resnet":
                return self._encode_resnet(cropped)
            else:
                # 极端回退：直接 resize 后的原始像素
                return self._encode_fallback(cropped)
        except Exception as e:
            logger.warning(f"[ObjectEmbedder] 编码失败 ({self._backend}): {e}")
            return self._encode_fallback(cropped)

    def _crop_with_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """使用 mask 裁剪对象区域"""
        # 找到 mask 边界框
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any() or not cols.any():
            return image

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # 添加边距
        h, w = image.shape[:2]
        pad = 10
        rmin = max(0, rmin - pad)
        rmax = min(h, rmax + pad)
        cmin = max(0, cmin - pad)
        cmax = min(w, cmin + pad)

        return image[rmin:rmax, cmin:cmax]

    def _encode_siglip(self, image: np.ndarray) -> np.ndarray:
        import torch
        from PIL import Image

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        inputs = self._processor(images=image, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._model.get_image_features(**inputs)
        emb = outputs.cpu().numpy().flatten()
        # L2 归一化
        emb = emb / (np.linalg.norm(emb) + 1e-12)
        return emb

    def _encode_dinov2(self, image: np.ndarray) -> np.ndarray:
        import torch
        import torchvision.transforms as T

        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        tensor = transform(image).unsqueeze(0).to(self._device)
        with torch.no_grad():
            emb = self._model(tensor).cpu().numpy().flatten()
        emb = emb / (np.linalg.norm(emb) + 1e-12)
        return emb

    def _encode_clip(self, image: np.ndarray) -> np.ndarray:
        import torch
        from PIL import Image

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        image_input = self._processor(image).unsqueeze(0).to(self._device)
        with torch.no_grad():
            emb = self._model.encode_image(image_input).cpu().numpy().flatten()
        emb = emb / (np.linalg.norm(emb) + 1e-12)
        return emb

    def _encode_resnet(self, image: np.ndarray) -> np.ndarray:
        import torch

        tensor = self._processor(image).unsqueeze(0).to(self._device)
        with torch.no_grad():
            emb = self._model(tensor).cpu().numpy().flatten()
        emb = emb / (np.linalg.norm(emb) + 1e-12)
        return emb

    def _encode_fallback(self, image: np.ndarray) -> np.ndarray:
        """极端回退：resize + 像素值"""
        import cv2
        resized = cv2.resize(image, (64, 64))
        emb = resized.flatten().astype(np.float32) / 255.0
        # 填充/截断到 dim
        if len(emb) < self._dim:
            emb = np.pad(emb, (0, self._dim - len(emb)))
        else:
            emb = emb[:self._dim]
        emb = emb / (np.linalg.norm(emb) + 1e-12)
        return emb
