# vision/low_level_attrs.py — 低层属性自动提取
# ============================================================================
# 视觉模块自动提取低级属性：
# - 颜色 (主导色)
# - 大小 (small / medium / large)
# - 位置
# - 运动状态
#
# 禁止直接判断高级概念（如"这是苹果"）。
# 高级概念由 LLM + 用户反馈 + 知识图谱共同确认。
# ============================================================================

import numpy as np
import logging

logger = logging.getLogger(__name__)

# FAISS 匹配阈值
SIMILARITY_THRESHOLD = 0.75


def extract_low_level_attributes(image: np.ndarray, mask: np.ndarray = None) -> dict:
    """
    提取对象的低级视觉属性

    Args:
        image: RGB 图像 (H, W, 3)
        mask: 对象 mask (H, W) bool

    Returns:
        {
            "color": str | None,       # 主导颜色名称
            "color_rgb": [r, g, b],    # 主导色 RGB
            "size_category": str,      # "small" | "medium" | "large"
            "relative_area": float,    # 对象占图像比例
        }
    """
    h, w = image.shape[:2]
    total_area = h * w

    if mask is not None and mask.any():
        # 只取 mask 区域的像素
        pixels = image[mask]
        obj_area = mask.sum()
    else:
        pixels = image.reshape(-1, 3)
        obj_area = total_area

    attrs = {}

    # ── 颜色提取 ──────────────────────────────────────────
    try:
        color_name, color_rgb = _extract_dominant_color(pixels)
        attrs["color"] = color_name
        attrs["color_rgb"] = [int(c) for c in color_rgb]
    except Exception as e:
        logger.debug(f"[Attrs] 颜色提取失败: {e}")
        attrs["color"] = None

    # ── 大小分类 ──────────────────────────────────────────
    relative_area = obj_area / total_area if total_area > 0 else 0
    attrs["relative_area"] = round(float(relative_area), 4)

    if relative_area < 0.02:
        attrs["size_category"] = "small"
    elif relative_area < 0.15:
        attrs["size_category"] = "medium"
    else:
        attrs["size_category"] = "large"

    return attrs


def _extract_dominant_color(pixels: np.ndarray) -> tuple[str, np.ndarray]:
    """提取主导颜色

    使用 K-Means 聚类（k=5）找主要颜色簇，取最大的。
    """
    if len(pixels) == 0:
        return "unknown", np.array([0, 0, 0])

    # 降采样以提高性能
    max_samples = 5000
    if len(pixels) > max_samples:
        idx = np.random.choice(len(pixels), max_samples, replace=False)
        pixels = pixels[idx]

    try:
        from sklearn.cluster import KMeans
        k = min(5, len(pixels))
        if k < 2:
            avg = np.mean(pixels, axis=0).astype(np.uint8)
            return _rgb_to_color_name(avg), avg

        kmeans = KMeans(n_clusters=k, n_init=2, random_state=42)
        labels = kmeans.fit_predict(pixels)

        # 找最大簇
        counts = np.bincount(labels)
        dominant_cluster = np.argmax(counts)
        dominant_color = kmeans.cluster_centers_[dominant_cluster].astype(np.uint8)

        return _rgb_to_color_name(dominant_color), dominant_color
    except ImportError:
        # sklearn 不可用时用平均值
        avg = np.mean(pixels, axis=0).astype(np.uint8)
        return _rgb_to_color_name(avg), avg


def _rgb_to_color_name(rgb: np.ndarray) -> str:
    """将 RGB 值映射到人类可读的颜色名称"""
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])

    # 灰度检测
    if abs(r - g) < 20 and abs(g - b) < 20 and abs(r - b) < 20:
        brightness = (r + g + b) // 3
        if brightness < 40:
            return "黑色"
        elif brightness < 100:
            return "深灰色"
        elif brightness < 180:
            return "灰色"
        elif brightness < 230:
            return "浅灰色"
        else:
            return "白色"

    # 颜色映射
    color_map = {
        "红色": (r > 150 and r > g * 1.3 and r > b * 1.3),
        "深红色": (r > 100 and r > g * 1.5 and r > b * 1.5),
        "绿色": (g > 120 and g > r * 1.2 and g > b * 1.2),
        "深绿色": (g > 80 and g > r * 1.3 and g > b * 1.3),
        "蓝色": (b > 120 and b > r * 1.2 and b > g * 1.2),
        "深蓝色": (b > 80 and b > r * 1.3 and b > g * 1.3),
        "黄色": (r > 180 and g > 180 and b < 100),
        "橙色": (r > 180 and g > 100 and g < 180 and b < 80),
        "紫色": (r > 100 and b > 100 and r + b > g * 1.5),
        "粉色": (r > 200 and g > 100 and b > 150),
        "棕色": (r > 100 and g > 50 and g < 150 and b < 80),
        "青色": (g > 150 and b > 150 and r < 120),
    }

    for name, condition in color_map.items():
        if condition:
            return name

    return "其他颜色"
