"""
Monolithic AI Edge — toàn bộ pipeline nhận diện chạy local (CPU-only).
Target NFR:
  - E2E ≤150ms / lượt (detect → anti-spoof → match)
  - Hiển thị video ≥25 FPS
  - Cold-start <3s
  - RAM idle ≤250MB, peak ≤1.2GB
  - Không GPU

Pipeline: Camera frame → YuNet (lọc mặt lớn nhất) → Anti-spoof → SFace → 1:N match
"""
from __future__ import annotations

import time
import random
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass

import numpy as np
from PIL import Image
import onnxruntime as ort

# Models nằm cạnh project (shared với web backend)
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
SFACE_PATH = MODELS_DIR / "face_recognition_sface_2021dec.onnx"
YUNET_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
# Anti-spoof ONNX (MiniFASNetV2); attendance fails closed if the model is missing.
ANTISPOOF_PATH = MODELS_DIR / "antispoof_minifasnet.onnx"

# Ngưỡng cosine SFace (OpenCV ~0.363). Hạ nhẹ để dễ khớp thực tế webcam.
SFACE_COSINE_THRESHOLD = 0.28
SFACE_COSINE_MARGIN = 0.06
# Session ONNX dùng chung (lazy, CPU only)
_sess_sface: Optional[ort.InferenceSession] = None
_sess_yunet: Optional[ort.InferenceSession] = None
_sess_spoof: Optional[ort.InferenceSession] = None
_cv_sface = None
_so_options = ort.SessionOptions()
_so_options.intra_op_num_threads = 2
_so_options.inter_op_num_threads = 1
_so_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL


@dataclass
class EdgeResult:
    ok: bool
    is_spoof: bool
    student_id: Optional[int]
    full_name: Optional[str]
    confidence: float
    latency_ms: float
    message: str
    bbox: Optional[List[float]] = None
    embedding: Optional[np.ndarray] = None
    liveness_pending: bool = False


class ActiveLiveness:
    """Randomized head-turn challenge layered on top of per-frame MiniFASNet."""

    def __init__(self):
        self.student_id = None
        self.started_at = 0.0
        self.baseline = None
        self.first_direction = random.choice((-1, 1))
        self.step = 0
        self.hold_count = 0
        self.complete_until = 0.0

    def reset(self, student_id=None):
        self.student_id = student_id
        self.started_at = time.monotonic() if student_id is not None else 0.0
        self.baseline = None
        self.first_direction = random.choice((-1, 1))
        self.step = 0
        self.hold_count = 0
        self.complete_until = 0.0

    def advance(self, student_id, face):
        now = time.monotonic()
        if self.student_id != student_id or (self.started_at and now - self.started_at > 15):
            self.reset(student_id)
        if self.step >= 4 and self.complete_until <= now:
            self.reset(student_id)
        if self.complete_until > now:
            return True, "Xác thực sống thành công"

        points = face.get("kps")
        if points is None:
            return False, "Cần camera YuNet: không thể xác thực chuyển động khuôn mặt"
        points = np.asarray(points, dtype=np.float32).reshape(5, 2)
        eye_distance = float(np.linalg.norm(points[1] - points[0]))
        if eye_distance < 8:
            return False, "Đưa khuôn mặt gần camera hơn"
        # YuNet nose displacement normalized by eye distance is a simple yaw cue.
        yaw = float((points[2][0] - (points[0][0] + points[1][0]) / 2) / eye_distance)

        if self.baseline is None:
            if abs(yaw) > 0.18:
                return False, "Nhìn thẳng vào camera để bắt đầu xác thực"
            self.baseline = yaw
            return False, self._prompt()

        # Steps: turn one way, return to center, turn the other way, return.
        targets = (self.first_direction, 0, -self.first_direction, 0)
        target = targets[self.step]
        delta = yaw - self.baseline
        reached = abs(delta) < 0.09 if target == 0 else delta * target > 0.17
        if reached:
            self.hold_count += 1
            if self.hold_count >= 2:
                self.step += 1
                self.hold_count = 0
                if self.step >= len(targets):
                    self.complete_until = now + 2.0
                    return True, "Xác thực sống thành công"
        else:
            self.hold_count = 0
        return False, self._prompt()

    def _prompt(self):
        if self.baseline is None:
            return "Nhìn thẳng vào camera để bắt đầu xác thực"
        first_side = "phải" if self.first_direction > 0 else "trái"
        other_side = "trái" if self.first_direction > 0 else "phải"
        prompts = (
            f"Quay đầu sang {first_side} màn hình và giữ nhẹ",
            "Nhìn thẳng lại camera",
            f"Quay đầu sang {other_side} màn hình và giữ nhẹ",
            "Nhìn thẳng lại để hoàn tất",
        )
        return prompts[min(self.step, len(prompts) - 1)]


def _providers():
    return ["CPUExecutionProvider"]


def warm_up() -> float:
    """Load models + 1 lần inference giả — cold-start target <3s."""
    t0 = time.perf_counter()
    _get_yunet()
    _get_sface()
    # dummy frame
    dummy = np.zeros((112, 112, 3), dtype=np.float32)
    try:
        extract_sface_embedding(dummy)
    except Exception:
        pass
    return (time.perf_counter() - t0) * 1000


def _get_sface() -> ort.InferenceSession:
    global _sess_sface
    if _sess_sface is None:
        if not SFACE_PATH.exists():
            raise FileNotFoundError(f"Thiếu SFace model: {SFACE_PATH}")
        _sess_sface = ort.InferenceSession(
            str(SFACE_PATH), sess_options=_so_options, providers=_providers()
        )
    return _sess_sface


def _get_yunet() -> ort.InferenceSession:
    global _sess_yunet
    if _sess_yunet is None:
        if not YUNET_PATH.exists():
            raise FileNotFoundError(f"Thiếu YuNet model: {YUNET_PATH}")
        _sess_yunet = ort.InferenceSession(
            str(YUNET_PATH), sess_options=_so_options, providers=_providers()
        )
    return _sess_yunet


def _get_spoof() -> Optional[ort.InferenceSession]:
    global _sess_spoof
    if _sess_spoof is None and ANTISPOOF_PATH.exists():
        _sess_spoof = ort.InferenceSession(
            str(ANTISPOOF_PATH), sess_options=_so_options, providers=_providers()
        )
    return _sess_spoof


def _antispoof_crop(image_bgr: np.ndarray, bbox: Optional[list], scale: float = 2.7) -> np.ndarray:
    """Crop an expanded BGR face patch for MiniFASNetV2 (80x80, 0..255)."""
    if bbox is None:
        raise ValueError("MiniFASNet cần bounding box khuôn mặt")
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    box_w, box_h = x2 - x1, y2 - y1
    if box_w <= 0 or box_h <= 0:
        raise ValueError("Bounding box khuôn mặt không hợp lệ")
    ratio = min((h - 1) / box_h, (w - 1) / box_w, scale)
    crop_w, crop_h = box_w * ratio, box_h * ratio
    cx, cy = x1 + box_w / 2, y1 + box_h / 2
    left = max(0, int(cx - crop_w / 2))
    top = max(0, int(cy - crop_h / 2))
    right = min(w - 1, int(cx + crop_w / 2))
    bottom = min(h - 1, int(cy + crop_h / 2))
    crop = image_bgr[top:bottom + 1, left:right + 1]
    if crop.size == 0:
        raise ValueError("Không crop được khuôn mặt cho MiniFASNet")
    import cv2
    return cv2.resize(crop, (80, 80), interpolation=cv2.INTER_LINEAR)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -15.0, 15.0)
    return 1.0 / (1.0 + np.exp(-x))


def _decode_yunet_outputs(
    outputs: list,
    out_names: list,
    input_w: int,
    input_h: int,
    scale_x: float,
    scale_y: float,
    orig_w: int,
    orig_h: int,
    score_thresh: float,
) -> Optional[dict]:
    """Giải mã output YuNet 2023 (cls/obj/bbox theo stride 8/16/32)."""
    by_name = {n: o for n, o in zip(out_names, outputs)}
    strides = [8, 16, 32]
    best = None
    best_area = 0.0

    for stride in strides:
        cls_k, obj_k, bbox_k = f"cls_{stride}", f"obj_{stride}", f"bbox_{stride}"
        if cls_k not in by_name or obj_k not in by_name or bbox_k not in by_name:
            continue
        cls = np.squeeze(by_name[cls_k]).astype(np.float32)
        obj = np.squeeze(by_name[obj_k]).astype(np.float32)
        bbox = np.squeeze(by_name[bbox_k]).astype(np.float32)
        if getattr(cls, "ndim", 0) == 0:
            continue
        if bbox.ndim == 1:
            bbox = bbox.reshape(1, -1)
        # OpenCV YuNet: score = sqrt(sigmoid(cls)*sigmoid(obj))
        cls_s = _sigmoid(cls)
        obj_s = _sigmoid(obj)
        scores = np.sqrt(np.maximum(cls_s * obj_s, 0.0))
        scores = np.asarray(scores).reshape(-1)
        cols = input_w // stride
        rows = input_h // stride
        n = min(len(scores), cols * rows)
        for idx in range(n):
            score = float(scores[idx])
            if score < score_thresh:
                continue
            r = idx // cols
            c = idx % cols
            dx, dy, dw, dh = [float(v) for v in bbox[idx][:4]]
            cx = (c + dx) * stride
            cy = (r + dy) * stride
            bw = float(np.exp(np.clip(dw, -10, 10))) * stride
            bh = float(np.exp(np.clip(dh, -10, 10))) * stride
            x1 = (cx - bw / 2) / scale_x
            y1 = (cy - bh / 2) / scale_y
            x2 = (cx + bw / 2) / scale_x
            y2 = (cy + bh / 2) / scale_y
            x1 = max(0.0, min(float(orig_w), x1))
            y1 = max(0.0, min(float(orig_h), y1))
            x2 = max(0.0, min(float(orig_w), x2))
            y2 = max(0.0, min(float(orig_h), y2))
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if area > best_area and (x2 - x1) > 12 and (y2 - y1) > 12:
                best_area = area
                best = {"bbox": [x1, y1, x2, y2], "score": score}
    return best


_cv_yunet = None
_cv_yunet_failed = False
_cv_size = (0, 0)
_haar = None


def _detect_with_opencv(image_bgr: np.ndarray, score_thresh: float) -> Optional[dict]:
    """OpenCV FaceDetectorYN."""
    global _cv_yunet, _cv_yunet_failed, _cv_size
    if _cv_yunet_failed:
        return None
    try:
        import cv2
    except ImportError:
        _cv_yunet_failed = True
        return None
    if not YUNET_PATH.exists():
        _cv_yunet_failed = True
        return None
    h, w = image_bgr.shape[:2]
    try:
        if _cv_yunet is None or _cv_size != (w, h):
            _cv_yunet = cv2.FaceDetectorYN.create(
                str(YUNET_PATH),
                "",
                (w, h),
                score_threshold=float(score_thresh),
                nms_threshold=0.3,
                top_k=5000,
            )
            _cv_size = (w, h)
        _cv_yunet.setInputSize((w, h))
        _cv_yunet.setScoreThreshold(float(score_thresh))
        _retval, faces = _cv_yunet.detect(image_bgr)
        if faces is None or len(faces) == 0:
            return None
        best = None
        best_area = 0.0
        for f in faces:
            x, y, bw, bh = float(f[0]), float(f[1]), float(f[2]), float(f[3])
            score = float(f[-1]) if len(f) > 4 else float(score_thresh)
            area = bw * bh
            if area > best_area:
                best_area = area
                kps = np.asarray(f[4:14], dtype=np.float32).reshape(5, 2) if len(f) >= 14 else None
                best = {"bbox": [x, y, x + bw, y + bh], "score": score, "kps": kps}
        return best
    except Exception:
        _cv_yunet_failed = True
        _cv_yunet = None
        return None


def _detect_with_haar(image_bgr: np.ndarray) -> Optional[dict]:
    """Fallback Haar cascade (luôn có sẵn trong OpenCV)."""
    global _haar
    try:
        import cv2
    except ImportError:
        return None
    try:
        if _haar is None:
            path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            _haar = cv2.CascadeClassifier(path)
            if _haar.empty():
                return None
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = _haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(48, 48))
        if faces is None or len(faces) == 0:
            return None
        x, y, bw, bh = max(faces, key=lambda f: f[2] * f[3])
        return {"bbox": [float(x), float(y), float(x + bw), float(y + bh)], "score": 0.6}
    except Exception:
        return None


def _detect_onnx_yunet(image_bgr: np.ndarray, score_thresh: float) -> Optional[dict]:
    """ONNX YuNet — letterbox + stretch (thử cả 2)."""
    session = _get_yunet()
    h, w = image_bgr.shape[:2]
    input_w = input_h = 640
    img_rgb = image_bgr[:, :, ::-1]
    out_names = [o.name for o in session.get_outputs()]

    def _run(canvas, sx, sy):
        blob = canvas.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...]
        feed = {session.get_inputs()[0].name: blob}
        for inp in session.get_inputs()[1:]:
            n = inp.name.lower()
            if "width" in n or n.endswith("_w"):
                feed[inp.name] = np.array([input_w], dtype=np.float32)
            elif "height" in n or n.endswith("_h"):
                feed[inp.name] = np.array([input_h], dtype=np.float32)
        outputs = session.run(out_names, feed)
        return _decode_yunet_outputs(
            outputs, out_names, input_w, input_h, sx, sy, w, h, score_thresh
        )

    # 1) Letterbox (giữ tỉ lệ)
    scale = min(input_w / w, input_h / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    pil = Image.fromarray(img_rgb).resize((nw, nh), Image.BILINEAR)
    canvas = np.zeros((input_h, input_w, 3), dtype=np.uint8)
    canvas[:nh, :nw] = np.array(pil)
    face = _run(canvas, scale, scale)
    if face is not None:
        return face

    # 2) Stretch full 640x640
    pil2 = Image.fromarray(img_rgb).resize((input_w, input_h), Image.BILINEAR)
    canvas2 = np.array(pil2)
    return _run(canvas2, input_w / w, input_h / h)


def detect_largest_face(image_bgr: np.ndarray, score_thresh: float = 0.35) -> Optional[dict]:
    """
    Phát hiện 1 khuôn mặt lớn nhất.
    Thứ tự: OpenCV FaceDetectorYN → ONNX YuNet → Haar cascade.
    """
    if image_bgr is None or image_bgr.size == 0:
        return None

    # Đảm bảo uint8 BGR 3 kênh
    if image_bgr.dtype != np.uint8:
        image_bgr = np.clip(image_bgr, 0, 255).astype(np.uint8)
    if image_bgr.ndim == 2:
        image_bgr = np.stack([image_bgr] * 3, axis=-1)
    elif image_bgr.shape[2] == 4:
        image_bgr = image_bgr[:, :, :3]

    face = _detect_with_opencv(image_bgr, score_thresh)
    if face is not None:
        return face

    try:
        face = _detect_onnx_yunet(image_bgr, score_thresh)
        if face is not None:
            return face
    except Exception:
        pass

    return _detect_with_haar(image_bgr)


def align_crop(image_bgr: np.ndarray, bbox: list, out_size: int = 112) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(bw, bh) * 1.25
    nx1 = int(max(0, cx - side / 2))
    ny1 = int(max(0, cy - side / 2))
    nx2 = int(min(w, cx + side / 2))
    ny2 = int(min(h, cy + side / 2))
    crop = image_bgr[ny1:ny2, nx1:nx2]
    if crop.size == 0:
        crop = image_bgr
    rgb = crop[:, :, ::-1]
    pil = Image.fromarray(rgb).resize((out_size, out_size), Image.BILINEAR)
    return np.array(pil)[:, :, ::-1]


def align_face(image_bgr: np.ndarray, face: dict, out_size: int = 112) -> np.ndarray:
    """Use OpenCV's reference SFace alignment when YuNet landmarks are present."""
    if face.get("kps") is not None:
        try:
            import cv2
            x1, y1, x2, y2 = face["bbox"]
            row = np.asarray([[x1, y1, x2 - x1, y2 - y1,
                               *np.asarray(face["kps"], dtype=np.float32).reshape(-1),
                               float(face.get("score", 1.0))]], dtype=np.float32)
            recognizer = _get_cv_sface()
            aligned = recognizer.alignCrop(image_bgr, row)
            if aligned is not None and aligned.size:
                return aligned
        except Exception:
            pass
    return align_crop(image_bgr, face["bbox"], out_size)


def _get_cv_sface():
    global _cv_sface
    if _cv_sface is None:
        import cv2
        _cv_sface = cv2.FaceRecognizerSF.create(str(SFACE_PATH), "")
    return _cv_sface


def extract_sface_embedding(aligned_bgr: np.ndarray) -> np.ndarray:
    # Use the same official OpenCV feature path for enrollment and webcam frames.
    try:
        out = _get_cv_sface().feature(aligned_bgr)
    except Exception:
        session = _get_sface()
        rgb = aligned_bgr[:, :, ::-1].astype(np.float32)
        if rgb.shape[0] != 112 or rgb.shape[1] != 112:
            pil = Image.fromarray(rgb.astype(np.uint8)).resize((112, 112), Image.BILINEAR)
            rgb = np.array(pil).astype(np.float32)
        blob = rgb.transpose(2, 0, 1)[np.newaxis, ...]
        out = session.run(None, {session.get_inputs()[0].name: blob})[0]
    emb = out.flatten().astype(np.float32)
    return emb / (np.linalg.norm(emb) + 1e-8)


def check_anti_spoof(image_bgr: np.ndarray, bbox: Optional[list] = None) -> Tuple[bool, float]:
    """Run MiniFASNetV2; fail closed when the model is missing or errors."""
    try:
        sess = _get_spoof()
        if sess is None:
            return False, 0.0
        crop = _antispoof_crop(image_bgr, bbox)
        blob = crop.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...]
        input_meta = sess.get_inputs()[0]
        output = np.asarray(sess.run(None, {input_meta.name: blob})[0], dtype=np.float32)
        logits = output.reshape(1, -1)
        if logits.shape[1] < 2 or not np.isfinite(logits).all():
            return False, 0.0
        logits -= np.max(logits, axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= np.sum(probs, axis=1, keepdims=True) + 1e-8
        real_confidence = float(probs[0, 1])  # MiniFASNetV2 class 1 = real
        return real_confidence >= 0.5, real_confidence
    except Exception:
        return False, 0.0


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-8) * (np.linalg.norm(b) + 1e-8)))


def train_templates(
    samples: List[Tuple[int, str, np.ndarray]],
) -> List[Tuple[int, str, np.ndarray, List[np.ndarray]]]:
    """
    Training gallery cục bộ trên Edge:
    - Gom nhiều embedding theo từng sinh viên
    - Tính centroid (trung bình rồi L2-normalize) làm template chính
    - Giữ toàn bộ samples để match max-similarity

    Input:  list (student_id, full_name, embedding)
    Output: list (student_id, full_name, centroid, [sample_emb, ...])
    """
    groups: Dict[int, Dict[str, Any]] = {}
    for uid, name, emb in samples:
        emb = np.asarray(emb, dtype=np.float32).flatten()
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        if uid not in groups:
            groups[uid] = {"name": name, "embs": []}
        groups[uid]["embs"].append(emb)

    templates: List[Tuple[int, str, np.ndarray, List[np.ndarray]]] = []
    for uid, g in groups.items():
        embs = g["embs"]
        mean = np.mean(np.stack(embs, axis=0), axis=0)
        mean = mean / (np.linalg.norm(mean) + 1e-8)
        templates.append((uid, g["name"], mean.astype(np.float32), embs))
    return templates


def match_gallery(
    emb: np.ndarray,
    gallery: List,
    threshold: float = SFACE_COSINE_THRESHOLD,
) -> Tuple[Optional[int], Optional[str], float]:
    """
    Match 1:N — hỗ trợ nhiều mẫu / SV:
    - gallery item: (id, name, emb)  hoặc  (id, name, centroid, [samples])
    - Điểm = max( sim(centroid), max_i sim(sample_i) )
    """
    emb = np.asarray(emb, dtype=np.float32).flatten()
    emb = emb / (np.linalg.norm(emb) + 1e-8)

    scores = []
    for item in gallery:
        if len(item) >= 4:
            uid, name, centroid, samples = item[0], item[1], item[2], item[3]
            sims = [cosine_sim(emb, centroid)]
            for s in samples:
                sims.append(cosine_sim(emb, s))
            sim = float(max(sims))
        else:
            uid, name, gemb = item[0], item[1], item[2]
            sim = cosine_sim(emb, gemb)
        scores.append((uid, name, sim))

    if not scores:
        return None, None, -1.0
    scores.sort(key=lambda item: item[2], reverse=True)
    best_id, best_name, best_sim = scores[0]
    # Require both a reasonable absolute score and a clear lead over the
    # second identity, so a weak glasses/pose match is not assigned to someone else.
    if best_sim < threshold:
        return None, None, best_sim
    if len(scores) > 1 and best_sim - scores[1][2] < SFACE_COSINE_MARGIN:
        return None, None, best_sim
    return best_id, best_name, best_sim


def process_frame(
    image_bgr: np.ndarray,
    gallery: List,
    liveness: Optional[ActiveLiveness] = None,
) -> EdgeResult:
    """
    Full edge pipeline. gallery = templates từ train_templates() hoặc list (id,name,emb).
    """
    t0 = time.perf_counter()

    face = detect_largest_face(image_bgr)
    if face is None:
        ms = (time.perf_counter() - t0) * 1000
        return EdgeResult(False, False, None, None, 0.0, ms, "Không có khuôn mặt")

    bbox = face["bbox"]
    if min(bbox[2] - bbox[0], bbox[3] - bbox[1]) < 60:
        ms = (time.perf_counter() - t0) * 1000
        return EdgeResult(
            False, False, None, None, 0.0, ms,
            "Vui lĂ²ng tiáº¿n láº¡i gáº§n camera hÆ¡n", bbox=bbox,
        )

    is_real, spoof_score = check_anti_spoof(image_bgr, bbox)
    if not is_real:
        ms = (time.perf_counter() - t0) * 1000
        message = (
            "Thiếu model anti-spoof MiniFASNet; điểm danh bị khóa an toàn"
            if not ANTISPOOF_PATH.is_file()
            else "Từ chối: nghi ngờ giả mạo"
        )
        return EdgeResult(
            False, True, None, None, spoof_score, ms,
            message, bbox=bbox,
        )

    aligned = align_face(image_bgr, face)
    emb = extract_sface_embedding(aligned)
    sid, name, sim = match_gallery(emb, gallery)

    ms = (time.perf_counter() - t0) * 1000
    if sid is None:
        return EdgeResult(False, False, None, None, sim, ms,
                          f"Không khớp (sim={sim:.3f})", bbox=bbox, embedding=emb)

    if liveness is not None:
        live, live_message = liveness.advance(sid, face)
        ms = (time.perf_counter() - t0) * 1000
        if not live:
            return EdgeResult(
                False, False, sid, name, sim, ms, live_message,
                bbox=bbox, liveness_pending=True,
            )

    return EdgeResult(
        True, False, sid, name, sim, ms,
        f"Thành công: {name} ({sim:.3f})", bbox=bbox,
    )
