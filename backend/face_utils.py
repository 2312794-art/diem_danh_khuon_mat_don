"""
Face recognition utilities using **SFace** (OpenCV Zoo: face_recognition_sface_2021dec.onnx).
Detection: YuNet (face_detection_yunet_2023mar.onnx).
Không dùng buffalo_l / InsightFace recognition — chỉ SFace.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List
from io import BytesIO

import numpy as np
from PIL import Image
import onnxruntime as ort

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
UPLOADS_DIR = Path(__file__).resolve().parent.parent / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SFACE_PATH = MODELS_DIR / "face_recognition_sface_2021dec.onnx"
YUNET_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
ANTISPOOF_PATH = MODELS_DIR / "antispoof_minifasnet.onnx"

# Ngưỡng cosine tham chiếu của SFace theo OpenCV. Gallery thực tế cần hiệu
# chỉnh theo dữ liệu enroll, nhưng không nên siết cao hơn mặc định tùy tiện.
SFACE_COSINE_THRESHOLD = 0.363

_sface_session: Optional[ort.InferenceSession] = None
_yunet_session: Optional[ort.InferenceSession] = None
_cv_sface = None
_cv_yunet = None
_cv_yunet_size = (0, 0)
_antispoof_session: Optional[ort.InferenceSession] = None


def _get_antispoof_session() -> Optional[ort.InferenceSession]:
    global _antispoof_session
    if _antispoof_session is None and ANTISPOOF_PATH.is_file():
        _antispoof_session = ort.InferenceSession(
            str(ANTISPOOF_PATH), providers=["CPUExecutionProvider"]
        )
    return _antispoof_session


def anti_spoof_model_available() -> bool:
    return ANTISPOOF_PATH.is_file()


def _get_sface() -> ort.InferenceSession:
    global _sface_session
    if _sface_session is None:
        if not SFACE_PATH.exists():
            raise FileNotFoundError(
                f"Không tìm thấy model SFace: {SFACE_PATH}\n"
                "Tải: https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
            )
        _sface_session = ort.InferenceSession(
            str(SFACE_PATH), providers=["CPUExecutionProvider"]
        )
    return _sface_session


def _get_yunet() -> ort.InferenceSession:
    global _yunet_session
    if _yunet_session is None:
        if not YUNET_PATH.exists():
            raise FileNotFoundError(f"Không tìm thấy model YuNet: {YUNET_PATH}")
        _yunet_session = ort.InferenceSession(
            str(YUNET_PATH), providers=["CPUExecutionProvider"]
        )
    return _yunet_session


def load_image_bgr(path: str) -> Optional[np.ndarray]:
    try:
        img = Image.open(path).convert("RGB")
        rgb = np.array(img)
        return rgb[:, :, ::-1].copy()  # RGB -> BGR
    except Exception:
        return None


def load_image_bgr_from_bytes(data: bytes) -> Optional[np.ndarray]:
    try:
        img = Image.open(BytesIO(data)).convert("RGB")
        rgb = np.array(img)
        return rgb[:, :, ::-1].copy()
    except Exception:
        return None


def _detect_faces_yunet(image_bgr: np.ndarray, score_thresh: float = 0.35) -> List[dict]:
    """Detect faces with YuNet and return boxes and five landmarks.

    Prefer OpenCV's YuNet wrapper because it decodes the ONNX output and
    landmark coordinates using the model's canonical implementation. Keep the
    raw ONNX decoder as a compatibility fallback.
    """
    global _cv_yunet, _cv_yunet_size
    try:
        import cv2
        size = (image_bgr.shape[1], image_bgr.shape[0])
        if _cv_yunet is None or _cv_yunet_size != size:
            _cv_yunet = cv2.FaceDetectorYN.create(
                str(YUNET_PATH), "", size,
                score_threshold=float(score_thresh), nms_threshold=0.3, top_k=5000,
            )
            _cv_yunet_size = size
        _cv_yunet.setScoreThreshold(float(score_thresh))
        _retval, detections = _cv_yunet.detect(image_bgr)
        if detections is not None and len(detections):
            faces = []
            for row in detections:
                x, y, bw, bh = [float(v) for v in row[:4]]
                faces.append({
                    "bbox": [x, y, x + bw, y + bh],
                    "kps": np.asarray(row[4:14], dtype=np.float32).reshape(5, 2),
                    "score": float(row[14]),
                })
            return faces
    except Exception:
        # Older OpenCV builds can still use the ONNX decoder below.
        pass

    session = _get_yunet()
    h, w = image_bgr.shape[:2]
    # Model YuNet 2023mar expect fixed 640x640
    input_w, input_h = 640, 640
    img_rgb = image_bgr[:, :, ::-1]
    scale = min(input_w / w, input_h / h)
    nw, nh = int(w * scale), int(h * scale)
    pil = Image.fromarray(img_rgb).resize((nw, nh), Image.BILINEAR)
    canvas = np.zeros((input_h, input_w, 3), dtype=np.uint8)
    canvas[:nh, :nw] = np.array(pil)

    blob = canvas.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...]
    input_name = session.get_inputs()[0].name
    feed = {input_name: blob}
    for inp in session.get_inputs()[1:]:
        n = inp.name.lower()
        if "width" in n or n.endswith("_w"):
            feed[inp.name] = np.array([input_w], dtype=np.float32)
        elif "height" in n or n.endswith("_h"):
            feed[inp.name] = np.array([input_h], dtype=np.float32)
    outputs = session.run(None, feed)

    # YuNet 2023 ONNX exposes raw cls/obj/bbox/kps heads, not a ready-made
    # Nx15 detection array. Decode each stride the same way as the Edge path.
    by_name = {out.name: value for out, value in zip(session.get_outputs(), outputs)}
    faces: List[dict] = []
    for stride in (8, 16, 32):
        names = (f"cls_{stride}", f"obj_{stride}", f"bbox_{stride}", f"kps_{stride}")
        if not all(name in by_name for name in names[:3]):
            continue
        cls = np.asarray(by_name[names[0]], dtype=np.float32).reshape(-1)
        obj = np.asarray(by_name[names[1]], dtype=np.float32).reshape(-1)
        boxes = np.asarray(by_name[names[2]], dtype=np.float32).reshape(-1, 4)
        kp_head = (np.asarray(by_name[names[3]], dtype=np.float32).reshape(-1, 10)
                   if names[3] in by_name else None)
        # This model exports logits for class/objectness.
        cls_prob = 1.0 / (1.0 + np.exp(-np.clip(cls, -15, 15)))
        obj_prob = 1.0 / (1.0 + np.exp(-np.clip(obj, -15, 15)))
        scores = np.sqrt(cls_prob * obj_prob)
        cols = input_w // stride
        for idx in np.flatnonzero(scores >= score_thresh):
            row, col = divmod(int(idx), cols)
            dx, dy, dw, dh = boxes[idx]
            cx, cy = (col + float(dx)) * stride, (row + float(dy)) * stride
            bw = float(np.exp(np.clip(dw, -10, 10))) * stride
            bh = float(np.exp(np.clip(dh, -10, 10))) * stride
            x1 = max(0.0, min(float(w), (cx - bw / 2) / scale))
            y1 = max(0.0, min(float(h), (cy - bh / 2) / scale))
            x2 = max(0.0, min(float(w), (cx + bw / 2) / scale))
            y2 = max(0.0, min(float(h), (cy + bh / 2) / scale))
            if x2 - x1 < 12 or y2 - y1 < 12:
                continue
            kps = None
            if kp_head is not None:
                points = kp_head[idx].reshape(5, 2)
                kps = np.asarray([[(col + float(px)) * stride / scale,
                                   (row + float(py)) * stride / scale]
                                  for px, py in points], dtype=np.float32)
            faces.append({"bbox": [x1, y1, x2, y2], "kps": kps,
                          "score": float(scores[idx])})

    return faces


def _align_crop_sface(image_bgr: np.ndarray, face: dict, out_size: int = 112) -> np.ndarray:
    """Crop + resize face về 112x112 cho SFace."""
    if face.get("kps") is not None:
        try:
            import cv2
            x1, y1, x2, y2 = face["bbox"]
            row = np.asarray([[x1, y1, x2 - x1, y2 - y1,
                               *np.asarray(face["kps"], dtype=np.float32).reshape(-1),
                               float(face.get("score", 1.0))]], dtype=np.float32)
            aligned = _get_cv_sface().alignCrop(image_bgr, row)
            if aligned is not None and aligned.size:
                return aligned
        except Exception:
            pass
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = face["bbox"]
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
    return np.array(pil)[:, :, ::-1]  # BGR


def extract_sface_embedding(aligned_bgr: np.ndarray) -> np.ndarray:
    """SFace → embedding 128 chiều, đã L2-normalize."""
    try:
        out = _get_cv_sface().feature(aligned_bgr)
    except Exception:
        session = _get_sface()
        img = aligned_bgr[:, :, ::-1].astype(np.float32)
        if img.shape[0] != 112 or img.shape[1] != 112:
            pil = Image.fromarray(img.astype(np.uint8)).resize((112, 112), Image.BILINEAR)
            img = np.array(pil).astype(np.float32)
        blob = img.transpose(2, 0, 1)[np.newaxis, ...]
        input_name = session.get_inputs()[0].name
        out = session.run(None, {input_name: blob})[0]
    emb = out.flatten().astype(np.float32)
    norm = np.linalg.norm(emb) + 1e-8
    return emb / norm


def _get_cv_sface():
    global _cv_sface
    if _cv_sface is None:
        import cv2
        _cv_sface = cv2.FaceRecognizerSF.create(str(SFACE_PATH), "")
    return _cv_sface


def extract_embedding(image_bgr: np.ndarray) -> Optional[Tuple[np.ndarray, dict]]:
    """Detect face lớn nhất + trích xuất SFace embedding 128-d."""
    faces = _detect_faces_yunet(image_bgr)
    if not faces:
        return None
    face = max(
        faces,
        key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]),
    )
    aligned = _align_crop_sface(image_bgr, face)
    emb = extract_sface_embedding(aligned)
    info = {
        "bbox": face["bbox"],
        "kps": face["kps"].tolist() if face.get("kps") is not None else None,
        "det_score": float(face["score"]),
        "model": "SFace",
    }
    return emb, info


def extract_embedding_from_file(path: str) -> Optional[Tuple[np.ndarray, dict]]:
    img = load_image_bgr(path)
    if img is None:
        return None
    return extract_embedding(img)


def embedding_to_bytes(embedding: np.ndarray) -> bytes:
    return embedding.astype(np.float32).tobytes()


def bytes_to_embedding(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def check_anti_spoof(image_bgr: np.ndarray, face_bbox: Optional[list] = None) -> Tuple[bool, float]:
    """Run MiniFASNetV2 on an expanded BGR face crop; fail closed on errors."""
    try:
        session = _get_antispoof_session()
        if session is None or face_bbox is None:
            return False, 0.0
        h, w = image_bgr.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in face_bbox]
        box_w, box_h = x2 - x1, y2 - y1
        if box_w <= 0 or box_h <= 0:
            return False, 0.0
        scale = min((h - 1) / box_h, (w - 1) / box_w, 2.7)
        crop_w, crop_h = box_w * scale, box_h * scale
        cx, cy = x1 + box_w / 2, y1 + box_h / 2
        left, top = max(0, int(cx - crop_w / 2)), max(0, int(cy - crop_h / 2))
        right, bottom = min(w - 1, int(cx + crop_w / 2)), min(h - 1, int(cy + crop_h / 2))
        crop = image_bgr[top:bottom + 1, left:right + 1]
        if crop.size == 0:
            return False, 0.0
        import cv2
        crop = cv2.resize(crop, (80, 80), interpolation=cv2.INTER_LINEAR)
        blob = crop.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...]
        input_meta = session.get_inputs()[0]
        output = np.asarray(session.run(None, {input_meta.name: blob})[0], dtype=np.float32)
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


def match_face(
    query_emb: np.ndarray,
    gallery: List[Tuple[int, np.ndarray]],
    threshold: float = SFACE_COSINE_THRESHOLD,
    margin: float = 0.05,  # Chênh lệch tối thiểu giữa Top 1 và Top 2
) -> Tuple[Optional[int], float]:
    """1:N matching nâng cao - Chống nhận nhầm giữa các sinh viên."""
    if not gallery:
        return None, -1.0

    # Several approved photos belong to the same student. Collapse them before
    # computing the runner-up margin; otherwise that student's second photo
    # can incorrectly compete with their own best match.
    best_by_student = {}
    for uid, emb in gallery:
        sim = cosine_similarity(query_emb, emb)
        best_by_student[uid] = max(sim, best_by_student.get(uid, -1.0))
    scores = list(best_by_student.items())

    # Sắp xếp giảm dần theo độ tương đồng
    scores.sort(key=lambda x: x[1], reverse=True)
    
    best_id, best_sim = scores[0]

    # Điều kiện 1: Phải vượt qua ngưỡng threshold
    if best_sim < threshold:
        return None, best_sim

    # Điều kiện 2: Nếu có từ 2 SV trở lên trong gallery, Top 1 phải vượt trội hơn Top 2
    if len(scores) > 1:
        second_sim = scores[1][1]
        # Nếu Top 1 và Top 2 quá suýt soát nhau (chênh lệch < margin), hủy kết quả để tránh nhận nhầm
        if (best_sim - second_sim) < margin:
            return None, best_sim

    return best_id, best_sim


def save_image(image_bgr: np.ndarray, subdir: str, filename: str) -> str:
    folder = UPLOADS_DIR / subdir
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    rgb = image_bgr[:, :, ::-1]
    Image.fromarray(rgb.astype(np.uint8)).save(str(path), quality=92)
    # Luôn trả về path dùng dấu / (tránh lỗi Windows khi hiển thị ảnh trên web)
    rel = path.relative_to(UPLOADS_DIR.parent)
    return str(rel).replace("\\", "/")
