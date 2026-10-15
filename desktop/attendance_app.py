"""
Desktop = Monolithic AI Edge
- Toàn bộ: camera → detect → anti-spoof → SFace match chạy LOCAL (CPU)
- Chỉ gọi Central Server để: (1) tải gallery session (2) ghi kết quả (3) upload ảnh spoof
- Mục tiêu: E2E ≤150ms, video ≥25 FPS, cold-start <3s, RAM kiểm soát
"""
from __future__ import annotations

import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from collections import deque

import numpy as np
import requests

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import customtkinter as ctk
    USE_CTK = True
except ImportError:
    import tkinter as ctk
    USE_CTK = False

# Edge AI monolithic
sys.path.insert(0, str(Path(__file__).resolve().parent))
from edge_ai import process_frame, warm_up, EdgeResult, train_templates

API_BASE = os.getenv("FACE_API", "http://127.0.0.1:8000").rstrip("/")
try:
    import platform
    _host = platform.node() or "pc"
except Exception:
    _host = "pc"
DEVICE_INFO = f"edge-{_host}"
# Xử lý nhận diện mỗi N frame để giữ ≥25 FPS hiển thị
PROCESS_EVERY_N_FRAMES = 2
MATCH_COOLDOWN_SEC = 2.0  # tránh điểm danh trùng liên tục cùng người


class FaceAttendanceApp:
    def __init__(self):
        if USE_CTK:
            ctk.set_appearance_mode("light")
            ctk.set_default_color_theme("blue")
            self.root = ctk.CTk()
        else:
            self.root = ctk.Tk()
        self.root.title("AI Edge — Điểm danh Gương mặt")
        self.root.geometry("1120x720")
        self.root.minsize(900, 600)
        if USE_CTK:
            self.root.configure(fg_color="#F3F1FD")

        self.cap = None
        self.running = False
        self.session_id = None
        self.room = None
        self.class_name = ""
        self.gallery = []  # [(id, name, emb np)]
        self.last_match_uid = None
        self.last_match_time = 0.0
        self.frame_i = 0
        self.fps_times = deque(maxlen=30)
        self.last_result: Optional[EdgeResult] = None
        self._lock = threading.Lock()

        self._build_ui()
        # Cold-start models trên thread phụ
        threading.Thread(target=self._cold_start, daemon=True).start()

    def _cold_start(self):
        try:
            ms = warm_up()
            self.log(f"Cold-start models: {ms:.0f} ms")
        except Exception as e:
            self.log(f"Warm-up lỗi: {e}")

    def _build_ui(self):
        if USE_CTK:
            header = ctk.CTkFrame(self.root, fg_color="#FFFFFF", corner_radius=16)
        else:
            header = ctk.Frame(self.root, bg="#FFFFFF")
        header.pack(fill="x", padx=16, pady=(14, 8))

        if USE_CTK:
            ctk.CTkLabel(header, text="▣", text_color="#7565E5",
                         font=("", 26, "bold")).pack(side="left", padx=(16, 8), pady=10)
            title_box = ctk.CTkFrame(header, fg_color="transparent")
            title_box.pack(side="left", pady=7)
            ctk.CTkLabel(title_box, text="Face Attendance",
                         text_color="#332D52", font=("", 16, "bold")).pack(anchor="w")
            ctk.CTkLabel(title_box, text="SMART CLASSROOM • EDGE DEVICE",
                         text_color="#9892AE", font=("", 9)).pack(anchor="w")
        top = ctk.CTkFrame(header, fg_color="transparent") if USE_CTK else ctk.Frame(header, bg="#FFFFFF")
        top.pack(side="left", fill="x", expand=True, padx=(22, 10), pady=8)

        def L(t):
            return ctk.CTkLabel(top, text=t, text_color="#77718E") if USE_CTK else ctk.Label(top, text=t, fg="#77718E", bg="#FFFFFF")

        def E(w=100):
            return ctk.CTkEntry(top, width=w, height=34, corner_radius=9,
                                fg_color="#F8F7FC", border_color="#E4E0F4",
                                text_color="#332D52") if USE_CTK else ctk.Entry(top, width=12)

        def B(text, cmd, **kw):
            if USE_CTK:
                return ctk.CTkButton(top, text=text, command=cmd, height=34,
                                     corner_radius=9, fg_color="#7565E5",
                                     hover_color="#5F50CA", **kw)
            return ctk.Button(top, text=text, command=cmd)

        L("API:").pack(side="left", padx=2)
        self.api_entry = E(180)
        self.api_entry.pack(side="left", padx=2)
        try:
            self.api_entry.insert(0, API_BASE)
        except Exception:
            pass
        L("Phòng:").pack(side="left", padx=4)
        self.room_entry = E(100)
        self.room_entry.pack(side="left", padx=4)
        self.connect_btn = B("Đồng bộ session", self.connect_session)
        self.connect_btn.pack(side="left", padx=6)
        self.start_btn = B("Bắt đầu camera", self.toggle_camera)
        if USE_CTK:
            self.start_btn.configure(state="disabled")
        else:
            self.start_btn.configure(state="disabled")
        self.start_btn.pack(side="left", padx=4)
        self.status_label = L("● Chưa kết nối")
        self.status_label.pack(side="left", padx=12)
        self.fps_label = L("FPS: — | Latency: —")
        self.fps_label.pack(side="right", padx=8)

        if USE_CTK:
            body = ctk.CTkFrame(self.root, fg_color="transparent")
        else:
            body = ctk.Frame(self.root, bg="#F3F1FD")
        body.pack(fill="both", expand=True, padx=16, pady=(4, 16))

        if USE_CTK:
            camera_card = ctk.CTkFrame(body, fg_color="#FFFFFF", corner_radius=16)
        else:
            camera_card = ctk.Frame(body, bg="#FFFFFF")
        camera_card.pack(side="left", fill="both", expand=True)
        if USE_CTK:
            ctk.CTkLabel(camera_card, text="KHU VỰC CAMERA", text_color="#9993AF",
                         font=("", 10, "bold")).pack(anchor="w", padx=18, pady=(14, 4))
            self.video_label = ctk.CTkLabel(camera_card, text="Camera chưa được bật\n\nĐồng bộ session để bắt đầu",
                                            text_color="#AAA5BC", font=("", 16), fg_color="#F7F6FC",
                                            corner_radius=12)
        else:
            self.video_label = ctk.Label(camera_card, text="Camera chưa được bật\n\nĐồng bộ session để bắt đầu",
                                         fg="#AAA5BC", bg="#F7F6FC")
        self.video_label.pack(fill="both", expand=True, padx=14, pady=(0, 14))

        right = ctk.CTkFrame(body, width=310, fg_color="#FFFFFF", corner_radius=16) if USE_CTK else ctk.Frame(body, width=310, bg="#FFFFFF")
        right.pack(side="right", fill="y", padx=(12, 0))
        if USE_CTK:
            ctk.CTkLabel(right, text="Nhật ký Edge", text_color="#332D52",
                         font=("", 14, "bold")).pack(anchor="w", padx=16, pady=(16, 2))
            ctk.CTkLabel(right, text="Hoạt động hệ thống theo thời gian thực",
                         text_color="#9993AF", font=("", 10)).pack(anchor="w", padx=16, pady=(0, 10))
            self.log_box = ctk.CTkTextbox(right, width=280, height=480,
                                          fg_color="#F8F7FC", border_width=1,
                                          border_color="#EEEAF8", text_color="#5C5676",
                                          corner_radius=10)
        else:
            ctk.Label(right, text="Nhật ký Edge", fg="#332D52", bg="#FFFFFF").pack()
            self.log_box = ctk.Text(right, width=36, height=32, bg="#F8F7FC", fg="#5C5676")
        self.log_box.pack(fill="both", expand=True, padx=14, pady=(0, 14))

    def log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}\n"
        try:
            self.log_box.insert("end", line)
            self.log_box.see("end")
        except Exception:
            pass
        print(line.strip())

    def _api_base(self) -> str:
        global API_BASE
        try:
            val = self.api_entry.get().strip().rstrip("/")
            if val:
                API_BASE = val
        except Exception:
            pass
        return API_BASE

    def connect_session(self):
        room = self.room_entry.get().strip()
        if not room:
            self.log("Nhập mã phòng")
            return
        base = self._api_base()
        url = f"{base}/api/desktop/room/{room}/session"
        self.log(f"Đang gọi: {url}")
        try:
            r = requests.get(url, timeout=8)
            text = (r.text or "").strip()
            if not text:
                self.log(
                    f"Lỗi đồng bộ: server trả body rỗng (HTTP {r.status_code}). "
                    f"Kiểm tra API đang chạy và địa chỉ {base}"
                )
                return
            try:
                data = r.json()
            except Exception:
                self.log(
                    f"Lỗi đồng bộ: không phải JSON (HTTP {r.status_code}): {text[:200]}"
                )
                return
            if r.status_code >= 400:
                self.log(f"Lỗi server HTTP {r.status_code}: {data}")
                return
            if not data.get("found"):
                self.log(
                    f"Không có session mở cho phòng '{room}'. "
                    "Hãy mở buổi điểm danh trên web (GV) với đúng mã phòng."
                )
                return
            self.session_id = data["session_id"]
            self.room = room
            self.class_name = data.get("class_name", "")
            # Thu thập mọi mẫu embedding (mỗi SV có thể nhiều ảnh)
            raw_samples = []
            sample_source_ids = {}
            for s in data.get("students", []):
                embs = s.get("embeddings") or []
                face_ids = s.get("embedding_face_ids") or []
                if not embs and s.get("embedding") is not None:
                    embs = [s["embedding"]]
                sample_source_ids[s["id"]] = []
                for index, e in enumerate(embs):
                    emb = np.array(e, dtype=np.float32).flatten()
                    emb = emb / (np.linalg.norm(emb) + 1e-8)
                    raw_samples.append((s["id"], s["full_name"], emb))
                    sample_source_ids[s["id"]].append(
                        face_ids[index] if index < len(face_ids) else index + 1
                    )
            # Train templates: centroid + giữ samples
            self.gallery = train_templates(raw_samples)
            n_sv = len(self.gallery)
            n_samples = len(raw_samples)
            self.log(
                f"Đồng bộ OK: {self.class_name} | session#{self.session_id} | "
                f"{n_sv} SV | {n_samples} mẫu (đã train centroid)"
            )
            for uid, name, centroid, samples in self.gallery:
                self.log(f"  · {name}: {len(samples)} ảnh")
                if len(samples) > 1:
                    pair_scores = [
                        (i, j, float(np.dot(samples[i], samples[j]) /
                              ((np.linalg.norm(samples[i]) + 1e-8) *
                               (np.linalg.norm(samples[j]) + 1e-8))))
                        for i in range(len(samples))
                        for j in range(i + 1, len(samples))
                    ]
                    source_ids = sample_source_ids.get(uid, [])
                    per_sample = []
                    for i in range(len(samples)):
                        related = [score for a, b, score in pair_scores if i in (a, b)]
                        sid = source_ids[i] if i < len(source_ids) else i + 1
                        per_sample.append((sum(related) / len(related), sid))
                    outlier_mean, outlier_id = min(per_sample)
                    self.log(
                        f"  Kiểm tra ảnh đăng ký của {name}: "
                        f"sim cặp thấp nhất={min(x[2] for x in pair_scores):.3f}, "
                        f"trung bình={sum(x[2] for x in pair_scores) / len(pair_scores):.3f}; "
                        f"mẫu lệch nhất=#{outlier_id} (sim TB {outlier_mean:.3f})"
                    )
            rebuilt = sum(int(s.get("embeddings_rebuilt", 0)) for s in data.get("students", []))
            cached = sum(int(s.get("embeddings_from_cache", 0)) for s in data.get("students", []))
            self.log(f"Embedding ảnh đăng ký: tạo lại {rebuilt} | dùng cache {cached}")
            if n_sv == 0:
                self.log(
                    "Cảnh báo: chưa có SV nào có khuôn mặt đã duyệt trong lớp này."
                )
            self.status_label.configure(text=f"Edge | {self.class_name} | {room}")
            self.start_btn.configure(state="normal")
        except requests.exceptions.ConnectionError:
            self.log(
                f"Không kết nối được server {base}. "
                "Chạy uvicorn --host 0.0.0.0 --port 8000 và kiểm tra IP/firewall."
            )
        except Exception as e:
            self.log(f"Lỗi đồng bộ server: {e}")

    def toggle_camera(self):
        if not HAS_CV2:
            self.log("Cần OpenCV (cv2) để mở camera")
            return
        if self.running:
            self.running = False
            if self.cap:
                self.cap.release()
                self.cap = None
            self.start_btn.configure(text="Bắt đầu camera")
            self.log("Dừng camera")
            return
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        if not self.cap.isOpened():
            self.log("Không mở được camera")
            return
        self.running = True
        self.start_btn.configure(text="Dừng camera")
        self.log("Camera ON — AI chạy local (không gửi frame lên server)")
        threading.Thread(target=self._capture_loop, daemon=True).start()

    def _capture_loop(self):
        """Luồng hiển thị ≥25 FPS; AI xử lý subsample frames."""
        from PIL import Image, ImageTk

        while self.running and self.cap:
            t_frame = time.perf_counter()
            ret, frame = self.cap.read()
            if not ret:
                break
            self.frame_i += 1
            display = frame.copy()
            res = None

            # Có session thì luôn nhận diện (kể cả gallery rỗng để hiện thông báo)
            if self.session_id and (self.frame_i % PROCESS_EVERY_N_FRAMES == 0):
                if not self.gallery:
                    res = EdgeResult(
                        False, False, None, None, 0.0, 0.0,
                        "Gallery rỗng — chưa có SV duyệt khuôn mặt",
                    )
                else:
                    try:
                        res = process_frame(frame, self.gallery)
                    except Exception as e:
                        res = EdgeResult(
                            False, False, None, None, 0.0, 0.0,
                            f"Lỗi AI: {e}",
                        )
                with self._lock:
                    self.last_result = res
                if res:
                    self._handle_result(res, frame)

            with self._lock:
                res = self.last_result

            # Banner trạng thái trên toàn khung hình
            h, w = display.shape[:2]
            if res is None:
                status = "Dang quet..."
                color = (200, 200, 200)
            elif res.is_spoof:
                status = "TU CHOI - Gia mao"
                color = (0, 0, 255)
            elif res.ok:
                status = f"DIEM DANH OK: {res.full_name or ''}"
                color = (0, 220, 0)
            else:
                status = res.message[:50] if res.message else "Khong khop"
                color = (0, 165, 255)

            # Thanh trạng thái phía trên
            cv2.rectangle(display, (0, 0), (w, 36), (30, 30, 30), -1)
            cv2.putText(
                display, status, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA,
            )

            # Vẽ bbox nếu có
            if res and res.bbox:
                x1, y1, x2, y2 = [int(v) for v in res.bbox]
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                conf = f"sim={res.confidence:.3f}" if res.confidence else ""
                cv2.putText(
                    display, conf, (x1, min(h - 8, y2 + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
                )

            # FPS
            self.fps_times.append(time.perf_counter() - t_frame)
            if len(self.fps_times) >= 5:
                fps = 1.0 / (sum(self.fps_times) / len(self.fps_times) + 1e-9)
                lat = res.latency_ms if res else 0
                try:
                    self.fps_label.configure(text=f"FPS: {fps:.1f} | Latency: {lat:.0f}ms")
                except Exception:
                    pass

            rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            try:
                if USE_CTK:
                    # CustomTkinter yêu cầu CTkImage (tránh cảnh báo PhotoImage)
                    ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(img.width, img.height))
                    self.video_label.configure(image=ctk_img, text="")
                    self.video_label._video_img = ctk_img  # giữ tham chiếu
                else:
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.video_label.imgtk = imgtk
                    self.video_label.configure(image=imgtk)
            except Exception:
                pass

            elapsed = time.perf_counter() - t_frame
            time.sleep(max(0.0, 0.033 - elapsed))

    def _handle_result(self, result: EdgeResult, frame_bgr):
        now = time.time()
        if result.is_spoof:
            # log spoof tối đa mỗi 3s
            if now - getattr(self, "_last_spoof_log", 0) > 3:
                self._last_spoof_log = now
                self.log(f"SPOOF — {result.message}")
                threading.Thread(
                    target=self._upload_spoof, args=(frame_bgr.copy(), result), daemon=True
                ).start()
            return

        if not result.ok or result.student_id is None:
            # Log không khớp mỗi 2s để người dùng biết hệ thống đang chạy
            if now - getattr(self, "_last_miss_log", 0) > 2:
                self._last_miss_log = now
                self.log(f"… {result.message}")
            return

        # Cooldown cùng một người
        if (
            result.student_id == self.last_match_uid
            and (now - self.last_match_time) < MATCH_COOLDOWN_SEC
        ):
            return
        self.last_match_uid = result.student_id
        self.last_match_time = now

        self.log(f"ĐIỂM DANH: {result.full_name} | sim={result.confidence:.3f} | {result.latency_ms:.0f} ms")
        try:
            self.status_label.configure(
                text=f"{result.full_name} — OK | {self.room}"
            )
        except Exception:
            pass
        threading.Thread(
            target=self._post_result, args=(result,), daemon=True
        ).start()

    def _post_result(self, result: EdgeResult):
        try:
            r = requests.post(
                f"{self._api_base()}/api/desktop/result",
                json={
                    "session_id": self.session_id,
                    "student_id": result.student_id,
                    "confidence": result.confidence,
                    "room": self.room,
                    "device_info": DEVICE_INFO,
                    "latency_ms": result.latency_ms,
                },
                timeout=3,
            )
            if r.status_code >= 400:
                self.log(f"Ghi kết quả lỗi: {r.text[:120]}")
        except Exception as e:
            self.log(f"Ghi kết quả lỗi: {e}")

    def _upload_spoof(self, frame_bgr, result: EdgeResult):
        try:
            if not HAS_CV2:
                return
            ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                return
            files = {"file": ("spoof.jpg", buf.tobytes(), "image/jpeg")}
            data = {
                "session_id": str(self.session_id or ""),
                "room": self.room or "",
                "device_info": DEVICE_INFO,
                "note": result.message,
            }
            requests.post(
                f"{self._api_base()}/api/desktop/spoof",
                data=data,
                files=files,
                timeout=5,
            )
        except Exception as e:
            self.log(f"Upload spoof lỗi: {e}")

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _on_close(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()


# typing helper
from typing import Optional

if __name__ == "__main__":
    if not HAS_CV2:
        print("Cảnh báo: chưa có opencv — pip install opencv-python")
    app = FaceAttendanceApp()
    app.run()
