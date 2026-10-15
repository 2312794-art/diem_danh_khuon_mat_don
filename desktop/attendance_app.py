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
import tkinter as tk
from tkinter import ttk
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
        self.students_by_id = {}
        self.attendance_rows = {}
        self.unknown_rows = 0
        self._unknown_active_iid = None
        self._unknown_embedding = None
        self._unknown_frame = None
        self._unknown_candidate = None
        self._candidate_query_embedding = None
        self._lookup_in_progress = False
        self._sent_request_student_ids = set()
        self._request_in_progress_ids = set()
        self.last_match_uid = None
        self.last_match_time = 0.0
        self.frame_i = 0
        self.fps_times = deque(maxlen=30)
        self.last_result: Optional[EdgeResult] = None
        self._lock = threading.Lock()
        self._ui_thread_id = threading.get_ident()

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

        body = tk.Frame(self.root, bg="#F3F1FD")
        body.pack(fill="both", expand=True, padx=16, pady=(4, 16))

        top_row = tk.Frame(body, bg="#F3F1FD", height=350)
        top_row.pack(fill="both", expand=True)
        top_row.pack_propagate(False)
        top_row.grid_columnconfigure(0, weight=3, uniform="top_cards")
        top_row.grid_columnconfigure(1, weight=2, uniform="top_cards")
        top_row.grid_rowconfigure(0, weight=1)

        camera_card = tk.Frame(top_row, bg="#FFFFFF", highlightthickness=1,
                               highlightbackground="#E7E4F0")
        camera_card.grid(row=0, column=0, sticky="nsew", padx=(0, 7))
        tk.Label(camera_card, text="KHU VỰC CAMERA", fg="#5184B8", bg="#FFFFFF",
                 font=("Segoe UI", 10, "bold"), anchor="w").pack(fill="x", padx=14, pady=(12, 6))
        self.video_label = tk.Label(
            camera_card, text="Camera chưa được bật\n\nĐồng bộ buổi điểm danh để bắt đầu",
            fg="#AAA5BC", bg="#F7F8FC", font=("Segoe UI", 14),
        )
        self.video_label.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        info_card = tk.Frame(top_row, bg="#FFFFFF",
                             highlightthickness=1, highlightbackground="#E7E4F0")
        info_card.grid(row=0, column=1, sticky="nsew", padx=(7, 0))
        tk.Label(info_card, text="Thông tin sinh viên", fg="#5184B8", bg="#FFFFFF",
                 font=("Segoe UI", 11, "bold"), anchor="w").pack(fill="x", padx=16, pady=(14, 12))
        self.student_name_label = tk.Label(info_card, text="Chưa nhận diện", fg="#28364A",
                                           bg="#FFFFFF", font=("Segoe UI", 17, "bold"), anchor="w")
        self.student_name_label.pack(fill="x", padx=16, pady=(4, 8))
        self.student_code_label = tk.Label(info_card, text="Mã sinh viên: —", fg="#667085",
                                           bg="#FFFFFF", font=("Segoe UI", 11), anchor="w")
        self.student_code_label.pack(fill="x", padx=16, pady=5)
        self.student_class_label = tk.Label(info_card, text="Lớp: —", fg="#667085",
                                            bg="#FFFFFF", font=("Segoe UI", 11), anchor="w")
        self.student_class_label.pack(fill="x", padx=16, pady=5)
        self.student_status_label = tk.Label(info_card, text="Đưa khuôn mặt vào khung hình",
                                             fg="#667085", bg="#F7F8FC", font=("Segoe UI", 10),
                                             anchor="w", padx=10, pady=9)
        self.student_status_label.pack(fill="x", padx=16, pady=(14, 8))
        self.student_time_label = tk.Label(info_card, text="Thời gian: —", fg="#667085",
                                           bg="#FFFFFF", font=("Segoe UI", 10), anchor="w")
        self.student_time_label.pack(fill="x", padx=16, pady=4)
        self.lookup_button = tk.Button(
            info_card, text="Tìm sinh viên trong toàn hệ thống",
            command=self.lookup_unknown_student, state="disabled",
            fg="#FFFFFF", bg="#3978A8", activebackground="#2D648F",
            activeforeground="#FFFFFF", relief="flat", padx=10, pady=8,
            font=("Segoe UI", 9, "bold"),
        )
        self.lookup_button.pack(fill="x", padx=16, pady=(14, 6))
        self.approve_button = tk.Button(
            info_card, text="Gửi yêu cầu giảng viên thêm vào lớp",
            command=self.send_student_request, state="disabled",
            fg="#FFFFFF", bg="#15805D", activebackground="#116A4D",
            activeforeground="#FFFFFF", relief="flat", padx=10, pady=8,
            font=("Segoe UI", 9, "bold"),
        )
        self.approve_button.pack(fill="x", padx=16, pady=6)

        list_card = tk.Frame(body, bg="#FFFFFF", height=220,
                             highlightthickness=1, highlightbackground="#E7E4F0")
        list_card.pack(fill="x", pady=(12, 0))
        list_card.pack_propagate(False)
        list_header = tk.Frame(list_card, bg="#7565D9", height=38)
        list_header.pack(fill="x", padx=10, pady=(10, 7))
        list_header.pack_propagate(False)
        tk.Label(list_header, text="  Danh sách điểm danh", fg="#FFFFFF", bg="#7565D9",
                 font=("Segoe UI", 10, "bold"), anchor="w").pack(side="left", fill="y")
        self.attendance_count_label = tk.Label(list_header, text="0", fg="#FFFFFF", bg="#8D80E1",
                                               font=("Segoe UI", 9, "bold"), padx=8)
        self.attendance_count_label.pack(side="right", padx=8, pady=5)

        tabs = ttk.Notebook(list_card)
        tabs.pack(fill="both", expand=True, padx=10, pady=(0, 8))
        self.attendance_tabs = tabs
        attendance_tab = tk.Frame(tabs, bg="#FFFFFF")
        unknown_tab = tk.Frame(tabs, bg="#FFFFFF")
        tabs.add(attendance_tab, text="Đã điểm danh")
        self.unknown_tab = unknown_tab
        tabs.add(unknown_tab, text="Sinh viên chưa nhận diện (0)")

        table_frame = tk.Frame(attendance_tab, bg="#FFFFFF")
        table_frame.pack(fill="both", expand=True, padx=2, pady=2)
        style = ttk.Style()
        style.configure("Attendance.Treeview", rowheight=30, font=("Segoe UI", 9))
        style.configure("Attendance.Treeview.Heading", font=("Segoe UI", 9, "bold"))
        columns = ("index", "student_code", "name", "class", "time", "status")
        self.attendance_table = ttk.Treeview(table_frame, columns=columns, show="headings",
                                             style="Attendance.Treeview", height=5)
        headings = {"index": "STT", "student_code": "Mã SV", "name": "Họ tên",
                    "class": "Lớp", "time": "Thời gian", "status": "Kết quả"}
        widths = {"index": 48, "student_code": 135, "name": 190, "class": 100,
                  "time": 150, "status": 100}
        for col in columns:
            self.attendance_table.heading(col, text=headings[col])
            self.attendance_table.column(col, width=widths[col], anchor="center" if col in ("index", "time", "status") else "w")
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.attendance_table.yview)
        self.attendance_table.configure(yscrollcommand=scrollbar.set)
        self.attendance_table.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        tk.Label(unknown_tab,
                 text="Các lượt khuôn mặt chưa khớp sinh viên trong danh sách lớp. "
                      "Vui lòng kiểm tra danh tính trước khi thêm sinh viên vào lớp.",
                 fg="#667085", bg="#FFFFFF", font=("Segoe UI", 9), anchor="w",
                 justify="left", wraplength=980).pack(fill="x", padx=8, pady=(5, 7))
        unknown_frame = tk.Frame(unknown_tab, bg="#FFFFFF")
        unknown_frame.pack(fill="both", expand=True, padx=2, pady=(0, 2))
        unknown_columns = ("index", "time", "similarity", "message")
        self.unknown_table = ttk.Treeview(unknown_frame, columns=unknown_columns,
                                          show="headings", style="Attendance.Treeview", height=4)
        for col, title, width in (("index", "STT", 55), ("time", "Thời gian", 180),
                                  ("similarity", "Độ khớp cao nhất", 155),
                                  ("message", "Trạng thái", 400)):
            self.unknown_table.heading(col, text=title)
            self.unknown_table.column(col, width=width,
                                      anchor="center" if col in ("index", "time", "similarity") else "w")
        unknown_scroll = ttk.Scrollbar(unknown_frame, orient="vertical",
                                       command=self.unknown_table.yview)
        self.unknown_table.configure(yscrollcommand=unknown_scroll.set)
        self.unknown_table.pack(side="left", fill="both", expand=True)
        unknown_scroll.pack(side="right", fill="y")
        self.log_box = None

    def log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}\n"
        if threading.get_ident() != self._ui_thread_id:
            try:
                self.root.after(0, self._append_log, line)
            except Exception:
                pass
            print(line.strip())
            return
        self._append_log(line)

    def _append_log(self, line: str):
        print(line.strip())

    def _set_status(self, text: str):
        try:
            if threading.get_ident() == self._ui_thread_id:
                self.status_label.configure(text=text)
            else:
                self.root.after(0, lambda: self.status_label.configure(text=text))
        except Exception:
            pass

    def _show_student_result(self, result: EdgeResult):
        """Update the student card and keep one attendance row per student."""
        student = self.students_by_id.get(int(result.student_id), {})
        name = result.full_name or student.get("full_name") or "Sinh viên"
        code = student.get("student_id") or str(result.student_id)
        now = datetime.now().astimezone().strftime("%d/%m/%Y %H:%M:%S")
        self.student_name_label.configure(text=name)
        self.student_code_label.configure(text=f"Mã sinh viên: {code}")
        self.student_class_label.configure(text=f"Lớp: {self.class_name or '—'}")
        self.student_status_label.configure(
            text=f"✓  Đã điểm danh  •  Độ khớp {result.confidence:.3f}", fg="#12805C"
        )
        self.student_time_label.configure(text=f"Thời gian: {now}")

        sid = int(result.student_id)
        existing = self.attendance_rows.get(sid)
        values = (len(self.attendance_rows) + (0 if existing else 1), code, name,
                  self.class_name or "—", now, "Có mặt")
        if existing:
            self.attendance_table.item(existing, values=values)
            self.attendance_table.move(existing, "", 0)
        else:
            iid = self.attendance_table.insert("", 0, values=values)
            self.attendance_rows[sid] = iid
        for index, iid in enumerate(self.attendance_table.get_children(), start=1):
            row_values = list(self.attendance_table.item(iid, "values"))
            row_values[0] = index
            self.attendance_table.item(iid, values=row_values)
        self.attendance_count_label.configure(text=str(len(self.attendance_rows)))
        self._unknown_active_iid = None
        self._unknown_candidate = None
        self._candidate_query_embedding = None
        self._unknown_embedding = None
        self._unknown_frame = None
        self._lookup_in_progress = False
        self.lookup_button.configure(state="disabled")
        self.approve_button.configure(state="disabled")

    def _show_unrecognized(self, result: EdgeResult):
        if self._unknown_embedding is not None and not self._lookup_in_progress:
            self.lookup_button.configure(state="normal")
        if self._unknown_candidate and self._candidate_query_embedding is not None:
            current = self._unknown_embedding
            if current is not None:
                similarity = float(np.dot(current, self._candidate_query_embedding) /
                                   ((np.linalg.norm(current) + 1e-8) *
                                    (np.linalg.norm(self._candidate_query_embedding) + 1e-8)))
                if similarity >= 0.55:
                    return
            self._unknown_candidate = None
            self._candidate_query_embedding = None
            self.approve_button.configure(state="disabled")
        now = datetime.now().astimezone().strftime("%d/%m/%Y %H:%M:%S")
        sim_text = f"{result.confidence:.3f}" if result.confidence >= 0 else "—"
        self.student_name_label.configure(text="Sinh viên chưa được nhận diện")
        self.student_code_label.configure(text="Mã sinh viên: Chưa xác định")
        self.student_class_label.configure(text=f"Lớp đang điểm danh: {self.class_name or '—'}")
        self.student_status_label.configure(
            text="Khuôn mặt chưa khớp sinh viên trong lớp", fg="#B54708"
        )
        self.student_time_label.configure(text=f"Thời gian quét: {now}  •  Độ khớp: {sim_text}")

        values = (self.unknown_rows + (0 if self._unknown_active_iid else 1), now,
                  sim_text, result.message or "Chưa khớp danh sách lớp")
        if self._unknown_active_iid:
            self.unknown_table.item(self._unknown_active_iid, values=values)
            self.unknown_table.move(self._unknown_active_iid, "", 0)
        else:
            self.unknown_rows += 1
            self._unknown_active_iid = self.unknown_table.insert("", 0, values=values)
            while len(self.unknown_table.get_children()) > 50:
                self.unknown_table.delete(self.unknown_table.get_children()[-1])
        for index, iid in enumerate(self.unknown_table.get_children(), start=1):
            row_values = list(self.unknown_table.item(iid, "values"))
            row_values[0] = index
            self.unknown_table.item(iid, values=row_values)
        self.attendance_tabs.tab(self.unknown_tab,
                                 text=f"Sinh viên chưa nhận diện ({self.unknown_rows})")

    def _show_no_face(self):
        self._unknown_active_iid = None
        self._unknown_candidate = None
        self._candidate_query_embedding = None
        self._unknown_embedding = None
        self._unknown_frame = None
        self._lookup_in_progress = False
        self.lookup_button.configure(state="disabled")
        self.approve_button.configure(state="disabled")
        self.student_name_label.configure(text="Đang chờ khuôn mặt")
        self.student_code_label.configure(text="Mã sinh viên: —")
        self.student_class_label.configure(text=f"Lớp đang điểm danh: {self.class_name or '—'}")
        self.student_status_label.configure(text="Đưa khuôn mặt vào khung hình", fg="#667085")
        self.student_time_label.configure(text="Thời gian: —")

    def lookup_unknown_student(self):
        if not self.session_id or self._unknown_embedding is None:
            return
        embedding = self._unknown_embedding.copy()
        self._lookup_in_progress = True
        self.lookup_button.configure(state="disabled", text="Đang tra cứu...")
        threading.Thread(target=self._lookup_student_worker, args=(embedding,), daemon=True).start()

    def _lookup_student_worker(self, embedding):
        try:
            response = requests.post(
                f"{self._api_base()}/api/desktop/face-lookup",
                json={"session_id": self.session_id, "embedding": embedding.tolist()},
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            self.root.after(0, self._show_lookup_result, data, embedding)
        except Exception as exc:
            self.root.after(0, self._lookup_failed, str(exc))

    def _show_lookup_result(self, data, embedding):
        self._lookup_in_progress = False
        self.lookup_button.configure(text="Tìm sinh viên trong toàn hệ thống", state="normal")
        if self._unknown_embedding is None:
            self._lookup_failed("Không còn khuôn mặt cần tra cứu; hãy quét lại.")
            return
        current_similarity = float(np.dot(self._unknown_embedding, embedding) /
                                   ((np.linalg.norm(self._unknown_embedding) + 1e-8) *
                                    (np.linalg.norm(embedding) + 1e-8)))
        if current_similarity < 0.55:
            self._lookup_failed("Khuôn mặt đã thay đổi; quét lại rồi tra cứu lần nữa.")
            return
        if not data.get("found"):
            self._unknown_candidate = None
            self._candidate_query_embedding = None
            self.approve_button.configure(state="disabled")
            if data.get("reason") == "below_threshold":
                refreshed = int(data.get("embeddings_refreshed", 0))
                refresh_note = f" Đã cập nhật {refreshed} mẫu từ ảnh gốc." if refreshed else ""
                self.student_status_label.configure(
                    text=(f"Có ảnh đã duyệt nhưng độ khớp thấp ({data['confidence']:.3f}; "
                          f"cần ≥ {data['threshold']:.3f}). Hãy nhìn thẳng/gần camera rồi quét lại."
                          f"{refresh_note}"),
                    fg="#B54708")
            elif data.get("reason") == "no_approved_templates":
                self.student_status_label.configure(
                    text="Không có embedding ảnh đã duyệt dùng được; cần kiểm tra lại ảnh/embedding trên hệ thống.",
                    fg="#B42318")
            elif data.get("reason") == "no_candidates":
                self.student_status_label.configure(
                    text="Các hồ sơ có ảnh đã duyệt có thể đã thuộc lớp này; không có SV ngoài lớp để gửi yêu cầu.",
                    fg="#667085")
            elif data.get("ambiguous"):
                self.student_status_label.configure(
                    text=(f"Có nhiều hồ sơ gần giống (điểm cao nhất {data['confidence']:.3f}); "
                          "không thể tự chọn."), fg="#B54708")
            else:
                self.student_status_label.configure(
                    text="Không có hồ sơ phù hợp; bỏ qua, không thêm sinh viên", fg="#667085")
            return

        self._unknown_candidate = data
        self._candidate_query_embedding = embedding.copy()
        student = data["student"]
        self.student_name_label.configure(text=student.get("full_name", "Sinh viên"))
        self.student_code_label.configure(text=f"Mã sinh viên: {student.get('student_id') or '—'}")
        self.student_class_label.configure(text=f"Lớp hiện tại: {self.class_name or '—'}")
        if int(student["id"]) in self._sent_request_student_ids:
            self.student_status_label.configure(
                text="Đã gửi thông báo lên web giảng viên; đang chờ duyệt.", fg="#12805C")
            self.approve_button.configure(state="disabled")
        else:
            self.student_status_label.configure(
                text=f"Đã tìm thấy hồ sơ • Độ khớp {data['confidence']:.3f}.", fg="#12805C")
            self.student_time_label.configure(text="Đang tự gửi yêu cầu lên web giảng viên...")
            self.approve_button.configure(state="disabled", text="Đang gửi yêu cầu...")
            self._queue_student_request(data)

    def _lookup_failed(self, message):
        self._lookup_in_progress = False
        state = "normal" if self._unknown_embedding is not None else "disabled"
        self.lookup_button.configure(text="Tìm sinh viên trong toàn hệ thống", state=state)
        self.student_status_label.configure(text=message[:100], fg="#B42318")

    def send_student_request(self):
        candidate = self._unknown_candidate
        if not candidate or not self.session_id:
            return
        self._queue_student_request(candidate)

    def _queue_student_request(self, candidate):
        student_id = int(candidate["student"]["id"])
        if student_id in self._sent_request_student_ids or student_id in self._request_in_progress_ids:
            return
        self._request_in_progress_ids.add(student_id)
        self.approve_button.configure(state="disabled", text="Đang gửi yêu cầu...")
        threading.Thread(target=self._send_student_request_worker,
                         args=(candidate,), daemon=True).start()

    def _send_student_request_worker(self, candidate):
        try:
            student = candidate["student"]
            response = requests.post(
                f"{self._api_base()}/api/desktop/session/{self.session_id}/enrollment-requests",
                json={"student_id": student["id"], "confidence": candidate["confidence"]},
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()
            self.root.after(0, self._student_request_sent, data, candidate)
        except Exception as exc:
            self.root.after(0, self._student_request_failed, str(exc))

    def _student_request_sent(self, data, candidate):
        student = candidate["student"]
        student_id = int(student["id"])
        self._request_in_progress_ids.discard(student_id)
        self._sent_request_student_ids.add(student_id)
        self.student_status_label.configure(
            text="Đã gửi thông báo đến giảng viên. Chờ duyệt trên web.", fg="#12805C")
        self.student_time_label.configure(text=f"Yêu cầu #{data.get('request_id', '—')} • chưa điểm danh")
        self.approve_button.configure(text="Đã gửi yêu cầu", state="disabled")
        self.log(f"Đã gửi yêu cầu thêm {student['full_name']} vào lớp {self.class_name}; chờ duyệt web")

    def _student_request_failed(self, message):
        if self._unknown_candidate:
            self._request_in_progress_ids.discard(int(self._unknown_candidate["student"]["id"]))
        self.approve_button.configure(text="Gửi yêu cầu giảng viên thêm vào lớp", state="normal")
        self.student_status_label.configure(text=message[:100], fg="#B42318")

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
            self.students_by_id = {int(s["id"]): s for s in data.get("students", [])}
            self.attendance_rows.clear()
            for row_id in self.attendance_table.get_children():
                self.attendance_table.delete(row_id)
            self.unknown_rows = 0
            self._unknown_active_iid = None
            self._unknown_candidate = None
            self._candidate_query_embedding = None
            self._unknown_embedding = None
            self._unknown_frame = None
            self._lookup_in_progress = False
            self._sent_request_student_ids.clear()
            self._request_in_progress_ids.clear()
            self.lookup_button.configure(state="disabled", text="Tìm sinh viên trong toàn hệ thống")
            self.approve_button.configure(state="disabled", text="Gửi yêu cầu giảng viên thêm vào lớp")
            for row_id in self.unknown_table.get_children():
                self.unknown_table.delete(row_id)
            self.attendance_tabs.tab(self.unknown_tab, text="Sinh viên chưa nhận diện (0)")
            self.attendance_count_label.configure(text="0")
            self.student_name_label.configure(text="Chưa nhận diện")
            self.student_code_label.configure(text="Mã sinh viên: —")
            self.student_class_label.configure(text=f"Lớp: {self.class_name or '—'}")
            self.student_status_label.configure(text="Đưa khuôn mặt vào khung hình", fg="#667085")
            self.student_time_label.configure(text="Thời gian: —")
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
        # Camera drivers on Windows do not all work with the default backend.
        # Try common backends and device indices, then verify an actual frame
        # can be read before reporting that the camera is ready.
        candidates = []
        if sys.platform == "win32":
            candidates.extend((i, backend) for i in range(3)
                              for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY))
        else:
            candidates.extend((i, cv2.CAP_ANY) for i in range(3))
        self.cap = None
        for index, backend in candidates:
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                # Some drivers need a few reads while the stream initializes.
                for _ in range(8):
                    ret, first_frame = cap.read()
                    if ret and first_frame is not None and first_frame.size:
                        self.cap = cap
                        self.log(f"Mở camera {index} (backend {backend})")
                        break
                    time.sleep(0.08)
            if self.cap is not None:
                break
            cap.release()
        if self.cap is None:
            self.log("Không mở được camera. Kiểm tra quyền truy cập, ứng dụng khác đang dùng camera và thiết bị camera.")
            return
        self.running = True
        self.start_btn.configure(text="Dừng camera")
        self.log("Camera ON — AI chạy local (không gửi frame lên server)")
        threading.Thread(target=self._capture_loop, daemon=True).start()

    def _capture_loop(self):
        """Luồng hiển thị ≥25 FPS; AI xử lý subsample frames."""
        from PIL import Image, ImageTk
        cap = self.cap

        while self.running and cap is not None:
            t_frame = time.perf_counter()
            ret, frame = cap.read()
            if not ret or frame is None:
                self.log("Mất tín hiệu camera")
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
                label_text = f"FPS: {fps:.1f} | Latency: {lat:.0f}ms"
                try:
                    self.root.after(0, lambda text=label_text: self.fps_label.configure(text=text))
                except Exception:
                    pass

            rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            try:
                self.root.after(0, self._show_frame, img, ImageTk)
            except Exception:
                pass

            elapsed = time.perf_counter() - t_frame
            time.sleep(max(0.0, 0.033 - elapsed))

        # Leave the UI and capture state consistent if the device is unplugged
        # or its driver stops returning frames.
        self.running = False
        if self.cap is cap:
            self.cap = None
        if cap is not None:
            cap.release()
        try:
            self.root.after(0, lambda: self.start_btn.configure(text="Bắt đầu camera"))
        except Exception:
            pass

    def _show_frame(self, img, ImageTk):
        try:
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk, text="")
        except Exception:
            pass

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
            if result.embedding is not None:
                self._unknown_embedding = result.embedding.copy()
                self._unknown_frame = frame_bgr.copy()
                if self._candidate_query_embedding is not None:
                    sim = float(np.dot(self._unknown_embedding, self._candidate_query_embedding) /
                                ((np.linalg.norm(self._unknown_embedding) + 1e-8) *
                                 (np.linalg.norm(self._candidate_query_embedding) + 1e-8)))
                    if sim < 0.55:
                        self._unknown_candidate = None
                        self._candidate_query_embedding = None
                        self.root.after(0, lambda: self.approve_button.configure(state="disabled"))
            # Log không khớp mỗi 2s để người dùng biết hệ thống đang chạy
            if now - getattr(self, "_last_miss_log", 0) > 2:
                self._last_miss_log = now
                self.log(f"… {result.message}")
                if "không có khuôn mặt" in (result.message or "").lower():
                    self.root.after(0, self._show_no_face)
                else:
                    self.root.after(0, self._show_unrecognized, result)
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
            self._set_status(f"{result.full_name} — OK | {self.room}")
            self.root.after(0, self._show_student_result, result)
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
