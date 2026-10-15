# Kiến trúc Phân tán lai (Hybrid Distributed)

**Monolithic AI Edge** + **Centralized Web Server**

Thiết kế để đáp ứng NFR thời gian thực và CPU-only.

---

## 1. Sơ đồ logic

```
                    ┌──────────────────────────────────────┐
                    │     CENTRALIZED WEB SERVER           │
                    │  FastAPI + PostgreSQL                │
                    │  - Auth / RBAC                       │
                    │  - User, Class, Course, Session      │
                    │  - Face enrollment & admin duyệt     │
                    │  - Nhận kết quả điểm danh (metadata) │
                    │  - Lưu spoof evidence + báo cáo      │
                    └───────────────▲──────────────────────┘
                                    │
              Chỉ khi cần:          │  REST (JSON nhẹ)
              • GET gallery session │  • POST /api/desktop/result
              • POST spoof image    │  • POST /api/desktop/spoof
                                    │
┌───────────────────────────────────┴──────────────────────┐
│              MONOLITHIC AI EDGE (Desktop)                │
│  Camera → YuNet → Anti-Spoof → SFace → 1:N Match         │
│  • Toàn bộ AI local, CPU x86, không GPU                  │
│  • Video preview ≥25 FPS                                 │
│  • E2E nhận diện ≤150ms / lượt                           │
│  • Gallery embeddings cache trong RAM sau khi đồng bộ    │
└──────────────────────────────────────────────────────────┘
```

### Nguyên tắc phân tách

| Việc | Chạy ở đâu | Lý do |
|------|------------|--------|
| Bắt camera, hiển thị FPS | Edge | Tránh độ trễ mạng, đạt ≥25 FPS |
| Detect mặt, anti-spoof, match | Edge | E2E ≤150ms |
| Gallery embeddings | Edge (cache) sau khi GET từ server | Match offline trong phiên |
| Ghi điểm danh / spoof log | Central | Đồng bộ dữ liệu, audit |
| Quản lý user/lớp/phê duyệt face | Central | Web multi-role |

---

## 2. Ánh xạ Non-Functional Requirements

| NFR | Cơ chế đáp ứng |
|-----|----------------|
| **E2E ≤150ms** | AI 100% trên Edge; không upload frame mỗi lượt. Pipeline YuNet→Spoof→SFace tối ưu ONNX (2 thread CPU). |
| **Video ≥25 FPS** | Vòng capture/UI ~33ms; AI chạy subsample (`PROCESS_EVERY_N_FRAMES=2`), không block hiển thị. |
| **APCER <0.5%** | Anti-spoof **trên Edge trước khi match**. Ưu tiên chặn spoof (siết heuristic + slot MiniFASNet ONNX `models/antispoof_minifasnet.onnx`). Spoof → từ chối ngay + upload bằng chứng. |
| **BPCER <2.0%** | Cần model MiniFASNet đã train/validate; heuristic chỉ là fallback. |
| **Nhận diện ≥98.5%** | SFace + gallery đã duyệt admin; ngưỡng cosine 0.363; ảnh enroll chất lượng. |
| **Cold-start <3s** | `warm_up()` load ONNX + 1 inference giả khi mở app. |
| **CPU-only (i5-8+)** | `CPUExecutionProvider`, `intra_op_num_threads=2`. |
| **RAM idle ≤250MB** | Một process Edge, model load lazy, không giữ frame queue lớn. |
| **RAM peak ≤1.2GB** | Gallery float32 128-d × N SV nhỏ; không buffer video dài. |

---

## 3. Luồng thời gian thực (Edge)

```
Frame t
  │
  ├─► Hiển thị ngay (UI thread / vòng lặp chính)     ≥25 FPS
  │
  └─► mỗi N frame:
        YuNet (mặt lớn nhất)
           │ không mặt → bỏ qua
           ▼
        Anti-spoof
           │ spoof → EdgeResult(is_spoof) + POST /spoof (async)
           ▼
        SFace embedding 128-d
           ▼
        Cosine 1:N (gallery local)
           │ khớp + hết cooldown
           ▼
        POST /api/desktop/result (async, chỉ JSON)
```

Đo `latency_ms` **chỉ phần AI local** (không gồm mạng).

---

## 4. API đồng bộ Edge ↔ Central

| Method | Path | Hướng | Nội dung |
|--------|------|-------|----------|
| GET | `/api/desktop/room/{room}/session` | Central→Edge | session_id, class, list `{id, name, embedding[]}` |
| POST | `/api/desktop/result` | Edge→Central | session_id, student_id, confidence, latency_ms |
| POST | `/api/desktop/spoof` | Edge→Central | ảnh bằng chứng + room/session |
| POST | `/api/desktop/checkin` | (legacy) | server-side AI — **không dùng cho NFR 150ms** |

---

## 5. Thành phần mã nguồn

```
desktop/
  edge_ai.py           # Pipeline AI monolithic (YuNet + SFace + anti-spoof)
  attendance_app.py    # UI + camera + gọi edge_ai + sync REST nhẹ
backend/
  main.py              # Central API (thêm /desktop/result, /desktop/spoof)
  face_utils.py        # SFace phía server (enroll web)
models/
  face_recognition_sface_2021dec.onnx
  face_detection_yunet_2023mar.onnx
  antispoof_minifasnet.onnx   # (khuyến nghị bổ sung để đạt APCER/BPCER)
```

---

## 6. Ràng buộc & khuyến nghị đạt số liệu đầy đủ

1. **APCER/BPCER đúng spec** cần file `antispoof_minifasnet.onnx` (Silent-Face / MiniFASNet). Heuristic hiện tại chỉ giảm spoof thô, chưa guarantee <0.5% APCER.
2. Đo E2E trên máy thật: log `latency_ms` trong Edge; điều chỉnh `PROCESS_EVERY_N_FRAMES` nếu CPU yếu.
3. Enroll ảnh rõ, đủ sáng, admin duyệt — ảnh kém làm giảm accuracy dưới 98.5%.
4. Gallery lớn (>500 SV/lớp): có thể thêm FAISS/Index flat CPU nếu match >30ms.
5. Không chạy AI recognition trên Central cho luồng phòng học nếu muốn giữ ≤150ms.


