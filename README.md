# Hệ thống Điểm danh Gương mặt (Face Attendance System)

## Công nghệ
- **Backend**: Python 3.12+, FastAPI, SQLAlchemy
- **Database**: PostgreSQL 16
- **Nhận diện**: InsightFace (buffalo_l – tương đương / vượt SFace)
- **Chống giả mạo**: MiniFASNetV2 ONNX; fail closed nếu thiếu model
- **Web UI**: HTML + Tailwind + Vanilla JS
- **Desktop**: CustomTkinter / Tkinter + OpenCV

## Cài đặt

```bash
# 1. PostgreSQL (đã tạo user/db)
# User: faceattend / Password: faceattend123 / DB: faceattend

# 2. Python deps
cd face_attendance
pip install -r requirements.txt

# 3. Chạy API
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 4. Mở Web
# Mở file web/index.html trong trình duyệt
# hoặc serve: python -m http.server 5500 --directory web

# 5. Desktop app
python desktop/attendance_app.py
```

## Tài khoản mặc định
- **Admin**: `admin` / `admin123`

## Luồng sử dụng

### Admin
1. Đăng nhập → Quản lý người dùng (thêm GV, SV)
2. Tạo học phần, lớp học (gán phòng, ví dụ A101)
3. Phê duyệt dữ liệu khuôn mặt SV gửi lên
4. Xem log ảnh giả mạo

### Giảng viên
1. Đăng nhập → Xem lớp của mình
2. Thêm sinh viên vào lớp
3. Mở buổi điểm danh (chọn class_id + phòng)
4. Mở mục **Yêu cầu thêm SV** trên web để duyệt sinh viên đã có hồ sơ nhưng chưa thuộc lớp
5. Đóng buổi → xem thống kê

### Sinh viên
1. Đăng nhập → Upload ảnh khuôn mặt (chờ admin duyệt)
2. Xem lịch sử điểm danh

### Desktop (phòng học)
1. Nhập mã phòng đã tạo trên web (vd: A101)
2. Bấm **Kết nối buổi học** (phải có session đang mở)
3. Bật camera → hệ thống tự nhận diện + chống spoof
4. Nếu tìm thấy hồ sơ SV ngoài lớp, desktop tự gửi yêu cầu lên web giảng viên (không cần đăng nhập giảng viên ở desktop); chỉ ghi danh/điểm danh sau khi giảng viên duyệt
5. Khi phát hiện ảnh giả mạo → log ảnh về server

## API chính
- `POST /api/auth/login`
- `CRUD /api/users`, `/api/courses`, `/api/classes`
- `POST /api/face/upload` (SV)
- `POST /api/face/{id}/review` (Admin)
- `POST /api/sessions` (mở buổi điểm danh)
- `GET  /api/desktop/room/{room}/session`
- `POST /api/desktop/session/{id}/enrollment-requests`
- `GET  /api/lecturer/enrollment-requests` / `POST /api/lecturer/enrollment-requests/{id}/review`
- `POST /api/desktop/checkin` (desktop gửi frame)

## Ghi chú chống giả mạo
Desktop và backend dùng MiniFASNetV2 ONNX ở `models/antispoof_minifasnet.onnx`.
Ứng dụng crop vùng mặt mở rộng theo đúng tiền xử lý của model, rồi phân lớp thật/giả.
Nếu model thiếu hoặc inference lỗi, điểm danh và upload ảnh khuôn mặt bị từ chối an toàn;
không dùng heuristic độ nét/màu vì ảnh trên màn hình điện thoại có thể vượt qua heuristic đó.

Model tham khảo từ [yakhyo/face-anti-spoofing](https://github.com/yakhyo/face-anti-spoofing),
MiniFASNetV2 ONNX; mã nguồn phát hành theo Apache-2.0. Xem `models/ANTI_SPOOFING_MODEL.md`.

## Cấu trúc thư mục
```
face_attendance/
├── backend/          # FastAPI
│   ├── main.py
│   ├── database.py
│   ├── auth.py
│   ├── schemas.py
│   └── face_utils.py
├── web/index.html    # Web UI
├── desktop/attendance_app.py
├── uploads/          # Ảnh face + spoof logs
├── models/
└── requirements.txt
```
