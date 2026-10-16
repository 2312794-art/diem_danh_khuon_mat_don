from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timezone
import os
import shutil
from pathlib import Path

from database import (
    get_db, init_db, User, UserRole, Course, Class, Enrollment,
    FaceData, FaceDataStatus, AttendanceSession, AttendanceRecord, EnrollmentRequest,
    AttendanceStatus, SpoofLog
)
import schemas, auth
from face_utils import (
    extract_embedding_from_file, extract_embedding, embedding_to_bytes,
    bytes_to_embedding, match_face, check_anti_spoof, save_image, UPLOADS_DIR,
    load_image_bgr, load_image_bgr_from_bytes, anti_spoof_model_available,
)
import numpy as np

# Keep cross-system lookup aligned with the desktop SFace matcher. A lecturer
# still reviews every request before the student is enrolled.
DESKTOP_LOOKUP_THRESHOLD = 0.28
DESKTOP_LOOKUP_MARGIN = 0.06

app = FastAPI(
    title="Face Attendance System API",
    description="Hệ thống điểm danh gương mặt - Web API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploads
uploads_path = Path(__file__).resolve().parent.parent / "uploads"
uploads_path.mkdir(exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(uploads_path)), name="uploads")


@app.on_event("startup")
def on_startup():
    init_db()
    # Seed admin if not exists
    db = next(get_db())
    try:
        admin = db.query(User).filter(User.username == "admin").first()
        if not admin:
            admin = User(
                username="admin",
                email="admin@faceattend.local",
                hashed_password=auth.get_password_hash("admin123"),
                full_name="Quản trị viên",
                role=UserRole.ADMIN,
            )
            db.add(admin)
            db.commit()
            print("Seeded default admin: admin / admin123")
    finally:
        db.close()


# ==================== AUTH ====================
@app.post("/api/auth/login", response_model=schemas.Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = auth.authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Sai tên đăng nhập hoặc mật khẩu")
    token = auth.create_access_token(data={"sub": user.username, "role": user.role.value})
    return schemas.Token(
        access_token=token,
        role=user.role.value,
        full_name=user.full_name,
        user_id=user.id,
    )


@app.get("/api/auth/me", response_model=schemas.UserOut)
def me(current_user: User = Depends(auth.get_current_user)):
    return current_user


# ==================== USERS (Admin) ====================
@app.get("/api/users", response_model=List[schemas.UserOut])
def list_users(
    role: Optional[UserRole] = None,
    include_inactive: bool = False,
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.require_role(UserRole.ADMIN, UserRole.LECTURER)),
):
    # Giảng viên chỉ được xem sinh viên đang hoạt động
    if current_user.role == UserRole.LECTURER:
        role = UserRole.STUDENT
        include_inactive = False
    q = db.query(User)
    if role:
        q = q.filter(User.role == role)
    if not include_inactive:
        q = q.filter(User.is_active == True)
    return q.order_by(User.id).all()


@app.post("/api/users", response_model=schemas.UserOut, status_code=201)
def create_user(
    data: schemas.UserCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.require_role(UserRole.ADMIN)),
):
    if db.query(User).filter(User.username == data.username).first():
        raise HTTPException(400, "Username đã tồn tại")
    if db.query(User).filter(User.email == data.email).first():
        raise HTTPException(400, "Email đã tồn tại")
    user = User(
        username=data.username,
        email=data.email,
        hashed_password=auth.get_password_hash(data.password),
        full_name=data.full_name,
        role=data.role,
        student_id=data.student_id,
        phone=data.phone,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@app.put("/api/users/{user_id}", response_model=schemas.UserOut)
def update_user(
    user_id: int,
    data: schemas.UserUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.require_role(UserRole.ADMIN)),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(404, "Không tìm thấy người dùng")
    if user.username == "admin" and user.role == UserRole.ADMIN:
        # Không cho hạ quyền / vô hiệu admin gốc
        payload = data.model_dump(exclude_unset=True)
        if payload.get("is_active") is False:
            raise HTTPException(400, "Không thể vô hiệu hóa admin mặc định")
        if "role" in payload and payload["role"] not in (None, UserRole.ADMIN, "admin"):
            raise HTTPException(400, "Không thể đổi role admin mặc định")
    for field, value in data.model_dump(exclude_unset=True).items():
        if field == "password":
            if value:
                user.hashed_password = auth.get_password_hash(value)
        elif field == "email" and value and value != user.email:
            if db.query(User).filter(User.email == value, User.id != user_id).first():
                raise HTTPException(400, "Email đã tồn tại")
            user.email = value
        else:
            setattr(user, field, value)
    db.commit()
    db.refresh(user)
    return user


@app.delete("/api/users/{user_id}")
def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.require_role(UserRole.ADMIN)),
):
    """Xóa mềm: chỉ đặt is_active=False, không xóa khỏi DB."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(404, "Không tìm thấy người dùng")
    if user.role == UserRole.ADMIN and user.username == "admin":
        raise HTTPException(400, "Không thể vô hiệu hóa admin mặc định")
    if user.id == current_user.id:
        raise HTTPException(400, "Không thể vô hiệu hóa chính tài khoản đang đăng nhập")
    user.is_active = False
    db.commit()
    return {"message": "Đã vô hiệu hóa tài khoản (dữ liệu vẫn lưu trong hệ thống)"}


@app.post("/api/users/{user_id}/restore")
def restore_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.require_role(UserRole.ADMIN)),
):
    """Khôi phục tài khoản đã vô hiệu hóa."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(404, "Không tìm thấy người dùng")
    user.is_active = True
    db.commit()
    db.refresh(user)
    return {"message": "Đã kích hoạt lại tài khoản", "user": schemas.UserOut.model_validate(user)}


# ==================== COURSES ====================
@app.get("/api/courses", response_model=List[schemas.CourseOut])
def list_courses(db: Session = Depends(get_db), current_user: User = Depends(auth.get_current_user)):
    return db.query(Course).order_by(Course.code).all()


@app.post("/api/courses", response_model=schemas.CourseOut, status_code=201)
def create_course(
    data: schemas.CourseCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.require_role(UserRole.ADMIN)),
):
    if db.query(Course).filter(Course.code == data.code).first():
        raise HTTPException(400, "Mã học phần đã tồn tại")
    course = Course(**data.model_dump())
    db.add(course)
    db.commit()
    db.refresh(course)
    return course


@app.put("/api/courses/{course_id}", response_model=schemas.CourseOut)
def update_course(
    course_id: int,
    data: schemas.CourseUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.require_role(UserRole.ADMIN)),
):
    course = db.query(Course).filter(Course.id == course_id).first()
    if not course:
        raise HTTPException(404, "Không tìm thấy học phần")
    for k, v in data.model_dump(exclude_unset=True).items():
        setattr(course, k, v)
    db.commit()
    db.refresh(course)
    return course


@app.delete("/api/courses/{course_id}")
def delete_course(
    course_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.require_role(UserRole.ADMIN)),
):
    course = db.query(Course).filter(Course.id == course_id).first()
    if not course:
        raise HTTPException(404)
    db.delete(course)
    db.commit()
    return {"message": "Đã xóa"}


# ==================== CLASSES ====================
@app.get("/api/classes", response_model=List[schemas.ClassOut])
def list_classes(
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.get_current_user),
):
    q = db.query(Class)
    if current_user.role == UserRole.LECTURER:
        q = q.filter(Class.lecturer_id == current_user.id)
    return q.order_by(Class.id).all()


@app.post("/api/classes", response_model=schemas.ClassOut, status_code=201)
def create_class(
    data: schemas.ClassCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.require_role(UserRole.ADMIN, UserRole.LECTURER)),
):
    if not db.query(Course).filter(Course.id == data.course_id).first():
        raise HTTPException(400, "Học phần không tồn tại")
    cls = Class(**data.model_dump())
    if current_user.role == UserRole.LECTURER:
        cls.lecturer_id = current_user.id
    db.add(cls)
    db.commit()
    db.refresh(cls)
    return cls


@app.put("/api/classes/{class_id}", response_model=schemas.ClassOut)
def update_class(
    class_id: int,
    data: schemas.ClassUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.require_role(UserRole.ADMIN, UserRole.LECTURER)),
):
    cls = db.query(Class).filter(Class.id == class_id).first()
    if not cls:
        raise HTTPException(404)
    if current_user.role == UserRole.LECTURER and cls.lecturer_id != current_user.id:
        raise HTTPException(403)
    for k, v in data.model_dump(exclude_unset=True).items():
        setattr(cls, k, v)
    db.commit()
    db.refresh(cls)
    return cls


@app.delete("/api/classes/{class_id}")
def delete_class(
    class_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.require_role(UserRole.ADMIN)),
):
    cls = db.query(Class).filter(Class.id == class_id).first()
    if not cls:
        raise HTTPException(404)
    db.delete(cls)
    db.commit()
    return {"message": "Đã xóa"}


# ==================== ENROLLMENTS ====================
@app.get("/api/classes/{class_id}/students", response_model=List[schemas.EnrollmentOut])
def list_students_in_class(
    class_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.get_current_user),
):
    return (
        db.query(Enrollment)
        .filter(Enrollment.class_id == class_id)
        .all()
    )


@app.post("/api/enrollments", response_model=schemas.EnrollmentOut, status_code=201)
def enroll_student(
    data: schemas.EnrollmentCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.require_role(UserRole.ADMIN, UserRole.LECTURER)),
):
    student = db.query(User).filter(User.id == data.student_id, User.role == UserRole.STUDENT).first()
    if not student:
        raise HTTPException(400, "Sinh viên không tồn tại")
    cls = db.query(Class).filter(Class.id == data.class_id).first()
    if not cls:
        raise HTTPException(400, "Lớp không tồn tại")
    if current_user.role == UserRole.LECTURER and cls.lecturer_id != current_user.id:
        raise HTTPException(403)
    exists = db.query(Enrollment).filter(
        Enrollment.student_id == data.student_id,
        Enrollment.class_id == data.class_id,
    ).first()
    if exists:
        raise HTTPException(400, "Sinh viên đã trong lớp")
    enr = Enrollment(student_id=data.student_id, class_id=data.class_id)
    db.add(enr)
    db.commit()
    db.refresh(enr)
    return enr


@app.delete("/api/enrollments/{enrollment_id}")
def unenroll(
    enrollment_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.require_role(UserRole.ADMIN, UserRole.LECTURER)),
):
    enr = db.query(Enrollment).filter(Enrollment.id == enrollment_id).first()
    if not enr:
        raise HTTPException(404)
    db.delete(enr)
    db.commit()
    return {"message": "Đã xóa khỏi lớp"}


# ==================== FACE DATA ====================
@app.post("/api/face/upload")
async def upload_face(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.get_current_user),
):
    """Student uploads face image for recognition."""
    if current_user.role != UserRole.STUDENT:
        raise HTTPException(403, "Chỉ sinh viên được upload dữ liệu khuôn mặt")
    if not anti_spoof_model_available():
        raise HTTPException(503, "Thiếu model anti-spoof MiniFASNet; chưa thể kiểm tra ảnh đăng ký")
    # Save temp
    ext = Path(file.filename).suffix or ".jpg"
    tmp_path = UPLOADS_DIR / "temp" / f"{current_user.id}_{datetime.now().strftime('%Y%m%d%H%M%S')}{ext}"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = extract_embedding_from_file(str(tmp_path))
    if result is None:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(400, "Không phát hiện khuôn mặt trong ảnh. Vui lòng chụp ảnh rõ mặt.")

    emb, info = result
    img_bgr = load_image_bgr(str(tmp_path))
    is_real, spoof_conf = check_anti_spoof(img_bgr, info.get("bbox"))
    if not is_real:
        # Log spoof
        spoof_path = save_image(img_bgr, "spoof", f"upload_{current_user.id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg")
        log = SpoofLog(image_path=spoof_path, note=f"Upload spoof from user {current_user.id}", device_info="web-upload")
        db.add(log)
        db.commit()
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(400, "Phát hiện ảnh giả mạo / không phải khuôn mặt thật. Đã ghi log.")

    # Lưu mỗi ảnh là 1 bản ghi riêng (SV có thể có nhiều ảnh)
    ts = datetime.now().strftime("%Y%m%d%H%M%S%f")
    final_rel = save_image(img_bgr, "faces", f"user_{current_user.id}_{ts}.jpg")
    tmp_path.unlink(missing_ok=True)

    face = FaceData(
        user_id=current_user.id,
        embedding=embedding_to_bytes(emb),
        image_path=final_rel,
        status=FaceDataStatus.PENDING,
    )
    db.add(face)
    db.commit()
    db.refresh(face)
    total = db.query(FaceData).filter(FaceData.user_id == current_user.id).count()
    return {
        "message": f"Đã gửi ảnh khuôn mặt (#{face.id}), chờ admin phê duyệt. Tổng ảnh: {total}",
        "status": "pending",
        "face_id": face.id,
        "total_images": total,
    }


@app.get("/api/face/pending", response_model=List[schemas.FaceDataOut])
def list_pending_faces(
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.require_role(UserRole.ADMIN)),
):
    from sqlalchemy.orm import joinedload
    return (
        db.query(FaceData)
        .options(joinedload(FaceData.user))
        .filter(FaceData.status == FaceDataStatus.PENDING)
        .order_by(FaceData.submitted_at.desc())
        .all()
    )


@app.post("/api/face/{face_id}/review")
def review_face(
    face_id: int,
    data: schemas.FaceReview,
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.require_role(UserRole.ADMIN)),
):
    # 1. Tìm bản ghi ảnh khuôn mặt
    face = db.query(FaceData).filter(FaceData.id == face_id).first()
    if not face:
        raise HTTPException(404, "Không tìm thấy dữ liệu khuôn mặt")
    if data.status not in (FaceDataStatus.APPROVED, FaceDataStatus.REJECTED):
        raise HTTPException(400, "Trạng thái phải là approved hoặc rejected")

    # 2. Xử lý TỰ ĐỘNG TRAIN / TRÍCH XUẤT EMBEDDING khi Duyệt (APPROVED)
    if data.status == FaceDataStatus.APPROVED:
        # Nếu ảnh chưa có embedding, tự động chạy SFace trích xuất lại ngay lập tức
        if not face.embedding:
            # Lấy đường dẫn ảnh tuyệt đối
            abs_image_path = Path(__file__).resolve().parent.parent / face.image_path
            
            if not abs_image_path.exists():
                raise HTTPException(400, f"Không tìm thấy file ảnh tại: {face.image_path}")

            # Trích xuất đặc trưng (Training/Embedding)
            result = extract_embedding_from_file(str(abs_image_path))
            if result is None:
                raise HTTPException(
                    400, "Không thể trích xuất đặc trưng khuôn mặt từ ảnh này. Vui lòng từ chối ảnh!"
                )
            
            emb, _ = result
            # Lưu Vector Embedding đã train vào CSDL dưới dạng Bytes
            face.embedding = embedding_to_bytes(emb)

    # 3. Cập nhật trạng thái và thông tin duyệt
    face.status = data.status
    face.note = data.note
    face.reviewed_at = datetime.now(timezone.utc)
    face.reviewed_by = current_user.id

    db.commit()
    db.refresh(face)

    return {
        "message": f"Đã {data.status.value} và cập nhật dữ liệu huấn luyện thành công!",
        "face_id": face.id,
        "status": face.status.value,
        "has_embedding": face.embedding is not None
    }

@app.get("/api/face/my")
def my_face(
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.get_current_user),
):
    faces = (
        db.query(FaceData)
        .filter(FaceData.user_id == current_user.id)
        .order_by(FaceData.submitted_at.desc())
        .all()
    )
    if not faces:
        return {"status": "none", "images": [], "total": 0}
    # Ưu tiên trạng thái: approved > pending > rejected
    statuses = [f.status.value if hasattr(f.status, "value") else str(f.status) for f in faces]
    if "approved" in statuses:
        overall = "approved"
    elif "pending" in statuses:
        overall = "pending"
    else:
        overall = statuses[0]
    return {
        "status": overall,
        "total": len(faces),
        "images": [
            {
                "id": f.id,
                "status": f.status.value if hasattr(f.status, "value") else str(f.status),
                "image_path": f.image_path,
                "submitted_at": f.submitted_at,
                "note": f.note,
            }
            for f in faces
        ],
    }


# ==================== ATTENDANCE SESSIONS ====================
@app.post("/api/sessions", response_model=schemas.SessionOut, status_code=201)
def create_session(
    data: schemas.SessionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.require_role(UserRole.ADMIN, UserRole.LECTURER)),
):
    cls = db.query(Class).filter(Class.id == data.class_id).first()
    if not cls:
        raise HTTPException(404, "Lớp không tồn tại")
    if current_user.role == UserRole.LECTURER and cls.lecturer_id != current_user.id:
        raise HTTPException(403)
    session = AttendanceSession(
        class_id=data.class_id,
        session_date=data.session_date,
        room=data.room or cls.room,
        notes=data.notes,
        created_by=current_user.id,
        is_open=True,
        start_time=datetime.now(timezone.utc),
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@app.get("/api/sessions", response_model=List[schemas.SessionOut])
def list_sessions(
    class_id: Optional[int] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.get_current_user),
):
    q = db.query(AttendanceSession)
    if class_id:
        q = q.filter(AttendanceSession.class_id == class_id)
    if current_user.role == UserRole.LECTURER:
        class_ids = [c.id for c in db.query(Class).filter(Class.lecturer_id == current_user.id).all()]
        q = q.filter(AttendanceSession.class_id.in_(class_ids))
    return q.order_by(AttendanceSession.session_date.desc()).all()


@app.post("/api/sessions/{session_id}/close")
def close_session(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.require_role(UserRole.ADMIN, UserRole.LECTURER)),
):
    session = db.query(AttendanceSession).filter(AttendanceSession.id == session_id).first()
    if not session:
        raise HTTPException(404)
    session.is_open = False
    session.end_time = datetime.now(timezone.utc)
    # Mark missing students as absent
    enrolled = db.query(Enrollment).filter(Enrollment.class_id == session.class_id).all()
    for enr in enrolled:
        exists = db.query(AttendanceRecord).filter(
            AttendanceRecord.session_id == session_id,
            AttendanceRecord.student_id == enr.student_id,
        ).first()
        if not exists:
            rec = AttendanceRecord(
                session_id=session_id,
                student_id=enr.student_id,
                status=AttendanceStatus.ABSENT,
            )
            db.add(rec)
    db.commit()
    return {"message": "Đã đóng buổi điểm danh"}


@app.get("/api/sessions/{session_id}/records", response_model=List[schemas.AttendanceRecordOut])
def session_records(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.get_current_user),
):
    return db.query(AttendanceRecord).filter(AttendanceRecord.session_id == session_id).all()


@app.post("/api/sessions/{session_id}/manual")
def manual_attendance(
    session_id: int,
    data: schemas.ManualAttendance,
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.require_role(UserRole.ADMIN, UserRole.LECTURER)),
):
    session = db.query(AttendanceSession).filter(AttendanceSession.id == session_id).first()
    if not session or not session.is_open:
        raise HTTPException(400, "Buổi điểm danh không mở")
    rec = db.query(AttendanceRecord).filter(
        AttendanceRecord.session_id == session_id,
        AttendanceRecord.student_id == data.student_id,
    ).first()
    if rec:
        rec.status = data.status
        rec.is_manual = True
        rec.check_in_time = datetime.now(timezone.utc)
    else:
        rec = AttendanceRecord(
            session_id=session_id,
            student_id=data.student_id,
            status=data.status,
            is_manual=True,
            check_in_time=datetime.now(timezone.utc),
        )
        db.add(rec)
    db.commit()
    return {"message": "OK"}


# ==================== STUDENT ATTENDANCE HISTORY ====================
@app.get("/api/my/attendance")
def my_attendance(
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.get_current_user),
):
    if current_user.role != UserRole.STUDENT:
        raise HTTPException(403)
    records = (
        db.query(AttendanceRecord)
        .filter(AttendanceRecord.student_id == current_user.id)
        .order_by(AttendanceRecord.created_at.desc())
        .all()
    )
    result = []
    for r in records:
        sess = r.session
        cls = sess.class_ if sess else None
        result.append({
            "session_id": r.session_id,
            "status": r.status.value,
            "check_in_time": r.check_in_time,
            "class_name": cls.name if cls else None,
            "session_date": sess.session_date if sess else None,
            "room": sess.room if sess else None,
        })
    return result


# ==================== DESKTOP API: Face check-in ====================
@app.get("/api/desktop/room/{room}/session")
def get_open_session_by_room(room: str, db: Session = Depends(get_db)):
    """Desktop app queries open session for a room."""
    session = (
        db.query(AttendanceSession)
        .filter(AttendanceSession.room == room, AttendanceSession.is_open == True)
        .order_by(AttendanceSession.created_at.desc())
        .first()
    )
    if not session:
        return {"found": False}
    cls = db.query(Class).filter(Class.id == session.class_id).first()
    if not cls:
        return {"found": False}
    students = []
    enrollments = db.query(Enrollment).filter(Enrollment.class_id == cls.id).all()
    approved_face_count = 0
    for enr in enrollments:
        student = db.query(User).filter(User.id == enr.student_id).first()
        if not student:
            continue
        # Lấy TẤT CẢ ảnh đã duyệt của SV (nhiều embedding)
        faces = (
            db.query(FaceData)
            .filter(
                FaceData.user_id == student.id,
                FaceData.status == FaceDataStatus.APPROVED,
            )
            .all()
        )
        approved_face_count += len(faces)
        embeddings = []
        embedding_face_ids = []
        rebuilt_count = 0
        stored_count = 0
        for face in faces:
            try:
                # Rebuild legacy embeddings from source photos. Older records
                # may have been generated with different crop/channel handling.
                image_path = Path(__file__).resolve().parent.parent / face.image_path
                extracted = extract_embedding_from_file(str(image_path)) if image_path.is_file() else None
                if extracted is not None:
                    emb = [float(x) for x in extracted[0]]
                    rebuilt_count += 1
                elif face.embedding:
                    emb = [float(x) for x in bytes_to_embedding(face.embedding)]
                    stored_count += 1
                else:
                    continue
                embeddings.append(emb)
                embedding_face_ids.append(face.id)
            except Exception:
                continue
        if not embeddings:
            continue
        students.append({
            "id": student.id,
            "full_name": student.full_name,
            "student_id": student.student_id,
            "embedding": embeddings[0],          # tương thích cũ
            "embeddings": embeddings,            # nhiều mẫu để train/match
            "embedding_face_ids": embedding_face_ids,
            "num_samples": len(embeddings),
            "embeddings_rebuilt": rebuilt_count,
            "embeddings_from_cache": stored_count,
        })
    return {
        "found": True,
        "session_id": session.id,
        "class_id": session.class_id,
        "class_name": cls.name,
        "room": session.room,
        "enrolled_count": len(enrollments),
        "approved_face_count": approved_face_count,
        "students": students,
    }


@app.post("/api/desktop/checkin", response_model=schemas.FaceMatchResult)
async def desktop_checkin(
    session_id: int = Form(...),
    room: str = Form(...),
    file: UploadFile = File(...),
    device_info: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    """Desktop sends captured frame for recognition + anti-spoof."""
    if not anti_spoof_model_available():
        raise HTTPException(503, "Thiếu model anti-spoof MiniFASNet; điểm danh đã bị khóa")
    # Save temp
    tmp = UPLOADS_DIR / "temp" / f"checkin_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.jpg"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    content = await file.read()
    with open(tmp, "wb") as f:
        f.write(content)
    img = load_image_bgr_from_bytes(content)
    if img is None:
        raise HTTPException(400, "Ảnh không hợp lệ")

    result = extract_embedding(img)
    if result is None:
        tmp.unlink(missing_ok=True)
        return schemas.FaceMatchResult(matched=False, is_spoof=False, message="Không phát hiện khuôn mặt")

    emb, info = result
    is_real, spoof_score = check_anti_spoof(img, info.get("bbox"))
    if not is_real:
        spoof_path = save_image(img, "spoof", f"room_{room}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg")
        log = SpoofLog(
            session_id=session_id,
            room=room,
            image_path=spoof_path,
            device_info=device_info,
            note=f"Anti-spoof failed, score={spoof_score:.2f}",
        )
        db.add(log)
        db.commit()
        tmp.unlink(missing_ok=True)
        return schemas.FaceMatchResult(
            matched=False, is_spoof=True, message="Phát hiện giả mạo! Đã ghi log hình ảnh."
        )

    # Load gallery for session
    session = db.query(AttendanceSession).filter(AttendanceSession.id == session_id).first()
    if not session or not session.is_open:
        tmp.unlink(missing_ok=True)
        return schemas.FaceMatchResult(matched=False, message="Buổi điểm danh đã đóng")

    gallery = []
    enrollments = db.query(Enrollment).filter(Enrollment.class_id == session.class_id).all()
    for enr in enrollments:
        faces = (
            db.query(FaceData)
            .filter(
                FaceData.user_id == enr.student_id,
                FaceData.status == FaceDataStatus.APPROVED,
                FaceData.embedding.isnot(None),
            )
            .all()
        )
        for face in faces:
            try:
                gallery.append((enr.student_id, bytes_to_embedding(face.embedding)))
            except Exception:
                continue

    matched_id, sim = match_face(emb, gallery)
    tmp.unlink(missing_ok=True)

    if matched_id is None:
        return schemas.FaceMatchResult(matched=False, confidence=sim, message=f"Không khớp (sim={sim:.3f})")

    student = db.query(User).filter(User.id == matched_id).first()
    # Record attendance
    rec = db.query(AttendanceRecord).filter(
        AttendanceRecord.session_id == session_id,
        AttendanceRecord.student_id == matched_id,
    ).first()
    now = datetime.now(timezone.utc)
    # Simple late rule: > 15 min after start
    status = AttendanceStatus.PRESENT
    if session.start_time and (now - session.start_time).total_seconds() > 15 * 60:
        status = AttendanceStatus.LATE

    if rec:
        if rec.status == AttendanceStatus.ABSENT:
            rec.status = status
            rec.check_in_time = now
            rec.confidence = sim
            rec.is_manual = False
    else:
        rec = AttendanceRecord(
            session_id=session_id,
            student_id=matched_id,
            status=status,
            check_in_time=now,
            confidence=sim,
            is_manual=False,
        )
        db.add(rec)
    db.commit()

    return schemas.FaceMatchResult(
        matched=True,
        student_id=matched_id,
        full_name=student.full_name if student else None,
        confidence=sim,
        is_spoof=False,
        message=f"Điểm danh thành công: {student.full_name if student else matched_id} ({status.value})",
    )



@app.post("/api/desktop/face-lookup")
def desktop_face_lookup(
    data: schemas.DesktopFaceLookup,
    db: Session = Depends(get_db),
):
    """Find an existing student for a live, open desktop attendance session."""
    session = db.query(AttendanceSession).filter(
        AttendanceSession.id == data.session_id, AttendanceSession.is_open == True
    ).first()
    if not session:
        raise HTTPException(404, "Buổi điểm danh không tồn tại hoặc đã đóng")

    enrolled_ids = {
        row.student_id for row in db.query(Enrollment.student_id)
        .filter(Enrollment.class_id == session.class_id).all()
    }
    rows = (
        db.query(FaceData, User)
        .join(User, User.id == FaceData.user_id)
        .filter(FaceData.status == FaceDataStatus.APPROVED, FaceData.embedding.isnot(None),
                User.role == UserRole.STUDENT, User.is_active == True)
        .all()
    )
    query = np.asarray(data.embedding, dtype=np.float32).flatten()
    if not np.isfinite(query).all() or np.linalg.norm(query) < 1e-8:
        raise HTTPException(400, "Embedding khuôn mặt không hợp lệ")
    query /= np.linalg.norm(query) + 1e-8
    def rank_templates(rebuild_from_images=False):
        best_by_student = {}
        usable_count = 0
        refreshed_count = 0
        for face, student in rows:
            try:
                sample = None
                if rebuild_from_images and face.image_path:
                    image_path = Path(__file__).resolve().parent.parent / face.image_path
                    if image_path.is_file():
                        extracted = extract_embedding_from_file(str(image_path))
                        if extracted is not None:
                            sample = extracted[0].astype(np.float32).flatten()
                            face.embedding = embedding_to_bytes(sample)
                            refreshed_count += 1
                if sample is None and face.embedding:
                    sample = bytes_to_embedding(face.embedding).astype(np.float32)
                if sample is None or sample.size != query.size or not np.isfinite(sample).all():
                    continue
                sample_norm = np.linalg.norm(sample)
                if sample_norm < 1e-8:
                    continue
                usable_count += 1
                if student.id in enrolled_ids:
                    continue
                sample /= sample_norm
                sim = float(np.dot(query, sample))
            except Exception:
                continue
            current = best_by_student.get(student.id)
            if current is None or sim > current["confidence"]:
                best_by_student[student.id] = {"student": student, "confidence": sim}
        ranked_matches = sorted(
            best_by_student.values(), key=lambda item: item["confidence"], reverse=True
        )
        return ranked_matches, usable_count, refreshed_count

    ranked, usable_template_count, refreshed_count = rank_templates()
    # Older approved embeddings may have been produced with a different crop
    # or color path. Rebuild from the approved source images when the cached
    # vectors do not produce a confident, unambiguous candidate.
    needs_rebuild = (
        not ranked
        or ranked[0]["confidence"] < DESKTOP_LOOKUP_THRESHOLD
        or (len(ranked) > 1 and
            ranked[0]["confidence"] - ranked[1]["confidence"] < DESKTOP_LOOKUP_MARGIN)
    )
    if needs_rebuild:
        ranked, usable_template_count, refreshed_count = rank_templates(rebuild_from_images=True)
        if refreshed_count:
            db.commit()
    if not ranked:
        reason = "no_approved_templates" if usable_template_count == 0 else "no_candidates"
        return {"found": False, "reason": reason,
                "usable_templates": usable_template_count,
                "embeddings_refreshed": refreshed_count}
    if ranked[0]["confidence"] < DESKTOP_LOOKUP_THRESHOLD:
        return {"found": False, "reason": "below_threshold",
                "confidence": ranked[0]["confidence"],
                "threshold": DESKTOP_LOOKUP_THRESHOLD,
                "usable_templates": usable_template_count,
                "embeddings_refreshed": refreshed_count}
    if len(ranked) > 1 and ranked[0]["confidence"] - ranked[1]["confidence"] < DESKTOP_LOOKUP_MARGIN:
        return {"found": False, "reason": "ambiguous", "ambiguous": True,
                "confidence": ranked[0]["confidence"],
                "margin": ranked[0]["confidence"] - ranked[1]["confidence"],
                "usable_templates": usable_template_count,
                "embeddings_refreshed": refreshed_count}
    best = ranked[0]
    student = best["student"]
    return {
        "found": True,
        "student": {"id": student.id, "full_name": student.full_name,
                    "student_id": student.student_id},
        "confidence": best["confidence"],
        "usable_templates": usable_template_count,
        "embeddings_refreshed": refreshed_count,
    }


@app.post("/api/desktop/session/{session_id}/enrollment-requests")
def desktop_create_enrollment_request(
    session_id: int,
    data: schemas.DesktopEnrollmentRequestCreate,
    db: Session = Depends(get_db),
):
    """Queue a known student from an open desktop session for lecturer approval."""
    session = db.query(AttendanceSession).filter(
        AttendanceSession.id == session_id, AttendanceSession.is_open == True
    ).first()
    if not session:
        raise HTTPException(404, "Buổi điểm danh không tồn tại hoặc đã đóng")
    cls = db.query(Class).filter(Class.id == session.class_id).first()
    if not cls:
        raise HTTPException(404, "Lớp của buổi điểm danh không tồn tại")
    student = db.query(User).filter(User.id == data.student_id, User.role == UserRole.STUDENT,
                                   User.is_active == True).first()
    if not student:
        raise HTTPException(404, "Không tìm thấy sinh viên đang hoạt động")
    if db.query(Enrollment).filter(
        Enrollment.student_id == student.id, Enrollment.class_id == session.class_id
    ).first():
        raise HTTPException(409, "Sinh viên đã thuộc lớp này")
    pending = db.query(EnrollmentRequest).filter(
        EnrollmentRequest.session_id == session_id,
        EnrollmentRequest.student_id == student.id,
        EnrollmentRequest.status == "pending",
    ).first()
    if pending:
        return {"request_id": pending.id, "status": pending.status,
                "message": "Yêu cầu đã được gửi, đang chờ giảng viên duyệt"}
    requester_id = session.created_by or cls.lecturer_id
    if requester_id is None:
        raise HTTPException(409, "Buổi điểm danh chưa gắn với tài khoản giảng viên")
    request = EnrollmentRequest(
        session_id=session.id, class_id=session.class_id, student_id=student.id,
        requested_by=requester_id, confidence=data.confidence, status="pending",
    )
    db.add(request)
    db.commit()
    db.refresh(request)
    return {"request_id": request.id, "status": request.status,
            "message": "Đã gửi yêu cầu đến giảng viên"}


@app.get("/api/lecturer/enrollment-requests")
def list_enrollment_requests(
    status_filter: str = "pending",
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.require_role(UserRole.ADMIN, UserRole.LECTURER)),
):
    query = db.query(EnrollmentRequest).join(
        Class, Class.id == EnrollmentRequest.class_id
    ).filter(EnrollmentRequest.status == status_filter)
    if current_user.role == UserRole.LECTURER:
        query = query.filter(Class.lecturer_id == current_user.id)
    requests_list = query.order_by(EnrollmentRequest.created_at.desc()).all()
    return [{
        "id": req.id,
        "student_id": req.student.id,
        "student_name": req.student.full_name,
        "student_code": req.student.student_id,
        "class_id": req.class_id,
        "class_name": req.class_.name,
        "room": req.session.room,
        "session_id": req.session_id,
        "confidence": req.confidence,
        "status": req.status,
        "created_at": req.created_at,
    } for req in requests_list]


@app.post("/api/lecturer/enrollment-requests/{request_id}/review")
def review_enrollment_request(
    request_id: int,
    data: schemas.EnrollmentRequestReview,
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.require_role(UserRole.ADMIN, UserRole.LECTURER)),
):
    request = db.query(EnrollmentRequest).filter(EnrollmentRequest.id == request_id).first()
    if not request:
        raise HTTPException(404, "Không tìm thấy yêu cầu")
    if current_user.role == UserRole.LECTURER and request.class_.lecturer_id != current_user.id:
        raise HTTPException(403, "Bạn không phụ trách lớp này")
    if request.status != "pending":
        raise HTTPException(409, "Yêu cầu đã được xử lý")

    now = datetime.now(timezone.utc)
    if data.approved:
        enrollment = db.query(Enrollment).filter(
            Enrollment.student_id == request.student_id,
            Enrollment.class_id == request.class_id,
        ).first()
        if not enrollment:
            db.add(Enrollment(student_id=request.student_id, class_id=request.class_id))
        session = request.session
        if session.is_open:
            record = db.query(AttendanceRecord).filter(
                AttendanceRecord.session_id == request.session_id,
                AttendanceRecord.student_id == request.student_id,
            ).first()
            attendance_status = AttendanceStatus.PRESENT
            session_start = session.start_time
            # SQLite may return timezone-aware columns as naive datetimes.
            # Session timestamps are written in UTC, so normalize before subtraction.
            if session_start and session_start.tzinfo is None:
                session_start = session_start.replace(tzinfo=timezone.utc)
            if session_start and (now - session_start).total_seconds() > 15 * 60:
                attendance_status = AttendanceStatus.LATE
            if record is None:
                db.add(AttendanceRecord(
                    session_id=request.session_id, student_id=request.student_id,
                    status=attendance_status, check_in_time=now,
                    confidence=request.confidence, is_manual=False,
                ))
            elif record.status == AttendanceStatus.ABSENT:
                record.status = attendance_status
                record.check_in_time = now
                record.confidence = request.confidence
                record.is_manual = False
        request.status = "approved"
    else:
        request.status = "rejected"
    request.reviewed_at = now
    request.reviewed_by = current_user.id
    db.commit()
    return {"message": "Đã thêm sinh viên vào lớp và ghi nhận điểm danh" if data.approved
            else "Đã từ chối yêu cầu", "status": request.status}


@app.post("/api/desktop/result")
def desktop_result(data: schemas.DesktopResult, db: Session = Depends(get_db)):
    """Edge gửi kết quả điểm danh (không gửi frame) — Central chỉ ghi nhận."""
    session = db.query(AttendanceSession).filter(AttendanceSession.id == data.session_id).first()
    if not session or not session.is_open:
        raise HTTPException(400, "Buổi điểm danh đã đóng hoặc không tồn tại")
    student = db.query(User).filter(User.id == data.student_id, User.role == UserRole.STUDENT).first()
    if not student:
        raise HTTPException(404, "Sinh viên không tồn tại")

    now = datetime.now(timezone.utc)
    status = AttendanceStatus.PRESENT
    if session.start_time and (now - session.start_time).total_seconds() > 15 * 60:
        status = AttendanceStatus.LATE

    rec = db.query(AttendanceRecord).filter(
        AttendanceRecord.session_id == data.session_id,
        AttendanceRecord.student_id == data.student_id,
    ).first()
    if rec:
        if rec.status == AttendanceStatus.ABSENT:
            rec.status = status
            rec.check_in_time = now
            rec.confidence = data.confidence
            rec.is_manual = False
    else:
        rec = AttendanceRecord(
            session_id=data.session_id,
            student_id=data.student_id,
            status=status,
            check_in_time=now,
            confidence=data.confidence,
            is_manual=False,
        )
        db.add(rec)
    db.commit()
    return {
        "message": "OK",
        "status": status.value,
        "student": student.full_name,
        "latency_ms_edge": data.latency_ms,
    }


@app.post("/api/desktop/spoof")
async def desktop_spoof(
    session_id: Optional[str] = Form(None),
    room: Optional[str] = Form(None),
    device_info: Optional[str] = Form(None),
    note: Optional[str] = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Edge chỉ upload ảnh bằng chứng khi phát hiện giả mạo."""
    content = await file.read()
    img = load_image_bgr_from_bytes(content)
    if img is None:
        raise HTTPException(400, "Ảnh không hợp lệ")
    sid = int(session_id) if session_id and session_id.isdigit() else None
    path = save_image(
        img, "spoof",
        f"edge_{room or 'na'}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg",
    )
    log = SpoofLog(
        session_id=sid,
        room=room,
        image_path=path,
        device_info=device_info,
        note=note or "Edge anti-spoof reject",
    )
    db.add(log)
    db.commit()
    return {"message": "Spoof log saved", "path": path}


@app.get("/api/spoof-logs")
def list_spoof_logs(
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.require_role(UserRole.ADMIN)),
):
    return db.query(SpoofLog).order_by(SpoofLog.detected_at.desc()).limit(100).all()

# ==================== USER FACES MANAGEMENT ====================

@app.get("/api/face/user/{user_id}", response_model=List[schemas.FaceDataOut])
def get_user_faces(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.require_role(UserRole.ADMIN, UserRole.LECTURER)),
):
    """API lấy toàn bộ danh sách ảnh của một sinh viên."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Không tìm thấy người dùng")
    
    faces = (
        db.query(FaceData)
        .filter(FaceData.user_id == user_id)
        .order_by(FaceData.submitted_at.desc())
        .all()
    )
    return faces


@app.delete("/api/face/{face_id}")
def delete_face(
    face_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.require_role(UserRole.ADMIN)),
):
    """API xóa ảnh khuôn mặt khỏi CSDL và ổ đĩa."""
    face = db.query(FaceData).filter(FaceData.id == face_id).first()
    if not face:
        raise HTTPException(status_code=404, detail="Không tìm thấy ảnh")

    # Xóa file ảnh vật lý trên ổ đĩa
    if face.image_path:
        abs_path = Path(__file__).resolve().parent.parent / face.image_path
        if abs_path.exists():
            try:
                abs_path.unlink()
            except Exception as e:
                print(f"Lỗi khi xóa file ảnh: {e}")

    # Xóa dữ liệu trong Database
    db.delete(face)
    db.commit()
    return {"message": f"Đã xóa thành công ảnh #{face_id}"}
# Web UI (HTML) — phục vụ trang đăng nhập tại /
WEB_DIR = Path(__file__).resolve().parent.parent / "web"


@app.get("/")
def root():
    index = WEB_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index), media_type="text/html; charset=utf-8")
    return {"message": "Face Attendance API", "docs": "/docs", "hint": "Thiếu web/index.html"}


@app.get("/app")
def web_app():
    """Alias trang web."""
    index = WEB_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index), media_type="text/html; charset=utf-8")
    raise HTTPException(404, "Không tìm thấy web/index.html")
