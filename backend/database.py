from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, ForeignKey, Float, Text, LargeBinary, Enum as SAEnum, UniqueConstraint
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from sqlalchemy.sql import func
from datetime import datetime
import enum
import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:151205@127.0.0.1:5432/faceattend")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class UserRole(str, enum.Enum):
    ADMIN = "admin"
    LECTURER = "lecturer"
    STUDENT = "student"


class AttendanceStatus(str, enum.Enum):
    PRESENT = "present"
    ABSENT = "absent"
    LATE = "late"


class FaceDataStatus(str, enum.Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100), nullable=False)
    role = Column(SAEnum(UserRole), nullable=False)
    is_active = Column(Boolean, default=True)
    student_id = Column(String(20), unique=True, nullable=True)  # for students
    phone = Column(String(20), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships — 1 SV có nhiều ảnh khuôn mặt
    face_data = relationship(
        "FaceData",
        back_populates="user",
        uselist=True,
        foreign_keys="[FaceData.user_id]",
    )
    taught_classes = relationship("Class", back_populates="lecturer", foreign_keys="Class.lecturer_id")
    enrollments = relationship("Enrollment", back_populates="student")
    attendance_records = relationship("AttendanceRecord", back_populates="student")


class FaceData(Base):
    """Mỗi dòng = 1 ảnh khuôn mặt. Một sinh viên có thể có nhiều ảnh."""
    __tablename__ = "face_data"

    id = Column(Integer, primary_key=True, index=True)
    # Bỏ unique: cho phép nhiều ảnh / SV
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    embedding = Column(LargeBinary, nullable=True)  # SFace 128-d float32
    image_path = Column(String(255), nullable=True)
    status = Column(SAEnum(FaceDataStatus), default=FaceDataStatus.PENDING)
    submitted_at = Column(DateTime(timezone=True), server_default=func.now())
    reviewed_at = Column(DateTime(timezone=True), nullable=True)
    reviewed_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    note = Column(Text, nullable=True)

    user = relationship("User", back_populates="face_data", foreign_keys=[user_id])


class Course(Base):
    __tablename__ = "courses"

    id = Column(Integer, primary_key=True, index=True)
    code = Column(String(20), unique=True, nullable=False)  # e.g. CS101
    name = Column(String(100), nullable=False)
    credits = Column(Integer, default=3)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    classes = relationship("Class", back_populates="course")


class Class(Base):
    __tablename__ = "classes"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)  # e.g. CS101-01
    course_id = Column(Integer, ForeignKey("courses.id"), nullable=False)
    lecturer_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    room = Column(String(50), nullable=True)  # room code for desktop app
    semester = Column(String(20), nullable=True)
    academic_year = Column(String(20), nullable=True)
    max_students = Column(Integer, default=50)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    course = relationship("Course", back_populates="classes")
    lecturer = relationship("User", back_populates="taught_classes", foreign_keys=[lecturer_id])
    enrollments = relationship("Enrollment", back_populates="class_")
    sessions = relationship("AttendanceSession", back_populates="class_")


class Enrollment(Base):
    __tablename__ = "enrollments"
    __table_args__ = (UniqueConstraint("student_id", "class_id", name="uq_student_class"),)

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    class_id = Column(Integer, ForeignKey("classes.id"), nullable=False)
    enrolled_at = Column(DateTime(timezone=True), server_default=func.now())

    student = relationship("User", back_populates="enrollments")
    class_ = relationship("Class", back_populates="enrollments")


class AttendanceSession(Base):
    __tablename__ = "attendance_sessions"

    id = Column(Integer, primary_key=True, index=True)
    class_id = Column(Integer, ForeignKey("classes.id"), nullable=False)
    session_date = Column(DateTime(timezone=True), nullable=False)
    start_time = Column(DateTime(timezone=True), nullable=True)
    end_time = Column(DateTime(timezone=True), nullable=True)
    room = Column(String(50), nullable=True)
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    notes = Column(Text, nullable=True)
    is_open = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    class_ = relationship("Class", back_populates="sessions")
    records = relationship("AttendanceRecord", back_populates="session")


class AttendanceRecord(Base):
    __tablename__ = "attendance_records"
    __table_args__ = (UniqueConstraint("session_id", "student_id", name="uq_session_student"),)

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("attendance_sessions.id"), nullable=False)
    student_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    status = Column(SAEnum(AttendanceStatus), default=AttendanceStatus.ABSENT)
    check_in_time = Column(DateTime(timezone=True), nullable=True)
    confidence = Column(Float, nullable=True)
    is_manual = Column(Boolean, default=False)
    image_path = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    session = relationship("AttendanceSession", back_populates="records")
    student = relationship("User", back_populates="attendance_records")


class SpoofLog(Base):
    __tablename__ = "spoof_logs"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("attendance_sessions.id"), nullable=True)
    room = Column(String(50), nullable=True)
    image_path = Column(String(255), nullable=False)
    detected_at = Column(DateTime(timezone=True), server_default=func.now())
    device_info = Column(String(255), nullable=True)
    note = Column(Text, nullable=True)


class EnrollmentRequest(Base):
    """A lecturer-facing request to add an existing student during attendance."""
    __tablename__ = "enrollment_requests"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("attendance_sessions.id"), nullable=False, index=True)
    class_id = Column(Integer, ForeignKey("classes.id"), nullable=False, index=True)
    student_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    requested_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    confidence = Column(Float, nullable=False)
    status = Column(String(20), nullable=False, default="pending", index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    reviewed_at = Column(DateTime(timezone=True), nullable=True)
    reviewed_by = Column(Integer, ForeignKey("users.id"), nullable=True)

    session = relationship("AttendanceSession")
    class_ = relationship("Class")
    student = relationship("User", foreign_keys=[student_id])
    requester = relationship("User", foreign_keys=[requested_by])
    reviewer = relationship("User", foreign_keys=[reviewed_by])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    Base.metadata.create_all(bind=engine)
    # Gỡ unique(user_id) trên face_data nếu DB cũ còn constraint (cho phép nhiều ảnh / SV)
    try:
        with engine.begin() as conn:
            conn.exec_driver_sql(
                "ALTER TABLE face_data DROP CONSTRAINT IF EXISTS face_data_user_id_key"
            )
            # Một số bản PostgreSQL đặt tên constraint khác
            conn.exec_driver_sql(
                "ALTER TABLE face_data DROP CONSTRAINT IF EXISTS uq_face_data_user_id"
            )
    except Exception:
        pass
