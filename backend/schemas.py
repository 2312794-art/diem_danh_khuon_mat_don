from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime
from database import UserRole, AttendanceStatus, FaceDataStatus


# ---------- Auth ----------
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    full_name: str
    user_id: int


class LoginRequest(BaseModel):
    username: str
    password: str


# ---------- User ----------
class UserBase(BaseModel):
    username: str
    email: str
    full_name: str
    role: UserRole
    student_id: Optional[str] = None
    phone: Optional[str] = None


class UserCreate(UserBase):
    password: str = Field(..., min_length=6)


class UserUpdate(BaseModel):
    email: Optional[str] = None
    full_name: Optional[str] = None
    phone: Optional[str] = None
    student_id: Optional[str] = None
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None
    password: Optional[str] = None


class UserOut(UserBase):
    id: int
    is_active: bool
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# ---------- Course ----------
class CourseCreate(BaseModel):
    code: str
    name: str
    credits: int = 3
    description: Optional[str] = None


class CourseUpdate(BaseModel):
    code: Optional[str] = None
    name: Optional[str] = None
    credits: Optional[int] = None
    description: Optional[str] = None


class CourseOut(CourseCreate):
    id: int
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# ---------- Class ----------
class ClassCreate(BaseModel):
    name: str
    course_id: int
    lecturer_id: Optional[int] = None
    room: Optional[str] = None
    semester: Optional[str] = None
    academic_year: Optional[str] = None
    max_students: int = 50


class ClassUpdate(BaseModel):
    name: Optional[str] = None
    lecturer_id: Optional[int] = None
    room: Optional[str] = None
    semester: Optional[str] = None
    academic_year: Optional[str] = None
    max_students: Optional[int] = None
    is_active: Optional[bool] = None


class ClassOut(BaseModel):
    id: int
    name: str
    course_id: int
    lecturer_id: Optional[int] = None
    room: Optional[str] = None
    semester: Optional[str] = None
    academic_year: Optional[str] = None
    max_students: int
    is_active: bool
    course: Optional[CourseOut] = None
    lecturer: Optional[UserOut] = None

    class Config:
        from_attributes = True


# ---------- Enrollment ----------
class EnrollmentCreate(BaseModel):
    student_id: int
    class_id: int


class EnrollmentOut(BaseModel):
    id: int
    student_id: int
    class_id: int
    enrolled_at: Optional[datetime] = None
    student: Optional[UserOut] = None

    class Config:
        from_attributes = True


# ---------- Face Data ----------
class FaceDataOut(BaseModel):
    id: int
    user_id: int
    status: FaceDataStatus
    image_path: Optional[str] = None
    submitted_at: Optional[datetime] = None
    reviewed_at: Optional[datetime] = None
    note: Optional[str] = None
    user: Optional[UserOut] = None

    class Config:
        from_attributes = True


class FaceReview(BaseModel):
    status: FaceDataStatus  # approved or rejected
    note: Optional[str] = None


# ---------- Attendance ----------
class SessionCreate(BaseModel):
    class_id: int
    session_date: datetime
    room: Optional[str] = None
    notes: Optional[str] = None


class SessionOut(BaseModel):
    id: int
    class_id: int
    session_date: datetime
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    room: Optional[str] = None
    is_open: bool
    notes: Optional[str] = None
    class_: Optional[ClassOut] = None

    class Config:
        from_attributes = True


class AttendanceRecordOut(BaseModel):
    id: int
    session_id: int
    student_id: int
    status: AttendanceStatus
    check_in_time: Optional[datetime] = None
    confidence: Optional[float] = None
    is_manual: bool
    student: Optional[UserOut] = None

    class Config:
        from_attributes = True


class ManualAttendance(BaseModel):
    student_id: int
    status: AttendanceStatus


class AttendanceStats(BaseModel):
    total_sessions: int
    present_count: int
    late_count: int
    absent_count: int
    attendance_rate: float


# ---------- Desktop / Spoof ----------
class SpoofLogCreate(BaseModel):
    session_id: Optional[int] = None
    room: Optional[str] = None
    device_info: Optional[str] = None
    note: Optional[str] = None


class FaceMatchResult(BaseModel):
    matched: bool
    student_id: Optional[int] = None
    full_name: Optional[str] = None
    confidence: Optional[float] = None
    is_spoof: bool = False
    message: str


# ---------- Edge → Central (hybrid) ----------
class DesktopResult(BaseModel):
    session_id: int
    student_id: int
    confidence: float
    room: Optional[str] = None
    device_info: Optional[str] = None
    latency_ms: Optional[float] = None
