# Architecture Diagram

Sơ đồ mô tả kiến trúc và liên kết màn hình của hệ thống Face Attendance. Hệ thống gồm giao diện web phân quyền, API FastAPI, cơ sở dữ liệu PostgreSQL và ứng dụng Desktop Edge xử lý nhận diện khuôn mặt tại phòng học.

## Application Architecture

<!-- mermaid-checked: no \n, no em-dash/en-dash, no {} in labels, subgraphs are id["label"], arrows are -->|"label"|, all subgraphs closed by end, ids unique -->
~~~mermaid
flowchart TD
    subgraph ClientLayer["Client Layer"]
        Browser["Web Browser"]
        Desktop["Desktop Edge App"]
        Camera["Room Camera"]
    end
    subgraph WebLayer["Web Application Layer"]
        WebUI["HTML Tailwind Vanilla JS"]
        API["FastAPI REST API"]
        Auth["JWT Authentication"]
        FaceService["Face Processing Service"]
    end
    subgraph DataLayer["Data Layer"]
        PostgreSQL[("PostgreSQL")]
        Uploads[("Uploads Storage")]
        Models["ONNX Face Models"]
    end
    subgraph EdgeLayer["Edge Processing Layer"]
        EdgePipeline["Local Recognition Pipeline"]
        AntiSpoof["Anti Spoof Check"]
    end

    Browser -->|"HTTP requests"| WebUI
    WebUI -->|"REST and file upload"| API
    API -->|"token validation"| Auth
    API -->|"CRUD and attendance"| PostgreSQL
    API -->|"face image storage"| Uploads
    API -->|"embedding and matching"| FaceService
    FaceService -->|"loads models"| Models
    Desktop -->|"room session sync"| API
    Camera -->|"video frames"| Desktop
    Desktop -->|"local frame processing"| EdgePipeline
    EdgePipeline -->|"spoof decision"| AntiSpoof
    EdgePipeline -->|"checkin and spoof logs"| API
~~~

### Technology Stack Summary

| Layer | Technology | Version | Purpose |
|---|---|---:|---|
| Web UI | HTML, Tailwind CSS CDN, Vanilla JavaScript | Current | Login, dashboards and role based screens |
| Web API | FastAPI | 1.0 application API | Authentication, users, courses, classes and attendance |
| ORM | SQLAlchemy | Project dependency | Database models and sessions |
| Database | PostgreSQL | 16 | Persistent users, classes, face data and attendance |
| Authentication | OAuth2 form login and JWT | Project dependency | Secure API access and role authorization |
| Face recognition | InsightFace and SFace | Project dependency | Face embeddings and face matching |
| Face detection | YuNet ONNX | Local model | Detect faces in images and camera frames |
| Anti spoof | MiniFASNet or heuristic fallback | Local model | Detect presentation attacks |
| Desktop UI | CustomTkinter or Tkinter | Project dependency | Room camera and Edge monitoring |
| Desktop inference | ONNX Runtime and OpenCV | Project dependency | CPU based local recognition pipeline |

### Data Storage & External Services

PostgreSQL stores account, course, class, enrollment, face metadata, attendance session, attendance record and spoof log data. Uploaded face images and spoof evidence are served from the local `uploads` directory. ONNX models are loaded from the local `models` directory. The web interface communicates with the FastAPI service through REST calls, while the Desktop Edge application synchronizes room sessions and sends attendance results to the same API.

### Key Architectural Decisions

- Uses a single page web shell where role based navigation replaces the `main-content` region without a full page reload.
- Keeps face recognition and anti spoof inference local in the Desktop Edge process to reduce camera latency and protect raw frames.
- Uses role authorization for Admin, Lecturer and Student flows, with JWT tokens attached to API requests.

## Component Relationships

<!-- mermaid-checked: no \n, no em-dash/en-dash, no {} in labels, subgraphs are id["label"], arrows are -->|"label"|, all subgraphs closed by end, ids unique -->
~~~mermaid
flowchart LR
    subgraph PresentationLayer["Presentation"]
        cLogin["Login View"]
        cAppShell["Application Shell"]
        cSidebar["Role Sidebar"]
        cMain["Main Content"]
        cDesktopUI["Desktop Edge UI"]
    end
    subgraph BusinessLayer["Business Logic"]
        cAuth["Auth Module"]
        cUserFlow["User Management"]
        cCourseFlow["Course and Class Management"]
        cAttendanceFlow["Attendance Service"]
        cFaceFlow["Face Review Service"]
        cEdgeFlow["Edge Recognition Pipeline"]
    end
    subgraph DataAccessLayer["Data Access"]
        cDbSession["Database Session"]
        cModels["SQLAlchemy Models"]
        cFileStore["Upload File Store"]
    end
    subgraph InfraLayer["Infrastructure"]
        cJwt["JWT Dependency"]
        cCamera["Camera Capture"]
        cOnnx["ONNX Runtime"]
        cAntiSpoof["Anti Spoof Detector"]
    end

    cLogin -->|"submits credentials"| cAuth
    cAuth -->|"returns role and token"| cAppShell
    cAppShell -->|"builds navigation"| cSidebar
    cSidebar -->|"selects screen"| cMain
    cMain -->|"calls endpoints"| cUserFlow
    cMain -->|"calls endpoints"| cCourseFlow
    cMain -->|"calls endpoints"| cAttendanceFlow
    cMain -->|"uploads and reviews"| cFaceFlow
    cAuth -->|"validates token"| cJwt
    cUserFlow -->|"queries"| cDbSession
    cCourseFlow -->|"queries"| cDbSession
    cAttendanceFlow -->|"queries"| cDbSession
    cFaceFlow -->|"queries"| cDbSession
    cDbSession -->|"maps entities"| cModels
    cFaceFlow -->|"stores images"| cFileStore
    cDesktopUI -->|"starts capture"| cCamera
    cCamera -->|"frames"| cEdgeFlow
    cEdgeFlow -->|"inference"| cOnnx
    cEdgeFlow -->|"spoof decision"| cAntiSpoof
    cEdgeFlow -->|"checkin request"| cAttendanceFlow
~~~

### Component Inventory

| Component | Layer | Type | Responsibility |
|---|---|---|---|
| Login View | Presentation | Web screen | Collects username and password and displays login errors |
| Application Shell | Presentation | Web shell | Switches between login view and authenticated application view |
| Role Sidebar | Presentation | Navigation component | Displays Admin, Lecturer or Student menu items |
| Main Content | Presentation | Dynamic view host | Renders the selected dashboard and management screen |
| Desktop Edge UI | Presentation | Desktop screen | Controls API address, room, session sync, camera and logs |
| Auth Module | Business Logic | FastAPI auth module | Authenticates users and creates JWT tokens |
| User Management | Business Logic | API route group | Creates, updates, deactivates and restores users |
| Course and Class Management | Business Logic | API route group | Manages courses, classes, rooms and lecturers |
| Attendance Service | Business Logic | API route group | Opens sessions, closes sessions and records checkins |
| Face Review Service | Business Logic | API route group | Receives face images and supports approval decisions |
| Edge Recognition Pipeline | Business Logic | Local service | Detects faces, matches embeddings and prepares checkin results |
| Database Session | Data Access | SQLAlchemy session | Provides transactional database access |
| SQLAlchemy Models | Data Access | ORM model set | Represents users, classes, face data and attendance records |
| Upload File Store | Data Access | Local storage | Stores face images and spoof evidence |
| JWT Dependency | Infrastructure | Security dependency | Protects API routes and enforces roles |
| Camera Capture | Infrastructure | OpenCV component | Reads camera frames from the classroom device |
| ONNX Runtime | Infrastructure | Inference runtime | Runs YuNet, SFace and anti spoof models on CPU |
| Anti Spoof Detector | Infrastructure | Detection component | Classifies live faces and presentation attacks |
