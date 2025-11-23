from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import func
from sqlalchemy_database import SessionLocal, Attendance, EmotionRecord
from datetime import datetime, date
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Attendance Monitoring API")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
def read_root():
    return {"message": "Attendance Monitoring API is running"}

@app.get("/attendance")
def get_attendance(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    Get recent attendance records.
    """
    records = db.query(Attendance).order_by(Attendance.timestamp.desc()).offset(skip).limit(limit).all()
    return records

@app.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    """
    Get summary statistics for the dashboard.
    """
    today = datetime.utcnow().date()
    
    # Total present today
    total_present_today = db.query(Attendance).filter(func.date(Attendance.timestamp) == today).count()
    
    # Total unique students today
    unique_students_today = db.query(Attendance.name).filter(func.date(Attendance.timestamp) == today).distinct().count()

    # Emotion distribution (all time for now, can be filtered)
    emotion_counts = db.query(EmotionRecord.emotion, func.count(EmotionRecord.emotion)).group_by(EmotionRecord.emotion).all()
    emotion_stats = {emotion: count for emotion, count in emotion_counts}

    return {
        "date": str(today),
        "total_present_today": total_present_today,
        "unique_students_today": unique_students_today,
        "emotion_distribution": emotion_stats
    }

@app.get("/emotions/{student_name}")
def get_student_emotions(student_name: str, db: Session = Depends(get_db)):
    """
    Get emotion history for a specific student.
    """
    # Find student attendance records
    attendance_ids = db.query(Attendance.id).filter(Attendance.name == student_name).all()
    attendance_ids = [id[0] for id in attendance_ids]
    
    if not attendance_ids:
        raise HTTPException(status_code=404, detail="Student not found")

    # Find emotions linked to these attendance records
    emotions = db.query(EmotionRecord).filter(EmotionRecord.attendance_id.in_(attendance_ids)).all()
    
    return emotions

@app.get("/students")
def get_students(db: Session = Depends(get_db)):
    """
    Get list of all students with their latest attendance info.
    """
    # Get distinct names
    students = db.query(Attendance.name).distinct().all()
    student_list = []
    
    for (name,) in students:
        # Get total attendance count
        count = db.query(Attendance).filter(Attendance.name == name).count()
        # Get last seen
        last_seen = db.query(Attendance.timestamp).filter(Attendance.name == name).order_by(Attendance.timestamp.desc()).first()
        
        student_list.append({
            "name": name,
            "attendance_count": count,
            "last_seen": last_seen[0] if last_seen else None,
            "status": "Active" # Placeholder logic
        })
        
    return student_list

@app.get("/reports/export")
def export_report(start_date: date = None, end_date: date = None, db: Session = Depends(get_db)):
    """
    Export attendance records as CSV.
    """
    from fastapi.responses import StreamingResponse
    import csv
    import io

    query = db.query(Attendance)
    if start_date:
        query = query.filter(func.date(Attendance.timestamp) >= start_date)
    if end_date:
        query = query.filter(func.date(Attendance.timestamp) <= end_date)
        
    records = query.order_by(Attendance.timestamp.desc()).all()
    
    # Create CSV in memory
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["ID", "Name", "Status", "Timestamp"])
    
    for record in records:
        writer.writerow([record.id, record.name, record.status, record.timestamp])
        
    stream.seek(0)
    
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=attendance_report.csv"
    return response

@app.get("/search")
def search(q: str, db: Session = Depends(get_db)):
    """
    Search for students or records.
    """
    students = db.query(Attendance.name).filter(Attendance.name.ilike(f"%{q}%")).distinct().limit(5).all()
    return {"students": [s[0] for s in students]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
