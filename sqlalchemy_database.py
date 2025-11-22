# sqlalchemy_database.py
import os
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from datetime import datetime
from dotenv import load_dotenv
import urllib.parse
from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import UniqueConstraint
import logging

# Load environment variables from .env file
load_dotenv()

# Retrieve database credentials from environment variables
DB_USERNAME = os.getenv('DB_USERNAME')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_NAME = os.getenv('DB_NAME', 'class_records')
DB_PORT = os.getenv('DB_PORT', '3306')
DB_SSL_MODE = os.getenv('DB_SSL_MODE', 'PREFERRED')

# URL-encode the password for special characters
DB_PASSWORD_ENCODED = urllib.parse.quote(DB_PASSWORD, safe="") if DB_PASSWORD else ""

# Construct the SQLAlchemy connection string
# Support both local and cloud databases
if DB_HOST and DB_HOST != 'localhost':
    # Cloud database connection
    DATABASE_URL = f"mysql+mysqlconnector://{DB_USERNAME}:{DB_PASSWORD_ENCODED}@{DB_HOST}:{DB_PORT}/{DB_NAME}?ssl_mode={DB_SSL_MODE}"
else:
    # Local database connection
    DATABASE_URL = f"mysql+mysqlconnector://{DB_USERNAME}:{DB_PASSWORD_ENCODED}@{DB_HOST}/{DB_NAME}"

# Create engine with connection pooling for better performance
engine = create_engine(
    DATABASE_URL, 
    echo=False,  # Set to True for debugging
    isolation_level="READ COMMITTED",
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,  # Verify connections before use
    pool_recycle=3600    # Recycle connections every hour
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Define ORM base class
Base = declarative_base()

# ✅ **Attendance Table**
class Attendance(Base):
    __tablename__ = 'attendance_records'
    __table_args__ = (UniqueConstraint("name", "timestamp", name="uix_name_date"),)
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    status = Column(String(20), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    emotions = relationship("EmotionRecord", back_populates="attendance", cascade="all, delete-orphan")

# ✅ **Emotion Table**
class EmotionRecord(Base):
    __tablename__ = 'emotion_records'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    attendance_id = Column(Integer, ForeignKey('attendance_records.id'))
    emotion = Column(String(50), nullable=False)
    face_encoding = Column(String(500), nullable=True)
    attendance = relationship("Attendance", back_populates="emotions")

# ✅ Create tables in the database
try:
    Base.metadata.create_all(engine)
    logger.info("✅ Database tables created successfully")
except SQLAlchemyError as e:
    logger.error(f"❌ Error creating database tables: {e}")
    raise

# Create a session factory
SessionLocal = sessionmaker(bind=engine)
session = SessionLocal()

class mysqlDatabase:
    def __init__(self):
        self.engine = engine
        self.Session = SessionLocal

    def test_connection(self):
        """Test database connection"""
        try:
            with self.engine.connect() as connection:
                result = connection.execute("SELECT 1")
                logger.info("✅ Database connection successful")
                return True
        except SQLAlchemyError as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False

    def add_attendance(self, name, status):
        session = self.Session()
        try:
            # Add a unique constraint check: attendance per day per name
            existing_record = session.query(Attendance).filter(
                Attendance.name == name,
                Attendance.timestamp >= datetime.utcnow().date()
            ).first()
            
            if existing_record:
                logger.warning("%s already marked as %s today.", name, existing_record.status)
                return
            
            new_record = Attendance(name=name, status=status)
            session.add(new_record)
            session.commit()
            logger.info("✅ Attendance recorded: %s - %s", name, status)
        except SQLAlchemyError as e:
            session.rollback()
            logger.error("Error in add_attendance: %s", e)
        finally:
            session.close()

    def batch_add_attendance(self, attendance_list):
        """
        Adds multiple attendance records in a single commit.
        attendance_list: List of (name, status) tuples
        """
        session = self.Session()
        try:
            today = datetime.utcnow().date()
            new_records = []

            for name, status in attendance_list:
                existing_record = session.query(Attendance).filter(
                    Attendance.name == name,
                    func.date(Attendance.timestamp) == today
                ).first()

                if not existing_record:
                    new_records.append(Attendance(name=name, status=status))

            if new_records:
                session.bulk_save_objects(new_records)
                session.commit()
                logger.info("✅ %d attendance records added successfully.", len(new_records))
            else:
                logger.warning("⚠️ No new attendance records to add.")
        except SQLAlchemyError as e:
            session.rollback()
            logger.error("Error in batch_add_attendance: %s", e)
        finally:
            session.close()

    # ✅ Add emotion separately
    def add_emotion(self, name, emotion, face_encoding=None):
        session = self.Session()
        try:
            attendance_record = session.query(Attendance).filter(
                Attendance.name == name,
                Attendance.timestamp >= datetime.utcnow().date()
            ).first()
            
            if not attendance_record:
                logger.warning("No attendance record found for %s. Cannot store emotion.", name)
                return
            
            new_emotion = EmotionRecord(
                attendance_id=attendance_record.id,
                emotion=emotion,
                face_encoding=face_encoding
            )
            session.add(new_emotion)
            session.commit()
            logger.info("✅ Emotion recorded: %s - %s", name, emotion)
        except SQLAlchemyError as e:
            session.rollback()
            logger.error("Error in add_emotion: %s", e)
        finally:
            session.close()

    # ✅ Retrieve all attendance records
    def get_attendance(self):
        session = self.Session()
        try:
            records = session.query(Attendance).all()
            if not records:
                logger.warning("No attendance records found.")
                return []
            for record in records:
                logger.info("%s: %s - %s - %s", record.id, record.name, record.status, record.timestamp)
            return records
        except SQLAlchemyError as e:
            logger.error("Error in get_attendance: %s", e)
            return []
        finally:
            session.close()

    # ✅ Retrieve all emotion records
    def get_emotions(self):
        session = self.Session()
        try:
            records = session.query(EmotionRecord).all()
            if not records:
                logger.warning("No emotion records found.")
                return []
            for record in records:
                logger.info("%s: Attendance ID %s - Emotion: %s", record.id, record.attendance_id, record.emotion)
            return records
        except SQLAlchemyError as e:
            logger.error("Error in get_emotions: %s", e)
            return []
        finally:
            session.close()

    # ✅ Delete attendance record (also deletes related emotions)
    def delete_attendance(self, record_id):
        session = self.Session()
        try:
            record = session.query(Attendance).filter(Attendance.id == record_id).first()
            if record:
                session.delete(record)
                session.commit()
                logger.info("✅ Attendance record %s deleted successfully.", record_id)
            else:
                logger.warning("Attendance record %s not found.", record_id)
        except SQLAlchemyError as e:
            session.rollback()
            logger.error("Error in delete_attendance: %s", e)
        finally:
            session.close()

    # ✅ Delete emotion record
    def delete_emotion(self, record_id):
        session = self.Session()
        try:
            record = session.query(EmotionRecord).filter(EmotionRecord.id == record_id).first()
            if record:
                session.delete(record)
                session.commit()
                logger.info("✅ Emotion record %s deleted successfully.", record_id)
            else:
                logger.warning("Emotion record %s not found.", record_id)
        except SQLAlchemyError as e:
            session.rollback()
            logger.error("Error in delete_emotion: %s", e)
        finally:
            session.close()

    def assignment(self, arr, unk, m, name_list, emotions_list, month_n):
        """
        Fixed assignment method to correctly log attendance and emotions
        based on the track history (arr) for the specific track ID (m).
        """
        try:
            # Filter arr for the specific track ID 'm'
            # arr structure: [track_id, name, distance, emotion]
            person_history = [item for item in arr if item[0] == m]
            
            if not person_history:
                logger.warning(f"No history found for track ID {m}")
                return

            # Determine the name (filter out 'unknown' if possible, or take the most frequent)
            names = [item[1] for item in person_history if item[1] != 'unknown']
            if not names:
                # If only unknown, check if there are any records at all
                names = [item[1] for item in person_history]
            
            if not names:
                 logger.warning(f"Could not determine name for track ID {m}")
                 return

            # Simple majority vote or just take the last known name
            # Here we take the most frequent name to be robust
            from collections import Counter
            name = Counter(names).most_common(1)[0][0]
            
            if name == 'unknown':
                logger.info(f"Track ID {m} is unknown, skipping DB insertion.")
                return

            # 1. Add Attendance
            self.add_attendance(name, "Present")
            
            # 2. Add Emotions
            # Collect all emotions recorded for this person
            emotions = [item[3] for item in person_history if len(item) > 3]
            
            for emotion in emotions:
                if emotion: # Ensure emotion is not None or empty
                    self.add_emotion(name, emotion)
            
            logger.info(f"✅ Assignment completed for {name} (Track ID: {m})")
            
        except Exception as e:
            logger.error(f"Error in assignment: {e}")

    def get_attendance_summary(self, date=None):
        """
        Get attendance summary for a specific date
        """
        session = self.Session()
        try:
            if date is None:
                date = datetime.utcnow().date()
            
            records = session.query(Attendance).filter(
                func.date(Attendance.timestamp) == date
            ).all()
            
            summary = {
                'date': date,
                'total_present': len(records),
                'attendance_list': [record.name for record in records],
                'emotions': {}
            }
            
            # Get emotions for the day
            for record in records:
                emotions = session.query(EmotionRecord).filter(
                    EmotionRecord.attendance_id == record.id
                ).all()
                if emotions:
                    summary['emotions'][record.name] = [e.emotion for e in emotions]
            
            return summary
        except SQLAlchemyError as e:
            logger.error("Error in get_attendance_summary: %s", e)
            return None
        finally:
            session.close()

    def get_attendance_stats(self, days=30):
        """
        Get attendance statistics for the last N days
        """
        session = self.Session()
        try:
            from datetime import timedelta
            end_date = datetime.utcnow().date()
            start_date = end_date - timedelta(days=days)
            
            records = session.query(Attendance).filter(
                Attendance.timestamp >= start_date,
                Attendance.timestamp <= end_date
            ).all()
            
            stats = {
                'period': f"{start_date} to {end_date}",
                'total_days': days,
                'total_records': len(records),
                'unique_students': len(set(record.name for record in records)),
                'daily_attendance': {}
            }
            
            # Group by date
            for record in records:
                date_str = record.timestamp.date().isoformat()
                if date_str not in stats['daily_attendance']:
                    stats['daily_attendance'][date_str] = []
                stats['daily_attendance'][date_str].append(record.name)
            
            return stats
        except SQLAlchemyError as e:
            logger.error("Error in get_attendance_stats: %s", e)
            return None
        finally:
            session.close()

# ✅ **Testing the Script**
if __name__ == "__main__":
    db = mysqlDatabase()

    # Add attendance
    db.add_attendance("Alice", "Present")
    db.add_attendance("Bob", "Absent")

    # Add emotions
    db.add_emotion("Alice", "Happy")
    db.add_emotion("Bob", "Sad")

    # Retrieve attendance records
    print("\n📌 Attendance Records:")
    db.get_attendance()

    # Retrieve emotion records
    print("\n📌 Emotion Records:")
    db.get_emotions()

    # Test connection
    print("\n📌 Testing database connection...")
    if db.test_connection():
        print("✅ Database connection test successful")
    else:
        print("❌ Database connection test failed")

    # Get attendance summary
    print("\n📌 Attendance Summary:")
    summary = db.get_attendance_summary()
    if summary:
        print(f"Date: {summary['date']}")
        print(f"Total Present: {summary['total_present']}")
        print(f"Attendance List: {summary['attendance_list']}")
        print(f"Emotions: {summary['emotions']}")
    else:
        print("❌ Error retrieving attendance summary")

    # Get attendance stats
    print("\n📌 Attendance Stats:")
    stats = db.get_attendance_stats()
    if stats:
        print(f"Period: {stats['period']}")
        print(f"Total Days: {stats['total_days']}")
        print(f"Total Records: {stats['total_records']}")
        print(f"Unique Students: {stats['unique_students']}")
        print(f"Daily Attendance: {stats['daily_attendance']}")
    else:
        print("❌ Error retrieving attendance stats")