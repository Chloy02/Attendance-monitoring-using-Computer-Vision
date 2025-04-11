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


# Load environment variables from .env file
load_dotenv()

# Retrieve database credentials from environment variables
DB_USERNAME = os.getenv('DB_USERNAME')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_NAME = "class_records"  # ✅ Renamed database

# URL-encode the password
DB_PASSWORD_ENCODED = urllib.parse.quote(DB_PASSWORD, safe="")

# Construct the correct SQLAlchemy connection string
DATABASE_URL = f"mysql+mysqlconnector://{DB_USERNAME}:{DB_PASSWORD_ENCODED}@{DB_HOST}/{DB_NAME}"

engine = create_engine(DATABASE_URL, echo=True, isolation_level="READ COMMITTED")

import logging

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
    attendance_id = Column(Integer, ForeignKey('attendance_records.id'))  # Foreign key to Attendance table
    emotion = Column(String(50), nullable=False)  # Detected emotion
    face_encoding = Column(String(500), nullable=True)  # Store face encoding

    # Relationship with Attendance table
    attendance = relationship("Attendance", back_populates="emotions")

# ✅ Create tables in the database
Base.metadata.create_all(engine)

# Create a session factory
SessionLocal = sessionmaker(bind=engine)
session = SessionLocal()


class mysqlDatabase:
    def __init__(self):
        self.engine = engine
        self.Session = SessionLocal

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
        record = self.session.query(Attendance).filter_by(id=record_id).first()
        if record:
            self.session.delete(record)
            self.session.commit()
            print(f"❌ Deleted attendance record ID {record_id}")
        else:
            print(f"⚠️ Record ID {record_id} not found.")

    # ✅ Delete emotion record
    def delete_emotion(self, record_id):
        record = self.session.query(EmotionRecord).filter_by(id=record_id).first()
        if record:
            self.session.delete(record)
            self.session.commit()
            print(f"❌ Deleted emotion record ID {record_id}")
        else:
            print(f"⚠️ Emotion record ID {record_id} not found.")


    def assignment(self, arr, unk, m, name_list, emotions_list, month_n):
        """
        Logs attendance for recognized individuals and separately logs emotions in batches.
        
        name_list: List of names corresponding to recognized faces.
        emotions_list: List of detected emotions corresponding to recognized faces.
        """
        session = self.Session()
        try:
            today = datetime.utcnow().date()

            # ✅ Prepare batch attendance records
            attendance_list = []
            emotion_list = []
            attendance_id_map = {}

            for i, name in enumerate(name_list):
                status = "Present" if name != "Unknown" else "Unknown"
                emotion = emotions_list[i] if i < len(emotions_list) else "Neutral"  # ✅ FIXED: Use emotions_list

                # Check if attendance already exists
                existing_record = session.query(Attendance).filter(
                    Attendance.name == name,
                    func.date(Attendance.timestamp) == today
                ).first()

                if not existing_record:
                    new_attendance = Attendance(name=name, status=status)
                    session.add(new_attendance)
                    session.flush()  # Get the ID before commit
                    attendance_id_map[name] = new_attendance.id
                else:
                    attendance_id_map[name] = existing_record.id  # Store existing ID

            session.commit()  # ✅ Batch commit attendance

            # ✅ Prepare batch emotion records
            for i, name in enumerate(name_list):
                if name in attendance_id_map:
                    emotion = emotions_list[i] if i < len(emotions_list) else "Neutral"  # ✅ FIXED: Use emotions_list
                    emotion_list.append(EmotionRecord(attendance_id=attendance_id_map[name], emotion=emotion))

            if emotion_list:
                session.bulk_save_objects(emotion_list)
                session.commit()  # ✅ Batch commit emotions

            logger.info("✅ Attendance and emotions successfully recorded for %d individuals.", len(name_list))

        except SQLAlchemyError as e:
            session.rollback()
            logger.error("⚠️ Error in assignment: %s", e)
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

















# sqlalchemy_database.py - modifications in the attendance database #MAR 26 6:07 PM
# import os
# from sqlalchemy import create_engine, Column, Integer, String, DateTime
# from sqlalchemy.orm import sessionmaker, declarative_base
# from datetime import datetime
# from dotenv import load_dotenv
# import urllib.parse

# # Load environment variables from .env file
# load_dotenv()

# # Retrieve database credentials from environment variables
# DB_USERNAME = os.getenv('DB_USERNAME')
# DB_PASSWORD = os.getenv('DB_PASSWORD')
# DB_HOST = os.getenv('DB_HOST')
# DB_NAME = os.getenv('DB_NAME')

# # URL-encode the password
# DB_PASSWORD_ENCODED = urllib.parse.quote(DB_PASSWORD, safe="")

# # Construct the correct SQLAlchemy connection string
# DATABASE_URL = f"mysql+mysqlconnector://{DB_USERNAME}:{DB_PASSWORD_ENCODED}@{DB_HOST}/{DB_NAME}"


# engine = create_engine(DATABASE_URL, echo=True, isolation_level="READ COMMITTED")  # Default safe option

# # Define ORM base class
# Base = declarative_base()

# # Modify Attendance table to include emotion and face encoding
# class Attendance(Base):
#     __tablename__ = 'attendance_records'
    
#     id = Column(Integer, primary_key=True, autoincrement=True)
#     name = Column(String(100), nullable=False)
#     status = Column(String(20), nullable=False)  # Present, Absent, Late
#     emotion = Column(String(50), nullable=True)  # Store detected emotion
#     timestamp = Column(DateTime, default=datetime.utcnow)
#     face_encoding = Column(String(500), nullable=True)  # Store face encoding for comparison


# # Create tables in the database
# Base.metadata.create_all(engine)

# # Create a session factory
# SessionLocal = sessionmaker(bind=engine)
# session = SessionLocal()

# # Insert new attendance record
# def add_attendance(name, status):
#     new_record = Attendance(name=name, status=status)
#     session.add(new_record)
#     session.commit()
#     print(f"✅ Attendance recorded: {name} - {status}")

# # Retrieve all attendance records
# def get_attendance():
#     records = session.query(Attendance).all()
#     for record in records:
#         print(f"{record.id}: {record.name} - {record.status} - {record.timestamp}")

# # Update an attendance record
# def update_attendance(record_id, new_status):
#     record = session.query(Attendance).filter_by(id=record_id).first()
#     if record:
#         record.status = new_status
#         session.commit()
#         print(f"✅ Updated ID {record_id} to {new_status}")
#     else:
#         print(f"⚠️ Record ID {record_id} not found.")

# # Delete an attendance record
# def delete_attendance(record_id):
#     record = session.query(Attendance).filter_by(id=record_id).first()
#     if record:
#         session.delete(record)
#         session.commit()
#         print(f"❌ Deleted record ID {record_id}")
#     else:
#         print(f"⚠️ Record ID {record_id} not found.")

# # Test the script
# if __name__ == "__main__":
#     # Add records
#     add_attendance("Alice", "Present")
#     add_attendance("Bob", "Absent")

#     # Get all records
#     print("\n📌 Attendance Records:")
#     get_attendance()

#     # Update a record
#     update_attendance(1, "Late")

#     # Get updated records
#     print("\n📌 Updated Attendance Records:")
#     get_attendance()

#     # Delete a record
#     delete_attendance(2)

#     # Get final records
#     print("\n📌 Final Attendance Records:")
#     get_attendance()

#     # Close the session
#     session.close()


# class mysqlDatabase:
#     def __init__(self, host=None, username=None, password=None, db_name=None):
#         self.host = os.getenv('DB_HOST')
#         self.username = os.getenv('DB_USERNAME')
#         self.password = os.getenv('DB_PASSWORD')
#         self.db_name = os.getenv('DB_NAME')

#         # Initialize database connection
#         DATABASE_URL = f"mysql+mysqlconnector://{self.username}:{urllib.parse.quote(self.password)}@{self.host}/{self.db_name}"
#         self.engine = create_engine(DATABASE_URL, echo=True, isolation_level="READ COMMITTED")
#         self.session = SessionLocal()



#     # Modify add_attendance to prevent duplicate entries and store emotion
#     def add_attendance(self, name, status, emotion=None, face_encoding=None):
#         existing_record = self.session.query(Attendance).filter_by(name=name, timestamp=datetime.utcnow().date()).first()
        
#         if existing_record:
#             print(f"⚠️ {name} already marked as {existing_record.status} today.")
#             return
        
#         new_record = Attendance(name=name, status=status, emotion=emotion, face_encoding=face_encoding)
#         self.session.add(new_record)
#         self.session.commit()
#         print(f"✅ Attendance recorded: {name} - {status} - Emotion: {emotion}")

#     def get_attendance(self):
#         """
#         Fetches and prints all attendance records from the database.
#         """
#         records = self.session.query(Attendance).all()
#         if not records:
#             print("⚠️ No attendance records found.")
#             return

#         for record in records:
#             print(f"{record.id}: {record.name} - {record.status} - {record.emotion} - {record.timestamp}")



#     def update_attendance(self, record_id, new_status):
#         record = self.session.query(Attendance).filter_by(id=record_id).first()
#         if record:
#             record.status = new_status
#             self.session.commit()
#             print(f"✅ Updated ID {record_id} to {new_status}")
#         else:
#             print(f"⚠️ Record ID {record_id} not found.")

#     def delete_attendance(self, record_id):
#         record = self.session.query(Attendance).filter_by(id=record_id).first()
#         if record:
#             self.session.delete(record)
#             self.session.commit()
#             print(f"❌ Deleted record ID {record_id}")
#         else:
#             print(f"⚠️ Record ID {record_id} not found.")
    
#     # Modify assignment to check for known vs unknown faces and log emotions
#     def assignment(self, arr, unk, m, name_list, month_n, emotions):
#         """
#         Logs attendance for recognized individuals, including emotion data.

#         arr: List of recognized face encodings
#         unk: List of unknown face encodings
#         m: Tracker ID (if used for tracking individuals)
#         name_list: List of names corresponding to recognized faces
#         month_n: Current month (used as table name if necessary)
#         emotions: List of detected emotions corresponding to recognized faces
#         """
#         try:
#             for i, name in enumerate(name_list):
#                 status = "Present" if name != "Unknown" else "Unknown"
#                 emotion = emotions[i] if i < len(emotions) else "Neutral"

#                 # Prevent duplicate attendance entries
#                 existing_record = self.session.query(Attendance).filter_by(name=name, timestamp=datetime.utcnow().date()).first()
#                 if existing_record:
#                     print(f"⚠️ {name} already marked as {existing_record.status} today.")
#                     continue
                
#                 new_record = Attendance(name=name, status=status, emotion=emotion)
#                 self.session.add(new_record)
            
#             self.session.commit()
#             print("✅ Attendance assignment successful.")
#         except Exception as e:
#             self.session.rollback()
#             print(f"⚠️ Error in assignment: {e}")












# # #sqlalchemy_database.py - working one - 5 40 pm Mar 26 
# import os
# from sqlalchemy import create_engine, Column, Integer, String, DateTime
# from sqlalchemy.orm import sessionmaker, declarative_base
# from datetime import datetime
# from dotenv import load_dotenv
# import urllib.parse

# # Load environment variables from .env file
# load_dotenv()

# # Retrieve database credentials from environment variables
# DB_USERNAME = os.getenv('DB_USERNAME')
# DB_PASSWORD = os.getenv('DB_PASSWORD')
# DB_HOST = os.getenv('DB_HOST')
# DB_NAME = os.getenv('DB_NAME')

# # URL-encode the password
# DB_PASSWORD_ENCODED = urllib.parse.quote(DB_PASSWORD, safe="")

# # Construct the correct SQLAlchemy connection string
# DATABASE_URL = f"mysql+mysqlconnector://{DB_USERNAME}:{DB_PASSWORD_ENCODED}@{DB_HOST}/{DB_NAME}"


# engine = create_engine(DATABASE_URL, echo=True, isolation_level="READ COMMITTED")  # Default safe option

# # Define ORM base class
# Base = declarative_base()

# # Define the Attendance table
# class Attendance(Base):
#     __tablename__ = 'attendance_records'
    
#     id = Column(Integer, primary_key=True, autoincrement=True)
#     name = Column(String(100), nullable=False)
#     status = Column(String(20), nullable=False)  # Present, Absent, Late
#     timestamp = Column(DateTime, default=datetime.utcnow)

# # Create tables in the database
# Base.metadata.create_all(engine)

# # Create a session factory
# SessionLocal = sessionmaker(bind=engine)
# session = SessionLocal()

# # Insert new attendance record
# def add_attendance(name, status):
#     new_record = Attendance(name=name, status=status)
#     session.add(new_record)
#     session.commit()
#     print(f"✅ Attendance recorded: {name} - {status}")

# # Retrieve all attendance records
# def get_attendance():
#     records = session.query(Attendance).all()
#     for record in records:
#         print(f"{record.id}: {record.name} - {record.status} - {record.timestamp}")

# # Update an attendance record
# def update_attendance(record_id, new_status):
#     record = session.query(Attendance).filter_by(id=record_id).first()
#     if record:
#         record.status = new_status
#         session.commit()
#         print(f"✅ Updated ID {record_id} to {new_status}")
#     else:
#         print(f"⚠️ Record ID {record_id} not found.")

# # Delete an attendance record
# def delete_attendance(record_id):
#     record = session.query(Attendance).filter_by(id=record_id).first()
#     if record:
#         session.delete(record)
#         session.commit()
#         print(f"❌ Deleted record ID {record_id}")
#     else:
#         print(f"⚠️ Record ID {record_id} not found.")

# # Test the script
# if __name__ == "__main__":
#     # Add records
#     add_attendance("Alice", "Present")
#     add_attendance("Bob", "Absent")

#     # Get all records
#     print("\n📌 Attendance Records:")
#     get_attendance()

#     # Update a record
#     update_attendance(1, "Late")

#     # Get updated records
#     print("\n📌 Updated Attendance Records:")
#     get_attendance()

#     # Delete a record
#     delete_attendance(2)

#     # Get final records
#     print("\n📌 Final Attendance Records:")
#     get_attendance()

#     # Close the session
#     session.close()


# class mysqlDatabase:
#     def __init__(self, host=None, username=None, password=None, db_name=None):
#         self.host = os.getenv('DB_HOST')
#         self.username = os.getenv('DB_USERNAME')
#         self.password = os.getenv('DB_PASSWORD')
#         self.db_name = os.getenv('DB_NAME')

#         # Initialize database connection
#         DATABASE_URL = f"mysql+mysqlconnector://{self.username}:{urllib.parse.quote(self.password)}@{self.host}/{self.db_name}"
#         self.engine = create_engine(DATABASE_URL, echo=True, isolation_level="READ COMMITTED")
#         self.session = SessionLocal()



#     def add_attendance(self, name, status):
#         new_record = Attendance(name=name, status=status)
#         self.session.add(new_record)
#         self.session.commit()
#         print(f"✅ Attendance recorded: {name} - {status}")

#     def get_attendance(self):
#         records = self.session.query(Attendance).all()
#         for record in records:
#             print(f"{record.id}: {record.name} - {record.status} - {record.timestamp}")

#     def update_attendance(self, record_id, new_status):
#         record = self.session.query(Attendance).filter_by(id=record_id).first()
#         if record:
#             record.status = new_status
#             self.session.commit()
#             print(f"✅ Updated ID {record_id} to {new_status}")
#         else:
#             print(f"⚠️ Record ID {record_id} not found.")

#     def delete_attendance(self, record_id):
#         record = self.session.query(Attendance).filter_by(id=record_id).first()
#         if record:
#             self.session.delete(record)
#             self.session.commit()
#             print(f"❌ Deleted record ID {record_id}")
#         else:
#             print(f"⚠️ Record ID {record_id} not found.")
    
#     def assignment(self, arr, unk, m, name_list, month_n):
#         """
#         Logs attendance for recognized individuals.
        
#         arr: List of recognized face encodings
#         unk: List of unknown face encodings
#         m: Tracker ID (if used for tracking individuals)
#         name_list: List of names corresponding to recognized faces
#         month_n: Current month (used as table name if necessary)
#         """
#         try:
#             for i, name in enumerate(name_list):
#                 status = "Present" if name != "Unknown" else "Unknown"
#                 new_record = Attendance(name=name, status=status)
#                 self.session.add(new_record)
#             self.session.commit()
#             print("✅ Attendance assignment successful.")
#         except Exception as e:
#             self.session.rollback()
#             print(f"⚠️ Error in assignment: {e}")












# import os
# from typing import List, Optional, Union

# from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, UniqueConstraint
# from sqlalchemy.orm import declarative_base, sessionmaker, Session
# from sqlalchemy.exc import SQLAlchemyError, IntegrityError

# from datetime import datetime
# import calendar
# import logging
# from dotenv import load_dotenv

# # Load environment variables from .env file
# load_dotenv()

# # Fetch database credentials from environment variables
# DB_USERNAME = os.getenv('DB_USERNAME')
# DB_PASSWORD = os.getenv('DB_PASSWORD')
# DB_HOST = os.getenv('DB_HOST', 'localhost')  # Default to localhost if not provided
# DB_NAME = os.getenv('DB_NAME')

# # Ensure variables are correctly loaded
# print(f"Loaded DB Credentials: {DB_USERNAME}, {DB_HOST}, {DB_NAME}")


# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s: %(message)s',
#     handlers=[
#         logging.StreamHandler(),
#         logging.FileHandler('attendance_tracker.log')
#     ]
# )
# logger = logging.getLogger(__name__)

# # Create Base class for declarative models
# Base = declarative_base()


# class AttendanceRecord(Base):
#     """
#     Enhanced SQLAlchemy model for storing comprehensive attendance records
#     """
#     __tablename__ = 'attendance_records'

#     id = Column(Integer, primary_key=True, autoincrement=True)
#     person_name = Column(String(50), nullable=False, index=True)
#     tracker_id = Column(Integer, nullable=False, index=True)
#     emotion = Column(String(20))
#     recognition_confidence = Column(Float)
#     timestamp = Column(DateTime, default=datetime.now, index=True)
#     month = Column(String(20), index=True)

#     __table_args__ = (UniqueConstraint('person_name', 'tracker_id', 'month', name='unique_attendance'),)

#     def __repr__(self):
#         """String representation of the record"""
#         return (f"<AttendanceRecord(name={self.person_name}, "
#                 f"tracker_id={self.tracker_id}, "
#                 f"emotion={self.emotion}, "
#                 f"timestamp={self.timestamp})>")


# class DatabaseManager:
#     def __init__(self,
#                  username: str = DB_USERNAME,
#                  password: str = DB_PASSWORD,
#                  host: str = DB_HOST,
#                  database: str = DB_NAME):
#         try:
#             # Debugging: Print out connection details (remove in production)
#             print(f"Connecting with: username={username}, host={host}, database={database}")
            
#             # Ensure all required parameters are present
#             if not all([username, password, host, database]):
#                 raise ValueError("Missing database connection parameters. Check your .env file.")

#             # Use function parameters instead of undefined global variables
#             connection_url = f"mysql+pymysql://{username}:{password}@{host}/{database}"

#             self.engine = create_engine(
#                 connection_url,
#                 echo=True,  # Enable detailed logging for debugging
#                 pool_pre_ping=True  # Test connection before using
#             )

#             self.SessionLocal = sessionmaker(bind=self.engine)
#             logging.info("Database connection established successfully")

#         except Exception as e:
#             logging.error(f"Database connection failed: {e}")
#             raise

        
#     def get_session(self) -> Session:
#         """
#         Create and return a new database session

#         Returns:
#             Session: SQLAlchemy database session
#         """
#         return self.SessionLocal()

#     def insert_attendance(
#         self,
#         name: str,
#         tracker_id: int,
#         emotion: str,
#         confidence: Optional[float] = None
#     ) -> bool:
#         """
#         Insert a new attendance record with improved error handling, checking for existing records first.

#         Args:
#             name (str): Name of the person
#             tracker_id (int): Unique tracker ID
#             emotion (str): Detected emotion
#             confidence (float, optional): Recognition confidence

#         Returns:
#             bool: True if insertion successful, False otherwise
#         """
#         session = self.get_session()

#         try:
#             # Get current month
#             today = datetime.now()
#             month_name = calendar.month_name[today.month]

#             # Check if the record already exists
#             existing_record = session.query(AttendanceRecord).filter_by(
#                 person_name=name,
#                 tracker_id=tracker_id,
#                 month=month_name
#             ).first()

#             if existing_record:
#                 logger.info(f"Attendance record already exists for {name} (tracker_id: {tracker_id}) in {month_name}")
#                 return True  # No need to insert again

#             # Create new attendance record
#             new_record = AttendanceRecord(
#                 person_name=name,
#                 tracker_id=tracker_id,
#                 emotion=emotion,
#                 recognition_confidence=confidence,
#                 month=month_name
#             )

#             # Add and commit the record
#             session.add(new_record)
#             session.commit()

#             logger.info(f"Attendance recorded for {name} (tracker_id: {tracker_id})")
#             return True

#         except IntegrityError as e:
#             session.rollback()
#             logger.warning(f"Integrity error inserting attendance for {name} (tracker_id: {tracker_id}): {e}")
#             return True # Handle unique constraint violation as "success"
        
#         except SQLAlchemyError as e:
#             session.rollback()
#             logger.error(f"Error inserting attendance for {name} (tracker_id: {tracker_id}): {e}")
#             return False

#         finally:
#             session.close()

#     def get_attendance_by_month(self, month: str) -> List[AttendanceRecord]:
#         """
#         Retrieve attendance records for a specific month

#         Args:
#             month (str): Month name to query

#         Returns:
#             List[AttendanceRecord]: List of attendance records
#         """
#         session = self.get_session()
#         try:
#             # Query attendance records for a specific month
#             records = session.query(AttendanceRecord).filter_by(month=month).all()
#             return records

#         except SQLAlchemyError as e:
#             logger.error(f"Error retrieving attendance for {month}: {e}")
#             return []

#         finally:
#             session.close()

#     def get_unique_attendees(self, month: Optional[str] = None) -> List[str]:
#         """
#         Get unique attendees for a given month

#         Args:
#             month (str, optional): Specific month to query

#         Returns:
#             List[str]: List of unique attendee names
#         """
#         session = self.get_session()
#         try:
#             if month:
#                 unique_attendees = session.query(AttendanceRecord.person_name) \
#                     .filter_by(month=month) \
#                     .distinct() \
#                     .all()
#             else:
#                 unique_attendees = session.query(AttendanceRecord.person_name) \
#                     .distinct() \
#                     .all()

#             return [attendee[0] for attendee in unique_attendees]

#         except SQLAlchemyError as e:
#             logger.error(f"Error retrieving unique attendees: {e}")
#             return []

#         finally:
#             session.close()



# '''
# 1. Unique Constraint: Added a UniqueConstraint to the AttendanceRecord model to prevent redundant entries based on person_name, tracker_id, and month.
# 2. Check for Existing Record Before Insert: The insert_attendance function now checks if a record with the same person_name, tracker_id, and month already exists before attempting to insert a new record. This prevents duplicate entries.
# 3. Handle IntegrityError: Added IntegrityError exception handling, and log that the unique record already exist.
# 4. Logging: Added tracker_id info the logs.'''