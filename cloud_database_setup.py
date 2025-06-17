#!/usr/bin/env python3
"""
Cloud Database Setup Script
Helps migrate from local to cloud databases for attendance monitoring system
"""

import os
import sys
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv
import urllib.parse
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_database_connection():
    """Create database connection with cloud database support"""
    try:
        # Get database credentials
        DB_USERNAME = os.getenv('DB_USERNAME')
        DB_PASSWORD = os.getenv('DB_PASSWORD')
        DB_HOST = os.getenv('DB_HOST')
        DB_NAME = os.getenv('DB_NAME', 'class_records')
        DB_PORT = os.getenv('DB_PORT', '3306')
        DB_SSL_MODE = os.getenv('DB_SSL_MODE', 'PREFERRED')
        
        if not all([DB_USERNAME, DB_PASSWORD, DB_HOST]):
            raise ValueError("Missing database credentials in .env file")
        
        # URL-encode password for special characters
        DB_PASSWORD_ENCODED = urllib.parse.quote(DB_PASSWORD, safe="")
        
        # Construct connection string
        if DB_HOST and DB_HOST != 'localhost':
            # Cloud database connection
            DATABASE_URL = f"mysql+mysqlconnector://{DB_USERNAME}:{DB_PASSWORD_ENCODED}@{DB_HOST}:{DB_PORT}/{DB_NAME}?ssl_mode={DB_SSL_MODE}"
        else:
            # Local database connection
            DATABASE_URL = f"mysql+mysqlconnector://{DB_USERNAME}:{DB_PASSWORD_ENCODED}@{DB_HOST}/{DB_NAME}"
        
        # Create engine with connection pooling
        engine = create_engine(
            DATABASE_URL,
            echo=False,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            pool_recycle=3600
        )
        
        return engine
        
    except Exception as e:
        logger.error(f"Failed to create database connection: {e}")
        return None

def test_connection(engine):
    """Test database connection"""
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            logger.info("✅ Database connection successful")
            return True
    except SQLAlchemyError as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False

def create_tables(engine):
    """Create database tables"""
    try:
        # Import the models
        from sqlalchemy_database import Base
        
        # Create all tables
        Base.metadata.create_all(engine)
        logger.info("✅ Database tables created successfully")
        return True
        
    except SQLAlchemyError as e:
        logger.error(f"❌ Error creating tables: {e}")
        return False

def insert_sample_data(engine):
    """Insert sample data for testing"""
    try:
        from sqlalchemy_database import Attendance, EmotionRecord
        from sqlalchemy.orm import sessionmaker
        from datetime import datetime
        
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Sample attendance records
        sample_attendance = [
            Attendance(name="John Doe", status="Present", timestamp=datetime.utcnow()),
            Attendance(name="Jane Smith", status="Present", timestamp=datetime.utcnow()),
            Attendance(name="Bob Johnson", status="Late", timestamp=datetime.utcnow())
        ]
        
        session.add_all(sample_attendance)
        session.commit()
        
        # Sample emotion records
        sample_emotions = [
            EmotionRecord(attendance_id=1, emotion="Happy"),
            EmotionRecord(attendance_id=2, emotion="Neutral"),
            EmotionRecord(attendance_id=3, emotion="Tired")
        ]
        
        session.add_all(sample_emotions)
        session.commit()
        
        logger.info("✅ Sample data inserted successfully")
        session.close()
        return True
        
    except SQLAlchemyError as e:
        logger.error(f"❌ Error inserting sample data: {e}")
        return False

def verify_database_setup(engine):
    """Verify database setup by checking tables and data"""
    try:
        with engine.connect() as connection:
            # Check if tables exist
            result = connection.execute(text("SHOW TABLES"))
            tables = [row[0] for row in result]
            logger.info(f"📋 Found tables: {tables}")
            
            # Check attendance records
            result = connection.execute(text("SELECT COUNT(*) FROM attendance_records"))
            attendance_count = result.fetchone()[0]
            logger.info(f"📊 Attendance records: {attendance_count}")
            
            # Check emotion records
            result = connection.execute(text("SELECT COUNT(*) FROM emotion_records"))
            emotion_count = result.fetchone()[0]
            logger.info(f"😊 Emotion records: {emotion_count}")
            
        return True
        
    except SQLAlchemyError as e:
        logger.error(f"❌ Error verifying database: {e}")
        return False

def main():
    """Main migration function"""
    logger.info("🚀 Starting database migration...")
    
    # Create database connection
    engine = create_database_connection()
    if not engine:
        logger.error("❌ Failed to create database connection")
        sys.exit(1)
    
    # Test connection
    if not test_connection(engine):
        logger.error("❌ Database connection test failed")
        sys.exit(1)
    
    # Create tables
    if not create_tables(engine):
        logger.error("❌ Failed to create tables")
        sys.exit(1)
    
    # Insert sample data (optional)
    insert_sample = input("Do you want to insert sample data? (y/n): ").lower().strip()
    if insert_sample == 'y':
        insert_sample_data(engine)
    
    # Verify setup
    if not verify_database_setup(engine):
        logger.error("❌ Database verification failed")
        sys.exit(1)
    
    logger.info("🎉 Database migration completed successfully!")
    logger.info("📝 Your attendance monitoring system is ready to use!")

if __name__ == "__main__":
    main() 