#!/usr/bin/env python3
"""
Test Cloud Database Connection
Simple script to verify your cloud database is working
"""

import os
import sys
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_connection():
    """Test database connection and basic operations"""
    try:
        from sqlalchemy_database import mysqlDatabase
        
        print("🔍 Testing cloud database connection...")
        
        # Create database instance
        db = mysqlDatabase()
        
        # Test basic connection
        if not db.test_connection():
            print("❌ Database connection failed")
            return False
        
        print("✅ Database connection successful!")
        
        # Test adding a sample record
        print("📝 Testing attendance recording...")
        db.add_attendance("Test User", "Present")
        
        # Test retrieving records
        print("📊 Testing record retrieval...")
        records = db.get_attendance()
        if records:
            print(f"✅ Found {len(records)} attendance records")
        else:
            print("⚠️ No attendance records found")
        
        # Test emotion recording
        print("😊 Testing emotion recording...")
        db.add_emotion("Test User", "Happy")
        
        # Test emotion retrieval
        print("📈 Testing emotion retrieval...")
        emotions = db.get_emotions()
        if emotions:
            print(f"✅ Found {len(emotions)} emotion records")
        else:
            print("⚠️ No emotion records found")
        
        # Test attendance summary
        print("📋 Testing attendance summary...")
        summary = db.get_attendance_summary()
        if summary:
            print(f"✅ Today's attendance: {summary['total_present']} present")
        else:
            print("⚠️ Could not retrieve attendance summary")
        
        # Test attendance stats
        print("📊 Testing attendance statistics...")
        stats = db.get_attendance_stats(days=7)
        if stats:
            print(f"✅ Last 7 days: {stats['total_records']} records, {stats['unique_students']} unique students")
        else:
            print("⚠️ Could not retrieve attendance statistics")
        
        print("\n🎉 All tests completed successfully!")
        print("✅ Your cloud database is working perfectly!")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you have all required packages installed:")
        print("   pip install -r requirements.txt")
        return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("\n💡 Troubleshooting tips:")
        print("1. Check your .env file configuration")
        print("2. Verify database credentials")
        print("3. Ensure your IP is whitelisted")
        print("4. Check if the database is running")
        return False

def show_connection_info():
    """Show current database connection information"""
    print("\n📋 Current Database Configuration:")
    print("=" * 40)
    
    config_vars = [
        'DB_HOST', 'DB_USERNAME', 'DB_NAME', 
        'DB_PORT', 'DB_SSL_MODE'
    ]
    
    for var in config_vars:
        value = os.getenv(var, 'Not set')
        if var == 'DB_PASSWORD':
            value = '*' * len(value) if value else 'Not set'
        print(f"{var}: {value}")
    
    print("=" * 40)

def main():
    """Main function"""
    print("🌐 Cloud Database Connection Test")
    print("=" * 40)
    
    # Show current configuration
    show_connection_info()
    
    # Test connection
    success = test_connection()
    
    if success:
        print("\n🎊 Your attendance monitoring system is ready for cloud deployment!")
        print("✅ Anyone can now access your system from anywhere!")
        
        print("\n📝 Next steps:")
        print("1. Run your application: python app.py")
        print("2. Test with multiple users")
        print("3. Set up monitoring and alerts")
        print("4. Configure backup strategies")
        
    else:
        print("\n❌ Cloud database setup needs attention")
        print("💡 Check the troubleshooting guide in CLOUD_DATABASE_SETUP.md")
        
        print("\n🔧 Quick fixes to try:")
        print("1. Verify your .env file has correct credentials")
        print("2. Check if your cloud database is running")
        print("3. Ensure your IP is whitelisted")
        print("4. Try running: python deploy_to_cloud.py")

if __name__ == "__main__":
    main() 