#!/usr/bin/env python3
"""
Cloud Deployment Script for Attendance Monitoring System
Automates the process of migrating to cloud databases
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CloudDeployer:
    def __init__(self):
        self.env_file = Path('.env')
        self.backup_file = Path('.env.backup')
        
    def backup_current_config(self):
        """Backup current .env configuration"""
        if self.env_file.exists():
            import shutil
            shutil.copy2(self.env_file, self.backup_file)
            logger.info("✅ Current configuration backed up to .env.backup")
        else:
            logger.warning("⚠️ No .env file found to backup")
    
    def restore_backup(self):
        """Restore configuration from backup"""
        if self.backup_file.exists():
            import shutil
            shutil.copy2(self.backup_file, self.env_file)
            logger.info("✅ Configuration restored from backup")
        else:
            logger.error("❌ No backup file found")
    
    def get_cloud_provider(self):
        """Get user's preferred cloud provider"""
        print("\n🌐 Choose your cloud database provider:")
        print("1. AWS RDS (Recommended)")
        print("2. Google Cloud SQL")
        print("3. Azure Database")
        print("4. Heroku (JawsDB)")
        print("5. Railway")
        print("6. Custom/Other")
        
        while True:
            try:
                choice = int(input("\nEnter your choice (1-6): "))
                if 1 <= choice <= 6:
                    providers = {
                        1: "aws_rds",
                        2: "google_cloud",
                        3: "azure",
                        4: "heroku",
                        5: "railway",
                        6: "custom"
                    }
                    return providers[choice]
                else:
                    print("❌ Please enter a number between 1 and 6")
            except ValueError:
                print("❌ Please enter a valid number")
    
    def get_aws_rds_config(self):
        """Get AWS RDS configuration from user"""
        print("\n🔧 AWS RDS Configuration")
        print("Please provide your AWS RDS connection details:")
        
        config = {}
        config['DB_HOST'] = input("Database Endpoint (e.g., attendance-db.xxxxx.region.rds.amazonaws.com): ").strip()
        config['DB_USERNAME'] = input("Username (default: admin): ").strip() or "admin"
        config['DB_PASSWORD'] = input("Password: ").strip()
        config['DB_NAME'] = input("Database Name (default: class_records): ").strip() or "class_records"
        config['DB_PORT'] = input("Port (default: 3306): ").strip() or "3306"
        config['DB_SSL_MODE'] = input("SSL Mode (default: PREFERRED): ").strip() or "PREFERRED"
        
        return config
    
    def get_google_cloud_config(self):
        """Get Google Cloud SQL configuration from user"""
        print("\n🔧 Google Cloud SQL Configuration")
        print("Please provide your Google Cloud SQL connection details:")
        
        config = {}
        config['DB_HOST'] = input("Instance Connection Name (e.g., project:region:instance): ").strip()
        config['DB_USERNAME'] = input("Username: ").strip()
        config['DB_PASSWORD'] = input("Password: ").strip()
        config['DB_NAME'] = input("Database Name (default: class_records): ").strip() or "class_records"
        config['DB_PORT'] = input("Port (default: 3306): ").strip() or "3306"
        config['DB_SSL_MODE'] = input("SSL Mode (default: PREFERRED): ").strip() or "PREFERRED"
        
        return config
    
    def get_heroku_config(self):
        """Get Heroku configuration from user"""
        print("\n🔧 Heroku Configuration")
        print("Please provide your Heroku JawsDB connection details:")
        
        config = {}
        jawsdb_url = input("JawsDB URL (e.g., mysql://user:pass@host:port/db): ").strip()
        
        # Parse JawsDB URL
        if jawsdb_url.startswith('mysql://'):
            parts = jawsdb_url.replace('mysql://', '').split('@')
            if len(parts) == 2:
                user_pass = parts[0].split(':')
                host_port_db = parts[1].split('/')
                
                if len(user_pass) >= 2 and len(host_port_db) >= 2:
                    config['DB_USERNAME'] = user_pass[0]
                    config['DB_PASSWORD'] = user_pass[1]
                    
                    host_port = host_port_db[0].split(':')
                    config['DB_HOST'] = host_port[0]
                    config['DB_PORT'] = host_port[1] if len(host_port) > 1 else "3306"
                    config['DB_NAME'] = host_port_db[1]
                    config['DB_SSL_MODE'] = "PREFERRED"
        
        # Fallback manual input if parsing fails
        if not config:
            print("⚠️ Could not parse JawsDB URL, please enter manually:")
            config['DB_HOST'] = input("Host: ").strip()
            config['DB_USERNAME'] = input("Username: ").strip()
            config['DB_PASSWORD'] = input("Password: ").strip()
            config['DB_NAME'] = input("Database: ").strip()
            config['DB_PORT'] = input("Port (default: 3306): ").strip() or "3306"
            config['DB_SSL_MODE'] = "PREFERRED"
        
        return config
    
    def get_custom_config(self):
        """Get custom database configuration from user"""
        print("\n🔧 Custom Database Configuration")
        print("Please provide your database connection details:")
        
        config = {}
        config['DB_HOST'] = input("Host: ").strip()
        config['DB_USERNAME'] = input("Username: ").strip()
        config['DB_PASSWORD'] = input("Password: ").strip()
        config['DB_NAME'] = input("Database Name (default: class_records): ").strip() or "class_records"
        config['DB_PORT'] = input("Port (default: 3306): ").strip() or "3306"
        config['DB_SSL_MODE'] = input("SSL Mode (default: PREFERRED): ").strip() or "PREFERRED"
        
        return config
    
    def update_env_file(self, config):
        """Update .env file with new configuration"""
        try:
            # Read existing .env file
            env_lines = []
            if self.env_file.exists():
                with open(self.env_file, 'r') as f:
                    env_lines = f.readlines()
            
            # Update or add database configuration
            updated_keys = set()
            new_lines = []
            
            for line in env_lines:
                line = line.strip()
                if line and '=' in line:
                    key = line.split('=')[0]
                    if key in config:
                        new_lines.append(f"{key}={config[key]}\n")
                        updated_keys.add(key)
                    else:
                        new_lines.append(line + '\n')
                else:
                    new_lines.append(line + '\n')
            
            # Add new keys that weren't in the original file
            for key, value in config.items():
                if key not in updated_keys:
                    new_lines.append(f"{key}={value}\n")
            
            # Write updated .env file
            with open(self.env_file, 'w') as f:
                f.writelines(new_lines)
            
            logger.info("✅ .env file updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error updating .env file: {e}")
            return False
    
    def test_connection(self):
        """Test database connection"""
        try:
            from sqlalchemy_database import mysqlDatabase
            db = mysqlDatabase()
            if db.test_connection():
                logger.info("✅ Database connection test successful!")
                return True
            else:
                logger.error("❌ Database connection test failed")
                return False
        except Exception as e:
            logger.error(f"❌ Error testing connection: {e}")
            return False
    
    def run_migration(self):
        """Run database migration"""
        try:
            logger.info("🚀 Running database migration...")
            result = subprocess.run([sys.executable, 'cloud_database_setup.py'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ Database migration completed successfully")
                return True
            else:
                logger.error(f"❌ Database migration failed: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"❌ Error running migration: {e}")
            return False
    
    def deploy(self):
        """Main deployment process"""
        print("🚀 Cloud Database Deployment for Attendance Monitoring System")
        print("=" * 60)
        
        # Backup current configuration
        self.backup_current_config()
        
        try:
            # Get cloud provider
            provider = self.get_cloud_provider()
            
            # Get configuration based on provider
            if provider == "aws_rds":
                config = self.get_aws_rds_config()
            elif provider == "google_cloud":
                config = self.get_google_cloud_config()
            elif provider == "heroku":
                config = self.get_heroku_config()
            elif provider == "custom":
                config = self.get_custom_config()
            else:
                print(f"⚠️ {provider} configuration not implemented yet")
                config = self.get_custom_config()
            
            # Update .env file
            if not self.update_env_file(config):
                raise Exception("Failed to update .env file")
            
            # Test connection
            print("\n🔍 Testing database connection...")
            if not self.test_connection():
                raise Exception("Database connection test failed")
            
            # Run migration
            print("\n📊 Setting up database tables...")
            if not self.run_migration():
                raise Exception("Database migration failed")
            
            print("\n🎉 Deployment completed successfully!")
            print("✅ Your attendance monitoring system is now connected to the cloud database")
            print("✅ Anyone can access it from anywhere with an internet connection")
            
            # Show next steps
            print("\n📝 Next Steps:")
            print("1. Test your application: python app.py")
            print("2. Set up monitoring for your cloud database")
            print("3. Configure backup strategies")
            print("4. Set up alerts for connection issues")
            
        except Exception as e:
            logger.error(f"❌ Deployment failed: {e}")
            print(f"\n❌ Deployment failed: {e}")
            
            # Ask if user wants to restore backup
            restore = input("\nDo you want to restore the previous configuration? (y/n): ").lower().strip()
            if restore == 'y':
                self.restore_backup()
                print("✅ Configuration restored")
            
            return False
        
        return True

def main():
    """Main function"""
    deployer = CloudDeployer()
    success = deployer.deploy()
    
    if success:
        print("\n🎊 Congratulations! Your system is now cloud-ready!")
    else:
        print("\n💡 Tips for troubleshooting:")
        print("- Check your database credentials")
        print("- Verify network connectivity")
        print("- Ensure your IP is whitelisted")
        print("- Check cloud provider documentation")

if __name__ == "__main__":
    main() 