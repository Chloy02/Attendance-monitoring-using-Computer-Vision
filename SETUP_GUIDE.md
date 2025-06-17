# Attendance Monitoring System - Linux Setup Guide

## Overview
This guide will help you set up the Student Attendance Monitoring System on Linux (Fedora/Ubuntu/Debian). The system uses computer vision for face detection, recognition, and emotion analysis with MySQL database storage.

## Prerequisites
- Linux system (Fedora/Ubuntu/Debian)
- Python 3.10 or higher
- MySQL Server
- Webcam or video file for testing

## Step-by-Step Setup

### 1. System Dependencies Installation

#### For Fedora:
```bash
sudo dnf update
sudo dnf install python3-dev python3-pip python3-venv libmysqlclient-dev build-essential
sudo dnf install libgl1-mesa-glx  # For OpenCV
sudo dnf install community-mysql-server
```

#### For Ubuntu/Debian:
```bash
sudo apt update
sudo apt install python3-dev python3-pip python3-venv libmysqlclient-dev build-essential
sudo apt install libgl1-mesa-glx  # For OpenCV
sudo apt install mysql-server
```

### 2. Start and Enable MySQL Service
```bash
sudo systemctl start mysqld
sudo systemctl enable mysqld
```

### 3. Set Up MySQL Database

#### Step 3a: Log into MySQL
```bash
# If no password is set (default on many systems)
mysql -u root

# If you have a password set
mysql -u root -p
```

#### Step 3b: Create Database and Tables
```sql
-- Create the database
CREATE DATABASE class_records;

-- Use the database
USE class_records;

-- The application will automatically create the required tables:
-- - attendance_records
-- - emotion_records
-- But you can verify they exist after running the app once

-- Exit MySQL
EXIT;
```

#### Step 3c: Verify Database Creation
```bash
mysql -u root -p -e "SHOW DATABASES;"
mysql -u root -p -e "USE class_records; SHOW TABLES;"
```

### 4. Clone/Download the Project
```bash
# Navigate to your desired directory
cd ~/Documents/College_code/projects/
# Copy your project files here
```

### 5. Create and Activate Virtual Environment
```bash
cd Attendance-monitoring-using-Computer-Vision
python3 -m venv .venv
source .venv/bin/activate
```

### 6. Fix Requirements.txt for Linux
The original requirements.txt has macOS-specific packages. Replace the content with:

```txt
## EMOTION DETECTION

# Core dependencies
numpy==1.23.5
tensorflow==2.15.0
torch==2.0.1
torchvision==0.15.2
torchaudio==2.0.2

# Deep learning models
mtcnn==1.0.0
keras==2.15.0
scipy==1.10.1
h5py==3.10.0
tensorboard>=2.15,<2.16

# Face detection and recognition
ultralytics==8.1.0
opencv-python==4.11.0.86
deepface==0.0.86
facenet-pytorch==2.5.3

# Face tracking (StrongSORT)
filterpy==1.4.5

# Database (MySQL)
mysql-connector-python==8.0.33

# Miscellaneous utilities
pandas==2.0.1
pillow==9.5.0

# Distance metrics
scikit-learn==1.6.1

# Additional dependencies
SQLAlchemy==2.0.23
pymysql==1.1.0
python-dotenv==1.0.0
cryptography==41.0.7
gdown==4.7.1
easydict==1.13
```

### 7. Install Python Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 8. Install Missing Dependencies (if needed)
```bash
# If you encounter missing module errors, install these:
pip install easydict
```

### 9. Create Environment Configuration File
```bash
# Create .env file
cat > .env << EOF
DB_USERNAME=root
DB_PASSWORD=
DB_HOST=localhost
DB_NAME=class_records
EOF
```

**Note:** If your MySQL root user has a password, replace the empty `DB_PASSWORD=` with your actual password.

### 10. Download Missing Model Files

#### Download FaceNet Keras Weights:
```bash
# Method 1: Using curl (recommended)
curl -L -o facenet_keras_weights.h5 https://github.com/nyoki-mtl/keras-facenet/raw/master/model/facenet_keras.h5

# Method 2: Using wget
wget https://github.com/nyoki-mtl/keras-facenet/raw/master/model/facenet_keras.h5 -O facenet_keras_weights.h5

# Method 3: Manual download
# Visit: https://github.com/nyoki-mtl/keras-facenet/blob/master/model/facenet_keras.h5?raw=true
# Download and save as 'facenet_keras_weights.h5' in project root
```

**Verify the download:**
```bash
ls -lh facenet_keras_weights.h5
# Should show ~88MB file size
```

### 11. Verify Required Files
```bash
# Check if all required files are present
ls -la *.h5 *.pt
ls -la encodings/
ls -la .env
```

Expected files:
- `facenet_keras_weights.h5` (~88MB)
- `yolov8n-face.pt` (~6MB)
- `yolov8n.pt` (~6MB)
- `encodings/encodings.pkl`
- `.env`

### 12. Test the Application

#### Test with Webcam:
```bash
source .venv/bin/activate
python main.py --source 0
```

#### Test with Video File:
```bash
source .venv/bin/activate
python main.py --source ./path/to/your/video.mp4
```

#### Test with Help:
```bash
source .venv/bin/activate
python main.py --help
```

### 13. Application Usage

#### Command Line Options:
- `--source 0`: Use webcam
- `--source /path/to/video.mp4`: Use video file
- `--conf 0.25`: Confidence threshold (default: 0.25)
- `--iou 0.45`: IoU threshold (default: 0.45)
- `--save-vid`: Save output video

#### Example Commands:
```bash
# Run with webcam
python main.py --source 0

# Run with video file
python main.py --source ./test/Class.mp4

# Run with custom confidence
python main.py --source 0 --conf 0.3

# Run and save output video
python main.py --source 0 --save-vid
```

### 14. Training New Faces (Optional)

If you want to add new people to the system:

1. Create a directory structure:
```bash
mkdir -p Faces/person_name
# Add multiple photos of the person in Faces/person_name/
```

2. Run the training script:
```bash
source .venv/bin/activate
python train.py
```

**Note:** The `train.py` script expects face images in a specific directory structure. Modify the `face_data` path in `train.py` to point to your face images directory.

### 15. Troubleshooting

#### Common Issues:

1. **MySQL Connection Error:**
   - Check if MySQL is running: `sudo systemctl status mysqld`
   - Verify database exists: `mysql -u root -p -e "SHOW DATABASES;"`
   - Check .env file credentials

2. **Missing Model Files:**
   - Ensure `facenet_keras_weights.h5` is ~88MB
   - Check if YOLO model files are present

3. **Import Errors:**
   - Activate virtual environment: `source .venv/bin/activate`
   - Install missing packages: `pip install package_name`
   - Common missing package: `pip install easydict`

4. **Camera Access Issues:**
   - Check camera permissions
   - Try different camera index: `python main.py --source 1`

5. **Performance Issues:**
   - The system runs on CPU by default
   - For better performance, consider GPU setup

### 16. System Requirements

#### Minimum Requirements:
- **CPU:** Multi-core processor (Intel i5 or equivalent)
- **RAM:** 8GB minimum, 16GB recommended
- **Storage:** 10GB free space
- **Camera:** USB webcam or built-in camera

#### Recommended Requirements:
- **CPU:** Intel i7 or AMD Ryzen 7 or better
- **RAM:** 16GB or more
- **GPU:** NVIDIA GPU with CUDA support (optional, for better performance)
- **Storage:** SSD with 20GB free space

### 17. Database Schema

The system creates two main tables:

1. **attendance_records:**
   - id (Primary Key)
   - name (Student name)
   - status (Attendance status)
   - timestamp (Date and time)

2. **emotion_records:**
   - id (Primary Key)
   - attendance_id (Foreign Key to attendance_records)
   - emotion (Detected emotion)
   - face_encoding (Face embedding)

### 18. Features

- **Face Detection:** Using YOLOv8
- **Face Recognition:** Using FaceNet embeddings
- **Emotion Detection:** Real-time emotion analysis
- **Attendance Tracking:** Automatic attendance logging
- **Database Storage:** MySQL integration
- **Video Processing:** Support for video files and live camera

### 19. Output

The system provides:
- Real-time face detection and recognition
- Emotion analysis results
- Attendance records in MySQL database
- Optional video recording with annotations

## Success!

Your Attendance Monitoring System is now fully functional on Linux! 

The system will:
- Detect faces in real-time
- Recognize known individuals
- Analyze emotions
- Log attendance to the database
- Display results with bounding boxes and labels

Press 'q' to quit the application when running. 