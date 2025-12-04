@echo off
REM Navigate to the directory
cd MTKRCE

REM Set the environment variable
set GEMINI_API_KEY=AIzaSyAdslTk5N9uP-ydjiKhIy-3ADWTATEiam0
set MONGO_DB_PASSWORD=xDW9wopR0U8oFQFH

REM Run the Python application
python app.py
python app_inceptionv3.py
python app_resnet50.py

REM Pause the window so it doesn't close immediately after finishing (helpful for seeing errors)
pause