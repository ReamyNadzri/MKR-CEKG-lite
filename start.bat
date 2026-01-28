@echo off
REM Navigate to the directory
cd MTKRCE

REM Set the environment variable
set GEMINI_API_KEY=AIzaSyCBSHWxuDXJG64Iyj0uQxMgRzEEx9t9ckE
set MONGO_DB_PASSWORD=xDW9wopR0U8oFQFH

REM Run the Python application
python app.py


REM Pause the window so it doesn't close immediately after finishing (helpful for seeing errors)
pause