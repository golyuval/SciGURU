@echo off
cd %~dp0
echo Running evaluation script...
python evaluate_preference_data.py
pause 