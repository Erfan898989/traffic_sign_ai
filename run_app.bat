@echo off
cd /d "%~dp0"

echo ========================================
echo       Traffic Sign Ai
echo ========================================
echo.
echo Starting application...
echo.

python -m streamlit run streamlit_app.py --server.headless true

pause