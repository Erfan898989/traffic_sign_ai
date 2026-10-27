@echo off
cd /d "%~dp0"

echo ========================================
echo       CAT vs DOG AI
echo ========================================
echo.
echo Starting application...
echo.

python -m streamlit run streamlit_app.py --server.headless true

pause