@echo off
title AI Clipper
cd /d "%~dp0"
 
if "%GROQ_API_KEY%"=="" (
    echo GROQ_API_KEY belum ter-set sebagai environment variable permanen.
    set /p GROQ_API_KEY="Masukkan Groq API Key kamu (gsk_...): "
)
 
echo.
echo Menjalankan AI Clipper...
echo Browser akan terbuka otomatis dalam beberapa detik.
echo (Biarkan jendela ini tetap terbuka selama memakai aplikasi)
echo.
 
if exist app.py (
    python app.py
) else (
    python web_app.py
)
 
echo.
echo Server berhenti. Tekan tombol apa saja untuk menutup jendela ini.
pause >nul