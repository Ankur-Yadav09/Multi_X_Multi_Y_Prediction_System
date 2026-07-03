@echo off
title YP OLF1 whatif Dashboard

:: Go to project folder
cd /d "D:\Multi X Multi Y Optimization\Multi X Multi Y Optimization V0\Scripts"

:: Activate the virtual environment (FD_venv)
call "D:\Ethane Cracking project\Models\Yanpet_OLF1_Virtual_env\Yanpet_OLF1_Venv\Scripts\activate.bat"

:: Run Streamlit
streamlit run dashboard_streamlit_copy.py

pause
