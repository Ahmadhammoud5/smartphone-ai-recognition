@echo off

call "C:\Users\user\anaconda3\Scripts\activate.bat" base

cd /d "C:\Users\user\Desktop\Smartphone_AI_Project\app"

python -m streamlit run streamlit_app.py

pause