@echo off
REM ---- 1: finde Python, lege ggf. venv an -------------------------------
IF NOT EXIST ".venv" (
    echo [+] Creating virtual environment ...
    py -3 -m venv .venv
)

echo [+] Activating virtual environment ...
call ".venv\Scripts\activate.bat"

REM ---- 2: installiere/aktualisiere Pakete -------------------------------
echo [+] Installing/Updating Python packages ...
pip install --upgrade pip
pip install -r requirements.txt

REM ---- 3: Starte Streamlit-Suite ---------------------------------------
echo [+] Launching Streamlit UI ...
python -m streamlit run edge_suite\ui_main.py --server.headless false
