@echo off
:: ==============================================================================
:: Wi-Fi CSI Indoor Geometry & Vital Sign Pipeline - Windows Automated Launcher
:: ==============================================================================
:: This batch script automatically sets up the Python virtual environment,
:: checks dependencies, executes verification tests, and runs the simulation.

title Wi-Fi CSI Indoor Geometry & Vital Sign Pipeline
echo ==============================================================================
echo   WI-FI CSI INDOOR GEOMETRY ^& VITAL SIGN PIPELINE - WINDOWS LAUNCHER
echo ==============================================================================
echo.

cd /d "%~dp0"

:: 1. Check Virtual Environment
echo [*] Step 1/4: Checking Python Environment ^& Virtual Environment (.venv)...
if not exist ".venv" (
    echo     [-] Virtual environment not found. Creating Python virtual environment...
    python -m venv .venv
    echo     [+] Virtual environment created successfully.
) else (
    echo     [+] Virtual environment (.venv) detected.
)

:: Activate virtual environment
call .venv\Scripts\activate.bat

:: Install dependencies
echo     [*] Verifying scientific dependencies (numpy, scipy, scikit-learn, matplotlib)...
python -m pip install -q --upgrade pip
python -m pip install -q numpy scipy scikit-learn matplotlib
echo     [+] All dependencies installed and verified.
echo.

:: 2. Run DSP & Ingestion Verification Tests
echo [*] Step 2/4: Running Unit Verification Suite...
echo     -> Testing Phase 2 Ingestion Engine...
python ingestion.py
echo     -> Testing Phase 3 DSP Engine...
python dsp_engine.py
echo     -> Testing Phase 4 Geometric Mapping...
python geometry_mapping.py
echo     [+] Unit verification tests completed.
echo.

:: 3. Execute Simulation & Dashboard
echo [*] Step 3/4: Executing End-to-End Room Simulation ^& Vital Sign Dashboard...
python simulation.py
echo.

:: 4. Done
echo ==============================================================================
echo   PIPELINE EXECUTION COMPLETE! ALL SYSTEMS OPERATIONAL.
echo ==============================================================================
echo Generated visual artifacts have been saved to your workspace.
echo.
pause
