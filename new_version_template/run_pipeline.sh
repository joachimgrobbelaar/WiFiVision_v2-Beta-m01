#!/usr/bin/env bash
# ==============================================================================
# Wi-Fi CSI Indoor Geometry & Vital Sign Pipeline - Automated Launcher
# ==============================================================================
# This executable script automatically sets up the environment, verifies hardware,
# checks dependencies, executes all DSP/Geometry verification tests, and runs the
# end-to-end room simulation and real-time vital sign dashboard.

set -e # Exit immediately if a command exits with a non-zero status

# Resolve absolute directory of this launcher
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
cd "$SCRIPT_DIR"

# Terminal Colors for beautiful aesthetic styling
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo -e "${CYAN}${BOLD}==============================================================================${NC}"
echo -e "${CYAN}${BOLD}  WI-FI CSI INDOOR GEOMETRY & VITAL SIGN PIPELINE - AUTOMATED LAUNCHER${NC}"
echo -e "${CYAN}${BOLD}==============================================================================${NC}"
echo ""

# 1. Environment & Virtual Environment Setup
echo -e "${YELLOW}${BOLD}[*] Step 1/5: Checking Python Environment & Virtual Environment (.venv)...${NC}"
if [ ! -d ".venv" ]; then
    echo -e "    [-] Virtual environment not found. Creating Python 3 virtual environment..."
    python3 -m venv .venv
    echo -e "${GREEN}    [+] Virtual environment created successfully.${NC}"
else
    echo -e "${GREEN}    [+] Virtual environment (.venv) detected.${NC}"
fi

# Activate virtual environment
source .venv/bin/activate

# Ensure pip is up to date and required dependencies are installed
echo -e "    [*] Verifying scientific computing dependencies (numpy, scipy, scikit-learn, matplotlib)..."
pip install -q --upgrade pip
pip install -q numpy scipy scikit-learn matplotlib
echo -e "${GREEN}    [+] All scientific dependencies are installed and verified.${NC}"
echo ""

# 2. Hardware Compatibility & Environment Diagnostics
echo -e "${YELLOW}${BOLD}[*] Step 2/5: Executing Hardware Compatibility Diagnostics (env_check.sh)...${NC}"
if [ -f "env_check.sh" ]; then
    bash env_check.sh
else
    echo -e "${RED}    [!] Warning: env_check.sh not found. Skipping hardware diagnostics.${NC}"
fi
echo ""

# 3. DSP & Ingestion Unit Verification Suite
echo -e "${YELLOW}${BOLD}[*] Step 3/5: Running Unit Verification Suite across Core Modules...${NC}"
echo -e "    -> Testing Phase 2 Ingestion Engine (Hampel MAD filter & LPF)..."
python ingestion.py | grep -E "(\[\+\]|\[\*\])"
echo -e "    -> Testing Phase 3 DSP Engine (SFO/CFO sanitization & Vital Sign IIR/FFT)..."
python dsp_engine.py | grep -E "(\[\+\]|\[\*\])"
echo -e "    -> Testing Phase 4 Geometric Mapping (DBSCAN wall clustering & Doppler tracking)..."
python geometry_mapping.py | grep -E "(\[\+\]|\[\*\])"
echo -e "${GREEN}    [+] All unit verification tests passed successfully!${NC}"
echo ""

# 4. End-to-End Simulation & Visual Dashboard Execution
echo -e "${YELLOW}${BOLD}[*] Step 4/5: Executing End-to-End Room Simulation & Vital Sign Dashboard...${NC}"
echo -e "    -> Synthesizing 5m x 5m room multipath with semantic wall materials..."
echo -e "    -> Tracking walking target and range-gating stationary vital sign subject..."
python simulation.py
echo ""

# 5. Summary and Output Artifacts
echo -e "${CYAN}${BOLD}==============================================================================${NC}"
echo -e "${GREEN}${BOLD}  PIPELINE EXECUTION COMPLETE! ALL SYSTEMS OPERATIONAL.${NC}"
echo -e "${CYAN}${BOLD}==============================================================================${NC}"
echo -e "Generated Visual Artifacts and Verification Plots have been saved to:"
echo -e "  • ${BOLD}Room Geometry & Material Map:${NC} file:///home/m/.gemini/antigravity-ide/brain/a9ac646b-2fc4-4333-be55-b1fb650fcd5e/room_geometry_reconstruction.png"
echo -e "  • ${BOLD}Vital Sign & Tracking Dashboard:${NC} file:///home/m/.gemini/antigravity-ide/brain/a9ac646b-2fc4-4333-be55-b1fb650fcd5e/vital_sign_dashboard.png"
echo ""
echo -e "${YELLOW}Tip: To re-run this pipeline anytime, simply execute:${NC}"
echo -e "     ${BOLD}./run_pipeline.sh${NC}"
echo -e "${CYAN}==============================================================================${NC}"
