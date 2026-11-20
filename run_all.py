#!/usr/bin/env python3
"""Run the full analysis pipeline in order.

Usage:
    python run_all.py

Reads the anonymised survey at `data/survey_anonymised.csv` (see data/README.md).
Outputs (figures/tables) are written to each script's working directory.
"""
import subprocess, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "survey_anonymised.csv")

PIPELINE = [
    "src/predictive_analysis.py",
    "src/advanced_analysis.py",
    "src/student_profiling.py",
    "src/predict_ethical_concerns.py",
    "src/generate_journal_assets.py",
    "src/first/predict_engagement.py",
    "src/compile_report.py",          # run last: assembles tables into a report
]

def main():
    if not os.path.exists(DATA):
        sys.exit("ERROR: data/survey_anonymised.csv not found. See data/README.md.")
    for rel in PIPELINE:
        path = os.path.join(HERE, rel)
        print(f"\n{'='*60}\nRunning {rel}\n{'='*60}")
        r = subprocess.run([sys.executable, path], cwd=os.path.dirname(path))
        if r.returncode != 0:
            print(f"WARNING: {rel} exited with code {r.returncode}")

if __name__ == "__main__":
    main()
