"""Run pose visualization."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parents[1]/"src/utils/visualization.py"),run_name="__main__")
