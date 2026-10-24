"""Thin experiment entry point."""
from pathlib import Path
import runpy
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT))
_source=ROOT/'scripts/evaluate_multihmr.py'
if __name__=="__main__":
    runpy.run_path(str(_source),run_name="__main__")
else:
    globals().update(runpy.run_path(str(_source),run_name=__name__))
