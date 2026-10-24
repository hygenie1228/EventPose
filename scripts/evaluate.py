"""Dispatch the requested model."""
import argparse
from pathlib import Path
import runpy
import sys
p=argparse.ArgumentParser(add_help=False)
p.add_argument("--method",choices=["multihmr","nlf"],default="multihmr")
a,rest=p.parse_known_args()
sys.argv=[sys.argv[0],*rest]
runpy.run_path(str(Path(__file__).resolve().parent/f"evaluate_{a.method}.py"),run_name="__main__")
