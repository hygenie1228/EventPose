"""Freeze Track 2 split and prepare independent RGB/SMPL-X training caches."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from multihmr.track2_data import plan, prepare_sample, finalize_indices


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--sample')
    parser.add_argument('--iterations', type=int, default=350)
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--finalize', action='store_true')
    args = parser.parse_args()
    if args.plan:
        manifest = plan()
        print({k:manifest[k] for k in ['train', 'test', 'training', 'validation']})
    if args.sample:
        prepare_sample(args.sample, args.iterations, args.batch_size, args.limit)
    if args.finalize:
        finalize_indices()
