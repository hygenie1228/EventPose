"""Frozen user scope for the twelve-hour encoding experiment."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
GOAL=ROOT/'_agents/research/261028/device3_encoding_goal_12h.json'
ENCODINGS=['count33','bins100','surface20']
COMBINATIONS=['rgb3_event4','event3_event4','rgb3_event3']
TRAIN_NAMES=['rgb3']+[c+'_'+e for e in ENCODINGS for c in COMBINATIONS]
EVAL_NAMES=['pre_rgb3']+TRAIN_NAMES
BASE_LABELS=dict(rgb3_event4='device3 RGB + device4 event',event3_event4='device3 event + device4 event',rgb3_event3='device3 RGB + device3 event')
LABELS=dict(pre_rgb3='원본 device3 RGB',rgb3='device3 RGB')|{c+'_'+e:BASE_LABELS[c]+' / '+e for e in ENCODINGS for c in COMBINATIONS}
METRICS=['first_frame_aligned_mpjpe_mm','first_frame_aligned_pa_mpjpe_mm','frame_aligned_mpjpe_mm','frame_aligned_pa_mpjpe_mm','pelvis_error_mm']
PAIRS=[('rgb3',c+'_'+e) for e in ENCODINGS for c in COMBINATIONS]+[(c+'_count33',c+'_'+e) for c in COMBINATIONS for e in ENCODINGS[1:]]
