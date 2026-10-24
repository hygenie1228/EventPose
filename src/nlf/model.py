"""NLF checkpoint loading and metric SMPL-X output conversion."""
from pathlib import Path
import torch
import torchvision

ROOT=Path(__file__).resolve().parents[2]

def load_model(checkpoint=None,device='cuda:0'):
    path=Path(checkpoint) if checkpoint else ROOT/'data/pretrained/nlf/nlf_l_multi.torchscript'
    # map_location also remaps scripted tensor attributes, unlike Module.to alone.
    return torch.jit.load(str(path),map_location=device).eval()

def regressed_joints_m(vertices_mm,regressor):
    """Normalize NLF's fitted vertex output; empty outputs may use subset topology."""
    if len(vertices_mm)==0:return torch.empty((0,55,3),device=regressor.device)
    assert vertices_mm.shape[-2:]==(10475,3)
    return regressor@vertices_mm.float()/1000

def standard_translation_m_v020(prediction):
    """Recover standard camera-space SMPL-X transl from the v0.2.0 release.

    This release fits centered points and adds their mean back to vertices/joints,
    but its raw `trans` field keeps the centered fitting translation. Apply this
    only to v0.2.0-derived checkpoints, not arbitrary newer upstream versions.
    """
    translations=[]
    for raw,vertices,joints in zip(prediction['trans'],prediction['vertices3d_nonparam'],
                                   prediction['joints3d_nonparam']):
        if len(raw)==0:translations.append(raw)
        else:
            center=torch.cat([vertices,joints],dim=1).mean(dim=1)/1000
            translations.append(raw+center)
    return translations
