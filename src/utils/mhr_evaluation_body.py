"""Deterministic local MHR evaluation protocol for undisclosed host subsets."""
from pathlib import Path

import numpy as np


def farthest_vertices(points, count, candidates=None):
    """Deterministic farthest-point sampling on the neutral body, no random seed."""
    candidates = np.arange(len(points)) if candidates is None else np.asarray(candidates)
    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} candidate vertices for {count} samples")
    xyz = np.asarray(points[candidates],dtype=np.float64)
    distance = np.full(len(xyz),np.inf)
    selected = []
    current = int(np.argmax(((xyz-xyz.mean(axis=0))**2).sum(axis=1)))
    for _ in range(count):
        selected.append(current)
        distance = np.minimum(distance,((xyz-xyz[current])**2).sum(axis=1))
        distance[selected] = -1
        current = int(np.argmax(distance))
    return candidates[selected]


class LocalMHRBody:
    """Use real LOD1 MHR with neutral-body fixed vertex subsets."""

    def __init__(self, path, joint_indices, device="cuda:0"):
        import torch
        self.torch = torch
        self.device = device
        self.model = torch.jit.load(str(path),map_location=device).eval()
        state = self.model.state_dict()
        n_vertices = len(state["character_torch.mesh.rest_vertices"])
        names = list(self.model.character_torch.skeleton.joint_names)
        parents = state["character_torch.skeleton.joint_parents"].cpu().numpy()
        skin = state["character_torch.linear_blend_skinning.skin_indices_flattened"].cpu().numpy()
        vertices = state["character_torch.linear_blend_skinning.vert_indices_flattened"].cpu().numpy()
        weights = state["character_torch.linear_blend_skinning.skin_weights_flattened"].cpu().numpy()
        self.roles = {}
        neutral,_ = self(np.zeros((1,204)),np.zeros(45))
        hands = []
        for side in ("left","right"):
            wrist = names.index("l_wrist" if side=="left" else "r_wrist")
            descendants = {wrist}
            changed = True
            while changed:
                previous = len(descendants)
                descendants.update(i for i,parent in enumerate(parents) if int(parent) in descendants)
                changed = len(descendants) != previous
            influence = np.zeros(n_vertices)
            chosen = np.isin(skin,list(descendants))
            np.add.at(influence,vertices[chosen],weights[chosen])
            candidates = np.flatnonzero(influence >= .5)
            self.roles[f"{side}_hand"] = farthest_vertices(neutral[0,:n_vertices],256,candidates)
            hands.extend(candidates.tolist())
        body = np.setdiff1d(np.arange(n_vertices),hands)
        self.roles["body"] = farthest_vertices(neutral[0,:n_vertices],64,body)
        self.roles["alignment"] = farthest_vertices(neutral[0,:n_vertices],27,body)
        self.vertex_count = n_vertices

    def __call__(self, model_params, shape_params, chunk=16):
        torch = self.torch
        params = np.atleast_2d(model_params)
        vertices,joints = [],[]
        flip = torch.tensor([1.,-1.,-1.],device=self.device)
        with torch.no_grad():
            for start in range(0,len(params),chunk):
                batch = torch.as_tensor(params[start:start+chunk],device=self.device,dtype=torch.float32)
                shape = torch.as_tensor(np.asarray(shape_params),device=self.device,dtype=torch.float32).reshape(1,45).expand(len(batch),-1).contiguous()
                expression = torch.zeros(len(batch),72,device=self.device)
                body,skeleton = self.model(shape,batch,expression,True)
                body = (body*flip/100.).cpu().numpy().astype(np.float64)
                joint = (skeleton[...,:3]*flip/100.).cpu().numpy().astype(np.float64)
                vertices.append(body)
                joints.append(joint)
        return np.concatenate(vertices),np.concatenate(joints)
