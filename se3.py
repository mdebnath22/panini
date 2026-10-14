from __future__ import annotations
import numpy as np
_EPS = 1e-10

def hat3(w):
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])

def geodesic_distance(T1, T2, trans_weight=1.0):
    R = T1[:3, :3].T @ T2[:3, :3]
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-8:
        dR = np.zeros(3)
    else:
        dR = theta / (2 * np.sin(theta)) * np.array(
            [R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
    dp = T2[:3, 3] - T1[:3, 3]
    return float(np.sqrt(np.dot(dR, dR) + trans_weight * np.dot(dp, dp)))
