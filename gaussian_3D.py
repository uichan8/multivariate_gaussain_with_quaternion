import torch
import numpy as np

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3))

    r = q[:, 0]
    x = q[:, 1] 
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float)
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def build_covariance(s, r, modifier = None):
    """
    나중에 midifier로 조정
    https://github.com/graphdeco-inria/gaussian-splatting/blob/2eee0e26d2d5fd00ec462df47752223952f6bf4e/scene/gaussian_model.py#L29
    """
    L = build_scaling_rotation(s, r)
    actual_covariance = L @ L.transpose(1, 2)
    return actual_covariance

class gaussian_3D:

    def __init__(self,dim = 1, device = 'cuda'):
        self.device = device

        # set param
        self.Q = torch.randn((dim, 4)) # 회전자
        self.S = torch.randn((dim, 3)) # 스케일러
        self.M = torch.randn((dim, 3)) # 평균(bias)

        #update formula
        self.C = build_covariance(self.S, self.Q) #대문자 시그마
        self.inv_C = torch.inverse(self.C)
        self.det_C = torch.det(self.C)

        def gaussian(x,y,z):
            x_minus_mu = torch.tensor([x - self.M[:,0], y - self.M[:,1], z - self.M[:,2]])
            return torch.exp(-0.5 * x_minus_mu @ self.inv_C @ x_minus_mu) # 주의 트렌스 포즈 안붙임
        
        self.fomula = gaussian

if __name__ == "__main__":
    G = gaussian_3D(device='cpu')
    print(G.fomula(0,0,0))

    





