from gaussian_3D import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
from matplotlib.widgets import Slider

# 3D 플롯 생성
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 슬라이더 추가
sz_ax_slider = plt.axes([0.1, 0.05, 0.8, 0.03])
sz_slider = Slider(sz_ax_slider, 'sz', 0, 1, valinit=0.1, valstep=0.1)

def update(val):
    ax.cla()
    ax.set_xlim([0, 10])  
    ax.set_ylim([0, 10])  
    ax.set_zlim([0, 10])
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_aspect('equal')

    user_sz = sz_slider.val
    # Q = c + sxi + syj + szk
    c = (1 - user_sz**2)**0.5
    Q = torch.tensor([[c,0,0,user_sz]])
    S = torch.tensor([[2,1,1]])
    M = torch.tensor([[5,5,5]])

    C = build_covariance(S, Q) #시그마

    # 평균값
    mean = np.array([5, 5, 5])

    # 공분산 행렬
    cov = C.numpy()[0]

    x, y, z = np.mgrid[0:10:.2, 0:10:.2, 0:10:.2]
    pos = np.empty(x.shape + (3,))
    pos[:, :, :, 0] = x; pos[:, :, :, 1] = y; pos[:, :, :, 2] = z

    # 3변수 가우시안 정의
    rv = multivariate_normal(mean, cov)

    # 확률 밀도 계산
    density = rv.pdf(pos)

    # 임계값 설정
    threshold_min = 0.01
    threshold_max = 0.013

    # 임계값 이상의 데이터만 필터링
    mask = (density > threshold_min) & (density < threshold_max)

    x_filtered = x[mask]
    y_filtered = y[mask]
    z_filtered = z[mask]
    density_filtered = density[mask]

    # 3D 산점도로 플롯
    scatter = ax.scatter(x_filtered, y_filtered, z_filtered, c=density_filtered ,alpha=0.5)

# 슬라이더 이벤트 리스너 설정
sz_slider.on_changed(update)
plt.show()