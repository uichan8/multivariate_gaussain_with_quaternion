from gaussian_3D import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
from matplotlib.widgets import Slider

# 3D 플롯 생성
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.25)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')

# 슬라이더 추가
sz_ax_slider = plt.axes([0.1, 0.05, 0.8, 0.03])
sz_slider = Slider(sz_ax_slider, 'sz', 0, 1, valinit=0.1, valstep=0.1)

z_ax_slider = plt.axes([0.95, 0.1, 0.03, 0.8])
z_slider = Slider(z_ax_slider, 'z', 0, 10, valinit=5, valstep=0.5,orientation='vertical')

def update(val):
    ax.cla()
    ax.set_xlim([0, 10])  
    ax.set_ylim([0, 10])  
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_aspect('equal')

    user_sz = sz_slider.val
    user_z = z_slider.val

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

    x, y = np.mgrid[0:10:.1, 0:10:.1]
    pos = np.empty(x.shape + (2,)) # (x, y) 위치
    pos[:, :, 0] = x
    pos[:, :, 1] = y

    # 3변수 가우시안 정의
    rv = multivariate_normal(mean, cov)
    density_xy = np.array([[rv.pdf([x_val, y_val, user_z]) for x_val, y_val in zip(x_row, y_row)] for x_row, y_row in zip(x, y)])
    cf = ax.contourf(x, y, density_xy, vmin = 0, vmax = 0.1)


# 슬라이더 이벤트 리스너 설정
sz_slider.on_changed(update)
z_slider.on_changed(update)

plt.show()