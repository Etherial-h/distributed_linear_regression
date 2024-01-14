import numpy as np
import matplotlib.pyplot as plt

n = 6
f = 1
d = 2
T = 500
att = "random" # "random
x_tr = np.ones(d)
msg = np.zeros([n, d])
A = [[1., 0.],
     [0.8, 0.5],
     [0.5, 0.8],
     [0., 1.],
     [-0.5, 0.8],
     [-0.8, 0.5]]
B = [0.9108, 1.3349, 1.3376, 1.0033, 0.2142, -0.3615]
N = [-0.0892, 0.0349, 0.0376, 0.0033, -0.0858, -0.0615]
A = np.array(A)
B = np.array(B)
N = np.array(N)
solution = np.array([1.0780, 0.9825])
x0 = np.array([-0.0085, -0.5643])


def dist(v1, v2):
    return np.linalg.norm(v1-v2)


def CGE(msg, n, f):
    norm = np.linalg.norm(msg, axis=1)
    index = np.argsort(norm)[0:n-f]
    return msg[index].mean(axis=0)


def CWTM(msg, n, f):
    msg.sort(axis=0)
    return msg[f:-(f), :].sum(axis=0)


def attack(msg, index, att):
    if att == "gr":
        for i in index:
            msg[i] *= -1
    elif att == "random":
        for i in index:
            msg[i] = np.random.normal(0., 200, 2)


def proj(x):
    for i in range(x.size):
        x[i] = max(min(x[i], 1000), -1000)

eta0 = 1.5
_A = A[1:, :]
_B = B[1:]
x_gd = x0.copy()
dist_gd = np.zeros(T+1) + dist(x0, solution)
diff = _A @ x0 - _B
loss_gd = np.zeros(T+1) + diff@diff
for t in range(T):
    eta = eta0 / (t+1)
    msg = 2 * A * (A@x_gd-B)[:, np.newaxis]
    attack(msg, [0,], att)
    x_gd -= eta * msg.mean(axis=0)
    proj(x_gd)
    dist_gd[t + 1] = min(1., dist(x_gd, solution))
    diff = _A @ x_gd - _B
    loss_gd[t+1] = min(1., diff@diff)
print(x_gd, dist_gd[-1])


x_cge = x0.copy()
dist_cge = np.zeros(T+1) + dist(x0, solution)
diff = _A @ x0 - _B
loss_cge = np.zeros(T+1) + diff@diff
for t in range(T):
    eta = eta0 / (t+1)
    msg = 2 * A * (A@x_cge-B)[:, np.newaxis]
    attack(msg, [0,], att)
    x_cge -= eta * CGE(msg, n, f)
    proj(x_cge)
    dist_cge[t+1] = min(1., dist(x_cge, solution))
    diff = _A @ x_cge - _B
    loss_cge[t+1] = min(1., diff@diff)
print(x_cge, dist_cge[-1])


x_cwtm = x0.copy()
dist_cwtm = np.zeros(T+1) + dist(x0, solution)
diff = _A @ x0 - _B
loss_cwtm = np.zeros(T+1) + diff@diff
for t in range(T):
    eta = eta0 / (t+1)
    msg = 2 * A * (A@x_cwtm-B)[:, np.newaxis]
    attack(msg, [0,], att)
    x_cwtm -= eta * CWTM(msg, n, f)
    proj(x_cwtm)
    dist_cwtm[t+1] = min(1., dist(x_cwtm, solution))
    diff = _A @ x_cwtm - _B
    loss_cwtm[t+1] = min(1., diff@diff)
print(x_cwtm, dist_cwtm[-1])


x_free = x0.copy()
dist_free = np.zeros(T+1) + dist(x0, solution)
diff = _A @ x0 - _B
loss_free = np.zeros(T+1) + diff@diff
for t in range(T):
    eta = eta0 / (t+1)
    grad = 2 * _A.transpose()@(_A@x_free-_B)
    x_free -= eta * grad
    proj(x_free)
    dist_free[t+1] = min(1., dist(x_free, solution))
    diff = _A @ x_free - _B
    loss_free[t+1] = min(1., diff@diff)
print(x_free, dist_free[-1])

import matplotlib
matplotlib.rc('font', family='Microsoft JhengHei')
fig, axs = plt.subplots(1, 2)
fig.set_size_inches((10, 4))
index = np.arange(0, 501)
FONT_SIZE = 23
TICK_SIZE = 16
# fig.subplots_adjust(wspace=2., hspace=0)
for column in range(2):
    axs[column].grid('True')
    axs[column].tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    axs[column].set_xlabel('迭代次数', fontsize=16)
    axs[column].set_xlim(0, 500)
    if column == 0:
        # axs[column].set_yscale('log')
        axs[column].set_ylim(0, 1.01)
        axs[column].set_ylabel("目标函数值", fontsize=16)
        axs[column].plot(index, loss_free, label="无攻击", marker='o', markevery=(0, 50))
        axs[column].plot(index, loss_cge, label="CGE", marker='v', markevery=(25, 50))
        axs[column].plot(index, loss_cwtm, label="CWTM")
        axs[column].plot(index, loss_gd, label="GD")
    else:
        axs[column].set_ylim(0, 1.01)
        axs[column].set_ylabel("距离", fontsize=16)
        axs[column].plot(index, dist_free, label="无攻击", marker='o', markevery=(0, 50))
        axs[column].plot(index, dist_cge, label="CGE", marker='v', markevery=(25, 50))
        axs[column].plot(index, dist_cwtm, label="CWTM")
        axs[column].plot(index, dist_gd, label="GD")

plt.subplots_adjust(wspace=0.3)
lines, labels = fig.axes[-1].get_legend_handles_labels()
plt.tight_layout()
plt.legend(lines, labels, loc='best', ncol=1, fontsize=16)
plt.savefig('./{}.pdf'.format(att), dpi=fig.dpi)

plt.show()
plt.cla()
