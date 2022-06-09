import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import random

alpha1 = 1.2
alpha2 = 0.1
beta = 10
gamma = 10

def du(u, v):
    return alpha1/(1+ v ** beta) -u

def dv(u, v):
    return alpha2/(1+ u ** gamma) -v
def vector(state, t):
    u, v = state
    nextu = du(u, v)
    nextv = dv(u, v)
    return nextu, nextv

#初期値は不安定固定点

u0 = 0 + random.uniform(0, 0.001)
v0 = 0 + random.uniform(0, 0.001)

t = np.arange(0.0, 10.0, 0.01)
v = integrate.odeint(vector, [u0, v0], t)

u_vec = v[:,0]
v_vec = v[:,1]

plt.figure()
plt.plot(u_vec, v_vec)
plt.xlabel("Repressor U")
plt.ylabel("Repressor V")
plt.title(f"Repressor Concentration alpha1={alpha1} alpha2={alpha2}")

#作図範囲を設定するパラメータ
p = 2.0

umax, umin = u_vec.max() + p, v_vec.min()- p
vmax, vmin = u_vec.max() + p, v_vec.min() - p

U, V = np.meshgrid(np.arange(umin, vmax, 0.1), np.arange(umin, vmax, 0.1))

dU = du(U, V)
dV = dv(U, V)

plt.quiver(U, V, dU, dV)
plt.scatter(u_vec[-1], v_vec[-1], color="pink", s= 100)
plt.contour(U, V, dV, levels=[0], colors="Blue")
plt.contour(U, V, dU, levels=[0], colors="Red")
plt.xlim([0, umax])
plt.ylim([0, vmax])
plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=18
)
plt.grid()
plt.savefig(f'image/a1-{alpha1}_a2-{alpha2}_b-{beta}_g-{gamma}.png')
#plt.show()