import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import sys
from charge import *
from field import *

sys.path.insert(1, os.path.join(sys.path[0], '..'))

width = 4
lim = 50e-9
grid_size = 150

charge = (OscillatingCharge(direction=(1, 0), start_position=(-4e-9, 0), amplitude=4e-9, speed=0.6 * c))
t = 0
x, y = np.meshgrid(np.linspace(-lim, lim, grid_size), np.linspace(-lim, lim, grid_size), indexing='ij')
field = Field(charge)

E_total = field.calculate_E( t=t, x=x, y=y)

fig, ax = plt.subplots(figsize=(width, width))
u = E_total[0]
v = E_total[1]
im = plt.imshow(np.sqrt(u ** 2 + v ** 2).T, origin='lower', extent=[-lim, lim, -lim, lim], vmax=7)
plt.xticks([-lim, -lim / 2, 0, lim / 2, lim], [-50, -25, 0, 25, 50])
plt.yticks([-lim, -lim / 2, 0, lim / 2, lim], [-50, -25, 0, 25, 50])
im.set_norm(mpl.colors.LogNorm(vmin=1e5, vmax=1e8))
r = np.power(np.add(np.power(u, 2), np.power(v, 2)), 0.5)  # arrows
cb = fig.colorbar(im, fraction=0.046, pad=0.04)
cb.ax.set_ylabel('$|\mathbf{E}|$ [N/C]', rotation=270, labelpad=12)
plt.xlabel('$x$ [nm]')
plt.ylabel('$y$ [nm]')

Q = plt.quiver(x, y, u / r, v / r, scale_units='xy')  # arrows

pos = ax.scatter(-2e-9, 0, s=5, c='red', marker='o')


def _update_animation(frame):
    text = f"\rProcessing frame {frame + 1}/{n_frames}."
    sys.stdout.write(text)
    sys.stdout.flush()
    t = frame * dt
    E_total = field.calculate_E(t=t, x=x, y=y)
    u = E_total[0]
    v = E_total[1]
    im.set_data(np.sqrt(u ** 2 + v ** 2).T)
    r = np.power(np.add(np.power(u, 2), np.power(v, 2)), 0.5)  # arrows
    Q.set_UVC(u / r, v / r)  # arrows
    pos.set_offsets((charge.xpos(t), 0))
    return im


def _init_animate():
    pass


n_frames = 36
dt = 2 * np.pi / charge.w / n_frames
anim = FuncAnimation(fig, _update_animation, frames=n_frames, blit=False, init_func=_init_animate)

plt.show()
