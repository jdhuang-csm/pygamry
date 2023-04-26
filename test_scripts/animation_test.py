import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

from pygamry.animation import LiveAxes, LiveFigure

# fig, ax = plt.subplots()
# xdata, ydata = [], []
# pts, = ax.plot(xdata, ydata, ls='', marker='.', ms=10, alpha=0.5)
# ln, = ax.plot(xdata, ydata, c='k')
#
# rs = np.random.RandomState(3503)
#
# times = np.arange(0, 10, 0.1)
# tau = 2
# v = 1 - np.exp(-times / tau)
#
#
# def init():
#     ax.set_xlim(0, 5)
#     ax.set_ylim(0, 0.5)
#     return pts, ln
#
#
# def update(frame):
#     xdata.append(times[frame])
#     ydata.append(v[frame])
#     pts.set_data(xdata, ydata)
#
#     v_rand = 1 - np.exp(-times * rs.normal(1, 0.01) / tau)
#     ln.set_data(times, v_rand)
#
#     # resize if necessary
#     if np.max(xdata) > ax.get_xlim()[1] or np.max(ydata) > ax.get_ylim()[1]:
#         print(frame, ax.get_xlim(), ax.get_ylim())
#         ax.set_xlim(0, ax.get_xlim()[1] + 5)
#         ax.set_ylim(0, ax.get_ylim()[1] + 0.5)
#         # fig.canvas.resize_event()
#         fig.canvas.flush_events()
#
#     return pts, ln
#
#
# ani = FuncAnimation(fig, update, frames=np.arange(len(times), dtype=int),
#                     init_func=init, blit=False, repeat=False, interval=100)
#
# plt.show()

fig, axes = plt.subplots(1, 2, figsize=(8, 3))
lax1 = LiveAxes(axes[0], fixed_xlim=None, fixed_ylim=None)
lax2 = LiveAxes(axes[1], fixed_xlim=None, fixed_ylim=None)

lax1.ax.set_xlabel('time')
lax1.ax.set_ylabel('v')

lax2.ax.set_xlabel('time')
lax2.ax.set_ylabel('v')

times = []
v_meas = []
data = (times, v_meas)
tau = 2

rs = np.random.RandomState(3503)


def data_update(frame):
    return times, v_meas


def pred_update(frame):
    times_full = np.arange(0, frame * 0.1 + 0.1, 0.1)
    v_pred = 1 - np.exp(-times_full * rs.normal(1, 0.01) / tau)
    return times_full, v_pred


start = time.time()
lax1.add_line_artist('data', data_update, marker='.', ms=10, alpha=0.5, ls='', label='Data')
lax1.add_line_artist('pred', pred_update, c='k', label='Fit')

lax2.add_line_artist('pred', pred_update, c='k')

lfig = LiveFigure([lax1, lax2])

ani = lfig.run(100, 100)

for i in range(100):
    times.append(i * 0.1)
    v_meas.append(1 - np.exp(-times[i] / tau))

    plt.pause(0.1)

plt.show()

print('Run time: {:.2f} s'.format(time.time() - start))





