import comtypes
import comtypes.client as client
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TKAgg')  # seems to make plotting work in console
import time

from pygamry.utils import gamry_error_decoder
from pygamry.dtaq import get_pstat, GamryDtaqEventSink, GamryCOM, DtaqEquilibration

import sqeis.square_drt as sqdrt

plt.rcParams['font.size'] = 16

# Get potentiostat
pstat = get_pstat()

#
dtaq = DtaqEquilibration()

dtaq.set_signal(pstat, 0, 0.5, 1e-4, 30, 0.1)
dtaq.run(pstat)


print(len(dtaq.acquired_points))
# print(dtaq_sink.counts)
# print(dtaq_sink.count_times)
columns = ['T', 'Vf', 'Vu', 'Im', 'Q', 'Vsig', 'Ach', 'IERange', 'Overload', 'StopTest']

out = np.array(dtaq.acquired_points)


# dtaq.drt.plot_distribution()

plt.show()
# fig, axes = plt.subplots(3, 3, figsize=(9, 9))
#
# for i, (ax, col) in enumerate(zip(axes.ravel(), columns[1:])):
#     ax.scatter(out[:, 0], out[:, i + 1], s=10, alpha=0.5)
#     ax.set_ylabel(col)
#
# fig.tight_layout()

# plt.show(block=False)

# eqdrt = EquilibrationDRT(np.logspace(-4, 3, 71), time_precision=3)
# eqdrt.precalculate_matrices(0.1, 0.2, 20)
# eqdrt.ridge_fit(out[:, 0], eqdrt.dummy_input_signal, out[:, 1], downsample=False, hyper_l2_lambda=False)
#
# eqdrt.plot_fit(transform_time=False)
# plt.show()

# drt = sqdrt.DRT(basis_tau=np.logspace(-4, 3, 71), tau_basis_type='zga', time_precision=3)
# drt.set_zga_params()
#
# check_interval = 1
# last_check = 0
#
# fig, axes = plt.subplots(1, 2, figsize=(8, 3))
#
# axes[0].scatter(out[:, 0], out[:, 1], s=10, alpha=0.5)
#
# while last_check <= out[-1, 0]:
#     check_time = last_check + check_interval
#     check_data = out[out[:, 0] <= check_time]
#
#     times = check_data[:, 0]
#     drt.basis_tau = drt.get_tau_from_times(np.concatenate([times, times[:-1] * 10]), [0])
#     drt.ridge_fit(out[:, 0], out[:, 3], out[:, 1], downsample=False, hyper_l2_lambda=True, nonneg=True,
#               l2_lambda_0=1e-6)
#
#     v_pred = drt.predict_response(out[:, 0])
#     print(v_pred)
#     axes[0].plot(out[:, 0], v_pred, alpha=0.5)
#
#     drt.plot_distribution(ax=axes[1], alpha=0.5)
#
#     last_check += 1
#
# fig.tight_layout()
# plt.show()

#
# fig, axes = plt.subplots(1, 2, figsize=(8, 3))
# drt.plot_fit(transform_time=False, ax=axes[0])
# drt.plot_distribution(ax=axes[1])
# plt.show()


