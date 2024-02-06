import matplotlib as mpl

mpl.use('TKAgg')
import numpy as np

from pygamry.dtaq import get_pstat, DtaqReadZ

pstat = get_pstat()

dc_amp = 0
ac_amp = 0.01
z_guess = 1

dtaq = DtaqReadZ(mode='galvanostatic', readzspeed='ReadZSpeedNorm', write_mode='continuous')
dtaq.leave_cell_on = False
dtaq.run(pstat, np.logspace(6, 3, 31), dc_amp, ac_amp, z_guess, timeout=400, show_plot=True,
         plot_interval=1, plot_type='all',
         result_file='../test_data/EIS_test.DTA')

# dtaq.write_mode = 'continuous'
# dtaq.run(pstat, np.logspace(6, 3, 31), 0.02, 0.02, 30, timeout=600, show_plot=False,
#          result_file='EIS_test_continuous.DTA')
# print('^Continuous write^\n')
#
# dtaq.write_mode = 'interval'
# dtaq.write_interval = 10
# dtaq.run(pstat, np.logspace(6, 3, 31), 0.02, 0.02, 30, timeout=600, show_plot=False,
#          result_file='EIS_test_interval.DTA')
# print('^Interval write^\n')

# print('Without plotting, continuous write')
# dtaq.run(pstat, np.logspace(6, 0, 61), 0.02, 0.02, 30, timeout=200, show_plot=False,
#          result_file='test.DTA', write_mode='continuous')
# plt.show()

# fig, ax = plt.subplots(figsize=(4, 3))
# ax.scatter(dtaq.data_array[:, 0], dtaq.data_array[:, 1], s=10, alpha=0.5)
# ax.set_xlabel('$i$')
# ax.set_ylabel('$v$')
# fig.tight_layout()
# plt.show()

# ax = sqp.plot_nyquist(dtaq.z_dataframe, set_aspect_ratio=False)
# axis_limits = get_nyquist_limits(ax, dtaq.z_dataframe['Zreal'].values + 1j * dtaq.z_dataframe['Zimag'].values)
