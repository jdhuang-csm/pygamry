import time
import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from sqeis import plotting as sqp

from pygamry.dtaq import get_pstat, DtaqOcv, DtaqChrono, DtaqReadZ, GamryCOM
from pygamry.file_utils import get_decimation_index

datadir = '..\\test_data'

pstat = get_pstat()

dt_ocv = DtaqOcv()
dt_chrono = DtaqChrono('galvanostatic', leave_cell_on=True, write_mode='once')
dt_eis = DtaqReadZ('galvanostatic', leave_cell_on=True)

# # Run ocv
# dt_ocv.run(pstat, 5, 0.1, None, show_plot=True)
# print('OCV:', dt_ocv.get_ocv())
# plt.close()
# plt.pause(1)


# EIS params
freq = np.logspace(5, 3, 11)
i_rms = 4e-5

# Chrono params
i_step = i_rms * np.sqrt(2)
t_pre = 1
t_step = 5
t_sample = 1e-4

# num steps
n_steps = 10

# dt_chrono.configure_decimation('run', 10, 10)
dt_chrono.configure_decimation('write', 10, 10, 2, 0.1)

eis_data = []
chrono_data = []
dt_chrono.start_with_cell_off = False
dt_chrono.leave_cell_on = True
dt_eis.start_with_cell_off = True
dt_eis.leave_cell_on = True

run_start = time.time()
sleep_times = []
sleep_currents = []
sleep_voltages = []
for i in range(n_steps):
    if i > 0:
        dt_eis.start_with_cell_off = False  #False
        # append_to_file = True

    if i == n_steps - 1:
        dt_chrono.leave_cell_on = False

    # Run EIS
    dt_eis.run(pstat, freq, i * i_step, i_rms, 10, timeout=60)
    eis_data.append(dt_eis.z_dataframe)
    eis_end_time = time.time()
    while time.time() - eis_end_time < 2:
        sleep_currents.append(pstat.MeasureI())
        sleep_voltages.append(pstat.MeasureV())
        sleep_times.append(time.time() - run_start)

    # Configure chrono step
    dt_chrono.configure_mstep_signal(i_step * i, i_step, t_pre, t_step, t_sample, 1)
    # dt_chrono.configure_dstep_signal(i_step * i, i_step * i, i_step * (i + 1), 0, t_pre, t_step, t_sample)

    # Run chrono
    dt_chrono.run(pstat, decimate=True, result_file=None)

    # Append data with time offset
    tmp_data = dt_chrono.decimated_data_array.copy()
    tmp_data[:, 0] += dt_chrono.start_time - run_start - dt_chrono.data_time_offset  # apply time offset
    chrono_data.append(tmp_data)

    # chrono_end_time = time.time()
    # while time.time() - chrono_end_time < 2:
    #     sleep_currents.append(pstat.MeasureI())
    #     sleep_voltages.append(pstat.MeasureV())
    #     sleep_times.append(time.time() - run_start)

# Close pstat
# pstat.SetCell(GamryCOM.CellOff)
# pstat.Close()

# Plot EIS data
fig, axes = plt.subplots(1, 3, figsize=(9, 2.75))
for df in eis_data:
    sqp.plot_eis(df, axes=axes)

fig.tight_layout()

# Plot chrono results
fig, axes = plt.subplots(1, 2, figsize=(8, 3))

chrono_data = np.concatenate(chrono_data)
data_df = pd.DataFrame(chrono_data, columns=dt_chrono.cook_columns)

axes[0].scatter(data_df['Time'], data_df['Im'], s=10, alpha=0.5)
axes[0].plot(sleep_times, sleep_currents)

axes[1].scatter(data_df['Time'], data_df['Vf'], s=10, alpha=0.5)
axes[1].plot(sleep_times, sleep_voltages)

fig.tight_layout()


# line plots
fig, axes = plt.subplots(1, 2, figsize=(8, 3))

# chrono_data = np.concatenate(chrono_data)
data_df = pd.DataFrame(chrono_data, columns=dt_chrono.cook_columns)

axes[0].plot(data_df['Time'], data_df['Im'], marker='.', ms=4)
# axes[0].plot(sleep_times, sleep_currents)

axes[1].plot(data_df['Time'], data_df['Vf'], marker='.', ms=4)
# axes[1].plot(sleep_times, sleep_voltages)

fig.tight_layout()
