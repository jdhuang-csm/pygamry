import time
import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

from pygamry.dtaq import get_pstat, DtaqOcv, DtaqChrono

datadir = '..\\test_data'

pstat = get_pstat()

dt_ocv = DtaqOcv()
dt_chrono = DtaqChrono('galvanostatic', leave_cell_on=True)

# # Run ocv
# dt_ocv.run(pstat, 5, 0.1, None, show_plot=True)
# print('OCV:', dt_ocv.get_ocv())
# plt.close()
# plt.pause(1)

# Run chrono steps
# i_rms = 3e-5
i_step = 5e-5  # i_rms * np.sqrt(2)
t_pre = 1
t_step = 5
t_sample = 1e-4
n_steps = 2

dt_chrono.configure_decimation('run', 10, 10)

chrono_data = []
chrono_file = os.path.join(datadir, 'CHRONOP_staircase.DTA')
dt_chrono.start_with_cell_off = True
dt_chrono.leave_cell_on = True
append_to_file = False

chrono_start = time.time()
for i in range(n_steps):
    if i > 0:
        dt_chrono.start_with_cell_off = False
        append_to_file = True

    if i == n_steps - 1:
        dt_chrono.leave_cell_on = False

    # Configure next step
    dt_chrono.configure_mstep_signal(i_step * i, i_step, t_pre, t_step, t_sample, 1)

    # Run
    dt_chrono.run(pstat, decimate=True, result_file=chrono_file, append_to_file=append_to_file)

    # Append data with time offset
    tmp_data = dt_chrono.data_array.copy()
    tmp_data[:, 0] += dt_chrono.start_time - chrono_start - dt_chrono.data_time_offset  # apply time offset
    chrono_data.append(tmp_data)

# Plot chrono results
fig, axes = plt.subplots(1, 2, figsize=(8, 3))

chrono_data = np.concatenate(chrono_data)
data_df = pd.DataFrame(chrono_data, columns=dt_chrono.cook_columns)

axes[0].scatter(data_df['Time'], data_df['Im'], s=10, alpha=0.5)

axes[1].scatter(data_df['Time'], data_df['Vf'], s=10, alpha=0.5)

fig.tight_layout()
