import time
import numpy as np
import pandas as pd
import os
import matplotlib

matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

from pygamry.dtaq import get_pstat, DtaqOcv, DtaqChrono, DtaqReadZ

datadir = '..\\test_data'

pstat = get_pstat()

dt_ocv = DtaqOcv()
dt_chrono = DtaqChrono('galvanostatic', leave_cell_on=True, write_mode='once')
dt_eis = DtaqReadZ('galvanostatic', start_with_cell_off=True)

# Run ocv
dt_ocv.run(pstat, 5, 0.1, None, show_plot=True)
print('OCV:', dt_ocv.get_ocv())
plt.close()
plt.pause(1)

# Run chrono steps
i_rms = 0.05
i_step = i_rms * np.sqrt(2)
t_pre = 1
t_step = 50
t_sample = 1e-3

chrono_data = []
chrono_file = os.path.join(datadir, 'CHRONOP_hybrid3.DTA')

dt_chrono.configure_decimation('write', 10, 10, 2, 0.1)

# Half step up
dt_chrono.configure_mstep_signal(0, i_step, t_pre, t_step, t_sample, 1)
chrono_start = time.time()
dt_chrono.run(pstat, decimate=True, result_file=chrono_file)
tmp_data = dt_chrono.data_array.copy()
tmp_data[:, 0] += dt_chrono.start_time - chrono_start - dt_chrono.data_time_offset  # apply time offset
chrono_data.append(tmp_data)

# Full step down, half step up
dt_chrono.start_with_cell_off = False
dt_chrono.configure_dstep_signal(i_step, -i_step, 0, t_pre, t_step, t_step, t_sample)
chrono_offset = time.time() - chrono_start
dt_chrono.run(pstat, decimate=True, result_file=chrono_file, append_to_file=True)
tmp_data = dt_chrono.data_array.copy()
tmp_data[:, 0] += chrono_offset - dt_chrono.data_time_offset  # apply time offset
chrono_data.append(tmp_data)

chrono_data = np.concatenate(chrono_data)

# # Run EIS
# freq = np.logspace(5, 3, 11)
# dt_eis.run(pstat, freq, 0, i_rms, 5, timeout=60, show_plot=True)

# Plot chrono results
# fig, axes = plt.subplots(1, 2, figsize=(8, 3))

# data_df = pd.DataFrame(chrono_data, columns=dt_chrono.cook_columns)
#
# axes[0].scatter(data_df['Time'], data_df['Im'])
# # axes[0].plot(sleep_data[:, 0], sleep_data[:, 2])
#
# axes[1].scatter(data_df['Time'], data_df['Vf'])
# # axes[1].plot(sleep_data[:, 0], sleep_data[:, 1])
#
# fig.tight_layout()





