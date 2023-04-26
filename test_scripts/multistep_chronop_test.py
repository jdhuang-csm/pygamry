import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

from pygamry.dtaq import get_pstat, DtaqChrono

pstat = get_pstat()


dtaq = DtaqChrono('galvanostatic', write_mode='once', write_precision=6,
                  leave_cell_on=True)

#
i_step = 1e-4
n_steps = 1

data_arrays = []
sleep_times = []
sleep_currents = []
sleep_voltages = []
start_time = time.time()

dtaq.configure_decimation('write', 10, 10, 2, 0.05)
for i in range(n_steps):
    # dtaq.configure_mstep_signal(i_step * i, i_step, 0.1, 1, 1e-4, 1)
    dtaq.configure_dstep_signal(0, i_step, 0, 0.1, 10, 10, 1e-4)

    if i > 0:
        dtaq.start_with_cell_off = False

    # if i == n_steps - 1:
    #     dtaq.leave_cell_on = False

    print(i, dtaq.start_with_cell_off, dtaq.leave_cell_on)

    step_time_offset = time.time() - start_time
    print('Step time offset: {:.2f} s'.format(step_time_offset))
    step_start = time.time()
    # for n in range(1):

    dtaq.run(pstat,
             result_file=f'multi_chronop_test_{i}.DTA',
             decimate=True,
             repeats=1)

    step_end_time = time.time()
    print("step time: {:.3f}".format(step_end_time - step_start))
    # if pstat.TestIsOpen():
    #     while time.time() - step_end_time < 2:
    #         sleep_currents.append(pstat.MeasureI())
    #         sleep_voltages.append(pstat.MeasureV())
    #         sleep_times.append(time.time() - start_time)

    # tmp_data = dtaq.data_array.copy()
    # tmp_data[:, 0] += step_time_offset
    # data_arrays.append(tmp_data)

pstat.Close()

print('Total time: {:.2f} s'.format(time.time() - start_time))


# data_array = np.concatenate(data_arrays)
# data_df = pd.DataFrame(data_array, columns=dtaq.cook_columns)
#
# # sleep_data = np.array([sleep_times, sleep_voltages, sleep_currents]).T
#
# fig, axes = plt.subplots(1, 2, figsize=(8, 3))
#
# axes[0].scatter(data_df['Time'], data_df['Im'])
# # axes[0].plot(sleep_data[:, 0], sleep_data[:, 2])
#
# axes[1].scatter(data_df['Time'], data_df['Vf'])
# # axes[1].plot(sleep_data[:, 0], sleep_data[:, 1])
#
# fig.tight_layout()
