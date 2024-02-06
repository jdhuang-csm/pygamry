import matplotlib as mpl
import numpy as np
import os
from pygamry.dtaq import get_pstat, DtaqOcv, HybridSequencer

mpl.use('TKAgg')

data_path = '..\\test_data'

pstat = get_pstat()

# Initialize sequences
seq = HybridSequencer()
seq.configure_decimation('write', 10, 10, 2, 0.1)

# Initialize OCV
dt_ocv = DtaqOcv()

# Current magnitudes (A)
i_init = 0  # Initial current for chrono
i_rms = 0.01  # RMS current for EIS. Also determines step size

# Chrono step time and sample period (s)
t_init = 0.5
t_step = 5
t_sample = 1e-3

file_suffix = '_Hybrid_test'


# Single measurement: triple step
# --------------------------------
# Equilibrate at OCV
print('Running OCV...')
dt_ocv.run(pstat, duration=10, t_sample=1, result_file=os.path.join(data_path, f'OCV{file_suffix}.DTA'),
           show_plot=False)

# Run hybrid measurement
seq.configure_triple_step(i_init, i_rms, t_init, t_step, t_sample, np.logspace(5, 3, 21), init_sign=-1)
seq.run(pstat, True, data_path=data_path, file_suffix=file_suffix, rest_time=1)

# Plot data after run
seq.dt_eis.plot_data()
seq.dt_chrono.plot_data()  # this will only show the final step
