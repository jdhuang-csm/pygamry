import matplotlib as mpl
import os
from pygamry.dtaq import get_pstat, DtaqChrono

mpl.use('TKAgg')

# Get potentiostat
pstat = get_pstat()

# Current for each step (A)
i_init = 0  # A
i_step1 = 0.01  # A
i_step2 = 0  # A

# Time for each step (s)
t_init = 1
t_step1 = 10
t_step2 = 10

# Sample period (s)
t_sample = 1e-3

# Initialize dtaq
dtaq = DtaqChrono('galvanostatic', write_mode='once')

# Configure step
dtaq.configure_dstep_signal(i_init, i_step1, i_step2, t_init, t_step1, t_step2, t_sample)

# Configure decimation
dtaq.configure_decimation('write', 10, 10, 2, 0.1)

# Run
dtaq.run(pstat, timeout=30, decimate=True, result_file=os.path.join('../test_data', 'CHRONOP_test.DTA'))

# Plot after run
dtaq.plot_data()









