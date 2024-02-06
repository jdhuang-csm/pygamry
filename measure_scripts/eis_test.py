import os
import numpy as np
import matplotlib.pyplot as plt
import time
from pygamry.dtaq import DtaqReadZ

from pygamry.dtaq import get_pstat, DtaqOcv

ocp_duration = 10
ocp_sample_period = 1
suffix = 'Test'
data_path = '.'
eis_max_freq = 1e5
eis_min_freq = 1e2
eis_ppd = 10
z_guess = 30


# Get pstat
pstat = get_pstat()

# Configure OCV
# Write to file every minute
ocv = DtaqOcv(write_mode='interval', write_interval=int(60 / ocp_sample_period))

# Configure EIS
# Write continuously
eis = DtaqReadZ(mode='potentiostatic', readzspeed='ReadZSpeedNorm', write_mode='continuous')


# Run OCV
# -------------------
print('Running OCV')
ocv_file = os.path.join(data_path, f'OCP_{suffix}.DTA')
ocv.run(pstat, ocp_duration, ocp_sample_period, show_plot=True, result_file=ocv_file)
print('OCV done\n')

plt.close()
time.sleep(1)

# Get measured OCV for EIS
V_oc = np.mean(ocv.dataframe['Vf'].values[-10:])  # average last 10 values
print('OCV: {:.3f} V'.format(V_oc))

# Run EIS
# -------------------
# Get frequencies to measure
num_decades = np.log10(eis_max_freq) - np.log10(eis_min_freq)
num_freq = int(eis_ppd * num_decades) + 1
eis_freq = np.logspace(np.log10(eis_max_freq), np.log10(eis_min_freq), num_freq)

# Determine DC voltage
V_dc = V_oc + 0.014

print('Running EIS')
eis_file = os.path.join(data_path, f'EIS_{suffix}.DTA')
eis.run(pstat, eis_freq, V_dc, 0.01, z_guess, timeout=60,
        show_plot=True, plot_interval=1, plot_type='all',
        result_file=eis_file)
print('EIS done\n')

plt.close()
time.sleep(1)
