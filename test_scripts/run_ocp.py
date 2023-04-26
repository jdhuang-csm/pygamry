import os
import matplotlib as mpl
mpl.use('TKAgg')

from pygamry.dtaq import DtaqOcv, get_pstat

pstat = get_pstat()

dtaq = DtaqOcv(write_mode='continuous', write_interval=10)

datadir = '..\\test_data'
result_file = os.path.join(datadir, 'OCP_test.DTA')
dtaq.run(pstat, 10, 1, show_plot=False, result_file=result_file)

# Dlot data after running
dtaq.plot_data()
