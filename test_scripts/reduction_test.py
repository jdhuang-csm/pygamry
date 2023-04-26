import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

from pygamry.dtaq import get_pstat
from pygamry.reduction import DtaqReduction





pstat = get_pstat()

dtaq = DtaqReduction(config_file)

dtaq.run(pstat, 10, 0.1, '.', show_plot=True)

plt.show()
