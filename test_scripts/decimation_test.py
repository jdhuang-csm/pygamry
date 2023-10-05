import numpy as np
import os
import sys
import argparse

rf_dir = 'C:\\Users\\zmap\\python\\pygamry\\measure_scripts'
if rf_dir not in sys.path:
    sys.path.append(rf_dir)

from pygamry.dtaq import get_pstat, GamryCOM, DtaqChrono
from pygamry.sequencers import HybridSequencer

import run_functions as rf
import arg_config as argc


pstat = get_pstat()

# Initialize dtaq
dtaq = DtaqChrono('galv', write_mode='once', write_precision=6)


dtaq.configure_decimation('write', 20, 10, 2, None)

# Configure step signal
dtaq.configure_triplestep_signal(0, 2e-10, 0.1, 1, 1e-3)
# Too complicated to configure geo step flexibly
# elif args.chrono_step_type == 'geostep':
#     dtaq.configure_geostep_signal(args.s_init, s_rms, args.t_init, args.t_step, args.t_sample)


print('Running CHRONO')
dtaq.run(pstat, result_file=None, decimate=True)

dtaq.decimate_index

seq = HybridSequencer('galv', 'pot')

parser = argparse.ArgumentParser()
argc.add_args_from_dict(parser, argc.common_args)
argc.add_args_from_dict(parser, argc.hybrid_args)

args = parser.parse_args(('.', 'TEST'))
args.hybrid_step_type = 'geo'
args.hybrid_t_sample = 1e-3
args.hybrid_geo_t_short = 1e-2


rf.run_hybrid(seq, pstat, args, 'TEST')
