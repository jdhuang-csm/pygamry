# Generate schedules for individual MFCs from channel config and channel composition schedule
import pandas as pd
import os
import numpy as np
from utils import check_equality
from file_proc import generate_mfc_schedule, get_channel_gases_from_config, get_channel_gases_from_schedule, \
    get_mfc_property, check_channel_gases, check_schedule_fractions
import argparse
import easygui
import matplotlib.pyplot as plt
import sys


parser = argparse.ArgumentParser(description='Generate flow schedule for individual MFCs')
parser.add_argument('channel_config_file', type=str)
parser.add_argument('channel_schedule_file', type=str)
args = parser.parse_args()


channel_config_file = args.channel_config_file
channel_schedule_file = args.channel_schedule_file

# Generate MFC schedule
schedule_path = os.path.split(channel_schedule_file)[0]
mfc_schedule, _ = generate_mfc_schedule(channel_config_file, channel_schedule_file, None, schedule_path)

# Plot channel composition schedule
# -----------------------------------
channel_config = pd.read_csv(channel_config_file)
channels = np.unique(channel_config['Channel'])
channel_schedule = pd.read_csv(channel_schedule_file)
sched_columns = channel_schedule.columns

# Generate times to plot
start_times = np.concatenate(([0.0], np.cumsum(channel_schedule['Duration'].values)))
times = np.array(sum([[t, t + 1e-3] for t in start_times], []))

fig, axes = plt.subplots(len(channels), 3, figsize=(10, 3 * len(channels)))

for channel, axrow in zip(channels, axes):
    # Row 0: gas fractions
    frac_columns = [col for col in sched_columns if col.find(f'{channel}Frac') == 0]
    for frac_col in frac_columns:
        plot_values = np.zeros(len(times))
        for st, val in zip(start_times[:-1], channel_schedule[frac_col].values):
            plot_values[times > st] = val
        gas_name = frac_col[len(f'{channel}Frac'):]
        axrow[0].plot(times, plot_values, label=gas_name)
    
    axrow[0].set_ylabel('Fraction')
    axrow[0].set_title(f'{channel} Composition')
    axrow[0].legend()
    
    # Row 1: total flow
    plot_values = np.zeros(len(times))
    for st, val in zip(start_times[:-1], channel_schedule[f'{channel}Flow'].values):
        plot_values[times > st] = val
    axrow[1].plot(times, plot_values)
    axrow[1].set_ylabel('Total flow (SCCM)')
    axrow[1].set_title(f'{channel} Total Flow')
    
    # Row 2: individual MFC flows
    # Row 0: gas fractions
    mfc_columns = mfc_schedule.columns[2:]
    for col in mfc_columns:
        port, mfc_id = col.split('_')
        if get_mfc_property(channel_config, 'Channel', mfc_id, port) == channel:
            gas_name = get_mfc_property(channel_config, 'Gas', mfc_id, port)
            plot_values = np.zeros(len(times))
            for st, val in zip(start_times[:-1], mfc_schedule[col].values):
                plot_values[times > st] = val
            axrow[2].plot(times, plot_values, label=f'{col} ({gas_name})')
            axrow[2].set_ylabel('MFC flow (SCCM)')
        
    axrow[2].set_title(f'{channel} MFC Flows')
    axrow[2].legend()
    
    for ax in axrow:
        ax.set_xlabel('Time')
    
fig.tight_layout()
fig_path = os.path.join(schedule_path, 'mfc_schedule.png')
fig.savefig(fig_path, dpi=350)

plt.show(block=False)

confirmation = easygui.ynbox(msg='Confirm channel schedule?')


if confirmation:
    
    print('Generated MFC schedule')
    sys.exit(0)
else:
    # with open(destination, 'f') as file:
        # file.write('')
    # to_csv(os.path.join(schedule_path, 'mfc_flow_schedule.csv'), index=False)
    print('Schedule not confirmed. Did not generate MFC schedule')
    sys.exit(1)
    # return False









