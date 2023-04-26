# Generate schedules for individual MFCs from channel config and channel composition schedule
import pandas as pd
import os
import numpy as np
from file_proc import generate_mfc_schedule, generate_cmd_kwargs, generate_tempctrl_schedule, get_mfc_property, \
    get_tempctrl_property
import argparse
import easygui
import matplotlib.pyplot as plt
import sys

parser = argparse.ArgumentParser(description='Generate equipment schedule files for experiment control')
parser.add_argument('experiment_schedule_file', type=str)
parser.add_argument('measurement_config_file', type=str)
parser.add_argument('channel_config_file', type=str)
parser.add_argument('tempctrl_config_file', type=str)
parser.add_argument('reduction_config_file', type=str)
# parser.add_argument('python_path', type=str)
# parser.add_argument('script_path', type=str)
# parser.add_argument('data_path', type=str)
args = parser.parse_args()
# print(vars(args))

try:
    # Generate MFC schedule
    schedule_path = os.path.split(args.experiment_schedule_file)[0]
    mfc_schedule_file = os.path.join(schedule_path, 'OUT_mfc_flow_schedule.csv')
    mfc_schedule, red_schedule = generate_mfc_schedule(args.channel_config_file, args.experiment_schedule_file,
                                                       args.reduction_config_file,
                                                       schedule_path)

    # Generate temp controller schedule
    tc_schedule_file = os.path.join(schedule_path, 'OUT_temp_controller_schedule.csv')
    tc_schedule = generate_tempctrl_schedule(args.channel_config_file, args.experiment_schedule_file,
                                             args.tempctrl_config_file, tc_schedule_file)

    # Generate cmd arg file
    # cmd_schedule_file = os.path.join(schedule_path, 'OUT_cmd_schedule.csv')
    # generate_cmd_schedule(args.experiment_schedule_file, args.experiment_config_file, args.reduction_config_file,
    #                       args.python_path, args.script_path, args.data_path, cmd_schedule_file)
    cmd_arg_file = os.path.join(schedule_path, 'OUT_cmd_args.csv')
    generate_cmd_kwargs(args.experiment_schedule_file, args.measurement_config_file, cmd_arg_file)

    # Plot schedule outputs for confirmation
    # ---------------------------------------
    channel_config = pd.read_csv(args.channel_config_file)
    tc_config = pd.read_csv(args.tempctrl_config_file)
    exp_schedule = pd.read_csv(args.experiment_schedule_file)
    red_config = pd.read_csv(args.reduction_config_file)
    sched_columns = exp_schedule.columns


    def plot_flow_schedule(input_schedule_df, output_flow_df, channel_config):
        channels = np.unique(channel_config['Channel'])

        # Generate times to plot
        start_indices = np.concatenate((input_schedule_df.index, [len(input_schedule_df)]))
        plot_index = np.array(sum([[t - 1e-3, t, t + 1e-3] for t in start_indices], []))
        plot_index = plot_index[1:]

        fig, axes = plt.subplots(len(channels), 3, figsize=(10, 3 * len(channels)))

        sched_columns = input_schedule_df.columns
        for channel, axrow in zip(channels, axes):
            # Row 0: gas fractions
            frac_columns = [col for col in sched_columns if col.find(f'{channel}Frac') == 0]
            for frac_col in frac_columns:
                plot_values = np.zeros(len(plot_index))
                for st, val in zip(start_indices[:-1], input_schedule_df[frac_col].values):
                    plot_values[plot_index >= st] = val
                gas_name = frac_col[len(f'{channel}Frac'):]
                axrow[0].plot(plot_index, plot_values, label=gas_name)

            axrow[0].set_ylabel('Fraction')
            axrow[0].set_title(f'{channel} Composition')
            axrow[0].legend()

            # Row 1: total flow
            plot_values = np.zeros(len(plot_index))
            for st, val in zip(start_indices[:-1], input_schedule_df[f'{channel}Flow'].values):
                plot_values[plot_index >= st] = val
            axrow[1].plot(plot_index, plot_values)
            axrow[1].set_ylabel('Total flow (SCCM)')
            axrow[1].set_title(f'{channel} Total Flow')

            # Row 2: individual MFC flows
            mfc_columns = output_flow_df.columns[2:]
            for col in mfc_columns:
                port, mfc_id = col.split('_')
                if get_mfc_property(channel_config, 'Channel', mfc_id, port) == channel:
                    gas_name = get_mfc_property(channel_config, 'Gas', mfc_id, port)
                    # display_name = get_mfc_property(channel_config, 'DisplayName', mfc_id, port)
                    plot_values = np.zeros(len(plot_index))
                    for st, val in zip(start_indices[:-1], output_flow_df[col].values):
                        plot_values[plot_index >= st] = val
                    axrow[2].plot(plot_index, plot_values, label=f'{col} ({gas_name})')
                    axrow[2].set_ylabel('MFC flow (SCCM)')

            axrow[2].set_title(f'{channel} MFC Flows')
            axrow[2].legend()

            for ax in axrow:
                ax.set_xlabel('Step Index')

        fig.tight_layout()

        return fig


    def plot_temp_schedule(output_temp_df, tempctrl_config):
        # channels = np.unique(channel_config['Channel'])

        # Generate indices to plot
        start_indices = np.concatenate((output_temp_df.index, [len(output_temp_df)]))
        plot_index = np.array(sum([[t - 1e-3, t, t + 1e-3] for t in start_indices], []))
        plot_index = plot_index[1:]

        fig, axes = plt.subplots(1, 3, figsize=(10, 3))

        for i, row in tempctrl_config.iterrows():
            # Get temperatures to plot
            plot_temps = np.zeros(len(plot_index))
            tc_id = '{}_{}'.format(row['Port'], row['ID'])
            for st, val in zip(start_indices[:-1], output_temp_df[tc_id].values):
                plot_temps[plot_index >= st] = val

            if row['Type'] == 'Furnace':
                # ax0: furnace and heatwrap temp
                axes[0].plot(plot_index, plot_temps, label=row['DisplayName'])
            elif row['Type'] == 'HeatWrap':
                axes[1].plot(plot_index, plot_temps, label=row['DisplayName'])
            elif row['Type'] == 'Bubbler':
                # ax1: bubbler temp
                axes[1].plot(plot_index, plot_temps, label=row['DisplayName'])

                # ax2: bubbler status
                # Get status values to plot
                plot_status = np.zeros(len(plot_index))
                status_col = '{}BubblerStatus'.format(row['Channel'])
                for st, val in zip(start_indices[:-1], output_temp_df[status_col].values):
                    plot_status[plot_index >= st] = val

                axes[2].plot(plot_index, plot_status, label=row['DisplayName'])

        for ax in axes:
            ax.set_xlabel('Step Index')
            ax.legend()

        axes[0].set_ylabel('T ($^\circ$C)')
        axes[0].set_title('Furnace Temp')

        axes[1].set_ylabel('T ($^\circ$C)')
        axes[1].set_title('Bubbler and Heat Wrap Temp')

        axes[2].set_ylabel('Status')
        axes[2].set_title('Bubbler Status')
        axes[2].set_yticks([0, 1])
        axes[2].set_yticklabels(['OFF', 'ON'])

        fig.tight_layout()

        return fig


    # Confirm reduction schedule
    exp_sched = pd.read_csv(args.experiment_schedule_file)
    num_red_steps = len(exp_sched[exp_sched['StepType'] == 'Reduction'])
    if num_red_steps == 0:
        confirmation = easygui.ynbox(
            msg='No reduction step is scheduled in the experiment sequence. Do you wish to proceed?'
        )
        if confirmation:
            print('Skipped reduction schedule')
        else:
            print('Canceled')
            sys.exit(1)
    elif num_red_steps == 1:
        red_fig = plot_flow_schedule(red_config, red_schedule, channel_config)
        red_fig_path = os.path.join(schedule_path, 'OUT_reduction_flow_schedule.png')
        red_fig.savefig(red_fig_path, dpi=350)

        plt.show(block=False)

        confirmation = easygui.ynbox(msg='Confirm reduction flow schedule?')
        if confirmation:
            print('Generated reduction schedule')
            plt.close()
        else:
            print('Reduction schedule not confirmed')
            sys.exit(1)
    else:
        confirmation = easygui.ynbox(
            msg='Multiple reduction steps are scheduled in the experiment sequence. Do you wish to proceed?'
        )
        if confirmation:
            red_fig = plot_flow_schedule(red_config, red_schedule, channel_config)
            red_fig_path = os.path.join(schedule_path, 'OUT_reduction_flow_schedule.png')
            red_fig.savefig(red_fig_path, dpi=350)

            plt.show(block=False)

            confirmation = easygui.ynbox(msg='Confirm reduction flow schedule?')
            if confirmation:
                print('Generated reduction schedule')
                plt.close()
            else:
                print('Reduction schedule not confirmed')
                sys.exit(1)
        else:
            print('Canceled')
            sys.exit(1)

    # Confirm channel flow schedule
    mfc_fig = plot_flow_schedule(exp_schedule, mfc_schedule, channel_config)
    mfc_fig_path = os.path.join(schedule_path, 'OUT_mfc_flow_schedule.png')
    mfc_fig.savefig(mfc_fig_path, dpi=350)

    plt.show(block=False)

    confirmation = easygui.ynbox(msg='Confirm channel flow schedule?')

    if confirmation:
        print('Generated MFC schedule')
        plt.close()
    else:
        print('Canceled')
        sys.exit(1)

    # Confirm temp controller and bubbler schedule
    tc_fig = plot_temp_schedule(tc_schedule, tc_config)
    tc_fig_path = os.path.join(schedule_path, 'OUT_temp_controller_schedule.png')
    tc_fig.savefig(mfc_fig_path, dpi=350)

    plt.show(block=False)

    confirmation = easygui.ynbox(msg='Confirm temperature and bubbler schedule?')

    if confirmation:
        print('Generated temp controller schedule')
        sys.exit(0)
    else:
        print('Canceled')
        sys.exit(1)

except Exception as err:
    easygui.msgbox(msg=f'The following error occurred while processing configuration files:'
                       f'\n\n{err.__class__.__name__}: {err}\n\n'
                       'Please correct the error and try again. Check the Python log for more information.',
                   title='Error processing configuration files')

    # Raise the error to ensure the full traceback is available in the log
    raise err
    # sys.exit(1)
