# Generate schedules for individual MFCs from channel config and channel composition schedule
import pandas as pd
import os
import numpy as np
import pydoc
from utils import check_equality, eq_ph2o, is_subset


def validate_channel_config(channel_config_df):
    mfc_gas_ids = channel_config_df.apply(lambda x: '{}_{}_{}'.format(x['Channel'], x['Gas'], x['Rank']), axis=1)
    unique_ids = np.unique(mfc_gas_ids)

    if len(mfc_gas_ids) != len(unique_ids):
        raise ValueError('Error in MFC config file: if multiple MFCs feed the same gas to the same channel, '
                         'they must be assigned unique ranks')


def validate_exp_schedule(exp_sched_df):
    steps = exp_sched_df['Step'].values
    if len(np.unique(steps)) != len(steps):
        raise ValueError('Step names in experiment schedule must be unique')


def split_flow(tot_flow, ranks, ranges):
    if len(ranks) != len(ranges):
        raise ValueError('Length of ranks must match length of ranges')

    if np.isscalar(tot_flow):
        flows = np.zeros(len(ranks))
        if tot_flow > np.sum(ranges):
            raise ValueError('Total requested flow ({} SCCM) exceeds total range of MFCs ({} SCCM)'.format(
                tot_flow, np.sum(ranges)
            ))

        ranks = np.array(ranks)
        ranges = np.array(ranges)

        sort_index = np.argsort(ranks)
        inverse_index = np.zeros_like(sort_index)
        inverse_index[sort_index] = np.arange(len(ranks))
        combo_ranges = np.cumsum(ranges[sort_index])
        num_required = np.where(combo_ranges >= tot_flow)[0][0] + 1

        if num_required == 1:
            # Full flow can be supplied by first MFC
            flows[0] = tot_flow
        else:
            # Split flow between multiple MFCs
            # Each active MFC provides the same % of its full range for best precision
            combo_range = combo_ranges[num_required - 1]
            range_frac = tot_flow / combo_range
            flows[:num_required] = range_frac * ranges[sort_index][:num_required]

        return flows[inverse_index]
    else:
        flows = [split_flow(tf, ranks, ranges) for tf in tot_flow]
        return np.array(flows)


def get_mfc_flows(channel_config, mfc_row, channel_flow, gas_frac):
    # Check for redundant MFCs
    try:
        dup_rows = channel_config[channel_config['Channel_Gas'] == mfc_row['Channel_Gas']]
        if len(dup_rows) == 1:
            # No redundant MFCs
            flow = channel_flow * gas_frac * mfc_row['CorrectionFactor']
        else:
            # Redundant MFCs - must distribute flow
            ranks = dup_rows['Rank'].values
            ranges = dup_rows['Range'].values / dup_rows['CorrectionFactor'].values
            dup_index = np.where(dup_rows['Full_ID'].values == mfc_row['Full_ID'])[0][0]
            dup_flows = split_flow(channel_flow * gas_frac, ranks, ranges)
            flow = dup_flows[:, dup_index] * mfc_row['CorrectionFactor']

        # Check for flows exceeding range
        index_over_range = np.where(flow > mfc_row['Range'])

        return flow, index_over_range
    except Exception as err:
        raise err.__class__('Error encountered while determining flow for MFC {}:\n{}'.format(
            mfc_row['DisplayName'], err
        ))


def generate_mfc_schedule(channel_config_file, exp_schedule_file, reduction_config_file, destination_dir):
    # Read files
    channel_config = pd.read_csv(channel_config_file)
    exp_schedule = pd.read_csv(exp_schedule_file)

    # Make sure that MFCs are assigned appropriate ranks if redundant MFCs exist
    validate_channel_config(channel_config)

    # Validate exp_schedule
    validate_exp_schedule(exp_schedule)

    # Verify that gases assigned to each channel are the same in channel config and exp schedule
    gas_config_ok = check_channel_gases(exp_schedule, channel_config)
    fractions_ok = check_schedule_fractions(exp_schedule, channel_config)

    mfc_schedule = exp_schedule.loc[:, ['Step']].copy()

    if reduction_config_file is not None:
        red_config = pd.read_csv(reduction_config_file)
        red_fractions_ok = check_schedule_fractions(red_config, channel_config)
        red_schedule = red_config.loc[:, ['ReductionStep']].copy()
    else:
        red_schedule = None

    # # Get exp step corresponding to reduction
    # exp_red_step = exp_schedule[exp_schedule['StepType'] == 'Reduction']
    # if len(exp_red_step) == 1:
    #     exp_red_step = exp_red_step.loc[0, :]
    # elif len(exp_red_step) > 1:
    #     raise ValueError(f'Multiple reduction steps found in {exp_schedule_file}')
    # else:
    #     exp_red_step = None

    channel_config['Full_ID'] = channel_config.apply(lambda x: '{}_{}'.format(x['Port'], x['ID']), axis=1)
    channel_config['Channel_Gas'] = channel_config.apply(lambda x: '{}_{}'.format(x['Channel'], x['Gas']), axis=1)

    for i, row in channel_config.iterrows():
        full_id = row['Full_ID']
        display_name = row['DisplayName']
        channel = row['Channel']  # get_mfc_property(channel_config, 'Channel', row['ID'], row['Port'])
        gas = row['Gas']  # get_mfc_property(channel_config, 'Gas', row['ID'], row['Port'])
        # cor_factor = row['CorrectionFactor']  # factor accounting for MFC miscalibration
        # max_range = row['Range']
        # channel_gas = row['Channel_Gas']

        # Determine MFC flows for regular steps
        channel_flow = exp_schedule[f'{channel}Flow']
        try:
            gas_frac = exp_schedule[f'{channel}Frac{gas}']
        except KeyError:
            # If gas_frac not specified, set to zero
            gas_frac = 0

        # # Check for redundant MFCs
        # dup_rows = channel_config[channel_config['Channel_Gas'] == channel_gas]
        # if len(dup_rows) == 1:
        #     # No redundant MFCs
        #     flow = channel_flow * gas_frac * cor_factor
        # else:
        #     # Redundant MFCs - must distribute flow
        #     ranks = dup_rows['Rank'].values
        #     ranges = dup_rows['Range'].values / dup_rows['CorrectionFactor'].values
        #     dup_index = np.where(dup_rows['Full_ID'].values == full_id)[0][0]
        #     dup_flows = split_flow(channel_flow * gas_frac, ranks, ranges)
        #     flow = dup_flows[:, dup_index] * cor_factor
        #
        # mfc_schedule[full_id] = flow
        #
        # # Validate flows
        # index_over_range = np.where(flow > max_range)

        # Get MFC flow
        flow, index_over_range = get_mfc_flows(channel_config, row, channel_flow, gas_frac)

        # Validate flows
        if len(index_over_range[0]) > 0:
            raise ValueError('Requested flow for MFC {} exceeds MFC range at the following steps: {}'.format(
                display_name, exp_schedule['Step'].values[index_over_range]
            ))

        mfc_schedule[full_id] = flow

        # Determine MFC flows for reduction step
        if reduction_config_file is not None:
            red_channel_flow = red_config[f'{channel}Flow']
            try:
                red_frac = red_config[f'{channel}Frac{gas}']
            except KeyError:
                # If gas_frac not specified, set to zero
                red_frac = 0

            # red_flow = red_channel_flow * red_frac * cor_factor
            # red_schedule[full_id] = red_flow
            #
            # # Validate flows
            # index_over_range = np.where(red_flow > max_range)

            # Get MFC flow
            red_flow, index_over_range = get_mfc_flows(channel_config, row, red_channel_flow, red_frac)

            # Validate flows
            if len(index_over_range[0]) > 0:
                raise ValueError('Requested flow for MFC {} exceeds MFC range at the following REDUCTION steps: {}'.format(
                    display_name, red_schedule['ReductionStep'].values[index_over_range]
                ))

            red_schedule[full_id] = red_flow

    mfc_schedule.to_csv(os.path.join(destination_dir, 'OUT_mfc_flow_schedule.csv'), index_label='Index')
    if red_schedule is not None:
        red_schedule.to_csv(os.path.join(destination_dir, 'OUT_reduction_flow_schedule.csv'), index_label='Index')

    return mfc_schedule, red_schedule


def generate_cmd_kwargs(exp_schedule_file, meas_config_file, destination):

    exp_schedule = pd.read_csv(exp_schedule_file)

    meas_config = pd.read_excel(meas_config_file, sheet_name=None)

    cmd_kw_schedule = exp_schedule.loc[:, ['Step', 'StepType']].copy()

    def generate_cmd(row):
        script_name = row['ExperimentScript']
        config_sheet = meas_config.get(script_name, None)
        if config_sheet is not None:
            # Get specified options
            config_options = config_sheet[~pd.isnull(config_sheet['ArgValue'])]
            # Handle simple arguments
            simple_args = config_options[~config_options['ArgType'].isin(['store_true', 'store_false'])]

            def get_simple_arg_string(arg_name, arg_type, arg_value):
                # Type cast to required type (there may be undesired type conversions when reading from config file)
                arg_value = pydoc.locate(arg_type)(arg_value)
                return '--{} {}'.format(arg_name, arg_value)

            arg_string = ' '.join([get_simple_arg_string(ar['ArgName'], ar['ArgType'], ar['ArgValue'])
                                   for i, ar in simple_args.iterrows()])
            # Handle action arguments
            action_args = config_options[config_options['ArgType'].isin(['store_true', 'store_false'])]

            def get_action_arg_string(arg_name, action, arg_value):
                if action == 'store_true' and (str(arg_value).upper() == 'TRUE' or arg_value == 1):
                    string = f'--{arg_name}'
                elif action == 'store_false' and (str(arg_value).upper() == 'FALSE' or arg_value == 0):
                    string = f'--{arg_name}'
                else:
                    string = ''
                # print(arg_name, arg_value, string)
                return string

            arg_string += ' ' + ' '.join([get_action_arg_string(ar['ArgName'], ar['ArgType'], ar['ArgValue'])
                                          for i, ar in action_args.iterrows()])
            arg_string = arg_string.strip()
        else:
            arg_string = ''

        if row['StepType'] in ('Measure', 'Reduction'):
            return arg_string
        elif row['StepType'] == 'End':
            return ''
        else:
            raise ValueError('Invalid step type {} at step {} in file {}'.format(row['StepType'],
                                                                                 row['Step'], exp_schedule_file
                                                                                 )
                             )

    cmd_kw_schedule['ArgText'] = exp_schedule.apply(generate_cmd, axis=1)

    cmd_kw_schedule.to_csv(destination, index_label='Index')


# Get ph2o(T) for interpolation
ph2o_temp_range = np.arange(25, 100.1, 0.1)
ph2o_values = eq_ph2o(ph2o_temp_range)


def get_bubbler_temp_and_status(ph2o):
    status = np.zeros(len(ph2o), dtype=int)

    active_index = np.where(ph2o > 0)
    status[active_index] = 1  # bubbler on

    # Leave bubbler temp at zero when status is off
    bub_temp = np.zeros(len(ph2o))
    bub_temp[active_index] = np.interp(ph2o[active_index], ph2o_values, ph2o_temp_range, left=-1, right=120)

    # Check for non-attainable ph2o values
    low_index = np.where((bub_temp < np.min(ph2o_temp_range)) & (bub_temp != 0))
    high_index = np.where(bub_temp > np.max(ph2o_temp_range))
    if len(low_index[0]) > 0:
        raise Exception('Requested pH2O values {} are too low at indices {}. Allowed pH2O range: {:.4f}, {:.4f}'.format(
            ph2o[low_index], low_index[0], np.min(ph2o_values), np.max(ph2o_values))
        )
    if len(high_index[0]) > 0:
        raise Exception('Requested pH2O values {} are too high at indices {}. Allowed pH2O range: {:.4f}, {:.4f}'.format(
            ph2o[high_index], high_index[0], np.min(ph2o_values), np.max(ph2o_values))
        )

    return bub_temp, status


def generate_tempctrl_schedule(channel_config_file, exp_schedule_file, tempctrl_config_file, destination):
    # Read files
    channel_config = pd.read_csv(channel_config_file)
    tc_config = pd.read_csv(tempctrl_config_file)
    exp_schedule = pd.read_csv(exp_schedule_file)

    # Check files
    check_tempctrl_config(exp_schedule, channel_config, tc_config)

    tc_schedule = exp_schedule.loc[:, ['Step']].copy()

    status_values = {}
    # Loop through temp controllers in order listed in config file
    # This ensures that LabVIEW can take schedule file columns in same order as config file rows without
    # needing to match IDs
    for i, row in tc_config.iterrows():
        tc_id = '{}_{}'.format(row['Port'], row['ID'])
        if row['Type'] == 'Furnace':
            tc_schedule[tc_id] = exp_schedule['Temperature']
        elif row['Type'] == 'HeatWrap':
            # Old logic: Turn heat wrap on whenever furnace temp is above RT to preheat gases
            # (not dependent on pH2O). Limit heat wrap temp to 80 C
            tc_schedule[tc_id] = np.minimum(exp_schedule['Temperature'].values, 90)

            # # Match heat wrap temperature to bubbler temperature to ensure that bubbler stays at target temperature
            # # when gas starts flowing through it
            # channel = row['Channel']
            # ph2o_col = f'{channel}PH2O'
            # if ph2o_col not in exp_schedule.columns:
            #     raise Exception(f'No pH2O specified for channel {channel} in exp_schedule_file')
            #
            # # Get bubbler temperature corresponding to specified pH2O
            # bub_temps, bub_status = get_bubbler_temp_and_status(exp_schedule[ph2o_col].values)
            # # Set heat wrap to 80 degrees to ensure no condensation
            # wrap_temps = np.zeros(len(bub_temps)) + 80
            # # If bubbler is at or near room temp, leave heat wrap cooler - this helps cool the bubbler when
            # # coming from hotter temps. The tubing downstream is heated enough via proximity to furnace
            # wrap_temps[bub_temps <= 30] = 45
            # wrap_temps[bub_status == 0] = 45
            # # Don't set the wrap temperature higher than the furnace temp
            # wrap_temps = np.minimum(exp_schedule['Temperature'].values, wrap_temps)
            # tc_schedule[tc_id] = wrap_temps

        elif row['Type'] == 'Bubbler':
            channel = row['Channel']
            ph2o_col = f'{channel}PH2O'
            if ph2o_col not in exp_schedule.columns:
                raise Exception(f'No pH2O specified for channel {channel} in exp_schedule_file')

            # Get bubbler temperature corresponding to specified pH2O
            bub_temps, bub_status = get_bubbler_temp_and_status(exp_schedule[ph2o_col].values)
            tc_schedule[tc_id] = bub_temps

            # Store bubbler status values - will be added to schedule after all temperature columns
            status_values[f'{channel}BubblerStatus'] = bub_status
        else:
            raise Exception('Invalid temperature controller type {} for {} in temperature controller '
                            'config file'.format(row['Type'], tc_id))

    # Add status columns after all temperature columns
    for col, values in status_values.items():
        tc_schedule[col] = values

    tc_schedule.to_csv(destination, index_label='Index')

    return tc_schedule


# Utility functions
# ---------------------------
def get_channel_gases_from_config(channel_config, channel):
    cdf = channel_config[channel_config['Channel'] == channel]

    return sorted(np.unique(cdf['Gas']))


def get_channel_gases_from_schedule(exp_schedule, channel):
    columns = exp_schedule.columns

    # Get columns corresponding to gas fractions for channel
    frac_cols = [col for col in columns if col.find(f'{channel}Frac') == 0]

    frac_gases = [col[len(f'{channel}Frac'):] for col in frac_cols]

    return sorted(frac_gases)


def get_mfc_property(channel_config, property_name, mfc_id, port=None):
    if property_name not in channel_config.columns:
        raise ValueError(f'Invalid property_name {property_name}. Property must be defined in channel_config. '
                         f'Valid options: {channel_config.columns}')

    if port is None:
        # Assume all MFCs on same port
        return dict(zip(channel_config['ID'].values, channel_config[property_name].values)).get(mfc_id, None)
    else:
        mfc_df = channel_config[(channel_config['ID'] == mfc_id) & (channel_config['Port'] == port)]
        if len(mfc_df) == 1:
            return mfc_df[property_name].values[0]
        elif len(mfc_df) == 0:
            return None
        elif len(mfc_df) > 1:
            raise ValueError(f'Multiple entries found for mfc_id {mfc_id} and port {port}')


def check_channel_gases(exp_schedule, channel_config):
    """Ensure gases in experiment schedule match gases in channel config"""
    channels = np.unique(channel_config['Channel'])
    for channel in channels:
        channel_gases = get_channel_gases_from_config(channel_config, channel)
        sched_gases = get_channel_gases_from_schedule(exp_schedule, channel)

        if not is_subset(sched_gases, channel_gases):
            raise ValueError(f'Gases for channel {channel} are inconsistent in channel configuration '
                             f'and channel schedule')

    return True


def check_schedule_fractions(exp_schedule, channel_config):
    """Ensure gas fractions sum to 1 for each channel"""
    channels = np.unique(channel_config['Channel'])
    columns = exp_schedule.columns
    for channel in channels:
        # Get columns corresponding to gas fractions for channel
        frac_cols = [col for col in columns if col.find(f'{channel}Frac') == 0]

        if len(frac_cols) > 0:
            # Exclude rows with zero flow through channel
            nonzero_df = exp_schedule[exp_schedule[f'{channel}Flow'] > 0]
            total_frac = nonzero_df.loc[:, frac_cols].sum(axis=1)

            if not check_equality(np.round(total_frac, 5), np.ones(len(total_frac))):
                bad_index = total_frac[total_frac.round(5) != 1.0].index
                raise ValueError(f'Gas fractions in channel schedule do not sum to 1 for channel {channel} '
                                 f'for the following rows: {bad_index.values}')

    return True


def get_tempctrl_property(tempctrl_config, property_name, tempctrl_id, port=None):
    if property_name not in tempctrl_config.columns:
        raise ValueError(f'Invalid property_name {property_name}. Property must be defined in tempctrl_config. '
                         f'Valid options: {tempctrl_config.columns}')

    if port is None:
        # Assume all temp controllers on same port
        return dict(zip(tempctrl_config['ID'].values, tempctrl_config[property_name].values)).get(tempctrl_id, None)
    else:
        mfc_df = tempctrl_config[(tempctrl_config['ID'] == tempctrl_id) & (tempctrl_config['Port'] == port)]
        if len(mfc_df) == 1:
            return mfc_df[property_name].values[0]
        elif len(mfc_df) == 0:
            return None
        elif len(mfc_df) > 1:
            raise ValueError(f'Multiple entries found for mfc_id {tempctrl_id} and port {port}')


def check_tempctrl_config(exp_schedule, channel_config, tempctrl_config):
    # Ensure 1 temp controller is assigned to furnace
    num_furnace = len(tempctrl_config[tempctrl_config['Type'] == 'Furnace'])
    if num_furnace != 1:
        raise Exception('There must be one and only one temperature controller with Type=Furnace in the temperature '
                        'controller config file')

    # Ensure that every channel for which a nonzero pH2O is scheduled has a bubbler assigned
    channels = np.unique(channel_config['Channel'])
    columns = exp_schedule.columns
    for channel in channels:
        ph2o_col = f'{channel}PH2O'
        if ph2o_col in columns:
            if np.max(exp_schedule[ph2o_col]) > 0:
                # Look for channel bubbler
                bub_df = tempctrl_config[(tempctrl_config['Channel'].isin([channel, 'All'])) &
                                         (tempctrl_config['Type'] == 'Bubbler')]
                if len(bub_df) == 0:
                    raise Exception(f'No bubbler assigned to channel {channel} in temp controller config file')
                elif len(bub_df) > 1:
                    raise Exception(f'Multiple bubblers assigned to channel {channel} in temp controller config file')

    return True


# Obsolete
# -----------------------------
# def generate_cmd_schedule(exp_schedule_file, exp_config_file, reduction_config_file, python_path, script_path,
#                           data_path, destination):
#
#     exp_schedule = pd.read_csv(exp_schedule_file)
#
#     exp_config = pd.read_excel(exp_config_file, sheet_name=None)
#
#     cmd_schedule = exp_schedule.loc[:, ['Step', 'StepType']].copy()
#
#     def generate_cmd(row):
#         script_name = row['ExperimentScript']
#         config_sheet = exp_config.get(script_name, None)
#         if config_sheet is not None:
#             # Get specified options
#             config_options = config_sheet[~pd.isnull(config_sheet['ArgValue'])]
#             # Handle simple arguments
#             simple_args = config_options[~config_options['ArgType'].isin(['store_true', 'store_false'])]
#
#             def get_simple_arg_string(arg_name, arg_type, arg_value):
#                 # Type cast to required type (there may be undesired type conversions when reading from config file)
#                 arg_value = pydoc.locate(arg_type)(arg_value)
#                 return '--{} {}'.format(arg_name, arg_value)
#
#             arg_string = ' '.join([get_simple_arg_string(ar['ArgName'], ar['ArgType'], ar['ArgValue'])
#                                    for i, ar in simple_args.iterrows()])
#             # Handle action arguments
#             action_args = config_options[config_options['ArgType'].isin(['store_true', 'store_false'])]
#
#             def get_action_arg_string(arg_name, action, arg_value):
#                 if action == 'store_true':
#                     if arg_value == 'True':
#                         string = f'--{arg_name}'
#                 elif action == 'store_false':
#                     if arg_value == 'False':
#                         string = f'--{arg_name}'
#                 else:
#                     string = ''
#                 return string
#
#             arg_string += ' ' + ' '.join([get_action_arg_string(ar['ArgName'], ar['ArgType'], ar['ArgValue'])
#                                           for i, ar in action_args.iterrows()])
#             arg_string = arg_string.strip()
#         else:
#             arg_string = ''
#
#         # NOTE: LabVIEW cannot handle double quotes in delimited files for some reason (???)
#         # Workaround: Put single quotes in cmd text where double quotes are desired,
#         # then replace with double quotes in LabVIEW after reading csv file
#         # /s /c flags allow entire command to be enclosed in quotes safely
#         if row['StepType'] == 'Measure':
#             return "cmd /s /c ''{}' '{}' '{}' '{}' --num_loops {} {}''".format(
#                 python_path,
#                 os.path.join(script_path, row['ExperimentScript']),
#                 data_path,
#                 'Step{}'.format(row['Step']),  # file suffix
#                 row['ExperimentCycles'],
#                 arg_string
#             )
#         elif row['StepType'] == 'Reduction':
#             return "cmd /s /c ''{}' '{}' '{}' '{}' '{}' {}''".format(
#                 python_path,
#                 os.path.join(script_path, row['ExperimentScript']),
#                 data_path,
#                 'Step{}'.format(row['Step']),  # file suffix
#                 reduction_config_file,
#                 arg_string
#             )
#         elif row['StepType'] == 'End':
#             return ''
#         else:
#             raise ValueError('Invalid step type {} at step {} in file {}'.format(row['StepType'],
#                                                                                  row['Step'], exp_schedule_file
#                                                                                  )
#                              )
#
#     cmd_schedule['CmdText'] = exp_schedule.apply(generate_cmd, axis=1)
#
#     cmd_schedule.to_csv(destination, index_label='Index')