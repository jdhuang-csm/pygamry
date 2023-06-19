import codecs


def decode_string_arg(arg):
    """Decode string argument with escape characters"""
    return codecs.decode(arg, 'unicode_escape')


def add_args_from_dict(parser, arg_dict):
    """Add arguments to parser from dict"""
    for name, kw in arg_dict.items():
        parser.add_argument(name, **kw)


def none_or_float(value):
    if value == 'None':
        return None
    return value


# Define common args for experiment scripts
common_args = {
    'data_path': dict(type=str),
    'file_suffix': dict(type=str),
    '--exp_notes': dict(type=decode_string_arg, default=''),
    '--num_loops': dict(type=int, default=1),
    '--kst_path': dict(type=str)
}

# Define kwargs for different measurement types
ocp_args = {
    '--ocp_duration': dict(type=float, default=600),
    '--ocp_sample_period': dict(type=float, default=1)
}

eis_args = {
    # '--eis_ctrl_mode': dict(type=str, default='pot'),
    '--eis_Z_guess': dict(type=float, default=5.0),
    '--eis_speed': dict(type=str, default='Norm'),
    '--eis_mode': dict(type=str, default='pot'),
    '--eis_max_freq': dict(type=float, default=1e6),
    '--eis_min_freq': dict(type=float, default=0.1),
    '--eis_ppd': dict(type=int, default=10),
    '--eis_SAC': dict(type=float, default=0.01),
    '--eis_SDC': dict(type=float, default=0.0),
    '--eis_VDC_vs_VRef': dict(default=False, action='store_true'),
    '--eis_condition_time': dict(type=float, default=0)
}

pstatic_args = {
    '--pstatic_VDC': dict(type=float, default=0.8),
    '--pstatic_duration': dict(type=float, default=600),
    '--pstatic_sample_period': dict(type=float, default=1),
    '--pstatic_i_min': dict(type=float, default=None),
    '--pstatic_i_max': dict(type=float, default=None),
    '--pstatic_VDC_vs_VRef': dict(default=False, action='store_true')
}

pwrpol_args = {
    '--pwrpol_i_final': dict(type=float, default=1),
    '--pwrpol_scan_rate': dict(type=float, default=0.001),
    '--pwrpol_sample_period': dict(type=float, default=5),
    '--pwrpol_v_min': dict(type=float, default=0.4),
    '--pwrpol_v_max': dict(type=float, default=1.5),
    '--pwrpol_direction': dict(type=str, default='both'),
    '--pwrpol_rest_time': dict(type=float, default=60)
}

chrono_decimate_args = {
    '--disable_decimation': dict(default=False, action='store_true'),
    '--decimate_during': dict(type=str, default='write'),
    '--decimation_prestep_points': dict(type=int, default=20),
    '--decimation_interval': dict(type=int, default=30),
    '--decimation_factor': dict(type=int, default=2),
    '--decimation_max_t_sample': dict(type=float, default=0.05),
    '--decimate_filter': dict(default=False, action='store_true')
}

chrono_step_args = {
    '--chrono_v_rms': dict(type=float, default=0.01),
    '--chrono_disable_find_i': dict(default=False, action='store_true'),
    '--chrono_step_type': dict(type=str, default='dstep'),
    '--chrono_s_init': dict(type=float, default=0.0),
    '--chrono_t_init': dict(type=float, default=0.1),
    '--chrono_t_sample': dict(type=float, default=1e-4),
    # Dstep
    '--chrono_s_step1': dict(type=float, default=-1e-4),
    '--chrono_s_step2': dict(type=float, default=0.0),
    '--chrono_t_step1': dict(type=float, default=1),
    '--chrono_t_step2': dict(type=float, default=1),
    # Mstep
    '--chrono_s_step': dict(type=float, default=1e-4),
    '--chrono_t_step': dict(type=float, default=1),
    '--chrono_n_steps': dict(type=int, default=1),
    # triplestep / geostep
    '--chrono_s_rms': dict(type=float, default=0.01),
    # geostep
    '--chrono_geo_s_final': dict(type=float, default=0),
    '--chrono_geo_s_min': dict(type=float, default=0),
    '--chrono_geo_s_max': dict(type=float, default=0),
    '--chrono_geo_t_short': dict(type=float, default=1e-3),
    '--chrono_geo_num_scales': dict(type=int, default=3),
    '--chrono_geo_steps_per_scale': dict(type=int, default=2),
}

hybrid_args = {
    '--hybrid_step_type': dict(type=str, default='triple'),
    '--hybrid_i_init': dict(type=float, default=0),
    '--hybrid_i_rms': dict(type=float, default=0.01),  # ignored if v_rms provided
    '--hybrid_v_rms': dict(type=float, default=0.01),
    '--hybrid_disable_find_i': dict(default=False, action='store_true'),
    '--hybrid_t_init': dict(type=float, default=0.1),
    '--hybrid_t_step': dict(type=float, default=1),
    '--hybrid_geo_t_short': dict(type=float, default=1e-3),
    '--hybrid_geo_num_scales': dict(type=int, default=3),
    '--hybrid_geo_steps_per_scale': dict(type=int, default=2),
    '--hybrid_geo_end_at_init': dict(action='store_true', default=False),
    '--hybrid_geo_end_time': dict(type=float, default=0),
    '--hybrid_chrono_first': dict(action='store_true', default=False),
    '--hybrid_t_sample': dict(type=float, default=1e-4),
    '--hybrid_eis_max_freq': dict(type=float, default=1e6),
    '--hybrid_eis_min_freq': dict(type=float, default=1e3),
    '--hybrid_eis_ppd': dict(type=int, default=10),
    '--hybrid_eis_mode': dict(type=str, default='galv'),
    '--hybrid_rest_time': dict(type=float, default=0.0),
    '--hybrid_Z_guess': dict(type=float, default=1),
}
# Include decimation options
hybrid_args.update(chrono_decimate_args)

staircase_args = {
    '--staircase_v_min': dict(type=float, default=0.4),
    '--staircase_v_max': dict(type=float, default=1.5),
    '--staircase_num_steps': dict(type=int, default=20),
    '--staircase_constant_step_size': dict(default=False, action='store_true'),
    '--staircase_equil_time': dict(type=float, default=60.0),
    '--staircase_ocv_equil': dict(default=False, action='store_true'),
    '--staircase_run_post_eis': dict(default=False, action='store_true'),
    '--staircase_run_pre_eis': dict(default=False, action='store_true'),
    '--staircase_direction': dict(type=str, default='both'),
    '--staircase_full_eis_max_freq': dict(type=float, default=1e6),
    '--staircase_full_eis_min_freq': dict(type=float, default=1e-1),
    '--staircase_full_eis_ppd': dict(type=int, default=10),
}

equil_args = {
    '--equil_mode': dict(type=str, default='pot'),
    '--equil_duration': dict(type=float, default=600),
    '--equil_sample_period': dict(type=float, default=0.01),
    '--equil_window_seconds': dict(type=float, default=15),
    '--equil_slope_thresh': dict(type=float, default=0.25),  # %/min for pstatic, mV/min for gstatic
    '--equil_min_wait_time': dict(type=float, default=0),  # minutes
    '--equil_require_consecutive': dict(type=int, default=10)
}

pstatic_equil_args = {
    '--pequil_VDC': dict(type=float, default=-0.1),
    '--pequil_i_min': dict(type=float, default=None),
    '--pequil_i_max': dict(type=float, default=None),
    '--pequil_VDC_vs_VRef': dict(default=False, action='store_true'),
}

gstatic_equil_args = {
    '--gequil_IDC': dict(type=float, default=0.0),
    '--gequil_VDC': dict(type=float, default=None),
    '--gequil_VDC_vs_VRef': dict(action='store_true', default=False),
    '--gequil_v_min': dict(type=float, default=None),
    '--gequil_v_max': dict(type=float, default=None),
}

vsweep_args = {
    '--vsweep_v_rms': dict(type=float, default=0.01),
    '--vsweep_t_init': dict(type=float, default=1),
    '--vsweep_t_sample': dict(type=float, default=1e-2),
    # Mstep
    '--vsweep_t_step': dict(type=float, default=5),
    '--vsweep_num_steps': dict(type=int, default=20),
    # General
    '--vsweep_direction': dict(type=str, default='both'),
    '--vsweep_v_min': dict(type=float, default=0.4),
    '--vsweep_v_max': dict(type=float, default=1.5),
    '--vsweep_rest_time': dict(type=float, default=60),
    '--vsweep_ocv_equil': dict(default=False, action='store_true')
}
