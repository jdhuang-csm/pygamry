import os
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.interpolate import interp1d

import pandas as pd
from pygamry.dtaq import DtaqChrono, DtaqOcv, GamryCOM, DtaqGstatic, DtaqPstatic
from pygamry.equilibration import DtaqPstaticEquil, DtaqGstaticEquil
from pygamry.file_utils import read_curve_data


def run_ocv(dtaq, pstat, args, file_suffix, show_plot=False):
    print('Running OCV')
    ocv_file = os.path.join(args.data_path, f'OCP_{file_suffix}.DTA')
    if args.kst_path is not None:
        kst_file = os.path.join(args.kst_path, 'Kst_OCP.DTA')
    else:
        kst_file = None

    dtaq.run(pstat, args.ocp_duration, args.ocp_sample_period, show_plot=show_plot,
             result_file=ocv_file, kst_file=kst_file)
    print('OCV done\n')

    if show_plot:
        plt.close()


def get_eis_frequencies(max_freq, min_freq, ppd):
    num_decades = np.log10(max_freq) - np.log10(min_freq)
    num_freq = int(ppd * num_decades) + 1
    eis_freq = np.logspace(np.log10(max_freq), np.log10(min_freq), num_freq)
    return eis_freq


def run_eis(dtaq, pstat, args, file_suffix, V_oc=None, show_plot=False):
    # Get frequencies to measure
    eis_freq = get_eis_frequencies(args.eis_max_freq, args.eis_min_freq, args.eis_ppd)

    # Set ReadZSpeed
    print('eis_speed:', args.eis_speed)
    dtaq.gc_readzspeed = f'ReadZSpeed{args.eis_speed}'

    # Determine DC offset
    if dtaq.mode == 'pot' and not args.eis_VDC_vs_VRef:
        # Offset relative to OCV
        if V_oc is None:
            V_oc = test_ocv(pstat)

        eis_SDC = args.eis_SDC + V_oc
    else:
        eis_SDC = args.eis_SDC

    print('Running EIS')
    print('EIS SDC: {:.3f}'.format(eis_SDC))
    eis_file = os.path.join(args.data_path, f'EIS{dtaq.mode.upper()}_{file_suffix}.DTA')
    if args.kst_path is not None:
        kst_file = os.path.join(args.kst_path, 'Kst_EIS.DTA')
    else:
        kst_file = None
    dtaq.run(pstat, eis_freq, eis_SDC, args.eis_SAC, args.eis_Z_guess, timeout=None,
             show_plot=show_plot, plot_interval=1, plot_type='all', condition_time=args.eis_condition_time,
             result_file=eis_file, kst_file=kst_file)
    print('EIS done\n')

    if show_plot:
        plt.close()


def equil_ocv(pstat, v_oc, duration, suffix, exp_notes, data_path ,kst_path):
    pstatic = DtaqPstatic(write_mode='interval', write_interval=1,
                          exp_notes=exp_notes)

    print("Equilibrating at OCV for {:.1f} s...".format(duration))

    pstatic_file = os.path.join(data_path, f'PSTATIC-EQUIL_{suffix}.DTA')
    if kst_path is not None:
        kst_file = os.path.join(kst_path, 'Kst_IVT.DTA')
    else:
        kst_file = None

    pstatic.run(pstat, v_oc, duration, 1.0,
                i_min=-1.0, i_max=1.0, timeout=None,
                show_plot=False, result_file=pstatic_file, kst_file=kst_file)



def run_pwrpol(dtaq, pstat, args, file_suffix, show_plot=False):
    print('Running PWRPOL')

    direction_options = ['both', 'charge', 'discharge']
    if args.pwrpol_direction not in direction_options:
        raise ValueError(f'Invalid direction {args.pwrpol_direction}. Options: {direction_options}')
    elif args.pwrpol_direction == 'both':
        directions = ['discharge', 'charge']
    else:
        directions = [args.pwrpol_direction]

    pwrpol_file = os.path.join(args.data_path, f'PWRPOLARIZATION_{file_suffix}.DTA')
    if args.kst_path is not None:
        kst_file = os.path.join(args.kst_path, 'Kst_PWRPOL.DTA')
    else:
        kst_file = None

    for i, direction in enumerate(directions):
        if direction == 'charge':
            dtaq.mode = 'CurrentCharge'
        else:
            dtaq.mode = 'CurrentDischarge'

        # If measuring both directions, append to file and rest between
        if i == 0:
            append_to_file = False
        else:
            append_to_file = True
            print(f'Resting for {args.pwrpol_rest_time} s between charge/discharge...')
            time.sleep(args.pwrpol_rest_time)

        print(f'Running in {dtaq.mode} mode...')
        dtaq.run(pstat, args.pwrpol_i_final, args.pwrpol_scan_rate, args.pwrpol_sample_period,
                 v_min=args.pwrpol_v_min, v_max=args.pwrpol_v_max,
                 show_plot=show_plot, result_file=pwrpol_file, kst_file=kst_file,
                 append_to_file=append_to_file)
    print('PWRPOL done\n')

    if show_plot:
        plt.close()


def get_pstatic_cutoff(pstatic_VDC, args, V_oc=None, pwrpol_dtaq=None):
    if V_oc is not None and pwrpol_dtaq is not None:
        # Determine current sign
        i_sign = np.sign(pstatic_VDC - V_oc)

        # Get cutoff current from polarization curve
        # Find current at which voltage = pstatic_VDC
        jv_df = pwrpol_dtaq.dataframe
        sort_index = np.argsort(jv_df.values)  # sort for interpolation
        v_sort = jv_df['Vf'].values[sort_index]
        i_sort = jv_df['Im'].values[sort_index]
        i_ref = np.interp(pstatic_VDC, v_sort, i_sort)

        if i_sign > 0 and args.pstatic_i_min is None:
            # Set cutoff at 50% of reference current
            pstatic_i_min = i_ref * 0.5
        else:
            pstatic_i_min = args.pstatic_i_min

        if i_sign < 0 and args.pstatic_i_max is None:
            # Set cutoff at 50% of reference current
            pstatic_i_max = i_ref * 0.5
        else:
            pstatic_i_max = args.pstatic_i_max

        print('Pstatic reference current:', i_ref)
    else:
        pstatic_i_min = args.pstatic_i_min
        pstatic_i_max = args.pstatic_i_max

    print('Pstatic current limits:', (pstatic_i_min, pstatic_i_max))

    return pstatic_i_min, pstatic_i_max


def run_pstatic(dtaq, pstat, args, file_suffix, V_oc=None, pwrpol_dtaq=None, show_plot=False):
    # Determine DC voltage
    if args.pstatic_VDC_vs_VRef:
        pstatic_VDC = args.pstatic_VDC
    else:
        pstatic_VDC = args.pstatic_VDC + V_oc

    # Get cutoff current from polarization curve
    pstatic_i_min, pstatic_i_max = get_pstatic_cutoff(pstatic_VDC, args, V_oc, pwrpol_dtaq)

    print('Running PSTATIC')

    pstatic_file = os.path.join(args.data_path, f'PSTATIC_{file_suffix}.DTA')
    if args.kst_path is not None:
        kst_file = os.path.join(args.kst_path, 'Kst_IVT.DTA')
    else:
        kst_file = None
    dtaq.run(pstat, pstatic_VDC, args.pstatic_duration, args.pstatic_sample_period,
             i_min=pstatic_i_min, i_max=pstatic_i_max, timeout=None,
             show_plot=show_plot, result_file=pstatic_file, kst_file=kst_file)
    print('PSTATIC done\n')

    if show_plot:
        plt.close()


def run_chrono(dtaq, pstat, args, file_suffix, show_plot=False, repeats=1):
    # Set step sizes based on chrono_v_rms. If provided, ignore other args
    if args.chrono_v_rms is not None and not args.chrono_disable_find_i:
        if dtaq.mode == 'galv':
            # Test current
            v_oc = test_ocv(pstat)
            s_rms = find_current(pstat, v_oc + args.chrono_v_rms, 2.0)
            time.sleep(1)  # rest
        else:
            s_rms = args.chrono_v_rms
        s_half_step = s_rms * np.sqrt(2)

        if args.chrono_step_type == 'dstep':
            args.chrono_s_step1 = args.chrono_s_init + 2 * s_half_step
            args.chrono_s_step2 = args.chrono_s_init
        elif args.chrono_step_type == 'geo':
            args.chrono_s_final = args.chrono_s_init + 2 * s_half_step
        else:
            args.chrono_s_step = 2 * s_half_step
    else:
        s_rms = args.chrono_s_rms

    decimate = not args.disable_decimation

    # Configure decimation
    if decimate:
        dtaq.configure_decimation(args.decimate_during, args.decimation_prestep_points, args.decimation_interval,
                                  args.decimation_factor, args.decimation_max_t_sample)

    # Configure step signal
    if args.chrono_step_type == 'dstep':
        dtaq.configure_dstep_signal(args.chrono_s_init, args.chrono_s_step1, args.chrono_s_step2, args.chrono_t_init,
                                    args.chrono_t_step1, args.chrono_t_step2,
                                    args.chrono_t_sample)
    elif args.chrono_step_type == 'mstep':
        dtaq.configure_mstep_signal(args.chrono_s_init, args.chrono_s_step, args.chrono_t_init, args.chrono_t_step,
                                    args.chrono_t_sample,
                                    args.chrono_n_steps)
    elif args.chrono_step_type == 'triple':
        dtaq.configure_triplestep_signal(args.chrono_s_init, s_rms, args.chrono_t_init, args.chrono_t_step,
                                         args.chrono_t_sample)
    # Too complicated to configure geo step flexibly
    # elif args.chrono_step_type == 'geostep':
    #     dtaq.configure_geostep_signal(args.s_init, s_rms, args.t_init, args.t_step, args.t_sample)
    else:
        raise ValueError(f'Invalid step_type {args.chrono_step_type}')

    # Get result file
    if dtaq.mode == 'galv':
        tag_letter = 'P'
    else:
        tag_letter = 'A'

    result_file = os.path.join(args.data_path, 'CHRONO{}_{}.DTA'.format(tag_letter, file_suffix))

    if args.kst_path is not None:
        kst_file = os.path.join(args.kst_path, 'Kst_IVT.DTA')
    else:
        kst_file = None

    print('Running CHRONO')
    dtaq.run(pstat, result_file=result_file, kst_file=kst_file,
             decimate=decimate, show_plot=show_plot, repeats=repeats)
    print('CHRONO done\n')

    if show_plot:
        plt.close()


def run_v_sweep(dtaq, pstat, args, file_suffix, V_oc):
    # Set step sizes based on chrono_v_rms. If provided, ignore other args
    if dtaq.mode != 'pot':
        raise ValueError('Chrono mode must be set to potentiostatic for voltage sweep')

    # Configure decimation
    decimate = not args.disable_decimation
    if decimate:
        dtaq.configure_decimation(args.decimate_during, args.decimation_prestep_points, args.decimation_interval,
                                  args.decimation_factor, args.decimation_max_t_sample)

    # Get voltages
    v_init = V_oc
    v_step = abs(args.vsweep_v_rms) * 2 * np.sqrt(2)

    # Check direction
    direction_options = ['both', 'charge', 'discharge']
    if args.vsweep_direction not in direction_options:
        raise ValueError(f'Invalid direction {args.vsweep_direction}. Options: {direction_options}')
    elif args.vsweep_direction == 'both':
        directions = ['discharge', 'charge']
    else:
        directions = [args.vsweep_direction]

    # Get result file
    result_file = os.path.join(args.data_path, 'VSWEEP_{}.DTA'.format(file_suffix))

    if args.kst_path is not None:
        kst_file = os.path.join(args.kst_path, 'Kst_IVT.DTA')
    else:
        kst_file = None

    # Run each direction
    iv_data = []
    for i, direction in enumerate(directions):
        if direction == 'charge':
            sign = 1
            v_lim = args.vsweep_v_max
        else:
            sign = -1
            v_lim = args.vsweep_v_min

        # If measuring both directions, append to file and rest between
        if i == 0:
            append_to_file = False
        else:
            append_to_file = True

            # Equilibrate at OCV between modes to ensure cell returns to steady state after deep discharge or charge
            if args.vsweep_ocv_equil:
                suffix = f'{args.file_suffix}_Vsweep-{direction}_pre'
                equil_ocv(pstat, V_oc, args.vsweep_rest_time, suffix, args.exp_notes, args.data_path, args.kst_path)

            print(f'Resting for {args.vsweep_rest_time} s between charge/discharge...')
            time.sleep(args.vsweep_rest_time)

        # Determine number of steps considering requested n_steps and voltage limits
        max_n_steps = int(np.ceil((v_lim - V_oc) / (sign * v_step)))
        n_steps = min(max_n_steps, args.vsweep_num_steps)

        # Configure mstep
        dtaq.configure_mstep_signal(v_init, sign * v_step, args.vsweep_t_init, args.vsweep_t_step,
                                    args.vsweep_t_sample,
                                    n_steps)

        print(f'Running voltage sweep in {direction} direction...')
        dtaq.run(pstat, result_file=result_file, kst_file=kst_file,
                 decimate=decimate, show_plot=False, repeats=1, append_to_file=append_to_file, i_max=1.0)

        # Retrieve step data
        # ------------------
        data = dtaq.data_array
        t = data[:, dtaq.cook_columns.index('Time')]
        vf = data[:, dtaq.cook_columns.index('Vf')]
        im = data[:, dtaq.cook_columns.index('Im')]

        # Get step times
        step_times = dtaq.get_step_times()
        # Append final measurement time
        step_times = np.append(step_times, t[-1])

        step_iv = np.empty((len(step_times), 2))

        prev_step = t[0]
        for k, step in enumerate(step_times):
            step_index = (prev_step < t) & (t < step)
            step_iv[k, 0] = np.median(im[step_index][-100:])
            step_iv[k, 1] = np.median(vf[step_index][-100:])
            prev_step = step

        iv_data.append(step_iv)

    iv_data = np.concatenate(iv_data)
    iv_df = pd.DataFrame(iv_data, columns=['Im', 'Vf'])
    iv_df.to_csv(os.path.join(args.data_path, 'VSWEEPIV_{}.DTA'.format(file_suffix)), sep='\t', index_label='Pt')

    print('VSWEEP done\n')

    return iv_df


def test_ocv(pstat, num_points=3):
    pstat.Open()
    pstat.SetCell(GamryCOM.CellOff)

    v_meas = np.empty(num_points)
    for i in range(num_points):
        v_meas[i] = pstat.MeasureV()

    v_oc = np.mean(v_meas)
    print('V_oc: {:.4f} V'.format(v_oc))
    return v_oc


def test_resistance(pstat, t_step=2.5, v_step=0.01):
    print('Testing resistance...')
    # dt_ocp = DtaqOcv()
    dt_chrono = DtaqChrono(mode='pot')

    # Measure OCV
    # dt_ocp.run(pstat, duration=1.0, t_sample=0.01, timeout=10, show_plot=False)
    # v_oc = dt_ocp.get_ocv(100)
    v_oc = test_ocv(pstat)
    print('OCV: {:.3f} V'.format(v_oc))

    dt_chrono.configure_dstep_signal(v_oc, v_oc + v_step, v_oc - v_step, 0.5, t_step, t_step, 1e-3)
    dt_chrono.run(pstat, timeout=None, decimate=False, show_plot=False)
    # result_file='C:\\Users\\jdhuang\\Documents\\Gamry_data\\220130\\220121_1b\\LabVIEW\\data\\CHRONO_r_test.DTA')

    r_est = dt_chrono.estimate_r_tot(window=None)
    print('Estimated resistance: {:.3f} ohms'.format(r_est))

    return r_est


def find_current(pstat, vdc, duration, num_points=3):
    pstat.Open()
    pstat.SetCtrlMode(GamryCOM.PstatMode)
    pstat.SetIERangeMode(True)
    pstat.SetVoltage(vdc)
    pstat.SetCell(GamryCOM.CellOn)

    start_time = time.time()
    i_meas = []
    while time.time() - start_time < duration:
        # Repeatedly measure current to update IERange
        i_meas.append(pstat.MeasureI())

    idc = np.median(i_meas[-num_points:])

    pstat.SetCell(GamryCOM.CellOff)
    pstat.SetVoltage(0)

    return idc


def run_hybrid(sequencer, pstat, args, file_suffix, show_plot=False, **kw):
    print('Running HYBRID')

    if sequencer.chrono_mode != 'galv':
        raise ValueError("run_hybrid expects a sequencer with chrono_mode = 'galv';"
                         f" received sequencer with mode {sequencer.chrono_mode}")

    if args.hybrid_eis_mode == 'pot':
        if args.hybrid_i_init != 0:
            raise ValueError('Potentiostatic EIS can only be used for hybrid measurements centered at open circuit')

    # Determine current signal amplitude to obtain desired voltage
    v_oc = None
    if args.hybrid_v_rms is not None and not args.hybrid_disable_find_i:
        # Test resistance
        v_oc = test_ocv(pstat)
        print('v_rms:', args.hybrid_v_rms)
        i_rms = find_current(pstat, v_oc + args.hybrid_v_rms, 2.0)
        z_guess = abs(args.hybrid_v_rms / i_rms)
        time.sleep(1)  # rest
        # if r_est is None:
        #     r_est = test_resistance(pstat, t_step=2.5)
        # i_rms = args.hybrid_v_rms / r_est
    else:
        # r_est = None
        i_rms = args.hybrid_i_rms
        z_guess = None

    print('i_rms: {:.2g} A'.format(i_rms))

    # Configure decimation
    decimate = not args.disable_decimation
    if decimate:
        sequencer.configure_decimation(args.decimate_during, args.decimation_prestep_points, args.decimation_interval,
                                       args.decimation_factor, args.decimation_max_t_sample)

    # Get EIS frequencies
    eis_freq = get_eis_frequencies(args.hybrid_eis_max_freq, args.hybrid_eis_min_freq, args.hybrid_eis_ppd)

    # Configure chrono step
    step_types = ['triple', 'geo']
    if args.hybrid_step_type == 'triple':
        sequencer.configure_triple_step(args.hybrid_i_init, i_rms, args.hybrid_t_init,
                                        args.hybrid_t_step, args.hybrid_t_sample, eis_freq, z_guess=z_guess)
    elif args.hybrid_step_type == 'geo':
        sequencer.configure_geo_step(args.hybrid_i_init, i_rms, args.hybrid_t_init, args.hybrid_geo_t_short,
                                     args.hybrid_t_step, args.hybrid_t_sample,
                                     args.hybrid_geo_num_scales, args.hybrid_geo_steps_per_scale,
                                     eis_freq,
                                     end_at_init=args.hybrid_geo_end_at_init, end_time=args.hybrid_geo_end_time,
                                     z_guess=z_guess)
        # s_init, s_rms, t_init, t_short, t_long, t_sample, num_scales, steps_per_scale,
        # frequencies
    else:
        raise ValueError(f'Invalid measurement_type {args.hybrid_step_type}. Options: {step_types}')
    # else:
    #     getattr(sequencer, f'configure_{args.hybrid_step_type}_step')(args.hybrid_i_init, i_rms, args.hybrid_t_init,
    #                                                                   args.hybrid_t_step, args.hybrid_t_sample,
    #                                                                   eis_freq)

    if args.hybrid_eis_mode == 'pot':
        if v_oc is None:
            v_oc = test_ocv(pstat)
        sequencer.configure_eis(eis_freq, dc_amp=v_oc, ac_amp=abs(args.hybrid_v_rms), z_guess=z_guess)

    sequencer.run(pstat, decimate=decimate, data_path=args.data_path, kst_path=args.kst_path,
                  file_suffix=file_suffix, rest_time=args.hybrid_rest_time, filter_response=args.decimate_filter,
                  eis_first=(not args.hybrid_chrono_first),
                  show_plot=show_plot, **kw)
    print('HYBRID done\n')

    if show_plot:
        plt.close()


def run_hybrid_staircase(sequencer, pstat, args, file_suffix,
                         jv_data=None):
    print('Running HYBRID staircase')

    if sequencer.chrono_mode != 'galv':
        raise ValueError("run_hybrid_staircase expects a sequencer with chrono_mode = 'galv';"
                         f" received sequencer with mode {sequencer.chrono_mode}")

    # Check staircase direction
    direction_options = ['both', 'charge', 'discharge']
    if args.staircase_direction not in direction_options:
        raise ValueError(f'Invalid direction {args.staircase_direction}. Options: {direction_options}')
    elif args.staircase_direction == 'both':
        directions = ['discharge', 'charge']
    else:
        directions = [args.staircase_direction]

    # Determine current signal amplitude to obtain desired voltage
    if jv_data is not None:
        if args.hybrid_v_rms is None:
            raise ValueError('hybrid_v_rms must be specified if jv_data is provided')
        i_rms = None

        # Get OCV from iv curve
        if type(jv_data) == pd.DataFrame:
            jv_df = jv_data
        else:
            jv_df = read_curve_data(jv_data)
        v_oc = interp1d(jv_df['Im'].values, jv_df['Vf'].values)(0)

    elif args.hybrid_v_rms is not None and not args.hybrid_disable_find_i:
        # Test resistance
        v_oc = test_ocv(pstat)
        i_rms = find_current(pstat, v_oc + args.hybrid_v_rms, 2.0)
        print('i_rms: {:.2e} A'.format(i_rms))
    else:
        v_oc = test_ocv(pstat)
        i_rms = args.hybrid_i_rms

    # print('i_rms: {:.2e} A'.format(i_rms))

    # Configure decimation
    decimate = not args.disable_decimation
    if decimate:
        sequencer.configure_decimation(args.decimate_during, args.decimation_prestep_points, args.decimation_interval,
                                       args.decimation_factor, args.decimation_max_t_sample)

    # Get EIS frequencies
    eis_freq = get_eis_frequencies(args.hybrid_eis_max_freq, args.hybrid_eis_min_freq, args.hybrid_eis_ppd)
    full_eis_freq = get_eis_frequencies(args.staircase_full_eis_max_freq, args.staircase_full_eis_min_freq,
                                        args.staircase_full_eis_ppd)

    # Configure staircase
    v_limits = (args.staircase_v_min, args.staircase_v_max)

    for i, direction in enumerate(directions):
        suffix = f'{file_suffix}_Staircase-{direction}'
        v_rms = None
        if direction == 'charge':
            if args.hybrid_v_rms is not None:
                v_rms = abs(args.hybrid_v_rms)
            if i_rms is not None:
                i_rms = abs(i_rms)
        else:
            if args.hybrid_v_rms is not None:
                v_rms = -abs(args.hybrid_v_rms)
            if i_rms is not None:
                i_rms = -abs(i_rms)

        print('jv_data:', jv_data)
        print('i_rms:', i_rms)
        print('v_rms:', v_rms)
        print('step_type:', args.hybrid_step_type)
        geo_kwargs = dict(
            t_short=args.hybrid_geo_t_short,
            num_scales=args.hybrid_geo_num_scales,
            steps_per_scale=args.hybrid_geo_steps_per_scale,
        )
        if jv_data is not None:
            sequencer.configure_staircase_from_jv(args.hybrid_i_init, v_rms, args.hybrid_t_init,
                                                  args.hybrid_t_step, args.hybrid_t_sample,
                                                  jv_data, eis_freq, args.staircase_num_steps, v_limits=v_limits,
                                                  step_type=args.hybrid_step_type, geo_kwargs=geo_kwargs)
        else:
            sequencer.configure_staircase(args.hybrid_i_init, i_rms, args.hybrid_t_init,
                                          args.hybrid_t_step, args.hybrid_t_sample,
                                          eis_freq, args.staircase_num_steps, v_limits=v_limits,
                                          v_rms_target=v_rms,
                                          step_type=args.hybrid_step_type, geo_kwargs=geo_kwargs)

        # Equilibrate at OCV between modes to ensure cell returns to steady state after deep discharge or charge
        if args.staircase_ocv_equil:
            pstatic_suffix = f'{args.file_suffix}_Staircase-{direction}_pre'
            equil_ocv(pstat, v_oc, args.staircase_equil_time, pstatic_suffix, args.exp_notes,
                      args.data_path, args.kst_path)

        # Run staircase
        print(f'Running staircase in {direction} direction')

        sequencer.run_staircase(pstat, decimate=decimate, data_path=args.data_path, kst_path=args.kst_path,
                                file_suffix=suffix,
                                equil_time=args.staircase_equil_time, rest_time=args.hybrid_rest_time,
                                run_full_eis_pre=args.staircase_run_pre_eis,
                                run_full_eis_post=args.staircase_run_post_eis,
                                # run_full_eis_post=post_eis,
                                full_frequencies=full_eis_freq, start_with_cell_off=True, leave_cell_on=False,
                                filter_response=args.decimate_filter)

    print('HYBRID staircase done\n')


def equilibrate_pstatic(pstat, args, file_suffix):
    print('Running PEQUIL')
    # Get DC voltage
    if not args.pequil_VDC_vs_VRef:
        # Measure OCV
        v_oc = test_ocv(pstat)
        pstatic_vdc = v_oc + args.pequil_VDC
    else:
        pstatic_vdc = args.pequil_VDC
    print('pequil VDC: {:.3f} V'.format(pstatic_vdc))

    # Configure dtaq. Write once per second
    dtaq = DtaqPstaticEquil(args.equil_window_seconds, slope_thresh_pct_per_minute=args.equil_slope_thresh,
                            min_wait_time_minutes=args.equil_min_wait_time, exp_notes=args.exp_notes,
                            write_mode='interval', write_interval=int(1 / args.equil_sample_period))

    # Get files
    result_file = os.path.join(args.data_path, 'PEQUIL_{}.DTA'.format(file_suffix))
    if args.kst_path is not None:
        kst_file = os.path.join(args.kst_path, 'Kst_IVT.DTA')
    else:
        kst_file = None

    dtaq.run(pstat, pstatic_vdc, args.equil_duration, args.equil_sample_period, result_file=result_file,
             kst_file=kst_file, i_min=args.pequil_i_min, i_max=args.pequil_i_max)
    print('PEQUIL done\n')


def equilibrate_gstatic(pstat, args, file_suffix):
    print('Running GEQUIL')

    # Determine DC current
    if args.gequil_VDC is not None:
        if args.gequil_VDC_vs_VRef:
            gstatic_vdc = args.gequil_VDC
        else:
            v_oc = test_ocv(pstat)
            gstatic_vdc = args.gequil_VDC + v_oc
        gstatic_idc = find_current(pstat, gstatic_vdc, 1.5)
    else:
        gstatic_idc = args.gequil_IDC
    print('gequil IDC: {:.6f} A'.format(gstatic_idc))

    # Configure dtaq. Write once per second
    dtaq = DtaqGstaticEquil(args.equil_window_seconds, slope_thresh_mv_per_minute=args.equil_slope_thresh,
                            min_wait_time_minutes=args.equil_min_wait_time, exp_notes=args.exp_notes,
                            write_mode='interval', write_interval=int(1 / args.equil_sample_period))

    # Get files
    result_file = os.path.join(args.data_path, 'GEQUIL_{}.DTA'.format(file_suffix))
    if args.kst_path is not None:
        kst_file = os.path.join(args.kst_path, 'Kst_IVT.DTA')
    else:
        kst_file = None

    dtaq.run(pstat, gstatic_idc, args.equil_duration, args.equil_sample_period, result_file=result_file,
             kst_file=kst_file, v_min=args.gequil_v_min, v_max=args.gequil_v_max)
    print('GEQUIL done\n')


def equilibrate(pstat, args, file_suffix):
    if args.equil_mode == 'pot':
        equilibrate_pstatic(pstat, args, file_suffix)
    elif args.equil_mode == 'galv':
        equilibrate_gstatic(pstat, args, file_suffix)
    else:
        raise ValueError(f'Invalid equil_mode {args.equil_mode}. Options: pot, galv')
