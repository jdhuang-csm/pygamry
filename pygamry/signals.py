import numpy as np


def check_signal_size(signal):
    max_signal_size = 262143
    if len(signal) > max_signal_size:
        raise ValueError(f'Length of signal array ({len(signal)}) exceeds maximum length ({max_signal_size})')


def make_triplestep_signal(s_init, s_rms, t_init, t_step, t_sample):
    duration = t_init + 3 * t_step
    times = np.arange(0, duration + 1e-8, t_sample)
    check_signal_size(times)

    signal = np.zeros(len(times)) + s_init
    s_half_step = s_rms * np.sqrt(2)

    step_scales = [1, -2, 1]
    step_time = t_init
    step_times = []
    for step_scale in step_scales:
        signal[times >= step_time] += step_scale * s_half_step
        step_times.append(step_time)
        step_time += t_step

    return times, signal, step_times


def make_geostep_signal(s_init, s_final, s_min, s_max, t_init, t_sample, t_short, t_long,
                        num_scales, steps_per_scale, flex_thresh=0.05, end_at_init=False, end_time=None):
    if s_max <= s_min:
        raise ValueError(f's_max must be greater than s_min. Received s_min={s_min}, s_max={s_max}')

    if t_short >= t_long:
        raise ValueError('t_short must be less than t_long, but received t_short={:.3e} '
                         'and t_long={:.3e}'.format(t_short, t_long))

    if t_short <= t_sample:
        raise ValueError('t_short must be greater than t_sample')

    if num_scales < 2:
        raise ValueError('Geostep requires at least 2 duration scales (num_scales); '
                         f'received num_scales={num_scales}')

    if steps_per_scale < 1 or steps_per_scale > 3:
        raise ValueError(f'steps_per_scale must be between 1 and 3; received {steps_per_scale}')

    # Build the signal array
    # Get logarithmically spaces step lengths
    step_durations = np.logspace(np.log10(t_short), np.log10(t_long), num_scales)
    # Round to integer number of sample periods
    step_durations = np.array([round(sd / t_sample, 0) * t_sample for sd in step_durations])
    print('geostep step_durations:', step_durations)

    # Make the first step in the direction that maximizes magnitude
    s_range = s_max - s_min
    if abs((s_init - s_min) - (s_max - s_init)) < s_range * flex_thresh:
        # s_init is at center of range - first step sign is flexible
        init_sign = 0
        print('init flex')
    elif abs(s_init - s_min) > abs(s_init - s_max):
        init_sign = -1
    else:
        init_sign = 1

    # Make the last step in the direction that maximizes magnitude
    if abs((s_final - s_min) - (s_max - s_final)) < s_range * flex_thresh and init_sign != 0:
        # Flexible last step
        final_sign = 0
        print('final flex')
    elif abs(s_final - s_min) > abs(s_final - s_max):
        final_sign = 1
    else:
        final_sign = -1

    # Check if the initial and final signs are aligned.
    # If not, add a "stutter step" at the shortest timescale to align them
    tot_num_steps = num_scales * steps_per_scale
    stutter = False
    sign_switch = (-1) ** (tot_num_steps - 1)
    if init_sign == 0:
        init_sign = final_sign * sign_switch
    elif final_sign == 0:
        final_sign = init_sign * sign_switch
    elif init_sign * sign_switch != final_sign:
        stutter = True

    # Starting offset to align steps (will be overwritten when s_init is set)
    if init_sign == -1:
        signal_offset = s_max
    else:
        signal_offset = s_min

    # Get sample times
    duration = t_init + steps_per_scale * np.sum(step_durations) + int(stutter) * step_durations[0]
    times = np.arange(0, duration + 1e-8, t_sample)
    check_signal_size(times)

    signal = np.zeros(len(times)) + signal_offset
    step_times = []

    print('init_sign:', init_sign)
    print('final_sign:', final_sign)
    print('stutter:', stutter)

    # Intermediate scales
    step_time = t_init
    sign = init_sign
    for i, step_duration in enumerate(step_durations):
        if stutter and i == 0:
            num_steps = steps_per_scale + 1
        else:
            num_steps = steps_per_scale

        for j in range(num_steps):
            signal[times >= step_time] += sign * s_range
            step_times.append(step_time)
            step_time += step_duration
            sign *= -1

    # Set initial and final values
    signal[times < step_times[0]] = s_init
    signal[times >= step_times[-1]] = s_final

    if end_at_init and s_final != s_init:
        # Add a step to return to the initial signal value
        if end_time is None:
            raise ValueError('end_time must be provided if end_at_init=True')

        end_times = np.arange(times[-1] + t_sample, times[-1] + end_time + 1e-8, t_sample)

        step_times.append(times[-1])
        times = np.concatenate([times, end_times])
        signal = np.concatenate([signal, np.zeros(len(end_times))])
        signal[times >= step_times[-1]] = s_init

    return times, signal, step_times
