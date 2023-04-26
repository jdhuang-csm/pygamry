import numpy as np
from scipy.optimize import minimize


def estimate_next_i(i_hist, v_hist, v_step, penalty_vec, deg=2, num_points=3, prev_step_prior=0,
                    i_offset=None, v_offset=0, i_lambda=0):
    """
    Estimate the next polarization current that will result in a voltage change of v_step
    :param i_hist: list of current values
    :param v_hist: list of voltage values
    :param v_step: desired voltage step size
    :param penalty_vec: ridge penalty strength for polynomial terms (ascending order)
    :param deg: polynomial degree
    :param num_points: number of points to fit
    :param prev_step_prior:
    :param i_offset: Offset applied to current. If None, offsets by initial current in fit window
    :param v_offset: Offset applied to voltage. If None, offsets by initial voltage in v_hist
    :param i_lambda: Penalty applied to current step size
    :return:
    """
    # Get target voltage
    v_next = v_hist[-1] + v_step

    # Select the last n points to use for fit
    i_opt = np.array(i_hist[-num_points:])
    v_opt = np.array(v_hist[-num_points:])

    # If only 2 points are available, use linear fit
    if len(i_opt) == 2:
        deg = 1
        penalty_vec = [0, 0]

    # Scale data
    i_scale = (np.max(i_opt) - np.min(i_opt)) / (len(i_opt) - 1)
    if i_offset is None:
        i_offset = i_opt[0]
    i_scaled = (i_opt - i_offset) / i_scale

    v_scale = (np.max(v_opt) - np.min(v_opt)) / (len(v_opt) - 1)
    if v_offset is None:
        v_offset = v_hist[0]
    v_scaled = (v_opt - v_offset) / v_scale
    v_next_scaled = (v_next - v_offset) / v_scale
    print('v_scale:', v_scale)

    # Basic polynomial fit
    pfit = np.polyfit(v_scaled, i_scaled, deg=deg)
    print('pfit:', pfit[::-1])

    # Regularized fit (ridge regression with additional priors)
    if np.max(penalty_vec) > 0 or prev_step_prior > 0 or i_lambda > 0:
        # Polynomial matrix for fit data
        A = np.vstack([v_scaled ** d for d in range(0, deg + 1)]).T
        # Poly vector for target voltage
        A_next = np.array([v_next_scaled ** d for d in range(0, deg + 1)])
        # Next i predicted by using previous step size
        prev_i_pred_scaled = (2 * i_hist[-1] - i_hist[-2] - i_offset) / i_scale
        # Ridge penalty matrix
        P = np.diag(penalty_vec)

        # Size of first step - used for determining scale of normal prior on current step size
        init_i_step = i_hist[1] - i_hist[0]

        def cost_func(x):
            # RSS with ridge penalty
            cost = x.T @ A.T @ A @ x - 2 * i_scaled.T @ A @ x + 0.5 * x.T @ P @ x
            # Add penalty for deviation from previous step size
            i_next = A_next @ x
            cost += prev_step_prior * (prev_i_pred_scaled - i_next) ** 2
            # Add penalty for large steps
            cost += i_lambda * ((i_next - i_scaled[-1]) / (init_i_step / i_scale)) ** 2
            return cost

        opt_res = minimize(cost_func, x0=pfit[::-1])
        x_opt = opt_res['x']
    else:
        x_opt = pfit[::-1]

    # Get next current
    i_pred = i_offset + i_scale * np.polyval(x_opt[::-1], v_next_scaled)
    # print('next i_scaled:', np.polyval(x_opt[::-1], v_next_scaled))

    return i_pred