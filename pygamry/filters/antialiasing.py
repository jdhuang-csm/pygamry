import numpy as np
from scipy import ndimage

from pygamry.filters._filters import empty_gaussian_filter1d
from pygamry.utils import identify_steps, split_steps, robust_std, pdf_normal


def nonuniform_gaussian_filter1d(a, sigma, axis=-1, empty=False,
                                 mode='reflect', cval=0.0, truncate=4, sigma_node_factor=1.5, min_sigma=0.25):
    if np.max(sigma) > 0:
        sigma = np.copy(sigma)
        sigma[sigma <= 0] = 1e-10

        # Get sigma nodes
        min_ls = max(np.min(np.log10(sigma)), np.log10(min_sigma))
        max_ls = max(np.max(np.log10(sigma)), np.log10(min_sigma))
        num_nodes = int(np.ceil((max_ls - min_ls) / np.log10(sigma_node_factor))) + 1
        sigma_nodes = np.logspace(min_ls, max_ls, num_nodes)

        if np.min(sigma) < min_sigma:
            # If smallest sigma is below min effective value, insert dummy node at lowest value
            # This node will simply return the original array

            # Determine factor for uniform node spacing
            if len(sigma_nodes) > 1:
                factor = sigma_nodes[-1] / sigma_nodes[-2]
            else:
                factor = sigma_node_factor

            # Limit requested sigma values to 2 increments below min effective sigma
            # This will ensure that any sigma values well below min_sigma will not be filtered, while those
            # close to min_sigma will receive mixed-lengthscale filtering as intended
            sigma[sigma < min_sigma / (factor ** 2)] = min_sigma / (factor ** 2)

            # Insert as many sigma values as needed to get to lowest requested value (max 2 inserts)
            while sigma_nodes[0] > np.min(sigma) * 1.001:
                sigma_nodes = np.insert(sigma_nodes, 0, sigma_nodes[0] / factor)

        # print(sigma_nodes)
        if len(sigma_nodes) > 1:
            node_delta = np.log(sigma_nodes[-1] / sigma_nodes[-2])
        else:
            node_delta = 1

        def get_node_weights(x):
            # Tile x and nodes to same shape with extra axis
            tile_shape = np.ones(np.ndim(x) + 1, dtype=int)
            tile_shape[0] = len(sigma_nodes)
            # print('x:', x)
            x_tile = np.tile(x, tile_shape)
            node_tile = np.tile(sigma_nodes, (*x.shape, 1))
            node_tile = np.moveaxis(node_tile, -1, 0)

            nw = np.abs(np.log(x_tile / node_tile)) / node_delta
            nw[nw >= 1] = 1
            nw = 1 - nw
            # print('min weight:', np.min(nw))
            # print('max weight:', np.max(nw))
            # print('min weight sum:', np.min(np.sum(nw, axis=0)))
            # print('max weight sum:', np.max(np.sum(nw, axis=0)))
            return nw

        node_outputs = np.empty((len(sigma_nodes), *a.shape))
        for i in range(len(sigma_nodes)):
            if sigma_nodes[i] < min_sigma:
                # Sigma is below minimum effective value
                if empty:
                    # For empty filter, still need to apply filter to determine central value
                    node_outputs[i] = empty_gaussian_filter1d(a, sigma=min_sigma, axis=axis, mode=mode, cval=cval,
                                                              truncate=truncate)
                else:
                    # For standard filter, reduces to original array
                    node_outputs[i] = a
            else:
                if empty:
                    node_outputs[i] = empty_gaussian_filter1d(a, sigma=sigma_nodes[i], axis=axis, mode=mode, cval=cval,
                                                              truncate=truncate)
                else:
                    node_outputs[i] = ndimage.gaussian_filter1d(a, sigma=sigma_nodes[i], axis=axis, mode=mode, cval=cval,
                                                                truncate=truncate)

        node_weights = get_node_weights(sigma)
        # print(node_weights.shape, node_outputs.shape)
        # print(np.sum(node_weights, axis=0))

        out = node_outputs * node_weights
        return np.sum(out, axis=0)

    else:
        # No filtering to perform on this axis
        return a


def filter_chrono_signal(times, y, step_index=None, input_signal=None, decimate_index=None,
                         sigma_factor=0.01, max_sigma=None,
                         remove_outliers=False, outlier_kw=None, median_prefilter=False, **kw):
    if step_index is None and input_signal is None:
        raise ValueError('Either step_index or input_signal must be provided')

    if step_index is None:
        step_index = identify_steps(input_signal, allow_consecutive=False)

    if remove_outliers:
        y = y.copy()

        # First, remove obvious extreme values
        # ext_index = identify_extreme_values(y, qr_size=0.8)
        # print('extreme value indices:', np.where(ext_index))
        # y[ext_index] = ndimage.median_filter(y, size=31)[ext_index]

        # Find outliers with difference from filtered signal
        # Use median prefilter to avoid spread of outliers
        y_filt = filter_chrono_signal(times, y, step_index=step_index,
                                      sigma_factor=sigma_factor, max_sigma=max_sigma,
                                      remove_outliers=False,
                                      empty=False, median_prefilter=True, **kw)
        if outlier_kw is None:
            outlier_kw = {}

        outlier_flag = flag_chrono_outliers(y, y_filt, **outlier_kw)

        print('outlier indices:', np.where(outlier_flag))

        # Set outliers to filtered value
        y[outlier_flag] = y_filt[outlier_flag]

    y_steps = split_steps(y, step_index)
    t_steps = split_steps(times, step_index)
    t_sample = np.median(np.diff(times))

    if max_sigma is None:
        max_sigma = sigma_factor / t_sample

    # Get sigmas corresponding to decimation index
    if decimate_index is not None:
        decimate_sigma = sigma_from_decimate_index(y, decimate_index)
        step_dec_sigmas = split_steps(decimate_sigma, step_index)
    else:
        step_dec_sigmas = None

    y_filt = []
    for i, (t_step, y_step) in enumerate(zip(t_steps, y_steps)):
        # Ideal sigma from inverse sqrt of maximum curvature of RC relaxation
        sigma_ideal = np.exp(1) * (t_step - (t_step[0] - t_sample)) / 2
        sigmas = sigma_factor * (sigma_ideal / t_sample)
        sigmas[sigmas > max_sigma] = max_sigma

        # Use decimation index to cap sigma
        if step_dec_sigmas is not None:
            sigmas = np.minimum(step_dec_sigmas[i], sigmas)

        if median_prefilter:
            y_in = ndimage.median_filter(y_step, 3, mode='nearest')
        else:
            y_in = y_step

        yf = nonuniform_gaussian_filter1d(y_in, sigmas, **kw)

        y_filt.append(yf)

    return np.concatenate(y_filt)


def sigma_from_decimate_index(y, decimate_index, truncate=4.0):
    sigmas = np.zeros(len(y)) #+ 0.25

    # Determine distance to nearest sample
    diff = np.diff(decimate_index)
    ldiff = np.insert(diff, 0, diff[0])
    rdiff = np.append(diff, diff[-1])
    min_diff = np.minimum(ldiff, rdiff)

    # Set sigma such that truncate * sigma reaches halfway to nearest sample
    sigma_dec = min_diff / (2 * truncate)
    sigma_dec[min_diff < 2] = 0  # Don't filter undecimated regions
    sigmas[decimate_index] = sigma_dec

    return sigmas


def flag_chrono_outliers(y_raw, y_filt, thresh=0.95, p_prior=0.01):
    dev = y_filt - y_raw
    std = robust_std(dev)
    sigma_out = np.maximum(np.abs(dev), 0.01 * std)
    print('std:', std)
    p_out = outlier_prob(dev, 0, std, sigma_out, p_prior)

    return p_out > thresh


def outlier_prob(x, mu_in, sigma_in, sigma_out, p_prior):
    """
    Estimate outlier probability using a Bernoulli prior
    :param ndarray x: data
    :param ndarray mu_in: mean of inlier distribution
    :param ndarray sigma_in: standard deviation of inlier distribution
    :param ndarray sigma_out: standard deviation of outlier distribution
    :param float p_prior: prior probability of any point being an outlier
    :return:
    """
    pdf_in = pdf_normal(x, mu_in, sigma_in)
    pdf_out = pdf_normal(x, mu_in, sigma_out)
    p_out = p_prior * pdf_out / ((1 - p_prior) * pdf_in + p_prior * pdf_out)
    dev = np.abs(x - mu_in)
    # Don't consider data points with smaller deviations than sigma_in to be outliers
    p_out[dev <= sigma_in] = 0
    return p_out