import matplotlib
matplotlib.use('Agg')

import numpy as np
import collections
from functools import partial
from os.path import expanduser
from tqdm import tqdm
import h5py

from surround.image_processing import *
from surround.data_handling import *
from surround.efficient_coding import *
from aesthetics.plotting import *
from surround.modeling import gaussian, difference_of_gaussians

import pyret.filtertools as ft
from sklearn.decomposition import PCA
from scipy.stats import sem
from scipy.optimize import curve_fit, minimize
from scipy.interpolate import interp1d
from scipy.stats import pearsonr

from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import convolve2d
from scipy.misc import imresize

import tensorflow as tf

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


# LOAD CONSTANTS AND DATA
signal = np.array(np.load(os.path.expanduser('~/data/surround/signal_3_23.npy')))
constants = np.load(os.path.expanduser('~/data/surround/2017_10_9_diversity_constants.npy')).item()
fits = np.load(os.path.expanduser('~/data/surround/fits_3_23.npy')).item()
all_params = np.load(os.path.expanduser('~/data/surround/params_3_23.npy')).item()
variances = np.load(os.path.expanduser('~/data/surround/variances_3_23.npy')).item()
mean_squared_errors = np.load(os.path.expanduser('~/data/surround/mse_3_23.npy')).item()
abs_errors = np.load(os.path.expanduser('~/data/surround/abserrs_3_23.npy')).item()


def generate_spatial_signals(batch_size, signal=signal):
    # We generate white noise sequences, then multiply their frequency spectra
    # by the signal frequency spectra to make it look like a natural sequence.
    random_seq = [np.random.randn(2 * len(signal) - 1) for b in range(batch_size)]
    spatial_seq = [np.fft.irfft(np.fft.rfft(s) * signal) for s in random_seq]
    return np.stack(spatial_seq)

def rf_model(horz_weight, center_weight):
    return center_weight*constants['center'] + (1-center_weight)*(
            horz_weight*constants['horz_pf'] + (1-horz_weight)*constants['ama_pf'])

def encoder(horz_weight, center_weight):
    tf_center = tf.constant(constants['center'], dtype=tf.float32)
    tf_horz_pf = tf.constant(constants['horz_pf'], dtype=tf.float32)
    tf_ama_pf = tf.constant(constants['ama_pf'], dtype=tf.float32)
    tf_surround = (horz_weight * tf_horz_pf + (1 - horz_weight) * tf_ama_pf)
    return center_weight * tf_center + (1.0 - center_weight) * tf_surround

diff_of_gauss_mu0 = partial(difference_of_gaussians, mu=0)
def center_and_surround(space, center_width, surround_width, center_strength, surround_strength):
    return diff_of_gauss_mu0(space, abs(center_width), abs(surround_width),
                            -abs(center_strength), abs(surround_strength))

batch_size = 320 
max_channel_depth = 10
decoder_size = 200
x = generate_spatial_signals(batch_size)
input_len = x.shape[1]
output_len = x.shape[1]
max_steps = 10000
learning_rate = 0.01
global_step = tf.Variable(1, name='global_step', trainable=False, dtype=tf.int32)
increment_global_step_op = tf.assign(global_step, global_step+1)
lr = tf.train.polynomial_decay(learning_rate, global_step, 1000, end_learning_rate=0.00001, power=1.0, cycle=False, name=None)
gain = 1.0   # 0.5
target_snrs = [20.0]

with tf.device('/gpu:0'):
    for target_snr in target_snrs:
        for random_initialization in [False, True]:
            all_results = []
            for channel_depth in range(1, max_channel_depth+1):
                # Universal variables
                results = collections.defaultdict(list)

                g = tf.Graph()
                errors = []
                with g.as_default():
                    # GET INPUT and OUTPUT NOISE
                    n_in = tf.get_variable('noise_in', shape=(1,), dtype=tf.float32,
                            initializer=tf.constant_initializer(constants['input_noise']))
                    n_out = tf.get_variable('noise_out', shape=(channel_depth,), dtype=tf.float32,
                            initializer=tf.constant_initializer(constants['output_noise']))

                    # GET DATA READY
                    label = tf.placeholder(tf.float32, shape=(batch_size, output_len))
                    in_noise = tf.random_normal(shape=label.shape, mean=0.0, stddev=n_in, name='noisy_input')
                    noisy_input = label + in_noise
                    # inputs, num_outputs, kernel_size, stride=1

                    # GET FILTERS READY
                    # RANDOMLY INITIALIZE
                    # Horizontal weight parameters.
                    if random_initialization:
                        hw_param = tf.get_variable('horz_weights', shape=(channel_depth,), dtype=tf.float32,
                                initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
                        ideal_horz_weights = tf.nn.sigmoid(gain * hw_param)  # hw_param
                    else:
                        hw_param = tf.get_variable('horz_weights', shape=(1,), dtype=tf.float32,
                                initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
                        ideal_horz_weights = tf.nn.sigmoid(gain * hw_param)  # hw_param
                        # hw_param = tf.constant(0.144, dtype=tf.float32, shape=(channel_depth,), name='horz_weights')
                        # ideal_horz_weights = hw_param # tf.nn.sigmoid(gain * hw_param)

                    # Center weight parameters.
                    cw_param = tf.get_variable('center_weights', shape=(channel_depth,), dtype=tf.float32,
                                               initializer=tf.constant_initializer(1.0))
                    ideal_center_weights = tf.nn.sigmoid(gain * cw_param)

                    filters = []
                    filtered_output = []
                    for c in range(channel_depth):
                        # e = encoder(ideal_horz_weights[c], ideal_center_weights[c])
                        if random_initialization:
                            filters.append(encoder(ideal_horz_weights[c], ideal_center_weights[c]))
                        else:
                            filters.append(encoder(ideal_horz_weights[0], ideal_center_weights[c]))
                    kernel = tf.stack(filters, axis=-1)
                    kernel = tf.expand_dims(kernel, axis=1)
                    print('Kernel has shape %s.' %(kernel.shape,))

                    # CONVOLUTION WITH IDEAL RFS
                    distortion = tf.expand_dims(noisy_input, axis=-1)
                    encoded = tf.nn.conv1d(distortion, kernel, stride=1, padding='SAME', name='encoded')
                    out_noise = tf.random_normal(shape=encoded.shape, mean=0.0, stddev=n_out, name='output_noise')
                    noisy_encoded = encoded + out_noise
                    print('Encoded has shape %s.' %(encoded.shape,))
                    print('Noisy encoded has shape %s.' %(noisy_encoded,))

                    out = tf.layers.conv1d(noisy_encoded, filters=1, kernel_size=decoder_size, 
                                           padding='same', name='decoder')
                    variables = tf.get_collection(tf.GraphKeys.VARIABLES)
                    weights = tf.get_default_graph().get_tensor_by_name('decoder/kernel:0')
                    print('Out has shape %s.' %(out.shape,))

                    signal_mean, signal_var = tf.nn.moments(
                        tf.nn.conv1d(tf.expand_dims(label, axis=-1), kernel, stride=1, padding='SAME'), axes=[1])
                    all_noise = tf.nn.conv1d(
                        tf.expand_dims(in_noise, axis=-1), kernel, stride=1, padding='SAME', name='noise') + out_noise
                    noise_mean, noise_var = tf.nn.moments(tf.squeeze(all_noise), axes=[1])
                    snr = signal_var/noise_var
                    snr_regularization = tf.losses.mean_squared_error(
                            tf.constant(target_snr, dtype=tf.float32, shape=snr.shape), snr)

                    global_step = tf.Variable(1, name='global_step', trainable=False, dtype=tf.int32)
                    increment_global_step_op = tf.assign(global_step, global_step+1)
                    lr = tf.train.polynomial_decay(learning_rate, global_step, max_steps, 
                                                   end_learning_rate=0.00001, power=1.0, 
                                                   cycle=False, name=None)

                    mse = tf.losses.mean_squared_error(label, tf.squeeze(out))
                    loss = mse + 2. * snr_regularization
                    # opt = tf.train.GradientDescentOptimizer(lr)
                    opt = tf.train.AdamOptimizer(learning_rate=lr)
                    train_op = opt.minimize(loss)

                    with tf.Session() as sess:
                        sess.run(tf.global_variables_initializer())
                        for step in range(max_steps):
                            y = generate_spatial_signals(batch_size)
                            update, step, error, k, hw, cw, decoder, this_snr, snr_reg, ni, no = sess.run(
                                [train_op, increment_global_step_op, mse, kernel, ideal_horz_weights, ideal_center_weights,
                                 weights, snr, snr_regularization, n_in, n_out], feed_dict={label: y})
                            errors.append(error)
                            if step % 100 == 0:
                                print('Error at step %04d is %0.4f' %(step, error))
                            if step % 5000 == 0:
                                output = sess.run([out], feed_dict={label: y})[0]
                                results['hw'].append(hw)
                                results['cw'].append(cw)
                                results['snr'].append(this_snr)
                                results['snr_reg'].append(snr_reg)
                            elif step == max_steps - 1:
                                print('Error at step %04d is %0.4f' %(step, error))
                                results['input'].append(x)
                                results['labels'].append(y)
                                output = sess.run([out], feed_dict={label: y})[0]
                                results['output'].append(output)
                                results['kernel'].append(k)
                                results['hw'].append(hw)
                                results['cw'].append(cw)
                                results['decoder'].append(decoder)
                                results['snr'].append(this_snr)
                                results['snr_reg'].append(snr_reg)
                                results['input_noise'].append(ni)
                                results['output_noise'].append(no)

                # Collect results.
                this_mse = np.mean((np.squeeze(results['output'][-1]) - results['labels'][-1])**2)
                # np.mean([np.mean(
                #    (results['output'][-1][j] - results['labels'][-1][j])**2) for j in range(batch_size)])
                this_err = sem((np.squeeze(results['output'][-1]) - results['labels'][-1])**2)
                # this_err = sem([np.mean(
                #    (results['output'][-1][j] - results['labels'][-1][j])**2) for j in range(batch_size)])

                results['errors'] = errors
                results['mean_errors'] = this_mse
                results['sem_errors'] = this_err
                results['channels'] = channel_depth
                all_results.append(results)

                tf.reset_default_graph()

            if random_initialization:
                np.save('/home/lane/code/ipython-notebooks/baccuslab/2017_10_27_diversity_random_results_20.npy', all_results)
            else:
                np.save('/home/lane/code/ipython-notebooks/baccuslab/2017_10_27_diversity_homogenous_results_20.npy', all_results)



