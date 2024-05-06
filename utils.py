'''
Author: Zhenyu Yuan
Email: zhenyuyuan@outlook.com
Brief: some utility functions for sequence super resolution model training
'''
import os
import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt
from scipy import signal
import h5py
import pickle
import pandas as pd

class History_trained_model(object):
    def __init__(self, history, epoch, params):
        self.history = history
        self.epoch = epoch
        self.params = params

def save_history(filename, history):
    with open(filename, 'wb') as file:
        model_history= History_trained_model(history.history, history.epoch, history.params)
        pickle.dump(model_history, file, pickle.HIGHEST_PROTOCOL)

def load_history(filename):
    with open(filename, 'rb') as file:
        history=pickle.load(file)
    # print(history.history)
    return history

def plot_history(history,xlbl='Epoch',ylbl='MSE',feature='sum_loss',fname=''):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel(xlbl, fontweight='bold')
    plt.ylabel(ylbl, fontweight='bold')
    plt.plot(hist['epoch'], hist[feature], label='Train Performance')
    plt.plot(hist['epoch'], hist['val_{}'.format(feature)], label = 'Val Performance')
#   plt.ylim([0,5])
    plt.legend()
    plt.show()
  
    if fname is not None and type(fname) is str:
        plt.savefig(fname, dpi=300, bbox_inches='tight')

def load_h5_list(dirname):
    datasets = []
    filenames = os.listdir(dirname)
    file_extensions = set(['.h5'])
    for filename in filenames:
        ext = os.path.splitext(filename)[-1]
        if ext in file_extensions:
            full_filename = os.path.join(dirname, filename)
            datasets.append(full_filename)
    return datasets

def load_h5(dataset_name):
    print('Loading dataset ',dataset_name)
    with h5py.File(dataset_name, 'r') as hf:
        X = (hf['data'][:])
        Y = (hf['label'][:])
    print(X.shape)
    print(Y.shape)
    return X, Y

def SNR(y_true, y_pred):
    P = y_pred
    Y = y_true
    sqrt_l2_loss = K.sqrt(K.mean((P-Y)**2 + 1e-6))
    sqrn_l2_norm = K.sqrt(K.mean(Y**2))
    snr = 20 * K.log(sqrn_l2_norm / sqrt_l2_loss + 1e-8) / K.log(10.)
    avg_snr = K.mean(snr)
    return avg_snr

def sum_loss(y_true, y_pred):
    P = y_pred
    Y = y_true
    loss = K.sum((P-Y)**2)
    return loss

def compile_model(model):
    model.compile(loss='mse', optimizer="adam", metrics=[sum_loss, SNR])
    return model

# load before compile
def load_weights(model, weights_file, load_weights=False):
    if load_weights: 
        model.load_weights(weights_file)
        print('load model weights success!')
    return model

# save weights
def save_weights(model, weights_file, save_weights=True):
    if save_weights:
        model.save_weights(weights_file)
        print('save model weights success!')

# load model
# NOTE: Loading model containing self-defined layer ('SubPixel1D') failed.
def load_model(model_file, load_model=False):
    if load_model and os.path.exists(model_file): 
        model = keras.models.load_model(model_file)
        print('load model success!')
        return model
    
    return None

# save model
def save_model(model, model_file, save_model=True):
    if save_model:
        model.save(model_file)
        print('save model success!')

# resize sequence to certain length
def sequence_resize(signal, length=256):
    '''
    assume initial shape of signal is [1,:,1]
    '''
    size = signal.size 
    if size<length:
        data = np.append(signal, np.zeros(length-size))
        data = data.reshape(1,length,1)
    elif size==length:
        data = signal
    else:
        dim1 = size//length + 1
        nadd = dim1 * length - size
        data = np.append(signal, np.zeros(nadd))
        data = data.reshape(-1,length,1)
    return data

def upsample(x_lr, r):
    x_lr = x_lr.flatten()
    x_hr_len = len(x_lr) * r
    x_sp = np.zeros(x_hr_len)

    i_lr = np.arange(x_hr_len, step=r)
    i_hr = np.arange(x_hr_len)

    f = interpolate.splrep(i_lr, x_lr)
    x_sp = interpolate.splev(i_hr, f)

    return x_sp
