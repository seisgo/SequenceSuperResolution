import numpy as np
import scipy as spy
import segyio
import utils
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import h5py
import sys
from shutil import copyfile

## Load seismic data to array
def loadSegy2D(fname):
    '''
    OUTPUT:
        data: seismic data stored in 2D array
        twt:  two-way traveling time
        dt:   time increment with unit 'ms'
        st:   begin time with unit 'ms'
    '''
    with segyio.open(fname, 'r', ignore_geometry=True) as sFile:
        dt = segyio.tools.dt(sFile)/1e3
        st = sFile.header[0][segyio.TraceField.DelayRecordingTime]
        twt = sFile.samples
        data = [np.copy(tr) for tr in sFile.trace]
        sFile.close()
        data = np.array(data).T
    
    print('Information of %s:\n' %fname, '\tShape: ', data.shape, 
          '\n\tst, dt: ', st, dt)
    return data, twt, dt, st

## Prepare HR and LR series
def pre_lrdata(hr_data, lr_data=None, r=2):

    if lr_data is not None:
        pass
    else:
        lr_data_t = hr_data[0::r , :]

    ns = hr_data.shape[0]
    lr_data = np.zeros(hr_data.shape)
    for i in range(lr_data_t.shape[-1]):
        lr_data[:,i] = utils.upsample(lr_data_t[:,i], r)[0:ns]
    
    return lr_data

## Prepare train and validataion data
def pre_train_val(lr_data, hr_data, dim=256, stride=8, tag='train', comment=None, **kw):
    '''
    Prepare train and validation dataset
    tag: 'train' or 'test', if 'test', lr_data and hr_data should be of
         single trace for mantaining trace indexing.
    '''
    lr_data = np.reshape(lr_data, (lr_data.size, 1))
    hr_data = np.reshape(hr_data, (hr_data.size, 1))
    if 'norm' in kw:
        print("Normalization Type ", kw['norm'])
        if kw['norm'] is '-1_1':
            lr_data = MinMaxScaler((-1,1)).fit_transform(lr_data)
            hr_data = MinMaxScaler((-1,1)).fit_transform(hr_data)
        elif kw['norm'] is '0_1':
            lr_data = MinMaxScaler().fit_transform(lr_data)
            hr_data = MinMaxScaler().fit_transform(hr_data)
        elif kw['norm'] is 'stand':
            lr_data = StandardScaler().fit_transform(lr_data)
            hr_data = StandardScaler().fit_transform(hr_data)

    lr_patches = list()
    hr_patches = list()

    for i in range(0, lr_data.size-dim, stride):
        lr_patch = lr_data[i:i+dim]
        hr_patch = hr_data[i:i+dim]

        lr_patches.append(lr_patch)
        hr_patches.append(hr_patch)

    hr_len = len(hr_patches)
    lr_len = len(lr_patches)
    
    hr_patches = np.array(hr_patches[0:hr_len])
    lr_patches = np.array(lr_patches[0:lr_len])
    
    print('High resolution(Y) dataset shape is ',hr_patches.shape)
    print('Low resolution(X) dataset shape is ',lr_patches.shape)
    
    if comment is None:
        dataset_name = 'data/dim%d-strd%d-%s.h5'%(dim, stride, tag)
    else:
        dataset_name = 'data/dim%d-strd%d-%s-%s.h5'%(dim, stride, tag, comment)

    return lr_patches, hr_patches, dataset_name
    
def save(dataset_name, X, Y):
    with h5py.File(dataset_name, 'w') as f:
        data_set = f.create_dataset('data', X.shape, np.float32) # lr
        label_set = f.create_dataset('label', Y.shape, np.float32) # hr
        data_set[...] = X
        label_set[...] = Y
    print('save complete -> %s'%(dataset_name))

def pre_mldata(lr_fname, hr_fname, dim=256, stride=8, lBeg=0, lEnd=None, lGap=1, comment=None, **kw):
    # load raw seismic data
    lr_seis, _, _, _ = loadSegy2D(lr_fname)
    hr_seis, _, _, _ = loadSegy2D(hr_fname) 
    
    if lEnd is None:
        lEnd = hr_seis.shape[-1]
    
    # preprocess raw seismic data
    for i in range(lBeg,lEnd,lGap):
        # get begin and end index of certain trace
        if 'skipzero' in kw and kw['skipzero'] is True:
            idx = np.nonzero(hr_seis[:,i])
            bidx = idx[0][0]
            eidx = idx[0][-1] + 1 # Note: the last value is valid, so for array slicing, idx + 1
        else:
            bidx = 0
            eidx = None
        # stack each trace to make the whole dataset
        if i==0:
            lr_data = lr_seis[bidx:eidx,i].squeeze()
            hr_data = hr_seis[bidx:eidx,i].squeeze()
        else:
            lr_data = np.hstack((lr_data, lr_seis[bidx:eidx,i].squeeze()))
            hr_data = np.hstack((hr_data, hr_seis[bidx:eidx,i].squeeze()))

    # prepare train and val data
    lr, hr, name = pre_train_val(lr_data, hr_data, dim=dim, stride=stride, tag='train', comment=comment, **kw)
    save(name, lr, hr)

def pre_val_data(lr_fname, hr_fname, lineRange=None, **kw):
    # load raw seismic data
    lr_seis, lr_twt, _, _ = loadSegy2D(lr_fname)
    hr_seis, hr_twt, _, _ = loadSegy2D(hr_fname) 

    if lineRange and lineRange[0]>=0 and lineRange[-1]<=hr_seis.shape[-1]:
        colRange = lineRange
    else:
        colRange = range(hr_seis.shape[-1])
    # preprocess raw seismic data
    lr_list = list()
    hr_list = list()
    for i in colRange:
        # get begin and end index of certain trace
        if 'skipzero' in kw and kw['skipzero'] is True:
            idx = np.nonzero(hr_seis[:,i])
            bidx = idx[0][0]
            eidx = idx[0][-1] + 1 # Note: the last value is valid, so for array slicing, idx + 1
        else:
            bidx = 0
            eidx = None
        lr_list.append(lr_seis[bidx:eidx,i])
        hr_list.append(hr_seis[bidx:eidx,i])
    
    return lr_list, hr_list

def pre_prdt_data(fname, **kw):
    '''
    Description: prepare seismic data for prediction
    INPUT:
        fname: input seismic segy file name
    OUTPUT:
        seisList: valid seismic data stored in list, whose length indicates trace number
        idxList:  trace idenx, begin and end time index of valid seismic data of each trace, 
                  same length with seisList 
        seisShape:  shape of input seismic data
    '''
    # load seismic data
    seisData, _, _, _ = loadSegy2D(fname)

    # preprocess raw seismic data
    seisList = list()
    idxList = list()
       
    for i in range(seisData.shape[-1]):
        idx = np.nonzero(seisData[:,i])
        if idx[0].size==0:
            continue
        # get begin and end index of certain trace
        if 'skipzero' in kw and kw['skipzero'] is True:
            bidx = idx[0][0]
            eidx = idx[0][-1] + 1 # Note: the last value is valid, so for array slicing, idx + 1
        else:
            bidx = 0
            eidx = None
        seisList.append(seisData[bidx:eidx,i])
        idxList.append((i, bidx, eidx))
    
    return seisList, idxList, seisData.shape

def write_segy(infile, outfile, data):
    '''
    Note: dtype of data should be np.single
    '''
    copyfile(infile, outfile)
    with segyio.open(outfile, 'r+', ignore_geometry=True) as sfile:
        for i in range(sfile.tracecount):
            # The usage of copy is to make array C_Contiguous = True
            sfile.trace[i] = data[:, i].copy(order='C')

def pre_h5file(flag=True):
    if flag is False:
        return
    fname = './data/l1126.amp_0.5ms.segy'
    hr_data, _, _, _ = loadSegy2D(fname)
    lr_data4 = pre_lrdata(hr_data, r=4) 
    lr, hr, name = pre_train_val(lr_data4, hr_data)
    print(lr.shape, hr.shape, name)
    save(name, lr, hr)

# @20201217
def prepData_marmousi():
    # Prepare machine learning data set
    rname = './data/syn_ricker.segy'
    sname = './data/syn_wide_b_ricker.segy'
    pre_mldata(rname, sname, dim=512, stride=256, lGap=20, comment='Mar2')

if __name__ == "__main__":
    prepData_marmousi()