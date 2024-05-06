'''
Author: Zhenyu Yuan
Email: zhenyuyuan@outlook.com
Brief: high resolution seismic processing by a sequential convolutional neural network
Parts: model training, validation and prediction
'''
import os
import tensorflow.keras as keras
import h5py
import pre_data
import model
import utils
from sklearn.model_selection import train_test_split
import numpy as np

# LOAD_WEIGHTS = True
WEIGHTS_PATH = 'weights/'
WEIGHTS_FILE = 'ssr-model-Mar2-ricker30-wbricker.h5'

def run_train(DATA_FILE=None, ns=256, log_path='./log', log_file='log.csv', hist_file='hist.pkl'):
    # Hyper params
    BATCH_SIZE = 32
    if DATA_FILE is None:
        DATA_FILE = './data/dim256-strd8-train-30_wb1-173Hz.h5'
    VALID_SPLIT = 0.1
    SHUFFLE = True
    MINI_EPOCH = 200 # set each dataset's epochs
    
    sr_model = model.ssr_model(ns=ns)
    sr_model = utils.load_weights(sr_model, os.path.join(WEIGHTS_PATH, WEIGHTS_FILE), False)
    sr_model = utils.compile_model(sr_model)
    
    # Train Model
    checkpointer = keras.callbacks.ModelCheckpoint(filepath=os.path.join(WEIGHTS_PATH, WEIGHTS_FILE),
                                                   verbose=1, save_best_only=True) 
    earlystopper = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0)
    os.makedirs(log_path,exist_ok=True)
    csv_log = keras.callbacks.CSVLogger(os.path.join(log_path,log_file))
    tensboard_cb = keras.callbacks.TensorBoard(log_dir=log_path,histogram_freq=1)

    lr_train, hr_train = utils.load_h5(DATA_FILE)

    hist = sr_model.fit(lr_train, hr_train, batch_size=BATCH_SIZE, epochs=MINI_EPOCH, shuffle=SHUFFLE,
                        callbacks=[checkpointer,earlystopper,csv_log,tensboard_cb], validation_split=VALID_SPLIT)
    print('Training Finish')
    utils.save_weights(sr_model, os.path.join(WEIGHTS_PATH, WEIGHTS_FILE))
    utils.save_history(os.path.join(log_path, hist_file), hist)

    _, lr_test, _, hr_test = train_test_split(lr_train, hr_train, test_size=0.2, random_state=42)
    score = sr_model.evaluate(lr_test, hr_test)
    print('Evaluating Finish, With Score:', score)
    
    return hist

def run_val(lr_name=None, hr_name=None, **kw):
    # Load ML Model
    if 'ns' in kw:
        ns = kw['ns']
    else:
        ns = 256
    sr_model = model.ssr_model(summary=False, ns=ns)
    sr_model = utils.load_weights(sr_model, os.path.join(WEIGHTS_PATH, WEIGHTS_FILE), True)
    sr_model = utils.compile_model(sr_model)
    
    # Evaluate Model
    if lr_name is None:
        lr_name = './data/syn_seis_30Hz.sgy'
    if hr_name is None:
        hr_name = './data/syn_seis_wb1-173Hz.sgy'
    if 'lineRange' in kw:
        lineRange = kw['lineRange']
    else:
        lineRange = None
    lr_list, hr_list = pre_data.pre_val_data(lr_name, hr_name, lineRange)
    print('Sampling number: ', len(lr_list))

    # Make Prediction
    sp_idx = 0
    if 'sp_idx' in kw:
        sp_idx = kw['sp_idx']
        if sp_idx>=len(lr_list):
            sp_idx = 0
    print('Sampling index: ', sp_idx)
    data = lr_list[sp_idx].reshape(1,-1,1)
    data = utils.sequence_resize(data, length=ns)[0,...]
    data = data.reshape(1,-1,1)
    result = sr_model.predict(data, verbose=1)
    print('Prediction Finish!')
    # print(result)
    pre_data.plot_lr_hr(data.flatten(), result.flatten(), 
                        typ1='LR Synthetic', typ2='HR Processed', crv_nmlz=True)
    
    # Compare real and predicted curve
    data_h = utils.sequence_resize(hr_list[sp_idx].reshape(1,-1,1),length=ns)[0,...]
    pre_data.plot_lr_hr(data_h.flatten(), result.flatten(), 
                        typ1='HR Synthetic', typ2='HR Processed', crv_nmlz=True)

    pre_data.plot_lr_hr(data.flatten(), data_h.flatten(), result.flatten(),
                        typ1='LR Synthetic', typ2='HR Synthetic', typ3='HR Processed',crv_nmlz=True,**kw)

    # evaluate
    data_h = data_h.reshape(1,-1,1)
    test_scores = sr_model.evaluate(data, data_h, verbose=1)
    print(test_scores)


def run_predict(fname=None, outfile=None, **kw):
    # Load ML Model
    if 'ns' in kw:
        ns = kw['ns']
    else:
        ns = 256
    sr_model = model.ssr_model(summary=False, ns=ns)
    sr_model = utils.load_weights(sr_model, os.path.join(WEIGHTS_PATH, WEIGHTS_FILE), True)
    sr_model = utils.compile_model(sr_model)
    
    # Load seismic data for prediction
    if fname is None:
        fname = './data/real_seis.sgy'
    seisList, idxList, sShape = pre_data.pre_prdt_data(fname, **kw)

    # Make Prediction
    ntrc = len(seisList)  # trace number with valid trace value
    prdtData = np.zeros(sShape, np.single)
    for i in range(ntrc):
        data = seisList[i].reshape(1,-1,1)
        data = utils.sequence_resize(data,length=ns)
        result = sr_model.predict(data, verbose=0)
        trcIdx = idxList[i][0]
        bIdx = idxList[i][1]
        eIdx = idxList[i][2]
        if eIdx is not None:
            nIdx = eIdx - bIdx
        else:
            nIdx = sShape[0]
        prdtData[bIdx:eIdx, trcIdx] = result.ravel()[0:nIdx]

    print('Prediction Finish!')

    if outfile is None:
        outfile = './data/prdt_seis.sgy'
    pre_data.write_segy(fname, outfile, prdtData)
    print('Write prediction to file success.')

def prc_marmousi():
    run_train(DATA_FILE='./data/dim512-strd256-train-Mar2.h5', ns=512)
    run_val(lr_name='./data/syn_ricker.segy', hr_name='./data/syn_wide_b_ricker.segy', 
            lineRange=[0,13000,100], ns=512)
    lr_name = './data/syn_seis_30Hz.sgy'
    pr_name = './data/syn_seis_30Hz_prdt_byMar2.segy'
    run_predict(fname=lr_name, outfile=pr_name, ns=512, skipzero=True)
    
if __name__ == '__main__':
    # prc_marmousi2()
    # hist=utils.load_history('./log/hist.pkl')
    # utils.plot_history(hist)
    # utils.plot_history(hist,feature='SNR')
    # prc_marmousi2_20210316()
    # prc_welllog_wb1_173_20210317()
    #seismic_show_20210317()
    seismic_show_202310()