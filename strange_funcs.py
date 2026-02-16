import numpy as np
from scipy.signal import butter, filtfilt

# band-pass funcs (the following three)
def butter_lowpass_filter(data, lowcut, fs, order):
  nyq = fs/2
  low = lowcut/nyq
  b, a = butter(order, low, btype='low')
  y = filtfilt(b, a, data) # zero-phase filter # data: [ch x time]
  return y

def butter_highpass_filter(data, highcut, fs, order):
  nyq = fs/2
  high = highcut/nyq
  b, a = butter(order, high, btype='high')
  y = filtfilt(b, a, data) # zero-phase filter
  return y

def butter_bandpass_filter(data, lowcut, highcut, fs, order):
  nyq = fs/2
  low = lowcut/nyq
  high = highcut/nyq
  b, a = butter(order, [low, high], btype='band')
  # demean before filtering
  meandat = np.mean(data, axis=1)
  data = data - meandat[:, np.newaxis]
  y = filtfilt(b, a, data)
  return y


# extract epoch from 2D data into 3D [ch x time x trial]
# input: event, baseline, frame
# extract epoch = baseline[0] to frame[2]
# for memory pre-allocation
def extractEpoch3D(data, event, srate, baseline, frame, opt_keep_baseline):
  if opt_keep_baseline == True:
    begin_tmp = int(np.floor(baseline[0]/1000*srate))
    end_tmp = int(begin_tmp+np.floor(frame[1]-baseline[0])/1000*srate)
  else:
    begin_tmp = int(np.floor(frame[0]/1000*srate))
    end_tmp = int(begin_tmp+np.floor(frame[1]-frame[0])/1000*srate)
  
  epoch3D = np.zeros((data.shape[0], end_tmp-begin_tmp, len(event)))
  nth_event = 0

  for i in event:
    if opt_keep_baseline == True:
      begin_id = int(i + np.floor(baseline[0]/1000 * srate))
      end_id = int(begin_id + np.floor((frame[1]-baseline[0])/1000*srate))
    else:
      begin_id = int(i + np.floor(frame[0]/1000 * srate))
      end_id = int(begin_id + np.floor((frame[1]-frame[0])/1000*srate))
    
    tmp_data = data[:, begin_id:end_id]

    begin_base = int(np.floor(baseline[0]/1000 * srate))
    end_base = int(begin_base + np.floor(np.diff(baseline)/1000 * srate)-1)
    base = np.mean(tmp_data[:, begin_base:end_base], axis=1)

    rmbase_data = tmp_data - base[:, np.newaxis]
    epoch3D[:, :, nth_event] = rmbase_data
    nth_event = nth_event + 1

  return epoch3D


def processRun(dataset, baseline = [-200, 0], frame = [0, 600]):
    data = np.asarray(dataset['data'])
    data -= data.mean(axis=1)[:, None]
    srate = dataset['srate']
    data = butter_bandpass_filter(data, 0.5, 10, srate, 4)
    markers = dataset['markers_target']

    ID = np.where(markers!=0)[0]
    eegmat = extractEpoch3D(data, ID, srate, baseline, frame, False)

    return([eegmat.transpose([2, 0, 1]), markers[ID] == 1, dataset['markers_seq'][ID]])

def infer_nseq_per_trial(flash_code, is_target):
    target_keys = np.sort(flash_code[is_target].reshape([-1, 2])).astype(int).astype(str)
    target_keys = np.char.add(np.char.add(target_keys[:,0], '-'), target_keys[:,1])
    return np.min(np.unique(target_keys, return_counts=True)[1])

def reformat_data(eeg_mat, is_target, flash_code):
  nseq = infer_nseq_per_trial(flash_code, is_target)
  #
  indi_sort_code = np.argsort(flash_code.reshape([-1, 12]), axis=1)
  n_flash, n_channel, n_time = eeg_mat.shape
  X = np.take_along_axis(
      eeg_mat.reshape([-1, 12, n_channel, n_time]), 
      indi_sort_code[:, :, None, None], 
      axis=1).reshape([-1, nseq, 2, 6, n_channel, n_time])
  y = np.take_along_axis(
      is_target.reshape([-1, 12]), 
      indi_sort_code, 
      axis=1).reshape([-1, nseq, 2, 6]).astype(int)
  return X, y

def processRun2(dataset, baseline = [-200, 0], frame = [0, 600]):
  if isinstance(dataset, list):
    Xys = [processRun2(ds, baseline, frame) for ds in dataset]
    return np.concatenate([Xy[0] for Xy in Xys]), np.concatenate([Xy[1] for Xy in Xys])
  eeg_mat, is_target, flash_code = processRun(dataset, baseline, frame)
  return reformat_data(eeg_mat, is_target, flash_code)

def evaluate_chr_accu(yhat, y):
  correct = yhat.cumsum(axis=1).argmax(axis=3) == y.argmax(axis=3)
  return np.mean(correct.sum(axis=2) == 2, axis=0)