import tensorflow as tf
from conv_lstm_batch_size import ConvLSTM #, get_data_keras, get_train_test
from sclouds.ml.ConvLSTM.utils import r2_keras, get_xarray_dataset_for_period, get_data_keras, get_train_test

num_vars = 4
# (seq_length, self.n_lat, self.n_lon, self.NUM_INPUT_VARS),
seq_length = 6

epochs = 40
batch_size = 10
#Xtrain_dummy = tf.ones((batch_size, seq_length, 81, 161, num_vars))
#ytrain_dummy = tf.ones((batch_size, seq_length, 81, 161))

# antall filrer i hver lag.
filters = [256, 256]
# size of filters used
kernels = [3, 3]

from utils import get_xarray_dataset_for_period, get_data_keras, get_train_test
#data = get_xarray_dataset_for_period(start = '2012-01-01', stop = '2012-01-31')
#print(data)
train_start = '2004-01-01'
train_stop  = '2011-12-31'

validation_start = '2012-01-01'
validation_stop = '2013-12-31'

#train_dataset, test_dataset =  get_train_test(start, stop, model = 'ar')
train_dataset = get_xarray_dataset_for_period(start = train_start, stop = train_stop)
X_train, y_train = get_data_keras(train_dataset, num_samples = None, seq_length = 24,
                                        batch_size = batch_size, data_format='channels_last')

data = get_xarray_dataset_for_period(start = validation_start, stop = validation_stop)
X_val, y_val = get_data_keras(data, num_samples = None, seq_length = 24, batch_size = batch_size,
                data_format='channels_last')

#batch_size = 10

ms_train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
ms_batch_train_ds = ms_train_ds.batch(batch_size, drop_remainder=True)

ms_train_ds_val = tf.data.Dataset.from_tensor_slices((X_val, y_val))
ms_batch_train_ds_val = ms_train_ds_val.batch(batch_size, drop_remainder=True)

model = ConvLSTM(ms_batch_train_ds, ms_batch_train_ds_val, filters=filters,
                 kernels=kernels, seq_length = seq_length,
                 epochs=epochs, batch_size = batch_size, validation_split=None,
                 name = 'Model2', result_path = '/home/hannasv/results/')
