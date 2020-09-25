import tensorflow as tf
from conv_lstm_batch_size import ConvLSTM #, get_data_keras, get_train_test
from sclouds.ml.ConvLSTM.utils import r2_keras, get_xarray_dataset_for_period, get_data_keras, get_train_test

from utils import get_xarray_dataset_for_period, get_data_keras, get_train_test

def train_convlstm(num_vars = 4, seq_length = 6, epochs = 40, batch_size = 10,
                   filters = [256, 256], kernels = [3, 3], overwrite_results=True, drop_remainder_batch = True):
    """ Train """

    train_start = '2004-01-01'
    train_stop  = '2011-12-31'

    validation_start = '2012-01-01'
    validation_stop = '2013-12-31'
    
    test_start = '2014-01-01'
    test_stop = '2018-12-31'

    train_dataset = get_xarray_dataset_for_period(start = train_start, stop = train_stop)
    X_train, y_train = get_data_keras(train_dataset, num_samples = None, seq_length = seq_length,
                                          batch_size = batch_size, data_format='channels_last')
    print(train_dataset)
    data = get_xarray_dataset_for_period(start = validation_start, stop = validation_stop)
    X_val, y_val = get_data_keras(data, num_samples = None, seq_length = seq_length, batch_size = batch_size,
                  data_format='channels_last')
    print(data)

    data = get_xarray_dataset_for_period(start = test_start, stop = test_stop)
    X_test, y_test = get_data_keras(data, num_samples = None, seq_length = seq_length, batch_size = batch_size,
                  data_format='channels_last')
    print(data)


    ms_train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    ms_batch_train_ds = ms_train_ds.batch(batch_size, drop_remainder=drop_remainder_batch)

    ms_train_ds_val = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    ms_batch_train_ds_val = ms_train_ds_val.batch(batch_size, drop_remainder=drop_remainder_batch)

    ms_train_ds_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    ms_batch_train_ds_test = ms_train_ds_test.batch(batch_size, drop_remainder=drop_remainder_batch)


    model = ConvLSTM(ms_batch_train_ds, ms_batch_train_ds_val, filters=filters,
                   kernels=kernels, seq_length = seq_length,
                   epochs=epochs, batch_size = batch_size, validation_split=None,
                   name = 'Model2', result_path = '/home/hannasv/results_new/', test_data = ms_batch_train_ds_test)
