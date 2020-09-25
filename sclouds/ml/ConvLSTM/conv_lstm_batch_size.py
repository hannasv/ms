"""Convolutional Long-Short Term Model.
"""
import os, sys
import glob
import numpy as np
import xarray as xr
from sclouds.helpers import get_lon_array, get_lat_array, path_convlstm_results
from sclouds.ml.ConvLSTM.utils import r2_keras, mae
from tensorflow.keras.losses import MeanSquaredError
mse = MeanSquaredError()

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

def save_prediction(test_mae, test_mse, sequence_prediction):
    """ Save prediction """
    data_dict = {'tcc': (['batch', 'sequence_length', 'latitude', 'longitude'], sequence_prediction[:, :, :, :, 0])}

    longitude  = np.arange(-15.0, 25.0+0.25, step = 0.25)
    latitude   =  np.arange(30.0, 50.0+0.25, step = 0.25)
    #seq_length = sequence_prediction
    #time = np.arange(seq_length)
    print(sequence_prediction[:, :, :, :, 0].shape)
    ds = xr.Dataset(data_dict,
                     coords={'longitude': (['longitude'], longitude),
                             'latitude': (['latitude'], latitude),
                             'sequence_length': (['sequence_length'], np.arange(24)), 
                             'batch':(['batch'], np.arange(10))
                            })
    ds['mse_test']  = test_mse
    ds['mae_test']  = test_mae
    ds['date_seq']  = '2014-01-01'

    ds.to_netcdf('/home/hannasv/results_summary/test_prediction_conv_lstm.nc')
    return 
my_callbacks = [
    #tf.keras.callbacks.EarlyStopping(patience=3),
    #tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
    #tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    tf.keras.callbacks.TerminateOnNaN()
]
#model.fit(dataset, epochs=10, callbacks=my_callbacks)

def gen_name(filters, kernal):
        name = 'ConvLSTM'
        for f in filters:
            name += '-{}'.format(f)
        for k in kernal:
            name += '-{}'.format(k)
        print('generated name {}'.format(name))
        return name

def gen_longname(filters, kernal, batch_size, seq_length):
        name = 'ConvLSTM'
        name += '-B{}'.format(batch_size)
        name += '-SL{}'.format(seq_length)
        for i in range(len(filters)):
            f = filters[i]
            k = kernal[i]
            name += '-{}'.format(f)
            name += '-{}x{}'.format(k, k)
        print('generated name {}'.format(name))
        return name



class ConvLSTM:
    """ A convoliutional lstm neural network.

    What about :
        recurrent_activation='hard_sigmoid'
        activation='tanh'

    Notes
    ----------------------------------------------------------------------------
    filters, kernel_size, strides=(1, 1), padding='valid', data_format=None,
    dilation_rate=(1, 1), activation='tanh', recurrent_activation='hard_sigmoid',
    use_bias=True, kernel_initializer='glorot_uniform',
    recurrent_initializer='orthogonal', bias_initializer='zeros',
    unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None,
    bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
    recurrent_constraint=None, bias_constraint=None, return_sequences=False,
    go_backwards=False, stateful=False, dropout=0.0, recurrent_dropout=0.0

    (x=x, y=y, batch_size=None, epochs=1, verbose=1, callbacks=None,
    validation_split=0.2, validation_data=None, shuffle=True, class_weight=None,
    sample_weight=None, initial_epoch=0, steps_per_epoch=None,
    validation_steps=None, validation_batch_size=None, validation_freq=1,
    max_queue_size=10,


    """

    DATA_FORMAT        = 'channels_last'
    PADDING            = 'same'
    RETURN_SEQUENCE    = True
    NUM_INPUT_VARS     = 4
    OUTPUT_KERNEL_SIZE = 1
    OUTPUT_FILTER      = 1
    KERNAL_INIT        = 'lecun_uniform'

    n_lat   = 81
    n_lon   = 161
    #WORKERS = 16 # identical to the number of cores requested in

    #USE_MULTIPROCESSING = True
    #early_stopping_monitor = EarlyStopping(patience=3)
    #CALLBACKS = [early_stopping_monitor, TensorBoard(log_dir='./logs')]

    def __init__(self, ms_batch_train_ds, ms_batch_train_ds_val, filters, kernels, seq_length = 24,
                 epochs=40, batch_size = 10, validation_split=None, name = None, result_path = None, test_data = None):

        self.filters = filters
        self.kernels = kernels
        self.seq_length = seq_length

        #if validation_split > 0.0 and batch_size > 1:
        #    raise ValueError('Validation split is incompatible with batchsize > 1')
        print(ms_batch_train_ds)
        self.ms_batch_train_ds = ms_batch_train_ds
        self.ms_batch_train_ds_val = ms_batch_train_ds_val
        self.test_data = test_data
        print(self.test_data)
        self.epochs = epochs
        self.batch_size = batch_size

        self.validation_split = validation_split
        print('Starts to build model ...')
        self.model = self.build_model(filters, kernels, seq_length)
        #self.model.build(ms_batch_train_ds[0].shape)
        #self.store_summary()
        print('Statrs compilation of model ...')
        self.name = name
        self.name = gen_longname(filters, kernels, batch_size, seq_length)
        result_path = '/home/hannasv/results_summary/'
        if result_path is not None:
            self.result_path = result_path
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            self.result_path = os.path.join(result_path, self.name)
            if not os.path.exists(os.path.join(result_path, self.name)):
                os.makedirs(os.path.join(result_path, self.name))
        else:
            self.result_path = '/home/hannasv/results_summary/'
       
        # self.result_path = '/home/hanna/lagrings/results/'
        self.model.compile(optimizer=keras.optimizers.Adam(
                            learning_rate=0.001,
                            beta_1=0.9,
                            beta_2=0.999,
                            epsilon=1e-07,
                            amsgrad=False,
                            name="Adam",),
                            loss='mean_squared_error',
                            metrics=['mean_squared_error', r2_keras, tf.keras.metrics.MeanAbsoluteError()])
        print('Starts training, files are stored in  {}'.format(self.result_path))
        #self.model.save(os.path.join(self.result_path,'{}_config.h5'.format(self.name)))

        #self.store_summary()
        self.history = self.model.fit(ms_batch_train_ds, #batch_size=batch_size,
                   epochs=self.epochs, verbose=1,
                   callbacks=my_callbacks,
                   #validation_split=0.2,
                   validation_data=ms_batch_train_ds_val,
                   shuffle=False,
                   #class_weight=None,
                   #sample_weight=None, initial_epoch=0,
                   #steps_per_epoch=100,
                   #validation_steps=None,
                   #validation_freq=1, max_queue_size=10,
                   #workers=self.WORKERS,
                   use_multiprocessing=False)


        self.store_history()
        self.store_summary()
        # serialize model to JSON
        model_json = self.model.to_json()
        # print(model_json)
        with open(self.result_path+"/model.json", "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(os.path.join(self.result_path,'model_weights.h5'))
        self.save_model_weights_config()
        # make prediction 
        #print("Evaluate on test data")
        #test_data = np.random.random((10, 24, 81, 161, 4)).astype('float32')
        #results = model.evaluate(x_test, y_test, batch_size=128)
        #print("test loss, test acc:", results)
        #print("Evaluate")
        result = self.model.evaluate(test_data)
        score = dict(zip(self.model.metrics_names, result))
        #print(score)
        with open(self.result_path+"/test_score.json", "w") as json_file:
            #json_file.write(score)
            import json
            json.dump(score, json_file)
        
        #test_np = np.stack(list(test_data))
        #print(type(test_np), test_np.shape)
        prediction = self.model.predict(test_data)
        print(prediction)
        print(prediction.shape)
        #print(prediction)
        #test_mse = mse(test_data[1], prediction).numpy()
        #print(test_mse)
        #test_mae = mae(test_data[1], prediction).numpy()
        #save_prediction(0.1, 0.1, prediction)
        data_dict = {'tcc': (['batch', 'sequence_length', 'latitude', 'longitude'], prediction[:, :, :, :, 0])}

        longitude  = np.arange(-15.0, 25.0+0.25, step = 0.25)
        latitude   =  np.arange(30.0, 50.0+0.25, step = 0.25)
        #seq_length = sequence_prediction
        #time = np.arange(seq_length)
        #print(sequence_prediction[:, :, :, :, 0].shape)
        ds = xr.Dataset(data_dict,
                     coords={'longitude': (['longitude'], longitude),
                             'latitude': (['latitude'], latitude),
                             'sequence_length': (['sequence_length'], np.arange(24)), 
                             'batch':(['batch'], np.arange(1820))
                            })
        #ds['mse_test']  = test_mse
        #ds['mae_test']  = test_mae
        ds['date_seq']  = '2014-01-01'
        print(ds)
        ds.to_netcdf(os.path.join(self.result_path,'prediction.nc'), engine = 'h5netcdf')
        

    def save_model_weights_config(self):
        self.model.save(os.path.join(self.result_path,'{}_config.h5'.format(self.name)))
        self.model.save_weights(os.path.join(self.result_path,'{}_weights.h5'.format(self.name)))
        print('finished model -- ')


    def build_model(self, filters, kernels, seq_length = 24):
        """ Building a ConvLSTM model for predicting cloud cover.
        All filters are squared. Adding the architecture.

        Parameteres
        ------------------------
        filters : array like
            use length of this to infer the depth of the network.

        Returns
        ------------------------
        model : tensorflow.keras.Sequential
            Builded model
        """

        model =  keras.Sequential()
        #input  = keras.layers.Input(batch_input_shape=(self.batch_size, seq_length, self.n_lat, self.n_lon,
        #                        self.NUM_INPUT_VARS), name='input')#batch_size = self.batch_size)
        model.add(tf.keras.layers.BatchNormalization(
                   axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
               beta_initializer='zeros', gamma_initializer='ones',
               moving_mean_initializer='zeros', moving_variance_initializer='ones',
               beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
               gamma_constraint=None, renorm=False, renorm_clipping=None, renorm_momentum=0.99,
               fused=None, trainable=True, virtual_batch_size=None, adjustment=None, name=None))

        # Adding the first layer
        model.add(keras.layers.ConvLSTM2D(filters = filters[0],
                           kernel_size = (kernels[0], kernels[0]), #, self.NUM_INPUT_VARS
                           #input_shape = (seq_length,
                           #                 self.n_lat, self.n_lon, self.NUM_INPUT_VARS),
                           kernel_initializer=self.KERNAL_INIT,
                           padding = self.PADDING,
                           return_sequences=self.RETURN_SEQUENCE,
                           data_format=self.DATA_FORMAT,
                           batch_size = self.batch_size, dtype = tf.float64))

        model.add(tf.keras.layers.BatchNormalization(
             axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
            beta_initializer='zeros', gamma_initializer='ones',
                moving_mean_initializer='zeros', moving_variance_initializer='ones',
             beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
             gamma_constraint=None, renorm=False, renorm_clipping=None, renorm_momentum=0.99,
              fused=None, trainable=True, virtual_batch_size=None, adjustment=None, name=None))


        prev_filter = filters[0]
        if len(filters) > 1 and len(kernels) > 1:
            print('Detected more than one layer ... ')
            for i, tuple in enumerate(zip(filters[1:], kernels[1:])):
                filter, kernal = tuple
                # Begin with 3D convolutional LSTM layer
                model.add(keras.layers.ConvLSTM2D(filters=filter,
                                                kernel_size=(kernal, kernal), # prev_filter
                                                #input_shape = (seq_length, self.n_lat,
                                                #                self.n_lon, prev_filter),
                                                kernel_initializer=self.KERNAL_INIT,
                                                padding = self.PADDING,
                                                return_sequences=self.RETURN_SEQUENCE,
                                                data_format=self.DATA_FORMAT,
                                                batch_size = self.batch_size, dtype = tf.float64, 
						))
                prev_filter = filter
                model.add(tf.keras.layers.BatchNormalization(
                              axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                beta_initializer='zeros', gamma_initializer='ones',
                               moving_mean_initializer='zeros', moving_variance_initializer='ones',
                             beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                                  gamma_constraint=None, renorm=False, renorm_clipping=None, renorm_momentum=0.99,
                                 fused=None, trainable=True, virtual_batch_size=None, adjustment=None, name=None))



        # Adding the last layer
        model.add(keras.layers.ConvLSTM2D(filters=self.OUTPUT_FILTER,
                                        kernel_size=(self.OUTPUT_KERNEL_SIZE, self.OUTPUT_KERNEL_SIZE), #prev_filter
                                        #input_shape = (seq_length, self.n_lat,
                                        #                self.n_lon, prev_filter),
                                        kernel_initializer=self.KERNAL_INIT,
                                        padding = self.PADDING,
                                        return_sequences=self.RETURN_SEQUENCE,
                                        data_format=self.DATA_FORMAT,
                                        batch_size = self.batch_size, dtype = tf.float64))

        return model

    def compile(self, lmd=0.001):
        """ Compile model.

        Parameters
        -------------
        model : tensorflow.keras.Sequential
            Build model.

        Returnes
        -------------
        model : tensorflow.keras.Sequential
            Compiled model.
        """
        self.model.compile(optimizer=keras.optimizers.Adam(
                            learning_rate=lmd,
                            beta_1=0.9,
                            beta_2=0.999,
                            epsilon=1e-07,
                            amsgrad=False,
                            name="Adam",),
                            loss='mean_squared_error',
                            metrics=['mean_squared_error', r2_keras, tf.keras.metrics.MeanAbsoluteError()])
        return self.model



    def store_history(self):
        """ Fit builded model.
        Parameters
        -------------
        model : tensorflow.keras.Sequential
            Builded model
        """
        import pandas as pd
        history = self.history

        # convert the history.history dict to a pandas DataFrame:
        hist_df = pd.DataFrame(history.history)

        # save to json:
        hist_json_file = os.path.join(self.result_path, 'history.json')
        with open(hist_json_file, mode='w') as f:
            hist_df.to_json(f)

        # or save to csv:
        hist_csv_file = os.path.join(self.result_path, 'history.csv')
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)

        return

    def store_summary(self):
        """ Store summary of tranings process.
        """
        ORIG_OUTPUT = sys.stdout
        with open(os.path.join(self.result_path, "summary_{}.txt".format(self.name)), "w") as text_file:
            sys.stdout = text_file
            self.model.summary()
        sys.stdout = ORIG_OUTPUT
        #self.model.save(os.path.join(self.result_path,'{}_config.h5'.format(self.name)))
        #self.model.save_weights(os.path.join(self.result_path,'{}_weights.h5'.format(self.name)))
        return

if __name__ == '__main__':
    import tensorflow as tf
    num_vars = 4
    # (seq_length, self.n_lat, self.n_lon, self.NUM_INPUT_VARS),
    seq_length = 24

    epochs = 5
    batch_size = 10
    #Xtrain_dummy = tf.ones((batch_size, seq_length, 81, 161, num_vars))
    #ytrain_dummy = tf.ones((batch_size, seq_length, 81, 161))

    # antall filrer i hver lag.
    filters = [32] #256, 128,
    # size of filters used 
    kernels = [3] #, 3, 3
    """
    Batch size and the validation split is incompatible.

    tf.keras.Input(
        shape=None,
        batch_size=None,
        name=None,
        dtype=None,
        sparse=False,
        tensor=None,
        ragged=False,
        **kwargs
    )"""
    def gen_name(filters, kernal):
        name = 'ConvLSTM'
        for f in filter:
            name += '-{}'.format(f)
        for k in kernals:
            name += '-{}'.format(k)
        return name

    from utils import get_xarray_dataset_for_period, get_data_keras
    data = get_xarray_dataset_for_period(start = '2012-01-01', stop = '2012-01-31')
    X_train, y_train = get_data_keras(data, num_samples = None, seq_length = 24, batch_size = 10,
                                      data_format='channels_last')
    print(X_train.shape)
    print(y_train.shape)
    #model = ConvLSTM(X_train=X_train, y_train=y_train, filters=filters,
    #                 kernels=kernels, seq_length = seq_length,
    #                 epochs=epochs, batch_size = batch_size, validation_split=None,
    #                 name = 'test_model', result_path = '/home/hannasv/results/')
