"""Convolutional Long-Short Term Model.
"""
import os, sys
import glob
import numpy as np

from sclouds.helpers import get_lon_array, get_lat_array, path_convlstm_results
from sclouds.ml.ConvLSTM.utils import r2_keras

from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

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
    WORKERS = 16 # identical to the number of cores requested in

    USE_MULTIPROCESSING = True
    early_stopping_monitor = EarlyStopping(patience=3)
    CALLBACKS = [early_stopping_monitor, TensorBoard(log_dir='./logs')]

    def __init__(self, X_train, y_train, filters, kernels, seq_length = 24,
                 epochs=40, batch_size = 20, validation_split=0.1, name = None, result_path = None):

        self.filters = filters
        self.kernels = kernels
        self.seq_length = seq_length

        self.X_train = X_train
        self.y_train = y_train
        self.epochs = epochs
        self.batch_size = batch_size

        self.validation_split = validation_split
        print('Starts to build model ...')
        self.model = self.build_model(filters, kernels, seq_length)
        print('Statrs compilation of model ...')
        self.name = name
        """
        if result_path is not None:
            self.result_path = result_path
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            self.result_path = os.path.join(result_path, name)
            if not os.path.exists(os.path.join(result_path, name)):
                os.makedirs(os.path.join(result_path, name))
        else:
            self.result_path = '/home/hannasv/results/'
        """
        self.result_path = '/home/hanna/lagrings/results/'
        self.model.compile(optimizer=keras.optimizers.Adam(
                            learning_rate=0.001,
                            beta_1=0.9,
                            beta_2=0.999,
                            epsilon=1e-07,
                            amsgrad=False,
                            name="Adam",),
                            loss='mean_squared_error',
                            metrics=['mean_squared_error', r2_keras])
        print('starts training')
        self.history = self.model.fit(X_train, y_train,
                                     batch_size=batch_size,
                                     epochs=epochs, verbose=1,
                                     callbacks=self.CALLBACKS,
                                     validation_split=self.validation_split,
                                     #validation_data=None,
                                     shuffle=False,
                                     #class_weight=None,
                                     #sample_weight=None, initial_epoch=0,
                                     #steps_per_epoch=100,
                                     #validation_steps=None,
                                     #validation_freq=1, max_queue_size=10,
                                     workers=self.WORKERS,
                                     use_multiprocessing= self.USE_MULTIPROCESSING)
        self.store_history()
        self.store_summary()
        print('finished model -- ')


    def build_model(self, filters, kernels, seq_length = 24):
        """" Building a ConvLSTM model for predicting cloud cover.
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

        input  = keras.layers.Input(shape=(seq_length, self.n_lat, self.n_lon,
                                self.NUM_INPUT_VARS), name='input')#batch_size = self.batch_size)
                                

        # Adding the first layer
        model.add(keras.layers.ConvLSTM2D(filters = filters[0],
                           kernel_size = (kernels[0], kernels[0]), #, self.NUM_INPUT_VARS
                           input_shape = (seq_length,
                                            self.n_lat, self.n_lon, self.NUM_INPUT_VARS),
                           kernel_initializer=self.KERNAL_INIT,
                           padding = self.PADDING,
                           return_sequences=self.RETURN_SEQUENCE,
                           data_format=self.DATA_FORMAT,))
                           #batch_size = self.batch_size

        prev_filter = filters[0]
        if len(filters) > 1 and len(kernels) > 1:
            print('Detected more than one layer ... ')
            for i, tuple in enumerate(zip(filters[1:], kernels[1:])):
                filter, kernal = tuple
                # Begin with 3D convolutional LSTM layer
                model.add(keras.layers.ConvLSTM2D(filters=filter,
                                                kernel_size=(kernal, kernal), # prev_filter
                                                input_shape = (self.batch_size,
                                                        seq_length, self.n_lat,
                                                        self.n_lon, prev_filter),
                                                kernel_initializer=self.KERNAL_INIT,
                                                padding = self.PADDING,
                                                return_sequences=self.RETURN_SEQUENCE,
                                                data_format=self.DATA_FORMAT,))
                                                #batch_size = self.batch_size
                prev_filter = filter
        # Adding the last layer
        model.add(keras.layers.ConvLSTM2D(filters=self.OUTPUT_FILTER,
                                        kernel_size=(self.OUTPUT_KERNEL_SIZE, self.OUTPUT_KERNEL_SIZE), #prev_filter
                                        input_shape = (self.batch_size, seq_length, self.n_lat,
                                                        self.n_lon, prev_filter),
                                        kernel_initializer=self.KERNAL_INIT,
                                        padding = self.PADDING,
                                        return_sequences=self.RETURN_SEQUENCE,
                                        data_format=self.DATA_FORMAT,))
                                        #batch_size = self.batch_size))

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
                            metrics=['mean_squared_error', r2_keras])
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
        self.model.save(os.path.join(self.result_path,'{}.h5'.format(self.name)))  # creates a HDF5 file 'my_model.h5'
        return

    def for_later(self):
        from keras.models import load_model

        model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'

        # returns a compiled model
        # identical to the previous one
        model = load_model('my_model.h5')

        save_weights(
            filepath, overwrite=True, save_format=None
        )

        test_on_batch(
            x, y=None, sample_weight=None, reset_metrics=True, return_dict=False
        )
        return


if __name__ == '__main__':
    import tensorflow as tf
    num_vars = 4
    seq_length = 24
    epochs = 40
    batch_size = 20
    #Xtrain_dummy = tf.ones((batch_size, seq_length, 81, 161, num_vars))
    #ytrain_dummy = tf.ones((batch_size, seq_length, 81, 161)
    # antall filrer i hver lag.
    filters = [32] #256, 128,
    # size of filters used 
    kernels = [3] #, 3, 3
    """
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
    from utils import get_xarray_dataset_for_period, get_data_keras, get_train_test
    #data = get_xarray_dataset_for_period(start = '2012-01-01', stop = '2012-01-31')
    #print(data)
    test_start = '2014-01-01'
    test_stop  = '2018-12-31'
    train_dataset, test_dataset =  get_train_test(test_start, test_stop, model = 'ar')
    X_train, y_train = get_data_keras(train_dataset, num_samples = None, seq_length = 24, batch_size = None,
                    data_format='channels_last')
    print(X_train.shape)
    print(y_train.shape)
    model = ConvLSTM(X_train=X_train, y_train=y_train, filters=filters,
                     kernels=kernels, seq_length = seq_length,
                     epochs=epochs, batch_size = batch_size, validation_split=0.1,
                     name = 'test_model', result_path = '/home/hannasv/results/')
