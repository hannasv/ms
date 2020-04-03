""" Description for Hyper Parameters Convolutional Long-Short Term model.
"""

#import tensorflow as tf
#from tensorflow import keras
from tensorflow.keras import losses, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D #, BatchNormalization


from kerastuner import HyperModel
from sclouds.helpers import get_lon_array, get_lat_array, path_convlstm_results

n_lon = len(get_lon_array())
n_lat = len(get_lat_array())

class HyperConvLSTM(HyperModel):
    """ HyperConvLSTM is a child class of kerastuner.HyperModel

    Attributes
    -------------------
    DATA_FORMAT        = 'channels_last'
    PADDING            = 'same'
    RETURN_SEQUENCE    = True
    NUM_INPUT_VARS     = 4
    OUTPUT_KERNEL_SIZE = 1
    OUTPUT_FILTER      = 1
    KERNAL_INIT        = 'glorot_uniform'

    # TODO : add the following
    # DROPOUT, RECURRENT DROPOUT
    # ACTIVATION, RECURRENT ACTIVATION
    """

    DATA_FORMAT        = 'channels_last'
    PADDING            = 'same'
    RETURN_SEQUENCE    = True
    NUM_INPUT_VARS     = 4
    OUTPUT_KERNEL_SIZE = 1
    OUTPUT_FILTER      = 1
    KERNAL_INIT        = 'lecun_uniform',
    # TODO : add the following
    # DROPOUT, RECURRENT DROPOUT
    # ACTIVATION, RECURRENT ACTIVATION

    def __init__(self, num_hidden_layers, seq_length):
        self.num_hidden_layers = num_hidden_layers
        self.seq_length = seq_length

    def build(self, hp):
        """ Building and compiling a hyper convolutional lstm model.

        Parameters
        --------------------
        hp : kerastuner.HyperParameters
            Container for both a hyperparameter space, and current values.

        Returns
        -----------------


        """
        model =  Sequential()
        # Adding the first layer
        model.add(ConvLSTM2D(filters= hp.Int('filters',
                                        min_value=32,
                                        max_value=512,
                                        step=32),
                           kernel_size = (3, 3),
                           input_shape = (self.seq_length, n_lat, n_lon, self.NUM_INPUT_VARS),
                           #kernel_initializer=self.KERNAL_INIT,
                           padding = self.PADDING,
                           return_sequences=self.RETURN_SEQUENCE,
                           data_format=self.DATA_FORMAT))

        #prev_filter = filters[0]
        for i in range(self.num_hidden_layers):
            # Begin with 3D convolutional LSTM layer
            model.add(ConvLSTM2D(filters= hp.Int('filters',
                                                        min_value=32,
                                                        max_value=512,
                                                        step=32),
                                            kernel_size=(3, 3),
                                            #kernel_initializer=self.KERNAL_INIT,
                                            padding = self.PADDING,
                                            return_sequences = self.RETURN_SEQUENCE,
                                            data_format = self.DATA_FORMAT))
        # Adding the last layer
        model.add(ConvLSTM2D(filters = self.OUTPUT_FILTER,
                                          kernel_size = (1, 1),
                                          #kernel_initializer = self.KERNAL_INIT,
                                          padding = self.PADDING,
                                          return_sequences = self.RETURN_SEQUENCE,
                                          data_format = self.DATA_FORMAT))



        model.compile(
            optimizer=optimizers.Adam(
                hp.Choice('learning_rate',
                          values=[1e-2, 1e-3, 1e-4])),
            loss=hp.Choice('loss',
                            values=['mean_squared_error', 'mean_absolute_error'],
                            default='mean_squared_error'),
            metrics=['mean_squared_error', 'mean_absolute_error'])

        return model

#if __name__ != '__main__':

    #from kerastuner.tuners import RandomSearch
    #from kerastuner import HyperParameters

    #hp = HyperParameters()
    #hypermodel = HyperConvLSTM(num_hidden_layers = 2, seq_length= 4)

    #tuner = RandomSearch(
            # hypermodel,
            # objective='mean_squared_error',
            # max_trials=10,
            # allow_new_entries = True,
            # directory=path_convlstm_results,
            # project_name='test_hyperparameters')

    # Read in x and y
    #tuner.search(X_train, y_train,
    #         epochs=5,
    #         validation_data=(X_test, y_test))

    #tuner.search_space_summary()
