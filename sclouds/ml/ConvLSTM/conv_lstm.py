"""Convolutional Long-Short Term Model.
"""
from sclouds.helpers import get_lon_array, get_lat_array, path_convlstm_results

# The impact of using return_sequences is that the model will classify each frame in one category.
DATA_FORMAT        = 'channels_last'
PADDING            = 'same'
RETURN_SEQUENCE    = True
NUM_INPUT_VARS     = 4
OUTPUT_KERNEL_SIZE = 1
OUTPUT_FILTER      = 1

n_lon = len(get_lon_array())
n_lat = len(get_lat_array())

# -- Preparatory code -- #############
# Model configuration    #############
batch_size = 100
no_epochs = 30
learning_rate = 0.001
validation_split = 0.2
verbosity = 1
########################################

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

    """

    def __init__(self):
        return

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
        # TODO : update with activations?


        model =  Sequential()
        # Adding the first layer
        seq.add(ConvLSTM3D(filters = filters[0],
                           kernel_size = (kernels[0], kernels[0], NUM_INPUT_VARS),
                           input_shape = (seq_length, n_lat, n_lon, NUM_INPUT_VARS),
                           kernel_initializer='glorot_uniform',
                           padding = PADDING,
                           return_sequences=RETURN_SEQUENCE,
                           data_format=DATA_FORMAT))

        prev_filter = filters[0]
        for i, filter, kernal in enumerate(zip(filters[1:], kernels[1:])):
            # Begin with 3D convolutional LSTM layer
            seq.add(keras.layers.ConvLSTM3D(filters=filter,
                                            kernel_size=(kernal, kernel, prev_filter),
                                            kernel_initializer='glorot_uniform',
                                            padding = PADDING,
                                            return_sequences=RETURN_SEQUENCE,
                                            data_format=DATA_FORMAT))
            prev_filter = filter
        # Adding the last layer
        seq.add(keras.layers.ConvLSTM3D(filters=OUTPUT_FILTER,
                                        kernel_size=(1, 1, prev_filter),
                                        kernel_initializer='glorot_uniform',
                                        padding = PADDING,
                                        return_sequences=RETURN_SEQUENCE,
                                        data_format=DATA_FORMAT))

        return model

    def compile(self, model):
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
        model.compile(optimizer=keras.optimizers.Adam(
                            hp.Choice('learning_rate',
                            values=[1e-2, 1e-3, 1e-4])),
        loss='mean_squared_error',
        metrics=['accuracy'])
        return model


    def fit(self, model):
        """ Fit builded model.
        Parameters
        -------------
        model : tensorflow.keras.Sequential
            Builded model
        """
        return model

if __name__ == '__main__':
    print('Nothing here yet.')
