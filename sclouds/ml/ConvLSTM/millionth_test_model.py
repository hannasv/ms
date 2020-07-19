import tensorflow as tf
import numpy as np

from utils import get_xarray_dataset_for_period, get_data_keras
data = get_xarray_dataset_for_period(start = '2012-01-01', stop = '2012-01-31')
X_train, y_train = get_data_keras(data, num_samples = None, seq_length = 24, batch_size = 10,
                data_format='channels_last', )

data = get_xarray_dataset_for_period(start = '2012-02-01', stop = '2012-02-28')
X_test, y_test = get_data_keras(data, num_samples = None, seq_length = 24, batch_size = 10,
                data_format='channels_last', )
batch_size = 10
ms_train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
#ms_train_ds
ms_batch_train_ds = ms_train_ds.batch(batch_size, drop_remainder=True)
#ms_batch_train_ds

ms_train_ds_val = tf.data.Dataset.from_tensor_slices((X_test, y_test))
#ms_train_ds
ms_batch_train_ds_val = ms_train_ds_val.batch(batch_size, drop_remainder=True)
#ms_batch_train_ds_val

DATA_FORMAT        = 'channels_last'
PADDING            = 'same'
RETURN_SEQUENCE    = True
NUM_INPUT_VARS     = 4
OUTPUT_KERNEL_SIZE = 1
OUTPUT_FILTER      = 1
KERNAL_INIT        = 'lecun_uniform'


filters = [32]
kernels = [1]
model =  tf.keras.Sequential()

#model.add( keras.layers.Input(batch_input_shape=(self.batch_size, seq_length, self.n_lat, self.n_lon,
#                        self.NUM_INPUT_VARS), name='input'))        #batch_size = self.batch_size)

# Adding the first layer
model.add(tf.keras.layers.ConvLSTM2D(filters = filters[0],
                   kernel_size = (kernels[0], kernels[0]), #, self.NUM_INPUT_VARS
                   #input_shape = (seq_length,
                   #                 self.n_lat, self.n_lon, self.NUM_INPUT_VARS),
                   kernel_initializer=KERNAL_INIT,
                   padding = PADDING,
                   return_sequences=RETURN_SEQUENCE,
                   #data_format=self.DATA_FORMAT,
                   batch_size = batch_size))

prev_filter = filters[0]
if len(filters) > 1 and len(kernels) > 1:
    print('Detected more than one layer ... ')
    for i, tuple in enumerate(zip(filters[1:], kernels[1:])):
        filter, kernal = tuple
        # Begin with 3D convolutional LSTM layer
        model.add(tf.keras.layers.ConvLSTM2D(filters=filter,
                                        kernel_size=(kernal, kernal), # prev_filter
                                        #input_shape = (seq_length, self.n_lat,
                                        #                self.n_lon, prev_filter),
                                        kernel_initializer=KERNAL_INIT,
                                        padding = PADDING,
                                        return_sequences=RETURN_SEQUENCE,
                                        data_format=DATA_FORMAT,
                                        batch_size = batch_size))
        prev_filter = filter
# Adding the last layer
model.add(tf.keras.layers.ConvLSTM2D(filters=OUTPUT_FILTER,
                                kernel_size=(OUTPUT_KERNEL_SIZE,OUTPUT_KERNEL_SIZE), #prev_filter
                                #input_shape = (seq_length, self.n_lat,
                                #                self.n_lon, prev_filter),
                                kernel_initializer=KERNAL_INIT,
                                padding = PADDING,
                                return_sequences=RETURN_SEQUENCE,
                                data_format=DATA_FORMAT,
                                batch_size = batch_size))

my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3, monitor = 'val_loss',), # loss
    tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    tf.keras.callbacks.TerminateOnNaN()

]

model.compile(optimizer=tf.keras.optimizers.Adam(
                            learning_rate=0.001,
                            beta_1=0.9,
                            beta_2=0.999,
                            epsilon=1e-07,
                            amsgrad=False,
                            name="Adam",),
                            loss='mean_squared_error',
                            metrics=['mean_squared_error'])

model.fit(ms_batch_train_ds, #batch_size=batch_size,
         epochs=10, verbose=1,
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

