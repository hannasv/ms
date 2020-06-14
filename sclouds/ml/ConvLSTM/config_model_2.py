    import tensorflow as tf
    num_vars = 4
    # (seq_length, self.n_lat, self.n_lon, self.NUM_INPUT_VARS),
    seq_length = 24

    epochs = 100
    batch_size = 20
    #Xtrain_dummy = tf.ones((batch_size, seq_length, 81, 161, num_vars))
    #ytrain_dummy = tf.ones((batch_size, seq_length, 81, 161))

    # antall filrer i hver lag.
    filters = [256, 256]
    # size of filters used
    kernels = [3, 3]

    from utils import get_xarray_dataset_for_period, get_data_keras, get_train_test
    #data = get_xarray_dataset_for_period(start = '2012-01-01', stop = '2012-01-31')
    #print(data)
    test_start = '2014-01-01'
    test_stop  = '2018-12-31'
    train_dataset, test_dataset =  get_train_test(test_start, test_stop, model = 'ar')
    X_train, y_train = get_data_keras(train_dataset, num_samples = None, seq_length = 24,
                                        batch_size = None, data_format='channels_last')

    model = ConvLSTM(X_train=X_train, y_train=y_train, filters=filters,
                     kernels=kernels, seq_length = seq_length,
                     epochs=epochs, batch_size = batch_size, validation_split=0.1,
                     name = 'Model2', result_path = '/home/hannasv/results/')
