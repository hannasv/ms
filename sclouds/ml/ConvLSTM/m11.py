from base_config import train_convlstm


# NON converging models
train_convlstm(num_vars = 4, seq_length = 24, epochs = 40, batch_size = 10,
               filters = [32, 32], kernels = [5, 5],
               drop_remainder_batch = True, overwrite_results = True)

#train_convlstm(num_vars = 4, seq_length = 6, epochs = 40, batch_size = 5,
#               filters = [64, 64], kernels = [3, 3],
#               drop_remainder_batch = True, overwrite_results = True)

#train_convlstm(num_vars = 4, seq_length = 6, epochs = 40, batch_size = 5,
#               filters = [32, 32, 32], kernels = [3, 3, 3],
#               drop_remainder_batch = True, overwrite_results = True)

#train_convlstm(num_vars = 4, seq_length = 6, epochs = 40, batch_size = 5,
#               filters = [64, 32], kernels = [5, 5],
#               drop_remainder_batch = True, overwrite_results = True)

# converged models.
#train_convlstm(num_vars = 4, seq_length = 24, epochs = 40, batch_size = 10,
#               filters = [128], kernels = [3],
#               drop_remainder_batch = True, overwrite_results = True)

#train_convlstm(num_vars = 4, seq_length = 24, epochs = 40, batch_size = 10,
#               filters = [128, 32], kernels = [3, 3],
#               drop_remainder_batch = True, overwrite_results = True)

#train_convlstm(num_vars = 4, seq_length = 24, epochs = 40, batch_size = 10,
#               filters = [16, 16], kernels = [3, 3],
#               drop_remainder_batch = True, overwrite_results = True)

#train_convlstm(num_vars = 4, seq_length = 24, epochs = 40, batch_size = 10,
#               filters = [8, 8, 8], kernels = [3, 3, 3],
#               drop_remainder_batch = True, overwrite_results = True)



