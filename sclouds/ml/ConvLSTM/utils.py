# Custom R2-score metrics for keras backend
def r2_keras(y_true, y_pred):
    import keas.backend as kb
    SS_res =  kb.sum(kb.square(y_true - y_pred))
    SS_tot = kb.sum(kb.square(y_true - kb.mean(y_true)))
    return ( 1 - SS_res/(SS_tot + kb.epsilon()) )

def keras_custom_loss_function(y_actual, y_predict):
    """Custum keras loss function, accumulated squared error. """
    import keas.backend as kb
    return np.square(np.subtract(y_actual, y_predict)).sum(axis = 0)
