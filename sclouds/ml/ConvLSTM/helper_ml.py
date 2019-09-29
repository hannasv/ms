import numpy as np

def receptive_field(f, s):
    # TODO dilation??

    # Implement equation(1)

    # Inputs:
    # f (list): Filter size for each layer
    # s (list): Stride for each layer

    # Output
    # R: The calculated receptive field for each layer as a numpy array

    # ToDo:
    R = [1]
    for kk in range(len(s)):
        S = 1
        for ii in range(kk):
            S = S * s[ii]
        fov = R[-1] + (f[kk] - 1) * S
        R.append(fov)
    return np.array(R)
