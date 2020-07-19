from base_ar import train_ar_model
# (transform, bias, sig, order, overwrite_results = True)

try:
    train_ar_model(transform=False, bias=True, sig=False, order=5, overwrite_results = True) # AR-L0
except OSError:
    #print('ende')
    print('Finished training AR-L5')

#train_ar_model(transform=True, bias=False, sig=False, order=5, overwrite_results = True) # AR-T-L0
#print('Finished training AR-T-L5')
#train_ar_model(transform=False, bias=True, sig=False, order=5, overwrite_results = True)
#print('Finished training AR-B-L5')
#train_ar_model(transform=False, bias=False, sig=True, order=5, overwrite_results = True)
#print('Finished training AR-S-L5')
#train_ar_model(transform=True, bias=True, sig=True, order=5, overwrite_results = True)
#print('Finished training AR-TSB-L5')
#train_ar_model(transform=True, bias=True, sig=False, order=5, overwrite_results = True)
#print('Finished training AR-TB-L5')
#train_ar_model(transform=False, bias=True, sig=True, order=5, overwrite_results = True)
#print('Finished training AR-BS-L5')

