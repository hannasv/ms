#from base_train_from_files import train_ar_model
from base_train_from_files import train_ar_model
import glob

print('num available files {}'.format(len(glob.glob('/uio/lagringshotell/geofag/students/metos/hannasv/ar_data/*.nc'))))

# pasted in m1.py
try:
    train_ar_model(transform=False, bias=True, sig=False, order=5, overwrite_results = False) # AR-L0
except Exception as e:
    #print('ende')
    print('Detected error: {} Model only bias true'.format(e))


#try:
#    train_ar_model(transform=True, bias=False, sig=False, order=5, overwrite_results = False) # AR-L0
#except Exception as e:
#    #print('ende')
#    print('Error {} Model only transform True')

#try:
#    train_ar_model(transform=False, bias=False, sig=True, order=5, overwrite_results = False) # AR-L0
#except Exception as e:
#    #print('ende')
#    print('Error Sigmoid true')

#try:
#    train_ar_model(transform=False, bias=True, sig=True, order=5, overwrite_results = False) # AR-L0
#except Exception as e:
#    #print('ende')
#    print('hehe bias and sigmoid true')



#try:
#    train_ar_model(transform=True, bias=False, sig=True, order=5, overwrite_results = False) # AR-L0
#except Exception as e:
#    #print('ende')
#    print('hihi Transform and Sigmoid True ')

#try:
#    train_ar_model(transform=False, bias=False, sig=False, order=5, overwrite_results = True) # AR-L0
#except Exception as e:
#    print('All False ')
#    print(e)

