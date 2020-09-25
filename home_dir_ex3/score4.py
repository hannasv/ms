import os
import json
import glob
import numpy as np
import xarray as xr

#path = '/home/hanna/lagrings/results/ar/models/' # TODO update when data is in this folderself.
#path = '/home/hanna/lagrings/results/ar/'

path = '/home/hannasv/results_ar/'
#example = glob.glob(path+'/performance*-5-*')
#model_base_path = glob.glob(path+'*')
#print(model_base_path)
models = ['AR-B-S-5']
#['AR-B-5','AR-T-5','AR-S-5','AR-B-S-5', 'AR-T-S-5', 'AR-5']
#unique_models = np.unique([f.split('_')[1] for f in example]) # unique combinations
for metric in ['mae']: #, 'mse']:
    print('started on computing global {}'.format(metric))
    #test_scores  = []
    #model_names  = []
    #train_scores = []
    #num_trained_models = []
    json_dict = {}
    train_key = '{}_train'.format(metric)
    test_key = '{}_test'.format(metric)
    for model in models:
        path_model = os.path.join(path, model)
        for type in ['AR', 'TR']:
            test_scores  = []
            train_scores  = []
            error_message  = []
            num_trained_models = []

            for i in range(6):
                if not (type == 'TR' and i==0):
                    print(type)
                    relevant_files = glob.glob(os.path.join(path_model,'*performance*{}*L{}*'.format(type, i)))
                    name= relevant_files[0].split('_')[-3]
                    print(name)
                    print('Detected {} relevant files'.format(len(relevant_files)))
                    try:
                        data = xr.open_mfdataset(relevant_files, combine = 'by_coords')
                        test  = data[test_key].mean().values
                        train = data[train_key].mean().values

                        test_scores.append(float(test))
                        train_scores.append(float(train))
                        error_message.append('OK')
                        num_trained_models.append(len(relevant_files))
                        print('Model: {}, train score: {:.4f} and test score {:.4f}. '.format(model, train, test))
                    except Exception as e:
                        print('problem  files for model {}, error: {}'.format(model, e))
                        #print(relevant_files)
                        train_scores.append(np.nan)
                        test_scores.append(np.nan)
                        num_trained_models.append(len(relevant_files))
                        error_message.append(e)
                else:
                    train_scores.append(np.nan)
                    test_scores.append(np.nan)
                    num_trained_models.append(0)
                    error_message.append('OK')
                    #for i, name in enumerate(model_names):
            json_dict[name] = {}
            json_dict[name][train_key] = train_scores
            json_dict[name][test_key]  = test_scores
            json_dict[name]['num_trained_models'] = num_trained_models
            #json_dict[name]['error_message'] = error_message

        fil = '/home/hannasv/ar_summary/ar_{}_performance_{}.json'.format(metric, model) #, type, i)
        with open(fil, 'w') as json_file:
            json.dump(json_dict, json_file)
        print('stored file:{} containing {}'.format(fil, len(test_scores)))

    fil = '/home/hannasv/ar_summary/ar_{}_performance_terminal.json'.format(metric) #, type, i)
    with open(fil, 'w') as json_file:
        json.dump(json_dict, json_file)
    print('stored file:{} containing {}'.format(fil, len(test_scores)))

