import json
import glob
import numpy as np
import xarray as xr

path = '/home/hanna/lagrings/results/ar/models/' # TODO update when data is in this folderself.
path = '/home/hanna/lagrings/results/ar/'

path = '/home/hannasv/results_ar/'
#example = glob.glob(path+'/performance*-5-*')
model_base_path = glob.glob(path+'/*'):
print(model_base_path)
#unique_models = np.unique([f.split('_')[1] for f in example]) # unique combinations
for metric in ['mae', 'mse']:
    print('started on computing global {}'.format(metric))
    test_scores  = []
    model_names  = []
    train_scores = []
    num_trained_models = []

    json_dict = {}
    train_key = '{}_train'.format(metric)
    test_key = '{}_test'.format(metric)
    for model in model_base_path:
        for type in ['AR', 'TR']:
            model_names.append(model)
            for i in range(6):
                if not type == 'TR' and i==0:
                    relevant_files = glob.glob(model+'performance*{}*L{}*'.format(type, i))
                    try:
                        data = xr.open_mfdataset(relevant_files, combine = 'by_coords')
                        test  = data[test_key].sum().values
                        train = data[train_key].sum().values
                        test_scores.append(float(test))
                        train_scores.append(float(train))

                        num_trained_models.append(len(relevant_files))
                        print('Model: {}, train score: {:.4f} and test score {:.4f}. '.format(model, train, test))
                    except Exception as e:
                        print('problem  files for model {}, error: {}'.format(model, e))
                        print(relevant_files)
                        test_scores.append(np.nan)
                        train_scores.append(np.nan)

                for i, name in enumerate(model_names):
                    json_dict[name] = {}
                    json_dict[name][train_key] = float(train_scores[i])
                    json_dict[name][test_key] = float(test_scores[i])

                fil = '/home/hannasv/ar_{}_performance.json'.format(metric, model)
                with open(fil, 'w') as json_file:
                    json.dump(json_dict, json_file)
                print('stored file:{} containing {}'.format(fil, len(test_scores)))
