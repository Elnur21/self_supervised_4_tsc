import sys
import os
import numpy as np
import pandas as pd
from utils.utils import load_data, znormalisation, create_directory, encode_labels, split_ypred,draw,draw_before
from model import MODEL
from apply_classifier import apply_classifier

def extract_args():

    if len(sys.argv) != 5:
        raise ValueError("No options were specified")
    
    else:

        return sys.argv[2], sys.argv[4]
    
    # example : python3 main.py -e inception -d Coffee


UNIVARIATE_DATASET_NAMES_2018 = ['ArrowHead', 'BeetleFly', 'Ham', 'MoteStrain', 'OliveOil', 'Wine', 'Lightning7', 'InlineSkate', 'Beef', 'ACSF1', 'Yoga', 'GunPointOldVersusYoung',
                 'FreezerSmallTrain', 'WordSynonyms', 'Car', 'ProximalPhalanxTW', 'InsectWingbeatSound','FaceAll', 'EOGVerticalSignal',  'Earthquakes']



if __name__ == "__main__":


    runs = 5

    n_dim = 32

    encoder_name, _ = extract_args()
    for file_name in UNIVARIATE_DATASET_NAMES_2018:
        output_directory_parent = 'results_lite/'
        # if os.path.exists(output_directory_parent+"fcn/run_0/"+file_name):
        #     print('Already done')
        #     continue

        create_directory(output_directory_parent)

        output_directory_parent = output_directory_parent + encoder_name + '/'
        create_directory(output_directory_parent)

        xtrain, ytrain, xtest, ytest = load_data(file_name=file_name)
        
        xtrain = znormalisation(xtrain)
        xtest = znormalisation(xtest)

        ytrain = encode_labels(ytrain)
        ytest = encode_labels(ytest)

        l = int(xtrain.shape[1])
        

        for _run in range(1):
        
            output_directory = output_directory_parent + 'run_'+str(_run) + '/'
            create_directory(output_directory)
            output_directory = output_directory + file_name + '/'
            create_directory(output_directory)

            model = MODEL(length_TS=l,n_dim=n_dim,encoder_name=encoder_name,output_directory=output_directory)

            model.fit(xtrain=xtrain,xval=xtest)

            ypred_train = model.predict(xtrain)
            ypred_test = model.predict(xtest)

            # draw(ypred_test=ypred_test,labels_test=ytest,output_directory=output_directory)
            # draw_before(xtest=xtest,ytest=ytest,output_directory=output_directory)

            new_xtrain, new_xtest = split_ypred(ypred_train=ypred_train,ypred_test=ypred_test)

            new_xtrain = np.asarray(new_xtrain)
            new_xtest = np.asarray(new_xtest)

            np.save(arr=new_xtrain,file=output_directory+'v_train.npy')
            np.save(arr=new_xtest,file=output_directory+'v_test.npy')
