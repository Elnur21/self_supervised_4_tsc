

import pandas as pd
from tensorflow import keras 

from utils.helper import *
from utils.utils import *

from triplet_loss import triplet_loss_function


custom_objects = {'temp': triplet_loss_function}



# with tf.device('/cpu:0'):

for dataset_name in ['ArrowHead', 'BeetleFly', 'Ham', 'MoteStrain', 'OliveOil', 'Wine', 'Lightning7', 'InlineSkate', 'Beef',]:
    print(dataset_name)
    df = read_dataset(dataset_name)
    x_train=df[0]
    y_train=df[1]
    x_val=df[2]
    y_val=df[3]
    with tf.keras.utils.custom_object_scope(custom_objects):
        model = tf.keras.models.load_model(f'./results_2.15/fcn/run_0/{dataset_name}/last_model.hdf5')
        fcn_model = [layer for layer in model.layers if 'model' in layer.name][0]

        supervised_model = keras.models.load_model(f"./supervised_fcn/{dataset_name}/best_model.hdf5")

        perplexities = [3,5,10,15]
        train_samples, test_samples = len(x_train),len(x_val)
        for  perp in perplexities:
            print("Perplexity : ",perp)

            plot_filters(supervised_model, fcn_model, dataset_name, perp)


    
