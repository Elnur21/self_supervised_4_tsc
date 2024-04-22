

import pandas as pd
from tensorflow import keras 

from utils.helper import *
from utils.utils import *

from triplet_loss import triplet_loss_function


class CustomNumpy(np.ndarray):
    def __new__(cls, value, dtype):
        obj = np.asarray(value).view(cls)
        return obj



custom_objects = {'temp': triplet_loss_function,'__numpy__':CustomNumpy}




with tf.keras.utils.custom_object_scope(custom_objects):
    for dataset_name in ['ArrowHead', 'BeetleFly', 'Ham', 'MoteStrain', 'OliveOil', 'Wine', 'Lightning7', 'InlineSkate', 'Beef', 'ACSF1', 'Yoga', 'GunPointOldVersusYoung',
                 'FreezerSmallTrain', 'WordSynonyms', 'Car', 'ProximalPhalanxTW', 'InsectWingbeatSound','FaceAll', 'EOGVerticalSignal',  'Earthquakes']:
        print(dataset_name)

        model = tf.keras.models.load_model(f'./results_lite/lite/run_0/{dataset_name}/last_model.hdf5')
        lite_model = [layer for layer in model.layers if 'model' in layer.name][0]

        supervised_model = keras.models.load_model(f"./supervised_lite/{dataset_name}/best_model.hdf5")

        perplexities = [3,5,10,15]
        for  perp in perplexities:
            print("Perplexity : ",perp)

            plot_filters(supervised_model, lite_model, dataset_name, perp)


    
