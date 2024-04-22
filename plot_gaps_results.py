from utils.helper import *
from utils.utils import *

from triplet_loss import triplet_loss_function


class CustomNumpy(np.ndarray):
    def __new__(cls, value, dtype):
        obj = np.asarray(value).view(cls)
        return obj



custom_objects = {'temp': triplet_loss_function,'__numpy__':CustomNumpy}



with tf.keras.utils.custom_object_scope(custom_objects):
    for dataset_name in np.array(['ArrowHead', 'BeetleFly', 'Ham', 'MoteStrain', 'OliveOil', 'Wine', 'Lightning7', 'InlineSkate', 'Beef', 'ACSF1', 'Yoga', 'GunPointOldVersusYoung',
                 'FreezerSmallTrain', 'WordSynonyms', 'Car', 'ProximalPhalanxTW', 'InsectWingbeatSound','FaceAll', 'EOGVerticalSignal',  'Earthquakes']):
        #plot GAP results
        plot_gap_results("LITE", dataset_name)

        # #plot GAP results according to 2 samples
        # plot_gap_results_2samples(model, dataset_name)
        # plot_gap_results_2samples(model, dataset_name, reverse=True)
        # plot_gap_results_50x50(model, dataset_name)

        # #reshape GAP results reverse and plot
        # plot_gap_results_reverse(model, dataset_name)