from utils.helper import *
from utils.utils import *

from triplet_loss import triplet_loss_function


custom_objects = {'temp': triplet_loss_function}



with tf.keras.utils.custom_object_scope(custom_objects):
    for dataset_name in np.array(['ArrowHead', 'BeetleFly', 'Ham', 'MoteStrain', 'OliveOil', 'Wine', 'Lightning7', 'InlineSkate', 'Beef',]):
        #plot GAP results
        plot_gap_results("FCN", dataset_name)

        # #plot GAP results according to 2 samples
        # plot_gap_results_2samples(model, dataset_name)
        # plot_gap_results_2samples(model, dataset_name, reverse=True)
        # plot_gap_results_50x50(model, dataset_name)

        # #reshape GAP results reverse and plot
        # plot_gap_results_reverse(model, dataset_name)