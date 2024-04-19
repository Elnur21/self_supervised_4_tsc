import tensorflow as tf
from aeon.datasets import load_classification

from triplet_loss import triplet_loss_function


custom_objects = {'temp': triplet_loss_function}

with tf.keras.utils.custom_object_scope(custom_objects):
    model = tf.keras.models.load_model('./results_2.15/fcn/run_0/ArrowHead/last_model.hdf5')
    # layers = model
    print("main model layers: ")
    for layer in model.layers:
        print(layer.name)
    print("\n\n")
    fcn_model = [layer for layer in model.layers if 'model' in layer.name][0]
    # pair_model_train = tf.keras.models.Model(inputs = model.input, outputs=[model.get_layer("concatenate").output])
    print("fcn layers: ")
    for layer in fcn_model.layers:
        print(layer.name)
    
    X_train, y_train = load_classification("ArrowHead",  split="train")
    X_train= X_train.reshape(X_train.shape[0],X_train.shape[2])
    pair_model_train = tf.keras.models.Model(inputs = fcn_model.input, outputs=[fcn_model.get_layer("global_average_pooling1d").output])
    train_results=pair_model_train.predict(X_train)[:200]
    print(train_results.shape)

