import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from dtaidistance import dtw
from aeon.datasets import load_classification
import tensorflow.keras as keras
import tensorflow as tf
import os
from aeon.distances import dtw_distance

def read_dataset(dataset_name):
    try:
        encoder = LabelEncoder()
        datasets_dict = {}
        X_train, y_train = load_classification(dataset_name,  split="train")
        X_test, y_test = load_classification(dataset_name,  split="test")
        x_train = X_train.reshape(X_train.shape[0],X_train.shape[2])
        x_test = X_test.reshape(X_test.shape[0],X_test.shape[2])
        y_train = encoder.fit_transform(y_train)
        y_test = encoder.fit_transform(y_test)
        # znorm
        std_ = x_train.std(axis=1, keepdims=True)
        std_[std_ == 0] = 1.0
        x_train = (x_train - x_train.mean(axis=1, keepdims=True)) / std_

        std_ = x_test.std(axis=1, keepdims=True)
        std_[std_ == 0] = 1.0
        x_test = (x_test - x_test.mean(axis=1, keepdims=True)) / std_

        datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                    y_test.copy())

        return datasets_dict[dataset_name]
    except:
        print("Error fetching data")
        return [None]


def get_percent(value):
    return float("%.2f" % float(value*100))


def plot_filters(model_train, model_test,dataset_name, perp,i=1):   
    filters_train = [layer for layer in model_train.layers if 'conv' in layer.name][-1].get_weights()[0]
    filters_test= [layer for layer in model_test.layers if 'conv' in layer.name][-1].get_weights()[0]

    shape = filters_train.shape

    filters_reshaped_train = filters_train.reshape(shape[1],shape[0])
    filters_reshaped_test = filters_test.reshape(shape[1],shape[0])

    concat = np.concatenate((filters_reshaped_train, filters_reshaped_test)) 
    concat =  (concat - np.min(concat)) / (np.max(concat) - np.min(concat))

    tsne = TSNE(n_components=2, init='random', perplexity=perp, random_state=0, metric="precomputed")
    concat_dtw = dtw.distance_matrix_fast(concat.astype(np.double))

    concat_tsne = tsne.fit_transform(concat_dtw)
    
    
    plt.scatter(concat_tsne[:shape[1],0], concat_tsne[:shape[1],1], label=f"Supervised", alpha=0.4)
    # Annotate each point with its index
    for i, txt in enumerate(range(len(concat_tsne[:shape[1],0]))):
        plt.annotate(txt, (concat_tsne[:shape[1],0][i], concat_tsne[:shape[1],1][i]))

    plt.scatter(concat_tsne[shape[1]:,0], concat_tsne[shape[1]:,1], label=f"Self-supervised", alpha=0.4)  
    # Annotate each point with its index
    for i, txt in enumerate(range(len(concat_tsne[:shape[1],0]))):
        plt.annotate(txt, (concat_tsne[shape[1]:,0][i], concat_tsne[shape[1]:,1][i]))

   
    plt.legend()
    plt.title(f'{dataset_name} Scatter Plot of Last Layer Filters')
    plt.xlabel('Filter')
    plt.ylabel('Filter')
    create_directory(f"filters/filters_perplexity_{perp}")
    plt.savefig(f"filters/filters_perplexity_{perp}/"+dataset_name+"_filters.png")
    plt.close()


def get_filters(model_train, model_test, model_name, perp):
    tsne = TSNE(n_components=2, init='random', perplexity=perp, random_state=0, metric="precomputed")
    if(model_name != "Inception" and model_name !="LITE"):
        filters_train, _ = [layer for layer in model_train.layers if 'conv' in layer.name][0].get_weights()
        filters_test, _ = [layer for layer in model_test.layers if 'conv' in layer.name][0].get_weights()
        shape = filters_train.shape

        filters_reshaped_train = filters_train.reshape(shape[2],shape[0])
        filters_reshaped_test = filters_test.reshape(shape[2],shape[0])

        concat = np.concatenate((filters_reshaped_train, filters_reshaped_test)) 
        concat =  (concat - np.min(concat)) / (np.max(concat) - np.min(concat))
        concat_dtw = dtw.distance_matrix_fast(concat.astype(np.double))

        concat_tsne_model = tsne.fit_transform(concat_dtw)

        return  concat_tsne_model, shape
    
    else:
        train_filters=[]
        test_filters=[]
        dtw_train_filters=np.zeros((96,96))
        dtw_test_filters=np.zeros((96,96))
        first_layer_count = 3
        differencies = [40,20,10]
        for i in range(first_layer_count):
            filters_train = [layer for layer in model_train.layers if 'conv' in layer.name][i].get_weights()[0].reshape(32,1,differencies[i])
            filters_test = [layer for layer in model_test.layers if 'conv' in layer.name][i].get_weights()[0].reshape(32,1,differencies[i])
            for arr in filters_train:
                train_filters.append(arr)
            for arr in filters_test:
                test_filters.append(arr)
        for i in range(len(train_filters)):
            for j in range(len(train_filters)):
                dtw_train_filters[i][j]=dtw_distance(train_filters[i],train_filters[j])
        for i in range(len(filters_test)):
            for j in range(len(filters_test)):
                dtw_test_filters[i][j]=dtw_distance(filters_test[i],filters_test[j])
        

        concat = np.concatenate((dtw_train_filters, dtw_test_filters)) 
        concat =  (concat - np.min(concat)) / (np.max(concat) - np.min(concat))

        concat_dtw = dtw.distance_matrix_fast(concat.astype(np.double))

        concat_tsne_model = tsne.fit_transform(concat_dtw)
        print(concat_tsne_model.shape)
        shape = concat_tsne_model.shape
        return  concat_tsne_model, (shape[1],1,int(shape[0]/2))


def plot_all_filters(model_train1, model_test1, model_name1,model_train2, model_test2, model_name2, dataset_name, perp):
    result1, shape1 = get_filters(model_train1, model_test1, model_name1,perp)
    result2, shape2 = get_filters(model_train2, model_test2, model_name2,perp)

    compare_plot(result1, result2, model_name1, model_name2, dataset_name,shape1,shape2, perp)


def compare_plot(model_train1, model_train2,model1,model2,dataset,shape1,shape2,perp, alpha=0.4):
    # Create the main figure with constrained layout
    fig = plt.figure(constrained_layout=True, figsize=(10, 10))
    fig.suptitle(f"Comparison of {model1} and {model2} on {dataset} dataset")
    ax0 = fig.subplots(2, 2)

    ax0[0][0].scatter(model_train1[:shape1[2],0], model_train1[:shape1[2],1],label="Train", color="red", alpha=alpha)
    ax0[0][0].scatter(model_train1[shape1[2]:,0], model_train1[shape1[2]:,1],label="Test", color='green', alpha=alpha)
    ax0[0][0].set_title(f'{model1} Train and Test filters')
    ax0[0][0].legend()

    ax0[0][1].scatter(model_train2[:shape2[2],0], model_train2[:shape2[2],1],label="Train", color="blue", alpha=alpha)
    ax0[0][1].scatter(model_train2[shape2[2]:,0], model_train2[shape2[2]:,1],label="Test", color='orange', alpha=alpha)
    ax0[0][1].set_title(f'{model2} Train and Test filters')
    ax0[0][1].legend()

    ax0[1][0].scatter(model_train1[:shape1[2],0], model_train1[:shape1[2],1],label=model1, color="red", alpha=alpha)
    ax0[1][0].scatter(model_train2[:shape2[2],0], model_train2[:shape2[2],1],label=model2, color='blue', alpha=alpha)
    ax0[1][0].set_title(f'{model1} and {model2} Train filters')
    ax0[1][0].legend()

    ax0[1][1].scatter(model_train1[shape1[2]:,0], model_train1[shape1[2]:,1],label=model1, color="green", alpha=alpha)
    ax0[1][1].scatter(model_train2[shape2[2]:,0], model_train2[shape2[2]:,1],label=model2, color='orange', alpha=alpha)
    ax0[1][1].set_title(f'{model1} and {model2} Test filters')
    ax0[1][1].legend()


    create_directory(f"compare/filters/{model1}_and_{model2}_alpha/filters_perplexity_{perp}/")
    plt.savefig(f"compare/filters/{model1}_and_{model2}_alpha/filters_perplexity_{perp}/"+dataset+f"_filters_alpha_{alpha}.png")
    plt.close()
  

def extract_features(model_name, dataset_name,dataset1=0, dataset2=0, random_gamma=1.5,random_betta=0.5):
    # model_train = keras.models.load_model(f'./results_2.15/fcn/run_0/{dataset_name}/last_model.hdf5')
    model = tf.keras.models.load_model(f'./results_2.15/fcn/run_0/{dataset_name}/last_model.hdf5')
    model_train = [layer for layer in model.layers if 'model' in layer.name][0]
    # model_train=change_bn_parameters(model_train,new_gamma,new_beta)
    model_test = keras.models.load_model(f"./supervised_fcn/{dataset_name}/best_model.hdf5")
    df = read_dataset(dataset_name)

    train_results, test_results = None, None
    gap_layer_train = [layer for layer in model_train.layers if 'global_average' in layer.name][0].name
    pair_model_train = keras.models.Model(inputs = model_train.input, outputs=[model_train.get_layer(gap_layer_train).output])
    train_results=pair_model_train.predict(df[dataset1])[:200]
    train_results = np.sort(train_results, axis=1)

    gap_layer_test = [layer for layer in model_test.layers if 'global_average' in layer.name][0].name
    pair_model_test = keras.models.Model(inputs = model_test.input, outputs=[model_test.get_layer(gap_layer_test).output])
    test_results=pair_model_test.predict(df[dataset2])[:200]
    test_results = np.sort(test_results, axis=1)

    concat = np.concatenate((train_results, test_results)) 

    from sklearn.preprocessing import StandardScaler
    concat = StandardScaler().fit_transform(concat)

    return concat


def change_bn_parameters(model, new_gamma, new_beta):
    # Create a new instance of the model
    new_model = tf.keras.models.clone_model(model)
    
    # Loop through the layers of the model
    for new_layer, layer in zip(new_model.layers, model.layers):
        # Check if the layer is a BatchNormalization layer
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            # Set the gamma and beta parameters to the new values
            new_layer.gamma.assign(tf.ones_like(layer.gamma) * new_gamma)
            new_layer.beta.assign(tf.ones_like(layer.beta) * new_beta)
    
    return new_model


def plot_gap_results_50x50(model_name, dataset_name):
    print(dataset_name)
    output_directory = model_name+'/results/' + dataset_name + '/'
    output_directory_reverse = model_name+'/results_reverse/' + dataset_name + '/'
    model_train = keras.models.load_model(output_directory+'best_model.hdf5')
    model_test = keras.models.load_model(output_directory_reverse+'best_model.hdf5')
    df = read_dataset(dataset_name)

    train_results, test_results = None, None

    half_len = int(len(df[0])/2)
    gap_layer_train = [layer for layer in model_train.layers if 'global_average' in layer.name][0].name
    pair_model_train = keras.models.Model(inputs = model_train.input, outputs=[model_train.get_layer(gap_layer_train).output])
    train_results=pair_model_train.predict(df[0])[:half_len]
    train_results=np.sort(train_results)

    gap_layer_test = [layer for layer in model_test.layers if 'global_average' in layer.name][0].name
    pair_model_test = keras.models.Model(inputs = model_test.input, outputs=[model_test.get_layer(gap_layer_test).output])
    test_results=pair_model_test.predict(df[0])[half_len:]
    test_results=np.sort(test_results)

    concat = np.concatenate((train_results, test_results)) 

    from sklearn.preprocessing import StandardScaler
    concat = StandardScaler().fit_transform(concat)

    perps=[3,5,10,15]
    for perp in perps:
        # if os.path.exists(f"{model_name}/gaps/gap_perplexity_{perp}/"+dataset_name+"_gap.png"):
            # print('Already done')
        # else:
            tsne = TSNE(n_components=2, init='random', perplexity=perp, random_state=0)

            concat_tsne = tsne.fit_transform(concat)

            shape = concat_tsne.shape

            plt.scatter(concat_tsne[:int(shape[0]/2),0], concat_tsne[:int(shape[0]/2),1], color="blue", label=f"Train", alpha=0.4)
            plt.scatter(concat_tsne[int(shape[0]/2):,0], concat_tsne[int(shape[0]/2):,1], color='orange', label=f"Test", alpha=0.4)
            plt.legend()
            plt.title(f'{dataset_name} Scatter Plot of GAP results')
            plt.xlabel('Feature')
            plt.ylabel('Feature')
            create_directory(f"{model_name}/sorted_gaps_50x50/gap_perplexity_{perp}")
            plt.savefig(f"{model_name}/sorted_gaps_50x50/gap_perplexity_{perp}/"+dataset_name+"_gap.png")
            plt.close()


def plot_gap_results(model_name, dataset_name, random_gamma=1.5,random_betta=0.5):
    print(dataset_name)
    concat = extract_features(model_name, dataset_name)

    perps=[3,5,10,15]
    for perp in perps:
        # if os.path.exists(f"{model_name}/gaps/gap_perplexity_{perp}/"+dataset_name+"_gap.png"):
            # print('Already done')
        # else:
            tsne = TSNE(n_components=2, init='random', perplexity=perp, random_state=0)

            concat_tsne = tsne.fit_transform(concat)

            shape = concat_tsne.shape

            plt.scatter(concat_tsne[:int(shape[0]/2),0], concat_tsne[:int(shape[0]/2),1], color="blue", label=f"Train", alpha=0.4)
            plt.scatter(concat_tsne[int(shape[0]/2):,0], concat_tsne[int(shape[0]/2):,1], color='orange', label=f"Test", alpha=0.4)
            plt.legend()
            plt.title(f'{dataset_name} Scatter Plot of GAP results')
            plt.xlabel('Feature')
            plt.ylabel('Feature')
            create_directory(f"{model_name}/sorted_gaps/gap_perplexity_{perp}")
            plt.savefig(f"{model_name}/sorted_gaps/gap_perplexity_{perp}/"+dataset_name+"_gap.png")
            plt.close()


def compare_gap_plot(model_name, dataset_name):
    print(dataset_name)
    concat_train = extract_features(model_name, dataset_name)
    concat_test = extract_features(model_name, dataset_name,2)

    perps=[3,5,10,15]
    for perp in perps:
        # if os.path.exists(f"{model_name}/gaps/gap_perplexity_{perp}/"+dataset_name+"_gap.png"):
            # print('Already done')
        # else:
            tsne = TSNE(n_components=2, init='random', perplexity=perp, random_state=0)

            concat_tsne_train = tsne.fit_transform(concat_train)

            shape_train = concat_tsne_train.shape

            concat_tsne_test = tsne.fit_transform(concat_test)

            shape_test = concat_tsne_test.shape

            # Create the main figure with constrained layout
            fig = plt.figure(constrained_layout=True, figsize=(15, 7))
            fig.suptitle(f"Comparison of {model_name} Test and Train on {dataset_name} dataset")
            ax0 = fig.subplots(1, 2)

            border_colors = ['green', 'red']

            ax0[0].spines['top'].set_color(border_colors[0])
            ax0[0].spines['bottom'].set_color(border_colors[0])
            ax0[0].spines['left'].set_color(border_colors[0])
            ax0[0].spines['right'].set_color(border_colors[0])

            ax0[1].spines['top'].set_color(border_colors[1])
            ax0[1].spines['bottom'].set_color(border_colors[1])
            ax0[1].spines['left'].set_color(border_colors[1])
            ax0[1].spines['right'].set_color(border_colors[1])

            ax0[0].scatter(concat_tsne_train[:int(shape_train[0]/2),0], concat_tsne_train[:int(shape_train[0]/2),1],label="Train dataset", alpha=0.4)
            ax0[0].scatter(concat_tsne_train[int(shape_train[0]/2):,0], concat_tsne_train[int(shape_train[0]/2):,1],label="Test dataset", alpha=0.4)
            ax0[0].set_title(f'Train and Test GAP features for train model')
            ax0[0].legend()

            ax0[1].scatter(concat_tsne_test[:int(shape_test[0]/2),0], concat_tsne_test[:int(shape_test[0]/2),1],label="Train dataset", color="orange", alpha=0.4)
            ax0[1].scatter(concat_tsne_test[int(shape_test[0]/2):,0], concat_tsne_test[int(shape_test[0]/2):,1],label="Test dataset", color='blue', alpha=0.4)
            ax0[1].set_title(f'Train and Test GAP features for test model')
            ax0[1].legend()



            create_directory(f"compare_features/{model_name}/perplexity_{perp}")
            plt.savefig(f"compare_features/{model_name}/perplexity_{perp}/"+dataset_name+"_gap_features.png")
            plt.close()


def plot_gap_results_reverse(model_name, dataset_name):
    print(dataset_name)
    features_num=128
    if(model_name=="LITE"):
        features_num=32
    output_directory = model_name+'/results/' + dataset_name + '/'
    output_directory_reverse = model_name+'/results_reverse/' + dataset_name + '/'
    model_train = keras.models.load_model(output_directory+'best_model.hdf5')
    model_test = keras.models.load_model(output_directory_reverse+'best_model.hdf5')
    df = read_dataset(dataset_name)

    train_results, test_results = None, None
    gap_layer_train = [layer for layer in model_train.layers if 'global_average' in layer.name][0].name
    pair_model_train = keras.models.Model(inputs = model_train.input, outputs=[model_train.get_layer(gap_layer_train).output])
    train_results=pair_model_train.predict(df[0])[:200]
    train_results = train_results.reshape(features_num,train_results.shape[0])

    gap_layer_test = [layer for layer in model_test.layers if 'global_average' in layer.name][0].name
    pair_model_test = keras.models.Model(inputs = model_test.input, outputs=[model_test.get_layer(gap_layer_test).output])
    test_results=pair_model_test.predict(df[0])[:200]
    test_results = test_results.reshape(features_num,test_results.shape[0])


    concat = np.concatenate((train_results, test_results)) 

    from sklearn.preprocessing import StandardScaler
    concat = StandardScaler().fit_transform(concat)
    concat_dtw = dtw.distance_matrix_fast(concat.astype(np.double))

    perps=[3,5,10,15]
    for perp in perps:
        if os.path.exists(f"{model_name}/gaps_reverse/gap_perplexity_{perp}/"+dataset_name+"_gap.png"):
            print('Already done')
        else:
            tsne = TSNE(n_components=2, init='random', perplexity=perp, random_state=0, metric="precomputed")

            concat_tsne = tsne.fit_transform(concat_dtw)

            shape = concat_tsne.shape

            plt.scatter(concat_tsne[:int(shape[0]/2),0], concat_tsne[:int(shape[0]/2),1], color="blue", label=f"Train", alpha=0.4)
            plt.scatter(concat_tsne[int(shape[0]/2):,0], concat_tsne[int(shape[0]/2):,1], color='orange', label=f"Test", alpha=0.4)
            plt.legend()
            plt.title(f'{dataset_name} Scatter Plot of GAP results')
            plt.xlabel('Feature')
            plt.ylabel('Feature')
            create_directory(f"{model_name}/gaps_reverse/gap_perplexity_{perp}")
            plt.savefig(f"{model_name}/gaps_reverse/gap_perplexity_{perp}/"+dataset_name+"_gap.png")
            plt.close()


def plot_gap_results_2samples(model_name, dataset_name, reverse=False):
    print(dataset_name)
    features_num=128
    if(model_name=="LITE"):
        features_num=32
    output_directory = model_name+'/results/' + dataset_name + '/'
    if(reverse):
        output_directory = model_name+'/results_reverse/' + dataset_name + '/'
    model_train = keras.models.load_model(output_directory+'best_model.hdf5')
    df = read_dataset(dataset_name)

    train_results, test_results = None, None
    gap_layer_train = [layer for layer in model_train.layers if 'global_average' in layer.name][0].name
    pair_model_train = keras.models.Model(inputs = model_train.input, outputs=[model_train.get_layer(gap_layer_train).output])
    train_results=pair_model_train.predict(df[0])[:2].reshape(features_num,2)
    train_results=np.array(train_results)
    test_results=pair_model_train.predict(df[2])[:2].reshape(features_num,2)
    test_results=np.array(test_results)

    concat = np.concatenate((train_results, test_results)) 
    from sklearn.preprocessing import StandardScaler
    concat = StandardScaler().fit_transform(concat)

    if os.path.exists(f"{model_name}/sorted_gaps_2samples/"+dataset_name+"_gap.png"):
        print('Already done')
    else:

        shape = concat.shape

        plt.scatter(concat[:int(shape[0]/2),0], concat[:int(shape[0]/2),1], color="blue", label=f"Train", alpha=0.4)
        plt.scatter(concat[int(shape[0]/2):,0], concat[int(shape[0]/2):,1], color='orange', label=f"Test", alpha=0.4)
        plt.legend()
        plt.title(f'{dataset_name} Scatter Plot of GAP results')
        plt.xlabel('Feature')
        plt.ylabel('Feature')
        create_directory(f"{model_name}/sorted_gaps_2samples/{dataset_name}")
        if(reverse):
            plt.savefig(f"{model_name}/sorted_gaps_2samples/"+dataset_name+"/train_model.png")
        else:
            plt.savefig(f"{model_name}/sorted_gaps_2samples/"+dataset_name+"/test_model.png")
        plt.close()


def dynamic_plot(dataset,perp):
    import plotly.graph_objects as go
    filters = []
    shape=None
    for run in range(7):
        model_train = keras.models.load_model(f'LITE/results_7trains/{run}/{dataset}/best_model.hdf5')
        filters_train = [layer for layer in model_train.layers if 'conv' in layer.name][-1].get_weights()[0]

        shape = filters_train.shape

        filters_reshaped_train = filters_train.reshape(shape[1],shape[0])

        filters.append(filters_reshaped_train)

    concat = np.vstack(filters)
    concat =  (concat - np.min(concat)) / (np.max(concat) - np.min(concat))

    tsne = TSNE(n_components=2, init='random', perplexity=perp, random_state=0, metric="precomputed")
    concat_dtw = dtw.distance_matrix_fast(concat.astype(np.double))

    concat_tsne = tsne.fit_transform(concat_dtw)


    fig = go.Figure()

    for i in range(1,8):
        fig.add_trace(go.Scatter(
            x=concat_tsne[32*(i-1):32*i,0], y=concat_tsne[32*(i-1):32*i,1],
            name=f'run {i}',
            mode='markers'
        ))


    # Set options common to all traces with fig.update_traces
    fig.update_traces(mode='markers', marker_line_width=2, marker_size=10)
    fig.update_layout(title=f'{dataset} 7 runs scatter',
                    yaxis_zeroline=False, xaxis_zeroline=False)


    fig.show()


def create_directory(directory_path):
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile !:(
            return None
        return directory_path