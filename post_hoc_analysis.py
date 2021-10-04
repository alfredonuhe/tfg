# %% [markdown]
# ## Saliency Maps and Layer Visualizations

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-08-30T10:50:43.314136Z","iopub.execute_input":"2021-08-30T10:50:43.314541Z","iopub.status.idle":"2021-08-30T10:50:51.44548Z","shell.execute_reply.started":"2021-08-30T10:50:43.314511Z","shell.execute_reply":"2021-08-30T10:50:51.444236Z"}}
!pip install tf_keras_vis tensorflow

%reload_ext autoreload
%autoreload 2

import os, subprocess
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline

import tensorflow as tf
from tf_keras_vis.utils import num_of_gpus

_, gpus = num_of_gpus()
print('Tensorflow recognized {} GPUs'.format(gpus))

# Load Pretrained Model

from tensorflow.keras.models import load_model

# Global variables
# 0 -> run nothing, 1 -> run saliency vis, 2 -> run salienvy + filter vis , 3 -> run saliency + filter + dense vis
RUN_CONFIG = 1
DATA = {
    'models': ['model_5_second.h5', 'model_5_5_first.h5', 'model_5_5_5_first.h5'],
    'model_names' : ['model_0', 'model_1', 'model_2'],
    'analysis_type':['imgs_predicted_classes', 'imgs_correct_classes'],
    'plot_title_addon': ['prediction', 'truth'],
    'imgs_selected': [0, 1, 2, 3, 4, 5],
    'imgs_names': [
        ['rb_0', 'rb_1', 'rb_2', 'rb_3', 'rb_4', 'rb_5'], 
        ['lfb_0', 'lfb_1', 'lfb_2', 'lfb_3', 'lfb_4', 'lfb_5'],
        ['lfl_0', 'lfl_1', 'lfl_2', 'lfl_3', 'lfl_4'],
        ['ng_0', 'ng_1', 'ng_2', 'ng_3', 'ng_4'],
        ['pl_0', 'pl_1', 'pl_2', 'pl_3', 'pl_4'],
        ['s_0', 's_1', 's_2', 's_3', 's_4'],
    ],
    'imgs_urls': [
        [
            'https://i.imgur.com/jN76Iiq.png',
            'https://i.imgur.com/Wz2YrRJ.png', 
            'https://i.imgur.com/uzltmTi.png', 
            'https://i.imgur.com/xBNadRv.png', 
            'https://i.imgur.com/fZZeBWP.png', 
            'https://i.imgur.com/BmJOBGG.png'
        ],
        [
            'https://i.imgur.com/focrVaO.png',
            'https://i.imgur.com/3C0rVfr.png', 
            'https://i.imgur.com/ZIlnzSw.png', 
            'https://i.imgur.com/vacwwdl.png', 
            'https://i.imgur.com/KiLtjw8.png', 
            'https://i.imgur.com/A5RZPVJ.png'
        ],
        [
            'https://i.imgur.com/0Emt9H0.png', 
            'https://i.imgur.com/1vOYYoo.png', 
            'https://i.imgur.com/iGrgsrr.png', 
            'https://i.imgur.com/27JvZQJ.png', 
            'https://i.imgur.com/vVGUtmU.png'
        ],
        [
            'https://i.imgur.com/OcaOe1C.png', 
            'https://i.imgur.com/UkxkwXs.png', 
            'https://i.imgur.com/A27MGBr.png', 
            'https://i.imgur.com/LhgKmT0.png', 
            'https://i.imgur.com/gxuuMqt.png'
        ],
        [
            'https://i.imgur.com/rRVnpVE.png', 
            'https://i.imgur.com/IC2HObr.png', 
            'https://i.imgur.com/EAZttug.png', 
            'https://i.imgur.com/8CCyjHX.png', 
            'https://i.imgur.com/dvoS8Z9.png'
        ],
        [
            'https://i.imgur.com/GQlXAX8.png', 
            'https://i.imgur.com/fZxVGXu.png', 
            'https://i.imgur.com/ym2Xb8B.png', 
            'https://i.imgur.com/wXdstsD.png', 
            'https://i.imgur.com/6fEXyGf.png'
        ]
    ],
    'class_names': ['Blip', 'Koi_Fish', 'Low_Frequency_Burst', 'Light_Modulation', 
        'Power_Line', 'Extremely_Loud', 'Low_Frequency_Lines', 'Scattered_Light', 
        'Violin_Mode', 'Scratchy', '1080Lines', 'Whistle', 'Helix', 'Repeating_Blips', 
        'No_Glitch', 'Tomte', 'None_of_the_Above', '1400Ripples', 'Chirp', 
        'Air_Compressor', 'Wandering_Line', 'Paired_Doves'],
    'imgs_correct_classes': [13, 2, 6, 14, 4, 9],
    'imgs_predicted_classes': [0, 6, 2, 6, 4, 9]
}

# Navigate to main directory
os.chdir('/kaggle/working/')

# Remove previous images directory
subprocess.check_call('rm -rf images', shell=True, cwd='/kaggle/working/')

# Create image directory for downloads
subprocess.check_call('mkdir images', shell=True, cwd='/kaggle/working/')


# %% [markdown]
# ## Gradient-based Saliency Map Visualization

# %% [code] {"execution":{"iopub.status.busy":"2021-08-30T10:50:51.447526Z","iopub.execute_input":"2021-08-30T10:50:51.447837Z","iopub.status.idle":"2021-08-30T10:57:33.161272Z","shell.execute_reply.started":"2021-08-30T10:50:51.447804Z","shell.execute_reply":"2021-08-30T10:57:33.160176Z"}}
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
    
if RUN_CONFIG > 0:
    for model_file_name in DATA['models']: 
        model = load_model('../input/saved-model/' + model_file_name)
        model_name = DATA['model_names'][DATA['models'].index(model_file_name)]
        model.summary()
        for i in DATA['imgs_selected']:
            # Inititalize plot
            f, ax = plt.subplots(nrows=len(DATA['imgs_names'][i]), ncols=7, figsize=(28, 4*len(DATA['imgs_names'][i])))
            for j, name in enumerate(DATA['imgs_names'][i]):
                # Download image for analysis
                subprocess.check_call('wget ' + '--output-document=' + str(name) + '.png ' + str(DATA['imgs_urls'][i][j]), shell=True, cwd='/kaggle/working/images')

                # Load images and Convert them to a Numpy array
                images = []
                images.append(np.array(load_img('images/' + name + '.png', target_size=(250, 250))))
                images = np.asarray(images)

                # Add plain image to plot
                ax[j][0].set_title(name, fontsize=16)
                ax[j][0].imshow(images[0], cmap='jet')
                ax[j][0].axis('off')

                # Preparing input data for VGG16
                X = preprocess_input(images)
                X.shape
                
                # Calculate predicted and correct classes
                prediction = DATA['class_names'][DATA['imgs_predicted_classes'][i]]
                truth = DATA['class_names'][DATA['imgs_correct_classes'][i]]
                
                for m, key in enumerate(DATA['analysis_type']):
                    plot_index_base = m * 3
                    
                    # Vanilla Saliency

                    from tf_keras_vis.utils.model_modifiers import ReplaceToLinear

                    replace2linear = ReplaceToLinear()

                    # Instead of using ReplaceToLinear instance,
                    # you can also define the function from scratch as follows:
                    # def model_modifier_function(cloned_model):
                    #     cloned_model.layers[-1].activation = tf.keras.activations.linear

                    from tf_keras_vis.utils.scores import CategoricalScore

                    # 1 is the imagenet index corresponding to Goldfish, 294 to Bear and 413 to Assault Rifle.
                    class_number = DATA[DATA['analysis_type'][m]][i]
                    score = CategoricalScore([class_number])

                    # Instead of using CategoricalScore object,
                    # you can also define the function from scratch as follows:
                    # def score_function(output):
                    #     # The `output` variable refers to the output of the model,
                    #     # so, in this case, `output` shape is `(3, 1000)` i.e., (samples, classes).
                    #     return (output[0][1], output[1][294], output[2][413])

                    from tensorflow.keras import backend as K
                    from tf_keras_vis.saliency import Saliency
                    # from tf_keras_vis.utils import normalize

                    # Create Saliency object.
                    saliency = Saliency(model,
                        model_modifier=replace2linear,
                        clone=True)

                    # Generate saliency map
                    saliency_map = saliency(score, X)

                    ## Since v0.6.0, calling `normalize()` is NOT necessary.
                    # saliency_map = normalize(saliency_map)

                    # Add vanilla saliency to plot
                    ax[j][plot_index_base + 1].set_title('Vanilla ' + DATA['plot_title_addon'][m], fontsize=16)
                    ax[j][plot_index_base + 1].imshow(saliency_map[0], cmap='jet')
                    ax[j][plot_index_base + 1].axis('off')

                    # SmoothGrad

                    # Generate saliency map with smoothing that reduce noise by adding noise
                    saliency_map = saliency(score,
                                            X,
                                            smooth_samples=20, # The number of calculating gradients iterations.
                                            smooth_noise=0.20) # noise spread level.

                    ## Since v0.6.0, calling `normalize()` is NOT necessary.
                    # saliency_map = normalize(saliency_map)

                    # Add smoothgrad to plot
                    ax[j][plot_index_base + 2].set_title('SmoothGrad ' + DATA['plot_title_addon'][m], fontsize=16)
                    ax[j][plot_index_base + 2].imshow(saliency_map[0], cmap='jet')
                    ax[j][plot_index_base + 2].axis('off')

                    # GradCAM

                    from matplotlib import cm
                    from tf_keras_vis.gradcam import Gradcam

                    # Create Gradcam object
                    gradcam = Gradcam(model,
                        model_modifier=replace2linear,
                        clone=True)

                    # Generate heatmap with GradCAM
                    cam = gradcam(score,
                                  X,
                                  penultimate_layer=-1)

                    ## Since v0.6.0, calling `normalize()` is NOT necessary.
                    # cam = normalize(cam)

                    # Add gradcam to plot
                    heatmap = np.uint8(cm.jet(cam[0])[..., :3] * 255)
                    ax[j][plot_index_base + 3].set_title('Grad-CAM ' + DATA['plot_title_addon'][m], fontsize=16)
                    ax[j][plot_index_base + 3].imshow(images[0])
                    ax[j][plot_index_base + 3].imshow(heatmap, cmap='jet', alpha=0.5) # overlay
                    ax[j][plot_index_base + 3].axis('off')
            
            plt.tight_layout()
            f.suptitle('Prediction: ' + str(prediction) + '. Truth: ' + str(truth) + '. Model: ' + str(model_name) + '.', fontsize=30)
            f.subplots_adjust(top=0.88)
            plt.savefig('images/' + truth + '_' + model_name + '_' + str(i) + '.png')
            plt.show()

# %% [markdown]
# ## Convolutional Filter Visualization

# %% [code] {"execution":{"iopub.status.busy":"2021-08-30T10:57:33.163296Z","iopub.execute_input":"2021-08-30T10:57:33.163638Z","iopub.status.idle":"2021-08-30T10:57:33.693815Z","shell.execute_reply.started":"2021-08-30T10:57:33.163602Z","shell.execute_reply":"2021-08-30T10:57:33.69286Z"}}
if RUN_CONFIG > 1:
    # Visualize Convolutional Filter

    # Now visualize a convolutional filter
    from tf_keras_vis.utils.model_modifiers import ExtractIntermediateLayer, ReplaceToLinear

    layer_name = 'conv2d_1' # The target layer that is the last layer of VGG16.

    # This instance constructs new model whose output is replaced to `block5_conv3` layer's output.
    extract_intermediate_layer = ExtractIntermediateLayer(index_or_name=layer_name)
    # This instance modify the model's last activation function to linear one.
    replace2linear = ReplaceToLinear()

    # Instead of using ExtractIntermediateLayer and ReplaceToLinear instance,
    # you can also define the function from scratch as follows:
    def model_modifier_function(current_model):
        target_layer = current_model.get_layer(name=layer_name)
        target_layer.activation = tf.keras.activations.linear
        new_model = tf.keras.Model(inputs=current_model.inputs,
                                   outputs=target_layer.output)
        return new_model

    from tf_keras_vis.utils.scores import CategoricalScore

    filter_number = 3
    score = CategoricalScore(filter_number)

    # Instead of using CategoricalScore object above,
    # you can also define the function from scratch as follows:
    def score_function(output):
        return output[..., filter_number]

    from tf_keras_vis.activation_maximization import ActivationMaximization

    activation_maximization = ActivationMaximization(model,
                                                     # Please note that `extract_intermediate_layer` has to come before `replace2linear`.
                                                     model_modifier=[extract_intermediate_layer, replace2linear],
                                                     clone=False)

    from tf_keras_vis.activation_maximization.callbacks import Progress

    # Generate maximized activation
    activations = activation_maximization(score,
                                          callbacks=[Progress()])

    ## Since v0.6.0, calling `astype()` is NOT necessary.
    # activations = activations[0].astype(np.uint8)

    # Render
    f, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(activations[0])
    ax.set_title('filter[{:03d}]'.format(filter_number), fontsize=16)
    ax.axis('off')
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Dense Layer Visualization

# %% [code] {"execution":{"iopub.status.busy":"2021-08-30T10:35:39.057264Z","iopub.execute_input":"2021-08-30T10:35:39.057655Z","iopub.status.idle":"2021-08-30T10:35:39.458339Z","shell.execute_reply.started":"2021-08-30T10:35:39.057622Z","shell.execute_reply":"2021-08-30T10:35:39.457494Z"}}
if RUN_CONFIG > 2:
    # Visualize Dense layer

    from tf_keras_vis.utils.model_modifiers import ReplaceToLinear

    replace2linear = ReplaceToLinear()

    # Instead of using ReplaceToLinear instance,
    # you can also define the function from scratch as follows:
    def model_modifier_function(cloned_model):
        cloned_model.layers[-1].activation = tf.keras.activations.linear

    from tf_keras_vis.utils.scores import CategoricalScore

    # 20 is the imagenet index corresponding to Ouzel.
    score = CategoricalScore(8)

    # Instead of using CategoricalScore object above,
    # you can also define the function from scratch as follows:
    def score_function(output):
        # The `output` variable refer to the output of the model,
        # so, in this case, `output` shape is `(1, 1000)` i.e., (samples, classes).
        return output[:, 8]

    from tf_keras_vis.activation_maximization import ActivationMaximization

    activation_maximization = ActivationMaximization(model,
                                                     model_modifier=replace2linear,
                                                     clone=True)
    
    from tf_keras_vis.activation_maximization.callbacks import Progress
    from tf_keras_vis.activation_maximization.input_modifiers import Jitter, Rotate2D, Scale
    from tf_keras_vis.activation_maximization.regularizers import Norm, TotalVariation2D

    # Generate maximized activation
    activations = activation_maximization(score,
                                          callbacks=[Progress()])

    ## Since v0.6.0, calling `astype()` is NOT necessary.
    # activations = activations[0].astype(np.uint8)

    # Render
    f, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(activations[0])
    ax.set_title('Koi Fish', fontsize=16)
    ax.axis('off')
    plt.tight_layout()
    plt.show()
