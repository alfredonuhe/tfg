# %% [markdown]
# # Deep Learning Model

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-08-30T06:49:15.032097Z","iopub.execute_input":"2021-08-30T06:49:15.032421Z","iopub.status.idle":"2021-08-30T06:49:19.518869Z","shell.execute_reply.started":"2021-08-30T06:49:15.032351Z","shell.execute_reply":"2021-08-30T06:49:19.518044Z"}}
# Load libraries
import numpy as np
np.random.seed(1)
import tensorflow as tf

from tensorflow.keras.models import Sequential, save_model, model_from_json
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import time
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
from keras.preprocessing import image

import matplotlib.image as mpimg

plt.rcParams.update({'figure.max_open_warning': 0})

# Required line to avoid issue issue https://github.com/slundberg/shap/issues/1694#issue-773518362
tf.compat.v1.disable_v2_behavior()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-08-30T06:49:19.520188Z","iopub.execute_input":"2021-08-30T06:49:19.520518Z","iopub.status.idle":"2021-08-30T06:49:19.52481Z","shell.execute_reply.started":"2021-08-30T06:49:19.520484Z","shell.execute_reply":"2021-08-30T06:49:19.524071Z"}}
#Time measure
total_start_time = time.time()
# Cofig variables
LOAD_MODEL = False
SAVE_MODEL = True
TEST_MODEL = True
RUN_CONFUSION_MATRIX = True
RUN_TENSORBOARD = True
CLEAR_TENSORBOARD_LOGS = True
ZIP_RESULTS = False
RUN_SHAP = True
RUN_SALIENCY_MAPS = False
FREE_GPU_MEMORY = False

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-08-30T06:49:19.526609Z","iopub.execute_input":"2021-08-30T06:49:19.527171Z","iopub.status.idle":"2021-08-30T06:49:19.537598Z","shell.execute_reply.started":"2021-08-30T06:49:19.52712Z","shell.execute_reply":"2021-08-30T06:49:19.53687Z"}}
# Point to the 3 directories
train_dir = '../input/gravity-spy-gravitational-waves/train/train/'
validation_dir = '../input/gravity-spy-gravitational-waves/validation/validation/'
test_dir = '../input/gravity-spy-gravitational-waves/test/test/'

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-08-30T06:49:19.538888Z","iopub.execute_input":"2021-08-30T06:49:19.5393Z","iopub.status.idle":"2021-08-30T06:49:19.5479Z","shell.execute_reply.started":"2021-08-30T06:49:19.539264Z","shell.execute_reply":"2021-08-30T06:49:19.547017Z"}}
# Create the data generators
train_datagen = ImageDataGenerator(rescale=1. / 255)
validation_datagen = ImageDataGenerator(rescale=1. / 255)  
test_datagen = ImageDataGenerator(rescale=1. / 255)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-08-30T06:49:19.551005Z","iopub.execute_input":"2021-08-30T06:49:19.551473Z","iopub.status.idle":"2021-08-30T06:49:21.146969Z","shell.execute_reply.started":"2021-08-30T06:49:19.551443Z","shell.execute_reply":"2021-08-30T06:49:21.146038Z"}}
# Test if GPU is available
device_name = tf.test.gpu_device_name()
print('GPU avaliable: ', device_name)
if device_name != '/device:GPU:0':
  print(
      '\n\nThis error most likely means that this notebook is not '
      'configured to use a GPU.  Change this in Notebook Settings via the '
      'command palette (cmd/ctrl-shift-P) or the Edit menu.\n\n')
  raise SystemError('GPU device not found')

# %% [code] {"execution":{"iopub.status.busy":"2021-08-30T06:49:35.669523Z","iopub.execute_input":"2021-08-30T06:49:35.669839Z","iopub.status.idle":"2021-08-30T06:49:35.760236Z","shell.execute_reply.started":"2021-08-30T06:49:35.669809Z","shell.execute_reply":"2021-08-30T06:49:35.759397Z"}}
# Get the class names
df = pd.read_csv("../input/gravity-spy-gravitational-waves/trainingset_v1d1_metadata.csv")
classes_list = df.label.value_counts().index
classes_list = list(classes_list)
print(classes_list)

# %% [markdown]
# ## Data Sources

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-08-28T15:57:28.168318Z","iopub.execute_input":"2021-08-28T15:57:28.16864Z","iopub.status.idle":"2021-08-28T15:57:46.8808Z","shell.execute_reply.started":"2021-08-28T15:57:28.168607Z","shell.execute_reply":"2021-08-28T15:57:46.879899Z"}}
# Data sources
training_batch_size = 64
validation_batch_size = 32
img_dim = 250

train_generator = train_datagen.flow_from_directory(
  train_dir,                                                  
  classes = classes_list,
  target_size = (img_dim, img_dim),            
  batch_size = training_batch_size,
  class_mode = "categorical",
  shuffle = True,
  seed = 123)

validation_generator = validation_datagen.flow_from_directory(
  validation_dir,
  classes = classes_list,
  target_size = (img_dim, img_dim),
  batch_size = validation_batch_size,
  class_mode = "categorical",
  shuffle = True,
  seed = 123)

test_size = !find '../input/gravity-spy-gravitational-waves/test/test/' -type f | wc -l
test_size = int(test_size[0])
test_batch_size = 1

test_generator = test_datagen.flow_from_directory(
  test_dir,
  classes = classes_list,
  target_size = (img_dim, img_dim),
  batch_size = test_batch_size,
  class_mode = "categorical",
  shuffle = False,
  seed = 3)

# %% [markdown]
# ## Model

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-08-28T15:57:46.883502Z","iopub.execute_input":"2021-08-28T15:57:46.883844Z","iopub.status.idle":"2021-08-28T16:03:20.966005Z","shell.execute_reply.started":"2021-08-28T15:57:46.883811Z","shell.execute_reply":"2021-08-28T16:03:20.96492Z"}}
# CNN
input_shape = (img_dim, img_dim, 3)
epochs = 10
model = None

# Define and train model or load a pretrained model from a different session to save time
if LOAD_MODEL:
    # Load json and create model
    json_file = open('../input/saved-model/model_kaggle.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # Load weights into new model
    loaded_model.load_weights("../input/saved-model/model_kaggle.h5")
    model = loaded_model
else:
    tf.keras.backend.clear_session()
    model = Sequential()
    model.add(tf.keras.layers.Conv2D(5, (3, 3), input_shape=input_shape))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(tf.keras.layers.Conv2D(5, (3, 3)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(tf.keras.layers.Conv2D(5, (3, 3)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(img_dim, activation="relu"))
    model.add(tf.keras.layers.Dense(22, activation="softmax"))

    # Now, train the model
    model.compile(loss = "categorical_crossentropy",  
                  optimizer = 'adam', 
                  metrics=["accuracy"])

    training_step_size = 32
    validation_step_size = 32
    
    # Run tensorboard for graph inspection
    if RUN_TENSORBOARD:
        if CLEAR_TENSORBOARD_LOGS:
            # Clear any logs from previous runs
            !rm -rf ./logs/ 
            !mkdir ./logs/
        
        !rm -rf ./ngork
        !rm -rf ./ngork-stable-linux-amd64.*

        # Download Ngrok to tunnel the tensorboard port to an external port
        !wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
        !unzip -o ngrok-stable-linux-amd64.zip

        # Run tensorboard as well as Ngrox (for tunneling as non-blocking processes)
        import os
        import datetime
        import multiprocessing


        pool = multiprocessing.Pool(processes = 10)
        results_of_processes = [pool.apply_async(os.system, args=(cmd, ), callback = None )
                                for cmd in [
                                f"tensorboard --logdir ./logs/ --host 0.0.0.0 --port 6006 &",
                                "./ngrok http 6006 &"
                                ]]

        # Get the url to access tensorboad
        ! curl -s http://localhost:4040/api/tunnels | python3 -c \
            "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"
                
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(
        train_generator,
        steps_per_epoch = training_step_size,
        epochs = epochs,
        validation_data = validation_generator,
        validation_steps = validation_step_size,
        verbose = 1)
    
    # Take a look at the accuracy and loss
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # Summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-08-28T16:03:20.967993Z","iopub.execute_input":"2021-08-28T16:03:20.968365Z","iopub.status.idle":"2021-08-28T16:03:21.168978Z","shell.execute_reply.started":"2021-08-28T16:03:20.968325Z","shell.execute_reply":"2021-08-28T16:03:21.168134Z"}}
# Save model for future use
if SAVE_MODEL:
    # Serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # Serialize weights to HDF5
    model.save("model.h5")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-08-28T16:03:21.173292Z","iopub.execute_input":"2021-08-28T16:03:21.175204Z","iopub.status.idle":"2021-08-28T16:04:23.404449Z","shell.execute_reply.started":"2021-08-28T16:03:21.175166Z","shell.execute_reply":"2021-08-28T16:04:23.403563Z"}}
# Make the predictions on the test set
# Line declared for use in the next code cells
df = None
if TEST_MODEL:
    # Test model
    print('Testing model...', end='')
    predictions = model.predict(test_generator, steps = test_size, verbose = 1)
    print('completed.')

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-08-28T16:04:23.40581Z","iopub.execute_input":"2021-08-28T16:04:23.406186Z","iopub.status.idle":"2021-08-28T16:04:23.509666Z","shell.execute_reply.started":"2021-08-28T16:04:23.406145Z","shell.execute_reply":"2021-08-28T16:04:23.508788Z"}}
if TEST_MODEL:
    # Accuracy
    df = pd.DataFrame(predictions)
    df['filename'] = test_generator.filenames
    df['truth'] = np.nan
    df['truth'] = df['filename'].str.split('/', 1, expand = True)
    df['prediction_index'] = df[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]].idxmax(axis=1).copy()
    df['prediction'] = np.nan

    for i in range(0,22):
        df['prediction'][df['prediction_index'] == i] = classes_list[i]

    accuracy = accuracy_score(df['truth'], df['prediction'])
    print(accuracy)

# %% [markdown]
# ## Confusion matrix

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-08-28T16:04:23.511013Z","iopub.execute_input":"2021-08-28T16:04:23.511378Z","iopub.status.idle":"2021-08-28T16:04:24.311837Z","shell.execute_reply.started":"2021-08-28T16:04:23.511342Z","shell.execute_reply":"2021-08-28T16:04:24.311001Z"}}
if TEST_MODEL:
    # Create a confusion matrix
    if RUN_CONFUSION_MATRIX:    
        cm = confusion_matrix(df['truth'], df['prediction'])
        cm_df = pd.DataFrame(cm)
        cm_df.columns = classes_list
        cm_df['signal'] = classes_list

        # Plot the confusion matrix as a correlation plot
        import seaborn as sns

        plt.figure(figsize=(12, 12))

        corr = cm_df.corr()
        # Warning raised in this line.
        ax = sns.heatmap(
            corr, 
            vmin=0, vmax=1, center=0.5,
            cmap=sns.diverging_palette(0, 200, n=200),
            square=True
        )
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            horizontalalignment='right'
        )
        
        # Calculate the N max values and their indexes in confusion matrix
        df_data = {
            'value': [],
            'row_index': [],
            'col_index': []
        }
        num_max_values = 5
        for i in range(len(corr)):
            for j in range(len(corr.iloc[i])):
                end_col = (corr.iloc[i,j] == 1)
                if end_col:
                    break
                df_data['value'].append(corr.iloc[i,j])
                df_data['row_index'].append(corr.iloc[i].index[i])
                df_data['col_index'].append(corr.iloc[i].index[j])
        corr_df = pd.DataFrame(df_data)
        print('Top values in confusion matrix:\n', corr_df.nlargest(10, ['value']))

# %% [markdown]
# ## Worst Predictions

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-08-28T16:04:24.313191Z","iopub.execute_input":"2021-08-28T16:04:24.313537Z","iopub.status.idle":"2021-08-28T16:04:24.466278Z","shell.execute_reply.started":"2021-08-28T16:04:24.3135Z","shell.execute_reply":"2021-08-28T16:04:24.465256Z"}}
# Line declared for use in the next code cells
pivot_table_df = None
worst_predictions_df = None
if TEST_MODEL:
    # Check the number of errors in the model and which predictions the least accurate.
    df['prediction_is_correct'] = df.apply(lambda x : True if x['truth'] == x['prediction'] else False, axis = 1)
    pd.set_option('display.max_rows', None)
    pivot_table_df = pd.pivot_table(df.loc[:,['truth', 'prediction_is_correct']], index=['truth'], \
        columns=['prediction_is_correct'], aggfunc=len, fill_value=0).sort_values(\
        by=False, ascending=False).head()
    print(pivot_table_df, '\n')
    pivot_table_df = pd.pivot_table(df.loc[:,['truth', 'prediction', 'prediction_is_correct']], \
        index=['truth', 'prediction'], columns=['prediction_is_correct'], aggfunc=len, \
        fill_value=0).sort_values(by='truth', ascending=True)
    print(pivot_table_df, '\n')
    pivot_table_df = pd.pivot_table(df.loc[:,['truth', 'prediction', 'prediction_is_correct']], \
        index=['truth', 'prediction'], columns=['prediction_is_correct'], aggfunc=len, \
        fill_value=0).sort_values(by=False, ascending=False).head()
    print(pivot_table_df, '\n')
    
    # Collect x images from the top worst prediction pairs
    num_images = 5
    df_aux_0 = df[(df.truth == pivot_table_df.index[0][0]) & \
        (df.prediction == pivot_table_df.index[0][1]) & \
        (df.prediction_is_correct == False) & \
        (df.filename.str.contains('4.0.png'))].head(num_images)
    df_aux_1 = df[(df.truth == pivot_table_df.index[1][0]) & \
        (df.prediction == pivot_table_df.index[1][1]) & \
        (df.prediction_is_correct == False) & \
        (df.filename.str.contains('4.0.png'))].head(num_images)
    worst_predictions_df = pd.concat([df_aux_0,df_aux_1], ignore_index=True)
    print(worst_predictions_df[['filename', 'truth', 'prediction', 'prediction_is_correct']])
