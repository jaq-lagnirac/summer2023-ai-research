# Neural Network Libraries
import gc
from tkinter import ROUND
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

import typing
from tensorflow import keras as K
if typing.TYPE_CHECKING:
    from keras.api._v2 import keras

# Discord Libraries
import os # allows program to read in discord bot token
import discord # allows interface with discord bot api
# import nest_asyncio # allows infinite running event loop
from dotenv import load_dotenv # loads in .env file
import time # allows for timestamps and elapsed time

# Discord Constants
BOT_NAME = 'Amara'
TIME_PRECISION = 2
STAT_PRECISION = 5
HOUR = 3600 # secs
OUTPUT_CHANNEL = 1001937090084356096
# general 975212076811296810
# testing 975212341232812052
# old-data-test-run-1 990408172344844288
# old-data-test-run-2 991027816705445888
# old-data-test-run-3 991038996572426302
# first-dataset-run-1 993535424138072064
# second-dataset-run-1 993543370016292945
# second-dataset-run-2 993919067473846282
# second-dataset-run-3 994288944323756073
# second-dataset-run-4 994769712221265990
# sand-testing 997238449654468789
# sand-2-classes-1 998745433813815366
# sand-2-classes-2 998758099781419100
# sand-2-classes-3 998767762191155260
# sand-2-classes-4 998772013323780187
# s2c-5 998785319258308728
# overkill-1 998793935872868402
# s2c-6 998997369729323058
# s2c-7 999035118238843012
# s2c-8 999111903869677649
# s2c-9 999143115514454066
# overkill-2 1001937090084356096

# Neural Network Constants
IMG_WIDTH = 400
IMG_HEIGHT = 400
TRAIN_DATA_DIR = 'third_dataset/train'
VALIDATION_DATA_DIR ='third_dataset/validation'
NB_TRAIN_SAMPLES = 75
NB_VALIDATION_SAMPLES = 50
EPOCHS = 10000
BATCH_SIZE = 20
MIN_LOSS_THRESHOLD = 0.005
MAX_ACC_THRESHOLD = 0.95
PATIENCE = 300
MIN_DELTA = 0.01
MONITOR = 'val_loss'
DROPOUT = 0.5

######################
### Nerual Network ###
######################

def build_model():
   
    if K.backend.image_data_format() == 'channels_first':
        in_shape = (3, IMG_WIDTH, IMG_HEIGHT)
    else:
        in_shape = (IMG_WIDTH, IMG_HEIGHT, 3)
       
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=in_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
   
    model.add(Conv2D(1024, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Conv2D(1024, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Conv2D(1024, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Conv2D(1024, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Conv2D(1024, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
   
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
   
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
   
    model.compile(loss = 'binary_crossentropy',
                  optimizer = 'rmsprop',
                  metrics = ['accuracy'])
    return model

def train_model(model):
    # Augmentation configuration for training
    train_datagen = ImageDataGenerator(
        rescale = 1./255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True
        )
    # Augmentation configuration for testing
    test_datagen = ImageDataGenerator(
        rescale = 1./255
        )
   
    train_generator = train_datagen.flow_from_directory(
            TRAIN_DATA_DIR,
            target_size = (IMG_WIDTH, IMG_HEIGHT),
            batch_size = BATCH_SIZE,
            class_mode = 'binary')
   
    validation_generator = test_datagen.flow_from_directory(
            VALIDATION_DATA_DIR,
            target_size = (IMG_WIDTH,IMG_HEIGHT),
            batch_size = BATCH_SIZE,
            class_mode = 'binary')
    
    # Added Early Stopping
    my_callback = [EarlyStopping(
        monitor = MONITOR,
        min_delta = MIN_DELTA,
        patience = PATIENCE,
        mode = 'auto',
        baseline = 1,
        restore_best_weights = True)]

    history_1 = model.fit_generator(
        train_generator,
        steps_per_epoch = NB_VALIDATION_SAMPLES // BATCH_SIZE,
        epochs = EPOCHS,
        validation_data = validation_generator,
        validation_steps = NB_VALIDATION_SAMPLES // BATCH_SIZE,
        callbacks = my_callback)
   
    return model, history_1
   
def save_model(model):
    json_model = model.to_json()
    with open('json_model.json', 'w') as json_file:
        json_file.write(json_model)
    model.save('saved_model.h5')
   
def run_model():
    built_model = None
    tf.keras.backend.clear_session()
    gc.collect()
    built_model = build_model()
    trained_model, history_1 = train_model(built_model)
    save_model(trained_model)

    # Prints out max validation accuracy and epoch that it occured at
    val_acc = history_1.history['val_acc']
    max_acc = max(val_acc)
    max_acc_epoch = val_acc.index(max_acc) + 1
    acc_str = "Max Accuracy at Epoch " + str(max_acc_epoch) + " ==> " + str(max_acc)
    print(acc_str)
    
    # Prints out min validation loss and epoch that it occured at
    val_loss = history_1.history['val_loss']
    min_loss = min(val_loss)
    min_loss_epoch = val_loss.index(min_loss) + 1 # plus 1 for display
    loss_str = "Min Loss at Epoch " + str(min_loss_epoch) + " ==> " + str(min_loss)
    print(loss_str)
    
#'''    
    #plt.plot(val_acc, 'r', 'val_acc')
    #plt.plot(val_loss, 'g', 'val_loss')
    #plt.plot(val_loss)
    #plt.title('Validation Accuracy and Loss Over Time')
    #plt.xlabel('Epoch')
    #plt.ylabel('Percent')
    #plt.legend(loc = 'upper left')
    #plt.show()
    
    num_epochs = len(val_loss)
    min_loss_acc = val_acc[min_loss_epoch - 1] # starts index at zero

    return min_loss, max_acc, num_epochs, min_loss_epoch, max_acc_epoch, min_loss_acc
#'''

#main()

#'''
#'''

######################
### Discord Code #####
######################

# nest_asyncio.apply() # allows infinite running event loop
load_dotenv() # loads in local .env file

client = discord.Client() # creates instance of client
TOKEN = os.getenv('DISCORD_TOKEN') # gets key from .env

######################
### Discord/Main #####
######################

@client.event
async def on_ready():
    """Start up routine and main function, lets developer and user know bot has connected"""
    
    # program start time
    start = time.time()
    start_var = time.ctime(start)
    
    # program start-up routine and info
    print('We have logged in as {0.user}'.format(client))
    print(f'Start time: {start_var}')
    await client.wait_until_ready()
    output = client.get_channel(OUTPUT_CHANNEL) # channel that will output
    start_str = f'***{BOT_NAME} online. Beginning neural network.***\n'
    start_str = start_str + f'Maximum Accuracy Threshold: **{MAX_ACC_THRESHOLD}**\n'
    start_str = start_str + f'Minimum Loss Threshold: **{MIN_LOSS_THRESHOLD}**\n'
    start_str = start_str + f'Program start time: {start_var}\n\n\n---'
    await output.send(start_str)

    # bulk of program run
    min_loss_iter = 100.0 # Just an initializing dummy value
    max_acc_iter = 0.0 # Just an initializing dummy value
    counter = 0
    while ((min_loss_iter >= MIN_LOSS_THRESHOLD) or (max_acc_iter <= MAX_ACC_THRESHOLD)):
        counter += 1

        # start of iteration
        iter_start = time.time()
        current_var = time.ctime(iter_start)
        iter_str = f'***Start of Iteration {counter}***\nStart time: {current_var}\n---'
        print(iter_str)
        await output.send(iter_str)
        
        # iteration run
        min_loss_iter, max_acc_iter, num_epochs, min_loss_epoch, max_acc_epoch, min_loss_acc = run_model() # runs model, stores variables

        # rounds numbers
        min_loss_str = round(min_loss_iter, STAT_PRECISION)
        max_acc_str = round(max_acc_iter, STAT_PRECISION)
        
        # calculates elapsed time
        iter_end = time.time()
        current_var = time.ctime(iter_end)
        iter_elapsed = round(iter_end - iter_start, TIME_PRECISION)

        # end of iteration
        iter_str = f'***End of Iteration {counter}***\nEnd time: {current_var}\n'
        iter_str = iter_str + f'Iteration Elapsed: {num_epochs} epochs, {iter_elapsed} seconds\n'
        iter_str = iter_str + f'Max Val Accuracy: Epoch {max_acc_epoch}, {max_acc_str}\n'
        iter_str = iter_str + f'Min Val Loss: Epoch {min_loss_epoch}, **{min_loss_str}**\n'
        iter_str = iter_str + f'Val Accuracy at Epoch {min_loss_epoch}: {min_loss_acc}\n\n\n---'
        print(iter_str)
        await output.send(iter_str)
    
    end = time.time()
    end_var = time.ctime(end)
    elapsed = round(end - start, TIME_PRECISION)
    elapsed_hrs = elapsed / HOUR
    end_str = f'End of model generation, best model achieved.\nEnd time: {end_var}\n'
    end_str = end_str + f'Program Elapsed: {elapsed} seconds ({elapsed_hrs} hours)'
    print(end_str)
    await output.send(end_str)

    # hotfix to prevent model generation from running again
    hours = 0
    while True:
        time.sleep(HOUR)
        hours += 1
        print(f'Program done for {hours} hour(s): {time.ctime()}')

client.run(TOKEN) # connects bot to server