# Importing all the required packages

from __future__ import print_function
import numpy as np
from keras.models import load_model
from utils_new import *
# should have utils_new.py code in the same directory as this main.py
from keras.models import Model
from keras.callbacks import ModelCheckpoint, CSVLogger
import keras.backend as K
import os,cv2, sys, math , csv, time
from m_resunet import ResUnetPlusPlus
# should have m_resunet.py code in the same directory as this main.py

# Input dictionary for datagenerator method

data_gen_train = dict(rotation_range=10,
                        horizontal_flip=True,
                            width_shift_range=5,
                                height_shift_range=5,
	   	                            vertical_flip=True,
                                        fill_mode='nearest')


# Path to the working directories

cwd = os.getcwd() # Getting the current working directory
val_name = os.path.join(cwd  ,'\val_directory')
# here the name of my validation directory is 'val_directory'
train_name = os.path.join(cwd , '\train_directory')
# here the name of my train directory is 'train_directory'

csv_name = os.path.join(cwd , '\results\csvlog.csv')
# cvs log path for saving the training procedure. I am saving it in a file called results inside my current working directory with the name 'csvlog.csv'    
weights_folder = os.path.join(cwd , '\weights_directory')
# weights of our models are also saved in case we want to pick out a particular weight for prediction
# of even training stopped due to unforseen events, we can continue our training by loading the weight at which the training was particularly stopped at.
model_name =  "resunet_model.h5" 
# this is the name of the model we are making use of

print(train_name)

# Directories and files 
val_name = "S:\2nd sem\Proj\thestart\val_new1"
train_name = "S:\\2nd Sem\\Proj\\thestart\\train_new1"
csv_name = "S:\\2nd Sem\\Proj\\thestart\\results_new\\training_resunet_neww.csv"      
weights_folder = "S:\\2nd Sem\\Proj\\thestart\\weights_new\\rou"
model_name =  "resunet_model.h5" 


print(train_name)
# Hyperparameters for training

batch_size = 4 # No. of images in a batch
size = 256 # I have resized the images to this size
weights = [1.1997 , 13.3052 , 29.424 , 17.429] # BG,T,NT,Bor
# These weights are calculated for doing implementing the loss function 'Weighted Categorical Crossentropy'
LR = 1e-5
# Learning Rate
NO_CLASSES = 4
# The four classes which I have chose to work on were Background, Non-Tumor Cells, Tumor Cells and Border of these Tumor Cells
num_train = num_of_images(train_name)
num_val = num_of_images(val_name)
# Getting the number of images from the respective directory
train = dataGenerator(batch_size, train_name, data_gen_train, size)
val = valGenerator(batch_size, val_name, size)
# Generate batches of tensor image data with real-time data augmentation as per the written specifications

# Function for implementing weighted categorical crossentropy

def weighted_ce(weights):
    def custom_ce(y_actual,y_predicted):
        weights2 = K.constant(weights, dtype=K.floatx())
        loss_value1 = -K.mean( K.sum(weights2 * y_actual * K.log( y_predicted + K.epsilon()), axis=-1))
        return loss_value1
    return custom_ce

# Initializing loss function and optimizer for training

loss_fn = weighted_ce(weights)
opt = tf.keras.optimizers.RMSprop(lr=LR)

# Getting the number of training steps

train_steps = np.ceil(num_train / batch_size)
val_steps = np.ceil(num_val / batch_size)

# Creating a weights folder if not created before hand.

if os.path.isdir(weights_folder) is not True:
    os.makedirs(weights_folder)

# initializing CSV Logger

csv_logger = CSVLogger(csv_name, append=True)
checkpointer = ModelCheckpoint(filepath='%s/weights.{epoch:03d}.hdf5'%weights_folder,save_weights_only= True)
start = time.time()

# Training Model Initialization

#checkpoint_path = model_name
checkpoint_path = None
if checkpoint_path is not None:
    # if is responsible for trainning from a particular checkpoint it will run if the varible checkpoint_path is set to model_name 
    # if it is None it will go to the else part 
    model = load_model(checkpoint_path, custom_objects={'fscore': fscore , 'custom_ce' : loss_fn})
    model.load_weights("%s/weights.050.hdf5"%weights_folder)                                  
    initial_epoch = 50
    # To continue from a particular epoch in the example here I am implying that I want to train from weight 50 saved last and continue training from there.
    # Note : change the initial_epoch variable also accordingly
else:
    # else part is for training from the beginning 
    arch = ResUnetPlusPlus(input_size=None,no_classes=NO_CLASSES)
    model = arch.build_model()
    initial_epoch = 0

# I have set my final epoch to 100

final_epoch = 100    
print(model.summary())

# Model Compilation and Training

model.compile(optimizer = opt, loss = loss_fn, metrics =['accuracy',fscore])
model.fit(train, steps_per_epoch=train_steps, epochs=final_epoch, verbose=1, validation_data=val, 
             validation_steps=val_steps,  callbacks=[csv_logger,checkpointer],shuffle=True, initial_epoch=initial_epoch)
model.save(model_name)
end = time.time()
print(f"time taken for training {final_epoch - initial_epoch} is {end - start} seconds")
