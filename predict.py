# Importing all the required packages


import keras.utils as image
import os,time,sys
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # Comment for running in gpu
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from keras.layers import Input
from utils_new import *
from keras.models import load_model
import keras.backend as K
from keras.models import Model

# Same loss function as defined in the main function, a weighted categorical cross entropy

def weighted_ce(weights):
    def custom_ce(y_actual,y_predicted):
        weights2 = K.constant(weights, dtype=K.floatx())
        loss_value1 = -K.mean( K.sum(weights2 * y_actual * K.log( y_predicted + K.epsilon()), axis=-1))
        return loss_value1
    return custom_ce

weights = [1.1997 , 13.3052 , 29.424 , 17.429] # BG,T,NT,Bor

loss_fn = weighted_ce(weights)

# Path to the working directories

cwd = os.getcwd()
input_folder_name = "testfolder"    # Name of the prediction folder where the images to be tested are present
predict_folder = os.path.join(cwd,  input_folder_name)             
output_folder = os.path.join(cwd, input_folder_name,'_results' ) 
model_name = "resunet_model.h5"

# Creating an output directory before hand

if os.path.isdir(output_folder) is not True:
    os.mkdir(output_folder)


# Note the 

val  = validation2(predict_folder)



# Loading model and Weights
model = load_model(model_name, custom_objects={'fscore': fscore , 'custom_ce' : loss_fn})
model.load_weights("C:\\Users\\ssrih\\2nd Sem\\Proj\\thestart\\weights_new\\proj_weights_new2\\weights.048.hdf5") 
# Load an appropraite model seeing at the training graph, can choose one with high fscore or with less loss. I have chosen weight 48 from a particular location

start = time.time()

z= model.predict(val_x, batch_size=1, verbose=0, steps=None )
z = colour_code(z,label_values)
val_y = colour_code(val_y,label_values)

for i in range(len(z)):
    a,c = next(val)
    print(c)
    z= model.predict(a[np.newaxis,...], batch_size=1, verbose=0, steps=None)
    z = colour_code(z,label_values)
    out = z[0]

    print(type(out))
    print(np.shape(out))
    #a = inputname[i]
    image_name = os.path.basename(c)
    print(image_name)
    image_name = image_name[:image_name.index('.')]

    #inp = val_y

    image.save_img('%s/%s_pred.png'%(output_folder,image_name), out)
    # image is predicted and saved in the output folder
    
end = time.time()
print(f"time taken for { len(z) } images is {end - start} seconds ")
