# Importing all the required packages

import os,cv2
import random
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from glob import glob
import tensorflow as tf
import keras.backend as K

# The Color Codedd Groundtruth Values

class3 = [0,0,255] #border
class2 = [0,255,0] #nonotumor
class1 = [255,0,0] #tumor
class0 = [0,0,0] #background which is the final class

label_values = [class0]  + [class1] + [class2] + [class3] 
num_classes = len(label_values)

# One_Hot encoding function

def one_hot(mask):
    semantic_map = []
    for colour in label_values:
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis = -1)
        semantic_map.append(class_map)
    semantic_map = (np.stack(semantic_map, axis=-1)).astype(float)
    return semantic_map

# This function basically normalizes the input image and then one_hot_encodes it

def adjustData(img,mask):
    img = img / 255
    mask = one_hot(mask)
    return (img,mask)

def colour_code(image, label_values):
    x = np.argmax(image, axis = -1)
    colour_codes = np.array(label_values)
    x = colour_codes[x.astype(int)]
    return x

def fscore(y_true,y_pred):
    y_true = y_true[:,:,:,1:]
    y_pred = y_pred[:,:,:,1:]
    y_true = K.round(K.flatten(y_true))
    y_pred = K.round(K.flatten(y_pred)) 
    true_positives = K.sum(y_true * y_pred)
    predicted_positives = K.sum(y_pred)
    possible_positives = K.sum(y_true)
    precision = true_positives / (predicted_positives+ K.epsilon())   
    recall = true_positives / (possible_positives+ K.epsilon())  
    f_score1 = 2*precision*recall/(precision+recall+ K.epsilon())
    return f_score1    


def num_of_images(path):
    image_datagen = ImageDataGenerator()
    image_generator = image_datagen.flow_from_directory(
        path,
        classes = ["gt"],
        class_mode = None)
    return image_generator.samples

def dataGenerator(batch_size,path,aug_dict,size,seed=1):
    image_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        path,
        target_size=(size, size),
        classes = ["input"],
        class_mode = None,
        batch_size = batch_size,
        seed = seed)
    mask_generator = image_datagen.flow_from_directory(
        path,
        target_size=(size, size),
        classes = ["gt"],
        class_mode = None,
        batch_size = batch_size,
        seed = seed)
    data_generator = zip(image_generator, mask_generator)
    for (image,mask) in data_generator:
        image,mask = adjustData(image,mask)
        yield (image,mask)

def valGenerator(batch_size,path,size,seed=1):
    image_datagen = ImageDataGenerator()
    image_generator = image_datagen.flow_from_directory(
        path,
        target_size=(size, size),
        classes = ["input"],
        class_mode = None,
        batch_size = batch_size,
        seed = seed)
    mask_generator = image_datagen.flow_from_directory(
        path,
        target_size=(size, size),
        classes = ["gt"],
        class_mode = None,
        batch_size = batch_size,
        seed = seed)
    data_generator = zip(image_generator, mask_generator)
    for (image,mask) in data_generator:
        image,mask = adjustData(image,mask)
        yield (image,mask)


# Mask or Grounfruth name should end with '_gt' following the image name

def validation(image_path,mask_path,image_prefix = ".png",mask_prefix = ".png"):
    image_name_arr = glob(os.path.join(image_path,"*%s"%image_prefix))
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_arr):

        img = cv2.cvtColor(cv2.imread(item),cv2.COLOR_BGR2RGB)
        mask= cv2.cvtColor(cv2.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix)),cv2.COLOR_BGR2RGB)
        img = img[:,:,:3]
        mask = mask[:,:,:3] if mask.ndim==3 else np.repeat(mask[:,:,np.newaxis],3,axis=-1)
        img,mask = adjustData(img,mask)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr, image_name_arr

# This validation2 would be used in the predict.py

def validation2(image_path,image_prefix = ".png"):
    image_name_arr = glob(os.path.join(image_path,"*%s"%image_prefix))
    image_arr = []
    for index,item in enumerate(image_name_arr):

        img = cv2.cvtColor(cv2.imread(item),cv2.COLOR_BGR2RGB)
        #mask= cv2.cvtColor(cv2.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix)),cv2.COLOR_BGR2RGB)
        img = img[:,:,:3]
        #mask = mask[:,:,:3] if mask.ndim==3 else np.repeat(mask[:,:,np.newaxis],3,axis=-1)
        #img,mask = adjustData(img,mask)
        img = img/255
        yield img , item   
