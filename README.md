# histopathologial_image_segmentation
Histopathological Image Segmentation.<br>
<br>
**Please refer the codes, I have elaborated whats going on in comments.** <br>
**First run main.py the system is trained and the weights are saved.** <br>
**Then run predict.py, this one predicts for test outputs.** <br>

**Note:** The train and val directories should be in this format, (ground truth images have suffix _gt)<br>
**train_directory**<br>
---------**input**<br>
--------------image1.png<br>
--------------image2.png<br>
(and so on..)<br>
---------**gt**<br>
--------------image1_gt.png<br>
--------------image2_gt.png<br>
(and so on..)<br>

**val_directory**<br>
---------**input**<br>
--------------image30.png<br>
--------------image31png<br>
(and so on..)<br>
---------**gt**<br>
--------------image30_gt.png<br>
--------------image31_gt.png<br>
(and so on..)<br>


I have employed ResUNet for segmentation purpose of given Histopathological Image.
The loss function used is Weighted Categorical Cross entropy. The reason to include weights is that, if 
we train without including weights as the background pixels are more it is easily getting classified, but 
the main part of this challenge is to segment tumor and nontumor cells used for Tumor Cellularity 
Calculation. <br>
<br>
The Weights are calculated as follows:<br>
N1 = Total number of tumor cell pixels in the training data <br>
N2 = Total number of nontumor cell pixels in the training data<br>
N3 = Total number of cell boundary pixels in the training data<br>
N4 = Total number of Background pixels in the training data<br>
N = Total pixels involved in the whole training data<br>
Weight for Tumor Segmentation = N/N1<br>
Weight for Non Tumor Segmentation = N/N2<br>
Weight for Cell Boundary Segmentation = N/N3<br>
Weight for Background Segmentation = N/N4<br>

After Including weights, the network is trained. Augmentation of the data is also included while training 
includes different augmentations like rotations and inverting for regularization purposes. The 
optimization algorithm used is RMSProps which stands for Root Mean Squared Propagation.
<br>
After training I predicted the output for test data with best accuracy and least error seen while 
training. The predicted image is then subjected to morphological opening operation in order to reduce 
noisy output dots in the predicted image which can prove to cause hindrance in calculating Tumor 
Cellularity Value.
For classes of segmentation, I have tumor class, non-tumor class, cell boundary class and background. 
Hence in total we have 4 classes with colors red, green, blue and black respectively
