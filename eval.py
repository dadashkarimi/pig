import os
import numpy as np
import pandas as pd
import nibabel as nib
import surfa as sf
import neurite as ne
from scipy import ndimage
from utils import find_bounding_box, extract_cube
from sklearn.metrics import jaccard_score
import itertools

# from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
from neurite_sandbox.tf.models import labels_to_labels
from neurite_sandbox.tf.utils.augment import add_outside_shapes
from neurite.tf.utils.augment import draw_perlin_full
from scipy.ndimage import distance_transform_edt
import scipy.ndimage as ndi
import tensorflow.keras.layers as KL
import voxelmorph as vxm
import argparse
from tensorflow.keras.callbacks import ReduceLROnPlateau
import pathlib
# import surfa as sf
import re
import json
from keras import backend as K
import param_3d
import data
import model_3d
from data_3d import *
import scipy.ndimage as ndimage

import nibabel as nib
from tqdm import tqdm
from tensorflow.keras.layers import Lambda

from utils import *
from help import *


# Define lists for k1 and k2
k1_list = [0, 1,2,3,4, 5, 6]  # You can change this based on your required values for k1
k2_list = [6,7,8, 9]  # Similarly, change k2 values as needed

two_steps=False
three_steps=True
olfactory=True
majority_vote = False
nima = False

def majority_vote_mask(mask1, mask2, mask3):
    """
    Performs majority voting on three binary masks.
    Voxels with values above 1 in the sum of the masks will be set to 1, others will be set to 0.
    
    Parameters:
    - mask1 (ndarray): First binary mask (same shape as the others).
    - mask2 (ndarray): Second binary mask (same shape as the others).
    - mask3 (ndarray): Third binary mask (same shape as the others).
    
    Returns:
    - final_mask (ndarray): A binary mask where the majority rule is applied.
    """
    # Sum the three masks element-wise
    combined_sum = mask1 + mask2 + mask3
    
    # Apply majority voting: voxels where sum > 1 are set to 1, else 0
    final_mask = (combined_sum > 1).astype(np.int32)
    
    return final_mask
    
def get_pig_model(k1,k2):
    epsilon =1e-7
    min_max_norm = Lambda(lambda x: (x - K.min(x)) / (K.max(x) - K.min(x)+ epsilon) * (1.0) )
    
    print("model is loading")
    en = [16 ,16 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64]
    de = [64 ,64 ,64 ,64, 64 ,64 ,64, 64, 64, 16 ,16 ,2]
    input_img = Input(shape=(param_3d.img_size_192,param_3d.img_size_192,param_3d.img_size_192, 1))
    unet_model = vxm.networks.Unet(inshape=(param_3d.img_size_192,param_3d.img_size_192,param_3d.img_size_192, 1), nb_features=(en, de),
                       nb_conv_per_level=2,
                       final_activation_function='softmax')
        
    latest_weight = max(glob.glob(os.path.join("models_gmm_"+str(k1)+"_"+str(k2), 'weights_epoch_*.h5')), key=os.path.getctime, default=None)
    print(latest_weight)
    generated_img_norm = min_max_norm(input_img)
    segmentation = unet_model(generated_img_norm)
    combined_model = Model(inputs=input_img, outputs=segmentation)
    combined_model.load_weights(latest_weight)
    return combined_model

def get_pig_model_128():
    epsilon =1e-7
    min_max_norm = Lambda(lambda x: (x - K.min(x)) / (K.max(x) - K.min(x)+ epsilon) * (1.0) )
    
    print("model is loading")
    en = [16 ,16 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64]
    de = [64 ,64 ,64 ,64, 64 ,64 ,64, 64, 64, 16 ,16 ,2]
    input_img = Input(shape=(param_3d.img_size_128,param_3d.img_size_128,param_3d.img_size_128, 1))
    unet_model = vxm.networks.Unet(inshape=(param_3d.img_size_128,param_3d.img_size_128,param_3d.img_size_128, 1), nb_features=(en, de),
                       nb_conv_per_level=2,
                       final_activation_function='softmax')
        
    latest_weight = max(glob.glob(os.path.join("models_gmm_4_8_128", 'weights_epoch_*.h5')), key=os.path.getctime, default=None)
    print(latest_weight)
    generated_img_norm = min_max_norm(input_img)
    segmentation = unet_model(generated_img_norm)
    combined_model = Model(inputs=input_img, outputs=segmentation)
    combined_model.load_weights(latest_weight)
    return combined_model

def get_pig_model_96():
    epsilon =1e-7
    min_max_norm = Lambda(lambda x: (x - K.min(x)) / (K.max(x) - K.min(x)+ epsilon) * (1.0) )
    
    print("model is loading")
    en = [16 ,16 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64]
    de = [64 ,64 ,64 ,64, 64 ,64 ,64, 64, 64, 16 ,16 ,2]
    input_img = Input(shape=(param_3d.img_size_96,param_3d.img_size_96,param_3d.img_size_96, 1))
    unet_model = vxm.networks.Unet(inshape=(param_3d.img_size_96,param_3d.img_size_96,param_3d.img_size_96, 1), nb_features=(en, de),
                       nb_conv_per_level=2,
                       final_activation_function='softmax')
        
    latest_weight = max(glob.glob(os.path.join("models_gmm_4_6_96", 'weights_epoch_*.h5')), key=os.path.getctime, default=None)
    print(latest_weight)
    generated_img_norm = min_max_norm(input_img)
    segmentation = unet_model(generated_img_norm)
    combined_model = Model(inputs=input_img, outputs=segmentation)
    combined_model.load_weights(latest_weight)
    return combined_model

def get_pig_model_olfactory_96():
    epsilon =1e-7
    min_max_norm = Lambda(lambda x: (x - K.min(x)) / (K.max(x) - K.min(x)+ epsilon) * (1.0) )
    
    print("model is loading")
    en = [16 ,16 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64]
    de = [64 ,64 ,64 ,64, 64 ,64 ,64, 64, 64, 16 ,16 ,2]
    input_img = Input(shape=(param_3d.img_size_96,param_3d.img_size_96,param_3d.img_size_96, 1))
    unet_model = vxm.networks.Unet(inshape=(param_3d.img_size_96,param_3d.img_size_96,param_3d.img_size_96, 1), nb_features=(en, de),
                       nb_conv_per_level=2,
                       final_activation_function='softmax')
        
    latest_weight = max(glob.glob(os.path.join("models_gmm_4_6_96_olfactory", 'weights_epoch_*.h5')), key=os.path.getctime, default=None)
    print(latest_weight)
    generated_img_norm = min_max_norm(input_img)
    segmentation = unet_model(generated_img_norm)
    combined_model = Model(inputs=input_img, outputs=segmentation)
    combined_model.load_weights(latest_weight)
    return combined_model
    
# Function to refine predictions
def refine_prediction(crop_img, mask, model, model_128, folder, new_image_size=(192, 192, 192), margin=0, cube_size=128):
    folder_path = os.path.join("results", folder)
    os.makedirs(folder_path, exist_ok=True)
    nib.save(nib.Nifti1Image(crop_img, np.eye(4)), os.path.join(folder_path, 'image.nii.gz'))

    mask.data[mask.data != 0] = 1
    nib.save(nib.Nifti1Image(mask.astype(np.int32), np.eye(4)), os.path.join(folder_path, 'mask.nii.gz'))

    ms = np.mean(np.column_stack(np.nonzero(mask)), axis=0).astype(int)

    # Step 1: Initial Prediction
    prediction_one_hot = model.predict(crop_img[None, ...], verbose=0)
    initial_prediction = np.argmax(prediction_one_hot, axis=-1)[0]
    labeled, num_components = ndimage.label(initial_prediction > 0)
    largest_mask = labeled == np.argmax(ndimage.sum(initial_prediction > 0, labeled, range(num_components + 1)))
    initial_prediction = ndi.binary_fill_holes(largest_mask)
    nib.save(nib.Nifti1Image(initial_prediction.astype(np.int32), np.eye(4)), os.path.join(folder_path, 'initial_prediction.nii.gz'))

    # Step 2: Use find_bounding_box function to get the bounding box
    x1, y1, z1, x2, y2, z2 = find_bounding_box(initial_prediction, cube_size=cube_size)
    cube = extract_cube(crop_img, x1, y1, z1, x2, y2, z2, cube_size=128)

    pred_192 = np.zeros((192, 192, 192))
    ms = np.mean(np.column_stack(np.nonzero(mask)), axis=0).astype(int)

    # Step 3: Re-run the Model with the cropped image
    prediction_cropped_one_hot = model_128.predict(cube[None, ...], verbose=0)
    final_prediction = np.argmax(prediction_cropped_one_hot, axis=-1)[0]
    pred_192[x1:x2, y1:y2, z1:z2] = final_prediction
    pred_192[pred_192 == 1] = 1
    
    labeled, num_components = ndimage.label(pred_192 > 0)
    largest_mask = labeled == np.argmax(ndimage.sum(pred_192 > 0, labeled, range(num_components + 1)))
    largest_mask = ndi.binary_fill_holes(largest_mask)
    pred_192 = largest_mask

    # Resize the final prediction to match the input size
    nib.save(nib.Nifti1Image(pred_192.astype(np.int32), np.eye(4)), os.path.join(folder_path, 'second_prediction.nii.gz'))
    return pred_192



def refine_prediction2(crop_img, mask, model, model_128,model_96, folder, new_image_size=(192, 192, 192), margin=0, cube_size=128):
    """
    Refines the segmentation prediction in two steps:
    1. Makes an initial prediction.
    2. Crops the image based on the prediction and runs the model again.
    
    Parameters:
    - crop_img (ndarray): The input image for prediction.
    - mask (ndarray): The binary mask.
    - model: The trained segmentation model.
    - new_image_size (tuple): The new voxel size for resizing (default is (192, 192, 192)).
    - margin (int): The margin to add around the bounding box (default is 10).
    - cube_size (int): The size of the bounding cube (default is 32).
    
    Returns:
    - final_prediction_resized (ndarray): The final refined prediction, resized to match the original input size.
    """
    folder_path = os.path.join("results", folder)
    os.makedirs(folder_path, exist_ok=True)
    nib.save(nib.Nifti1Image(crop_img, np.eye(4)), os.path.join(folder_path, 'image.nii.gz'))

    # Step 1: Initial Prediction
    # Binarize the mask
    mask.data[mask.data != 0] = 1
    nib.save(nib.Nifti1Image(mask.astype(np.int32), np.eye(4)), os.path.join(folder_path, 'mask2.nii.gz'))

    # Compute mask center (using the provided find_bounding_box function)
    ms = np.mean(np.column_stack(np.nonzero(mask)), axis=0).astype(int)
    print(crop_img.shape)
    
    # Make an initial prediction
    prediction_one_hot = model.predict(crop_img[None, ...], verbose=0)
    initial_prediction = np.argmax(prediction_one_hot, axis=-1)[0]
    ne.plot.volume3D(crop_img, slice_nos=ms)
    print("Initial Prediction Result:")

    labeled, num_components = ndimage.label(initial_prediction > 0)
    largest_mask = labeled == np.argmax(ndimage.sum(initial_prediction > 0, labeled, range(num_components + 1)))
    initial_prediction = ndi.binary_fill_holes(largest_mask)
    initial_prediction = (initial_prediction > 0).astype(np.int32)
    nib.save(nib.Nifti1Image(initial_prediction.astype(np.int32), np.eye(4)), os.path.join(folder_path, 'initial_prediction.nii.gz'))

    ne.plot.volume3D(initial_prediction, slice_nos=ms)
    print("first step: ",my_hard_dice(mask.data, initial_prediction))

    # Step 2: Use find_bounding_box function to get the bounding box
    x1, y1, z1, x2, y2, z2 = find_bounding_box(initial_prediction, cube_size=cube_size)
    cube = extract_cube(crop_img, x1, y1, z1, x2, y2, z2, cube_size=128)


    pred_192_1 = np.zeros((192,192,192))
    pred_192_2 = np.zeros((192,192,192))
    pred = np.zeros((192,192,192))

    ms = np.mean(np.column_stack(np.nonzero(mask)), axis=0).astype(int)
    ne.plot.volume3D(cube, slice_nos=ms)

    # Step 3: Re-run the Model with the cropped image
    prediction_cropped_one_hot = model_128.predict(cube[None, ...], verbose=0)
    final_prediction = np.argmax(prediction_cropped_one_hot, axis=-1)[0]
    pred_192_1[x1:x2, y1:y2, z1:z2] = final_prediction
    pred_192_1[pred_192_1==1]=1
    
    labeled, num_components = ndimage.label(pred_192_1 > 0)
    largest_mask = labeled == np.argmax(ndimage.sum(pred_192_1 > 0, labeled, range(num_components + 1)))
    largest_mask = ndi.binary_fill_holes(largest_mask)
    pred_192_1 = largest_mask
    pred_192_1 = (pred_192_1 > 0).astype(np.int32)
    ne.plot.volume3D(pred_192_1, slice_nos=ms)
    print("second step: ",my_hard_dice(mask.data, pred_192_1))
    pred = (pred_192_1 > 0).astype(np.int32)
    pred[pred != 0] = 2

    nib.save(nib.Nifti1Image(pred.astype(np.int32), np.eye(4)), os.path.join(folder_path, 'second_prediction.nii.gz'))

    pred = np.zeros((192,192,192))
    x1, y1, z1, x2, y2, z2 = find_bounding_box(pred_192_1, cube_size=96)
    cube = extract_cube(crop_img, x1, y1, z1, x2, y2, z2, cube_size=96)

    prediction_cropped_one_hot = model_96.predict(cube[None, ...], verbose=0)
    final_prediction = np.argmax(prediction_cropped_one_hot, axis=-1)[0]
    pred_192_2[x1:x2, y1:y2, z1:z2] = final_prediction
    pred_192_2[pred_192_2==1]=1
    print("@@@@@",np.max(pred_192_2))
    labeled, num_components = ndimage.label(pred_192_2 > 0)
    largest_mask = labeled == np.argmax(ndimage.sum(pred_192_2 > 0, labeled, range(num_components + 1)))
    largest_mask = ndi.binary_fill_holes(largest_mask)
    pred_192_2 = largest_mask
    pred_192_2 = (pred_192_2 > 0).astype(np.int32)
    ne.plot.volume3D(pred_192_2, slice_nos=ms)
    # print("second step: ",my_hard_dice(mask.data, pred_192_2))


    if majority_vote:
        pred_192 = majority_vote_olfactory_mask(initial_prediction,pred_192_1,pred_192_2)
    else:
        pred_192 = pred_192_2
    # pred_192 = majority_vote_mask(initial_prediction,pred_192_1,pred_192_2)
    ne.plot.volume3D(pred_192, slice_nos=ms)
    print("third step: ",my_hard_dice(mask.data, pred_192))
    pred = (pred_192 > 0).astype(np.int32)
    pred[pred != 0] = 3 
    # Step 4: Resize the final prediction to the original crop_img size
    # final_prediction_resized = np.resize(final_prediction, (192, 192, 192))
    nib.save(nib.Nifti1Image(pred.astype(np.int32), np.eye(4)), os.path.join(folder_path, 'third_prediction.nii.gz'))
    return initial_prediction, pred_192_1, pred_192_2


def majority_vote_olfactory_mask(mask1, mask2, mask3):
    """
    Performs majority voting on three binary masks, but gives special preference to 
    mask3 (capturing olfactory areas) if a connected component is found in mask3 but not in mask1 or mask2.
    
    Parameters:
    - mask1 (ndarray): First binary mask (same shape as the others).
    - mask2 (ndarray): Second binary mask (same shape as the others).
    - mask3 (ndarray): Third binary mask (same shape as the others, capturing olfactory areas).
    
    Returns:
    - final_mask (ndarray): A binary mask where the majority rule is applied, but 
                              connected components in mask3 are prioritized.
    """
    # Sum the three masks element-wise for majority voting
    combined_sum = mask1 + mask2 + mask3
    
    # Apply majority voting: voxels where sum > 1 are set to 1, else 0
    final_mask = (combined_sum > 1).astype(np.int32)

    # Identify connected components in mask3 that are not in mask1 and mask2
    mask3_only = mask3 & (~mask1) & (~mask2)
    
    if np.any(mask3_only):  # If there are any components in mask3 only
        # Label connected components in mask3_only (where mask3 is 1 and mask1, mask2 are 0)
        labeled_components, num_components = label(mask3_only)
        
        # For each connected component in mask3_only, we ensure it's included in the final mask
        for i in range(1, num_components + 1):
            component = (labeled_components == i)
            final_mask[component] = 1  # Ensure all voxels in this component are set to 1 in the final mask
    
    return final_mask


def first_prediction(crop_img, mask, model, model_128, folder, new_image_size=(192, 192, 192), margin=0, cube_size=128):
    """
    Refines the segmentation prediction in two steps:
    1. Makes an initial prediction.
    2. Crops the image based on the prediction and runs the model again.
    
    Parameters:
    - crop_img (ndarray): The input image for prediction.
    - mask (ndarray): The binary mask.
    - model: The trained segmentation model.
    - new_image_size (tuple): The new voxel size for resizing (default is (192, 192, 192)).
    - margin (int): The margin to add around the bounding box (default is 10).
    - cube_size (int): The size of the bounding cube (default is 32).
    
    Returns:
    - final_prediction_resized (ndarray): The final refined prediction, resized to match the original input size.
    """
    folder_path = os.path.join("results", folder)
    os.makedirs(folder_path, exist_ok=True)
    nib.save(nib.Nifti1Image(crop_img, np.eye(4)), os.path.join(folder_path, 'image.nii.gz'))

    # Step 1: Initial Prediction
    # Binarize the mask
    mask.data[mask.data != 0] = 1
    # nib.save(nib.Nifti1Image(mask.astype(np.int32), np.eye(4)), os.path.join(folder_path, 'mask.nii.gz'))

    # Compute mask center (using the provided find_bounding_box function)
    ms = np.mean(np.column_stack(np.nonzero(mask)), axis=0).astype(int)
    print(crop_img.shape)
    
    # Make an initial prediction
    prediction_one_hot = model.predict(crop_img[None, ...], verbose=0)
    initial_prediction = np.argmax(prediction_one_hot, axis=-1)[0]
    print("Initial Prediction Result:")

    # labeled, num_components = ndimage.label(initial_prediction > 0)
    # largest_mask = labeled == np.argmax(ndimage.sum(initial_prediction > 0, labeled, range(num_components + 1)))
    # initial_prediction = ndi.binary_fill_holes(largest_mask)
    nib.save(nib.Nifti1Image(initial_prediction.astype(np.int32), np.eye(4)), os.path.join(folder_path, 'initial_prediction.nii.gz'))


    return initial_prediction

# Load subfolders in the validation folder
validation_folder_path = "/cubic/projects/Pig_TBI/JohnWolf/Protocols/T1_mask"

if nima:
    validation_folder_path = "/gpfs/fs001/cbica/home/broodman/Pig_project"
subfolders = [f.name for f in os.scandir(validation_folder_path) if f.is_dir()]

# Prepare to store results in a CSV file
results = []
dice_list = []
icc_list = []

def icc(mask1, mask2):
    # Flatten masks
    m1 = mask1.flatten()
    m2 = mask2.flatten()
    
    # Calculate mean-centered arrays
    mean_m1 = np.mean(m1)
    mean_m2 = np.mean(m2)
    
    ss_total = np.sum((m1 - mean_m1)**2 + (m2 - mean_m2)**2)
    ss_residual = np.sum((m1 - m2)**2)
    
    if ss_total == 0:
        return 1.0  # Identical masks
    return 1 - (ss_residual / ss_total)

def compare_masks(mask1_path, mask2_path):
    # Load NIfTI files
    m1 = nib.load(mask1_path).get_fdata()
    m2 = nib.load(mask2_path).get_fdata()

    # Optional: Binarize if needed
    m1 = (m1 > 0).astype(np.uint8)
    m2 = (m2 > 0).astype(np.uint8)

    dice = dice_coefficient(m1, m2)
    icc_val = icc(m1, m2)

    return dice, icc_val

# Loop through all combinations of k1 and k2
for k1, k2 in itertools.product(k1_list, k2_list):
    # Get the combined models for each k1, k2 combination
    combined_model = get_pig_model(k1, k2)
    combined_model_128 = get_pig_model_128()

    if olfactory:
        combined_model_96 = get_pig_model_olfactory_96()
    else:
        combined_model_96 = get_pig_model_96()


    # Store Dice coefficients for each combination
    dice_scores = []
    for folder in subfolders:
        folder_path = os.path.join(validation_folder_path, folder)
        folder_name = os.path.basename(folder_path)
        
        mask1_path = os.path.join(folder_path, "mask1.nii.gz")
        mask2_path = os.path.join(folder_path, "mask2.nii.gz")
    
        if not os.path.isfile(mask1_path) or not os.path.isfile(mask2_path):
            print(f"Skipping {folder} — mask not found")
            continue
    
        dice, icc_val = compare_masks(mask1_path, mask2_path)
        dice_list.append(dice)
        icc_list.append(icc_val)
    
        print(f"{folder}: Dice = {dice:.4f}, ICC = {icc_val:.4f}")

if dice_list and icc_list:
    avg_dice = sum(dice_list) / len(dice_list)
    avg_icc = sum(icc_list) / len(icc_list)
    print(f"\nOverall Dice: {avg_dice:.4f}")
    print(f"Overall ICC: {avg_icc:.4f}")
else:
    print("No valid mask pairs found for comparison.")
    
        # if not nima:
        #     filename = os.path.join(folder_path, f"{folder_name}_T1.nii.gz")
        #     mask_filename = os.path.join(folder_path, f"{folder_name}_T1_mask.nii.gz")
        # else:
        #     filename = os.path.join(folder_path, f"image.nii.gz")
        #     mask_filename = os.path.join(folder_path, f"mask.nii.gz")            

        # if "JAW-106_6month" in filename:
        #     print(f"Skipping {filename} due to no mask")
        #     continue
        
        # # Load mask
        # if os.path.isfile(mask_filename):
        #     mask = sf.load_volume(mask_filename).resize([1, 1, 1], method="linear")
        #     mask = mask.resize([1, 1, 1]).reshape([192, 192, 192, 1])
        # else:
        #     mask = sf.Volume(np.ones((192, 192, 192)))
        
        # # Load image
        # if not os.path.isfile(filename):
        #     continue
        
        # image = sf.load_volume(filename)
        # crop_img = image.resize([1, 1, 1], method="linear").reshape([192, 192, 192, 1])

        # # Compute mask center
        # ms = np.mean(np.column_stack(np.nonzero(mask)), axis=0).astype(int)

        # # Perform prediction
        # if three_steps:
        #     # prediction = refine_prediction2(crop_img, mask, combined_model,combined_model_128,combined_model_96, folder, new_image_size=(192, 192, 192))
        #     initial_pred, second_pred, third_pred = refine_prediction2(crop_img, mask, combined_model,combined_model_128,combined_model_96, folder, new_image_size=(192, 192, 192))

        # elif two_steps:
        #     prediction = refine_prediction(crop_img, mask, combined_model, combined_model_128, folder)
        # else:
        #     prediction = first_prediction(crop_img, mask, combined_model, combined_model_128, folder)

        # # Compute Dice coefficient
        # mask_flat = mask.data.flatten()
        
        # initial_flat = initial_pred.flatten()
        # second_flat = second_pred.flatten()
        # third_flat = third_pred.flatten()
        
        # dice_initial = 2 * np.sum(mask_flat * initial_flat) / (np.sum(mask_flat) + np.sum(initial_flat))
        # dice_second = 2 * np.sum(mask_flat * second_flat) / (np.sum(mask_flat) + np.sum(second_flat))
        # dice_third = 2 * np.sum(mask_flat * third_flat) / (np.sum(mask_flat) + np.sum(third_flat))

        # if np.sum(mask.data) < 1000:
        #     continue
        # dice_scores.append((dice_initial, dice_second, dice_third))


        # print(f"Dice coefficient for {folder_name}: {dice_score:.4f}")

    # Calculate the overall Dice coefficient for this k1, k2 combination
    # if dice_scores:
    #     initial_mean = np.mean([d[0] for d in dice_scores])
    #     second_mean = np.mean([d[1] for d in dice_scores])
    #     third_mean = np.mean([d[2] for d in dice_scores])
    # else:
    #     initial_mean = second_mean = third_mean = 0
    
    # results.append({
    #     'k1': k1,
    #     'k2': k2,
    #     'dice_initial': initial_mean,
    #     'dice_second': second_mean,
    #     'dice_third': third_mean
    # })

    # overall_dice = np.mean(dice_scores) if dice_scores else 0
    # print(f"Overall Dice coefficient for k1={k1}, k2={k2}: {overall_dice:.4f}")

    # # Save the results into a list to export later
    # results.append({'k1': k1, 'k2': k2, 'dice_score': overall_dice})

# Convert results to DataFrame and save as CSV
df = pd.DataFrame(results)
if three_steps:
    if olfactory:
        df.to_csv('dice_scores_three_steps_olfactory_results.csv', index=False)
    else:
        df.to_csv('dice_scores_three_steps_results.csv', index=False)
elif two_steps:
    df.to_csv('dice_scores_two_steps_results.csv', index=False)
else:
    df.to_csv('dice_scores_one_step_results.csv', index=False)

print("Results saved!'.")

