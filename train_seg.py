# from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
from neurite_sandbox.tf.models import labels_to_labels
from neurite_sandbox.tf.utils.augment import add_outside_shapes
from neurite.tf.utils.augment import draw_perlin_full

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

from utils import *
from help import *


import pathlib
import json
import nibabel as nib
import numpy as np
import tensorflow as tf
from utils import *
import param_3d
import scipy.ndimage as ndimage
import warnings
from keras import backend as K
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
import sys
from tensorflow.keras.models import load_model
import argparse
import cv2

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-lr','--learning_rate',type=float, default=0.00001, help="learning rate")
parser.add_argument('-zb','--zero_background',type=float, default=0.2, help="zero background")
parser.add_argument('-nc','--nb_conv_per_level',type=int, default=2, help="learning rate")
parser.add_argument('-ie','--initial_epoch',type=int,default=0,help="initial epoch")
parser.add_argument('-sc','--scale',type=float,default=0.2,help="scale")
parser.add_argument('-b','--batch_size',default=8,type=int,help="initial epoch")
parser.add_argument('-m','--num_dims',default=192,type=int,help="number of dims")
parser.add_argument('-k1','--num_brain_classes',default=6,type=int,help="number of dims")
parser.add_argument('-k2','--num_anat_classes',default=6,type=int,help="number of dims")
parser.add_argument('-model', '--model', choices=['gmm','192Net'], default='192Net')
parser.add_argument('-o', '--olfactory', action='store_true', help="Flag to disable number of brain classes")

args = parser.parse_args()


log_dir = 'logs'
models_dir = 'models'
num_epochs=param_3d.epoch_num
lr=args.learning_rate
scaling_factor = 0.8

if args.model=='gmm':
    log_dir += '_gmm_seg_'
    models_dir += '_gmm_seg_' 

k1=args.num_brain_classes
k2=args.num_anat_classes

log_dir +=str(args.num_brain_classes)+"_"+str(args.num_anat_classes)
models_dir +=str(args.num_brain_classes)+"_"+str(args.num_anat_classes)

if args.num_dims!=192:
    log_dir +="_"+str(args.num_dims)
    models_dir +="_"+str(args.num_dims)

    
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.95, patience=20, verbose=1, min_lr=1e-7)

latest_weight = max(glob.glob(os.path.join(models_dir, 'weights_epoch_*.h5')), key=os.path.getctime, default=None)

latest_epoch = 0
if latest_weight is not None:
    latest_epoch = int(latest_weight.split('_')[-1].split('.')[0])
    checkpoint_path = latest_weight
else:
    checkpoint_path = os.path.join(models_dir, 'weights_epoch_0.h5')

weights_saver = PeriodicWeightsSaver(filepath=models_dir, latest_epoch=latest_epoch, save_freq=20)  # Save weights every 100 epochs


early_stopping_callback = EarlyStoppingByLossVal(monitor='loss', value=1e-4, verbose=1)


TB_callback = CustomTensorBoard(
    base_log_dir=log_dir,
    models_dir=models_dir,
    histogram_freq=1000,
    write_graph=True,
    write_images=False,
    write_steps_per_second=False,
    update_freq='epoch',
    profile_batch=0,
    embeddings_freq=0,
    embeddings_metadata=None
)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    
if not os.path.exists(models_dir):
    os.makedirs(models_dir)


# path = "/cbica/home/dadashkj/uiuc_pig_brain_atlas/v2.1_12wk_Atlas/Combined_Maps_12wk/Combined_thr50_12wk.nii"
# pig_brain_map = [sf.load_volume(str(path)).resize(0.7).reshape([param_3d.img_size_192,]*3).data]

path = "/cbica/home/dadashkj/uiuc_pig_brain_atlas/v2.1_12wk_Atlas/Combined_Maps_12wk/Combined_thr50_12wk.nii"
pig_brain = sf.load_volume(str(path)).resize(1.2).reshape([param_3d.img_size_192,]*3).data
pig_brain = extend_label_map_with_surfa(pig_brain,scale_factor=80)
pig_brain = dilate_label_map(pig_brain)
pig_brain_map = [pig_brain]


def fill_holes_per_class(mask, labels=None):
    filled_mask = np.zeros_like(mask)
    if labels is None:
        labels = np.unique(mask)
        labels = labels[labels != 0]  # skip background

    for label in labels:
        class_mask = (mask == label)
        filled_class = ndi.binary_fill_holes(class_mask)
        filled_mask[filled_class] = label

    return filled_mask

    

def build_gmm_label_map(k1=6,k2=6):
    from sklearn.mixture import GaussianMixture
    from scipy.ndimage import gaussian_filter
    
    # folders_path = ["/cbica/home/dadashkj/neuroconnlab_pig_data/dwi_PigAnatomical/7646/anat",
    #                 "/cbica/home/dadashkj/neuroconnlab_pig_data/dwi_PigAnatomical/7778/anat",
    #                "/cbica/home/dadashkj/neuroconnlab_pig_data/dwi_PigAnatomical/7665/anat",
    #                "/cbica/home/dadashkj/neuroconnlab_pig_data/dwi_PigAnatomical/8030/anat",
    #                "/cbica/home/dadashkj/neuroconnlab_pig_data/dwi_PigAnatomical/8031/anat",
                        # "/cbica/home/dadashkj/neuroconnlab_pig_data/dwi_PigAnatomical/template",

    folders_path = [
                    "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/template/",
                    "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/106_6month/",
                    "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/82",
                    "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/7646",
                    "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/7665",
                    "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/7778"
                   ]

    if args.olfactory:
        folders_path = ["/cbica/home/dadashkj/neuroconnlab_pig_data/dwi_PigAnatomical/john1",
                   "/cbica/home/dadashkj/neuroconnlab_pig_data/dwi_PigAnatomical/john2",
                   "/cbica/home/dadashkj/neuroconnlab_pig_data/dwi_PigAnatomical/john3",
                   "/cbica/home/dadashkj/neuroconnlab_pig_data/dwi_PigAnatomical/john4",
                        "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/template/"]

    predicted_anat_labels=[]
    for folder_path in folders_path:

        from scipy.ndimage import zoom
        
        # geom_data = sf.load_volume(os.path.join(folder_path, 'anat.nii.gz')).geom
        # pig_anat = sf.load_volume(os.path.join(folder_path, 'anat.nii.gz')).reshape([param_3d.img_size_256,]*3).data
        # pig_brain_mask = sf.load_volume(os.path.join(folder_path, 'anat_brain_mask.nii.gz')).reshape([param_3d.img_size_256,]*3).data
        def load_volume_file(folder_path, file_name):
            # Try both extensions in a loop
            for ext in ['.nii.gz', '.nii']:
                file_path = os.path.join(folder_path, file_name + ext)
                print(f"Checking: {repr(file_path)}")  # Debug print to see which path is being checked
                if os.path.exists(file_path):
                    return sf.load_volume(file_path).reshape([param_3d.img_size_256,]*3).data
        
            # If neither file is found
            raise FileNotFoundError(f"{file_name} file not found in {folder_path}.")

            
        # geom_data = sf.load_volume(os.path.join(folder_path, 'anat.nii.gz')).geom
        geom_data = sf.load_volume(os.path.join(folder_path, 'anat.nii.gz')).geom
        # pig_seg = sf.load_volume(os.path.join(folder_path, 'fast_segmentation_seg.nii.gz')).resize(1).reshape([param_3d.img_size_256,]*3).data
        
        pig_anat = sf.load_volume(os.path.join(folder_path, 'anat.nii.gz')).reshape([param_3d.img_size_256,]*3).data
        sigma = 1  # Adjust sigma for desired smoothing effect
        pig_anat = gaussian_filter(pig_anat, sigma=sigma)
        
        pig_brain_mask = sf.load_volume(os.path.join(folder_path, 'anat_brain_olfactory_mask.nii.gz')).reshape([param_3d.img_size_256,]*3).data
        pig_brain_mask = fill_holes_per_class(pig_brain_mask)
        
        
        
        # pig_brain_mask = ndi.binary_fill_holes(pig_brain_mask)
        pig_brain = pig_anat * ((pig_brain_mask == 1) | (pig_brain_mask == 2))
        
        
        pig_anat = sf.Volume(zoom(pig_anat, scaling_factor, order=1)).reshape((256,)*3).data
        pig_brain = sf.Volume(zoom(pig_brain, scaling_factor, order=1)).reshape((256,)*3).data
        pig_brain_mask = sf.Volume(zoom(pig_brain_mask, scaling_factor, order=1)).reshape((256,)*3).data
        pig_brain_mask = fill_holes_per_class(pig_brain_mask)
        
        # pig_seg = sf.Volume(zoom(pig_seg, scaling_factor, order=1)).reshape((256,)*3)

        pig_skull = np.copy(pig_anat)
        pig_skull[pig_brain_mask>0] = 0
        sigma = 1  # Adjust sigma for desired smoothing effect
        smoothed_anat = gaussian_filter(pig_anat, sigma=sigma)
        brain_data = pig_brain.flatten().reshape(-1, 1)
        non_brain_data = pig_skull.flatten().reshape(-1, 1)


        def make_smooth(label_map):
            smoothed_labels = gaussian_filter(label_map.astype(float), sigma=1)
            return np.round(smoothed_labels).astype(int)
            
        main_brain_voxels = (pig_brain_mask == 1)
        olfactory_voxels = (pig_brain_mask == 2)
        non_brain_voxels = ~((pig_brain_mask == 1) | (pig_brain_mask == 2))
        
        # --- Extract data ---
        main_brain_data = pig_anat[main_brain_voxels].reshape(-1, 1)
        non_brain_data = pig_anat[non_brain_voxels].reshape(-1, 1)
        
        # --- GMM for main brain (excluding olfactory) ---
        gmm_brain = GaussianMixture(n_components=k1 - 1, random_state=42)
        main_brain_labels = gmm_brain.fit_predict(main_brain_data)
        main_brain_labels += 1  # Shift to start from class 1
        
        # --- GMM for non-brain tissue ---
        gmm_non_brain = GaussianMixture(n_components=k2, random_state=42)
        non_brain_labels = gmm_non_brain.fit_predict(non_brain_data)
        # non_brain_labels += (k1 + 1)  # Shift to avoid overlap with brain classes
        
        # --- Initialize label maps ---
        predicted_brain_labels = np.zeros_like(pig_anat, dtype=int)
        predicted_non_brain_labels = np.zeros_like(pig_anat, dtype=int)
        
        # --- Assign labels ---
        predicted_brain_labels[main_brain_voxels] = main_brain_labels
        predicted_brain_labels[olfactory_voxels] = k1  # Fixed class for olfactory
        
        predicted_non_brain_labels[non_brain_voxels] = non_brain_labels
        
        temp_map = np.zeros_like(predicted_brain_labels)
        temp_map[main_brain_voxels] = main_brain_labels  # before smoothing
        
        smoothed_main = make_smooth(temp_map)
        
        # --- Optional: smooth maps ---
        predicted_brain_labels[main_brain_voxels] = smoothed_main[main_brain_voxels]
        predicted_brain_labels[olfactory_voxels] = k1  # restore fixed class
        predicted_non_brain_labels = make_smooth(predicted_non_brain_labels)
        predicted_non_brain_labels[pig_anat==0]=0
        
        # --- Reshape for plotting (if needed) ---
        predicted_brain_labels = predicted_brain_labels.reshape((256, 256, 256))
        predicted_non_brain_labels = predicted_non_brain_labels.reshape((256, 256, 256))
        # predicted_non_brain_labels = shift_non_zero_elements(predicted_non_brain_labels,k1)


        # if k1==0:
        #     pig_seg = sf.load_volume(os.path.join(folder_path, 'fast_segmentation_seg.nii.gz')).resize(1).reshape([param_3d.img_size_256,]*3).data
        #     pig_seg = sf.Volume(zoom(pig_seg, scaling_factor, order=1)).reshape((256,)*3)
        #     predicted_brain_labels = pig_seg
            
        # predicted_non_brain_labels = gmm_non_brain.predict(non_brain_data)
        
        # predicted_brain_labels = make_smooth(predicted_brain_labels)
        # predicted_non_brain_labels = make_smooth(predicted_non_brain_labels)
        

        # predicted_brain_labels = predicted_brain_labels.reshape((256,256,256))
        # predicted_non_brain_labels = predicted_non_brain_labels.reshape((256,256,256))

        # predicted_non_brain_labels[pig_anat==0]=0

        # if args.num_dims == 96:
        #     from scipy.ndimage import binary_dilation
        #     structure = np.ones((3, 3, 3), dtype=bool)
        #     dial_mask = binary_dilation(predicted_brain_labels>0, structure=structure, iterations=10)
        #     predicted_non_brain_labels = predicted_non_brain_labels*(dial_mask>0)

        predicted_non_brain_labels = shift_non_zero_elements(predicted_non_brain_labels,7)
        predicted_anat_label = np.where(predicted_brain_labels != 0, predicted_brain_labels, predicted_non_brain_labels)




        zoomed_predicted_anat_labels = sf.Volume(predicted_anat_label).reshape([args.num_dims,]*3)
        predicted_anat_labels.append(zoomed_predicted_anat_labels)
    return predicted_anat_labels


def build_gmm_label_map2(k1=6,k2=6):
    from sklearn.mixture import GaussianMixture
    from scipy.ndimage import gaussian_filter
    
    folders_path = [
                    "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/template/",
                    "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/106_6month/",
                    "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/82",
                    "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/7646",
                    "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/7665",
                    "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/7778"
                   ]
    if args.olfactory:
        folders_path = ["/cbica/home/dadashkj/neuroconnlab_pig_data/dwi_PigAnatomical/john1",
                   "/cbica/home/dadashkj/neuroconnlab_pig_data/dwi_PigAnatomical/john2",
                   "/cbica/home/dadashkj/neuroconnlab_pig_data/dwi_PigAnatomical/john3",
                   "/cbica/home/dadashkj/neuroconnlab_pig_data/dwi_PigAnatomical/john4",
                        "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/template/"]

    predicted_anat_labels=[]
    for folder_path in folders_path:

        from scipy.ndimage import zoom

        def load_volume_file(folder_path, file_name):
            # Try both extensions in a loop
            for ext in ['.nii.gz', '.nii']:
                file_path = os.path.join(folder_path, file_name + ext)
                print(f"Checking: {repr(file_path)}")  # Debug print to see which path is being checked
                if os.path.exists(file_path):
                    return sf.load_volume(file_path).reshape([param_3d.img_size_256,]*3).data
        
            # If neither file is found
            raise FileNotFoundError(f"{file_name} file not found in {folder_path}.")

            
        # geom_data = sf.load_volume(os.path.join(folder_path, 'anat.nii.gz')).geom
        geom_data = sf.load_volume(os.path.join(folder_path, 'anat.nii.gz')).geom
        # pig_seg = sf.load_volume(os.path.join(folder_path, 'fast_segmentation_seg.nii.gz')).resize(1).reshape([param_3d.img_size_256,]*3).data
        
        pig_anat = sf.load_volume(os.path.join(folder_path, 'anat.nii.gz')).reshape([param_3d.img_size_256,]*3).data
        sigma = 1  # Adjust sigma for desired smoothing effect
        pig_anat = gaussian_filter(pig_anat, sigma=sigma)
        
        pig_brain_mask = sf.load_volume(os.path.join(folder_path, 'anat_brain_olfactory_mask.nii.gz')).reshape([param_3d.img_size_256,]*3).data
        pig_brain_mask = fill_holes_per_class(pig_brain_mask)
        
        
        
        # pig_brain_mask = ndi.binary_fill_holes(pig_brain_mask)
        pig_brain = pig_anat * ((pig_brain_mask == 1) | (pig_brain_mask == 2))
        
        
        pig_anat = sf.Volume(zoom(pig_anat, scaling_factor, order=1)).reshape((256,)*3).data
        pig_brain = sf.Volume(zoom(pig_brain, scaling_factor, order=1)).reshape((256,)*3).data
        pig_brain_mask = sf.Volume(zoom(pig_brain_mask, scaling_factor, order=1)).reshape((256,)*3).data
        pig_brain_mask = fill_holes_per_class(pig_brain_mask)
        
        # pig_seg = sf.Volume(zoom(pig_seg, scaling_factor, order=1)).reshape((256,)*3)

        pig_skull = np.copy(pig_anat)
        pig_skull[pig_brain_mask>0] = 0
        sigma = 1  # Adjust sigma for desired smoothing effect
        smoothed_anat = gaussian_filter(pig_anat, sigma=sigma)
        brain_data = pig_brain.flatten().reshape(-1, 1)
        non_brain_data = pig_skull.flatten().reshape(-1, 1)


        def make_smooth(label_map):
            smoothed_labels = gaussian_filter(label_map.astype(float), sigma=1)
            return np.round(smoothed_labels).astype(int)
            
        main_brain_voxels = (pig_brain_mask == 1)
        olfactory_voxels = (pig_brain_mask == 2)
        
        # --- Extract data ---
        main_brain_data = pig_anat[main_brain_voxels].reshape(-1, 1)
        
        # --- GMM for main brain (excluding olfactory) ---
        gmm_brain = GaussianMixture(n_components=k1 - 1, random_state=42)
        main_brain_labels = gmm_brain.fit_predict(main_brain_data)
        main_brain_labels += 1  # Shift to start from class 1
        
        # --- GMM for non-brain tissue ---
        gmm_non_brain = GaussianMixture(n_components=k2, random_state=42)
        
        # --- Initialize label maps ---
        predicted_brain_labels = np.zeros_like(pig_anat, dtype=int)
        
        # --- Assign labels ---
        predicted_brain_labels[main_brain_voxels] = main_brain_labels
        predicted_brain_labels[olfactory_voxels] = k1  # Fixed class for olfactory
        
        
        temp_map = np.zeros_like(predicted_brain_labels)
        temp_map[main_brain_voxels] = main_brain_labels  # before smoothing
        
        smoothed_main = make_smooth(temp_map)
        
        # --- Optional: smooth maps ---
        predicted_brain_labels[main_brain_voxels] = smoothed_main[main_brain_voxels]
        predicted_brain_labels[olfactory_voxels] = k1  # restore fixed class
        
        # --- Reshape for plotting (if needed) ---
        predicted_brain_labels = predicted_brain_labels.reshape((256, 256, 256))

        zoomed_predicted_anat_labels = sf.Volume(predicted_brain_labels).reshape([args.num_dims,]*3)
        predicted_anat_labels.append(zoomed_predicted_anat_labels)
    print("########### number of label maps",len(predicted_anat_labels))
    return predicted_anat_labels



# predicted_anat_labels=build_gmm_label_map(5,5)
pig_gmm_brain_map = build_gmm_label_map(6,6)
pig_brain_map = pig_gmm_brain_map
config_file = "params_gmm_seg_192.json"

if args.num_dims==param_3d.img_size_128 and args.model=="gmm":
    pig_brain_map = pig_gmm_brain_map
    config_file = "params_gmm_seg_128.json"
elif args.num_dims==param_3d.img_size_96 and args.model=="gmm":
    pig_brain_map = pig_gmm_brain_map
    config_file = "params_gmm_seg_96.json"
    

with open(config_file, "r") as json_file:
    config = json.load(json_file)
    
gen=generator_brain_window_Net(pig_brain_map,args.num_dims)

model_pig_config = config["pig_48"]
model_shapes_config = config["shapes"]
model_veins_config = config["veins"]
model_shapes_config["labels_in"] = [0] + list(range(7, 17))
model_veins_config["labels_in"] = [0] + list(range(7, 17))

num_forground_classes = 6

model_pig_config["in_shape"]=[ args.num_dims, args.num_dims, args.num_dims]
model_shapes_config["in_shape"]=[ args.num_dims, args.num_dims, args.num_dims]
model_veins_config["in_shape"]=[ args.num_dims, args.num_dims, args.num_dims]

model3_config = config["labels_to_image_model_48"]

model3_config["labels_out"] = {int(key): value for key, value in model3_config["labels_out"].items()}

model_pig_config["labels_out"] = {
    int(key): int(key) if int(key) in model_pig_config["labels_in"] else 0
    for key in model3_config["labels_out"].keys()
}

model3_config["in_shape"]=[ args.num_dims, args.num_dims, args.num_dims]
model_pig = create_model(model_pig_config)
model_shapes = create_model(model_shapes_config)

model_shapes_config["labels_out"] = {
    int(key): int(key) if int(key) in model_shapes_config["labels_in"] else 0
    for key in model3_config["labels_out"].keys()
}

model_veins_config["labels_out"] = {
    int(key): int(key) if int(key) in model_veins_config["labels_in"] else 0
    for key in model3_config["labels_out"].keys()
}

shapes = draw_shapes_easy(shape = (args.num_dims,)*3)   

labels_to_image_model = create_model(model3_config)
model_veins = create_model(model_veins_config)

# shapes = draw_shapes_easy(shape = (args.num_dims,)*3)   

import tensorflow as tf
from tensorflow.keras.layers import Layer

def shift_non_zero_back_elements(bg, shift_value):
    non_zero_mask = tf.not_equal(bg, 0)
    shifted_non_zero_elements = tf.where(non_zero_mask, bg - shift_value, bg)
    return shifted_non_zero_elements
    
def mask_bg_near_fg(fg, bg, dilation_iter=8):
    d_iter = tf.random.uniform([], minval=1, maxval=dilation_iter + 1, dtype=tf.int32)
    k = 2 * d_iter + 1
    fg_mask = tf.cast(fg > 0, tf.float32)
    fg_mask = tf.reshape(fg_mask, [1, *fg_mask.shape, 1])
    fg_mask = tf.nn.max_pool3d(fg_mask, ksize=[1, k, k, k, 1], strides=[1, 1, 1, 1, 1], padding='SAME')
    fg_mask = tf.squeeze(fg_mask > 0)

    bg_masked = tf.where(fg_mask, bg, tf.zeros_like(bg[0, ..., 0]))
    result = tf.where(fg > 0, fg, bg_masked)
    return result

import tensorflow as tf

def density_penalty(pred, threshold=0.5, min_density=0.01):
    """
    Penalizes classes that are too sparse (fragmented or weak).

    Parameters:
        pred: tf.Tensor [B, Z, Y, X, C]
        threshold: binarization threshold
        min_density: desired fraction of active voxels

    Returns:
        scalar penalty
    """
    p_bin = tf.cast(pred > threshold, tf.float32)
    B, Z, Y, X, C = tf.unstack(tf.shape(pred))
    total_voxels = tf.cast(Z * Y * X, tf.float32)

    class_densities = tf.reduce_sum(p_bin, axis=[1, 2, 3]) / total_voxels  # shape [B, C]
    penalty = tf.reduce_mean(tf.nn.relu(min_density - class_densities[:, 1:]))  # skip background

    return penalty


def final_loss(y, pred):
    loss = soft_dice(y, pred) + 0.3 * density_penalty(pred)
    return loss


    
combined_map = load_retina_vessels_with_volume("retina_blood_vessles",
    shape=(args.num_dims, args.num_dims, args.num_dims))

if __name__ == "__main__":
    en = [16 ,16 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64]
    de = [64 ,64 ,64 ,64, 64 ,64 ,64, 64, 64, 16 ,16 ,3]
    random.seed(3000)
    epsilon =1e-7
    steps_per_epoch = 100
    min_max_norm = Lambda(lambda x: (x - K.min(x)) / (K.max(x) - K.min(x)+ epsilon) * (1.0) )

    unet_model = vxm.networks.Unet(inshape=(args.num_dims,args.num_dims,args.num_dims, 1), nb_features=(en, de),
                   nb_conv_per_level=2,
                   final_activation_function='softmax')


    if args.model=="gmm" and args.num_dims == param_3d.img_size_96:
        input_img = Input(shape=(param_3d.img_size_96,param_3d.img_size_96,param_3d.img_size_96,1))

        _, fg = model_pig(input_img)
        
        _ , bg = model_shapes(input_img)
        
        bg = shift_non_zero_back_elements(bg[0,...,0],6)

        _, cm = model_veins(combined_map[None,...,None])
        cm = cm[0,...,0]
        final_shapes = tf.where(cm == 0, bg, cm)
        final_shapes = shift_non_zero_elements(final_shapes,6)

        
        result = mask_bg_near_fg(fg[0,...,0], final_shapes , dilation_iter=5)
        
        result = result[None, ..., None]

        generated_img , y = labels_to_image_model(result)
        generated_img_norm = min_max_norm(generated_img)
        
        segmentation = unet_model(generated_img_norm)
        combined_model = Model(inputs=input_img, outputs=segmentation)
        combined_model.add_loss(soft_dice(y,segmentation))
        combined_model.compile(optimizer=Adam(learning_rate=lr))


    elif args.model=="gmm" and args.num_dims == param_3d.img_size_128:
        input_img = Input(shape=(param_3d.img_size_128,param_3d.img_size_128,param_3d.img_size_128,1))

        _, fg = model_pig(input_img)
        
        _ , bg = model_shapes(input_img)
        
        # bg = shift_non_zero_back_elements(bg[0,...,0],6)
        bg = bg[0,...,0]
        _, cm = model_veins(combined_map[None,...,None])
        cm = cm[0,...,0]
        final_shapes = tf.where(cm == 0, bg, cm)
        final_shapes = shift_non_zero_elements(final_shapes,6)

        
        result = mask_bg_near_fg(fg[0,...,0], final_shapes , dilation_iter=7)
        
        result = result[None, ..., None]

        generated_img , y = labels_to_image_model(result)
        generated_img_norm = min_max_norm(generated_img)
        
        segmentation = unet_model(generated_img_norm)
        combined_model = Model(inputs=input_img, outputs=segmentation)
        combined_model.add_loss(soft_dice(y,segmentation))
        combined_model.compile(optimizer=Adam(learning_rate=lr))

    elif args.model=="gmm":
        input_img = Input(shape=(param_3d.img_size_192,param_3d.img_size_192,param_3d.img_size_192,1))
        
        _, fg = model_pig(input_img[None,...,None])
        _, bg = model_shapes(input_img[None,...,None])

        result = fg[0,...,0] + bg[0,...,0] * tf.cast(fg[0,...,0] == 0,tf.int32)
        result = result[None,...,None]


        generated_img , y = labels_to_image_model(result)
        generated_img_norm = min_max_norm(generated_img)
        
        segmentation = unet_model(generated_img_norm)
        combined_model = Model(inputs=input_img, outputs=segmentation)
        combined_model.add_loss(soft_dice(y, segmentation))
        combined_model.compile(optimizer=Adam(learning_rate=lr))

    callbacks_list = [TB_callback, weights_saver,reduce_lr]
    
    if os.path.exists(checkpoint_path):
        print(checkpoint_path)
        combined_model.load_weights(checkpoint_path)
        print("Loaded weights from the checkpoint and continued training.")
    else:
        print(checkpoint_path)
        print("Checkpoint file not found.")

    combined_model.fit(gen, epochs=num_epochs, batch_size=1, steps_per_epoch=steps_per_epoch,  callbacks=callbacks_list)

    