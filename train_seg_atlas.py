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
import os

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
parser.add_argument('-n', '--new_labels', action='store_true', help="Flag to specify new or old labels")

args = parser.parse_args()

gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available:", len(gpus))
print("################# GPUs:", gpus)

log_dir = 'logs'
models_dir = 'models'
num_epochs=param_3d.epoch_num
lr=args.learning_rate
scaling_factor = 1

if args.model=='gmm':
    log_dir += '_gmm_seg_atlas_'
    models_dir += '_gmm_seg_atlas_' 

k1=args.num_brain_classes
k2=args.num_anat_classes

log_dir +=str(args.num_brain_classes)+"_"+str(args.num_anat_classes)
models_dir +=str(args.num_brain_classes)+"_"+str(args.num_anat_classes)

if args.num_dims!=192:
    log_dir +="_"+str(args.num_dims)
    models_dir +="_"+str(args.num_dims)

if args.new_labels:
    log_dir +="_new"
    models_dir +="_new"

if args.num_dims==96:
    scaling_factor = 0.5

num_forground_classes = 102 if args.new_labels else 250
# csf_label = 102 if args.new_labels else 250

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


    folders_path = ["/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/template/"]


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

        # folder_path = "/cbica/home/dadashkj/neuroconnlab_pig_data/dwi_PigAnatomical/7646/anat"
        from scipy.ndimage import binary_fill_holes
        structure = np.ones((3, 3, 3), dtype=bool)

        # geom_data = sf.load_volume(os.path.join(folder_path, 'anat.nii.gz')).geom
        pig_histo = sf.load_volume(os.path.join(folder_path, 'anat_brain.nii.gz')).reshape([192, 192, 192]).data

        if args.new_labels:
            pig_seg = sf.load_volume(os.path.join(folder_path, 'anat_brain_atlas102.nii.gz')).reshape([192, 192, 192]).data
        else:
            pig_seg = sf.load_volume(os.path.join(folder_path, 'anat_brain_atlas.nii.gz')).reshape([192, 192, 192]).data

        pig_brain = pig_histo
        shapes = draw_shapes_easy(shape = (192,)*3)   
        
        brain_data = pig_brain.flatten().reshape(-1, 1)
        non_brain_data = shapes.numpy().flatten().reshape(-1, 1)

        predicted_brain_labels = pig_seg
        predicted_non_brain_labels = non_brain_data.reshape((192,192,192))

        
       
        predicted_non_brain_labels = shift_non_zero_elements(predicted_non_brain_labels,num_forground_classes)
        predicted_anat_label = np.where(predicted_brain_labels > 0, predicted_brain_labels, predicted_non_brain_labels)

        zoomed_predicted_anat_labels = sf.Volume(predicted_anat_label).reshape([args.num_dims,]*3)
        print("unique labels: ",np.unique(zoomed_predicted_anat_labels))
        predicted_anat_labels.append(zoomed_predicted_anat_labels)
    return predicted_anat_labels

def build_gmm_label_map2(k1=6,k2=6):
    from sklearn.mixture import GaussianMixture
    from scipy.ndimage import gaussian_filter


    folders_path = ["/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/template/"]


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

        # folder_path = "/cbica/home/dadashkj/neuroconnlab_pig_data/dwi_PigAnatomical/7646/anat"
        from scipy.ndimage import binary_fill_holes
        structure = np.ones((3, 3, 3), dtype=bool)

        # geom_data = sf.load_volume(os.path.join(folder_path, 'anat.nii.gz')).geom
        pig_histo = sf.load_volume(os.path.join(folder_path, 'anat_brain.nii.gz')).reshape([192, 192, 192]).data

        if args.new_labels:
            pig_seg = sf.load_volume(os.path.join(folder_path, 'anat_brain_atlas102.nii.gz')).reshape([192, 192, 192]).data
        else:
            pig_seg = sf.load_volume(os.path.join(folder_path, 'anat_brain_atlas.nii.gz')).reshape([192, 192, 192]).data

        pig_brain = pig_histo
        shapes = draw_shapes_easy(shape = (192,)*3)   
        
        brain_data = pig_brain.flatten().reshape(-1, 1)
        # non_brain_data = shapes.numpy().flatten().reshape(-1, 1)

        predicted_brain_labels = pig_seg
        # predicted_non_brain_labels = non_brain_data.reshape((192,192,192))

        
       
        # predicted_non_brain_labels = shift_non_zero_elements(predicted_non_brain_labels,num_forground_classes)
        predicted_anat_label = predicted_brain_labels# np.where(predicted_brain_labels > 0, predicted_brain_labels, predicted_non_brain_labels)

        zoomed_predicted_anat_labels = sf.Volume(predicted_anat_label).reshape([args.num_dims,]*3)
        print("unique labels: ",np.unique(zoomed_predicted_anat_labels))
        predicted_anat_labels.append(zoomed_predicted_anat_labels)
    return predicted_anat_labels
    
# predicted_anat_labels=build_gmm_label_map(5,5)
pig_gmm_brain_map = build_gmm_label_map2(6,6)
pig_brain_map = pig_gmm_brain_map
config_file = "params_gmm_seg_atlas_192.json"

with open(config_file, "r") as json_file:
    config = json.load(json_file)

# if args.num_dims==param_3d.img_size_96 and args.model=="gmm":
#     pig_brain_map = pig_gmm_brain_map
#     config_file = "params_gmm_seg_atlas_96.json"

gen=generator_brain_window_Net(pig_brain_map,args.num_dims)
all_possible_labels = list(range(0,num_forground_classes+10))  # or use np.unique(pig_seg) if available

model_pig_config = config["pig_48"]
model_veins_config = config["veins"]

model_pig_config["labels_in"] = list(range(num_forground_classes+1))

model_shapes_config = config["shapes"]
model_shapes_config["labels_in"] = [0] + list(range(num_forground_classes+1, num_forground_classes+10))
model_veins_config["labels_in"] = [0] + list(range(num_forground_classes+1, num_forground_classes+10))

model_pig_config["in_shape"]=[ args.num_dims, args.num_dims, args.num_dims]
model_shapes_config["in_shape"]=[ args.num_dims, args.num_dims, args.num_dims]
model_veins_config["in_shape"]=[ args.num_dims, args.num_dims, args.num_dims]

model3_config = config["labels_to_image_model_48"]
model3_config["labels_in"] =all_possible_labels

model3_config["labels_out"] = {
    label: label if 1 <= label <= num_forground_classes else 0
    for label in all_possible_labels  # Only up to 250, so output shape = 251
}

model_pig_config["labels_out"] = {
    int(key): int(key) if int(key) in model_pig_config["labels_in"] else 0
    for key in model3_config["labels_out"].keys()
}


model_veins_config["labels_out"] = {
    int(key): int(key) if int(key) in model_veins_config["labels_in"] else 0
    for key in model3_config["labels_out"].keys()
}

model3_config["in_shape"]=[ args.num_dims, args.num_dims, args.num_dims]
model_pig = create_model(model_pig_config)
model_shapes = create_model(model_shapes_config)
model_veins = create_model(model_veins_config)

shapes = draw_shapes_easy(shape = (args.num_dims,)*3)   

labels_to_image_model = create_model(model3_config)

print("model pig",model_pig_config["labels_in"])
print("model shapes",model_shapes_config["labels_in"])
print("model 3",model3_config["labels_in"],model3_config["labels_out"])

combined_map = load_retina_vessels_with_volume("retina_blood_vessles",
    shape=(192, 192, 192))


# Or more detailed:

if __name__ == "__main__":
    num_classes = num_forground_classes+1
    en = [16 ,16 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64]
    de = [64 ,64 ,64 ,64, 64 ,64 ,64, 64, 64, 16 ,16 ,num_classes]
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
        _, bg = model_shapes(input_img)

        result = fg[0,...,0] + bg[0,...,0] * tf.cast(fg[0,...,0] == 0,tf.int32)
        result = result[None,...,None]


        generated_img , y = labels_to_image_model(result)
        generated_img_norm = min_max_norm(generated_img)
        
        segmentation = unet_model(generated_img_norm)
        combined_model = Model(inputs=input_img, outputs=segmentation)
        combined_model.add_loss(soft_dice(y, segmentation))
        combined_model.compile(optimizer=Adam(learning_rate=lr))

    elif args.model=="gmm" and args.num_dims == param_3d.img_size_128:
        input_img = Input(shape=(param_3d.img_size_128,param_3d.img_size_128,param_3d.img_size_128,1))
        
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

    elif args.model=="gmm":
        input_img = Input(shape=(param_3d.img_size_192,param_3d.img_size_192,param_3d.img_size_192,1))

        _, fg = model_pig(input_img)
        
        _, cm = model_veins(combined_map[None,...,None])
        cm = cm[0,...,0]
        final_shapes = shift_non_zero_elements(cm,num_forground_classes)
        
        result = mask_bg_near_fg(fg, final_shapes[None,...,None] , dilation_iter=5)
        result = result[None, ..., None]

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