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

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-lr','--learning_rate',type=float, default=0.00001, help="learning rate")
parser.add_argument('-zb','--zero_background',type=float, default=0.2, help="zero background")
parser.add_argument('-nc','--nb_conv_per_level',type=int, default=2, help="learning rate")
parser.add_argument('-ie','--initial_epoch',type=int,default=0,help="initial epoch")
parser.add_argument('-sc','--scale',type=float,default=0.2,help="scale")
parser.add_argument('-b','--batch_size',default=8,type=int,help="initial epoch")
parser.add_argument('-m','--num_dims',default=192,type=int,help="number of dims")
parser.add_argument('-model', '--model', choices=['gmm','192Net'], default='192Net')


args = parser.parse_args()


log_dir = 'logs'
models_dir = 'models'
num_epochs=param_3d.epoch_num
lr=args.learning_rate

if args.model=='gmm':
    log_dir += '_gmm'
    models_dir += '_gmm' 


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

def build_gmm_label_map(k1=5,k2=5):
    from sklearn.mixture import GaussianMixture
    from scipy.ndimage import gaussian_filter
    
    folders_path = ["/cbica/home/dadashkj/neuroconnlab_pig_data/dwi_PigAnatomical/7646/anat",
                    "/cbica/home/dadashkj/neuroconnlab_pig_data/dwi_PigAnatomical/7778/anat",
                   "/cbica/home/dadashkj/neuroconnlab_pig_data/dwi_PigAnatomical/7665/anat",
                   "/cbica/home/dadashkj/neuroconnlab_pig_data/dwi_PigAnatomical/8030/anat",
                   "/cbica/home/dadashkj/neuroconnlab_pig_data/dwi_PigAnatomical/8031/anat",
                    "/cbica/home/dadashkj/neuroconnlab_pig_data/dwi_PigAnatomical/template"
                   ]
                   # "/cbica/home/dadashkj/neuroconnlab_pig_data/dwi_PigAnatomical/john1",
                   # "/cbica/home/dadashkj/neuroconnlab_pig_data/dwi_PigAnatomical/john2",
                   # "/cbica/home/dadashkj/neuroconnlab_pig_data/dwi_PigAnatomical/john3",
                   # "/cbica/home/dadashkj/neuroconnlab_pig_data/dwi_PigAnatomical/john4"]
    predicted_anat_labels=[]
    for folder_path in folders_path:
        geom_data = sf.load_volume(os.path.join(folder_path, 'anat.nii.gz')).geom
        pig_anat = sf.load_volume(os.path.join(folder_path, 'anat.nii.gz')).reshape([param_3d.img_size_192,]*3).data
        pig_brain = sf.load_volume(os.path.join(folder_path, 'anat_brain.nii.gz')).reshape([param_3d.img_size_192,]*3).data
        pig_brain_mask = sf.load_volume(os.path.join(folder_path, 'anat_brain_mask.nii.gz')).reshape([param_3d.img_size_192,]*3).data
    
    
        pig_skull = np.copy(pig_anat)
        pig_skull[pig_brain_mask == 1] = 0
        
        sigma = 3  # Adjust sigma for desired smoothing effect
        smoothed_anat = gaussian_filter(pig_anat, sigma=sigma)
        brain_data = pig_brain.flatten().reshape(-1, 1)
        non_brain_data = pig_skull.flatten().reshape(-1, 1)
    
        # Apply GMM for brain regions (assumes 29 brain regions to be classified)
        gmm_brain = GaussianMixture(n_components=k1, random_state=42)
        gmm_brain.fit(brain_data)  # Fit GMM on the brain data
        gmm_non_brain = GaussianMixture(n_components=k2, random_state=42)  # 0 for background, 30-40 for other tissues
        gmm_non_brain.fit(non_brain_data)  # Fit GMM on the non-brain data
        predicted_brain_labels = gmm_brain.predict(brain_data)
        predicted_non_brain_labels = gmm_non_brain.predict(non_brain_data)
        predicted_non_brain_labels = shift_non_zero_elements(predicted_non_brain_labels,k1)
        predicted_brain_labels = predicted_brain_labels.reshape((192,192,192))
        predicted_non_brain_labels = predicted_non_brain_labels.numpy().reshape((192,192,192))
        predicted_anat_labels.append(predicted_brain_labels+predicted_non_brain_labels)
    return predicted_anat_labels

# predicted_anat_labels=build_gmm_label_map(5,5)
pig_gmm_brain_map = build_gmm_label_map(5,5)
config_file= "params_192.json"

if args.model=="gmm":
    pig_brain_map = pig_gmm_brain_map
    config_file = "params_gmm_192.json"

with open(config_file, "r") as json_file:
    config = json.load(json_file)
    
gen=generator_brain_window_Net(pig_brain_map,param_3d.img_size_192)

model_pig_config = config["pig_48"]
model_shapes_config = config["shapes"]
model_pig_config["in_shape"]=[ param_3d.img_size_192, param_3d.img_size_192, param_3d.img_size_192]
model_shapes_config["in_shape"]=[ param_3d.img_size_192, param_3d.img_size_192, param_3d.img_size_192]

model3_config = config["labels_to_image_model_48"]
model3_config["labels_out"] = {int(key): value for key, value in model3_config["labels_out"].items()}
model3_config["in_shape"]=[ param_3d.img_size_192, param_3d.img_size_192, param_3d.img_size_192]
model_pig = create_model(model_pig_config)
model_shapes = create_model(model_shapes_config)
shapes = draw_shapes_easy(shape = (param_3d.img_size_192,)*3)   

labels_to_image_model = create_model(model3_config)

if __name__ == "__main__":
    en = [16 ,16 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64]
    de = [64 ,64 ,64 ,64, 64 ,64 ,64, 64, 64, 16 ,16 ,2]
    random.seed(3000)
    epsilon =1e-7
    steps_per_epoch = 100
    min_max_norm = Lambda(lambda x: (x - K.min(x)) / (K.max(x) - K.min(x)+ epsilon) * (1.0) )

    unet_model = vxm.networks.Unet(inshape=(param_3d.img_size_192,param_3d.img_size_192,param_3d.img_size_192, 1), nb_features=(en, de),
                   nb_conv_per_level=2,
                   final_activation_function='softmax')
    if args.model=="192Net":
        input_img = Input(shape=(param_3d.img_size_192,param_3d.img_size_192,param_3d.img_size_192,1))
        _, fg = model_pig(input_img)
        
        shapes = draw_shapes_easy(shape = (param_3d.img_size_192,)*3,num_label=10)
        
        shapes = tf.squeeze(shapes)
        shapes = tf.cast(shapes, tf.int32)
        
        bones = draw_bones_only(shape = (param_3d.img_size_192,)*3,num_labels=16,num_bones=20)
        bones = tf.cast(bones, tf.int32)
        bones = shift_non_zero_elements(bones,29)
        
        shapes2 = draw_layer_elipses(shape=(param_3d.img_size_192,)*3, num_labels=8, num_shapes=50, sigma=2)
        shapes2 = tf.squeeze(shapes2)
        shapes2 = tf.cast(shapes2, tf.int32)
        shapes2 = shift_non_zero_elements(shapes2,29)  
        
        shapes2 = bones + shapes2 * tf.cast(bones == 0,tf.int32)
        result = fg[0,...,0] + shapes2 * tf.cast(fg[0,...,0] == 0,tf.int32)
        result= result[None,...,None]
    
        
        generated_img , y = labels_to_image_model(result)
        generated_img_norm = min_max_norm(generated_img)
        
        segmentation = unet_model(generated_img_norm)
        combined_model = Model(inputs=input_img, outputs=segmentation)
        combined_model.add_loss(soft_dice(y, segmentation))
        combined_model.compile(optimizer=Adam(learning_rate=0.00001))
    elif args.model=="gmm":
        input_img = Input(shape=(param_3d.img_size_192,param_3d.img_size_192,param_3d.img_size_192,1))
        a = input_img[0, ...,0] 
        fg_mask = a <6
        fragment_brain = tf.where(fg_mask, a, 0)

        # Apply the models
        _, fg = model_pig(input_img[None,...,None])

        bg_mask = (a > 5) | (a == 0)
        fragment_bg = tf.where(bg_mask, a, 0)

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

    