import os
import json
import numpy as np

from src.wedge_video import GelSightWedgeVideo, ORIGINAL_IMG_SIZE
from src.contact_force import ContactForce
from src.gripper_width import GripperWidth
from src.grasp_data import GraspData
from src.analytical_estimate import EstimateModulus
from src.preprocess_data import preprocess_grasp, TRAINING_FORCE_THRESHOLD
from src.train import ModulusModel, N_FRAMES

#######################################################
# HOW TO USE GELSIGHT YOUNG'S MODULUS ESTIMATION CODE #
#######################################################

# -----------------------------------------------------
# FOR A SINGLE GRASP:
# -----------------------------------------------------

# 1. Record grasp
#       - We record grasps using src/collect_grasp_data.py
#       - Depending on hardware, create your own method of recording these measurements
#       - For now, we will enter some dummy values to showcase the estimation workflow

wedge_video = GelSightWedgeVideo(config_csv='./wedge_config/config_no_markers.csv') # Calibrate new config for your sensor
wedge_video._raw_rgb_frames = 100*[np.zeros((ORIGINAL_IMG_SIZE[0], ORIGINAL_IMG_SIZE[1], 3))]
wedge_video_markers = GelSightWedgeVideo(config_csv='./wedge_config/config_markers.csv')
wedge_video_markers._raw_rgb_frames = 100*[np.zeros((ORIGINAL_IMG_SIZE[0], ORIGINAL_IMG_SIZE[1], 3))]

contact_force = ContactForce()
contact_force._forces = np.linspace(0, 30, 100)
gripper_width = GripperWidth()
gripper_width._widths = np.linspace(0.08, 0.02, 100)

grasp_data = GraspData(
                wedge_video=wedge_video, wedge_video_markers=wedge_video_markers,
                contact_force=contact_force, gripper_width=gripper_width
            ) # Object for wrapping all recorded data

# Alternatively, can load from a saved location...
# grasp_data = GraspData()
# grasp_data.load(path_to_file)

# 2. Compute analytical estimates
#       - Using grasp data, apply analytical algorithms to generate estimates

analytical_estimator    = EstimateModulus(grasp_data=grasp_data, use_gripper_width=True, use_markers_video=True)
E_hat_simple_elastic    = analytical_estimator.fit_modulus_simple()
E_hat_hertz_MDR         = analytical_estimator.fit_modulus_hertz_MDR()

print(f"Simple estimate of Young's Modulus... {E_hat_simple_elastic:.3e} Pa")
print(f"Hertzian (MDR) estimate of Young's Modulus... {E_hat_hertz_MDR:.3e} Pa")

# 3. Preprocess the data
    # - Sample tactile images, forces, and widths equidistantly across loading sequence
    # - Here, we choose to use RGB images with markers as our input

grasp_data.auto_clip()
i_contact = np.argmax(grasp_data.forces() >= TRAINING_FORCE_THRESHOLD) # First contact
i_peak = np.argmax(grasp_data.forces()) # Peak force

video_sample_indices = np.linspace(i_contact, i_peak, N_FRAMES, endpoint=True, dtype=int)
force_sample_indices = np.linspace(i_contact, i_peak, N_FRAMES, endpoint=True, dtype=int)

sampled_tactile_images = wedge_video_markers.diff_images()[video_sample_indices, :, :, :]
sampled_forces = contact_force.forces()[force_sample_indices, None] # F
sampled_widths = gripper_width.widths()[force_sample_indices, None] # w

# 4. Execute learned model
#       - Using grasp data and estimations, generate estimate on a previously trained model
#       - Weights are saved for model trained on the full_dataset, or rubber_only

path_to_model = './model/full_dataset' # './model/rubber_only'
with open(f'{path_to_model}/config.json', 'r') as file:
    config = json.load(file)
config['use_wandb'] = False

modulus_model = ModulusModel(config)
modulus_model.load_model(path_to_model)

E_hat = modulus_model.estimate(
                sampled_tactile_images, forces=sampled_forces, widths=sampled_widths, \
                E_hat_simple=E_hat_simple_elastic, E_hat_hertz=E_hat_hertz_MDR
            ).item()

print(f"Learned estimate of Young's Modulus... {E_hat:.3e} Pa")



# -----------------------------------------------------
# TRAIN OVER MANY COLLECTED GRASPS:
# -----------------------------------------------------

# 1. Record grasps
#       - Use src/collect_grasp_data.py to record data for multiple grasps of many objects
#       - Over grasping, video from GelSight Wedge's alongside forces and gripper width measurements are recorded
#       - GelSightWedgeVideo objects could be augmented to facilitate recording of other camera-based tactile sensors

# Data will be saved into the following file structure...
#   data
#     └── raw_data
#          └── object_name
#               └── grasp={i}
#                    ├── {object_name}.avi            (GelSight video without markers)
#                    ├── {object_name}_markers.avi    (GelSight video with markers)
#                    ├── {object_name}_forces.avi     (Normal force measurments)
#                    └── {object_name}_widths.avi     (Gripper width measurements)

# Code can also be augmented to only record from a single camera-based tactile sensor

# 2. Compute analytical estimates
#       - With raw collected data, we can estimate the Young's Modulus using analytical methods
#       - This is facilitated by EstimateModulus in src/analytical_estimations.py
#       - NOTE: This step is not critical for training as estimations will be computed alongside data preprocessing

'''
wedge_video = GelSightWedgeVideo(config_csv='./wedge_config/config_no_markers.csv') # Calibrate new config for your sensor
wedge_video_markers = GelSightWedgeVideo(config_csv='./wedge_config/config_markers.csv')
contact_force = ContactForce()
gripper_width = GripperWidth()

grasp_data = GraspData(
                wedge_video=wedge_video, wedge_video_markers=wedge_video_markers,
                contact_force=contact_force, gripper_width=gripper_width
            ) # Object for wrapping all recorded data
 
simple_predictions = {} # Save predictions of Young's Modulus per grasp for all objects
hertz_predictions = {}

raw_collected_data_dir = './data/raw_data'
for object_name in os.listdir(raw_collected_data_dir):

    simple_predictions[object_name] = []
    hertz_predictions[object_name] = []

    for grasp_folder in os.listdir(f'{raw_collected_data_dir}/{object_name}'):
        
        grasp_data.load(f'{raw_collected_data_dir}/{object_name}/{grasp_folder}/{object_name}.avi')

        analytical_estimator    = EstimateModulus(grasp_data=grasp_data, use_gripper_width=True, use_markers_video=True)
        E_hat_simple_elastic    = analytical_estimator.fit_modulus_simple()
        E_hat_hertz_MDR         = analytical_estimator.fit_modulus_hertz_MDR()

        simple_predictions[object_name].append(E_hat_simple_elastic)
        hertz_predictions[object_name].append(E_hat_hertz_MDR)
'''

# 3. Preprocess the data
#       - We sample a set number of tactile images across each grasp 
#       - Alongside these images, forces and widths are sampled and analytical estimations are computed
#       - Multiple sampled augmentations are created for each grasp

'''
N_FRAMES = 3
DESTINATION_DIR = './data/gelsight_youngs_modulus_dataset'

# Loop through all data files
for object_name in os.listdir(raw_collected_data_dir):

    for grasp_folder in os.listdir(f'{raw_collected_data_dir}/{object_name}'):
        for file_name in os.listdir(f'{raw_collected_data_dir}/{object_name}/{grasp_folder}'):
            if os.path.splitext(file_name)[1] != '.avi' or file_name.count("_markers") > 0:
                continue

            # Preprocess the raw grasp
            preprocess_grasp(f'{raw_collected_data_dir}/{object_name}/{grasp_folder}/{os.path.splitext(file_name)[0]}', \
                             destination_dir=DESTINATION_DIR, num_frames_to_sample=N_FRAMES, grasp_data=grasp_data
                            )
'''

# The resulting file structure is...
#   data
#    └── gelsight_youngs_modulus_dataset
#         └── {object_name}
#               ├── metadata.json
#               └── grasp={grasp_number}
#                    └── augmentation={augmentation_number}
#                         ├── RGB.pkl
#                         ├── depth.pkl
#                         ├── RGB_markers.pkl
#                         ├── depth_markers.pkl
#                         ├── forces.pkl
#                         ├── widths.pkl
#                         ├── elastic_estimate.pkl
#                         └── hertz_estimate.pkl

# 4. Train and validate
#       - Once data is preprocessed, we are ready to train!

# config = {}   # Reference src/train.py for model configuration options
#               # Should match those of saved models if you would like to utilize these weights

# train_modulus = ModulusModel(config, device=device)
# train_modulus.train()