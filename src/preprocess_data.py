import os
import random
import csv
import pickle
import numpy as np

from tqdm import tqdm

from src.wedge_video import GelSightWedgeVideo
from src.gripper_width import GripperWidth
from src.contact_force import ContactForce
from src.grasp_data import GraspData
from src.analytical_estimate import EstimateModulus

TRAINING_FORCE_THRESHOLD = 5 # [N]

# Read CSV files with objects and labels tabulated
object_to_modulus = {}
object_to_shape = {}
object_to_material = {}
csv_file_path = f'./data/dataset_objects_and_compliance.csv'
with open(csv_file_path, 'r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader) # Skip title row
    for row in csv_reader:
        if row[14] != '':
            object_to_modulus[row[1]] = float(row[14])
            object_to_shape[row[1]] = row[2]
            object_to_material[row[1]] = row[3]

'''
Preprocess raw recorded data for training / evaluation...
    - Clip to the static loading sequence only
    - Create analytical estiamtes
    - Down sample frames to small number and save
'''
def preprocess_grasp(path_to_file, grasp_data=GraspData(), destination_dir=f'./data/gelsight_youngs_modulus_dataset', \
                     auto_clip=False, num_frames_to_sample=3, max_num_aug=5):
    
    object_name = os.path.basename(path_to_file)
    grasp_folder = os.path.basename(os.path.dirname(path_to_file))
    grasp = int(grasp_folder[grasp_folder.find('=')+1:])

    # Make necessary directories
    object_dir  = f'{destination_dir}/{object_name}'
    grasp_dir   = f'{object_dir}/grasp={str(grasp)}'
    
    if not os.path.exists(object_dir):
        os.mkdir(object_dir)
    if not os.path.exists(grasp_dir):
        os.mkdir(grasp_dir)
    else:
        return
    
    # Load video and forces
    grasp_data._reset_data()
    grasp_data.load(path_to_file)
    if auto_clip:
        grasp_data.auto_clip()

    # Skip files with a low peak force
    if grasp_data.forces().max() < 10: return

    # Skip files where gripper width does not change
    if grasp_data.gripper_widths().max() == grasp_data.gripper_widths().min(): return

    # Get analytical estimates
    estimator = EstimateModulus(grasp_data=grasp_data, use_gripper_width=True, use_markers_video=False)
    estimator._reset_data()
    estimator.load_from_file(path_to_file, auto_clip=False)
    estimator.grasp_data.clip_to_press(force_threshold=TRAINING_FORCE_THRESHOLD/2, pct_peak_threshold=0.9)
    
    # Clip to the point where at least one of the depths is deep compared to peak
    i_min_depth = min(np.argmax(estimator.max_depths() >= 0.075*estimator.max_depths().max()),
                        np.argmax(estimator.grasp_data.max_depths(marker_finger=True) >= 0.075*estimator.grasp_data.max_depths(marker_finger=True).max()))
    if i_min_depth > 0:
        estimator.grasp_data.clip(i_min_depth, len(estimator.forces()))

    # Remove stagnant gripper values across measurement frames
    estimator.interpolate_gripper_widths()

    # Fit estimates using respective methods
    E_hat_simple =  estimator.fit_modulus_simple()
    E_hat_hertz_MDR = estimator.fit_modulus_hertz_MDR()

    i_start = np.argmax(grasp_data.forces() >= TRAINING_FORCE_THRESHOLD) # 0.25*grasp_data.forces().max()) # FORCE_THRESHOLD)
    i_peak = np.argmax(grasp_data.forces() >= 0.975*grasp_data.forces().max()) + 1
    if (i_peak - i_start + 1) < num_frames_to_sample:
        print('Skipping!')
        return
    
    grasp_data.clip(i_start, i_peak + max_num_aug)
    force_sample_indices            = np.linspace(0, i_peak - i_start - 1, num_frames_to_sample, endpoint=True, dtype=int)
    video_sample_indices            = np.linspace(0, i_peak - i_start - 1, num_frames_to_sample, endpoint=True, dtype=int)
    video_sample_indices_markers    = np.linspace(0, i_peak - i_start - 1, num_frames_to_sample, endpoint=True, dtype=int)
    
    # Consider force when limiting augmentations
    num_aug = 1
    for i in range(1, max_num_aug):
        if grasp_data.forces()[force_sample_indices[-1] + i] >= 0.95*grasp_data.forces().max():
            num_aug += 1
        else:
            break
    assert num_aug > 0

    # Shift for latency between frames based on perceived contact
    contact_shift = -1
    for i in range(0, max_num_aug):
        if grasp_data.max_depths()[i] >= 0.075*grasp_data.max_depths().max():
            contact_shift = i
            break
    contact_shift_markers = -1
    for i in range(0, max_num_aug):
        if grasp_data.max_depths(other_finger=True)[i] >= 0.075*grasp_data.max_depths(other_finger=True).max():
            contact_shift_markers = i
            break

    contact_shift = max(0, contact_shift)
    contact_shift_markers = max(0, contact_shift_markers)

    force_sample_indices[0] += min(contact_shift, contact_shift_markers)
    video_sample_indices[0] += contact_shift
    video_sample_indices_markers[0] += contact_shift_markers
    
    if video_sample_indices[1] > video_sample_indices[0]:
        np.linspace(video_sample_indices[0], video_sample_indices[-1], num_frames_to_sample, endpoint=True, dtype=int)
    if video_sample_indices_markers[1] > video_sample_indices_markers[0]:
        np.linspace(video_sample_indices_markers[0], video_sample_indices_markers[-1], num_frames_to_sample, endpoint=True, dtype=int)

    num_aug = min(num_aug, max(video_sample_indices[1] - video_sample_indices[0], \
                               video_sample_indices_markers[1] - video_sample_indices_markers[0], 2))
    if min(video_sample_indices[-1] - video_sample_indices[0], \
           video_sample_indices_markers[-1] - video_sample_indices_markers[0]) < num_frames_to_sample:
        num_aug = 1

    for i in range(num_aug):

        # Don't continue if the depth from the first frame is similar to the final
        if i > 0 and object_to_modulus[object_name] >= 5e5 and \
            grasp_data.wedge_video.depth_images()[video_sample_indices[0] + i, :, :].max() >= \
            0.8*grasp_data.wedge_video.depth_images()[video_sample_indices[-1] + i, :, :].max() and \
            grasp_data.wedge_video_markers.depth_images()[video_sample_indices_markers[0] + i, :, :].max() >= \
            0.8*grasp_data.wedge_video_markers.depth_images()[video_sample_indices_markers[-1] + i, :, :].max():
            break

        RGB_images              = grasp_data.wedge_video.diff_images()[video_sample_indices + i, :, :]
        depth_images            = grasp_data.wedge_video.depth_images()[video_sample_indices + i, :, :]
        RGB_images_markers      = grasp_data.wedge_video_markers.diff_images()[video_sample_indices_markers + i, :, :]
        depth_images_markers    = grasp_data.wedge_video_markers.depth_images()[video_sample_indices_markers + i, :, :]
        forces                  = grasp_data.forces()[force_sample_indices + i]
        widths                  = grasp_data.gripper_widths()[force_sample_indices + i]
            
        # Make necessary directories
        aug_dir = f'{grasp_dir}/augmentation={i}'
        if not os.path.exists(aug_dir):
            os.mkdir(aug_dir)

        # Save to respective areas
        with open(f'{aug_dir}/RGB.pkl', 'wb') as file:
            pickle.dump(RGB_images, file)
        with open(f'{aug_dir}/depth.pkl', 'wb') as file:
            pickle.dump(depth_images, file)
        with open(f'{aug_dir}/RGB_markers.pkl', 'wb') as file:
            pickle.dump(RGB_images_markers, file)
        with open(f'{aug_dir}/depth_markers.pkl', 'wb') as file:
            pickle.dump(depth_images_markers, file)
        with open(f'{aug_dir}/forces.pkl', 'wb') as file:
            pickle.dump(forces, file)
        with open(f'{aug_dir}/widths.pkl', 'wb') as file:
            pickle.dump(widths, file)
        with open(f'{aug_dir}/elastic_estimate.pkl', 'wb') as file:
            pickle.dump(np.array(E_hat_simple), file)
        with open(f'{aug_dir}/hertz_estimate.pkl', 'wb') as file:
            pickle.dump(np.array(E_hat_hertz_MDR), file)

    return

if __name__ == "__main__":

    wedge_video         = GelSightWedgeVideo(config_csv="./wedge_config/config_no_markers.csv") # Force-sensing finger
    wedge_video_markers = GelSightWedgeVideo(config_csv="./wedge_config/config_markers.csv") # Marker finger
    grasp_data          = GraspData(wedge_video=wedge_video, wedge_video_markers=wedge_video_markers, \
                                    gripper_width=GripperWidth(), contact_force=ContactForce())

    N_FRAMES = 3
    DATA_DIR = f'./data/raw_data'
    DESTINATION_DIR = f'./data/gelsight_youngs_modulus_dataset'

    # Loop through all data files
    for object_name in tqdm(os.listdir(DATA_DIR)):
        for grasp_folder in os.listdir(f'{DATA_DIR}/{object_name}'):
            for file_name in os.listdir(f'{DATA_DIR}/{object_name}/{grasp_folder}'):
                if os.path.splitext(file_name)[1] != '.avi' or file_name.count("_markers") > 0:
                    continue           

                # Preprocess the raw grasp
                preprocess_grasp(f'{DATA_DIR}/{grasp_folder}/{object_name}/{os.path.splitext(file_name)[0]}', \
                                 destination_dir=DESTINATION_DIR, num_frames_to_sample=N_FRAMES, grasp_data=grasp_data
                                )