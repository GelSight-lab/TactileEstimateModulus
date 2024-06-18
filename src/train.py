import os
import math
import pickle
import csv
import json
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import wandb

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import matplotlib.pyplot as plt

from src.nn_modules import *

from torchinfo import summary
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
torch.autograd.set_detect_anomaly(True)

DATA_DIR = './data' # '/media/mike/Elements/data'
N_FRAMES = 3
WARPED_CROPPED_IMG_SIZE = (250, 350)

# Get the tree of all video files from a directory in place
def list_files(folder_path, file_paths, config):
    # Iterate through the list
    for item in os.listdir(folder_path):
        item_path = f'{folder_path}/{item}'
        marker_signal = '_markers' if config['use_markers'] else '.pkl'
        if os.path.isfile(item_path) and item.count(config['img_style']) > 0 and item.count(marker_signal) > 0:
            file_paths.append(item_path)
        elif os.path.isdir(item_path):
            list_files(item_path, file_paths, config)

class CustomDataset(Dataset):
    def __init__(self, config, paths_to_files, labels, normalization_values, \
                 validation_dataset=False,
                 frame_tensor=torch.zeros((N_FRAMES, 3, WARPED_CROPPED_IMG_SIZE[0], WARPED_CROPPED_IMG_SIZE[1])),
                 force_tensor=torch.zeros((N_FRAMES, 1)),
                 width_tensor=torch.zeros((N_FRAMES, 1)),
                 estimation_tensor=torch.zeros((2, 1)),
                 label_tensor=torch.zeros((1, 1))
        ):
        # Data parameters 
        self.data_dir               = config['data_dir']
        self.training_data_folder   = config['training_data_folder']
        self.n_frames               = config['n_frames']
        self.img_size               = config['img_size']
        self.img_style              = config['img_style']
        self.n_channels             = config['n_channels']
        self.rubber_only            = config['rubber_only']
        self.val_on_seen_objects    = config['val_on_seen_objects']
        self.use_markers            = config['use_markers']
        self.use_force              = config['use_force']
        self.use_width              = config['use_width']
        self.use_estimations         = config['use_estimations']
        self.use_transformations    = config['use_transformations']
        self.use_width_transforms   = config['use_width_transforms']
        self.exclude                = config['exclude']

        # Define training parameters
        self.epochs             = config['epochs']
        self.batch_size         = config['batch_size']
        self.img_feature_size   = config['img_feature_size']
        self.fwe_feature_size   = config['fwe_feature_size']
        self.val_pct            = config['val_pct']
        self.learning_rate      = config['learning_rate']
        self.gamma              = config['gamma']
        self.random_state       = config['random_state']

        self.validation_dataset = validation_dataset
        self.normalization_values = normalization_values
        self.input_paths = paths_to_files
        self.normalized_modulus_labels = labels

        if self.use_width_transforms:
            self.input_paths = 2*self.input_paths
            self.normalized_modulus_labels = 2*self.normalized_modulus_labels
            self.noise_force = [ i > len(self.input_paths)/2 and i % 2 == 1 for i in range(len(self.input_paths)) ]
            self.noise_width = [ i > len(self.input_paths)/2 for i in range(len(self.input_paths)) ]

        # Define attributes to use to conserve memory
        self.base_name      = ''
        self.x_frames       = frame_tensor
        self.x_forces       = force_tensor
        self.x_widths       = width_tensor
        self.x_estimations  = estimation_tensor
        self.y_label        = label_tensor
    
    def __len__(self):
        return len(self.normalized_modulus_labels)
    
    def __getitem__(self, idx):
        self.x_frames       = self.x_frames.zero_()
        self.x_forces       = self.x_forces.zero_()
        self.x_widths       = self.x_widths.zero_()
        self.x_estimations  = self.x_estimations.zero_()
        self.y_label        = self.y_label.zero_()

        folders = self.input_paths[idx].split('/')
        object_name = folders[folders.index(self.training_data_folder) + 1]

        # Read and store frames in the tensor
        with open(self.input_paths[idx], 'rb') as file:
            if self.img_style == 'depth':
                self.x_frames[:] = torch.from_numpy(pickle.load(file).astype(np.float32)).unsqueeze(3).permute(0, 3, 1, 2)
                self.x_frames /= self.normalization_values['max_depth']
            else:
                self.x_frames[:] = torch.from_numpy(pickle.load(file).astype(np.float32)).permute(0, 3, 1, 2)

        # Unpack force measurements
        self.base_name = os.path.dirname(self.input_paths[idx])
        if self.use_force:
            with open(self.base_name + '/forces.pkl', 'rb') as file:
                self.x_forces[:] = torch.from_numpy(pickle.load(file).astype(np.float32)).unsqueeze(1)
                self.x_forces /= self.normalization_values['max_force']

        # Unpack gripper width measurements
        if self.use_width:
            with open(self.base_name + '/widths.pkl', 'rb') as file:
                self.x_widths[:] = torch.from_numpy(pickle.load(file).astype(np.float32)).unsqueeze(1)
                self.x_widths[:] /= self.normalization_values['max_width']

            if self.use_width_transforms:
                if self.noise_width[idx]:
                    noise_amplitude = min(
                        1 - self.x_widths.max(),
                        self.x_widths.min()
                    )
                    if random.random() > 0.5:
                        self.x_widths += noise_amplitude * random.random()
                    else:
                        self.x_widths -= noise_amplitude * random.random()
        
        # Unpack modulus estimations
        if self.use_estimations:
            with open(self.base_name + '/elastic_estimate.pkl', 'rb') as file:
                self.x_estimations[0] = torch.from_numpy(pickle.load(file).astype(np.float32)).unsqueeze(1)
            with open(self.base_name + '/hertz_estimate.pkl', 'rb') as file:
                self.x_estimations[1] = torch.from_numpy(pickle.load(file).astype(np.float32)).unsqueeze(1)
        
        # Unpack label
        self.y_label[0] = self.normalized_modulus_labels[idx]

        return self.x_frames.clone(), self.x_forces.clone(), self.x_widths.clone(), self.x_estimations.clone(), self.y_label.clone(), object_name


class ModulusModel():
    def __init__(self, config, device=None):
        self.config = config
        self._unpack_config(config)

        # Use GPU by default
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            torch.cuda.empty_cache()
        else:
            self.device = device

        # Create max values for scaling
        self.normalization_values = { # Based on acquired data maximums
            'min_modulus': 1e3,
            'max_modulus': 1e12,
            'min_estimate': 1e2,
            'max_estimate': 1e14,
            'max_depth': 7.0,
            'max_width': 0.08, # Measured from Franka Panda gripper
            'max_force': 60.0,
        }

        # Reduce prediction range if objects are all rubber
        if self.rubber_only:
            self.normalization_values['min_modulus'] = 1e5
            self.normalization_values['max_modulus'] = 1e8

        self.x_max_cuda = torch.Tensor([self.normalization_values['max_modulus']]).to(device)
        self.x_min_cuda = torch.Tensor([self.normalization_values['min_modulus']]).to(device)

        self.video_encoder = EncoderCNN(
            img_x=self.img_size[0],
            img_y=self.img_size[1],
            input_channels=self.n_channels,
            CNN_embed_dim=self.img_feature_size,
            dropout_pct=self.dropout_pct
        )

        # Compute the size of the input to the decoder based on config
        self.decoder_input_size = self.n_frames * self.img_feature_size
        if self.use_force: 
            self.decoder_input_size += self.fwe_feature_size
        if self.use_width: 
            self.decoder_input_size += self.fwe_feature_size
        self.decoder_output_size = config['decoder_output_size'] if self.use_estimations else 1

        # Initialize force, width, estimation based on config
        self.force_encoder = ForceFC(
                                    input_dim=self.n_frames,
                                    hidden_size=self.fwe_feature_size, 
                                    output_dim=self.fwe_feature_size, 
                                    dropout_pct=self.dropout_pct
                                ) if self.use_force else None
        self.width_encoder = WidthFC(
                                    input_dim=self.n_frames,
                                    hidden_size=self.fwe_feature_size,
                                    output_dim=self.fwe_feature_size,
                                    dropout_pct=self.dropout_pct
                                ) if self.use_width else None
        self.estimation_decoder = EstimationDecoderFC(
                                    input_dim=2 + self.decoder_output_size,
                                    FC_layer_nodes=config['est_decoder_size'],
                                    output_dim=1,
                                    dropout_pct=self.dropout_pct
                                ) if self.use_estimations else None
        self.decoder = DecoderFC(input_dim=self.decoder_input_size, FC_layer_nodes=config['decoder_size'], output_dim=self.decoder_output_size, dropout_pct=self.dropout_pct)

        # Send models to device
        self.video_encoder.to(self.device)
        if self.use_force:
            self.force_encoder.to(self.device)
        if self.use_width:
            self.width_encoder.to(self.device)
        if self.use_estimations:
            self.estimation_decoder.to(self.device)
        self.decoder.to(self.device)

        # Concatenate parameters of all models
        self.params = list(self.video_encoder.parameters())
        if self.use_force: 
            self.params += list(self.force_encoder.parameters())
        if self.use_width: 
            self.params += list(self.width_encoder.parameters())
        if self.use_estimations: 
            self.params += list(self.estimation_decoder.parameters())
        self.params += list(self.decoder.parameters())
        
        # Create optimizer, use Adam
        self.optimizer      = torch.optim.Adam(self.params, lr=self.learning_rate)
        if self.gamma is not None:
            self.scheduler  = lr_scheduler.StepLR(self.optimizer, step_size=self.lr_step_size, gamma=self.gamma)

        # Normalize based on mean and std computed over the dataset
        if self.n_channels == 3:
            # Use the diff mean and std computed for our dataset
            self.image_normalization = torchvision.transforms.Normalize( \
                                            [0.49638007, 0.49770336, 0.49385751], \
                                            [0.04634926, 0.06181679, 0.07152624] \
                                        )

        # Apply random flipping transformations
        if self.use_transformations:
            self.random_transformer = torchvision.transforms.Compose([
                    torchvision.transforms.RandomHorizontalFlip(0.5),
                    torchvision.transforms.RandomVerticalFlip(0.5),
                ])

        # Data structures for training
        self.object_names = []
        self.object_to_modulus = {}

        if self.use_wandb:
            wandb.init(
                # Set the wandb project where this run will be logged
                project="TrainModulus",
                name=self.run_name,
                
                # Track hyperparameters and run metadata
                config={
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                    "n_frames": self.n_frames,
                    "n_channels": self.n_channels,
                    "img_size": self.img_size,
                    "img_style": self.img_style,
                    "img_feature_size": self.img_feature_size,
                    "fwe_feature_size": self.fwe_feature_size,
                    "learning_rate": self.learning_rate,
                    "gamma": self.gamma,
                    "lr_step_size": self.lr_step_size,
                    "validation_pct": self.val_pct,
                    "dropout_pct": self.dropout_pct,
                    "random_state": self.random_state,
                    "num_params": len(self.params),
                    "optimizer": "Adam",
                    "scheduler": "StepLR",
                    "use_markers": self.use_markers,
                    "use_force": self.use_force,
                    "use_width": self.use_width,
                    "use_estimations": self.use_estimations,
                    "use_transformations": self.use_transformations,
                    "use_width_transforms": self.use_width_transforms,
                    "exclude": self.exclude,
                }
            )
        
        # Log memory usage
        self.memory_allocated = torch.cuda.memory_allocated()
        self.memory_cached = torch.cuda.memory_reserved()
        if self.use_wandb:
            wandb.log({
                "epoch": 0,
                "memory_allocated": self.memory_allocated,
                "memory_reserved": self.memory_cached,
            })

        return
    
    def _unpack_config(self, config):
        # Data parameters 
        self.data_dir               = config['data_dir']
        self.training_data_folder   = config['training_data_folder']
        self.n_frames               = config['n_frames']
        self.img_size               = config['img_size']
        self.img_style              = config['img_style']
        self.n_channels             = config['n_channels']
        self.rubber_only            = config['rubber_only']
        self.val_on_seen_objects    = config['val_on_seen_objects']
        self.use_markers            = config['use_markers']
        self.use_force              = config['use_force']
        self.use_width              = config['use_width']
        self.use_estimations         = config['use_estimations']
        self.use_transformations    = config['use_transformations']
        self.use_width_transforms   = config['use_width_transforms']
        self.exclude                = config['exclude']
        
        self.use_wandb              = config['use_wandb']
        self.run_name               = config['run_name']

        # Define training parameters
        self.epochs             = config['epochs']
        self.batch_size         = config['batch_size']
        self.img_feature_size   = config['img_feature_size']
        self.fwe_feature_size   = config['fwe_feature_size']
        self.val_pct            = config['val_pct']
        self.dropout_pct        = config['dropout_pct']
        self.learning_rate      = config['learning_rate']
        self.gamma              = config['gamma']
        self.lr_step_size       = config['lr_step_size']
        self.random_state       = config['random_state']
        self.criterion          = nn.MSELoss()
        return
    
    # Normalize labels to maximum on log scale
    def log_normalize(self, x, x_max=None, x_min=None, use_torch=False):
        if x_max is None: x_max = self.normalization_values['max_modulus']
        if x_min is None: x_min = self.normalization_values['min_modulus']
        if use_torch:
            return (torch.log10(x) - torch.log10(self.x_min_cuda)) / (torch.log10(self.x_max_cuda) - torch.log10(self.x_min_cuda))
        return (np.log10(x) - np.log10(x_min)) / (np.log10(x_max) - np.log10(x_min))
    
    # Unnormalize labels from maximum on log scale
    def log_unnormalize(self, x_normal, x_max=None, x_min=None):
        if x_max is None: x_max = self.normalization_values['max_modulus']
        if x_min is None: x_min = self.normalization_values['min_modulus']
        return x_min * (x_max/x_min)**(x_normal)

    # Create data loaders based on configuration
    def _load_data_paths(self, labels_csv_name='dataset_objects_and_compliance.csv', csv_modulus_column=14, csv_shape_column=2, csv_material_column=3):
        # Read CSV files with objects and labels tabulated
        self.object_names = []
        self.object_to_modulus = {}
        self.object_to_material = {}
        self.object_to_shape = {}
        csv_file_path = f'{self.data_dir}/{labels_csv_name}'
        with open(csv_file_path, 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader) # Skip title row
            for row in csv_reader:
                if row[csv_modulus_column] != '' and float(row[csv_modulus_column]) > 0:
                    modulus = float(row[csv_modulus_column])
                    self.object_to_modulus[row[1]] = modulus
                    self.object_to_material[row[1]] = row[csv_material_column]
                    self.object_to_shape[row[1]] = row[csv_shape_column]

        # Extract object names as keys from data
        object_names = self.object_to_modulus.keys()
        if self.rubber_only:
            object_names = [x for x in object_names if self.object_to_material[x] == 'Rubber']
        else:
            object_names = [x for x in object_names if (x not in self.exclude)]

        # Extract corresponding elastic modulus labels for each object
        elastic_moduli = [self.object_to_modulus[x] for x in object_names]

        # Split objects into validation or training
        self.objects_train, self.objects_val, _, _ = train_test_split(object_names, elastic_moduli, test_size=self.val_pct, random_state=self.random_state)
        del object_names, elastic_moduli

        # Get all the paths to grasp data within directory
        paths_to_files = []
        list_files(f'{self.data_dir}/{self.training_data_folder}', paths_to_files, self.config)
        self.paths_to_files = paths_to_files

        # Remove those with no estimation
        if self.use_estimations:
            clean_paths_to_files = []
            for file_path in self.paths_to_files:
                if os.path.isfile(os.path.dirname(file_path) + '/hertz_estimate.pkl'):
                    clean_paths_to_files.append(file_path)
            self.paths_to_files = clean_paths_to_files

        # Remove those with no force change
        if self.use_force:
            clean_paths_to_files = []
            for file_path in self.paths_to_files:
                with open(os.path.dirname(file_path) + '/forces.pkl', 'rb') as file:
                    F = pickle.load(file)
                if F[-1] > F[0]:
                    clean_paths_to_files.append(file_path)
            self.paths_to_files = clean_paths_to_files

        # Remove those with no width change
        if self.use_width:
            clean_paths_to_files = []
            for file_path in self.paths_to_files:
                with open(os.path.dirname(file_path) + '/widths.pkl', 'rb') as file:
                    w = pickle.load(file)
                if w[-1] < w[0]:
                    clean_paths_to_files.append(file_path)
            self.paths_to_files = clean_paths_to_files

        # Remove those where depth is not monotonically increasing
        clean_paths_to_files = []
        marker_str = '_markers' if self.use_markers else ''
        for file_path in self.paths_to_files:
            with open(os.path.dirname(file_path) + f'/depth{marker_str}.pkl', 'rb') as file:
                depth = pickle.load(file)
            if depth[-1].max() > depth[-2].max():
                clean_paths_to_files.append(file_path)
        self.paths_to_files = clean_paths_to_files

        # Create data loaders based on training / validation break-up
        self._create_data_loaders()
        return
    
    def _create_data_loaders(self):
        # Divide paths up into training and validation data
        x_train, x_val = [], []
        y_train, y_val = [], []
        self.object_names = []
        for file_path in self.paths_to_files:
            folders = file_path.split('/')
            object_name = folders[folders.index(self.training_data_folder) + 1]
            if object_name in self.exclude: continue

            if object_name in self.objects_train:
                self.object_names.append(object_name)
                x_train.append(file_path)
                y_train.append(self.log_normalize(self.object_to_modulus[object_name]))

            elif object_name in self.objects_val:
                self.object_names.append(object_name)
                x_val.append(file_path)
                y_val.append(self.log_normalize(self.object_to_modulus[object_name]))

        # For seen objects, randomly shuffle
        if self.val_on_seen_objects:
            x_train, x_val, y_train, y_val = train_test_split(x_train + x_val, y_train + y_val, test_size=self.val_pct, random_state=self.random_state)
        
        # Create tensor's on device to send to dataset
        empty_frame_tensor        = torch.zeros((self.n_frames, self.n_channels, self.img_size[0], self.img_size[1]), device=self.device)
        empty_force_tensor        = torch.zeros((self.n_frames, 1), device=self.device)
        empty_width_tensor        = torch.zeros((self.n_frames, 1), device=self.device)
        empty_estimation_tensor   = torch.zeros((2, 1), device=self.device)
        empty_label_tensor        = torch.zeros((1), device=self.device)
    
        # Construct datasets
        kwargs = {'num_workers': 0, 'pin_memory': False, 'drop_last': True}
        self.train_dataset  = CustomDataset(self.config, x_train, y_train,
                                            self.normalization_values,
                                            validation_dataset=False,
                                            frame_tensor=empty_frame_tensor, 
                                            force_tensor=empty_force_tensor,
                                            width_tensor=empty_width_tensor,
                                            estimation_tensor=empty_estimation_tensor,
                                            label_tensor=empty_label_tensor)
        self.val_dataset    = CustomDataset(self.config, x_val, y_val,
                                            self.normalization_values,
                                            validation_dataset=True,
                                            frame_tensor=empty_frame_tensor, 
                                            force_tensor=empty_force_tensor,
                                            width_tensor=empty_width_tensor,
                                            estimation_tensor=empty_estimation_tensor,
                                            label_tensor=empty_label_tensor)
        self.train_loader   = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, **kwargs)
        self.val_loader     = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True, **kwargs)

        return
    
    # Forward pass on model from inputs
    def _execute(self, x_frames, x_widths, x_forces, x_estimations, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        x_frames = x_frames.view(-1, self.n_channels, self.img_size[0], self.img_size[1])

        # Normalize images
        if self.n_channels == 3:
            x_frames = self.image_normalization(x_frames)

        # Apply random transformations for training
        if self.use_transformations:
            x_frames = self.random_transformer(x_frames) # Apply V/H flips
            
        x_frames = x_frames.view(batch_size, self.n_frames, self.n_channels, self.img_size[0], self.img_size[1])

        # Concatenate features across frames into a single vector
        features = []
        for i in range(N_FRAMES):            
            # Execute CNN on video frames
            features.append(self.video_encoder(x_frames[:, i, :, :, :]))

        # Execute FC layers on other data and append
        if self.use_force: # Force measurements
            features.append(self.force_encoder(x_forces[:, :, :].squeeze(-1)))
        if self.use_width: # Width measurements
            features.append(self.width_encoder(x_widths[:, :, :].squeeze(-1)))

        # Send aggregated features to the FC decoder
        features = torch.cat(features, -1)
        outputs = self.decoder(features)

        # Send to decoder with deterministic estimations
        if self.use_estimations:
            x_estimations = torch.clamp(x_estimations, min=self.normalization_values['min_estimate'], max=self.normalization_values['max_estimate'])
            x_estimations = self.log_normalize(x_estimations, x_max=self.normalization_values['max_estimate'], x_min=self.normalization_values['min_estimate'], use_torch=True)
            outputs = self.estimation_decoder(torch.cat([outputs, x_estimations.squeeze(-1)], -1))

        return outputs

    def _train_epoch(self, train_loader=None):
        self.video_encoder.train()
        if self.use_force:
            self.force_encoder.train()
        if self.use_width:
            self.width_encoder.train()
        if self.use_estimations:
            self.estimation_decoder.train()
        self.decoder.train()

        train_stats = {
            'loss': 0,
            'log_acc': 0,
            'avg_log_diff': 0,
            'pct_w_100_factor_err': 0,
            'batch_count': 0,
        }
        if train_loader is None: train_loader = self.train_loader
        for batch_data in train_loader:
            self.optimizer.zero_grad()

            # Unpack data
            x_frames, x_forces, x_widths, x_estimations, y, object_names = batch_data

            # Forward model
            outputs = self._execute(x_frames, x_forces, x_widths, x_estimations)
           
            loss = self.criterion(outputs.squeeze(1), y.squeeze(1))
            
            # Add regularization to loss
            l2_reg = torch.tensor(0., device=self.device)
            for param in self.params:
                l2_reg += torch.norm(param)
            loss += 0.00005 * l2_reg

            loss.backward()
            self.optimizer.step()

            train_stats['loss'] += loss.item()
            train_stats['batch_count'] += 1

            # Calculate performance metrics
            abs_log_diff = torch.abs(torch.log10(self.log_unnormalize(outputs.cpu())) - torch.log10(self.log_unnormalize(y.cpu()))).detach().numpy()
            train_stats['avg_log_diff'] += abs_log_diff.sum()
            for i in range(self.batch_size):
                if abs_log_diff[i] <= 1:
                    train_stats['log_acc'] += 1
                if abs_log_diff[i] >= 2:
                    train_stats['avg_log_diff'] += 1
                    
        # Return loss
        train_stats['loss']                     /= train_stats['batch_count']
        train_stats['log_acc']                  /= (self.batch_size * train_stats['batch_count'])
        train_stats['avg_log_diff']             /= (self.batch_size * train_stats['batch_count'])
        train_stats['pct_w_100_factor_err']     /= (self.batch_size * train_stats['batch_count'])

        return train_stats

    def _val_epoch(self, track_predictions=False):
        self.video_encoder.eval()
        if self.use_force:
            self.force_encoder.eval()
        if self.use_width:
            self.width_encoder.eval()
        if self.use_estimations:
            self.estimation_decoder.eval()
        self.decoder.eval()

        if track_predictions:
            predictions = { obj : [] for obj in self.objects_val }

        val_stats = {
            'loss': 0,
            'log_acc': 0,
            'soft_log_acc': 0,
            'hard_log_acc': 0,
            'avg_log_diff': 0,
            'soft_avg_log_diff': 0,
            'hard_avg_log_diff': 0,
            'pct_w_100_factor_err': 0,
            'pct_w_1000_factor_err': 0,
            'soft_count': 0,
            'hard_count': 0,
            'batch_count': 0,
        }
        for batch_data in self.val_loader:

            # Unpack data
            x_frames, x_forces, x_widths, x_estimations, y, object_names = batch_data

            # Forward model
            outputs = self._execute(x_frames, x_forces, x_widths, x_estimations)                

            loss = self.criterion(outputs.squeeze(1), y.squeeze(1))
            
            # Add regularization to loss
            l2_reg = torch.tensor(0., device=self.device)
            for param in self.params:
                l2_reg += torch.norm(param)
            loss += 0.00005 * l2_reg

            val_stats['loss'] += loss.item()
            val_stats['batch_count'] += 1

            # Calculate performance metrics
            abs_log_diff = torch.abs(torch.log10(self.log_unnormalize(outputs.cpu())) - torch.log10(self.log_unnormalize(y.cpu()))).detach().numpy()
            val_stats['avg_log_diff'] += abs_log_diff.sum()
            for i in range(self.batch_size):
                if self.object_to_modulus[object_names[i]] < 1e8:
                    val_stats['soft_avg_log_diff'] += abs_log_diff[i]
                else:
                    val_stats['hard_avg_log_diff'] += abs_log_diff[i]

                if abs_log_diff[i] <= 1:
                    val_stats['log_acc'] += 1
                    if self.object_to_modulus[object_names[i]] < 1e8:
                        val_stats['soft_log_acc'] += 1
                    else:
                        val_stats['hard_log_acc'] += 1

                if abs_log_diff[i] >= 2:
                    val_stats['pct_w_100_factor_err'] += 1

                if abs_log_diff[i] >= 3:
                    val_stats['pct_w_1000_factor_err'] += 1

                if self.object_to_modulus[object_names[i]] < 1e8: val_stats['soft_count'] += 1
                else: val_stats['hard_count'] += 1

                if track_predictions:
                    predictions[object_names[i]].append(self.log_unnormalize(outputs[i][0].cpu()).detach().numpy())
            
        # Return loss and accuracy
        val_stats['loss']                   /= val_stats['batch_count']
        val_stats['log_acc']                /= (self.batch_size * val_stats['batch_count'])
        val_stats['avg_log_diff']           /= (self.batch_size * val_stats['batch_count'])
        val_stats['pct_w_100_factor_err']   /= (self.batch_size * val_stats['batch_count'])
        val_stats['pct_w_1000_factor_err']  /= (self.batch_size * val_stats['batch_count'])
        val_stats['soft_log_acc']           /= val_stats['soft_count']
        val_stats['soft_avg_log_diff']      /= val_stats['soft_count']
        if val_stats['hard_count'] > 0:
            val_stats['hard_log_acc']           /= val_stats['hard_count']
            val_stats['hard_avg_log_diff']      /= val_stats['hard_count']

        if track_predictions:
            return predictions, val_stats
        else:
            return val_stats

    def train(self):
        learning_rate = self.learning_rate 
        min_val_loss = 1e10

        # Load data
        self._load_data_paths()        

        for epoch in range(self.epochs):

            # Train batch
            train_stats = self._train_epoch()

            # Validation statistics
            val_stats = self._val_epoch()

            # Increment learning rate
            for param_group in self.optimizer.param_groups:
                learning_rate = param_group['lr']
            if self.gamma is not None:
                self.scheduler.step()

            # Save the best model based on validation loss and accuracy
            if val_stats['loss'] <= min_val_loss:
                min_val_loss = val_stats['loss']
                self.save_model()

            # Log information to W&B
            if self.use_wandb:
                self.memory_allocated = torch.cuda.memory_allocated()
                self.memory_cached = torch.cuda.memory_reserved()
                wandb.log({
                    "epoch": epoch,
                    "learning_rate": learning_rate,
                    "memory_allocated": self.memory_allocated,
                    "memory_reserved": self.memory_cached,
                    "train_loss": train_stats['loss'],
                    "train_avg_log_diff": train_stats['avg_log_diff'],
                    "train_log_accuracy": train_stats['log_acc'],
                    "train_pct_with_100_factor_err": train_stats['pct_w_100_factor_err'],
                    "val_loss": val_stats['loss'],
                    "val_avg_log_diff": val_stats['avg_log_diff'],
                    "val_log_accuracy": val_stats['log_acc'],
                    "val_pct_with_100_factor_err": val_stats['pct_w_100_factor_err'],
                    "val_pct_with_1000_factor_err": val_stats['pct_w_1000_factor_err'],
                    "val_soft_avg_log_diff": val_stats['soft_avg_log_diff'],
                    "val_soft_log_acc": val_stats['soft_log_acc'],
                    "val_hard_avg_log_diff": val_stats['hard_avg_log_diff'],
                    "val_hard_log_acc": val_stats['hard_log_acc'],
                })

        if self.use_wandb: wandb.finish()

        return

    # Given grasp inputs, execute estimate for Young's Modulus
    def estimate(self, tactile_frames, forces=None, widths=None, E_hat_simple=None, E_hat_hertz=None):
        if forces is not None:
            assert self.use_force
        if widths is not None:
            assert self.use_width
        if E_hat_hertz is not None or E_hat_simple is not None:
            assert self.use_estimations
            assert E_hat_hertz is not None and E_hat_simple is not None

        estimations = torch.zeros((2, 1), device=self.device)
        estimations[0] = E_hat_simple
        estimations[1] = E_hat_hertz

        # Convert to tensor
        if isinstance(tactile_frames, np.ndarray):
            tactile_frames = torch.tensor(tactile_frames.astype(np.float32), device=self.device)
        if isinstance(forces, np.ndarray):
            forces = torch.tensor(forces.astype(np.float32), device=self.device)
        if isinstance(widths, np.ndarray):
            widths = torch.tensor(widths.astype(np.float32), device=self.device)

        # Add batch dimension
        tactile_frames = tactile_frames.unsqueeze(0)
        forces = forces.unsqueeze(0)
        widths = widths.unsqueeze(0)
        estimations = estimations.unsqueeze(0)

        # Forward model
        output = self._execute(tactile_frames, forces, widths, estimations, batch_size=1)
        
        # Unnormalize the output into Pascals
        E_hat = self.log_unnormalize(output)

        return E_hat
    
    def save_model(self):
        model_save_dir = f'./model/{self.run_name}'
        if not os.path.exists(model_save_dir):
            os.mkdir(model_save_dir)
        else:
            [os.remove(f'{model_save_dir}/{filename}') for filename in os.listdir(model_save_dir)]

        # Save configuration dictionary and all files for the model(s)
        with open(f'{model_save_dir}/config.json', 'w') as json_file:
            json.dump(self.config, json_file)
        torch.save(self.video_encoder.state_dict(), f'{model_save_dir}/video_encoder.pth')
        if self.use_force: 
            torch.save(self.force_encoder.state_dict(), f'{model_save_dir}/force_encoder.pth')
        if self.use_width: 
            torch.save(self.width_encoder.state_dict(), f'{model_save_dir}/width_encoder.pth')
        if self.use_estimations: 
            torch.save(self.estimation_decoder.state_dict(), f'{model_save_dir}/estimation_decoder.pth')
        torch.save(self.decoder.state_dict(), f'{model_save_dir}/decoder.pth')

        return
    
    def load_model(self, folder_path):
        with open(f'{folder_path}/config.json', 'r') as file:
            config = json.load(file)
        self._unpack_config(config)

        self.video_encoder.load_state_dict(torch.load(f'{folder_path}/video_encoder.pth', map_location=self.device))
        if self.use_force: 
            self.force_encoder.load_state_dict(torch.load(f'{folder_path}/force_encoder.pth', map_location=self.device))
        if self.use_width: 
            self.width_encoder.load_state_dict(torch.load(f'{folder_path}/width_encoder.pth', map_location=self.device))
        if self.use_estimations: 
            self.estimation_decoder.load_state_dict(torch.load(f'{folder_path}/estimation_decoder.pth', map_location=self.device))
        self.decoder.load_state_dict(torch.load(f'{folder_path}/decoder.pth', map_location=self.device))

        return

    def make_performance_plot(self, plot=False, return_stats=False):
        material_to_color = {
            'Foam': 'firebrick',
            'Plastic': 'forestgreen',
            'Wood': 'goldenrod',
            'Paper': 'yellow',
            'Glass': 'darkgray',
            'Ceramic': 'pink',
            'Rubber': 'slateblue',
            'Metal': 'royalblue',
            'Food': 'darkorange',
        }
        material_prediction_data = {
            mat : [] for mat in material_to_color.keys()
        }
        material_label_data = {
            mat : [] for mat in material_to_color.keys()
        }

        # Run validation epoch to get out all predictions
        predictions, val_stats = self._val_epoch(track_predictions=True)
        log_diff_predictions = { key:[] for key in predictions.keys() }

        # Turn predictions into plotting data
        count = 0
        log_diff_count = 0
        log_acc_count = 0
        outlier_count = 0
        for obj in predictions.keys():
            if len(predictions[obj]) == 0: continue
            if obj in self.exclude: continue
            assert obj in self.object_to_material.keys()
            mat = self.object_to_material[obj]
            shape = self.object_to_shape[obj]
            for E in predictions[obj]:
                log_diff = np.log10(E) - np.log10(self.object_to_modulus[obj])
                log_diff_predictions[obj].append(log_diff)

                if E > 0 and not math.isnan(E):
                    assert not math.isnan(E)
                    
                    log_diff = abs(np.log10(E) - np.log10(self.object_to_modulus[obj]))
                    log_diff_count += log_diff
                    if log_diff <= 1:
                        log_acc_count += 1
                    if log_diff >= 2:
                        outlier_count += 1

                    material_prediction_data[mat].append(float(E))
                    material_label_data[mat].append(float(self.object_to_modulus[obj]))
                    count += 1

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.float32):
                    return float(obj)
                return json.JSONEncoder.default(self, obj)
    
        if not os.path.exists(f'./plotting_data/{self.run_name}'):
            os.mkdir(f'./plotting_data/{self.run_name}')
        with open(f'./plotting_data/{self.run_name}/obj_log_diff.json', 'w') as json_file:
            json.dump(log_diff_predictions, json_file, indent=4, sort_keys=True, cls=NumpyEncoder)
        with open(f'./plotting_data/{self.run_name}/predictions.json', 'w') as json_file:
            json.dump(material_prediction_data, json_file, cls=NumpyEncoder)
        with open(f'./plotting_data/{self.run_name}/labels.json', 'w') as json_file:
            json.dump(material_label_data, json_file, cls=NumpyEncoder)

        if plot:
            # Create plot
            mpl.rcParams['font.family'] = ['serif']
            mpl.rcParams['font.serif'] = ['Times New Roman']
            plt.figure()
            plt.plot([100, 10**12], [100, 10**12], 'k--', label='_')
            plt.fill_between([100, 10**12], [10**1, 10**11], [10**3, 10**13], color='gray', alpha=0.2)
            plt.xscale('log')
            plt.yscale('log')

            for mat in material_to_color.keys():
                plt.plot(material_label_data[mat], material_prediction_data[mat], '.', markersize=10, color=material_to_color[mat], label=mat)

            plt.xlabel("Ground Truth Modulus ($E$) [$\\frac{N}{m^2}$]", fontsize=12)
            plt.ylabel("Predicted Modulus ($\\tilde{E}$) [$\\frac{N}{m^2}$]", fontsize=12)
            plt.xlim([100, 10**12])
            plt.ylim([100, 10**12])
            plt.title('Neural Network', fontsize=14)

            plt.legend()
            plt.grid(True, which='both', linestyle='--', linewidth=0.25)
            plt.tick_params(axis='both', which='both', labelsize=10)

            plt.savefig('./figures/nn.png')
            plt.show()

        if return_stats:                    
            return {
                'loss': val_stats['loss'],
                'log_diff': log_diff_count / count,
                'log_acc': log_acc_count / count,
                'pct_outliers': outlier_count / count,
            }

        return


if __name__ == "__main__":

    # Training and model settings
    config = {
        # Data parameters
        'data_dir': DATA_DIR,                                       # Data location
        'training_data_folder': 'gelsight_youngs_modulus_dataset',  # Subfolder data location
        'n_frames': N_FRAMES,                                       # Number of tactile frames as input
        'img_size': WARPED_CROPPED_IMG_SIZE,    # Input size of tactile images
        'img_style': 'RGB',                     # Use RGB difference images or tactile depth as input
        'rubber_only': False,                   # Only consider rubber objects
        'val_on_seen_objects': False,           # Validate over seen objects, else only unseen
        'use_markers': False,                   # Use tactile images from sensor with markers
        'use_force': True,                      # Use grasping force measurements as an input
        'use_width': True,                      # Use grasping width measurements as an input
        'use_estimations': True,                # Use analytical estimations as an input
        'use_transformations': False,           # Transform images during training
        'use_width_transforms': True,           # Transform width measurements during training
        'exclude': [
                    'playdoh', 'silly_puty', 'blue_sponge_dry', 'blue_sponge_wet',
                    'apple', 'orange', 'strawberry', 'ripe_banana', 'unripe_banana',
                    'lacrosse_ball', 'baseball', 'racquet_ball', 'tennis_ball', 
                ], # Exclude objects without known ground truth

        # Logging on/off
        'use_wandb': True,                      # Report data to Weights & Biases
        'run_name': 'test_run',                 # Name of W&B run

        # Training and model parameters
        'epochs'                : 80,
        'batch_size'            : 32,
        'img_feature_size'      : 128,
        'fwe_feature_size'      : 32,
        'val_pct'               : 0.175,
        'dropout_pct'           : 0.3,
        'learning_rate'         : 3e-6,
        'gamma'                 : 0.975,
        'lr_step_size'          : 1,
        'random_state'          : 27,
        'decoder_size'          : [512, 512, 128],
        'est_decoder_size'      : [64, 64, 32],
        'decoder_output_size'   : 3,
    }
    assert config['img_style'] in ['RGB', 'depth']
    config['n_channels'] = 3 if config['img_style'] == 'RGB' else 1
    
    # Train the model
    train_modulus = ModulusModel(config, device=device)
    train_modulus.train()