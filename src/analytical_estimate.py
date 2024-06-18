import os
import cv2
import math
import colorsys
import random
import webcolors
import numpy as np
import warnings
import matplotlib.pyplot as plt

from src.wedge_video import GelSightWedgeVideo, DEPTH_THRESHOLD, WARPED_IMG_SIZE
from src.contact_force import ContactForce, FORCE_THRESHOLD
from src.gripper_width import GripperWidth
from src.grasp_data import GraspData

grasp_data = GraspData()

# Archived measurements from calipers on surface in x and y directions
WARPED_PX_TO_MM = (11, 11)
RAW_PX_TO_MM = (12.5, 11)

# Use sensor size and warped data to choose conversion
SENSOR_PAD_DIM_MM = (35, 25.5) # [mm]
PX_TO_MM = np.sqrt((WARPED_IMG_SIZE[0] / SENSOR_PAD_DIM_MM[0])**2 + (WARPED_IMG_SIZE[1] / SENSOR_PAD_DIM_MM[1])**2)
MM_TO_PX = 1/PX_TO_MM

# Fit an ellipse bounding the True space of a 2D binary array
def fit_ellipse_from_binary(binary_array, plot_result=False):
    # Find contours in the binary array
    binary_array_uint8 = binary_array.astype(np.uint8)
    contours, _ = cv2.findContours(binary_array_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError('No ellipse found!')

    # Iterate through contours
    max_ellipse_area = 0
    for contour in contours:
        if contour.shape[0] < 5: continue
        # Fit ellipse to the contour
        ellipse = cv2.fitEllipse(contour)

        # Calculate the area of the fitted ellipse
        ellipse_area = (np.pi * ellipse[1][0] * ellipse[1][1]) / 4

        # Check if the ellipse area is above the minimum threshold
        if ellipse_area > max_ellipse_area:
            max_ellipse_area = ellipse_area
            max_ellipse = ellipse

    if plot_result:
        # Draw the ellipse on a blank image for visualization
        ellipse_image = np.zeros_like(binary_array, dtype=np.uint8)
        cv2.ellipse(ellipse_image, max_ellipse, 255, 1)

        # Display the results
        cv2.imshow("Original Binary Array", (binary_array * 255).astype(np.uint8))
        cv2.imshow("Ellipse Fitted", ellipse_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if max_ellipse_area == 0: return None

    return max_ellipse

# Fit an ellipse bounding the True space of a 2D non-binary array
def fit_ellipse_from_float(float_array, plot_result=False):
    # Normalize array
    float_array_normalized = (float_array - float_array.min()) / (float_array.max() - float_array.min())

    # Threshold into binary array based on range
    binary_array = (255 * (float_array_normalized >= 0.5)).astype(np.uint8)

    # Fit to ellipse
    max_ellipse = fit_ellipse_from_binary(binary_array, plot_result=False)

    if plot_result:
        # Draw the ellipse on a blank image for visualization
        ellipse_image = np.zeros_like(float_array, dtype=np.uint8)
        cv2.ellipse(ellipse_image, max_ellipse, 255, 1)

        # Display the results
        cv2.imshow("Normalized Array", float_array_normalized)
        cv2.imshow("Binary Array", binary_array)
        cv2.imshow("Ellipse Fitted", ellipse_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return max_ellipse

# Random shades for consistent plotting over multiple grasps
def random_shade_of_color(color_name):
    try:
        rgb = webcolors.name_to_rgb(color_name)
        hls = colorsys.rgb_to_hls(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)

        # Randomize the lightness while keeping hue and saturation constant
        lightness = random.uniform(0.5, 1.0)
        rgb_shaded = colorsys.hls_to_rgb(hls[0], lightness, hls[2])
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(rgb_shaded[0] * 255),
            int(rgb_shaded[1] * 255),
            int(rgb_shaded[2] * 255)
        )
        return hex_color

    except ValueError:
        raise ValueError("Invalid color name")

class EstimateModulus():
    def __init__(
            self, grasp_data=None, depth_threshold=0.001*DEPTH_THRESHOLD, force_threshold=FORCE_THRESHOLD, 
            assumed_poissons_ratio=0.45, use_gripper_width=True, use_markers_video=False
        ):

        self.assumed_poisson_ratio = assumed_poissons_ratio # [\]
        self.depth_threshold = depth_threshold # [m]
        self.force_threshold = force_threshold # [N]
        self.contact_area_threshold = 1e-5 # [m^2]

        self.use_gripper_width = use_gripper_width # Boolean of whether or not to include gripper width
        self.use_markers_video = use_markers_video # Boolean of whether or not to use video recorded from the other finger

        if grasp_data is None:
            grasp_data = GraspData(use_gripper_width=self.use_gripper_width)
        self.grasp_data = grasp_data
        if self.use_markers_video:
            assert self.grasp_data.wedge_video_markers is not None

        # Gel material: Silicone XP-565
        # Datasheet:
        #     "https://static1.squarespace.com/static/5b1ecdd4372b96e84200cf1d/t/5b608e1a758d46a989c0bf93/1533054492488/XP-565+%281%29.pdf"
        # Estimated from Shore 00 hardness:
        #     "https://www.dow.com/content/dam/dcc/documents/en-us/tech-art/11/11-37/11-3716-01-durometer-hardness-for-silicones.pdf"
        self.shore_00_hardness = 60 # From durometer measurement
        self.E_gel = 150000 # N/m^2
        self.nu_gel = 0.485 # [\]
        self.gel_width = 0.035 # [m]
        self.gel_depth = 0.0055 # [m] (slightly adjusted because of depth inaccuracies)

        self._F = []                 # Contact forces for fitting
        self._d = []                 # Contact depth for fitting
        self._a = []                 # Contact radius for fitting
        self._R = []                 # Estimated radius of object
        self._contact_areas = []     # Contact areas for fitting
        self._x_data = []            # Save fitting data for plotting
        self._y_data = []            # Save fitting data for plotting

    # Clear out the data values
    def _reset_data(self):
        self.grasp_data._reset_data()
        self._F = []
        self._d = []
        self._a = []
        self._R = []
        self._contact_areas = []
        self._x_data = []
        self._y_data = []

    # Load data from a file
    def load_from_file(self, path_to_file, auto_clip=True):
        self._reset_data()
        self.grasp_data.load(path_to_file)
        if auto_clip: # Clip to the entire press
            self.grasp_data.auto_clip()
        assert len(self.grasp_data.depth_images(marker_finger=self.use_markers_video)) == len(self.grasp_data.forces())
        return

    # Return forces
    def forces(self):
        return self.grasp_data.forces()
    
    # Return gripper widths
    def gripper_widths(self):
        assert self.use_gripper_width
        return self.grasp_data.gripper_widths()
    
    # Return depth images (in meters)
    def depth_images(self, marker_finger=None):
        if marker_finger is None: marker_finger = self.use_markers_video
        return 0.001 * self.grasp_data.depth_images(marker_finger=marker_finger)
    
    # Return maximum value from each depth image (in meters)
    def max_depths(self, depth_images=None):
        if depth_images is None: depth_images = self.depth_images()
        return np.max(depth_images, axis=(1,2))
    
    # Return mean value from each depth image (in meters)
    def mean_depths(self, depth_images=None):
        if depth_images is None: depth_images = self.depth_images()
        return np.mean(depth_images, axis=(1,2))
    
    # Return mean of neighborhood around max value from each depth image (in meters)
    def mean_max_depths(self, depth_images=None, kernel_radius=5):
        if depth_images is None: depth_images = self.depth_images()
        mean_max_depths = []
        for i in range(len(depth_images)):
            max_index = np.argmax(depth_images[i])
            r, c = np.unravel_index(max_index, depth_images[i].shape)
            mean_max_depth = depth_images[i, r-kernel_radius:r+kernel_radius, c-kernel_radius:c+kernel_radius].mean()
            mean_max_depths.append(mean_max_depth)
        return np.array(mean_max_depths)
    
    # Return highest percentile of depth population (in meters)
    def top_percentile_depths(self, depth_images=None, percentile=97):
        if depth_images is None: depth_images = self.depth_images()
        top_percentile_depths = []
        for i in range(len(depth_images)):
            top_percentile_depths.append(np.percentile(depth_images[i,:,:], percentile))
        return np.array(top_percentile_depths)
    
    # Use a user-specified contact mask
    def input_peak_depth_method(self, depth_method_str, depth_images=None):
        if depth_images is None: depth_images = self.depth_images()
        peak_depth_functions = {
            'top_percentile_depths': self.top_percentile_depths,
            'mean_max_depths': self.mean_max_depths,
            'max_depths': self.max_depths,
            'mean_depths': self.mean_depths,
        }
        assert depth_method_str in peak_depth_functions.keys()
        return peak_depth_functions[depth_method_str](depth_images=depth_images)
    
    # Return mask of which pixels are in contact with object based on constant threshold
    def constant_threshold_contact_mask(self, depth, depth_threshold=None):
        if depth_threshold is None:
            depth_threshold = self.depth_threshold
        return depth >= depth_threshold

    # Return mask of which pixels are in contact with object based on mean of image
    def mean_threshold_contact_mask(self, depth):
        return depth >= depth.mean()
    
    # Return mask of which pixels are in contact with object based on mean of all images
    def total_mean_threshold_contact_mask(self, depth):
        return depth >= self.mean_depths().mean()

    # Return mask of which pixels are in upper half of depth range
    def std_above_mean_contact_mask(self, depth):
        return depth >= depth.mean() + np.std(depth)

    # Return mask of which pixels are in upper half of depth range
    def normalized_threshold_contact_mask(self, depth, threshold_pct=0.5):
        return (depth - depth.min()) / (depth.max() - depth.min()) >= threshold_pct

    # Return mask of which pixels are in upper half of depth range for whole video
    def total_normalized_threshold_contact_mask(self, depth, threshold_pct=0.1):
        total_min_depth = 0 
        if self.max_depths().max() < 0.00075: np.min(self.depth_images(), axis=(1,2)).min()
        return (depth - total_min_depth) / (self.max_depths().max() - total_min_depth) >= threshold_pct
    
    # Threshold and fit depth to an ellipse
    def ellipse_contact_mask(self, depth):
        ellipse = fit_ellipse_from_float(depth)
        ellipse_mask = np.zeros_like(depth, dtype=np.uint8)
        if ellipse is None:
            return ellipse_mask
        cv2.ellipse(ellipse_mask, ellipse, 1, -1)
        return ellipse_mask
    
    # Use regular threshold for whole video unless mean depth of peak is small
    def total_conditional_contact_mask(self, depth, depth_threshold=1e-4, threshold_pct=0.1):
        if self.max_depths().max() > 0.001: # = depth_threshold / threshold_pct
            mask = self.constant_threshold_contact_mask(depth, depth_threshold=depth_threshold)
        else:
            mask = self.total_normalized_threshold_contact_mask(depth, threshold_pct=threshold_pct)
        return mask
    
    # Wrap the chosen contact mask function into one place
    def contact_mask(self, depth, contact_mask=None):
        if contact_mask is not None:
            return self.input_contact_mask(contact_mask, depth)
        return self.constant_threshold_contact_mask(depth) # Default masking method
    
    # Use a user-specified contact mask
    def input_contact_mask(self, contact_mask_str, depth):
        contact_mask_functions = {
            'constant_threshold_contact_mask': self.constant_threshold_contact_mask,
            'mean_threshold_contact_mask': self.mean_threshold_contact_mask,
            'total_mean_threshold_contact_mask': self.total_mean_threshold_contact_mask,
            'std_above_mean_contact_mask': self.std_above_mean_contact_mask,
            'normalized_threshold_contact_mask': self.normalized_threshold_contact_mask,
            'total_normalized_threshold_contact_mask': self.total_normalized_threshold_contact_mask,
            'ellipse_contact_mask': self.ellipse_contact_mask,
            'total_conditional_contact_mask': self.total_conditional_contact_mask,
        }
        assert contact_mask_str in contact_mask_functions.keys()
        return contact_mask_functions[contact_mask_str](depth)

    # Fit linear equation with least squares
    def linear_coeff_fit(self, x, y):
        # Solve for best A given data and equation of form y = A*x
        return np.dot(x, y) / np.dot(x, x)
    
    # Clip a press sequence to only the loading sequence (positive force)
    def clip_to_press(self, use_force=True):
        if use_force:
            # Clip from initial to peak force
            self.grasp_data.clip_to_press(force_threshold=FORCE_THRESHOLD)
        else:
            # Find peak and start over depth values
            i_start = np.argmax(self.max_depths() >= self.depth_threshold)
            i_peak = np.argmax(self.max_depths())
            i_end = i_peak

            if i_start >= i_end:
                warnings.warn("No press detected! Cannot clip.", Warning)
            else:
                # Clip from start to peak depth
                self.grasp_data.clip(i_start, i_end+1)
        return
    
    # Return the gripper width at first contact, assuming already clipped to press
    def length_of_first_contact(self, depth_images=None):
        if depth_images is None: depth_images = self.depth_images(0)
        top_percentile_depths = self.top_percentile_depths(depth_images=depth_images)
        if depth_images.max() < 5*self.depth_threshold:
            return self.gripper_widths()[0]
        else:
            i_contact = np.argmax(top_percentile_depths >= self.depth_threshold)
            return self.gripper_widths()[i_contact] + 2*top_percentile_depths[i_contact]

    # Linearly interpolate gripper widths wherever measurements are equal
    def interpolate_gripper_widths(self, plot_result=False):
        self.grasp_data.interpolate_gripper_widths(plot_result=plot_result)
        return
    
    # Preprocess depth by taking mean over square kernels
    # Blurring depth in this way may help reduce geometric noise
    def lower_resolution_depth(self, depth_images=None, kernel_size=10):
        if depth_images is None:
            depth_images = self.depth_images()
        assert depth_images.shape[1] % kernel_size == depth_images.shape[2] % kernel_size == 0
        blurred_depth_images = []
        for i in range(depth_images.shape[0]):
            reshaped_depth_image = depth_images[i].reshape(depth_images.shape[1] // kernel_size, kernel_size, \
                                                           depth_images.shape[2] // kernel_size, kernel_size)
            blurred_depth_images.append(reshaped_depth_image.mean(axis=(1, 3)))

        # Extend such that the array becomes the same size again
        blurred_depth_images = np.kron(np.array(blurred_depth_images), np.ones((kernel_size, kernel_size)))

        return blurred_depth_images
    
    # Convert depth image to 3D data
    def depth_to_XYZ(self, depth, remove_zeros=True, remove_outliers=True):
        # Extract data
        X, Y, Z = [], [], []
        for i in range(depth.shape[0]):
            for j in range(depth.shape[1]):
                if (not remove_zeros) or (remove_zeros and depth[i][j] >= 1e-10):
                    X.append(0.001 * i / PX_TO_MM) # Convert pixels to meters
                    Y.append(0.001 * j / PX_TO_MM)
                    Z.append(depth[i][j])
        data = np.vstack((np.array(X), np.array(Y), np.array(Z)))   
        
        # Remove outliers via ~3-sigma rule
        if remove_outliers == True:
            X, Y, Z = [], [], []
            centroid = np.mean(data, axis=1)
            sigma = np.std(data, axis=1)
            for i in range(data.shape[1]):
                point = data[:, i]
                if (np.abs(centroid - point) <= 3*sigma).all():
                    X.append(point[0])
                    Y.append(point[1])
                    Z.append(point[2])

        return np.array(X), np.array(Y), np.array(Z)

    # Fit depth points to a sphere in 3D space to get contact depth and radius
    def fit_depth_to_sphere(self, depth, min_datapoints=10):
        '''
        Modified approach from: https://jekel.me/2015/Least-Squares-Sphere-Fit/
        '''
        # Extract data
        X, Y, Z = self.depth_to_XYZ(depth)
        if X.shape[0] < min_datapoints:
            return [0, 0, 0, 0]

        # Construct data matrix
        A = np.zeros((len(X),4))
        A[:,0] = X*2
        A[:,1] = Y*2
        A[:,2] = Z*2
        A[:,3] = 1

        # Assemble the f matrix
        f = np.zeros((len(X),1))
        f[:,0] = (X*X) + (Y*Y) + (Z*Z)

        # Solve least squares for sphere
        C, res, rank, s = np.linalg.lstsq(A, f, rcond=None)

        # Solve for the radius
        radius = np.sqrt((C[0]*C[0]) + (C[1]*C[1]) + (C[2]*C[2]) + C[3])

        return [radius, C[0], C[1], C[2]] # [ radius, center_x, center_y, center_z ]
    
    # Compute stiffness the old fashioned way, without tactile sensing
    def fit_modulus_no_tactile(self):
        L0 = self.gripper_widths()[0]
        A = (0.001 / PX_TO_MM)**2 * (self.depth_images()[0].shape[0]*self.depth_images()[0].shape[1]) # m^2

        x_data = []
        y_data = []
        for i in range(len(self.forces())):
            dL = L0 - self.gripper_widths()[i]
            x_data.append(dL / L0)
            y_data.append(self.forces()[i] / A)

        self._x_data = np.array(x_data)
        self._y_data = np.array(y_data)
        E = self.linear_coeff_fit(x_data, y_data)

        return E
    
    # Estimate modulus based on gripper width change
    # Modification on simple Hookean estimate using tactile data
    def fit_modulus_simple(self, contact_mask=None, depth_method=None, use_mean=False, use_ellipse_mask=False, \
                          fit_mask_to_ellipse=False, use_lower_resolution_depth=False,
                        ):
        assert self.use_gripper_width
        assert not (use_ellipse_mask and fit_mask_to_ellipse)

        # Find initial length of first contact
        L0 = self.length_of_first_contact()

        if use_lower_resolution_depth:
            depth_images = self.lower_resolution_depth(kernel_size=5)
        else:
            depth_images = self.depth_images()
        
        # Precompute peak depths
        if depth_method is None:
            peak_depths = self.top_percentile_depths(depth_images=depth_images)
        else:
            peak_depths = self.input_peak_depth_method(depth_method, depth_images=depth_images)

        contact_areas, a, F = [], [], []
        x_data, y_data, d = [], [], []
        for i in range(len(depth_images)):
            depth_i = depth_images[i]

            # Skip images without significant contact
            if depth_i.max() < 0.075*peak_depths.max(): continue
            
            if use_ellipse_mask:
                # Compute mask using ellipse fit
                mask = self.ellipse_contact_mask(depth_i)
                assert contact_mask is None
            else:
                # Compute mask using traditional thresholding alone
                mask = self.contact_mask(depth_images[i], contact_mask=contact_mask)
                if mask.max() == 0: continue

                if fit_mask_to_ellipse:
                    # Fit an ellipse to the mask and modify
                    ellipse = fit_ellipse_from_binary(mask)
                    if ellipse is None: continue
                    mask = np.zeros_like(mask, dtype=np.uint8)
                    cv2.ellipse(mask, ellipse, 1, -1)

            if use_mean:
                # Take average over masked contact area
                d_i = np.sum(depth_i * mask) / np.sum(mask)
            else:
                # Take deepest point
                d_i = peak_depths[i]
                
            # Use mask to compute contact area
            contact_area_i = (0.001 / PX_TO_MM)**2 * np.sum(mask)
            a_i = np.sqrt(contact_area_i / np.pi)

            # Compute total change in object size using tactile data
            dL = L0 - (self.gripper_widths()[i] + 2*d_i)

            if dL >= 0 and contact_area_i >= self.contact_area_threshold:
                # Save data for fitting where sufficient contact is registered
                x_data.append(dL/L0) # Strain
                y_data.append(abs(self.forces()[i]) / contact_area_i) # Stress
                contact_areas.append(contact_area_i)
                a.append(a_i)
                d.append(d_i)
                F.append(self.forces()[i])

        # Save stuff for plotting
        self._x_data = np.array(x_data)
        self._y_data = np.array(y_data)
        self._contact_areas =  np.array(contact_areas)
        self._a = np.array(a)
        self._d = np.array(d)
        self._F = np.array(F)

        if len(d) == 0:
            return 0
        
        d_norm = self._d / self._d.max()
        F_norm = self._F / self._F.max()
        correlation_mask = (d_norm <= F_norm + 0.25)*(d_norm >= F_norm - 0.25)
        x_data = self._x_data[correlation_mask]
        y_data = self._y_data[correlation_mask]

        # Fit to modulus
        E = self.linear_coeff_fit(x_data, y_data)

        return E
    
    # Estimate modulus based on gripper width change
    # Modification on simple Hookean estimate using tactile data
    # Aggregate data from opposing GelSight tactile sensors
    def fit_modulus_simple_both_sides(self, contact_mask=None, depth_method=None, use_mean=False, use_ellipse_mask=False, \
                            fit_mask_to_ellipse=False, use_lower_resolution_depth=False,
                        ):
        assert self.use_gripper_width
        assert not (use_ellipse_mask and fit_mask_to_ellipse)
        assert self.grasp_data._wedge_video_count > 1

        depth_images = self.depth_images()
        depth_images_markers = self.depth_images(marker_finger=True)
        
        if use_lower_resolution_depth:
            depth_images = self.lower_resolution_depth(depth_images=depth_images, kernel_size=5)
            depth_images_markers = self.lower_resolution_depth(depth_images=depth_images_markers)

        # Precompute peak depths
        if depth_method is None:
            peak_depths = self.top_percentile_depths(depth_images=depth_images)
            peak_depths_markers = self.top_percentile_depths(depth_images=depth_images_markers)
        else:
            peak_depths = self.input_peak_depth_method(depth_method, depth_images=depth_images)
            peak_depths_markers = self.input_peak_depth_method(depth_method, depth_images=depth_images_markers)

        # Find initial length of first contact
        if depth_images.max() < 5*self.depth_threshold and depth_images_markers.max() < 5*self.depth_threshold:
            L0 = self.gripper_widths()[0]
        else:
            i_contact = np.argmax((peak_depths >= self.depth_threshold) * (peak_depths_markers >= self.depth_threshold))
            L0 = self.gripper_widths()[i_contact] + peak_depths[i_contact] + peak_depths_markers[i_contact]

        contact_areas, a, F = [], [], []
        x_data, y_data, d = [], [], []
        for i in range(len(depth_images)):
            depth_i = depth_images[i]
            depth_i_markers = depth_images_markers[i]

            # Skip images without significant contact
            if depth_i.max() < 0.075*peak_depths.max() or depth_i_markers.max() < 0.075*peak_depths_markers.max(): continue
            
            if use_ellipse_mask:
                assert contact_mask is None
                # Compute mask using ellipse fit
                mask = self.ellipse_contact_mask(depth_i)
                mask_markers = self.ellipse_contact_mask(depth_i_markers)
            else:
                # Compute mask using traditional thresholding alone
                mask = self.contact_mask(depth_images[i], contact_mask=contact_mask)
                mask_markers = self.contact_mask(depth_images_markers[i], contact_mask=contact_mask)
                if mask.max() * mask_markers.max() == 0: continue

                if fit_mask_to_ellipse:
                    # Fit an ellipse to the mask and modify
                    ellipse = fit_ellipse_from_binary(mask)
                    if ellipse is None: continue
                    mask = np.zeros_like(mask, dtype=np.uint8)
                    cv2.ellipse(mask, ellipse, 1, -1)
                    ellipse_markers = fit_ellipse_from_binary(mask_markers)
                    if ellipse_markers is None: continue
                    mask_markers = np.zeros_like(mask_markers, dtype=np.uint8)
                    cv2.ellipse(mask_markers, ellipse_markers, 1, -1)

            if use_mean:
                # Take average over masked contact area
                d_i = np.sum(depth_i * mask) / np.sum(mask)
                d_i_markers = np.sum(depth_i_markers * mask_markers) / np.sum(mask_markers)
            else:
                # Take deepest point
                d_i = peak_depths[i]
                d_i_markers = peak_depths_markers[i]
                
            # Use mask to compute contact area
            contact_area_i = (0.001 / PX_TO_MM)**2 * (np.sum(mask) + np.sum(mask_markers))/2
            a_i = np.sqrt(contact_area_i / np.pi)

            # Compute total change in object size using tactile data
            dL = -(self.gripper_widths()[i] + d_i + d_i_markers - L0)

            if dL >= 0 and contact_area_i >= self.contact_area_threshold:
                # Save data for fitting where sufficient contact is registered
                x_data.append(dL/L0) # Strain
                y_data.append(abs(self.forces()[i]) / contact_area_i) # Stress
                contact_areas.append(contact_area_i)
                a.append(a_i)
                d.append((d_i + d_i_markers) / 2)
                F.append(self.forces()[i])

        # Save stuff for plotting
        self._x_data = np.array(x_data)
        self._y_data = np.array(y_data)
        self._contact_areas =  np.array(contact_areas)
        self._a = np.array(a)
        self._d = np.array(d)
        self._F = np.array(F)

        if len(d) == 0:
            return 0
        
        d_norm = self._d / self._d.max()
        F_norm = self._F / self._F.max()
        correlation_mask = (d_norm <= F_norm + 0.25)*(d_norm >= F_norm - 0.25)
        x_data = self._x_data[correlation_mask]
        y_data = self._y_data[correlation_mask]

        # Fit to modulus
        E = self.linear_coeff_fit(x_data, y_data)

        return E
    
    # Fit data to Hertizan model with apparent deformation
    # (Notably only requires gripper width data, not tactile depth)
    def fit_modulus_hertz(self, contact_mask=None, use_ellipse_mask=False, fit_mask_to_ellipse=False, use_lower_resolution_depth=False):
        assert not (use_ellipse_mask and fit_mask_to_ellipse)

        # Calculate apparent deformation using gripper width
        # Pretend that the contact geometry is cylindrical
        # This gives the relation...
        #       F  =  2 E* d a
        #       [From (2.3.2) in "Handbook of Contact Mechanics" by V.L. Popov]

        # Find initial length of first contact
        L0 = self.length_of_first_contact()

        if use_lower_resolution_depth:
            depth_images = self.lower_resolution_depth(kernel_size=5)
        else:
            depth_images = self.depth_images()

        x_data, y_data = [], []
        d, contact_areas, a, F = [], [], [], []
        for i in range(len(depth_images)):
            depth_i = depth_images[i]
            d_i = (L0 - self.gripper_widths()[i])
            
            # Skip images without significant contact
            if depth_i.max() < 0.075*self.max_depths(depth_images=depth_images).max(): continue

            if use_ellipse_mask:
                # Compute mask using ellipse fit
                mask = self.ellipse_contact_mask(depth_i)
                assert contact_mask is None
            else:
                # Compute mask using traditional thresholding alone
                mask = self.contact_mask(depth_images[i], contact_mask=contact_mask)
                if mask.max() == 0: continue

                if fit_mask_to_ellipse:
                    # Fit an ellipse to the mask and modify
                    ellipse = fit_ellipse_from_binary(mask)
                    if ellipse is None: continue
                    mask = np.zeros_like(mask, dtype=np.uint8)
                    cv2.ellipse(mask, ellipse, 1, -1)
                
            # Use mask to compute contact area
            contact_area_i = (0.001 / PX_TO_MM)**2 * np.sum(mask)
            a_i = np.sqrt(contact_area_i / np.pi)

            if contact_area_i >= self.contact_area_threshold and d_i > 0:
                # Save data for fitting where sufficient contact is registered
                x_data.append(2*d_i*a_i)
                y_data.append(self.forces()[i])
                contact_areas.append(contact_area_i)
                d.append(d_i)
                a.append(a_i)
                F.append(self.forces()[i])

        self._x_data = np.array(x_data)
        self._y_data = np.array(y_data)
        self._contact_areas =  np.array(contact_areas)
        self._a = np.array(a)
        self._d = np.array(d)
        self._F = np.array(F)

        if len(d) == 0:
            return 0
        
        d_norm = self._d / self._d.max()
        F_norm = self._F / self._F.max()
        correlation_mask = (d_norm <= F_norm + 0.25)*(d_norm >= F_norm - 0.25)
        x_data = self._x_data[correlation_mask]
        y_data = self._y_data[correlation_mask]

        E_agg = self.linear_coeff_fit(x_data, y_data)
        E = (1/E_agg - 1/(10*self.E_gel))**(-1)

        return E
    
    # Fit data to Hertzian model with apparent deformation
    # (Notably only requires gripper width data, not tactile depth)
    # Aggregate data from opposing GelSight tactile sensors
    def fit_modulus_hertz_both_sides(self, contact_mask=None, use_ellipse_mask=False, fit_mask_to_ellipse=False, use_lower_resolution_depth=False):
        assert not (use_ellipse_mask and fit_mask_to_ellipse)
        assert self.grasp_data._wedge_video_count > 1
        
        # Calculate apparent deformation using gripper width
        # Pretend that the contact geometry is cylindrical
        # This gives the relation...
        #       F  =  2 E* d a
        #       [From (2.3.2) in "Handbook of Contact Mechanics" by V.L. Popov]

        depth_images = self.depth_images()
        depth_images_markers = self.depth_images(marker_finger=True)
        
        if use_lower_resolution_depth:
            depth_images = self.lower_resolution_depth(depth_images=depth_images, kernel_size=5)
            depth_images_markers = self.lower_resolution_depth(depth_images=depth_images_markers)

        # Precompute peak depths based on percentile
        peak_depths = self.top_percentile_depths(depth_images=depth_images)
        peak_depths_markers = self.top_percentile_depths(depth_images=depth_images_markers)

        # Find initial length of first contact
        if depth_images.max() < 5*self.depth_threshold and depth_images_markers.max() < 5*self.depth_threshold:
            L0 = self.gripper_widths()[0]
        else:
            i_contact = np.argmax((peak_depths >= self.depth_threshold) * (peak_depths_markers >= self.depth_threshold))
            L0 = self.gripper_widths()[i_contact] + peak_depths[i_contact] + peak_depths_markers[i_contact]

        x_data, y_data = [], []
        d, contact_areas, a, F = [], [], [], []
        for i in range(len(depth_images)):
            depth_i = depth_images[i]
            depth_i_markers = depth_images[i]
            apparent_d_i = (L0 - self.gripper_widths()[i])
            
            # Skip images without significant contact
            if depth_i.max() < 0.075*peak_depths.max() or depth_i_markers.max() < 0.075*peak_depths_markers.max(): continue

            if use_ellipse_mask:
                # Compute mask using ellipse fit
                mask = self.ellipse_contact_mask(depth_i)
                mask_markers = self.ellipse_contact_mask(depth_i_markers)
                assert contact_mask is None
            else:
                # Compute mask using traditional thresholding alone
                mask = self.contact_mask(depth_images[i], contact_mask=contact_mask)
                mask_markers = self.contact_mask(depth_images_markers[i], contact_mask=contact_mask)
                if mask.max() * mask_markers.max() == 0: continue

                if fit_mask_to_ellipse:
                    # Fit an ellipse to the mask and modify
                    ellipse = fit_ellipse_from_binary(mask)
                    if ellipse is None: continue
                    mask = np.zeros_like(mask, dtype=np.uint8)
                    cv2.ellipse(mask, ellipse, 1, -1)
                    ellipse_markers = fit_ellipse_from_binary(mask_markers)
                    if ellipse_markers is None: continue
                    mask_markers = np.zeros_like(mask_markers, dtype=np.uint8)
                    cv2.ellipse(mask_markers, ellipse_markers, 1, -1)
                
            # Use mask to compute contact area
            contact_area_i = (0.001 / PX_TO_MM)**2 * (np.sum(mask) + np.sum(mask_markers)) / 2
            a_i = np.sqrt(contact_area_i / np.pi)

            if contact_area_i >= self.contact_area_threshold and apparent_d_i > 0:
                # Save data for fitting where sufficient contact is registered
                x_data.append(2*apparent_d_i*a_i)
                y_data.append(self.forces()[i])
                contact_areas.append(contact_area_i)
                d.append(apparent_d_i)
                a.append(a_i)
                F.append(self.forces()[i])

        self._x_data = np.array(x_data)
        self._y_data = np.array(y_data)
        self._contact_areas =  np.array(contact_areas)
        self._a = np.array(a)
        self._d = np.array(d)
        self._F = np.array(F)
        
        if len(d) == 0:
            return 0
        
        d_norm = self._d / self._d.max()
        F_norm = self._F / self._F.max()
        correlation_mask = (d_norm <= F_norm + 0.25)*(d_norm >= F_norm - 0.25)
        x_data = self._x_data[correlation_mask]
        y_data = self._y_data[correlation_mask]

        E_agg = self.linear_coeff_fit(x_data, y_data)
        E = (1/E_agg - 1/(10*self.E_gel))**(-1)

        return E
    
    # Use Hertzian contact models and MDR to compute the modulus of unknown object
    def fit_modulus_hertz_MDR(self, contact_mask=None, depth_method=None, use_ellipse_mask=False, fit_mask_to_ellipse=False, \
                        use_apparent_deformation=True, use_lower_resolution_depth=False):
        if use_apparent_deformation:
            assert self.use_gripper_width

        # Following MDR algorithm from (2.3.2) in "Handbook of Contact Mechanics" by V.L. Popov
        # p_0     = f(E*, F, a)
        # p(r)    = p_0 sqrt(1 - r^2/a^2)
        # q_1d(x) = 2 integral(r p(r) / sqrt(r^2 - x^2) dr)
        # w_1d(x) = (1-v^2)/E_sensor * q_1d(x)
        # w_1d(0) = max_depth

        x_data, y_data = [], []
        d, a, F, R = [], [], [], []
        
        # Find initial length of first contact
        L0 = self.length_of_first_contact()

        if use_lower_resolution_depth:
            depth_images = self.lower_resolution_depth(kernel_size=5)
        else:
            depth_images = self.depth_images()

        # Precompute peak depths
        if depth_method is None:
            peak_depths = self.mean_max_depths(depth_images=depth_images)
        else:
            peak_depths = self.input_peak_depth_method(depth_method, depth_images=depth_images)

        for i in range(len(depth_images)):
            F_i = abs(self.forces()[i])
            d_i = peak_depths[i] # Depth
            apparent_d_i = (L0 - self.gripper_widths()[i]) / 2

            # Skip images without significant contact
            if depth_images[i].max() < 0.075*peak_depths.max(): continue

            if use_ellipse_mask:
                # Compute mask using ellipse fit
                mask = self.ellipse_contact_mask(depth_images[i])
                assert contact_mask is None
            else:
                # Compute mask using traditional thresholding alone
                mask = self.contact_mask(depth_images[i], contact_mask=contact_mask)
            
            if mask.max() == 0: continue
        
            if fit_mask_to_ellipse:
                # Fit mask to an ellipse and take contact radius
                ellipse = fit_ellipse_from_binary(mask, plot_result=False)
                if ellipse is None: continue
                major_axis, minor_axis = ellipse[1]
                contact_area_i = np.pi * (0.001 / PX_TO_MM)**2 * major_axis * minor_axis
                a_i = np.sqrt(contact_area_i / np.pi)
            else:
                # Use mask to directly compute contact area
                contact_area_i = (0.001 / PX_TO_MM)**2 * np.sum(mask)
                a_i = np.sqrt(contact_area_i / np.pi)

            # Estimate object radius R
            if use_apparent_deformation:
                # Use apparent deformation and contact area to compute object radius
                if apparent_d_i <= 1e-4: continue
                R_i = a_i**2 / apparent_d_i
            elif use_ellipse_mask:
                # Compute circle radius using ellipse fit
                ellipse = fit_ellipse_from_float(depth_images[i], plot_result=False)
                if ellipse is None: continue
                major_axis, minor_axis = ellipse[1]
                r_i = 0.5 * (0.001 / PX_TO_MM) * (major_axis + minor_axis)/2
                R_i = d_i + (r_i**2 - d_i**2)/(2*d_i)
            else:
                # Compute estimated radius based on depth (d) and contact radius (a)
                R_i = d_i + (a_i**2 - d_i**2)/(2*d_i)

            if F_i > 0 and contact_area_i >= self.contact_area_threshold and d_i > 0:
                p_0 = (1/np.pi) * (6*F_i/(R_i**2))**(1/3) # times E_star^2/3        # From Wiki
                u_1D_0 = p_0 * np.pi * a_i / 4
                w_1D_0 = (1 - self.nu_gel**2) * u_1D_0 / self.E_gel

                # Save data for fitting where sufficient contact is registered
                d.append(d_i)
                a.append(a_i)
                R.append(R_i)
                F.append(F_i)
                x_data.append(w_1D_0)
                y_data.append(d_i)

        self._d = np.array(d)
        self._a = np.array(a)
        self._F = np.array(F)
        self._R = np.array(R)
        self._x_data = np.array(x_data)
        self._y_data = np.array(y_data)
        
        if len(d) == 0:
            return 0
        
        d_norm = self._d / self._d.max()
        F_norm = self._F / self._F.max()
        correlation_mask = (d_norm <= F_norm + 0.25)*(d_norm >= F_norm - 0.25)
        x_data = self._x_data[correlation_mask]
        y_data = self._y_data[correlation_mask]

        # Fit for E_star
        E_star = self.linear_coeff_fit(x_data, y_data)**(3/2)

        return E_star
    
    # Use Hertzian contact models and MDR to compute the modulus of unknown object
    # Aggregate data from opposing GelSight tactile sensors
    def fit_modulus_hertz_MDR_both_sides(self, contact_mask=None, depth_method=None, use_ellipse_mask=False, fit_mask_to_ellipse=False, \
                                   use_apparent_deformation=True, use_lower_resolution_depth=False):
        if use_apparent_deformation:
            assert self.use_gripper_width
        assert self.grasp_data._wedge_video_count > 1

        # Following MDR algorithm from (2.3.2) in "Handbook of Contact Mechanics" by V.L. Popov
        # p_0     = f(E*, F, a)
        # p(r)    = p_0 sqrt(1 - r^2/a^2)
        # q_1d(x) = 2 integral(r p(r) / sqrt(r^2 - x^2) dr)
        # w_1d(x) = (1-v^2)/E_sensor * q_1d(x)
        # w_1d(0) = max_depth

        x_data, y_data = [], []
        d, a, F, R = [], [], [], []
        
        depth_images = self.depth_images()
        depth_images_markers = self.depth_images(marker_finger=True)
        
        if use_lower_resolution_depth:
            depth_images = self.lower_resolution_depth(depth_images=depth_images, kernel_size=5)
            depth_images_markers = self.lower_resolution_depth(depth_images=depth_images_markers)

        # Precompute peak depths
        if depth_method is None:
            peak_depths = self.mean_max_depths(depth_images=depth_images)
            peak_depths_markers = self.mean_max_depths(depth_images=depth_images_markers)
        else:
            peak_depths = self.input_peak_depth_method(depth_method, depth_images=depth_images)
            peak_depths_markers = self.input_peak_depth_method(depth_method, depth_images=depth_images_markers)

        # Find initial length of first contact
        if depth_images.max() < 5*self.depth_threshold and depth_images_markers.max() < 5*self.depth_threshold:
            L0 = self.gripper_widths()[0]
        else:
            i_contact = np.argmax((peak_depths >= self.depth_threshold) * (peak_depths_markers >= self.depth_threshold))
            L0 = self.gripper_widths()[i_contact] + peak_depths[i_contact] + peak_depths_markers[i_contact]

        for i in range(len(depth_images)):
            F_i = abs(self.forces()[i])
            d_i = peak_depths[i] # Depth
            d_i_markers = peak_depths_markers[i] # Depth
            mean_d_i = (d_i_markers + d_i)/2
            apparent_d_i = (L0 - self.gripper_widths()[i]) / 2

            # Skip images without significant contact
            if depth_images[i].max() < 0.075*peak_depths.max() or depth_images_markers[i].max() < 0.075*peak_depths_markers.max(): continue

            if use_ellipse_mask:
                # Compute mask using ellipse fit
                mask = self.ellipse_contact_mask(depth_images[i])
                mask_markers = self.ellipse_contact_mask(depth_images_markers[i])
                assert contact_mask is None
            else:
                # Compute mask using traditional thresholding alone
                mask = self.contact_mask(depth_images[i], contact_mask=contact_mask)
                mask_markers = self.contact_mask(depth_images_markers[i], contact_mask=contact_mask)
            
            if mask.max() * mask_markers.max() == 0: continue
        
            if fit_mask_to_ellipse:
                # Fit mask to an ellipse and take contact radius
                ellipse = fit_ellipse_from_binary(mask, plot_result=False)
                if ellipse is None: continue
                major_axis, minor_axis = ellipse[1]
                contact_area_i = np.pi * (0.001 / PX_TO_MM)**2 * major_axis * minor_axis
                ellipse_markers = fit_ellipse_from_binary(mask_markers, plot_result=False)
                if ellipse_markers is None: continue
                major_axis_markers, minor_axis_markers = ellipse_markers[1]
                contact_area_i_markers = np.pi * (0.001 / PX_TO_MM)**2 * major_axis_markers * minor_axis_markers
                contact_area_i = (contact_area_i + contact_area_i_markers)/2
                a_i = np.sqrt(contact_area_i / np.pi)
            else:
                # Use mask to directly compute contact area
                contact_area_i = (0.001 / PX_TO_MM)**2 * (np.sum(mask) + np.sum(mask_markers))/2
                a_i = np.sqrt(contact_area_i / np.pi)

            # Estimate object radius R
            if use_apparent_deformation:
                # Use apparent deformation and contact area to compute object radius
                if apparent_d_i <= 1e-4: continue
                R_i = a_i**2 / apparent_d_i
            elif use_ellipse_mask:
                # Compute circle radius using ellipse fit
                ellipse = fit_ellipse_from_float(depth_images[i], plot_result=False)
                if ellipse is None: continue
                major_axis, minor_axis = ellipse[1]
                ellipse_markers = fit_ellipse_from_float(depth_images_markers[i], plot_result=False)
                if ellipse_markers is None: continue
                major_axis_markers, minor_axis_markers = ellipse_markers[1]
                r_i = 0.5 * (0.001 / PX_TO_MM) * (major_axis + minor_axis + major_axis_markers + minor_axis_markers)/2
                R_i = (d_i_markers + d_i)/2 + (r_i**2 - (mean_d_i)**2)/(2*mean_d_i)
            else:
                # Compute estimated radius based on depth (d) and contact radius (a)
                R_i = mean_d_i + (a_i**2 - mean_d_i**2)/(2*mean_d_i)

            if F_i > 0 and contact_area_i >= self.contact_area_threshold and d_i > 0:
                p_0 = (1/np.pi) * (6*F_i/(R_i**2))**(1/3) # times E_star^2/3        # From Wiki
                u_1D_0 = p_0 * np.pi * a_i / 4
                w_1D_0 = (1 - self.nu_gel**2) * u_1D_0 / self.E_gel

                # Save data for fitting where sufficient contact is registered
                d.append(mean_d_i)
                a.append(a_i)
                R.append(R_i)
                F.append(F_i)
                x_data.append(w_1D_0)
                y_data.append(d_i)

        self._d = np.array(d)
        self._a = np.array(a)
        self._F = np.array(F)
        self._R = np.array(R)
        self._x_data = np.array(x_data)
        self._y_data = np.array(y_data)
    
        if len(d) == 0:
            return 0
        
        d_norm = self._d / self._d.max()
        F_norm = self._F / self._F.max()
        correlation_mask = (d_norm <= F_norm + 0.25)*(d_norm >= F_norm - 0.25)
        x_data = self._x_data[correlation_mask]
        y_data = self._y_data[correlation_mask]
        
        # Fit for E_star
        E_star = self.linear_coeff_fit(x_data, y_data)**(3/2)

        return E_star
    
    def Estar_to_E(self, E_star):
        # Compute compliance from E_star by assuming Poisson's ratio
        nu = self.assumed_poisson_ratio
        E = (1 - nu**2) / (1/E_star - (1 - self.nu_gel**2)/(self.E_gel))
        return E, nu

    # Display raw data from a depth image in 3D
    def plot_depth(self, depth):
        # Extract 3D data
        X, Y, Z = self.depth_to_XYZ(depth, remove_zeros=False, remove_outliers=False)

        # Plot sphere in 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X, Y, Z, s=8, c=Z, cmap='winter', rasterized=True)
        ax.set_xlabel('$X$ [m]', fontsize=16)
        ax.set_ylabel('\n$Y$ [m]', fontsize=16)
        ax.set_zlabel('\n$Z$ [m]', fontsize=16)
        ax.set_title('Sphere Fitting', fontsize=16)
        plt.show()
        return
    
    # Display raw data from a depth image as 2D heightmap
    def plot_depth_2D(self, depth):
        plt.figure()
        plt.imshow(depth, cmap="winter")
        plt.title(f'Depth')
        plt.xlabel('Y')
        plt.ylabel('X')
        plt.colorbar()
        plt.show()
    
    # Watch evolution of depth images over time
    def watch_depth_2D(self):
        plt.ion()
        _, ax = plt.subplots()
        ax.set_title(f'Depth')
        ax.set_xlabel('Y')
        ax.set_ylabel('X')
        im = ax.imshow(self.depth_images()[0], cmap="winter")
        for i in range(len(self.depth_images())):
            im.set_array(self.depth_images()[i])
            plt.draw()
            plt.pause(0.5)
        plt.ioff()
        plt.show()
        return
    
    # Display computed contact mask for a given depth image
    def plot_contact_mask(self, depth, contact_mask=None):
        plt.figure()
        plt.imshow(self.contact_mask(depth, contact_mask=contact_mask), cmap=plt.cm.gray)
        plt.title(f'Contact Mask')
        plt.xlabel('Y')
        plt.ylabel('X')
        plt.colorbar()
        plt.show()
        return
    
    # Watch evolution of computed contact mask over time
    def watch_contact_mask(self, contact_mask=None):
        plt.ion()
        _, ax = plt.subplots()
        ax.set_title(f'Contact Mask')
        ax.set_xlabel('Y')
        ax.set_ylabel('X')
        im = ax.imshow(self.contact_mask(self.depth_images()[0], contact_mask=contact_mask), cmap=plt.cm.gray)
        for i in range(len(self.depth_images())):
            im.set_array(self.contact_mask(self.depth_images()[i], contact_mask=contact_mask))
            plt.draw()
            plt.pause(0.5)
        plt.ioff()
        plt.show()
        return
    
    # Check sphere fit by plotting data and fit shape
    def plot_sphere_fit(self, depth, sphere):
        # Extract 3D data
        X, Y, Z = self.depth_to_XYZ(depth)

        # Create discrete graph of sphere mesh
        r, x0, y0, z0 = sphere
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi/10:10j] # Plot top half of sphere
        # u, v = np.mgrid[0:2*np.pi:20j, 0:2*np.pi:20j] # Plot full sphere
        sphere_x = x0 + np.cos(u)*np.sin(v)*r
        sphere_y = y0 + np.sin(u)*np.sin(v)*r
        sphere_z = z0 + np.cos(v)*r

        # Plot sphere in 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X, Y, Z, s=8, c=Z, cmap='winter', rasterized=True)
        ax.plot_wireframe(sphere_x, sphere_y, sphere_z, color="r")
        ax.set_xlabel('$X$ [m]', fontsize=16)
        ax.set_ylabel('\n$Y$ [m]', fontsize=16)
        ax.set_zlabel('\n$Z$ [m]', fontsize=16)
        ax.set_title('Sphere Fitting', fontsize=16)
        plt.show()
        return
    
    # Plot different ways of aggregating depth from each image
    def plot_depth_metrics(self):
        plt.figure()
        plt.plot(self.max_depths(), label="Max Depth")
        plt.plot(self.mean_max_depths(), label="Mean Max Depth")
        plt.plot(self.top_percentile_depths(), label="Top Percentile Depth")
        plt.plot(self.mean_depths(), label="Mean Depth")
        plt.xlabel('Index [/]')
        plt.ylabel('Depth [m]')
        plt.legend()
        plt.show()
        return
    
    # Plot all normalized grasp data over indices
    def plot_grasp_data(self):
        plt.figure()
        plt.plot(abs(self.forces()) / abs(self.forces()).max(), label="Normalized Forces")
        plt.plot(self.gripper_widths() / self.gripper_widths().max(), label="Normalized Gripper Width")
        plt.plot(self.max_depths() / self.max_depths().max(), label="Normalized Depth")
        plt.legend()
        plt.show()
        return


if __name__ == "__main__":

    #####################################################
    # GET ESTIMATED MODULUS (E) FOR SET OF EXAMPLE DATA #
    #####################################################

    # Choose which mechanical model to use
    use_method = "simple"
    assert use_method in ["simple", "hertz", "MDR"]

    fig1 = plt.figure(1)
    sp1 = fig1.add_subplot(211)
    sp1.set_xlabel('Measured Sensor Deformation (d) [m]')
    sp1.set_ylabel('Force [N]')
    
    if use_method == "simple":
        # Set up stress / strain axes for simple method
        fig2 = plt.figure(2)
        sp2 = fig2.add_subplot(211)
        sp2.set_xlabel('Strain (dL/L) [/]')
        sp2.set_ylabel('Stress (F/A) [Pa]')
    
    elif use_method == "hertz":
        # Set up stress / strain axes for simple method
        fig2 = plt.figure(2)
        sp2 = fig2.add_subplot(211)
        sp2.set_xlabel('Area [m^2]')
        sp2.set_ylabel('Force [N]')

    elif use_method == "MDR":
        # Set up axes for MDR method
        fig2 = plt.figure(2)
        sp2 = fig2.add_subplot(211)
        sp2.set_xlabel('[Pa]^(-2/3)')
        sp2.set_ylabel('Depth [m]')

        # Set up axes for checking radius estimates
        fig3 = plt.figure(3)
        sp3 = fig3.add_subplot(211)
        sp3.set_xlabel('Radius [m]')
        sp3.set_ylabel('Index [/]')

    wedge_video             = GelSightWedgeVideo(config_csv="./wedge_config/config_no_markers.csv") # Force-sensing finger
    wedge_video_markers     = GelSightWedgeVideo(config_csv="./wedge_config/config_markers.csv") # Non-sensing finger
    contact_force           = ContactForce()
    gripper_width           = GripperWidth()
    grasp_data              = GraspData(wedge_video=wedge_video, wedge_video_markers=wedge_video_markers, \
                                        contact_force=contact_force, gripper_width=gripper_width, use_gripper_width=True)

    # For plotting
    obj_to_color = {
        "yellow_foam_brick_softest" : "yellow",
        "red_foam_brick_softer"     : "red",
        "blue_foam_brick_harder"    : "blue",
        "orange_ball_softest"       : "orange",
        "green_ball_softer"         : "green",
        "purple_ball_hardest"       : "indigo",
        "rigid_strawberry"          : "purple",
        "golf_ball"                 : "gray",
        "bolt"                      : "black",
        "rose_eraser"               : "brown",
    }

    # Unload data from folder
    data_folder = "./example_data/2023-12-16"
    data_files = os.listdir(data_folder)
    for i in range(len(data_files)):
        file_name = data_files[i]
        if os.path.splitext(file_name)[1] != '.avi' or file_name.count("_markers") > 0:
            continue
        obj_name = os.path.splitext(file_name)[0].split('__')[0]

        # Load data into modulus estimator
        grasp_data._reset_data()
        estimator = EstimateModulus(grasp_data=grasp_data, use_gripper_width=True)
        estimator.load_from_file(data_folder + "/" + os.path.splitext(file_name)[0], auto_clip=True)
        
        # Clip to loading sequence
        estimator.clip_to_press()
        assert len(estimator.depth_images()) == len(estimator.forces()) == len(estimator.gripper_widths())

        # Remove stagnant gripper values across measurement frames
        estimator.interpolate_gripper_widths()

        if use_method == "simple":
            # Fit using simple estimator
            E_object = estimator.fit_modulus_simple(use_mean=False, use_ellipse_mask=False, fit_mask_to_ellipse=True, use_lower_resolution_depth=True)

        elif use_method == "hertz":
            # Fit using simple Hertzian estimator
            E_object = estimator.fit_modulus_hertz(use_ellipse_mask=True)

        elif use_method == "MDR":
            # Fit using our MDR estimator
            E_star = estimator.fit_modulus_hertz_MDR(use_ellipse_mask=False, fit_mask_to_ellipse=True, use_apparent_deformation=True, use_lower_resolution_depth=False)
            E_object, v_object = estimator.Estar_to_E(E_star)

        print('Object:', obj_name)
        print(f'Maximum depth of {obj_name}:', np.max(estimator.max_depths()))
        print(f'Maximum force of {obj_name}:', np.max(estimator.forces()))
        print(f'Estimated modulus of {obj_name}:', E_object)
        print('\n')

        # Plot
        plotting_color = random_shade_of_color(obj_to_color[obj_name])
        sp1.plot(estimator.max_depths(), estimator.forces(), ".", label=obj_name, markersize=8, color=plotting_color)
        sp2.plot(estimator._x_data, estimator._y_data, ".", label=obj_name, markersize=8, color=plotting_color)

        if use_method == "simple":
            # Plot simple fit
            sp2.plot(estimator._x_data, E_object*np.array(estimator._x_data), "-", label=obj_name, markersize=8, color=plotting_color)

        elif use_method == "hertz":
            # Plot simple Hertzian fit
            E_agg = (1/E_object + 1/estimator.E_gel)**(-1)
            sp2.plot(estimator._x_data, E_agg*np.array(estimator._x_data), "-", label=obj_name, markersize=8, color=plotting_color)
            
        elif use_method == "MDR":
            # Plot MDR fit
            sp2.plot(estimator._x_data, (E_star**(2/3))*np.array(estimator._x_data), "-", label=obj_name, markersize=8, color=plotting_color)
            sp3.plot(estimator._R[3:], ".", label=obj_name, markersize=8, color=plotting_color)

    fig1.legend()
    fig1.set_figwidth(10)
    fig1.set_figheight(10)
    fig2.legend()
    fig2.set_figwidth(10)
    fig2.set_figheight(10)
    if use_method == "MDR":
        fig3.legend()
        fig3.set_figwidth(10)
        fig3.set_figheight(10)
    plt.show()
    print('Done.')