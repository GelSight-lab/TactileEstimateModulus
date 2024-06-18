import os
import time
import numpy as np
import warnings
import matplotlib.pyplot as plt

from src.wedge_video import GelSightWedgeVideo, DEPTH_THRESHOLD, AUTO_CLIP_OFFSET
from src.contact_force import ContactForce, FORCE_THRESHOLD
from src.gripper_width import GripperWidth

from threading import Thread

class GraspData():
    '''
    Class to streamline recording of data from GelSight Wedge's / force gauge and package into training
    '''
    def __init__(self, 
                 wedge_video=GelSightWedgeVideo(config_csv="./wedge_config/config_no_markers.csv"),
                 wedge_video_markers=None,
                 contact_force=ContactForce(), 
                 gripper_width=GripperWidth(), 
                 use_gripper_width=True
        ):
        self.wedge_video = wedge_video                      # Object containing all video data for wedge on force-sensing finger
        self.wedge_video_markers = wedge_video_markers      # Object containing all video data for wedge on marker finger
        self.contact_force = contact_force                  # Object containing all force measurements
        self.gripper_width = gripper_width                  # Object containing gripper width measurements
        self.use_gripper_width = use_gripper_width          # Boolean of whether or not to record gripper width data

        # How many wedge's are we streaming from?
        self._wedge_video_count = 1
        if self.wedge_video_markers is not None:
            self._wedge_video_count += 1

        self._stream_thread = Thread        # Streaming thread
        self._plot_thread = Thread          # Plotting thread
        self._stream_active = False         # Boolean of whether or not we're currently streaming
        self._plotting = False              # Boolean of whether or not we're plotting during stream
    
    # Clear all data from the object
    def _reset_data(self):
        self.wedge_video._reset_frames()
        if self._wedge_video_count > 1:
            self.wedge_video_markers._reset_frames()
        self.contact_force._reset_values()
        if self.use_gripper_width:
            self.gripper_width._reset_values()

    # Return forces
    def forces(self):
        return self.contact_force.forces()
    
    # Return gripper widths
    def gripper_widths(self):
        assert self.use_gripper_width
        return self.gripper_width.widths()
    
    # Return depth images
    def depth_images(self, marker_finger=False):
        if marker_finger:
            assert self._wedge_video_count > 1
            return self.wedge_video_markers.depth_images()
        return self.wedge_video.depth_images()
    
    # Return maximum from each depth image
    def max_depths(self, marker_finger=False):
        if marker_finger:
            assert self._wedge_video_count > 1
            return self.wedge_video_markers.max_depths()
        return self.wedge_video.max_depths()
    
    # Return mean value from each depth image
    def mean_depths(self, marker_finger=False):
        if marker_finger:
            assert self._wedge_video_count > 1
            return self.wedge_video_markers.mean_depths()
        return self.wedge_video.mean_depths()
    
    # Initiate streaming thread
    def start_stream(self, verbose=True, plot=False, plot_markers=False, plot_diff=False, plot_depth=False, _open_socket=True):
        self._reset_data()
        self._stream_active = True

        # Get video stream ready
        self.wedge_video._prepare_stream()
        if self._wedge_video_count > 1:
            self.wedge_video_markers._prepare_stream()

        # Start recording contact force and gripper width
        self.contact_force.start_stream(read_only=True, _open_socket=_open_socket)
        if self.use_gripper_width:
            self.gripper_width.start_stream(read_only=True)

        self._stream_thread = Thread(target=self._stream, kwargs={"verbose": verbose})
        self._stream_thread.daemon = True
        self._stream_thread.start()
        time.sleep(1)

        # Argument plot_markers determines which video is plotted
        if plot and not plot_markers:
            self.wedge_video._start_plotting(plot_diff=plot_diff, plot_depth=plot_depth)
        elif plot and plot_markers:
            assert self._wedge_video_count > 1          
            self.wedge_video_markers._start_plotting(plot_diff=plot_diff, plot_depth=plot_depth)
        return
    
    # Facilitate streaming thread, read data from Raspberry Pi camera
    def _stream(self, verbose=False):
        while self._stream_active:
            if verbose: print('Streaming...')
            img_found = self.wedge_video._decode_image_from_stream()
            if self._wedge_video_count > 1:
                _ = self.wedge_video_markers._decode_image_from_stream()
            if img_found:
                self.contact_force._request_value()
                if self.use_gripper_width:
                    self.gripper_width._request_value()
        return

    # Terminate streaming thread
    def end_stream(self, verbose=False, _close_socket=True):
        self._stream_active = False
        self._stream_thread.join()
        
        if self.wedge_video._plotting:
            self.wedge_video._stop_plotting()
        if self._wedge_video_count > 1 and self.wedge_video_markers._plotting:
            self.wedge_video_markers._stop_plotting()

        self.wedge_video._wipe_stream_info()
        if self._wedge_video_count > 1:
            self.wedge_video_markers._wipe_stream_info()

        self.contact_force.end_stream(verbose=False, _close_socket=_close_socket)
        if self.use_gripper_width:
            self.gripper_width.end_stream(verbose=False)

        time.sleep(1)
        if verbose: print('Done streaming.')

        if self.use_gripper_width:
            assert len(self.contact_force.forces()) == len(self.gripper_width.widths()) == len(self.wedge_video._raw_rgb_frames)
        else:
            assert len(self.contact_force.forces()) == len(self.wedge_video._raw_rgb_frames)
        return
    
    # Clip data between frame indices
    def clip(self, i_start, i_end):
        assert 0 <= i_start < i_end <= len(self.wedge_video._raw_rgb_frames)
        self.wedge_video.clip(i_start, i_end)
        if self._wedge_video_count > 1:
            self.wedge_video_markers.clip(i_start, i_end)
        self.contact_force.clip(i_start, i_end)
        if self.use_gripper_width:
            self.gripper_width.clip(i_start, i_end)
        return
    
    # Clip data to first press via thresholding
    def auto_clip(self, use_force=True, force_threshold=FORCE_THRESHOLD, depth_threshold=DEPTH_THRESHOLD, clip_offset=AUTO_CLIP_OFFSET):
        if use_force:
            self.auto_clip_by_force(force_threshold=force_threshold, clip_offset=clip_offset)
        else:
            self.auto_clip_by_depth(depth_threshold=depth_threshold, clip_offset=clip_offset)
        return

    # Auto clip based on force
    def auto_clip_by_force(self, force_threshold=FORCE_THRESHOLD, clip_offset=AUTO_CLIP_OFFSET):
        i_start, i_end = self.contact_force.auto_clip(force_threshold=force_threshold, clip_offset=clip_offset, return_indices=True)
        self.wedge_video.clip(i_start, i_end)
        if self._wedge_video_count > 1:
            self.wedge_video_markers.clip(i_start, i_end)
        if self.use_gripper_width:
            self.gripper_width.clip(i_start, i_end)
        return
            
    # Auto clip based on depth of frames
    def auto_clip_by_depth(self, depth_threshold=DEPTH_THRESHOLD, clip_offset=AUTO_CLIP_OFFSET):
        i_start, i_end = self.wedge_video.auto_clip(depth_threshold=depth_threshold, clip_offset=clip_offset, return_indices=True)
        if self._wedge_video_count > 1:
            self.wedge_video_markers.clip(i_start, i_end)
        self.contact_force.clip(i_start, i_end)
        if self.use_gripper_width:
            self.gripper_width.clip(i_start, i_end)
        return

    # Clip a press sequence to only the loading sequence (positive force)
    def clip_to_press(self, force_threshold=FORCE_THRESHOLD, start_offset=0, peak_offset=0, pct_peak_threshold=0.975):
        # Find initial and peak force over press
        i_start = max(np.argmax(self.forces() >= force_threshold)-1, 0)
        i_peak = np.argmax(self.forces())

        # Grab index before below 97.5% of peak
        i_end = i_peak
        if pct_peak_threshold < 1:
            for i in range(len(self.forces())):
                if i > i_peak and self.forces()[i] < pct_peak_threshold*self.forces()[i_peak]:
                    i_end = i-1
                    break

        if i_start >= i_end:
            warnings.warn("No press detected! Cannot clip.", Warning)
        else:
            self.clip(i_start + start_offset, i_end + peak_offset + 1)
        return
    
    # Linearly interpolate gripper widths wherever measurements are equal
    def interpolate_gripper_widths(self, plot_result=False):
        interpolated_gripper_widths = self.gripper_widths()
        i = 0
        while i < len(interpolated_gripper_widths)-1:
            if self.gripper_widths()[i] == self.gripper_widths()[-1]: break
            if self.gripper_widths()[i] == self.gripper_widths()[i+1]:
                for k in range(i, len(interpolated_gripper_widths)):
                    if self.gripper_widths()[i] != self.gripper_widths()[k]:
                        break
                slope = (self.gripper_widths()[k] - self.gripper_widths()[i]) / (k-i)
                for j in range(i+1, k):
                    interpolated_gripper_widths[j] = self.gripper_widths()[i] + slope*(j-i)
                i = k
            else:
                i += 1

        if plot_result:
            plt.figure()
            plt.plot(self.gripper_widths(), '.')
            plt.plot(interpolated_gripper_widths, '-')
            plt.xlabel('Index [/]')
            plt.ylabel('Gripper Width [m]')
            plt.show()

        # Adjust the wrapped grasp_data object
        self.gripper_width._widths = interpolated_gripper_widths
        return
    
    # Plot all data over indices
    def plot_grasp_data(self):
        forces      = abs(self.forces())
        widths      = self.gripper_widths()
        max_depths  = self.max_depths()
        if self._wedge_video_count > 1:
            max_depth_markers = self.max_depths(marker_finger=True)
        plt.figure()
        plt.plot(forces / forces.max(), label="Normalized Contact Forces")
        plt.plot(widths / widths.max(), label="Normalized Gripper Widths")
        plt.plot(max_depths / max_depths.max(), label="Normalized Max Depths")
        if self._wedge_video_count > 1:
            plt.plot(max_depth_markers / max_depth_markers.max(), label="Normalized Max Depths (Markers)")
        plt.xlabel('Index [/]')
        plt.legend()
        plt.show(block=False)
        return

    # Plot video for your viewing pleasure
    def watch(self, plot_diff=False, plot_depth=False, marker_finger=False):
        if not marker_finger:
            self.wedge_video.watch(plot_diff=plot_diff, plot_depth=plot_depth)
        else:
            assert self._wedge_video_count > 1
            self.wedge_video_markers.watch(plot_diff=plot_diff, plot_depth=plot_depth)
        return
    
    # Read frames from a video file and associated pickle files
    def load(self, path_to_file):
        self.wedge_video.load(path_to_file + '.avi')
        if self._wedge_video_count > 1:
            self.wedge_video_markers.load(path_to_file + '_markers.avi')
        self.contact_force.load(path_to_file + '_forces.pkl')
        if self.use_gripper_width:
            self.gripper_width.load(path_to_file + '_widths.pkl')
        return

    # Save collected data to video and pickle files
    def save(self, path_to_file):
        self.wedge_video.save(path_to_file + '.avi')
        if self._wedge_video_count > 1:
            self.wedge_video_markers.save(path_to_file + '_markers.avi')
        self.contact_force.save(path_to_file + '_forces.pkl')
        if self.use_gripper_width:
            self.gripper_width.save(path_to_file + '_widths.pkl')
        return
    

if __name__ == "__main__":
    # Typical data collection workflow might be...

    # Define streaming addresses
    wedge_video         =   GelSightWedgeVideo(IP="172.16.0.100", config_csv="./wedge_config/config_no_markers.csv") # Force-sensing finger
    # wedge_video_markers   =   GelSightWedgeVideo(IP="172.16.0.200", config_csv="./wedge_config/config_markers.csv") # Marker finger
    contact_force       =   ContactForce(IP="172.16.0.50", port=8888)
    grasp_data          =   GraspData(wedge_video=wedge_video, contact_force=contact_force)

    # Record example data and save
    grasp_data.start_stream(verbose=True, plot=True, plot_diff=True, plot_depth=True)
    time.sleep(3)
    grasp_data.end_stream(verbose=True)
    grasp_data.auto_clip()