import time
import cv2
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from threading import Thread

from gelsight_wedge.src.gelsight.util.Vis3D import ClassVis3D

from src.wedge_video import GelSightWedgeVideo, DEPTH_THRESHOLD, AUTO_CLIP_OFFSET
from src.contact_force import ContactForce, FORCE_THRESHOLD
from src.gripper_width import GripperWidth
from src.grasp_data import GraspData

from frankapy import FrankaArm

franka_arm = FrankaArm()

def open_gripper(_franka_arm): {
    _franka_arm.goto_gripper(
        0.08, # Maximum width in meters [m]
        speed=0.05, # Desired operation speed in [m/s]
        block=True,
        skill_desc="OpenGripper"
    )
}

def close_gripper(_franka_arm): {
    _franka_arm.goto_gripper(
        0.0, # Minimum width in meters [m]
        force=100, # Maximum force in Newtons [N]
        speed=0.0375, # Desired operation speed in [m/s]
        epsilon_inner=0.005, # Maximum tolerated deviation [m]
        epsilon_outer=0.005, # Maximum tolerated deviation [m]
        grasp=True,     
        block=True,
        skill_desc="CloseGripper"
    )
}

def collect_data_for_object(object_name, num_grasps, folder_name=None, plot_collected_data=False):
    # Define streaming addresses
    wedge_video         =   GelSightWedgeVideo(IP="172.16.0.100", config_csv="./wedge_config/config_no_markers.csv", markers=False)
    wedge_video_markers   =   GelSightWedgeVideo(IP="172.16.0.200", config_csv="./wedge_config/config_markers.csv", markers=True)
    contact_force       =   ContactForce(IP="172.16.0.69", port=8888)
    gripper_width       =   GripperWidth(franka_arm=franka_arm)
    grasp_data          =   GraspData(wedge_video=wedge_video, wedge_video_markers=wedge_video_markers, contact_force=contact_force, gripper_width=gripper_width)

    if folder_name is None:
        # Choose folder name as YYYY-MM-DD by default
        folder_name = datetime.now().strftime('%Y-%m-%d')

    if not os.path.exists(f'./data/raw_data/{folder_name}'):
        os.mkdir(f'./data/raw_data/{folder_name}')

    # Use these variables to keep force reading socket open over multiple grasps
    _open_socket = True
    _close_socket = False

    # Dedicated plotting function for multi-grasp plotting
    # Note: this became necessary because cv2.imshow() is not great with multiple threads
    def plot_stream(grasp_data=None, plot_diff=False, plot_depth=False, verbose=False):
        Vis3D = None
        while _plot_stream:
            if type(grasp_data.wedge_video._curr_rgb_image) != type(None):
                if verbose: print('Plotting...')

                if plot_diff or plot_depth:
                    diff_img = grasp_data.wedge_video.calc_diff_image(grasp_data.wedge_video.warp_image(grasp_data.wedge_video._raw_rgb_frames[0]), grasp_data.wedge_video.warp_image(grasp_data.wedge_video._curr_rgb_image))

                # Plot depth in 3D
                if plot_depth:
                    if no_data and type(Vis3D) == type(None): # Re-initialize 3D plotting
                        Vis3D = ClassVis3D(n=grasp_data.wedge_video.warped_size[0], m=grasp_data.wedge_video.warped_size[1])
                    Vis3D.update(grasp_data.wedge_video.img2depth(diff_img) / grasp_data.wedge_video.PX_TO_MM)

                if type(grasp_data.wedge_video._curr_rgb_image) != type(None):

                    # Plot raw RGB image
                    cv2.imshow('raw_RGB', grasp_data.wedge_video._curr_rgb_image)

                    # Plot difference image
                    if plot_diff:
                        cv2.imshow('diff_img', diff_img)

                    if cv2.waitKey(1) & 0xFF == ord('q'): # Exit windows by pressing "q"
                        break
                    if cv2.waitKey(1) == 27: # Exit window by pressing Esc
                        break

                no_data = False

            else:
                if verbose: print('No data. Not plotting.')
                no_data = True; Vis3D = None
                cv2.destroyAllWindows()
            
        cv2.destroyAllWindows()
        return

    # NOTE: - Do not stream / plot video while recording training data!
    #       - Uncomment below only for debugging purposes
    # _plot_stream = True
    # _stream_thread = Thread(target=plot_stream, kwargs={"grasp_data": grasp_data, "plot_diff": True, "plot_depth": True})
    # _stream_thread.daemon = True
    # _stream_thread.start()

    # Execute number of data collection grasps
    for i in range(num_grasps):
        print(f'Grasp #{i}:')

        if not os.path.exists(f'./data/raw_data/{folder_name}/grasp={i}'):
            os.mkdir(f'./data/raw_data/{folder_name}/grasp={i}')
        else:
            raise ValueError, 'Grasp index has already been recorded. Will not overwrite.'

        # Start with gripper in open position
        open_gripper(franka_arm)

        # Start recording
        if i > 0: _open_socket = False
        print('Starting stream...')
        grasp_data.start_stream(plot=False, verbose=False, _open_socket=_open_socket)

        # Close gripper
        close_gripper(franka_arm)
        time.sleep(0.5)

        # Open gripper
        open_gripper(franka_arm)
        time.sleep(2)

        # Stop recording
        if i == num_grasps - 1: _close_socket = True
        grasp_data.end_stream(_close_socket=_close_socket)
        print('Done streaming.')

        # Make some assertions to ensure data collection is okay
        assert grasp_data.forces()[0] < FORCE_THRESHOLD and grasp_data.forces()[-1] < FORCE_THRESHOLD
        assert grasp_data.forces().max() > FORCE_THRESHOLD
        assert grasp_data.gripper_widths().max() > grasp_data.gripper_widths().min() + 0.01
        
        # Clip conservatively based on force threshold
        grasp_data.auto_clip(clip_offset=AUTO_CLIP_OFFSET+10)

        ultimate_depth = max(grasp_data.max_depths().max(), grasp_data.max_depths(marker_finger=True).max())
        if ultimate_depth < DEPTH_THRESHOLD:
            warnings.warn(f'The maximum recorded depth was {ultimate_depth}mm. This is lower than the currently defined depth threshold.')

        # Make sure depths are aligned, smart shift based on correlation
        shift = np.argmax(
                        np.correlate(
                            grasp_data.max_depths() / grasp_data.max_depths().max(), \
                            grasp_data.max_depths(marker_finger=True) / grasp_data.max_depths(marker_finger=True).max(),
                            mode="full" \
                        )
                    ) - len(grasp_data.max_depths()) + 1
        if shift >= 15 or shift < 0:
            warnings.warn(f'The chosen video shift was {shift}. This value is bigger than typically expected', UserWarning)

        # Apply the shift
        shift = max(shift, 0)
        grasp_data.wedge_video.clip(shift, len(grasp_data.wedge_video._raw_rgb_frames))
        grasp_data.wedge_video_markers.clip(0, len(grasp_data.wedge_video_markers._raw_rgb_frames)-shift)
        grasp_data.contact_force.clip(shift, len(grasp_data.contact_force.forces()))
        grasp_data.gripper_width.clip(shift, len(grasp_data.gripper_width.widths()))

        if plot_collected_data:
            plt.plot(abs(grasp_data.forces()) / abs(grasp_data.forces()).max(), label="Forces")
            plt.plot(grasp_data.gripper_widths() / grasp_data.gripper_widths().max(), label="Gripper Widths")
            plt.plot(grasp_data.max_depths() / grasp_data.max_depths().max(), label="Max Depths")
            plt.plot(grasp_data.max_depths(marker_finger=True) / grasp_data.max_depths(marker_finger=True).max(), label="Max Depths (Markers)")
            plt.legend()
            plt.show()

        # Save
        grasp_data.save(f'./data/raw_data/{folder_name}/grasp={str(i)}/{object_name}')

        print('Depth shifting:', shift)
        print('Max depth in mm:', grasp_data.max_depths().max())
        print('Max force of N:', grasp_data.forces().max())
        print('Length of data:', len(grasp_data.forces()))
        print('\n')

        # Reset data
        grasp_data._reset_data()

    _plot_stream = False
    # _stream_thread.join()

    return True


if __name__ == "__main__":
    # Record grasp data for the given object
    object_name     = "unripe_banana"
    num_grasps      = 5
    collect_data_for_object(
        object_name, \
        num_grasps, \
        folder_name=object_name
    )