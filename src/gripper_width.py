import pickle
import numpy as np
import time
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

from threading import Thread

SMOOTHING_POLY_ORDER = 5

class GripperWidth():
    '''
    Class to read and record contact gripper width over grasping
    '''
    def __init__(self, franka_arm=None):
        self._franka_arm = franka_arm       # Object to interface with the Panda arm
        self._stream_thread = None          # Thread to receive positions and update value
        self._stream_active = False         # Boolean of whether or not we're currently streaming

        self._widths = []                   # Gripper width in meters at times (after smoothing)
        self._times_requested = []          # Times we would like measurements for. Will interpolate measurements to get these values
        self._widths_recorded = []          # Widths recorded at each respective time
        self._times_recorded = []           # Times when measurement recorded

    # Clear all width measurements from the object
    def _reset_values(self):
        self._widths_recorded = []
        self._times_recorded = []
        self._times_requested = []
        self._widths = []

    # Return array of width measurements
    def widths(self):
        return np.array(self._widths)

    # Open socket to begin streaming values
    def start_stream(self, read_only=False, verbose=False):
        self._reset_values()
        self._stream_active = True
        self._stream_thread = Thread(target=self._stream, kwargs={'read_only': read_only, 'verbose': verbose})
        self._stream_thread.daemon = True
        self._stream_thread.start()
        return
    
    # Function to facilitate continuous reading of values from stream
    def _stream(self, read_only=False, verbose=False):
        while self._stream_active:
            if verbose: print('Streaming gripper width...')
            self._read_value()
            if not read_only:
                self._request_value()
        return
    
    # Save the latest measurement from stream to local data
    def _read_value(self):
        self._times_recorded.append(time.time())
        self._widths_recorded.append(self._franka_arm.get_gripper_width() - 0.0005) # [m]
        return
    
    # Request interpolation to this time point
    def _request_value(self):
        self._times_requested.append(time.time())
        return
    
    # Smooth measurements based on time requested / recorded
    # Necessary because width measurement bandwidth is slower than video
    def _post_process_measurements(self, plot_interpolation=False):
        self._widths = np.interp(self._times_requested, self._times_recorded, self._widths_recorded).tolist()

        if plot_interpolation:
            plt.plot(self._times_requested, self._widths, 'b-')
            plt.plot(self._times_recorded, self._widths_recorded, 'r.')
            plt.show()

        return
    
    # Close socket when done measuring
    def end_stream(self, verbose=False):
        self._stream_active = False
        self._stream_thread.join()
        self._post_process_measurements()
        if verbose: print('Done streaming.')
        return

    # Clip measurements between provided indices
    def clip(self, i_start, i_end):
        i_start = max(0, i_start)
        i_end = min(i_end, len(self._widths))
        self._widths = self._widths[i_start:i_end]
        return
    
    # Fit widths to continuous function and down sample to smooth measurements
    def smooth_gripper_widths(self, plot_smoothing=False, poly_order=SMOOTHING_POLY_ORDER):
        smooth_widths = []
        indices = np.arange(len(self._widths))
        p = np.polyfit(indices, self._widths, poly_order)
        for i in indices.tolist():
            w = 0
            for k in range(len(p)):
                w += p[k] * i**(poly_order-k)
            smooth_widths.append(w)

        if plot_smoothing:
            # Plot to check how the smoothing of data looks
            plt.plot(indices, self._widths, 'r.')
            plt.plot(indices, smooth_widths, 'b-')
            plt.show()

        self._widths = smooth_widths
        return

    # Save array of width measurements to path
    def load(self, path_to_file):
        self._reset_values()
        assert path_to_file[-4:] == '.pkl'
        with open(path_to_file, 'rb') as file:
            self._widths = pickle.load(file)
        return

    # Save array of width measurements to path
    def save(self, path_to_file):
        assert path_to_file[-4:] == '.pkl'
        with open(path_to_file, 'wb') as file:
            pickle.dump(self.widths(), file)
        return