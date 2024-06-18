import socket
import time
import warnings
import pickle
import numpy as np

from threading import Thread

from src.wedge_video import AUTO_CLIP_OFFSET

# For automatically clipping
FORCE_THRESHOLD = 0.75 # [N]

class ContactForce():
    '''
    Class to read and record contact force data sent from grasping gauge
    '''
    def __init__(self, IP=None, port=8888):
        self._IP = IP           # IP address where force values are written to from Raspberry Pi
        self._port = port       # Port where force values are written to from Raspberry Pi

        self._socket = None                 # Grants access to data from URL
        self._client_socket = None          # Socket that we read from
        self._stream_thread = None          # Thread to receive measurements from sensor and update value
        self._stream_active = False         # Boolean of whether or not we're currently streaming

        self._forces = []                   # Force value in Newtons at times requested
        self._times_requested = []          # Times of measurement requested
        self._forces_recorded = []          # All measurements from gauge at recorded times
        self._times_recorded = []           # Times of measurements recorded

    # Clear all force measurements from the object
    def _reset_values(self):
        self._times_requested = []
        self._times_recorded = []
        self._forces_recorded = []
        self._forces = []

    # Return array of force measurements
    def forces(self):
        return np.array(self._forces)

    # Open socket to begin streaming values
    def start_stream(self, IP=None, port=None, read_only=False, verbose=False, _open_socket=True):
        if IP != None:      self._IP = IP
        if port != None:    self._port = port
        assert self._IP != None and self._port != None

        self._reset_values()
        self._stream_active = True

        # Create a socket object and bind it to the specified address and port
        if _open_socket:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.bind((self._IP, self._port))
            self._socket.listen(1)
            self._client_socket, _ = self._socket.accept()

        self._stream_thread = Thread(target=self._stream, kwargs={'read_only': read_only, 'verbose': verbose})
        self._stream_thread.daemon = True
        self._stream_thread.start()
        return
    
    # Function to facilitate continuous reading of values from stream
    # (If read_only is True, will not save measurements to local force array.
    #   Instead, must directly execute _record_latest() to record the most recently recieved measurement.)
    def _stream(self, read_only=False, verbose=False):
        while self._stream_active:
            if verbose: print('Streaming force measurements...')
            self._read_value()
            if not read_only:
                self._request_value()
        return

    # Read force measurement from socket
    def _read_value(self, verbose=False):
        received_data = self._client_socket.recv(1024)
        if not received_data:
            raise ValueError()
        
        # Interpret data
        self._times_recorded.append(time.time())
        float_str = received_data.decode()
        if float_str.count('.') > 1:
            float_str = float_str[float_str.rfind('.', 0, float_str.rfind('.'))+3:]
        if float_str[1:].count('-') > 0:
            float_str = float_str[:float_str.find('-', 1)]
        self._forces_recorded.append(-float(float_str) * 0.00002)
        if verbose: print(self._forces_recorded[-1])
        return
    
    # Save the latest measurement from stream to local data
    def _request_value(self):
        self._times_requested.append(time.time())
        return
    
    # Smooth measurements based on time requested / recorded
    # Necessary because force bandwidth is slower than video
    def _post_process_measurements(self):
        self._forces = []
        for t_req in self._times_requested:
            for i in range(len(self._forces_recorded) - 1):
                if t_req > self._times_recorded[i] and t_req <= self._times_recorded[i+1]:
                    # Interpolate between measured values
                    F_t = self._forces_recorded[i] + (self._forces_recorded[i+1] - self._forces_recorded[i]) * \
                            (t_req - self._times_recorded[i])/(self._times_recorded[i+1] - self._times_recorded[i])
                    self._forces.append(F_t)
                    break
                elif i == len(self._forces_recorded) - 2:
                    # Take last measured
                    self._forces.append(self._forces_recorded[i+1])
        return
    
    # Close socket when done measuring
    def end_stream(self, verbose=False, _close_socket=True):
        self._stream_active = False
        self._stream_thread.join()
        if _close_socket:
            self._socket.close()
        self._post_process_measurements()
        if verbose: print('Done streaming.')
        return
    
    # Clip measurements between provided indices
    def clip(self, i_start, i_end):
        i_start = max(0, i_start)
        i_end = min(i_end, len(self._forces))
        self._forces = self._forces[i_start:i_end]
        return
    
    # Automatically clip based on threshold
    def auto_clip(self, force_threshold=FORCE_THRESHOLD, clip_offset=AUTO_CLIP_OFFSET, return_indices=False):
        forces = self.forces()
        i_start, i_end = len(forces), len(forces)-1
        for i in range(2, len(forces)-2):
            # Check if next 3 consecutive indices are aboue threshold
            penetration = forces[i] > force_threshold and forces[i+1] > force_threshold and forces[i+2] > force_threshold
            if penetration and i <= i_start:
                i_start = i

            # Check if past 3 consecutive indices are below threshold
            no_penetration = forces[i] < force_threshold and forces[i-1] < force_threshold and forces[i-2] < force_threshold
            if no_penetration and i >= i_start and i <= i_end:
                i_end = i
            if no_penetration and i >= i_start: break

        if i_start >= i_end:
            warnings.warn("No press detected! Cannot clip.", Warning)            
        else:
            i_start_offset = max(0, i_start - clip_offset)
            i_end_offset = min(i_end + clip_offset, len(forces)-1)
            self.clip(i_start_offset, i_end_offset)
            if return_indices:
                return i_start_offset, i_end_offset

    # Save array of force measurements to path
    def load(self, path_to_file):
        self._reset_values()
        assert path_to_file[-4:] == '.pkl'
        with open(path_to_file, 'rb') as file:
            self._forces = pickle.load(file)
        return

    # Save array of force measurements to path
    def save(self, path_to_file):
        assert path_to_file[-4:] == '.pkl'
        with open(path_to_file, 'wb') as file:
            pickle.dump(self.forces(), file)
        return
    

if __name__ == "__main__":
    # Read data from source
    contact_force = ContactForce(IP="172.16.0.69") # IP="172.16.0.50")
    contact_force.start_stream(verbose=True)
    print('If measurements not being received, ssh directly into the pi.')
    time.sleep(10)
    contact_force.end_stream()

    print(f'Read {len(contact_force.forces())} values in 3 seconds.')