import cv2
import numpy as np
import time
import urllib.request
import csv
import warnings

from gelsight_wedge.src.gelsight.util.processing import warp_perspective
from gelsight_wedge.src.gelsight.util.fast_poisson import poisson_reconstruct
from gelsight_wedge.src.gelsight.util.helper import find_marker, interpolate_grad
from gelsight_wedge.src.gelsight.util.Vis3D import ClassVis3D

from threading import Thread

def read_csv(filename="config.csv"):
    rows = []
    with open(filename, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        _ = next(csvreader)
        for row in csvreader:
            rows.append((int(row[1]), int(row[2])))
    return rows

def trim(img):
    img[img < 0] = 0
    img[img > 255] = 255

# Measured from calipers on surface in x and y directions
WARPED_PX_TO_MM = (11, 11)
RAW_PX_TO_MM = (12.5, 11)
PX_TO_MM = np.sqrt(WARPED_PX_TO_MM[0]**2 + WARPED_PX_TO_MM[1]**2)

# Derived from linear fit from max depth measured to known calibration ball diameter
DEPTH_TO_MM = 21.5

# Threshold which is considered more than noise (significant penetration)
DEPTH_THRESHOLD = 0.1 # 0.03 # [mm]
AUTO_CLIP_OFFSET = 10 # [frames]

# Constants for cropping the edges of images
EDGE_CROP_MARGIN = 25 # [pixels]

# Size before and after warping + cropping
ORIGINAL_IMG_SIZE = (480, 640)
WARPED_IMG_SIZE = (400, 300)
WARPED_CROPPED_IMG_SIZE = (400 - EDGE_CROP_MARGIN*2, 300 - EDGE_CROP_MARGIN*2)

# Streaming parameters
STREAM_FPS = 30.0 # [1/s]

class GelSightWedgeVideo():
    '''
    Class to streamline processing of data from GelSight Wedge's
    '''
    def __init__(self, config_csv="./config.csv", IP=None, markers=False, warped_size=WARPED_IMG_SIZE, edge_crop_margin=EDGE_CROP_MARGIN):
        self.corners = read_csv(config_csv)         # CSV with pixel coordinates of mirror corners in the order (topleft,topright,bottomleft,bottomright)
        self.image_size = ORIGINAL_IMG_SIZE         # The size of original image from camera
        self.warped_size = warped_size              # The size of the image to output from warping process
        self.edge_crop_margin = edge_crop_margin    # Number of pixels to crop off edge after warping
        self.FPS = STREAM_FPS                       # Default FPS from Raspberry Pi camera
        self.PX_TO_MM = PX_TO_MM                    # Conversion from pixels to mm

        # If markers are being used, we will demark the image
        self.markers = markers

        # Compute size of the image after warping and edges are cropped
        self.cropped_size = (warped_size[0] - 2*self.edge_crop_margin, \
                             warped_size[1] - 2*self.edge_crop_margin)

        self._IP = IP                       # IP address of Raspberry Pi stream via mjpg_streamer
        self._url = ''                      # URL address of Raspberry Pi stream via mjpg_streamer
        if self._IP != None:
            self._url = self.IP_to_URL(self._IP)

        self._bytes = b''                   # Bytes data from URL during stream
        self._url_stream = None             # Access to URL data with urllib request
        self._curr_rgb_image = None         # Current raw RGB image streamed from camera
        self._stream_thread = Thread        # Streaming thread
        self._plot_thread = Thread          # Plotting thread
        self._stream_active = False         # Boolean of whether or not we're currently streaming
        self._plotting = False              # Boolean of whether or not we're plotting during stream

        self._raw_rgb_frames = []           # List of raw RGB frames recorded from camera (in shape: [(640,480,3)])
        self._warped_rgb_frames = []        # Images from camera now cropped / warped to the mirror (in shape: [(640,480,3)])
        self._diff_images = []              # Difference images relative to first frame (after warping) (in shape: [(640,480,3)])
        self._grad_images = []              # Gradient images estimated from RGB (in shape: [(640,480,2)])
        self._depth_images = []             # Depth images calculated (in shape: [(640,480)])
    
    # Clear all video data from the object
    def _reset_frames(self):
        self._curr_rgb_image = None
        self._raw_rgb_frames = []
        self._warped_rgb_frames = []
        self._diff_images = []
        self._grad_images = []
        self._depth_images = []
        return

    # Return stored raw frames from Raspberry Pi camera
    def raw_RGB_frames(self):
        return np.stack(self._raw_rgb_frames, axis=0)
    
    # Return cropped and warped images, compute if necessary
    def warped_RGB_frames(self):
        if len(self._warped_rgb_frames) != len(self._raw_rgb_frames):
            self._warped_rgb_frames = []
            for img in self.raw_RGB_frames():
                self._warped_rgb_frames.append(self.crop_image(self.warp_image(img)))
        return np.stack(self._warped_rgb_frames, axis=0)

    # Return cropped and warped difference images from initial frame to rest of video
    def diff_images(self):
        if len(self._diff_images) != len(self._raw_rgb_frames):
            self._diff_images = []
            ref_img = self.warped_RGB_frames()[0]
            # ref_img = cv2.GaussianBlur(ref_img, (13, 13), 0)
            for img in self.warped_RGB_frames():
                self._diff_images.append(self.calc_diff_image(ref_img, img))
        return np.stack(self._diff_images, axis=0)
    
    # Return gradients across images, compute if necessary
    def grad_images(self):
        if len(self._grad_images) != len(self._raw_rgb_frames):
            self._grad_images = []
            for frame in self.diff_images():
                self._grad_images.append(self.img2grad(frame))
        return np.stack(self._grad_images, axis=0)
    
    # Return depth at each frame, compute if necessary
    def depth_images(self):
        if len(self._depth_images) != len(self._raw_rgb_frames):
            self._depth_images = []
            for frame in self.diff_images():
                self._depth_images.append(self.img2depth(frame))
        return np.stack(self._depth_images, axis=0)
    
    # Return maximum depth from each depth image
    def max_depths(self):
        return np.max(self.depth_images(), axis=(1,2))
    
    # Return the maximum depth across all frames
    def max_depth(self):
        return self.max_depths().max()
    
    # Return mean depth from each depth image
    def mean_depths(self):
        return np.mean(self.depth_images(), axis=(1,2))
    
    # Warp raw image to mirror shape based on config corners
    def warp_image(self, img):
        return warp_perspective(img, self.corners, self.warped_size)
    
    # Crop edges of the image to avoid inaccuracies
    def crop_image(self, img):
        return img[EDGE_CROP_MARGIN:img.shape[0]-EDGE_CROP_MARGIN, EDGE_CROP_MARGIN:img.shape[1]-EDGE_CROP_MARGIN]
    
    # Calculate difference image from reference frame
    def calc_diff_image(self, ref_img, img):
        return (img * 1.0 - cv2.GaussianBlur(ref_img, (11, 11), 0) * 1.0) / 255 + 0.5
    
    # Remove markers from image with mask
    def demark_grad(self, diff_img, dx, dy):
        mask = find_marker(diff_img)
        dx = interpolate_grad(dx, mask)
        dy = interpolate_grad(dy, mask)
        return dx, dy

    # Calculate gradients from a cropped / warped difference image
    def img2grad(self, diff_img):
        dx = (diff_img[:, :, 1] - (diff_img[:, :, 0] + diff_img[:, :, 2]) * 0.5)
        dy = (diff_img[:, :, 0] - diff_img[:, :, 2])
        dx = dx / (1 - dx ** 2) ** 0.5 / 128
        dy = dy / (1 - dy ** 2) ** 0.5 / 128
        return dx, dy
    
    # Calculate depth based on image gradients
    def grad2depth(self, diff_img, dx, dy):
        if self.markers:
            dx, dy = self.demark_grad(diff_img, dx, dy)
        zeros = np.zeros_like(dx)
        unitless_depth = poisson_reconstruct(dy, dx, zeros)
        depth_in_mm = DEPTH_TO_MM * unitless_depth # Derived from linear fit of ball calibration
        return depth_in_mm

    # Calculate depth from a cropped / warped difference image
    def img2depth(self, diff_img):
        dx, dy = self.img2grad(diff_img)
        depth = self.grad2depth(diff_img, dx, dy)
        return depth
    
    # Convert IP addres to streaming url
    def IP_to_URL(self, IP, port=8080):
        return 'http://{}:{}/?action=stream'.format(IP, port)
    
    # Clear bytes data and open url
    def _prepare_stream(self):
        self._url_stream = urllib.request.urlopen(self._url)
        self._bytes = b''

    # Initiate streaming thread
    def start_stream(self, IP=None, plot=False, plot_diff=False, plot_depth=False):
        if IP != None:
            self._IP = IP
            self._url = self.IP_to_URL(self._IP)
        assert self._IP != None

        self._reset_frames()
        self._stream_active = True

        self._prepare_stream()

        self._stream_thread = Thread(target=self._stream, kwargs={})
        self._stream_thread.daemon = True
        self._stream_thread.start()
        time.sleep(1)

        if plot:
            self._start_plotting(plot_diff=plot_diff, plot_depth=plot_depth)
        return
    
    # Start plotting thread
    def _start_plotting(self, plot_diff=False, plot_depth=False, verbose=False):
        self._plotting = True
        self._plot_thread = Thread(target=self._plot, kwargs={'plot_diff': plot_diff, 'plot_depth': plot_depth, 'verbose': verbose})
        self._plot_thread.daemon = True
        self._plot_thread.start()
        return
    
    # During streaming, read object from URL
    def _decode_image_from_stream(self):
        self._bytes += self._url_stream.read(1024)
        a = self._bytes.find(b'\xff\xd8')
        b = self._bytes.find(b'\xff\xd9')
        if a != -1 and b != -1:
            jpg = self._bytes[a:b+2]
            self._bytes = self._bytes[b+2:]
            self._curr_rgb_image = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            self._raw_rgb_frames.append(self._curr_rgb_image)
            return True # Return whether or not an image was saved
        return False
    
    # Facilitate streaming thread, read data from Raspberry Pi camera
    def _stream(self, verbose=False):
        while self._stream_active:
            if verbose: print('Streaming...')
            _ = self._decode_image_from_stream()
        return

    # Plot relevant images during streaming
    def _plot(self, plot_diff=False, plot_depth=False, verbose=False):
        if plot_depth:  Vis3D = ClassVis3D(n=self.cropped_size[0], m=self.cropped_size[1])
        while self._plotting:
            if verbose: print('Plotting...')

            # Plot raw RGB image
            cv2.imshow('raw_RGB', self._curr_rgb_image)

            if plot_diff or plot_depth:
                diff_img = self.calc_diff_image(self.crop_image(self.warp_image(self._raw_rgb_frames[0])), self.crop_image(self.warp_image(self._curr_rgb_image)))

            # Plot difference image
            if plot_diff:
                cv2.imshow('diff_img', diff_img)

            # Plot depth in 3D
            if plot_depth:
                Vis3D.update(self.img2depth(diff_img) / PX_TO_MM)

            if cv2.waitKey(1) & 0xFF == ord('q'): # Exit windows by pressing "q"
                break
            if cv2.waitKey(1) == 27: # Exit window by pressing Esc
                break
        
        cv2.destroyAllWindows()
        return
    
    # Close URL and clear stream data
    def _wipe_stream_info(self):
        self._bytes = b''
        self._url_stream = None

    # Terminate streaming thread
    def end_stream(self, verbose=False):
        self._stream_active = False
        self._stream_thread.join()
        if self._plotting:
            self._stop_plotting()

        self._wipe_stream_info()
        time.sleep(1)
        if verbose: print('Done streaming.')
        return
    
    # Terminate plotting thread
    def _stop_plotting(self):
        self._plotting = False
        self._plot_thread.join()
        self._plot_thread = None
        return
    
    # Clip frames to indices
    def clip(self, i_start, i_end):
        i_start = max(0, i_start)
        i_end = min(i_end, len(self._raw_rgb_frames))

        # Clip all other frame data, if applicable
        if len(self._warped_rgb_frames) == len(self._raw_rgb_frames):
            self._warped_rgb_frames = self._warped_rgb_frames[i_start:i_end]
        if len(self._diff_images) == len(self._raw_rgb_frames):
            self._diff_images = self._diff_images[i_start:i_end]
        if len(self._grad_images) == len(self._raw_rgb_frames):
            self._grad_images = self._grad_images[i_start:i_end]
        if len(self._depth_images) == len(self._raw_rgb_frames):
            self._depth_images = self._depth_images[i_start:i_end]

        self._raw_rgb_frames = self._raw_rgb_frames[i_start:i_end]
        return
    
    # Automatically clip frames to first press via thresholding
    def auto_clip(self, depth_threshold=0.3*DEPTH_THRESHOLD, clip_offset=AUTO_CLIP_OFFSET, return_indices=False):
        i_start, i_end = len(self._raw_rgb_frames), len(self._raw_rgb_frames)-1
        max_depths = self.max_depths()
        for i in range(2, len(self._raw_rgb_frames)-2):
            # Check if next 3 consecutive indices are aboue threshold
            penetration = max_depths[i] > depth_threshold and max_depths[i+1] > depth_threshold and max_depths[i+2] > depth_threshold
            if penetration and i <= i_start:
                i_start = i

            # Check if past 3 consecutive indices are below threshold
            no_penetration = max_depths[i] < depth_threshold and max_depths[i-1] < depth_threshold and max_depths[i-2] < depth_threshold
            if no_penetration and i >= i_start and i <= i_end:
                i_end = i
            if no_penetration and i >= i_start: break

        if i_start >= i_end:
            warnings.warn("No press detected! Cannot clip.", Warning)            
        else:
            i_start_offset = max(0, i_start - clip_offset)
            i_end_offset = min(i_end + clip_offset, len(self._raw_rgb_frames)-1)
            self.clip(i_start_offset, i_end_offset)
            if return_indices:
                return i_start_offset, i_end_offset

    # Plot video for your viewing pleasure
    def watch(self, plot_diff=False, plot_depth=False):
        if plot_depth or plot_diff:
            diff_images = self.diff_images()
        if plot_depth:
            depth_images = self.depth_images()
            Vis3D = ClassVis3D(n=self.cropped_size[0], m=self.cropped_size[1])
        for i in range(len(self._raw_rgb_frames)):
            cv2.imshow('raw_RGB', self._raw_rgb_frames[i])
            if plot_diff:   cv2.imshow('diff_img', diff_images[i])
            if plot_depth:  Vis3D.update(depth_images[i] / PX_TO_MM)
            if cv2.waitKey(1) & 0xFF == ord('q'): # Exit windows by pressing "q"
                break
            if cv2.waitKey(1) == 27: # Exit window by pressing Esc
                break
            time.sleep(1/self.FPS)
        cv2.destroyAllWindows()
        return
    
    # Read frames from a video file
    def load(self, path_to_file):
        self._reset_frames()
        cap = cv2.VideoCapture(path_to_file)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self._raw_rgb_frames.append(frame)
        cap.release()
        return

    # Write recorded frames to video file
    def save(self, path_to_file):
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        video_writer = cv2.VideoWriter(path_to_file, fourcc, self.FPS, (self.image_size[1], self.image_size[0]))
        for frame in self._raw_rgb_frames:
            video_writer.write(frame)
        video_writer.release()
        return
    

if __name__ == "__main__":
    # Typical video recording workflow might be...
    wedge_video = GelSightWedgeVideo(IP="172.16.0.200", config_csv="./wedge_config/config_markers.csv")
    wedge_video.start_stream(plot=True, plot_diff=True, plot_depth=True)
    time.sleep(20)
    wedge_video.end_stream()
    print(wedge_video.max_depth())
    print('here')