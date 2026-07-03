# Auto detect the camera type
import sensor
NICLA = 1
OPENMV = 2
board = NICLA
if sensor.get_id() == sensor.GC2145:
    board = NICLA
elif sensor.get_id() == sensor.OV5640:
    board = OPENMV

import time
import micropython
import array
if board == NICLA:
    from pyb import UART, LED
elif board == OPENMV:
    from machine import LED, UART
from machine import I2C
#from vl53l1x import VL53L1X
import image
import math
from machine import Pin
import omv
import random
import asyncio
from ulab import numpy as np
from fast_np_count import fast_np_count as _fnc_call
_fnc_buf = bytearray(294 * 6 * 4)

# night mode:
if board == OPENMV:
    sensor.ioctl(sensor.IOCTL_SET_NIGHT_MODE, False)

# framesize setup, nicla cannot handle anything above HQVGA
# openmv, on the other hand, is recommended to run with QVGA
FRAME_SIZE = sensor.HQVGA
if board == NICLA:
    FRAME_SIZE = sensor.HQVGA
FRAME_PARAMS = None
if FRAME_SIZE == sensor.VGA:
    FRAME_PARAMS = [0, 0, 640, 480]  # Upper left corner x, y, width, height
elif FRAME_SIZE == sensor.HVGA:
    FRAME_PARAMS = [0, 0, 480, 320]
elif FRAME_SIZE == sensor.QVGA:
    FRAME_PARAMS = [0, 0, 320, 240]
elif FRAME_SIZE == sensor.HQVGA:
    FRAME_PARAMS = [0, 0, 240, 160]

# manual white balance - to be used with *get_gains.py* in the repository
# - see RGB gain readings in the console
if board == OPENMV:
    R_GAIN, G_GAIN, B_GAIN = [62, 60, 65]
elif board == NICLA:
    R_GAIN, G_GAIN, B_GAIN = [70, 66, 115]

# NOTE: MACROS for balloon detection #############
# Grid setup
N_ROWS = 14
N_COLS = 21

random.seed(time.time())
NUM_CELLS = 2
START_ROW = random.randint(0, N_ROWS-NUM_CELLS-1)
START_COL = random.randint(0, N_COLS-NUM_CELLS-1)

# Probablistic filter for balloon detection
FILTER = True
L_MAX = 4
L_MIN = -4

# print the stats in a 3x3 cell - for balloon color data collection
PRINT_CORNER = False
PLOT_METRIC = not PRINT_CORNER

# whether combine detction of green and purple as target balloons
# setting to False makes the detector to ignore colors other than the currently tracked one
COMBINE_TARGETS = False

# Color distribution - From the data analysis on balloon detection
# to be used with
if board == OPENMV:
    COLOR_DATA = {
        "purple": [[23.36438596491228, -39.03035087719298] ,  [[0.0648033747896978, 0.055988642505018865], [0.05598864250501887, 0.058994799618367275]]],
        "green": [[-20.875354509359045, 8.56792399319342] ,  [[0.060236044299697374, 0.04894408879238302], [0.04894408879238302, 0.0646636883366453]]],
        "blue": [[-0.08622366288492707, -21.085143165856294] ,  [[0.405955173972352, 0.11825244006317702], [0.11825244006317703, 0.04203593483634022]]],
        "red": [[35.494601328903656, 32.24958471760797] ,  [[0.24644136648942605, -0.2762493355717791], [-0.2762493355717791, 0.3291565103824226]]]
    }  # openmv
elif board == NICLA:
    COLOR_DATA = {
        "purple": [[23.36438596491228, -39.03035087719298] ,  [[0.0648033747896978, 0.055988642505018865], [0.05598864250501887, 0.058994799618367275]]],
        "green": [[-20.875354509359045, 8.56792399319342] ,  [[0.060236044299697374, 0.04894408879238302], [0.04894408879238302, 0.0646636883366453]]],
        "blue": [[13.596253902185223, -33.79474505723205] ,  [[0.13004108243366663, 0.06653979100910323], [0.06653979100910323, 0.04018167036620918]]],
        "red": [[53.568024861878456, 11.724102209944752] ,  [[0.03232165802049636, -0.021282694514752485], [-0.021282694514752485, 0.03727551356108449]]]
    }  # nicla


# color detection sensitivities:
# [0]: higher value for more sensitive detection
# [1]: for filtering out uniform colors such as a light source, higher -> less positive detection
# [2]: for filtering out messy background/environment, lower -> less positive detection
COLOR_SENSITIVITY = {
    "purple": [2.0, 3.0, 20.0],
    "green": [1.5, 5.0, 18.0],
    "blue": [1.5, 3.0, 18.0],
    "red": [3.0, 3.0, 24.0]
}

# range of the L channel values that guarantee valid detection
L_RANGE = {
    "purple": [30, 70],
    "green": [10, 80],
    "blue": [5, 40],
    "red": [2, 80]
}
# the minimum value given by a cell that we consider a positive detection
COLOR_CONFIDENCE = 0.3

# target balloon colors {color id: (RGB value for visulization)}
COLOR_TARGET = {"purple": (255,0,255),
                "green": (0,255,0),}
               # {"red": (255,0,0)}

# peer blimp color
COLOR_PEER = {} # {"red": (255, 0, 0)}

# parameters for removing noises and neighbor colors
NEIGHBOR_REMOVAL = True
NEIGHBOR_REMOVAL_FACTOR = 10.0
# NEIGHBOR FORMAT: {$neighbor: ($target, RGB value for visulization)}
COLOR_NEIGHBOR = {"blue": ("purple", (0,0,255))}


# tracking parameters
MAX_LOST_FRAME_TARGET = 20 # maximum number of frames without a positive balloon detection
                           # with data of the current detection still reported to the controller
MAX_LOST_FRAME_GOAL = 40   # maximum number of frames without a positive goal detection
MOMENTUM = 0.0 # keep the tracking result move if the detection is momentarily lost

# action kernel setup
FILTER_KERNEL = False #TODO (low priority): using LogOdd filter-based kernel
KERNEL_SIZE = 5 # a n x n kernel that summarizes the possibility scores inside to give the most
                # approachable detection, for target balloon detections only
KERNEL_CENTER_EMPH = 3 # how much weight we want the center of the kernel to count
kernel_scale = KERNEL_SIZE ** 2 + KERNEL_CENTER_EMPH - 1
# setting up the kernel matrix
kernel_matrix = []
for i in range(KERNEL_SIZE):
    kernel_row = []
    for j in range(KERNEL_SIZE):
        if i == KERNEL_SIZE // 2 and j == KERNEL_SIZE // 2:
            kernel_row.append(KERNEL_CENTER_EMPH)
        else:
            kernel_row.append(1)
    kernel_matrix.append(kernel_row)

# Performance profiling
PROFILING = True
CRAZY_RANDOM_OPTIMIZATION = False # TODO: not implemented with numpy batch processing yet
OPTIMIZATION_LEVEL = 0.2 # the higher, the faster, the less accurate
RANDOM_DACAY = 0.5*2*OPTIMIZATION_LEVEL # if we skip the detection of a cell, its confidence of a
                                        # positive detection decays exponentially


########## NOTE: MACROS for goal detection #############
NORM_LEVEL = 2  # Default to use 2-norm, change to 1 to reduce computation
MAX_FEATURE_DIST = 32767  # The maximum feature distance
# TODO: the following 4 parameters are questionable, viewing from now
# they basically control the motion speed of the tracking roi in the precense of a new/lost detection
# whenever the detection is lost on a frame, a nonzero `FF_POSITION` and `FF_SIZE`
# will try to expand the tracking roi to the entire frame, anchored in the frame center
# whenever a new detection is available, a nonzero `GF_POSITION` and `GF_SIZE`
# will try move the tracking roi to the detection smoothly
FF_POSITION = 0.0 # The forgetting factor for the position
FF_SIZE = 0.0 # The forgetting factor for the size
GF_POSITION = 0.3 # The gain factor for the position
GF_SIZE = 0.3 # The gain factor for the size
# setting FF_POSITION, FF_SIZE to 0 and GF_POSITION, GF_SIZE to 1
# will make the tracking identical to the current detection
frame_rate = 80 # target framerate that is a lie
TARGET_ORANGE = [(16, 80, 36, 56, 10, 12)] #(12, 87, -9, 62, 15, 50)
TARGET_COLOR2 = [(56, 76, -36, -15, 29, 58)]
TARGET_YELLOW = [(25, 88, -42, -17, 30, 56)] #[(40, 67, -31, -15, 27, 55)]
TARGET_COLOR = TARGET_YELLOW
WAIT_TIME_US = 1000000//frame_rate

SATURATION = 64 # global saturation for goal detection mode - not affected by ADVANCED_SENSOR_SETUP, defeult 64
CONTRAST = 40 # global contrast for goal detection mode - not affected by ADVANCED_SENSOR_SETUP, defeult 48
ADVANCED_SENSOR_SETUP = True # fine-tune the sensor for goal

################### NOTE: Balloon grid detection classes and functions #######################
# color confidence filter
class LogOddFilter:
    def __init__(self, row, col, p_det=0.99, p_ndet=0.01, p_x=0.5):
        # p_x: Probability of having a balloon in a cell
        self.init_belif = math.log(p_x/(1-p_x))
        self.l_det = math.log(p_det/(1-p_det))
        self.l_ndet = math.log(p_ndet / (1 - p_ndet))

        self.L = np.zeros(row*col, dtype=np.float)
        self.P = np.zeros(row*col, dtype=np.float)
        print("Initial belief=", self.init_belif,
              "L for detection=", self.l_det,
              "L for not detection", self.l_ndet )


    def update(self, measurements, l_max=L_MAX, l_min=L_MIN):
        measurement_array = np.array(measurements, dtype=np.float)
        measurement_array[measurement_array < 0.1] = self.l_ndet
        p_x_array = np.clip(0.48 + 0.52 * measurement_array, 0.0001, 0.9999)
        l_array = np.log(p_x_array / (1. - p_x_array))
        if FILTER_KERNEL: # TODO: to implement
            self.L += l_array - self.init_belif
            self.L = np.clip(self.L, l_min, l_max)
        else:
            self.L += l_array - self.init_belif
            self.L = np.clip(self.L, l_min, l_max)
        return self.L


    def probabilities(self):
        self.P = 1. / (1. + np.exp(-self.L))
        # for i, l in enumerate(self.L):
        #     self.P[i] = 1. / (1. + np.exp(-l))
        return self.P


# color detector based on distance to a line segment on L(AB) color space
class ColorDetector:
    # detector types: (T)arget, (P)eer, (N)eighbor
    def __init__(self, color_id,
                 detector_type: str,
                 line_ref,
                 max_dist, std_range, light_range, rgb,
                 mahalanobis=True, mu=None, sigma_inv=None, decay=4.,
                 neighbor=None):
        self.color_id = color_id
        self.line_ref = line_ref
        self.max_dist = max_dist
        self.std_range = std_range
        self.l_range = light_range
        self.rgb = rgb
        self.detector_type = detector_type[0].upper()
        if self.detector_type == "N":
            self.neighbor = neighbor

        # Color Gaussian distribution
        self.mahalanobis = mahalanobis
        if mahalanobis:
            self.mu_ndarray = np.array(mu, dtype=np.float)
            self.sigma_inv_ndarray = np.array(sigma_inv, dtype=np.float)
            self.decay = decay

        self.metric = [0. for _ in range(N_COLS*N_ROWS)]
        # Filter
        self.filter = LogOddFilter(N_ROWS, N_COLS)
        self.P = [0. for _ in range(N_COLS*N_ROWS)]


    def batch_distance_mahalanobis(self, stats):
        # Compute the Mahalanobis distance between a set of points and a distribution.
        stats_mean_ab = stats[:, 1:3]
        diff_mean_ab = stats_mean_ab - self.mu_ndarray
        # Compute the Mahalanobis distance in batch
        VQ = np.dot(diff_mean_ab, self.sigma_inv_ndarray)
        mahalanobis_sq = np.sum(VQ * diff_mean_ab, axis=1)
        # mahalanobis_dist = mahalanobis_sq ** 0.5
        return mahalanobis_sq


    def batch_update_cell(self, stats_array):
        d2_batch = self.batch_distance_mahalanobis(stats_array)
        score_batch = np.exp(-d2_batch / self.decay)

        # Check if standard deviation is in range
        std_cols = stats_array[:, 3:6]
        mask_std = np.any((std_cols < self.std_range[0]) | (std_cols > self.std_range[1]), axis=1)
        mask_l = (stats_array[:, 0] > self.l_range[1]) | (stats_array[:, 0] < self.l_range[0])
        mask = mask_std | mask_l
        score_batch[mask] = 0.0
        self.metric = score_batch
        return None


    def update_filter(self):
        self.filter.update(self.metric)
        self.P = self.filter.probabilities()
        return self.P


# grid-based color detector divisions
class Grid:
    def __init__(self, num_rows, num_cols, img_width, img_height):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.cell_width = int(img_width / num_cols)
        self.cell_height = int(img_height / num_rows)
        # self.grid_img_bf = sensor.alloc_extra_fb(num_cols, num_rows, sensor.GRAYSCALE).bytearray()
        self.image_bytearray = bytearray(num_cols * num_rows)



    def np_count(self, img, detectors):
        raw = img.bytearray()
        _fnc_call(raw, _fnc_buf)
        stats_array = np.frombuffer(_fnc_buf, dtype=np.float).reshape((294, 6))
        for detector in detectors.values():
            detector.batch_update_cell(stats_array)

    # an optimized counting method that ``batch-counts'' all cells using ulab numpy
    # CRAZY_RANDOM_OPTIMIZATION works in a different way
    """
    def np_count(self, img, detectors):
        raw = img.bytearray()
        pixels = np.frombuffer(raw, dtype=np.uint16)
        pf = np.array(pixels, dtype=np.float)   # convert to float first
        R = (pf // 2048) * (255.0 / 31.0)       # equivalent to >> 11
        G = ((pf // 32) % 64) * (255.0 / 63.0) # equivalent to >> 5 & 0x3F
        B = (pf % 32) * (255.0 / 31.0)          # equivalent to & 0x1F
        R, G, B = R / 255.0, G / 255.0, B / 255.0
        # Approximate gamma linearization (replacement for np.where which micropython doesn't have
        R_lin = (R / 12.92) * (R <= 0.04045) + (((R + 0.055) / 1.055) ** 2.4) * (R > 0.04045)
        G_lin = (G / 12.92) * (G <= 0.04045) + (((G + 0.055) / 1.055) ** 2.4) * (G > 0.04045)
        B_lin = (B / 12.92) * (B <= 0.04045) + (((B + 0.055) / 1.055) ** 2.4) * (B > 0.04045)
        X = 0.4124 * R_lin + 0.3576 * G_lin + 0.1805 * B_lin
        Y = 0.2126 * R_lin + 0.7152 * G_lin + 0.0722 * B_lin
        Z = 0.0193 * R_lin + 0.1192 * G_lin + 0.9505 * B_lin
        # XYZ → LAB
        X, Y, Z = X / 0.95047, Y / 1.00000, Z / 1.08883  # normalize by D65 white point
        # L = 116 * f(Y) - 16
        L = 116 * ((Y > 0.008856)*(Y ** (1/3)) + ((Y < 0.008856)*(7.787 * Y + 16/116))) - 16
        # A = 500 * (f(X) - f(Y))
        A = 500 * (((X > 0.008856)*(X ** (1/3)) + ((X < 0.008856)*(7.787 * X + 16/116)))-((Y > 0.008856)*(Y ** (1/3)) + ((Y < 0.008856)*(7.787 * Y + 16/116))))
        # B_lab = 200 * (f(Y) - f(Z))
        B_lab = 200 * (((Y > 0.008856)*(Y ** (1/3)) + ((Y < 0.008856)*(7.787 * Y + 16/116))) - ((Z > 0.008856)*(Z ** (1/3)) + ((Z < 0.008856)*(7.787 * Z + 16/116))))
        lab = np.array([L, A, B_lab]).transpose()  # (38400, 3)
        img_arr = lab.reshape(160, 240, 3)
        img_arr = img_arr[:14*11, :21*11, :]  # crop → (154, 231, 3)
        grid = img_arr.reshape(14, 11, 21, 11, 3)
        grid = grid.transpose(0, 2, 1, 3, 4)
        cells = grid.reshape(294, 121, 3)
        means = np.mean(cells, axis=1)  # (294, 3)
        stds = np.std(cells, axis=1)  # (294, 3)
        stats_array = np.concatenate((means, stds), axis=1)
        for detector in detectors.values():
            detector.batch_update_cell(stats_array)
    @micropython.native  # to delete if not necessary. It did improve the inference speed by roughly 1 ms though
    def np_count(self, img, detectors):
        num_rows = self.num_rows
        num_cols = self.num_cols
        cell_width = self.cell_width
        cell_height = self.cell_height
        detector_values = detectors.values()

        stats_array = np.zeros((num_rows*num_cols, 6), dtype=np.float)
        for row in range(num_rows):
            row_start = row * cell_height
            print_corner_row = PRINT_CORNER and START_ROW < row < START_ROW + NUM_CELLS + 1

            for col in range(num_cols):
                col_start = col * cell_width
                roi = (col_start, row_start, cell_width, cell_height)
                s = img.get_statistics(roi=roi)

                # Cache the statistical values
                l_mean = s.l_mean()
                a_mean = s.a_mean()
                b_mean = s.b_mean()
                l_stdev = s.l_stdev()
                a_stdev = s.a_stdev()
                b_stdev = s.b_stdev()

                stats_array[self._matrix_to_index(row, col), :] = np.array([l_mean, a_mean, b_mean, l_stdev, a_stdev, b_stdev])

                if print_corner_row and START_COL < col < START_COL + NUM_CELLS+1:
                    print((a_mean, b_mean), end=', ')

            if print_corner_row:
                print()

        for detector in detector_values:
            detector.batch_update_cell(stats_array)
    """

    def plot_metric(self, metrics, rgb):
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                roi = (col * self.cell_width, row * self.cell_height, self.cell_width+1, self.cell_height+1)

                metric = metrics[row*self.num_cols + col]

                # Draw the number of ones in the corner of the cell
                # img.draw_string(roi[0], roi[1],str(int(metric*10)) , color=(0,255,0))
                if metric >= 0.2:
                    # Draw the ROI on the image
                    img.draw_rectangle(roi, color=(int(metric*rgb[0]),int(metric*rgb[1]),int(metric*rgb[2])), thickness=1)


    def plot_data_collection(self):
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                if START_ROW<row<START_ROW+NUM_CELLS+1 and START_COL<col<START_COL+NUM_CELLS+1:
                    # Draw the ROI on the image
                    roi = (col * self.cell_width, row * self.cell_height, self.cell_width+1, self.cell_height+1)
                    img.draw_rectangle(roi, color=(0, 0, 0), thickness=1)


    def statistics(self, nones):
        # Compute the mean of the sample vector
        mean = sum(nones) / len(nones)
        # Compute the sum of squared differences from the mean
        squared_diff_sum = sum((x - mean) ** 2 for x in nones)
        # Compute the variance
        variance = squared_diff_sum / len(nones)

        return mean, variance


    def _index_to_matrix(self, i:int) -> Tuple[int, int]:
        row = i // self.num_cols
        col = i % self.num_cols
        return row, col


    def _matrix_to_index(self, row: int, col: int) -> int:
        index = row * self.num_cols + col
        return index


    # an ulab-assisted optimized kernel count method
    def kernel_count(self, metric):
        pixel_scale = 256
        metric_image = None
        if FILTER_KERNEL:
            ndarray_morphed_metric = np.array(metric, dtype=np.float)*pixel_scale/kernel_scale
        else:
            ndarray_metric = np.array(metric*pixel_scale, dtype=np.float).reshape((N_ROWS, N_COLS))
            metric_image = image.Image(ndarray_metric, buffer=self.image_bytearray)
            metric_image.morph(KERNEL_SIZE//2, kernel_matrix)
            ndarray_morphed_metric = metric_image.to_ndarray(dtype="f").flatten()

        max_val = np.max(ndarray_morphed_metric)
        # print(np.mean(abs(ndarray_morphed_metric - max_val)))
        # print(np.std(abs(ndarray_morphed_metric - max_val)))
        indices = np.where(abs(ndarray_morphed_metric - max_val) < 1, np.arange(len(ndarray_morphed_metric)), -1)
        matching_indices = indices[indices != -1]
        same_val_rc = [self._index_to_matrix(i) for i in matching_indices]
        return None, None, max_val*kernel_scale//pixel_scale, same_val_rc


    @micropython.native
    def optimized_loop(self, metric):
        num_rows = self.num_rows
        num_cols = self.num_cols
        metric_len = len(metric)
        half_kernel = KERNEL_SIZE // 2
        max_col = -1
        max_row = -1
        max_val = -1
        same_val_rc = []

        # Precompute kernel offsets
        kernel_offsets = [
            (dr, dc)
            for dr in range(-half_kernel, half_kernel + 1)
            for dc in range(-half_kernel, half_kernel + 1)
        ]

        for i in range(metric_len):
            row = i // num_cols
            col = i % num_cols
            total = 0

            for dr, dc in kernel_offsets:
                r = row + dr
                c = col + dc
                if 0 <= r < num_rows and 0 <= c < num_cols:
                    mk = r * num_cols + c
                    val = metric[mk]
                    if dr == 0 and dc == 0:
                        total += KERNEL_CENTER_EMPH * val
                    else:
                        total += val

            if total > max_val:
                max_row = row
                max_col = col
                max_val = total
                same_val_rc = [[max_row, max_col]]
            elif total == max_val:
                same_val_rc.append([row, col])

        return max_row, max_col, max_val, same_val_rc


    def action(self, metric):
        if PROFILING:
            start_time = time.ticks_us()
        _, _, max_val, same_val_rc = self.kernel_count(metric)
        if PROFILING:
            print("Pooling: {} us".format(time.ticks_diff(time.ticks_us(), start_time)))
        sum_row, sum_col = 0, 0
        for rc in same_val_rc:
            sum_row += rc[0]
            sum_col += rc[1]

        max_row = sum_row/len(same_val_rc)
        max_col = sum_col/len(same_val_rc)

        x, y = max_col * self.cell_width + self.cell_width // 2, max_row * self.cell_height + self.cell_height // 2
        return x, y, max_val


# Class for the balloon detection to use with the tracker in the main function
class BalloonTracker:
    def __init__(self, grid, detectors, filter_on=FILTER):
        self.grid = grid
        self.detectors = detectors
        self.filter_on = filter_on
        self.flag = int(0)
        if board == NICLA:
            self.led_red = LED(1)
            self.led_green = LED(2)
            self.led_blue = LED(3)
        elif board == OPENMV:
            self.led_red = LED("LED_RED")
            self.led_green = LED("LED_GREEN")
            self.led_blue = LED("LED_BLUE")
        self.detection_count = 0
        self.ux = -1
        self.uy = -1
        self.val = 0
        self.velx = 0
        self.vely = 0
        self.balloon_color = None


    def track(self, img):
        if PROFILING:
            start_time = time.ticks_ms()
        self.grid.np_count(img, self.detectors)
        if PROFILING:
            print("counting total: {} ms".format(time.ticks_diff(time.ticks_ms(), start_time)))
        for color, detector in self.detectors.items():
            metric = detector.metric

            if self.filter_on:
                metric = detector.update_filter()
            else:
                detector.P = detector.metric
            if PLOT_METRIC:
                grid.plot_metric(metric, detector.rgb)

        if PRINT_CORNER:
            grid.plot_data_collection()

        # Discard neighbor colors of certain targets
        # What we do here is to comapre the designated neighbor value and the associated target value,
        #   then discard the cells with a high probability of false positive on the neighbor color over the target color
        # TODO: we are not considering peer detection here yet
        true_positive_targets = {}
        for color, detector in self.detectors.items():
            if detector.detector_type == "T":
                if color in true_positive_targets:
                    pass
                else:
                    true_positive_targets[color] = detector.P
            elif detector.detector_type == "N":
                neighbor_target = detector.neighbor
                true_positive_targets[neighbor_target] = [t if (t > NEIGHBOR_REMOVAL_FACTOR*n+0.8 and n < 0.02) else 0 for t,n in zip(self.detectors[neighbor_target].P, detector.P)]

        # decide which color to track
        if self.balloon_color == None:
            # initialize color of the balloon to track
            for target, value in true_positive_targets.items():
                if self.balloon_color == None or max(value) > max(true_positive_targets[self.balloon_color]):
                    self.balloon_color = target


        if self.balloon_color == None:
            metric_grid = [0.0 for _ in range(N_ROWS * N_COLS)]
        else:
            metric_grid = self.detectors[self.balloon_color].P

        if COMBINE_TARGETS:
            for target, value in true_positive_targets.items():
                for i in range(len(metric_grid)):
                    metric_grid[i] = max(metric_grid[i], value[i])

        total_score = max(metric_grid)
        ux, uy, val = grid.action(metric_grid)

        if total_score > COLOR_CONFIDENCE:
            if self.ux == -1:
                self.ux = ux
                self.uy = uy
                self.val = val
                self.velx = ux - img.width()/2
                self.vely = uy - img.height()/2
                self.flag = 1
            else:
                self.velx = ux - self.ux
                self.vely = uy - self.uy
                self.ux = 0.55*ux + 0.45*self.ux
                self.uy = 0.55*uy + 0.45*self.uy
                self.val = 0.35*self.val + 0.65*val
                self.flag = 3 - self.flag
                if PLOT_METRIC:
                    img.draw_circle(int(ux), int(uy), int(5), color=(0,255,255), thickness=4, fill=True)
            self.detection_count = MAX_LOST_FRAME_TARGET

            if COMBINE_TARGETS:
                self.led_red.on()
                self.led_blue.on()
                self.led_green.on()
            elif self.balloon_color == "purple":
                self.led_red.on()
                self.led_blue.on()
                self.led_green.off()
            elif self.balloon_color == "green":
                self.led_red.off()
                self.led_blue.off()
                self.led_green.on()
            elif self.balloon_color == "red":
                self.led_red.on()
                self.led_blue.off()
                self.led_green.off()
            elif self.balloon_color == "blue":
                self.led_red.off()
                self.led_blue.on()
                self.led_green.off()
            else:
                self.led_red.on()
                self.led_blue.on()
                self.led_green.on()
        else:
            self.detection_count -= 1

            self.led_red.toggle()
            self.led_blue.toggle()
            self.led_green.toggle()

            if not self.ux == -1:
                self.ux += self.velx * MOMENTUM
                self.uy += self.vely * MOMENTUM
                self.velx *= 0.75
                self.vely *= 0.75

        if self.ux > img.width():
            self.ux = img.width()
        elif self.ux < 0:
            self.ux = 0
        if self.uy > img.height():
            self.uy = img.height()
        elif self.uy < 0:
            self.uy = 0

        # LED indicator and flag toggle
        if self.detection_count > 0 and PLOT_METRIC:
            # Draw the ROI on the image
            img.draw_circle(int(self.ux), int(self.uy),
                            int(5*self.val), color=(255,0,0),
                            thickness=1, fill=False)
        else:
            self.balloon_color = None
            self.detection_count = 0
            self.ux = -1
            self.uy = -1
            self.val = 0

            self.led_red.on()
            self.led_blue.on()
            self.led_green.on()

            self.flag = 0
            self.velx = 0
            self.vely = 0

        x_roi, y_roi = int(self.ux), int(self.uy)
        w_roi, h_roi = int(10*self.val), int(10*self.val)
        x_value, y_value = int(self.ux), int(self.uy)
        return [x_roi, y_roi, w_roi, h_roi, x_value, y_value, w_roi, h_roi, 0.0], self.flag | 0x40


################### NOTE: Goal detection classes and functions ###################
# a moving ROI that shrinks towards a current detection
# and expands to the framesize without an active detection
class MemROI:
    def __init__(self, frame_params:list = FRAME_PARAMS,
                 min_windowsize:int=20, ffp:float=FF_POSITION, ffs:float=FF_SIZE,
                 gfp:float=GF_POSITION, gfs:float=GF_SIZE)->None:
        # @description: Constructor of the ROI object that memorizes previous states.
        # @param {list} frame_params: The parameters of the frame [x0, y0, max_w, max_h]
        # @param {int} min_windowsize: The minimum size of the tracking window
        # @param {float} ffp: The forgetting factor for the position
        # @param {float} ffs: The forgetting factor for the size
        # @param {float} gfp: The gain factor for the position
        # @param {float} gfs: The gain factor for the size
        print(frame_params)
        self.roi = frame_params # [x0, y0, w, h]
        self.frame_params = frame_params  # [x0, y0, max_w, max_h]
        self.min_windowsize = min_windowsize
        self.ffp = ffp
        self.ffs = ffs
        self.gfp = gfp
        self.gfs = gfs


    def _clamp(self)->None:
        # @description: Clamp the ROI to be within the frame.
        # Ensure the ROI's top-left corner is within the bounds.
        self.roi[0] = max(self.frame_params[0], self.roi[0])
        self.roi[1] = max(self.frame_params[1], self.roi[1])

        # Ensure the ROI's bottom-right corner is within the bounds.
        self.roi[2] = min(self.frame_params[2] - self.roi[0], self.roi[2])
        self.roi[3] = min(self.frame_params[3] - self.roi[1], self.roi[3])


    def _center(self, rect:list)->tuple:
        # @description: Calculate the center of the rectangle.
        # @param {list} rect: The rectangle to be calculated [Upper left corner x, y, w, h]
        # @return {tuple} The center of the rectangle
        if len(rect) != 4:
            raise ValueError("Cannot calculate the center of the rectangle! The rectangle must be in the form of [x0, y0, w, h]")
        return (rect[0] + rect[2] / 2, rect[1] + rect[3] / 2)

    def _map(self, rect1:list, rect2:list, flag:int)->list:
        # @description: Map rect1 to rect2 by the forgetting factors.
        # @param       {list} rect1: Rectangle to be mapped [x0, y0, w, h]
        # @param       {list} rect2: Rectangle to be mapped to [x0, y0, w, h]
        # @param       {int} flag: 0 for forgetting factor, 1 for gain factor
        # @return      {list} The mapped rectangle [x0, y0, w, h]
        # Get the centers of the rectangles
        cx1, cy1 = self._center(rect1) # Center x, y
        cx2, cy2 = self._center(rect2) # Center x, y

        fp = 0.0
        fs = 0.0
        if flag == 0:
            fp = self.ffp
            fs = self.ffs
        elif flag == 1:
            fp = self.gfp
            fs = self.gfs
        else:
            raise ValueError("Invalid factor setting! flag must be 0(forget) or 1(gain).")

        # Calculate new center by shifting rect1's center towards rect2's center by alpha
        new_cx = cx1 + fp * (cx2 - cx1)
        new_cy = cy1 + fp * (cy2 - cy1)

        # Shift the size of rect1 towards rect2's size by beta
        new_w = rect1[2] + fs * (rect2[2] - rect1[2])
        new_h = rect1[3] + fs * (rect2[3] - rect1[3])
        return [new_cx - new_w / 2, new_cy - new_h / 2, new_w, new_h]


    def update(self, new_roi:list=None)->None:
        # @description: Update the ROI with a new ROI.
        # @param {list} new_roi: The new roi to map to [x0, y0, w, h]
        if not new_roi: # No new detection is found in the maximum tracking window
            self.roi = self._map(self.roi, self.frame_params, 0) # Map the ROI to the frame by the forgetting factors
        else:
            # Scale up the new_roi
            expanded_roi = [new_roi[0] - 0.1 * new_roi[2],
                            new_roi[1] - 0.1 * new_roi[3],
                            1.2 * new_roi[2],
                            1.2 * new_roi[3]]

            self.roi = self._map(self.roi, expanded_roi, 1) # Map the ROI to the new_roi by the gain factors
        self._clamp() # Clamp the ROI to be within the frame

    def reset(self)->None:
        # @description: Reset the ROI to the frame.
        self.roi = self.frame_params

    def get_roi(self)->list:
        # @description: Get the ROI.
        # @return      {list} The ROI [x0, y0, w, h]
        return [math.ceil(value)+1 for value in self.roi]


# determine the shape of a detected goal
class ShapeDetector:
    def __init__(self, gridsize):
        self.gridsize = gridsize
        self.binary_image = sensor.alloc_extra_fb(gridsize, gridsize, sensor.BINARY)
        # Pre-create shapes
        self.img_triangle = self.create_triangle(gridsize)
        self.tot_tri = 1#sum(self.img_triangle.get_pixel(j, i) for i in range(self.gridsize) for j in range(self.gridsize))
        self.img_circle = self.create_circle(gridsize)
        self.tot_cir = 1#sum(self.img_circle.get_pixel(j, i) for i in range(self.gridsize) for j in range(self.gridsize))
        self.img_square = self.create_square(gridsize)
        self.tot_squ = 1#sum(self.img_square.get_pixel(j, i) for i in range(self.gridsize) for j in range(self.gridsize))


    def destruct(self):
        sensor.dealloc_extra_fb()
        sensor.dealloc_extra_fb()
        sensor.dealloc_extra_fb()


    def create_triangle(self, gridsize):
        # Allocate frame buffer for triangle
        # img = sensor.alloc_extra_fb(gridsize, gridsize, sensor.BINARY)
        # # Draw an isosceles triangle
        # img.draw_line(gridsize // 2, gridsize - 1, 0, 0, color=255, thickness=1)
        # img.draw_line(gridsize // 2, gridsize - 1, gridsize - 1, 0, color=255, thickness=1)
        # img.draw_line(0, 0, gridsize - 1, 0, color=255, thickness=2)
        img = sensor.alloc_extra_fb(gridsize, gridsize, sensor.BINARY)
        # Flipped isosceles triangle
        img.draw_line(gridsize // 2, 0, 0, gridsize - 1, color=255, thickness=1)  # Apex to left base
        img.draw_line(gridsize // 2, 0, gridsize - 1, gridsize - 1, color=255, thickness=1)  # Apex to right base
        img.draw_line(0, gridsize - 1, gridsize - 1, gridsize - 1, color=255, thickness=1)  # Base line
        return img


    def create_circle(self, gridsize):
        # Allocate frame buffer for circle
        img = sensor.alloc_extra_fb(gridsize, gridsize, sensor.BINARY)
        radius = (gridsize)// 2
        img.draw_circle(gridsize // 2, gridsize // 2, radius, color=255, fill=False, thickness=2)
        if (gridsize % 2 == 0):
            img.draw_circle((gridsize) // 2 -1, (gridsize) // 2 -1, radius, color=255, fill=False)
            img.draw_circle((gridsize) // 2 , (gridsize) // 2 -1, radius, color=255, fill=False)
            img.draw_circle((gridsize) // 2 -1, (gridsize) // 2 , radius, color=255, fill=False)
        return img


    def create_square(self, gridsize):
        # Allocate frame buffer for square
        img = sensor.alloc_extra_fb(gridsize, gridsize, sensor.BINARY)
        # Draw a square
        img.draw_rectangle(0, 0, gridsize-0, gridsize-0, color=255, fill=False, thickness=1)
        return img


    def downsample_and_average(self, roi_img):
        # Custom function to process and downsample the ROI
        # Use the mean_pooled function to simplify the pooling
        src_width, src_height = roi_img.width(), roi_img.height()
        block_width = src_width // self.gridsize
        block_height = src_height // self.gridsize
        width_remainder = src_width % self.gridsize
        height_remainder = src_height % self.gridsize
        if (block_width == 0 or block_height ==0):
            return self.binary_image

        # Iterate over each block
        for i in range(self.gridsize):
            for j in range(self.gridsize):
                current_block_width = block_width + (1 if j < width_remainder else 0)
                current_block_height = block_height + (1 if i < height_remainder else 0)
                x = sum(block_width + (1 if m < width_remainder else 0) for m in range(j))
                y = sum(block_height + (1 if n < height_remainder else 0) for n in range(i))

                # Define the sub ROI for this block
                sub_roi = (x, y, current_block_width, current_block_height)
                # Get statistics for the sub ROI
                stats = roi_img.get_statistics(roi=sub_roi)
                # Calculate the mean and determine if the block is predominantly white or black
                mean_val = stats.mean()
                binary_val = 1 if mean_val > 60 else 0  # Threshold the mean to create a binary image
                self.binary_image.set_pixel(j, i, binary_val)
        return self.binary_image


    def detect_shape(self, roi_img):
        mean_pooled_img = self.downsample_and_average(roi_img.to_grayscale())

        center_size = 3#1#3 if self.gridsize > 3 else 1  # Only 3x3 or 1x1, adjust if needed
        start = (self.gridsize - center_size) // 2
        end = start + center_size
        center_sum = sum(mean_pooled_img.get_pixel(j, i) for i in range(start, end) for j in range(start, end))
        # print(center_sum)
        if (center_sum >= center_size**2 * .66):
            return "not"
        tot_sum = sum(mean_pooled_img.get_pixel(j, i) for i in range(0, self.gridsize) for j in range(0, self.gridsize))
        if (tot_sum == 0):
            return "not"

        # Prepare for shape comparison
        overlap_triangle = 0
        overlap_circle = 0
        overlap_square = 0

        # Calculate overlaps by comparing each pixel
        for i in range(self.gridsize):
            for j in range(self.gridsize):
                if mean_pooled_img.get_pixel(j, i) == 1:  # Check if the ROI pixel is white
                    if self.img_triangle.get_pixel(j, i) == 1:
                        overlap_triangle += 1
                    if self.img_circle.get_pixel(j, i) == 1:
                        overlap_circle += 1
                    if self.img_square.get_pixel(j, i) == 1:
                        overlap_square += 1

        # print("Overlap Triangle:", overlap_triangle/self.tot_tri, "Overlap Circle:", overlap_circle/self.tot_cir, "Overlap Square:", overlap_square/self.tot_squ)

        # Identify which shape it is based on maximum overlap
        if overlap_triangle/self.tot_tri > overlap_circle/self.tot_cir and overlap_triangle/self.tot_tri > overlap_square //self.tot_squ:
            return "triangle"
        elif overlap_square/self.tot_squ > overlap_circle/self.tot_cir:
            return "square"
        else:
            return "circle"


    def extract_valid_roi(self, img, blob, thresholds, min_edge_distance=0):
        # Extracts and validates the ROI from the given blob based on minimum distance to the edge
        left_distance = blob.x()
        right_distance = img.width() - (blob.x() + blob.w())
        top_distance = blob.y()
        bottom_distance = img.height() - (blob.y() + blob.h())
        min_distance = min(left_distance, right_distance, top_distance, bottom_distance)

        if min_distance >= min_edge_distance:
            roi_width = min(int(img.width() * 1), blob.w())
            roi_height = min(int(img.height() * 1), blob.h())
            if roi_width // self.gridsize > 0 and roi_height // self.gridsize > 0:
                roi = (max(0, blob.x()), max(0, blob.y()), roi_width, roi_height)
                x_scale = 1
                y_scale = 1
                if roi_width > img.width()/3:
                    y_scale = 1- roi_height/img.height()
                if roi_height > img.height()/3:
                    y_scale = 1- roi_height/img.height()
                try:
                    roi_img = img.copy(x_scale = x_scale, y_scale = y_scale, roi=roi).binary(thresholds)
                except:
                    return None, None
                return roi_img, roi

        return None, None  # Return None if no valid ROI found


# a moving blob object that memorizes previous blobs detected
class CurBLOB:
    def __init__(
        self,
        initial_blob,
        norm_level: int = NORM_LEVEL,
        feature_dist_threshold: int = 400,
        window_size=4,
        blob_id=0,
    ) -> None:
        # @description: Constructor of the blob object that memorizes previous states.
        # @param {*} initial_blob: The first blob appeared after the reset
        # @param {int} norm_level: The norm level for the feature distance (default to L2)
        # @param {int} feature_dist_threshold: The threshold for the feature distance (default to 100)
        # @param {*} window_size: The window size for the moving average (default to 3)
        # @param {*} blob_id: The id of the blob
        if initial_blob:
            self.blob_history = [initial_blob]
            self.feature_vector = [
                initial_blob.x(),
                initial_blob.y(),
                initial_blob.w(),
                initial_blob.h(),
                initial_blob.rotation_deg(),
            ]
        else:
            self.blob_history = None
            self.feature_vector = None


        self.norm_level = norm_level
        self.untracked_frames = 0  # number of frames that the blob is not tracked
        self.feature_dist_threshold = (
            feature_dist_threshold  # threshold for feature distance
        )
        self.window_size = window_size  # window size for moving average
        self.id = blob_id  # id of the blob

    def reset(self) -> None:
        # @description: Reset the current blob
        self.blob_history = None
        self.feature_vector = None
        self.untracked_frames = 0

    def reinit(self, blob: image.blob) -> None:
        # @description: Reinitialize the current blob with a new blob
        # @param       {image.blob} blob: The new blob to be reinitialized with
        self.blob_history = [blob]  # reset the blob history
        self.feature_vector = [
            blob.x(),
            blob.y(),
            blob.w(),
            blob.h(),
            blob.rotation_deg(),
        ]
        self.untracked_frames = 0  # reset the untracked frames

    def compare(self, new_blob: image.blob) -> int:
        # @description: Compare the feature distance between the current blob and a new blob
        # @param {image.blob} new_blob: The new blob to be compared with
        # @return {int} The feature distance between the current blob and the new blob
        new_feature = (
            new_blob.x(),
            new_blob.y(),
            new_blob.w(),
            new_blob.h(),
            new_blob.rotation_deg(),
        )  # get the feature vector of the new blob
        old_feature = self.feature_vector  # get the feature vector of the current blob
        if (
            not new_blob.code() == self.blob_history[-1].code()
        ):  # Check if the color is the same
            return MAX_FEATURE_DIST  # Different colors automatically grant a maximum distance
        elif self.norm_level == 1:  # The norm level is L1
            return sum([abs(new_feature[i] - old_feature[i]) for i in range(5)])
        elif self.norm_level == 2:  # The norm level is L2
            return math.sqrt(
                sum([(new_feature[i] - old_feature[i]) ** 2 for i in range(5)])
            )

    def update(self, list_of_blob: list) -> list:
        # @description: Update the current blob with the best candidate blob in the list of blobs
        # @param {list} list_of_blob: The list of blobs to be compared with
        # @return {list} The rectangle of the best candidate blob
        if list_of_blob is None:  # For the case that no blob is detected
            self.untracked_frames += 1
            return None

        min_dist = 32767
        candidate_blob = None
        # Find the blob with minimum feature distance
        for b in list_of_blob:  # This should reference the input parameter 'list_of_blob', not 'blobs'
            dist = self.compare(b)
            if dist < min_dist:
                min_dist = dist
                candidate_blob = b

        if min_dist < self.feature_dist_threshold:
            # Update the feature history if the feature distance is below the threshold
            self.untracked_frames = 0  # Reset the number of untracked frames
            history_size = len(
                self.blob_history
            )  # Get the number of blobs in the history
            self.blob_history.append(candidate_blob)
            # Calculate the feature vector of the candidate blob
            candidate_feature = (
                candidate_blob.x(),
                candidate_blob.y(),
                candidate_blob.w(),
                candidate_blob.h(),
                candidate_blob.rotation_deg(),
            )

            if history_size < self.window_size:
                # Calculate the moving average directly if the blob history is not filled
                for i in range(5):
                    # calculate the moving average
                    self.feature_vector[i] = (self.feature_vector[i]*history_size +
                        candidate_feature[i])/(history_size + 1)
            else:
                # Remove the oldest blob from the history and calculate the moving average
                oldest_blob = self.blob_history[0]
                oldest_feature = (
                    oldest_blob.x(),
                    oldest_blob.y(),
                    oldest_blob.w(),
                    oldest_blob.h(),
                    oldest_blob.rotation_deg(),
                )
                self.feature_vector = [
                    (current * self.window_size - old + new) / history_size
                    for current, old, new in zip(
                        self.feature_vector, oldest_feature, candidate_feature
                    )
                ]
                self.blob_history.pop(0)
            return candidate_blob.rect()
        else:
            # Increase the number of untracked frames if no good candidate is found
            self.untracked_frames += 1
            return None


# tracking bounding box for the goal
class GoalTracker:
    def __init__(
        self,
        thresholds: list,
        clock: time.clock,
        show: bool = True,
        max_untracked_frames: int = 8,
        LEDpin: str = "PG12",
        sensor_sleep_time: int = 50000,
    ) -> None:
        # @param {list} thresholds: The list of thresholds for the goal
        # @param {time} clock: The clock to track the time
        # @param {bool} show: Whether to show the image (default: True)
        # @param {int} max_untracked_frames: The maximum number of untracked frames until the tracker resets (default: 5)
        # @param {str} LEDpin: The pin of the IR LED (default: "PG12")
        # @param {int} sensor_sleep_time: The time to sleep after the sensor captures a new image (default: 50000)
        # @param {int} threshold_update_rate: The rate of threshold update (default: 0)
        self.thresholds = thresholds
        self.clock = clock
        self.show = show
        self.max_untracked_frames = max_untracked_frames  # The maximum number of untracked frames
        if board == OPENMV:
            self.r_LED = LED("LED_RED")
            self.g_LED = LED("LED_GREEN")
            self.b_LED = LED("LED_BLUE")
        elif board == NICLA:
            self.r_LED = LED(1)  # The red LED
            self.g_LED = LED(2)  # The green LED
            self.b_LED = LED(3)  # The blue LED
        self.shape_detector = ShapeDetector(gridsize=9)
        self.num_blob_hist = 0
        self.LED_STATE = False
        self.time_last_snapshot = time.time_ns()  # wait for the sensor to capture a new image
        self.extra_fb = sensor.alloc_extra_fb(sensor.width(), sensor.height(), sensor.RGB565)
        self.extra_fb2 = sensor.alloc_extra_fb(sensor.width(), sensor.height(), sensor.RGB565)
        self.extra_fb3 = sensor.alloc_extra_fb(sensor.width(), sensor.height(), sensor.RGB565)
        self.IR_LED = Pin(LEDpin, Pin.OUT)
        self.IR_LED.value(0)
        self.roi = MemROI(ffp=0.01, ffs=0.02, gfp=.3, gfs=0.03)  # The ROI of the blob
        self.tracked_blob = None
        blob, _ = self.find_reference()
        self.tracked_blob = CurBLOB(blob)
        self.flag = 0x80
        self.sensor_sleep_time = sensor_sleep_time


    def destruct(self):
        sensor.dealloc_extra_fb()
        sensor.dealloc_extra_fb()
        sensor.dealloc_extra_fb()
        self.shape_detector.destruct()


    def _find_max(self, nice_blobs: list) -> image.blob:
        # @description: Find the blob with the largest area
        # @param {list} nice_blobs: The list of blobs to be compared
        # @return {image.blob} The blob with the largest area
        max_blob = None
        max_area = 0
        for blob in nice_blobs:
            if blob.area() > max_area:
                max_blob = blob
                max_area = blob.pixels()
        return max_blob


    def update_leds(self, tracking: bool = False, detecting: bool = False, lost: bool = True) -> None:
        # @description: Update the LEDs state
        # @param {bool} tracking: If we are tracking the blob in the roi
        # @param {bool} detecting: If we are actually detecting the blob
        # @param {bool} lost: If we lost the blob
        if tracking and detecting and not lost:
            self.g_LED.off()
            self.r_LED.off()
            self.b_LED.off()
        elif tracking and not detecting and not lost:
            self.g_LED.off()
            self.b_LED.off()
            self.r_LED.off()
        elif lost:
            self.g_LED.off()
            self.b_LED.off()
            self.r_LED.off()
        else:
            print("Error: Invalid LED state")
            pass


    def draw_initial_blob(self, img: image, blob: image.blob, sleep_us: int = 50000) -> None:
        # @description:
        # @param {image} img: The image to be drawn on
        # @param {image.blob} blob: The blob to be drawn
        # @param {int} sleep_us: The time to sleep after drawing the blob (default: 500000)
        if not blob or sleep_us < 41000:
            # No need to show anything if we do not want to show
            # sleep_us is beyond human's 24fps classy eyes' capability
            return
        else:
            img.draw_edges(blob.min_corners(), color=(255, 0, 0))
            img.draw_line(blob.major_axis_line(), color=(0, 255, 0))
            img.draw_line(blob.minor_axis_line(), color=(0, 0, 255))
            img.draw_rectangle(blob.rect())
            img.draw_cross(blob.cx(), blob.cy())
            img.draw_keypoints([(blob.cx(), blob.cy(), int(math.degrees(blob.rotation())))], size=20)
            # Sleep for 500ms for initial blob debut
            time.sleep_us(sleep_us)


    def track(self, edge_removal: bool = True) -> tuple:
        # @description: Track the blob with dynamic threshold and ROI
        # @param       {bool} edge_removal: Whether to remove the edge noises (default: True)
        # @return      {tuple} The feature vector of the tracked blob and whether the blob is tracked
        # the 8-bit flag variable
        # From MSB to LSB
        # [7]: 1 for goal
        # [6]: reserved for balloons
        # [5:2]: reserved
        # [1:0]: toggling between 1 and 2 for new detections, 0 for no detection
        self.update_leds(tracking=True, detecting=True, lost=False)  # Set the LEDs to indicate tracking

        # Initialize the blob with the max blob in view if it is not initialized
        if not self.tracked_blob.blob_history:
            self.flag = 0x80
            self.update_leds(tracking=False, detecting=False, lost=True)  # Set the LEDs to indicate tracking
            reference_blob, statistics = self.find_reference(time_show_us=0)  # Find the blob with the largest area
            if reference_blob:
                self.num_blob_hist = 1
                self.tracked_blob.reinit(reference_blob)  # Initialize the tracked blob with the reference blob
                # median_lumen = statistics.median()
                # if median_lumen <= self.thresholds[0][1]:
                #     # solid!
                #     flag |= 0x04
                self.roi.update(self.tracked_blob.feature_vector[0:4])  # Update the ROI
                self.update_leds(tracking=True, detecting=True, lost=False)
                # color_id = self.tracked_blob.blob_history[-1].code()
                # if color_id & 0b1:
                #     # green
                #    flag |= 0b10
                # elif color_id & 0b10:
                #     # orange
                #     flag &= 0xfd
                self.flag |= 0x01
                return self.tracked_blob.feature_vector, self.flag
            else:
                return None, self.flag

        # Track the blob
        img, list_of_blobs = self.detect(isColored=True, edge_removal=edge_removal)
        blob_rect = self.tracked_blob.update(list_of_blobs)

        if self.tracked_blob.untracked_frames >= self.max_untracked_frames or (not blob_rect and self.num_blob_hist <= 2):
            # If the blob fails to track for self.max_untracked_frames frames,
            # reset the tracking and find a new reference blob
            self.update_leds(tracking=False, detecting=False, lost=True)
            self.tracked_blob.reset()
            # self.roi.reset() (NOTE: ROI is not reset since we are assuming that the blob tends to appear in the same region when it is lost)
            print("Goal lost")
            self.num_blob_hist = 0
            self.flag = 0x80
            return None, self.flag
        elif self.flag == 0x80 and self.num_blob_hist == 2:
            self.flag = 0x81

        if blob_rect:
            self.num_blob_hist += 1
            # color_id = self.tracked_blob.blob_history[-1].code()
            # if color_id & 0b1:
            #     # green
            #     flag |= 0b10
            # elif color_id & 0b10:
            #     # orange
            #     flag &= 0xfd
            # If we discover the reference blob again
            if self.num_blob_hist > 2:
                flag_toggling = self.flag & 0x03
                flag_toggling = 3 - flag_toggling
                self.flag = 0x80 | flag_toggling
            self.roi.update(blob_rect)
            # We wnat to have a focus on the center of the blob
            shurnk_roi = list(blob_rect)
            shurnk_roi[0] += round(0.25*shurnk_roi[2])
            shurnk_roi[1] += round(0.25*shurnk_roi[3])
            shurnk_roi[2] //= 2
            shurnk_roi[3] //= 2
            statistics = img.get_statistics(roi=shurnk_roi)
            # median_lumen = statistics.median()
            # if median_lumen <= self.thresholds[0][1]:
            #     # solid!
            #     flag |= 0x04
            self.update_leds(tracking=True, detecting=True, lost=False)
        else:
            # If we do not discover the reference blob
            self.update_leds(tracking=True, detecting=False, lost=False)
            self.roi.update()

        if self.show:
            x0, y0, w, h = [math.floor(self.tracked_blob.feature_vector[i]) for i in range(4)]
            img.draw_rectangle(x0, y0, w, h, color=(255, 0, 0))
            img.draw_rectangle(self.roi.get_roi(), color=(128, 128, 0))
            img.flush()

        return self.tracked_blob.feature_vector, self.flag

    def find_reference(
        self,
        time_show_us: int = 50000,
    ) -> tuple:
        # @description: Find the a good blob to be the reference blob
        # @param {*} self:
        # @param {int} time_show_us: The time to show the blob on the screen
        # @return {tuple} The reference blob and its color statistics
        # omv.disable_fb(False)
        img, nice_blobs = self.detect(isColored=True, edge_removal=False)
        img.flush()
        if not nice_blobs:
            return None, None

        best_blob = self._find_max(nice_blobs)  # Find the best blob, will never return None if nice_blobs is not empty
        self.draw_initial_blob(img, best_blob, time_show_us)  # Draw the initial blob
        # omv.disable_fb(True)

        # We want to have a focus on the center of the blob
        shurnk_roi = list(best_blob.rect())
        shurnk_roi[0] += round(0.25*shurnk_roi[2])
        shurnk_roi[1] += round(0.25*shurnk_roi[3])
        shurnk_roi[2] //= 2
        shurnk_roi[3] //= 2

        statistics = img.get_statistics(roi=shurnk_roi)  # Get the color statistics of the blob in actual image
        return best_blob, statistics

    def detect(self, isColored=False, edge_removal=True):
        ############ NOTE: the following part is the discarded IR blinking method ############
        if False:
            omv.disable_fb(True)  # No show on screen
            self.LED_STATE = True
            self.IR_LED.value(not self.LED_STATE)
            sensor.skip_frames(1)
            while(not sensor.get_frame_available()):
                pass

            self.extra_fb.replace(sensor.snapshot())
            self.time_last_snapshot = time.time_ns() # wait for the sensor to capture a new image
            ###################################################################

            self.IR_LED.value(self.LED_STATE)

            while(not sensor.get_frame_available()):
                pass
            # time.sleep_us(20)
            #     time.sleep_us(1)
            # time.sleep_us(int(self.sensor_sleep_time/2))

            self.IR_LED.value(not self.LED_STATE)
            self.extra_fb2.replace(sensor.snapshot())
            self.time_last_snapshot = time.time_ns()  # wait for the sensor to capture a new image
            ######################################################################
            # self.sensor_sleep(self.time_last_snapshot)
            # sensor.skip_frames(1)
            while(not sensor.get_frame_available()):
                pass
            # time.sleep_us(20)
            #     time.sleep_us(1)

            img = sensor.snapshot()
            self.time_last_snapshot = time.time_ns()  # wait for the sensor to capture a new image

            self.IR_LED.value(False)
            img.difference(self.extra_fb2, reverse=self.LED_STATE)
            self.extra_fb3.replace(img)
            img.replace(self.extra_fb)
            img.sub(self.extra_fb2, reverse=self.LED_STATE)
            img.difference(self.extra_fb3)
            self.extra_fb2.replace(img)
            img.replace(self.extra_fb3)
            img.difference(self.extra_fb2, reverse=self.LED_STATE)
            # Remove the edge noises

            img.negate()
            omv.disable_fb(False)
        else:
            self.LED_STATE = True
            img = sensor.snapshot()
            self.clock.tick()

        # if self.num_blob_hist > 5 and self.thresholds == TARGET_ORANGE:
        #     list_of_blob = img.find_blobs(
        #         TARGET_COLOR2,
        #         area_threshold=3,
        #         pixels_threshold=3,
        #         margin=10,
        #         x_stride=1,
        #         y_stride=1,
        #         merge=True,
        #     )
        # else:
        list_of_blob = img.find_blobs(
            self.thresholds,
            area_threshold=3,
            pixels_threshold=3,
            margin=10,
            x_stride=1,
            y_stride=1,
            merge=True,
        )
        # largest_blob = max(list_of_blob, key=lambda b: b.area(), default=None)
        # # shape detection/determination
        # if largest_blob:
        #     roi_img, roi = self.shape_detector.extract_valid_roi(img, largest_blob, self.thresholds)
        #     if roi_img:
        #         detected_shape = self.shape_detector.detect_shape(roi_img)
        #         print("Detected Shape:", detected_shape)
        #         mean_pooled_img = self.shape_detector.downsample_and_average(roi_img)
        #         gridsize = 9
        #         # Visually represent the data (example code)
        #         scale_x = roi[2] / gridsize
        #         scale_y = roi[3] / gridsize
        #         for i in range(gridsize):
        #             for j in range(gridsize):
        #                 gray_value = mean_pooled_img.get_pixel(j, i) *255
        #                 rect_x = roi[0] + j * int(scale_x)
        #                 rect_y = roi[1] + i * int(scale_y)
        #                 rect_width = max(int(scale_x), 1)
        #                 rect_height = max(int(scale_y), 1)
        #                 img.draw_rectangle(rect_x, rect_y, rect_width, rect_height, color=(gray_value, gray_value, gray_value), fill=True)
        #         img.draw_rectangle(largest_blob.rect(), color=(127, 0, 127))  # Highlight the blob
        #         img.draw_string(largest_blob.x(), largest_blob.y(), detected_shape, color=(255, 0, 255))
        st = "FPS: {}".format(str(round(self.clock.fps(), 2)))
        img.draw_string(0, 0, st, color=(255, 0, 0))
        # sensor.dealloc_extra_fb()
        big_blobs=[]
        for blob in list_of_blob:
            if blob.area() > 20 and line_length(blob.minor_axis_line())> 5:
                # if self.tracked_blob != None and self.num_blob_hist > 5:
                #     big_blobs.append(blob)
                # else:
                roi_img, roi = self.shape_detector.extract_valid_roi(img, blob, self.thresholds)
                if roi_img:
                    # mean_pooled_img = self.shape_detector.downsample_and_average(roi_img)
                    # gridsize = 9
                    # scale_x = roi[2] / gridsize
                    # scale_y = roi[3] / gridsize
                    # for i in range(gridsize):
                    #     for j in range(gridsize):
                    #         gray_value = mean_pooled_img.get_pixel(j, i) *255
                    #         rect_x = roi[0] + j * int(scale_x)
                    #         rect_y = roi[1] + i * int(scale_y)
                    #         rect_width = max(int(scale_x), 1)
                    #         rect_height = max(int(scale_y), 1)
                    # img.draw_rectangle(rect_x, rect_y, rect_width, rect_height, color=(gray_value, gray_value, gray_value), fill=True)
                    detected_shape = self.shape_detector.detect_shape(roi_img)
                    # if detected_shape != "triangle" and detected_shape != "not":
                    if detected_shape != "not":
                        big_blobs.append(blob)

                    img.draw_string(blob.x(), blob.y(), detected_shape[0], color=(255, 0, 255))
                    img.draw_rectangle(blob.rect(), color=(255, 0, 255))
                    del(roi_img)
        # else: # if shape cannot be determined add blob to anyway
        #     big_blobs.append(blob)
        #     img.draw_rectangle(blob.rect(), color=(255, 0, 0))  # Red rectangle around the blob for visibility
        # img.draw_edges(blob.min_corners(), color=(255, 0, 0))
        # img.draw_line(blob.major_axis_line(), color=(0, 255, 0))
        # img.draw_line(blob.minor_axis_line(), color=(0, 0, 255))
        # img.draw_rectangle(blob.rect())
        # img.draw_cross(blob.cx(), blob.cy())
        # img.draw_keypoints([(blob.cx(), blob.cy(), int(math.degrees(blob.rotation())))], size=20)
        return img, big_blobs


    def sensor_sleep(self, last_time_stamp) -> None:
        # @description: Wait for the sensor for some time from the last snapshot to avoid a partial new image
        # @param       {*} last_time_stamp: The time stamp of the last snapshot
        elapsed = self.sensor_sleep_time - (int((time.time_ns() - last_time_stamp) / 1000))
        while elapsed < 0:
            elapsed += self.sensor_sleep_time
        # if elapsed >= 0:
        time.sleep_us(elapsed)
        # else:
        #     time.sleep_us(self.sensor_sleep_time+ elapsed%self.sensor_sleep_time)
        return None


# helper function for calculating the length of a line segment
def line_length(coords):
    # Calculate the length of a line segment given its coordinates.
    # Args:
    # coords (tuple): A tuple of four elements (x1, y1, x2, y2) representing
    #                 the coordinates of the two endpoints of the line segment.
    # Returns:
    # float: The length of the line segment.
    x1, y1, x2, y2 = coords
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


# Sensor initialization based on the object to track: 0 for balloon and 1 for goal
def init_sensor_target(tracking_type:int, framesize=FRAME_SIZE, windowsize=None) -> None:
    # Initialize sensors by updating the registers
    # for the two different purposes
    #     @param       {int} tracking_type: 0 for balloons and 1 for goals
    #     @return      {*} None
    #
    if tracking_type == 1:
        # goal detection sensor setup
        sensor.reset()
        if board == NICLA:
            sensor.ioctl(sensor.IOCTL_SET_FOV_WIDE, True)
        sensor.set_auto_exposure(True)
        sensor.set_pixformat(sensor.RGB565)
        sensor.set_framesize(framesize)
        if board == NICLA:
            sensor.ioctl(sensor.IOCTL_SET_FOV_WIDE, True)
            sensor.__write_reg(0xfe, 0b00000000) # change to registers at page 0
            sensor.__write_reg(0x80, 0b01111110) # [7] reserved, [6] gamma enable, [5] CC enable,
            if ADVANCED_SENSOR_SETUP and TARGET_COLOR == TARGET_ORANGE:
                sensor.__write_reg(0xfe, 0) # change to registers at page 0
                sensor.__write_reg(0x80, 0b01111110)    # [7] reserved, [6] gamma enable, [5] CC enable,
                                                        # [4] Edge enhancement enable
                                                        # [3] Interpolation enable, [2] DN enable, [1] DD enable,
                                                        # [0] Lens-shading correction enable - gives you uneven
                                                        #                                      shade in the dark
                                                        #                                      badly!!!!!
                sensor.__write_reg(0x81, 0b01010100)    # [7] BLK dither mode, [6] low light Y stretch enable
                                                        # [5] skin detection enable, [4] reserved, [3] new skin mode
                                                        # [2] autogray enable, [1] reserved, [0] BFF test image mode
                sensor.__write_reg(0x82, 0b00000100)    # [2] ABS enable, [1] AWB enable
                #sensor.__write_reg(0x87, 0b00000001)    # [0] auto_edge_effect
                sensor.__write_reg(0x9a, 0b00001111)    # [3] smooth Y, [2] smooth Chroma,
                                                        # [1] neighbor average mode, [0] subsample extend opclk
                sensor.skip_frames(2)
                print("block enabling done")

                # Edge enhancements
                sensor.__write_reg(0xfe, 2)             # change to registers at page 2
                sensor.__write_reg(0x90, 0b11101101)    # [7]edge1_mode, [6]HP3_mode, [5]edge2_mode, [4]Reserved,
                                                        # [3]LP_intp_en, [2]LP_edge_en, [1]NA, [0] half_scale_mode_en
                sensor.__write_reg(0x91, 0b11000000)    # [7]HP_mode1, [6]HP_mode2,
                                                        # [5]only 2 direction - only two direction H and V, [4]NA
                                                        # [3]only_defect_map, [2]map_dir, [1:0]reserved
                sensor.__write_reg(0x96, 0b00001100)    # [3:2] edge leve
                sensor.__write_reg(0x97, 0x88)          # [7:4] edge1 effect, [3:0] edge2 effect
                sensor.__write_reg(0x9b, 0b00100010)    # [7:4] edge1 threshold, [3:0] edge2 threshold
                sensor.skip_frames(2)
                print("edge enhancement done")

                # color correction -- this is very tricky: the color shifts on the color wheel it seems
                sensor.__write_reg(0xfe, 2) # change to registers at page 2
                # WARNING: uncomment the two lines to invert the color
                #sensor.__write_reg(0xc1, 0x80)          # CC_CT1_11, feels like elements in a matrix
                #sensor.__write_reg(0xc5, 0x80)          # CC_CT1_22 , feels like elements in a matrix
                print("color correction setup done")

                # ABS - anti-blur
                sensor.__write_reg(0xfe, 1)             # change to registers at page 1
                sensor.__write_reg(0x9a, 0b11110111)    # [7:4] add dynamic range, [2:0] abs adjust every frame
                sensor.__write_reg(0x9d, 0xff)          # [7:0] Y stretch limit
                sensor.skip_frames(2)
                print("anti-blur setup done")

                # color settings -- AWB
                # Ranting about the trickiness of the setup:
                # Even the auto white balance is disabled, the AWB gains will persist to
                # take effect. Although the correcponding registers are read-only in the
                # document, they are actually manually writeable, and such writings are
                # effective. On top of these messes, another set of registers that are
                # R/W have the exact same effect on the RGB gains, but they are not
                # controlled by the AWB.
                sensor.set_auto_exposure(False)
                sensor.set_auto_whitebal(False) # no, the gain_rgb_db does not work
                # # reset RGB auto gains
                # sensor.__write_reg(0xb3, 64)    # reset R auto gain
                # sensor.__write_reg(0xb4, 64)    # reset G auto gain
                # sensor.__write_reg(0xb5, 64)    # reset B auto gain
                # sensor.__write_reg(0xfe, 0)     # change to registers at page 0
                #                                 # manually set RGB gains to fix color/white balance
                # sensor.__write_reg(0xad, int(R_GAIN))    # R gain ratio
                # sensor.__write_reg(0xae, int(G_GAIN))    # G gain ratio
                # sensor.__write_reg(0xaf, int(B_GAIN))    # B gain ratio
                sensor.set_auto_exposure(True)
                sensor.set_auto_whitebal(True)
                sensor.__write_reg(0xfe, 1) # change to registers at page 1
                sensor.__write_reg(0x13, 96) # brightness level
                # sensor.__write_reg(0xb2, 255)   # post-gain, default 64
                sensor.skip_frames(2)
                print("AWB Gain setup done.")

                # color setup - saturation
                sensor.__write_reg(0xfe, 2)     # change to registers at page 2
                sensor.__write_reg(0xd0, 72)    # change global saturation,
                sensor.__write_reg(0xd1, 48)    # Cb saturation
                sensor.__write_reg(0xd2, 48)    # Cr saturation
                sensor.__write_reg(0xd3, 20)    # contrast
                sensor.__write_reg(0xd5, 0)     # luma offset
        elif board == OPENMV:
            pass

        sensor.skip_frames(time = 1000)

    elif tracking_type == 0:
        # balloon detection sensor setup
        sensor.reset()
        sensor.set_auto_whitebal(True)
        sensor.set_auto_exposure(True)
        sensor.set_pixformat(sensor.RGB565)
        if board == NICLA:
            sensor.ioctl(sensor.IOCTL_SET_FOV_WIDE, True) # wide FOV
        sensor.set_framesize(framesize)

        sensor.set_auto_whitebal(False)
        sensor.set_auto_exposure(False)

        # sensor setup
        if board == NICLA:
            sensor.__write_reg(0xfe, 0) # change to registers at page 0
            sensor.__write_reg(0x80, 0b01111110)    # [7] reserved, [6] gamma enable, [5] CC enable,
                                                    # [4] Edge enhancement enable
                                                    # [3] Interpolation enable, [2] DN enable, [1] DD enable,
                                                    # [0] Lens-shading correction enable - gives you uneven
                                                    #                                      shade in the dark
                                                    #                                      badly!!!!!
            sensor.__write_reg(0x81, 0b01010100)    # [7] BLK dither mode, [6] low light Y stretch enable
                                                    # [5] skin detection enable, [4] reserved, [3] new skin mode
                                                    # [2] autogray enable, [1] reserved, [0] BFF test image mode
            sensor.__write_reg(0x82, 0b00000100)    # [2] ABS enable, [1] AWB enable
            #sensor.__write_reg(0x87, 0b00000001)    # [0] auto_edge_effect
            sensor.__write_reg(0x9a, 0b00001111)    # [3] smooth Y, [2] smooth Chroma,
                                                    # [1] neighbor average mode, [0] subsample extend opclk
            sensor.skip_frames(2)
            print("block enabling done")

            # Edge enhancements
            sensor.__write_reg(0xfe, 2)             # change to registers at page 2
            sensor.__write_reg(0x90, 0b11101101)    # [7]edge1_mode, [6]HP3_mode, [5]edge2_mode, [4]Reserved,
                                                    # [3]LP_intp_en, [2]LP_edge_en, [1]NA, [0] half_scale_mode_en
            sensor.__write_reg(0x91, 0b11000000)    # [7]HP_mode1, [6]HP_mode2,
                                                    # [5]only 2 direction - only two direction H and V, [4]NA
                                                    # [3]only_defect_map, [2]map_dir, [1:0]reserved
            sensor.__write_reg(0x96, 0b00001100)    # [3:2] edge leve
            sensor.__write_reg(0x97, 0x88)          # [7:4] edge1 effect, [3:0] edge2 effect
            sensor.__write_reg(0x9b, 0b00100010)    # [7:4] edge1 threshold, [3:0] edge2 threshold
            sensor.skip_frames(2)
            print("edge enhancement done")

            # color correction -- this is very tricky: the color shifts on the color wheel it seems
            sensor.__write_reg(0xfe, 2) # change to registers at page 2
            # WARNING: uncomment the two lines to invert the color
            #sensor.__write_reg(0xc1, 0x80)          # CC_CT1_11, feels like elements in a matrix
            #sensor.__write_reg(0xc5, 0x80)          # CC_CT1_22 , feels like elements in a matrix
            print("color correction setup done")

            # ABS - anti-blur
            sensor.__write_reg(0xfe, 1)             # change to registers at page 1
            sensor.__write_reg(0x9a, 0b11110111)    # [7:4] add dynamic range, [2:0] abs adjust every frame
            sensor.__write_reg(0x9d, 0xff)          # [7:0] Y stretch limit
            sensor.skip_frames(2)
            print("anti-blur setup done")


            # color settings -- AWB
            # Ranting about the trickiness of the setup:
            # Even the auto white balance is disabled, the AWB gains will persist to
            # take effect. Although the correcponding registers are read-only in the
            # document, they are actually manually writeable, and such writings are
            # effective. On top of these messes, another set of registers that are
            # R/W have the exact same effect on the RGB gains, but they are not
            # controlled by the AWB.
            sensor.set_auto_exposure(False)
            sensor.set_auto_whitebal(False) # no, the gain_rgb_db does not work

            # reset RGB auto gains
            sensor.__write_reg(0xb3, 64)    # reset R auto gain
            sensor.__write_reg(0xb4, 64)    # reset G auto gain
            sensor.__write_reg(0xb5, 64)    # reset B auto gain

            sensor.__write_reg(0xfe, 0)     # change to registers at page 0
                                            # manually set RGB gains to fix color/white balance
            sensor.__write_reg(0xad, int(R_GAIN))    # R gain ratio
            sensor.__write_reg(0xae, int(G_GAIN))    # G gain ratio
            sensor.__write_reg(0xaf, int(B_GAIN))    # B gain ratio
            sensor.set_auto_exposure(True)
            sensor.__write_reg(0xfe, 1) # change to registers at page 1
            sensor.__write_reg(0x13, 120) # brightness level
            # sensor.__write_reg(0xb2, 255)   # post-gain, default 64
            sensor.skip_frames(2)
            print("AWB Gain setup done.")

            # color setup - saturation
            sensor.__write_reg(0xfe, 2)     # change to registers at page 2
            sensor.__write_reg(0xd0, 72)    # change global saturation,
            sensor.__write_reg(0xd1, 56)    # Cb saturation
            sensor.__write_reg(0xd2, 56)    # Cr saturation
            sensor.__write_reg(0xd3, 40)    # contrast
            sensor.__write_reg(0xd5, 0)     # luma offset
        elif board == OPENMV:
            # sensor.set_contrast(-3)
            # sensor.set_brightness(1)
            # sensor.__write_reg(0x3008, 0x42) # software power down
            # sensor.__write_reg(0x3103, 0x03) # SCCB system control

            # ISP setup:
            sensor.__write_reg(0x5000, 0b00100111)  # [7]: lens correction, [5]: raw gamma
                                                    # [2:1]: black/white pixel cancellation
                                                    # [0]: color interpolation
            sensor.__write_reg(0x5001, sensor.__read_reg(0x5001) | 0b10000110)# [7]: SFX, [5]: scaling
            # sensor.__write_reg(0x5001, sensor.__read_reg(0x5001) & 0b11011111)  # [2]: UV average,
                                                                                # [1]: color matrix
                                                                                # [0]: AWB
            openmv_set_saturation_brightness_contrast(saturation=1, brightness=1, contrast=3, ev=0)

            # lens correction parameters
            # BR_h_rec = 0
            # BR_v_rec = 0
            # G_h_rec = 256
            # G_v_rec = 256
            # sensor.__write_reg(0x583e, 255)  # Maximum gain: default 64
            # sensor.__write_reg(0x583f, 0)  # Minimum gain: default 32
            # sensor.__write_reg(0x5840, 0)  # Minimum Q: default 24
            # sensor.__write_reg(0x5841, 0b00000001)  # default 1101
            #                                         # Bit[3]: Add BLC enable
            #                                         #   0: Disable BLC add back function
            #                                         #   1: Enable BLC add back function
            #                                         # Bit[2]: BLC enable
            #                                         #   0: Disable BLC function
            #                                         #   1: Enable BLC function
            #                                         # Bit[1]: Gain manual enable
            #                                         # Bit[0]: Auto Q enable
            #                                         #   0: Used constant Q (0x40)
            #                                         #   1: Used calculated Q
            # sensor.__write_reg(0x5842, BR_h_rec >> 8)   # BR h[10:8]
            # sensor.__write_reg(0x5843, BR_h_rec & 0xff) # BR h[7:0]
            # sensor.__write_reg(0x5844, BR_v_rec >> 8)   # BR v[10:8]
            # sensor.__write_reg(0x5845, BR_v_rec & 0xff) # BR v[7:0]
            # sensor.__write_reg(0x5846, G_h_rec >> 8)    # G  h[10:8]
            # sensor.__write_reg(0x5847, G_h_rec & 0xff)  # G  h[7:0]
            # sensor.__write_reg(0x5848, G_v_rec >> 8)    # G  v[10:8]
            # sensor.__write_reg(0x5849, G_v_rec & 0xff)  # G  v[7:0]
            # sensor.__write_reg(0x3008, 0x02) # software power up

            sensor.set_auto_exposure(True)
            print(sensor.get_rgb_gain_db())
            sensor.set_auto_whitebal(False, rgb_gain_db=(R_GAIN, G_GAIN, B_GAIN))

    else:
        raise ValueError("Not a valid sensor-detection mode!")

    if windowsize is not None:
        sensor.set_windowing(windowsize)

def openmv_set_saturation_brightness_contrast(saturation: int=0, brightness: int=0, contrast: int=0, ev: int=0):
    # color settings - contrast, brightness, and saturation
    # Do refer to page 49 of this document
    # https://www.arducam.com/downloads/modules/OV5640/OV5640_Software_app_note_parallel.pdf

    # contrast
    sensor.__write_reg(0x3212, 0x03)
    if contrast == 3:
        sensor.__write_reg(0x5586, 0x2c)
        sensor.__write_reg(0x5585, 0x1c)
    elif contrast == 2:
        sensor.__write_reg(0x5586, 0x28)
        sensor.__write_reg(0x5585, 0x18)
    elif contrast == 1:
        sensor.__write_reg(0x5586, 0x24)
        sensor.__write_reg(0x5585, 0x10)
    elif contrast == 0:
        sensor.__write_reg(0x5586, 0x20)
        sensor.__write_reg(0x5585, 0x00)
    elif contrast == -1:
        sensor.__write_reg(0x5586, 0x1c)
        sensor.__write_reg(0x5585, 0x1c)
    elif contrast == -2:
        sensor.__write_reg(0x5586, 0x18)
        sensor.__write_reg(0x5585, 0x18)
    elif contrast == -3:
        sensor.__write_reg(0x5586, 0x14)
        sensor.__write_reg(0x5585, 0x14)

    # brightness
    if brightness == 4:
        sensor.__write_reg(0x5587, 0x40)
        sensor.__write_reg(0x5588, 0x01)
    elif brightness == 3:
        sensor.__write_reg(0x5587, 0x30)
        sensor.__write_reg(0x5588, 0x01)
    elif brightness == 2:
        sensor.__write_reg(0x5587, 0x20)
        sensor.__write_reg(0x5588, 0x01)
    elif brightness == 1:
        sensor.__write_reg(0x5587, 0x10)
        sensor.__write_reg(0x5588, 0x01)
    elif brightness == 0:
        sensor.__write_reg(0x5587, 0x00)
        sensor.__write_reg(0x5588, 0x01)
    elif brightness == -1:
        sensor.__write_reg(0x5587, 0x10)
        sensor.__write_reg(0x5588, 0x09)
    elif brightness == -2:
        sensor.__write_reg(0x5587, 0x20)
        sensor.__write_reg(0x5588, 0x09)
    elif brightness == -3:
        sensor.__write_reg(0x5587, 0x30)
        sensor.__write_reg(0x5588, 0x09)
    elif brightness == -4:
        sensor.__write_reg(0x5587, 0x40)
        sensor.__write_reg(0x5588, 0x09)

    # saturation
    sensor.__write_reg(0x5381, 0x1c)
    sensor.__write_reg(0x5382, 0x5a)
    sensor.__write_reg(0x5383, 0x06)
    if saturation == 3:
        sensor.__write_reg(0x5384, 0x2b)
        sensor.__write_reg(0x5385, 0xab)
        sensor.__write_reg(0x5386, 0xd6)
        sensor.__write_reg(0x5387, 0xda)
        sensor.__write_reg(0x5388, 0xd6)
        sensor.__write_reg(0x5389, 0x04)
    elif saturation == 2:
        sensor.__write_reg(0x5384, 0x24)
        sensor.__write_reg(0x5385, 0x8f)
        sensor.__write_reg(0x5386, 0xb3)
        sensor.__write_reg(0x5387, 0xb6)
        sensor.__write_reg(0x5388, 0xb3)
        sensor.__write_reg(0x5389, 0x03)
    elif saturation == 1:
        sensor.__write_reg(0x5384, 0x1f)
        sensor.__write_reg(0x5385, 0x7a)
        sensor.__write_reg(0x5386, 0x9a)
        sensor.__write_reg(0x5387, 0x9c)
        sensor.__write_reg(0x5388, 0x9a)
        sensor.__write_reg(0x5389, 0x02)
    elif saturation == 0:
        sensor.__write_reg(0x5384, 0x1a)
        sensor.__write_reg(0x5385, 0x66)
        sensor.__write_reg(0x5386, 0x80)
        sensor.__write_reg(0x5387, 0x82)
        sensor.__write_reg(0x5388, 0x80)
        sensor.__write_reg(0x5389, 0x02)
    elif saturation == -1:
        sensor.__write_reg(0x5384, 0x15)
        sensor.__write_reg(0x5385, 0x52)
        sensor.__write_reg(0x5386, 0x66)
        sensor.__write_reg(0x5387, 0x68)
        sensor.__write_reg(0x5388, 0x66)
        sensor.__write_reg(0x5389, 0x02)
    elif saturation == -2:
        sensor.__write_reg(0x5384, 0x10)
        sensor.__write_reg(0x5385, 0x3d)
        sensor.__write_reg(0x5386, 0x4d)
        sensor.__write_reg(0x5387, 0x4e)
        sensor.__write_reg(0x5388, 0x4d)
        sensor.__write_reg(0x5389, 0x01)
    elif saturation == -3:
        sensor.__write_reg(0x5384, 0x0c)
        sensor.__write_reg(0x5385, 0x30)
        sensor.__write_reg(0x5386, 0x3d)
        sensor.__write_reg(0x5387, 0x3e)
        sensor.__write_reg(0x5388, 0x3d)
        sensor.__write_reg(0x5389, 0x01)

    if ev == 3:
        sensor.__write_reg(0x3a0f, 0x60)
        sensor.__write_reg(0x3a10, 0x58)
        sensor.__write_reg(0x3a11, 0xa0)
        sensor.__write_reg(0x3a1b, 0x60)
        sensor.__write_reg(0x3a1e, 0x58)
        sensor.__write_reg(0x3a1f, 0x20)
    elif ev == 2:
        sensor.__write_reg(0x3a0f, 0x50)
        sensor.__write_reg(0x3a10, 0x48)
        sensor.__write_reg(0x3a11, 0x90)
        sensor.__write_reg(0x3a1b, 0x50)
        sensor.__write_reg(0x3a1e, 0x48)
        sensor.__write_reg(0x3a1f, 0x20)
    elif ev == 1:
        sensor.__write_reg(0x3a0f, 0x40)
        sensor.__write_reg(0x3a10, 0x38)
        sensor.__write_reg(0x3a11, 0x71)
        sensor.__write_reg(0x3a1b, 0x40)
        sensor.__write_reg(0x3a1e, 0x38)
        sensor.__write_reg(0x3a1f, 0x10)
    elif ev == 0:
        sensor.__write_reg(0x3a0f, 0x38)
        sensor.__write_reg(0x3a10, 0x30)
        sensor.__write_reg(0x3a11, 0x61)
        sensor.__write_reg(0x3a1b, 0x38)
        sensor.__write_reg(0x3a1e, 0x30)
        sensor.__write_reg(0x3a1f, 0x10)
    elif ev == -1:
        sensor.__write_reg(0x3a0f, 0x30)
        sensor.__write_reg(0x3a10, 0x28)
        sensor.__write_reg(0x3a11, 0x61)
        sensor.__write_reg(0x3a1b, 0x30)
        sensor.__write_reg(0x3a1e, 0x28)
        sensor.__write_reg(0x3a1f, 0x10)
    elif ev == -2:
        sensor.__write_reg(0x3a0f, 0x20)
        sensor.__write_reg(0x3a10, 0x18)
        sensor.__write_reg(0x3a11, 0x41)
        sensor.__write_reg(0x3a1b, 0x20)
        sensor.__write_reg(0x3a1e, 0x18)
        sensor.__write_reg(0x3a1f, 0x10)
    elif ev == -3:
        sensor.__write_reg(0x3a0f, 0x10)
        sensor.__write_reg(0x3a10, 0x08)
        sensor.__write_reg(0x3a11, 0x10)
        sensor.__write_reg(0x3a1b, 0x08)
        sensor.__write_reg(0x3a1e, 0x20)
        sensor.__write_reg(0x3a1f, 0x10)

    sensor.__write_reg(0x538b, 0x98)
    sensor.__write_reg(0x538a, 0x01)
    sensor.__write_reg(0x3212, 0x13)
    sensor.__write_reg(0x3212, 0xa3)


# IBus communication functions
# checksum that we can but we are not using on the ESP side for verifying data integrity
def checksum(arr, initial= 0):
    # The last pair of byte is the checksum on iBus
    sum = initial
    for a in arr:
        sum += a
    checksum = 0xFFFF - sum
    chA = checksum >> 8
    chB = checksum & 0xFF
    return chA, chB

# send an ibus message array to uart, each element is 2-byte
# for some of them we are only using 1 byte anyways
def IBus_message(message_arr_to_send):
    msg = bytearray(32)
    msg[0] = 0x20
    msg[1] = 0x40
    for i in range(len(message_arr_to_send)):
        msg_byte_tuple = bytearray(message_arr_to_send[i].to_bytes(2, 'little'))
        msg[int(2*i + 2)] = msg_byte_tuple[0]
        msg[int(2*i + 3)] = msg_byte_tuple[1]

    # Perform the checksume
    chA, chB = checksum(msg[:-2], 0)
    msg[-1] = chA
    msg[-2] = chB
    return msg


def mode_initialization(input_mode, mode, grid=None, detectors=None):
    # Switching between blinking goal tracker and balloon tracker
    if mode == input_mode:
        print("already in the mode")
        return None
    else:
        if input_mode == 0:
            # balloon tracking mode
            init_sensor_target(tracking_type=0)
            tracker = BalloonTracker(grid, detectors)
            print("balloon mode!")
        elif input_mode == 1:
            thresholds = TARGET_COLOR
            init_sensor_target(tracking_type=1)
            if board == NICLA:
                pin_str = "PG12"
            elif board == OPENMV:
                pin_str = "P7"
            tracker = GoalTracker(
                thresholds, clock,
                max_untracked_frames = MAX_LOST_FRAME_GOAL,
                LEDpin=pin_str,
                sensor_sleep_time=WAIT_TIME_US
            )
            print("Goal mode!")
        else:
            raise ValueError("Invalid mode selection")

        return input_mode, tracker


if __name__ == "__main__":
    # Necessary for both modes
    clock = time.clock()
    mode = 0 # 0 for balloon detection and 1 for goal

    # Initialize inter-board communication
    # time of flight sensor initialization
    # tof = VL53L1X(I2C(2)) # seems to interfere with the uart

    # Grid detection setup for balloons
    sensor.reset()
    sensor.set_pixformat(sensor.RGB565)
    if board == NICLA:
        sensor.ioctl(sensor.IOCTL_SET_FOV_WIDE, True) # wide FOV
    sensor.set_framesize(FRAME_SIZE)
    sensor.skip_frames(time = 1000)
    img = sensor.snapshot()
    grid = Grid(N_ROWS, N_COLS, img.width(), img.height())

    # Initialize color detectors for target balloons, peers, and neighbors of the target balloons
    detectors = {}
    color_collection = list(set(COLOR_TARGET.keys()).union(COLOR_PEER.keys()).union(COLOR_NEIGHBOR.keys()))
    neighbor = None
    for color in color_collection:
        std_range = COLOR_SENSITIVITY[color][1:]
        if color in COLOR_TARGET:
            detector_type = "T"
            rgb = COLOR_TARGET[color]
        elif color in COLOR_PEER:
            detector_type = "P"
            rgb = COLOR_PEER[color]
        else:
            detector_type = "N"
            rgb = COLOR_NEIGHBOR[color][1]
            neighbor = COLOR_NEIGHBOR[color][0]
        mu = COLOR_DATA[color][0]
        sigma_inv = COLOR_DATA[color][1]
        decay = COLOR_SENSITIVITY[color][0]
        l_range = L_RANGE[color]

        detectors[color] = ColorDetector(
            color, line_ref = None,
            detector_type=detector_type,
            max_dist=None, std_range=std_range, light_range=l_range,
            rgb=rgb, mahalanobis=True,
            mu=mu, sigma_inv=sigma_inv,
            decay=decay,
            neighbor=neighbor
        )

    # Initializing the tracker
    mode, tracker = mode_initialization(mode, -1, grid, detectors)
    del img
    img = None

    # Initialize UART
    if board == NICLA:
        uart = UART("LP1", baudrate= 115200, timeout_char=10) # (TX, RX) = (P1, P0) = (PB14, PB15) = "LP1"
    elif board == OPENMV:
        uart = UART(1, baudrate= 115200, timeout_char=10)

    # Main loop
    while True:
        clock.tick()
        """
        start = time.ticks_us()
        if mode == 1:
            feature_vector, flag = tracker.track()
        elif mode == 0:
            clock.tick()
            img = sensor.snapshot() #.gamma(gamma=2.0, contrast=0.5, brightness=0.0)
            feature_vector, flag = tracker.track(img)
        """
#  to seperate snapshot and inference process so that we can just calculate inference time
        if mode == 1:
            img = None
        elif mode == 0:
            img = sensor.snapshot()
        start = time.ticks_us()
        if mode == 1:
            feature_vector, flag = tracker.track()
        elif mode == 0:
            feature_vector, flag = tracker.track(img)
        end = time.ticks_us()
        inference_ms = time.ticks_diff(end, start) / 1000
        print(inference_ms)

        if flag & 0x80:
            # goal detection mode
            if flag & 0x03:
                # check if tracking (not necessarily detection) is still active
                roi = tracker.roi.get_roi()
                feature_vec = tracker.tracked_blob.feature_vector
                x_roi = roi[0] + roi[2]//2
                y_roi = roi[1] + roi[3]//2
                w_roi = roi[2]
                h_roi = roi[3]

                x_value = int(feature_vec[0] + feature_vec[2]/2)
                y_value = int(feature_vec[1] + feature_vec[3]/2)
                w_value = int(feature_vec[2])
                h_value = int(feature_vec[3])
                if board == NICLA:
                    msg = IBus_message([flag, x_value, y_roi, w_roi, h_roi,
                                        x_value, y_value, w_value, h_value, 8])
                    # print(x_value, y_roi, w_roi, h_roi, x_value, y_value, w_value, h_value)
                elif board == OPENMV:
                    msg = IBus_message([flag, FRAME_PARAMS[2] - x_value, FRAME_PARAMS[3] - y_roi, w_roi, h_roi,
                                        FRAME_PARAMS[2] - x_value, FRAME_PARAMS[3] - y_value, w_value, h_value, 9999])
            else:
                msg = IBus_message([flag, 0, 0, 0, 0, 0, 0, 0, 0, 9999])
        elif flag & 0b01000000:
            # balloon detection mode
            x_roi, y_roi, w_roi, h_roi, x_value, y_value, w_value, h_value, just_zero = feature_vector
            if flag & 0x03:
                if board == NICLA:
                    msg = IBus_message([flag, x_roi, y_roi, w_roi, h_roi,
                                        x_value, y_value, w_value, h_value, 9999])
                elif board == OPENMV:
                    msg = IBus_message([flag, FRAME_PARAMS[2] - x_roi, FRAME_PARAMS[3] - y_roi, w_roi, h_roi,
                                        FRAME_PARAMS[2] - x_value, FRAME_PARAMS[3] - y_value, w_value, h_value, 9999])
            else:
                msg = IBus_message([flag, 0, 0, 0, 0, 0, 0, 0, 0, 9999])
        else:
            print("0 flag!")
            assert(flag == 0)

        # print("fps: ", clock.fps())

        uart.write(msg)
        if uart.any():
            uart_input = uart.read()
            print(uart_input)
            if uart_input[-1] == 0x40 and mode == 1:
                tracker.destruct()
                res = mode_initialization(0, mode, grid, detectors)
                if res:
                    mode, tracker = res
            elif uart_input[-1] == 0x80 and mode == 0:
                TARGET_COLOR = TARGET_YELLOW
                res = mode_initialization(1, mode)
                if res:
                    mode, tracker = res
            elif uart_input[-1] == 0x81 and mode == 0:
                TARGET_COLOR = TARGET_ORANGE
                res = mode_initialization(1, mode)
                if res:
                    mode, tracker = res
            else:
                print("possible mode transition error")
