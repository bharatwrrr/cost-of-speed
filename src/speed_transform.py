'''
This file contains the functions used to generate the synthetic velocity profiles from the original velocity profiles from the dataset according
to descriptions of the two interventions (SC1 and SC2) described in the paper.


# Sample Usage (assumes you already have the original speed trace (speed_data), the corresponding timestamps (seconds_list), the corresponding 
elevation data (elevation_data), and the corresponding speed limit values for every timestamp (SPEED_LIMIT))

# SC1
adjusted_speeds, adjusted_seconds,  adjusted_speed_limit = adjust_speeds_without_coasting_numba(speed_data, seconds_list, SPEED_LIMIT)
# Or for SC2
adjusted_speeds, adjusted_seconds,  adjusted_speed_limit = speed_transform.adjust_speeds_with_coasting_numba(speed_data, seconds_list, SPEED_LIMIT)

# Calculating distance travelled in both traces
dist_list = speed_transform.distance_travelled(speed_data, seconds_list)
adjusted_dist_list = speed_transform.distance_travelled(adjusted_speeds, adjusted_seconds)

# Adjust elevation data according to modified speed trace
adjusted_elevation = speed_transform.adjust_elevation(elevation_data, dist_list, adjusted_dist_list)

# Create a figure with 2 subplots one below the other (2 rows, 1 column)
fig, axes = plt.subplots(2, 1, figsize=(14, 7))

# First subplot (top)
axes[0].plot(speed_data, label="Speed Data")
axes[0].plot(SPEED_LIMIT[:test_endpoint], label="Speed Limit")
axes[0].set_xlim(-250, 4000)  # Corrected to use set_xlim()

axes[0].set_title('Speed Data vs Speed Limit')
axes[0].legend()

# Second subplot (bottom)
axes[1].plot(adjusted_speeds, label="Adjusted Speeds")
axes[1].plot(adjusted_speed_limit, label="Speed Limit")
axes[1].set_xlim(-250, 4000)  # Corrected to use set_xlim()
axes[1].set_title('Adjusted Speeds vs Speed Limit')
axes[1].legend()

# Adjust layout to avoid overlap and make the plots fit well
plt.tight_layout()

# Show the figure
plt.show()

print(f'Increase in time: {len(adjusted_speeds)-len(speed_data)} s')
'''





import numpy as np
from scipy.interpolate import interp1d
import math
from numba.experimental import jitclass
from numba import njit, types

from enum import Enum
from dataclasses import dataclass


# Constants for acceleration/deceleration constraints (in mph/s)
MAX_ACCELERATION = 2.5  # mph/s
MAX_DECELERATION = 3.5  # mph/s

# Constant for exceed runway speed limit
MAX_EXCEED = 0.5 # mph

# Constant for speed compliance on highway only
ONLY_HWY = True
HWY_SPEED_LIMIT = np.array([45, 50, 55, 60, 65, 70, 75])

# Binary for acceleration constraint
LIMIT_ACCEL = False
LIMIT_ACCEL_OVERSPEEDING_ONLY = False

# Small constant
EPSILON = 1e-6

def get_next_anchor(speeds, sl):
    '''
    Get next anchor point including the first point
    '''
    for j in range(1, len(speeds)):
        if sl[j] != sl[j-1]:
            # if sl[j+1] == sl[j]:
            #     return j-1
            # elif sl[j-2] != sl[j-1]:

            #     continue
            return j-1, 'speed_limit_change' 
        if speeds[j-1] >= sl[j-1] and speeds[j] <= sl[j]:
            return j, 'underspeeding'
        
    return None, None

@njit
def get_next_anchor_numba(speeds, sl):
    """
    Returns (relative_index, anchor_type_code):
      anchor_type_code == 0 → speed_limit_change
      anchor_type_code == 1 → underspeeding
      (-1, -1) if no anchor found
    """
    n = speeds.shape[0]
    for j in range(1, n):
        # speed limit change
        if sl[j] != sl[j-1]:
            return j-1, 0
        # crossing back under the limit
        if speeds[j-1] >= sl[j-1] and speeds[j] <= sl[j]:
            return j, 1
    return -1, -1

# function to calculate distance travelled at between two points
def distance_travelled(speeds, time_steps):
    '''
    NOTE: speeds must be in mph or kmph
    '''
    if isinstance(speeds, np.ndarray):
        dt = np.diff(time_steps) # if speeds is np array, then so is time_steps
        seg = (speeds[:-1] + speeds[1:]) / 2 * dt / 3600
        return np.concatenate(([0.0], np.cumsum(seg)))
    
    distance = [0]
    for i in range(1, len(speeds)):
        delta_t = time_steps[i] - time_steps[i-1]
        distance.append(distance[i-1] + (speeds[i] + speeds[i-1]) / 2 * delta_t / 3600)
    return distance
    
    
def area_under_curve(x_values, y_values):
    area = 0.0
    for i in range(1, len(x_values)):
        delta_t = x_values[i] - x_values[i-1]
        area += delta_t * (y_values[i] + y_values[i-1]) / 2
    return area

def apply_acceleration_constraints(speeds, time_steps):
    constrained_speeds = [speeds[0]]
    for i in range(1, len(speeds)):
        delta_t = time_steps[i] - time_steps[i-1]
        max_speed = constrained_speeds[-1] + MAX_ACCELERATION * delta_t
        min_speed = constrained_speeds[-1] + MAX_DECELERATION * delta_t
        constrained_speeds.append(np.clip(speeds[i], min_speed, max_speed))
    return constrained_speeds

def get_decel_event(speeds, i):
    '''
    find the starting and ending points of the deceleration event centered at i
    Return None if no deceleration event is found, or if it is acceleration event at i
    '''
    if i == 0:
        return None, None
    if speeds[i] > speeds[i-1]:
        return None, None
    
    # find the starting point
    start = i
    while start > 0 and (speeds[start] < speeds[start - 1]):
        start -= 1
    # find the ending point
    end = i
    while end < len(speeds) - 1 and (speeds[end] > speeds[end + 1]):
        end += 1
    return start, end

def compute_coast_duration(distance, initial_speed, target_speed):
    '''
    Compute the duration to coast from initial_speed to target_speed such that the distance covered is distance
    '''
    if initial_speed == target_speed:
        return 0
    if initial_speed < target_speed:
        raise ValueError(f'Initial speed ({initial_speed}) must be greater than target speed ({target_speed}).')
    if distance <= 0:
        raise ValueError(f'Distance ({distance}) must be positive')
    # time = 0
    # while distance > 0:
    #     time += 1
    #     distance -= (initial_speed + target_speed) / 2 / 3600

    # dist_ = (0.5 * coast_time * (initial_speed - target_speed) + (target_speed * coast_time)) / 3600
    # coast_dist = 0.5 * coast_time * (initial_speed + target_speed) / 3600
    coast_time = 2 * distance / (initial_speed + target_speed) *3600
    return coast_time

def calculate_duration_to_cover_dist(v1, slope, distance):
    """
    Calculate the exact duration to cover a given distance while maintaining a constant slope (acceleration or deceleration).
    
    Args:
    v1 (float): Initial velocity (m/s)
    slope (float): Slope of the speed trace (acceleration or deceleration) (m/s^2)
    distance (float): Distance to be covered (m)
    
    Returns:
    float: Duration to cover the distance (s)
    """
    if slope == 0:
        print('Slope is zero')
        if v1 == 0:
            raise ValueError("Initial velocity and slope cannot both be zero.")
        return distance / v1
    
    # Coefficients of the quadratic equation
    a = 0.5 * slope * 3600
    b = v1
    c = -distance
    
    # Calculate the discriminant
    discriminant = b**2 - 4*a*c
    
    if discriminant < 0:
        raise ValueError("No real solution exists for the given parameters.")
    
    # Calculate the two possible solutions
    t1 = (-b + math.sqrt(discriminant)) / (2*a) * 3600 # in sec
    t2 = (-b - math.sqrt(discriminant)) / (2*a) * 3600 # in sec
#     print(f't1:{t1}, t2:{t2}')
    # Return the positive solution
    if t1 >= 0 and t2 >= 0:
        return min(t1, t2)
    elif t1 >= 0:
        return t1
    elif t2 >= 0:
        return t2
    else:
        raise ValueError("No positive solution exists for the given parameters.")
    
@njit
def calculate_duration_to_cover_dist_numba(v1, slope, distance):
    """
    Numba-compatible version.
    """
    # Handle zero slope
    if slope == 0.0:
        if v1 == 0.0:
            raise ValueError("Initial velocity and slope cannot both be zero.")
        return distance / v1
    
    # Quadratic coefficients
    a = 0.5 * slope * 3600.0
    b = v1
    c = -distance
    
    # Discriminant
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        raise ValueError("No real solution exists for the given parameters.")
    
    # Possible times
    sqrt_d = math.sqrt(disc)
    t1 = (-b + sqrt_d) / (2.0 * a) * 3600.0
    t2 = (-b - sqrt_d) / (2.0 * a) * 3600.0
    
    # Return the smallest non-negative solution
    if t1 >= 0.0 and t2 >= 0.0:
        return t1 if t1 < t2 else t2
    elif t1 >= 0.0:
        return t1
    elif t2 >= 0.0:
        return t2
    else:
        raise ValueError("No positive solution exists for the given parameters.")

# Function to find the next inflection point where the sign of acceleration changes, including zero crossings, only until the next window of points
def get_next_inflection(speeds, window=10):
    for i in range(1, min(len(speeds)-1, window)):
        delta_speed1 = speeds[i] - speeds[i - 1]
        delta_speed2 = speeds[i + 1] - speeds[i]
        # if (delta_speed1 * delta_speed2 < 0) or (delta_speed1 == 0 and delta_speed2 != 0) or (delta_speed1 != 0 and delta_speed2 == 0):
        #     return i

        if (delta_speed1 < 0 and delta_speed2 >= 0):
            return i

    return None

@njit
def get_next_inflection_numba(speeds, window=10):
    """
    Find the first index i (1 <= i < min(len(speeds)-1, window))
    where speed goes from decreasing to non-decreasing.
    Returns -1 if no such inflection exists.
    """
    n = speeds.shape[0]
    max_i = window if window < n-1 else n-1
    for i in range(1, max_i):
        delta1 = speeds[i]   - speeds[i-1]
        delta2 = speeds[i+1] - speeds[i]
        if delta1 < 0.0 and delta2 >= 0.0:
            return i
    return -1

def remove_duplicates(speed_data, seconds_list, speed_limit):
    '''
    If there are duplicate seconds_list values, remove that index for all three lists
    '''
    unique_indices = np.unique(seconds_list, return_index=True)[1]
    return [speed_data[i] for i in unique_indices], [seconds_list[i] for i in unique_indices], [speed_limit[i] for i in unique_indices]



@njit
def adjust_speeds_without_coasting_numba(speed_data, seconds_list, speed_limit,
                                             only_hwy=ONLY_HWY, HWY_SPEED_LIMIT=HWY_SPEED_LIMIT, MAX_EXCEED=MAX_EXCEED):
    n = speed_data.shape[0]
    # preallocate worst‐case
    out_s = np.empty(n * 2, np.float64)
    out_t = np.empty(n * 2, np.float64)
    out_sl = np.empty(n * 2, np.float64)

    # precompute distance travelled
    cum_dist = np.empty(n, np.float64)
    cum_dist[0] = 0.0
    for t in range(1, n):
        cum_dist[t] = cum_dist[t-1] + (speed_data[t] + speed_data[t-1]) / 2.0 * (seconds_list[t] - seconds_list[t-1]) / 3600.0
    i = 1
    out_s[0] = speed_data[0]
    out_t[0] = seconds_list[0]
    out_sl[0] = speed_limit[0]
    write_i = 1 # pointer for adjusted speeds
    total_distance = 0.0

    while i < n:
        dt = seconds_list[i] - seconds_list[i-1]
        sd = speed_data[i]
        sl = speed_limit[i]

        if (sd <= sl + MAX_EXCEED) or (np.isnan(sl)):
            out_s[write_i] = sd
            out_t[write_i] = out_t[write_i-1] + dt
            out_sl[write_i] = sl
            total_distance += (sd + speed_data[i-1]) / 2 / 3600 if i != 0 else 0
            write_i += 1
            i += 1
            continue

        if only_hwy and (sl not in HWY_SPEED_LIMIT):
            out_s[write_i] = sd
            out_t[write_i] = out_t[write_i-1] + dt
            out_sl[write_i] = sl
            total_distance += (sd + speed_data[i-1]) / 2 / 3600 if i != 0 else 0
            write_i += 1
            i += 1
            continue

        # find next anchor
        next_anchor, _ = get_next_anchor_numba(speed_data[i:], speed_limit[i:])
        if next_anchor < 0:
            next_anchor = n - i - 1
            raise ValueError('No anchor point found till the end. Need to add functionality for this.')
        else:
            next_anchor += i
            
            # calculate the number of points to hold at sl
            seg_dist = cum_dist[next_anchor] - cum_dist[i-1]
            pre_seg_dist = (out_s[write_i-1] + sl) / 2 * dt / 3600
            time_h = (seg_dist - pre_seg_dist) / sl * 3600
            hold_pts = int(np.ceil(time_h))
            hold_dt = time_h / (hold_pts-1) if hold_pts > 1 else 1  # avoid division by zero

            for _ in range(hold_pts): 
                out_s[write_i] = sl
                out_t[write_i] = out_t[write_i-1] + hold_dt if write_i > 0 else 0
                out_sl[write_i] = sl
                total_distance += sl * hold_dt / 3600
                write_i += 1

        i = next_anchor + 1

    return (out_s[:write_i], out_t[:write_i], out_sl[:write_i])


@njit
def get_next_anchor_numba(speeds, sl):
    """
    Returns (relative_index, code):
      code 0 = speed_limit_change
      code 1 = underspeeding
      (-1, -1) if none found
    """
    n = speeds.shape[0]
    for j in range(1, n):
        # speed limit change
        if sl[j] != sl[j-1]:
            return j-1, 0
        # crossing back under the limit
        if speeds[j-1] >= sl[j-1] and speeds[j] <= sl[j]:
            return j, 1
    return -1, -1

@njit
def get_decel_event_numba(speeds, i):
    """
    Find the start and end of the deceleration event centered at index i.
    Returns (start, end) or (-1, -1) if no deceleration event.
    """
    n = speeds.shape[0]
    # no event at boundary or if accelerating at i
    if i <= 0 or speeds[i] > speeds[i-1]:
        return -1, -1

    # find the start: move backward while decelerating
    start = i
    while start > 0 and speeds[start] < speeds[start - 1]:
        start -= 1

    # find the end: move forward while decelerating
    end = i
    while end < n - 1 and speeds[end] > speeds[end + 1]:
        end += 1

    return start, end


@njit
def get_next_anchor_numba(speeds, sl):
    """
    Returns (relative_index, code):
      code 0 = speed_limit_change
      code 1 = underspeeding
      (-1, -1) if none found
    """
    n = speeds.shape[0]
    for j in range(1, n):
        if sl[j] != sl[j-1]:
            return j-1, 0
        if speeds[j-1] >= sl[j-1] and speeds[j] <= sl[j]:
            return j, 1
    return -1, -1

@njit
def adjust_speeds_with_coasting_numba(speed_data, seconds_list, speed_limit,
                                      only_hwy=True, HWY_SPEED_LIMIT=HWY_SPEED_LIMIT, MAX_EXCEED=MAX_EXCEED):
    n = speed_data.shape[0]
    max_len = n * 2  # allocate double the length
    # outputs
    out_s   = np.empty(max_len, np.float64)
    out_t   = np.empty(max_len, np.float64)
    out_sl  = np.empty(max_len, np.float64)
    out_o   = np.empty(max_len, np.uint8)

    # cumulative distance (miles)
    cumd = np.empty(n, np.float64)
    cumd[0] = 0.0
    for i in range(1, n):
        dt = seconds_list[i] - seconds_list[i-1]
        cumd[i] = cumd[i-1] + (speed_data[i] + speed_data[i-1]) * dt / 2.0 / 3600.0

    # seed first point
    out_s[0]  = speed_data[0]
    out_t[0]  = seconds_list[0]
    out_sl[0] = speed_limit[0]
    out_o[0]  = 0
    write_i = 1
    i = 1

    while i < n:
        dt = seconds_list[i] - seconds_list[i-1]
        sd = speed_data[i]
        sl = speed_limit[i]

        # within limit or NaN
        if sd <= sl + MAX_EXCEED or np.isnan(sl):
            out_s[write_i]  = sd
            out_t[write_i]  = out_t[write_i-1] + dt
            out_sl[write_i] = sl
            out_o[write_i]  = 0
            write_i += 1
            i += 1
            continue

        # highway-only?
        if only_hwy:
            allowed = False
            for h in HWY_SPEED_LIMIT:
                if sl == h:
                    allowed = True
                    break
            if not allowed:
                out_s[write_i]  = sd
                out_t[write_i]  = out_t[write_i-1] + dt
                out_sl[write_i] = sl
                out_o[write_i]  = 0
                write_i += 1
                i += 1
                continue

        # find next anchor
        rel, code = get_next_anchor_numba(speed_data[i:], speed_limit[i:])
        if rel < 0:
            raise ValueError("No anchor point found")
        next_i = i + rel

        if code == 0:
            # speed_limit_change: hold at limit over segment distance
            seg_dist = cumd[next_i] - cumd[i-1]

            time_h = seg_dist / sl
            time_s = time_h * 3600.0
            N = int(np.ceil(time_s))

            tm = out_t[write_i-1]
            for k in range(N):
                tm += 1.0
                out_s[write_i]  = sl
                out_t[write_i]  = tm
                out_sl[write_i] = sl
                out_o[write_i]  = 1
                write_i += 1
            i = next_i + 1
            continue

        else:
            # underspeeding: coast from decel_start to next_i
            decel_start, decel_end = get_decel_event_numba(speed_data, next_i)
            if decel_start < i:
                decel_start = i

            # hold at speed limit until decel_start
            hold_t = seconds_list[decel_start] - seconds_list[i]
            hold_pts = int(np.ceil(hold_t)) + 1

            tm = out_t[write_i-1]
            dt = hold_t / hold_pts if hold_pts > 0 else 1.0
            for k in range(hold_pts):
                tm += 1
                out_s[write_i]  = sl
                out_t[write_i]  = tm
                out_sl[write_i] = sl
                out_o[write_i]  = 1
                write_i += 1
    
            dist_to_coast = cumd[decel_end] - cumd[i-1] - sl * hold_t / 3600.0 - (speed_data[i-1] + sl) / 2.0 / 3600.0

            # coast time computed using trapezoidal rule: avg speed = (v0+vt)/2
            v0 = out_s[write_i - 1]
            vt = speed_data[decel_end]
            avg_v = (v0 + vt) / 2.0
            if avg_v <= 0:
                coast_time_s = 0.0
            else:
                coast_time_s = dist_to_coast / avg_v * 3600.0

            # now coast for N points over coast_time_s
            N = int(np.ceil(coast_time_s))
            if N < 1:
                N = 1
            dv = (vt - v0) / (N-1) if N > 1 else 0.0
            dt = coast_time_s / (N-1) if N > 1 else 1.0       
            for k in range(N-1):
                tm += dt
                out_s[write_i]  = v0 + dv * (k+1)
                out_t[write_i]  = tm
                out_sl[write_i] = sl
                out_o[write_i]  = 1
                write_i += 1

        # advance to next anchor
        i = decel_end + 1

    return (out_s[:write_i],
            out_t[:write_i],
            out_sl[:write_i])



@njit
def get_next_zero_crossing_numba(speeds, start_speed, window=10):
    """
    Find the next zero crossing point in the adjusted speed data starting from a given index.
    A zero crossing is defined as a point where the adjusted speed changes from positive to negative.
    """
    n = speeds.shape[0]
    if start_speed <= 0:
        return -1
    for i in range(1, n):
        adj_s = start_speed + (speeds[i] - speeds[0])
        if adj_s <= 0:  # Check for zero crossing
            return i
        if i >= window:
            break
    return -1  # No zero crossing found within the window

NO_LIMIT = 0
EXCESSIVE_ACCEL = 1
LOCAL_MAXIMUM = 2
DECREASING_SPEED = 3
ZERO_SPEED_DIFF_WITH_SEP = 4  # New case
DEFAULT = 5

# Define the specification for the jitclass
spec = [
    ('speeds', types.float64[:]),           # 1D float64 array
    ('times', types.float64[:]),            # 1D float64 array
    ('speed_limits', types.float64[:]),     # 1D float64 array
    ('limit_accel', types.int64[:]),        # 1D int64 array (keep as int64)
    ('i', types.int64),                     # integer
    ('out_s', types.float64[:]),            # 1D float64 array
    ('out_t', types.float64[:]),            # 1D float64 array
    ('out_sl', types.float64[:]),           # 1D float64 array
    ('write_i', types.int64),               # integer
    ('cumd', types.float64[:]),             # 1D float64 array
    ('adj_d', types.float64[:]),            # 1D float64 array
    ('max_accel', types.float64),           # float
    ('eps', types.float64),                 # float
]

@jitclass(spec)
class SegmentState:
    def __init__(self, speeds, times, speed_limits, limit_accel, i, out_s, out_t, out_sl, 
                 write_i, cumd, adj_d, max_accel, eps):
        self.speeds = speeds
        self.times = times
        self.speed_limits = speed_limits
        self.limit_accel = limit_accel
        self.i = i
        self.out_s = out_s
        self.out_t = out_t
        self.out_sl = out_sl
        self.write_i = write_i
        self.cumd = cumd
        self.adj_d = adj_d
        self.max_accel = max_accel
        self.eps = eps


@njit
def apply_accel_limit_numba(speeds: np.ndarray, times: np.ndarray, 
                           speed_limits: np.ndarray, limit_accel: np.ndarray, 
                           max_accel=MAX_ACCELERATION, eps=EPSILON):
    """Main function - now much cleaner"""
    state = _initialize_state(speeds, times, speed_limits, limit_accel, max_accel, eps)
    
    while state.i < len(speeds):
        case = _classify_segment_new(state)
        _process_segment_new(state, case)
    
    return state.out_s[:state.write_i], state.out_t[:state.write_i], state.out_sl[:state.write_i]

@njit
def _initialize_state(speeds, times, speed_limits, limit_accel, max_accel, eps):
    """Initialize all arrays and state"""
    n = speeds.shape[0]
    max_len = 3 * n
    
    # Initialize output arrays
    out_s = np.empty(max_len, dtype=np.float64)
    out_t = np.empty(max_len, dtype=np.float64)
    out_sl = np.empty(max_len, dtype=np.float64)
    out_s[0] = speeds[0]
    out_t[0] = times[0]
    out_sl[0] = speed_limits[0]
    
    # Precompute cumulative distances
    cumd = np.empty(n, dtype=np.float64)
    cumd[0] = 0.0
    for i in range(1, n):
        cumd[i] = cumd[i-1] + (speeds[i] + speeds[i-1]) / 2 * (times[i] - times[i-1]) / 3600
    
    adj_d = np.empty(max_len, dtype=np.float64)
    adj_d[0] = 0.0
    
    return SegmentState(
        speeds,           # speeds
        times,            # times
        speed_limits,     # speed_limits
        limit_accel,      # limit_accel
        1,                # i
        out_s,            # out_s
        out_t,            # out_t
        out_sl,           # out_sl
        1,                # write_i
        cumd,             # cumd
        adj_d,            # adj_d
        max_accel,        # max_accel
        eps               # eps
    )

@njit
def _classify_segment(state):  # Remove -> AccelCase
    """Determine which case applies to current segment"""
    i = state.i
    
    # Case 1: No acceleration limiting
    if state.limit_accel[i] == 0:
        return NO_LIMIT
    
    speed_diff = state.speeds[i] - state.speeds[i-1]
    max_speed_change = state.max_accel * (state.times[i] - state.times[i-1])
    
    # Case 2: Excessive acceleration
    if speed_diff > max_speed_change:
        return EXCESSIVE_ACCEL
    
    # Case 3: Local maximum
    if (i < len(state.speeds) - 1 and 
        speed_diff <= 0 and 
        state.speeds[i-1] > state.speeds[i-2]):
        return LOCAL_MAXIMUM
    
    # Case 4: Zero speed difference with separation
    if abs(speed_diff) < state.eps:
        speed_sep = state.speeds[i] - state.out_s[state.write_i-1]
        if speed_sep > state.eps:
            return ZERO_SPEED_DIFF_WITH_SEP  # New case constant
        
    # Case 5: Decreasing speed
    if state.speeds[i] < state.speeds[i-1]:
        return DECREASING_SPEED
    
    return DEFAULT

@njit
def _classify_segment_new(state):  # Remove -> AccelCase
    """Determine which case applies to current segment"""
    i = state.i
    
    # Case 1: No acceleration limiting
    if state.limit_accel[i] == 0:
        return NO_LIMIT
    
    speed_diff = state.speeds[i] - state.speeds[i-1]
    max_speed_change = state.max_accel * (state.times[i] - state.times[i-1])
    
    # Case 2: Excessive acceleration
    if speed_diff > max_speed_change:
        return EXCESSIVE_ACCEL
    
    # Case 3: Local maximum
    if (i < len(state.speeds) - 1 and 
        speed_diff <= 0 and 
        state.speeds[i-1] > state.speeds[i-2]):
        return LOCAL_MAXIMUM
    
    # Case 4: Zero speed difference with separation
    if abs(speed_diff) < state.eps:
        speed_sep = state.speeds[i] - state.out_s[state.write_i-1]
        if speed_sep > state.eps:
            return ZERO_SPEED_DIFF_WITH_SEP  # New case constant
        
    # Case 5: Decreasing speed
    if state.speeds[i] < state.speeds[i-1]:
        return DECREASING_SPEED
    
    return DEFAULT


@njit
def _process_segment(state, case):
    """Process segment based on classified case"""
    if case == NO_LIMIT:
        _handle_no_limit(state)
    elif case == EXCESSIVE_ACCEL:
        _handle_excessive_accel(state)
    elif case == LOCAL_MAXIMUM:
        _handle_local_maximum(state)
    elif case == DECREASING_SPEED:
        _handle_decreasing_speed(state)
    elif case == ZERO_SPEED_DIFF_WITH_SEP:
        _handle_zero_speed_diff_with_separation(state)
    else:  # DEFAULT
        _handle_default(state)

@njit
def _process_segment_new(state, case):
    """Process segment based on classified case"""
    if case == NO_LIMIT:
        _handle_no_limit(state)
    elif case == EXCESSIVE_ACCEL:
        _handle_excessive_accel(state)
    elif case == LOCAL_MAXIMUM:
        _handle_local_maximum(state)
    elif case == DECREASING_SPEED:
        _handle_decreasing_speed(state)
    elif case == ZERO_SPEED_DIFF_WITH_SEP:
        _handle_zero_speed_diff_with_separation_new(state)
    else:  # DEFAULT
        _handle_default(state)

@njit
def _handle_no_limit(state):
    """Handle segments where acceleration limiting is disabled"""
    i, wi = state.i, state.write_i
    dt = state.times[i] - state.times[i-1]
    
    state.out_s[wi] = state.speeds[i]
    state.out_t[wi] = state.out_t[wi-1] + dt
    state.out_sl[wi] = state.speed_limits[i]
    state.adj_d[wi] = state.adj_d[wi-1] + (state.out_s[wi] + state.out_s[wi-1]) / 2 * dt / 3600
    
    state.write_i += 1
    state.i += 1

@njit
def _handle_excessive_accel(state):
    """Handle segments where acceleration exceeds limits"""
    i, wi = state.i, state.write_i
    dt = state.times[i] - state.times[i-1]

    # Apply maximum allowed acceleration
    new_speed = state.out_s[wi-1] + state.max_accel * dt
    state.out_s[wi] = new_speed
    state.out_t[wi] = state.out_t[wi-1] + dt
    state.out_sl[wi] = state.speed_limits[i]
    state.adj_d[wi] = state.adj_d[wi-1] + (state.out_s[wi] + state.out_s[wi-1]) / 2 * dt / 3600
    state.write_i += 1
    
    # Handle remaining distance gap
    dist_gap = state.cumd[i] - state.adj_d[state.write_i-1]
    if dist_gap > state.eps:
        _add_interpolated_points(state, dist_gap, state.max_accel, state.speeds[i])
    
    state.i += 1

@njit
def _force_next_point(state):
    """Forces speed at write_i to equal speed at i"""
    i, wi = state.i, state.write_i
    dt = state.times[i] - state.times[i-1]

    state.out_s[wi] = state.speeds[i]
    state.out_t[wi] = state.out_t[wi-1] + dt
    state.out_sl[wi] = state.speed_limits[i]
    state.adj_d[wi] = state.adj_d[wi-1] + (state.out_s[wi] + state.out_s[wi-1]) / 2 * dt / 3600
    state.write_i += 1

    state.i += 1

@njit
def _handle_local_maximum(state):
    """Handle speed plateaus after local maxima"""
    i, wi = state.i, state.write_i
    
    # Calculate speed separation between target peak speed and current adjusted speed
    speed_sep = state.speeds[i-1] - state.out_s[wi-1]
    fw_speed_sep = state.speeds[i] - state.out_s[wi-1]

    # Only proceed if there's a positive speed separation
    if speed_sep > 0:
        if fw_speed_sep <= 0:
            # print(f'forcing at local max: out_t: {state.out_t[wi-1]}, out_s: {state.out_s[wi-1]}')
            _force_next_point(state)
            return
        # Calculate speed slope, handling edge cases for small time differences
        if state.out_t[wi-1] - state.out_t[wi-2] < state.eps:
            speed_slope = (state.out_s[wi-1] - state.out_s[wi-3]) / (state.out_t[wi-1] - state.out_t[wi-3])
        else:
            speed_slope = (state.out_s[wi-1] - state.out_s[wi-2]) / (state.out_t[wi-1] - state.out_t[wi-2])
        
        # Calculate distance gap that needs to be covered
        dist_og = state.cumd[i]
        dist_adj = state.adj_d[wi-1]
        dist_diff = dist_og - dist_adj
        
        # Calculate time constraints
        time_to_top_speed = fw_speed_sep / speed_slope if speed_slope > state.eps else 1e5
        time_to_cover_dist = calculate_duration_to_cover_dist_numba(
            state.out_s[wi-1], speed_slope, dist_diff
        ) if dist_diff > state.eps else 1e5
        
        # Use the minimum time constraint
        additional_time = min(time_to_top_speed, time_to_cover_dist)
        if additional_time < state.eps:
            additional_time = 0
        
        # Only add points if we have meaningful additional time
        if additional_time < 1e5:
            num_points = int(np.ceil(additional_time))
            corr_dt = additional_time / num_points if num_points > 0 else 1

            # Add interpolated points to reach peak speed while covering distance
            for j in range(1, num_points + 1):
                state.out_s[wi] = state.out_s[wi-1] + speed_slope * corr_dt
                state.out_t[wi] = state.out_t[wi-1] + corr_dt
                state.out_sl[wi] = state.speed_limits[i]
                state.adj_d[wi] = state.adj_d[wi-1] + (state.out_s[wi] + state.out_s[wi-1]) / 2 * corr_dt / 3600
                state.write_i += 1
                wi += 1

        # Calculate distance gap that needs to be covered
        dist_og = state.cumd[i]
        dist_adj = state.adj_d[state.write_i-1]
        dist_diff = dist_og - dist_adj

        if dist_diff > state.eps:
            const_speed_t = dist_diff / state.out_s[wi-1] * 3600
            state.out_s[wi] = state.out_s[wi-1]
            state.out_t[wi] = state.out_t[wi-1] + const_speed_t
            state.out_sl[wi] = state.speed_limits[i]
            state.adj_d[wi] = state.adj_d[wi-1] + (state.out_s[wi] + state.out_s[wi-1]) / 2 * const_speed_t / 3600
            
            state.write_i += 1
        
        state.i += 1
        return   
  
    # If speed_sep <= 0, use default handling
    _handle_default(state)

@njit
def _handle_decreasing_speed(state):
    """Handle decreasing speed segments with multiple sub-cases"""
    speed_sep = state.speeds[state.i-1] - state.out_s[state.write_i-1]
    speed_slope = _calculate_speed_slope(state)
    
    if speed_slope > 0 and speed_sep > 0:
        
        _handle_positive_slope_decrease(state, speed_slope, speed_sep)
    elif speed_sep > 0 and abs(speed_slope) < state.eps:
        
        _handle_zero_slope_speed_decrease(state, speed_sep)
    else:
        _handle_coasting_decrease(state)

@njit
def _handle_default(state):
    """Handle normal segments with no special processing"""
    i, wi = state.i, state.write_i
    dt = state.times[i] - state.times[i-1]
    speed_diff = state.speeds[i] - state.speeds[i-1]
    
    state.out_s[wi] = max(0, state.out_s[wi-1] + speed_diff)
    
    state.out_t[wi] = state.out_t[wi-1] + dt
    state.out_sl[wi] = state.speed_limits[i]
    state.adj_d[wi] = state.adj_d[wi-1] + (state.out_s[wi] + state.out_s[wi-1]) / 2 * dt / 3600
    
    state.write_i += 1
    state.i += 1

@njit
def _handle_positive_slope_decrease(state, speed_slope: float, speed_sep: float):
    """Handle decreasing speed when there's a positive slope and speed separation"""
    i, wi = state.i, state.write_i
    dt = state.times[i] - state.times[i-1]
    
    # Check if forward speed separation is negative
    fw_speed_sep = state.speeds[i] - state.out_s[wi-1]
    if fw_speed_sep < 0:
        # If current adjusted speed > target speed, just set to target
        state.out_s[wi] = state.speeds[i]
        state.out_t[wi] = state.out_t[wi-1] + dt
        state.out_sl[wi] = state.speed_limits[i]
        state.adj_d[wi] = state.adj_d[wi-1] + (state.out_s[wi] + state.out_s[wi-1]) / 2 * dt / 3600
        state.write_i += 1
        state.i += 1
        return
    
    # Calculate time to reach top speed
    time_to_top_speed = speed_sep / speed_slope
    
    # Calculate distance gap and time to cover it
    dist_og = state.cumd[i]
    dist_adj = state.adj_d[wi-1]
    dist_diff = dist_og - dist_adj
    
    if dist_diff < state.eps:
        state.i += 1
        return
        
    dist_to_cover = dist_diff + (state.speeds[i] + state.speeds[i-1]) / 2 * dt / 3600
    time_to_cover_dist = calculate_duration_to_cover_dist_numba(
        state.out_s[wi-1], speed_slope, dist_to_cover
    )
    
    # Use minimum of the three constraints
    corr_t = min(time_to_top_speed, time_to_cover_dist, dt)
    if corr_t < state.eps:
        corr_t = 0
        
    num_points = int(np.ceil(corr_t))
    corr_dt = corr_t / num_points if num_points > 0 else 1
    
    # Add interpolated points
    for j in range(1, num_points + 1):
        # state.out_s[wi] = min(state.out_s[wi-1] + speed_slope * corr_dt, state.speeds[i])
        state.out_s[wi] = state.out_s[wi-1] + speed_slope * corr_dt
        state.out_t[wi] = state.out_t[wi-1] + corr_dt
        if state.out_s[wi] > state.speeds[i]:
            state.out_s[wi] = state.speeds[i]
            if corr_dt > dt:
                state.out_t[wi] = state.out_t[wi-1] + dt
        state.out_sl[wi] = state.speed_limits[i]
        state.adj_d[wi] = state.adj_d[wi-1] + (state.out_s[wi] + state.out_s[wi-1]) / 2 * corr_dt / 3600
        state.write_i += 1
        wi += 1
    
    state.i += 1

@njit
def _handle_zero_slope_speed_decrease(state, speed_sep: float):
    """Handle decreasing speed when slope is zero (constant speed)"""
    i, wi = state.i, state.write_i
    dt = state.times[i] - state.times[i-1]
    
    # Calculate distance gap
    dist_diff = state.cumd[i] - state.adj_d[wi-1]
    if dist_diff < state.eps:
        state.i += 1
        return
     
    v_const = state.out_s[wi-1]  # constant speed to maintain
    if v_const < state.eps:
        state.i += 1
        return
        
    # Time to cover distance at constant speed
    cover_dist_t = dist_diff / v_const * 3600
    if cover_dist_t < state.eps:
        cover_dist_t = 0
    
    # Check forward speed separation
    fw_speed_sep = state.speeds[i] - v_const
    if fw_speed_sep < 0:
        # If adjusted speed > target speed, just set to target
        state.out_s[wi] = state.speeds[i]
        state.out_t[wi] = state.out_t[wi-1] + dt
        state.out_sl[wi] = state.speed_limits[i]
        state.adj_d[wi] = state.adj_d[wi-1] + (state.out_s[wi] + state.out_s[wi-1]) / 2 * dt / 3600
        
        state.write_i += 1
        state.i += 1
        return
    
    # Add constant speed points to cover distance
    num_points = int(np.ceil(cover_dist_t))
    cover_dt = cover_dist_t / num_points if num_points > 0 else 1
    
    for j in range(1, num_points + 1):
        state.out_s[wi] = v_const
        state.out_t[wi] = state.out_t[wi-1] + cover_dt
        state.out_sl[wi] = state.speed_limits[i]
        state.adj_d[wi] = state.adj_d[wi-1] + (state.out_s[wi] + state.out_s[wi-1]) / 2 * cover_dt / 3600
        
        state.write_i += 1
        wi += 1
    
    state.i += 1

@njit
def _handle_coasting_decrease(state):
    """Handle coasting when speed is decreasing and no positive slope/separation"""
    i, wi = state.i, state.write_i
    
    # Find next inflection point and zero crossing
    next_inflection_point = get_next_inflection_numba(state.speeds[i:], window=10)
    next_zero_crossing = get_next_zero_crossing_numba(state.speeds[i:], state.out_s[wi-1], window=10)
    
    # Determine which anchor to use
    next_anchor = next_inflection_point
    if next_zero_crossing != -1:
        if next_inflection_point != -1:
            next_anchor = min(next_inflection_point, next_zero_crossing)
        else:
            next_inflection_point = get_next_inflection_numba(state.speeds[i:], window=20)
            next_anchor = next_zero_crossing
    
    if next_anchor != -1:
        speed_at_inflection = state.speeds[i + next_anchor] if next_zero_crossing == -1 else 0
        speed_sep = state.speeds[i-1] - state.out_s[wi-1]
        
        # Only coast if current speed > target speed and there's separation
        if state.out_s[wi-1] > speed_at_inflection and speed_sep > 0:
            # Calculate distances
            dist_og = state.cumd[i]
            dist_adj = state.adj_d[wi-1]
            dist_diff = dist_og - dist_adj
            
            if dist_diff > state.eps:
                # Distance to inflection point
                if next_inflection_point != -1:
                    dist_to_infl = state.cumd[i + next_inflection_point] - state.cumd[i]
                else:
                    dist_to_infl = state.cumd[i + next_anchor] - state.cumd[i]
                
                coast_dist = dist_diff + dist_to_infl
                
                # Time to coast using average speed
                coast_time_s = coast_dist / ((state.out_s[wi-1] + speed_at_inflection) / 2 / 3600)
                if coast_time_s < state.eps:
                    coast_time_s = 0
                
                num_points = int(np.ceil(coast_time_s))
                coast_dv = (speed_at_inflection - state.out_s[wi-1]) / num_points if num_points > 0 else 0
                coast_dt = coast_time_s / num_points if num_points > 0 else 1
                
                # Add coasting points
                for j in range(1, num_points + 1):
                    state.out_s[wi] = max(0, state.out_s[wi-1] + coast_dv)
                    state.out_t[wi] = state.out_t[wi-1] + coast_dt
                    state.out_sl[wi] = state.speed_limits[i]
                    state.adj_d[wi] = state.adj_d[wi-1] + (state.out_s[wi] + state.out_s[wi-1]) / 2 * coast_dt / 3600
                    state.write_i += 1
                    wi += 1
                
                # Skip to after the inflection point
                if next_inflection_point != -1:
                    state.i = i + next_inflection_point + 1
                else:
                    state.i = i + next_anchor + 1
                return
    
    # Default case: no special coasting needed, use default handler
    _handle_default(state)

@njit
def _handle_zero_speed_diff_with_separation(state):
    """Handle cases where speed_diff = 0 but speed_sep > 0"""
    i, wi = state.i, state.write_i
    dt = state.times[i] - state.times[i-1]
    
    # Calculate speed separation
    speed_sep = state.speeds[i-1] - state.out_s[wi-1]
    
    if speed_sep <= state.eps:
        # No meaningful separation, use default handling
        _handle_default(state)
        return
    
    # Calculate the last known speed slope
    speed_slope = _calculate_speed_slope(state)
    speed_slope = state.max_accel

    # Calculate distance gap that needs to be covered
    dist_og = state.cumd[i]
    dist_adj = state.adj_d[wi-1]
    dist_diff = dist_og - dist_adj
    
    # Calculate time constraints
    time_to_top_speed = speed_sep / speed_slope
    time_to_cover_dist = calculate_duration_to_cover_dist_numba(
        state.out_s[wi-1], speed_slope, dist_diff
    ) if dist_diff > state.eps else 1e5
    # time_to_cover_dist = 100
    
    # Use the minimum time constraint, but don't exceed the original time step
    acceleration_time = min(time_to_top_speed, time_to_cover_dist, dt)
    
    if acceleration_time < state.eps:
        acceleration_time = 0
    
    # Add points to reach toward the top speed
    if acceleration_time > state.eps:
        num_points = int(np.ceil(acceleration_time))
        corr_dt = acceleration_time / num_points if num_points > 0 else dt
        
        for j in range(1, num_points + 1):
            new_speed = min(state.out_s[wi-1] + speed_slope * corr_dt, state.speeds[i-1])
            state.out_s[wi] = new_speed
            state.out_t[wi] = state.out_t[wi-1] + corr_dt
            state.out_sl[wi] = state.speed_limits[i]
            state.adj_d[wi] = state.adj_d[wi-1] + (state.out_s[wi] + state.out_s[wi-1]) / 2 * corr_dt / 3600
            # if state.out_t[wi-1] > 900 and state.out_t[wi-1] < 940:
            #     print(state.out_t[wi-1], state.i)
            state.write_i += 1
            wi += 1
    
    state.i += 1

@njit
def _handle_zero_speed_diff_with_separation_new(state):
    """Handle cases where speed_diff = 0 but speed_sep > 0"""
    i, wi = state.i, state.write_i
    dt = state.times[i] - state.times[i-1]
    
    # Calculate speed separation
    speed_sep = state.speeds[i-1] - state.out_s[wi-1]
    
    if speed_sep <= state.eps:
        # No meaningful separation, use default handling
        _handle_default(state)
        return
    
    # Calculate the last known speed slope
    speed_slope = _calculate_speed_slope(state)
    speed_slope = state.max_accel    

    # Calculate distance gap that needs to be covered
    dist_og = state.cumd[i]
    dist_adj = state.adj_d[wi-1]
    dist_diff = dist_og - dist_adj
    
    # Calculate time constraints
    time_to_top_speed = speed_sep / speed_slope
    time_to_cover_dist = calculate_duration_to_cover_dist_numba(
        state.out_s[wi-1], speed_slope, dist_diff
    ) if dist_diff > state.eps else 1e5
    # time_to_cover_dist = 100
    
    # Use the minimum time constraint, but don't exceed the original time step
    acceleration_time = min(time_to_top_speed, time_to_cover_dist)
    
    # Add points to reach toward the top speed
    if acceleration_time > state.eps:
        num_points = int(np.ceil(acceleration_time))
        corr_dt = acceleration_time / num_points if num_points > 0 else dt
        
        for j in range(1, num_points + 1):
            new_speed = min(state.out_s[wi-1] + speed_slope * corr_dt, state.speeds[i-1])
            state.out_s[wi] = new_speed
            state.out_t[wi] = state.out_t[wi-1] + corr_dt
            state.out_sl[wi] = state.speed_limits[i]
            state.adj_d[wi] = state.adj_d[wi-1] + (state.out_s[wi] + state.out_s[wi-1]) / 2 * corr_dt / 3600
            # if state.out_t[wi-1] > 900 and state.out_t[wi-1] < 940:
            #     print(state.out_t[wi-1], state.i)
            state.write_i += 1
            wi += 1
    
    # Handle any remaining distance with constant speed
    remaining_dist = state.cumd[i] - state.adj_d[state.write_i-1]
    if remaining_dist > state.eps and state.out_s[state.write_i-1] > state.eps:
        const_speed_time = remaining_dist / state.out_s[state.write_i-1] * 3600
        state.out_s[wi] = state.out_s[wi-1]
        state.out_t[wi] = state.out_t[wi-1] + const_speed_time
        state.out_sl[wi] = state.speed_limits[i]
        state.adj_d[wi] = state.adj_d[wi-1] + state.out_s[wi-1] * const_speed_time / 3600
        
        state.write_i += 1
    
    state.i += 1

@njit
def _add_interpolated_points(state, distance: float, accel: float, target_speed: float):
    """Add interpolated points to cover distance gap"""
    if distance <= state.eps:
        return
    
    wi = state.write_i
    
    # Calculate time needed to cover distance
    dist_cover_t = calculate_duration_to_cover_dist_numba(state.out_s[wi-1], accel, distance)
    if dist_cover_t < state.eps:
        dist_cover_t = 0
        
    num_points = int(np.ceil(dist_cover_t))
    if num_points == 0:
        return
        
    corr_dt = dist_cover_t / num_points
    for j in range(1, num_points + 1):
        new_speed = min(state.out_s[wi-1] + accel * corr_dt, target_speed)
        state.out_s[wi] = new_speed
        state.out_t[wi] = state.out_t[wi-1] + corr_dt
        state.out_sl[wi] = state.speed_limits[state.i]
        state.adj_d[wi] = state.adj_d[wi-1] + (state.out_s[wi] + state.out_s[wi-1]) / 2 * corr_dt / 3600
        state.write_i += 1
        wi += 1


# Helper functions
@njit
def _calculate_speed_slope(state):
    """Calculate speed slope, handling edge cases"""
    wi = state.write_i
    if state.out_t[wi-1] - state.out_t[wi-2] < state.eps:
        if wi >= 3 and state.out_t[wi-1] - state.out_t[wi-3] >= state.eps:
            return (state.out_s[wi-1] - state.out_s[wi-3]) / (state.out_t[wi-1] - state.out_t[wi-3])
        return 0.0
    return (state.out_s[wi-1] - state.out_s[wi-2]) / (state.out_t[wi-1] - state.out_t[wi-2])


def adjust_elevation(elevation_og, distance_og, distance_adjusted):
    epsilon = 1e-10
    distance_og = np.array(distance_og)
    unique_indices = np.unique(distance_og, return_index=True)[1]
    distance_og = distance_og[unique_indices]
    elevation_og = np.array(elevation_og)[unique_indices]
    f = interp1d(distance_og, elevation_og, fill_value="extrapolate")
    return f(distance_adjusted).tolist()

