from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union
from config import MAIN_DRIVE_PATH, BACKUP_DRIVE_PATH, validate_paths
from config import ICEV_MODELS_PATH, BEV_MODELS_PATH, PHEV_MODELS_PATH, HEV_MODELS_PATH, VEHICLE_MODELS_PATH
from config import JOURNEY_TO_VEHICLE_DAY1, JOURNEY_TO_VEHICLE_DAY2, JOURNEY_TO_VEHICLE_DAY3, JOURNEY_TO_VEHICLE_DAY4, KWH_PER_GGE
from config import *
import gzip
import ast
import numpy as np
import pandas as pd
from datetime import datetime
import fastsim as fsim
import json
import os
import re
import ast  # Import ast to safely evaluate strings that look like lists

# Logging
import logging
logger = logging.getLogger(__name__)


def load_data(which_data: Dict, which_step: Optional[str] = 'ws04'):
    """
    Load data from the configured directory after validating paths.
    inputs:
        which_data = {'day': 1, 'hour': 0, 'geo': '9zv'}
        which_step = 'ws03'
    """
    # Validate paths
    validate_paths(which_data)

    # Check for correct structure of which_data
    assert all(key in which_data for key in ['day', 'hour','geo']), f"Incorrect structure for argument at position 0. the dictionary must contain 'day', 'hour', and 'geo' keys."
    assert isinstance(which_data['day'], int), "which_data['day'] is not an integer"
    assert which_data['day'] in range(1, 5), "which_data['day'] is not in the range 1-4"
    assert isinstance(which_data['hour'], int), "which_data['hour'] is not an integer"
    assert which_data['hour'] in range(0, 24), "which_data['hour'] is not in the range 0-23"
    assert isinstance(which_data['geo'], str), "which_data['geo'] is not a string"
    assert len(which_data['geo']) == 3, "which_data['geo'] should be a 3 characted string"
    
    ##TODO: for whichever path, get the file name acc to which_data. check if the geohash file exists
    if which_step == 'ws04':
        filepath_post = f"interpolated_hr={which_data['hour']}"
    else:
        return "Functionality to get data from geohash and organized folders is not present at the moment."
    
    if which_data['day'] == 1:
        filepath_pre = os.path.join(MAIN_DRIVE_PATH, 'VM2021', 'dt=2021-01-06', f"hr={which_data['hour']}", filepath_post)
    elif which_data['day'] == 2:
        filepath_pre = os.path.join(MAIN_DRIVE_PATH, 'VM2021', 'dt=2021-04-28', f"hr={which_data['hour']}", filepath_post)
    elif which_data['day'] == 3:
        filepath_pre = os.path.join(BACKUP_DRIVE_PATH, 'dt=2021-08-11', f"hr={which_data['hour']}", filepath_post)
    elif which_data['day'] == 4:
        filepath_pre = os.path.join(BACKUP_DRIVE_PATH, 'dt=2021-10-06', f"hr={which_data['hour']}", filepath_post)

    filepath = os.path.join(filepath_pre, f"processed_trajectories_{which_data['geo']}.json.gz")
    assert os.path.exists(filepath), f"filepath {filepath} does not exists."
    
    with gzip.open(filepath, 'rt') as f:
        data = json.load(f)
    
    return data
    
def load_journey_vehicle_mapping(day: int):
    """
    Load the journey to vehicle model mapping for the given day.
    """
    assert day in range(1, 5), "day should be in the range 1-4"
    if day == 1:
        filepath = JOURNEY_TO_VEHICLE_DAY1
    elif day == 2:
        filepath = JOURNEY_TO_VEHICLE_DAY2
    elif day == 3:
        filepath = JOURNEY_TO_VEHICLE_DAY3
    elif day == 4:
        filepath = JOURNEY_TO_VEHICLE_DAY4
    
    assert os.path.exists(filepath), f"filepath {filepath} does not exists."
    
    with open(filepath, 'r') as f:
        journey_veh_map = json.load(f)
    
    return journey_veh_map

def get_epa_class(journey: str, journey_veh_map: pd.DataFrame):
    """
    Get the epa class for the given journey.
    """
    veh_model = journey_veh_map.loc[journey, 'epa_class']
    return veh_model


@dataclass
class Trajectory:
    time: List[str]  # Assuming time is stored as strings (e.g., ISO format)
    speed_mph: List[float]
    elevation_m: List[float]
    geohash: List[str]
    latitude: Optional[List[float]] = None
    longitude: Optional[List[float]] = None
    speed_limit_mph: Optional[List[float]] = None
    veh_model: Optional[str] = None
    veh_make: Optional[str] = None
    veh_year: Optional[int] = None
    epa_class: Optional[str] = None
    time_s: List[float] = field(init=False, default_factory=list)  # This field is not included in the __init__ method
    speed_mps: List[float] = field(init=False, default_factory=list)
    accel_mpss: List[float] = field(init=False, default_factory=list)  
    jerk_mpsss: List[float] = field(init=False, default_factory=list)
    grade: List[float] = field(init=False, default_factory=list) # Grade in fraction
    road_type: Optional[List[str]] = None


    def __post_init__(self):
        """
        Perform basic validation to ensure all lists have the same length.
        """
        lengths = [
            len(value) for key, value in self.__dict__.items()
            if isinstance(value, list) and value is not None and key not in ['time_s', 'speed_mps', 'accel_mpss', 'jerk_mpsss', 'grade']
        ]
        if lengths and len(set(lengths)) != 1:
            raise ValueError("All present list attributes must have the same number of elements.")

    
        # Convert time to seconds and speed to mps
        if isinstance(self.time[0], str):
            self.time_s = self._convert_time_to_seconds(self.time)  
        else:
            self.time_s = self.time
        self.speed_mph = np.clip(self.speed_mph, a_min=0.0, a_max=None).tolist()          
        self.speed_mps = [speed * 0.44704 for speed in self.speed_mph]

        # Compute acceleration and jerk
        self.accel_mpss = self._compute_acceleration()
        self.jerk_mpsss = self._compute_jerk()

        # Compute grade
        self.grade = self._compute_grade()
        
        if self.speed_limit_mph is not None:
            self.speed_limit_mph = [np.nan if s is None else s for s in self.speed_limit_mph]
            self.speed_limit_mph = self._transform_speed_limit(self.speed_limit_mph)
            self.speed_limit_mph = self.smooth_speed_limit(self.speed_limit_mph)

    @staticmethod
    def _transform_speed_limit(speed_limit_mph):
        """
        TODO: complete this method to cover all possible cases of speed_limit_mph
        """


        # Convert speed_limit_mph from a list of str to a list of float
        for i, sl in enumerate(speed_limit_mph):
            try:
                if isinstance(sl, str):
                    if sl.startswith('[') and sl.endswith(']'):
                        s_list = ast.literal_eval(sl)
                        if len(s_list)>1:
                            s_list_cleaned = []
                            for item in s_list:
                                if 'nan' not in item and item != '':
                                    try:
                                        # Handle both ' mph' and 'mph' cases
                                        item_cleaned = int(item.replace(' mph', '').replace('mph', '').strip().split(';')[0])
                                    except:
                                        item_cleaned = np.nan
                                    s_list_cleaned.append(item_cleaned)
                                else:
                                    s_list_cleaned.append(1e10)
                            cleaned_speed_limit = min(s_list_cleaned)
                        else:
                            cleaned_speed_limit = int(s_list[0].replace(' mph', '').replace('mph', '').strip())
                        speed_limit_mph[i] = cleaned_speed_limit
                    elif sl == 'nan':
                        speed_limit_mph[i] = float('nan')
                    elif ';' in sl:
                        speed_limit_mph[i] = float(sl.split(';')[0])
                    elif ',' in sl:
                        speed_limit_mph[i] = float(sl.split(',')[0])
                    elif '-' in sl:
                        speed_limit_mph[i] = float(sl.split('-')[0])
                    elif ' ' in sl:
                        speed_limit_mph[i] = float(sl.split()[0])
                    
                    else:
                        try:
                            speed_limit_mph[i] = float(sl)
                        except Exception as e:
                            # logger.warning(f'[ERROR] Speed limit entry {sl} has invalid format')
                            speed_limit_mph[i] = float('nan')
                elif isinstance(sl, (float, int)):
                    speed_limit_mph[i] = sl
                    # logger.warning(f"[ERROR] Invalid speed limit value at index {i}: {sl}")
            except:
                # logger.warning(f"[ERROR] Invalid speed limit value at index {i}: {sl}")
                speed_limit_mph[i] = float('nan')
        return speed_limit_mph
    
    @staticmethod
    def _transform_speed_limit_np(speed_limit_mph):
        arr = np.array(speed_limit_mph, dtype=object)

        # 1) Mask out literal 'nan' → actual NaN
        mask_nan = np.equal(arr, 'nan')

        # 2) Identify bracketed-list entries
        mask_list = np.fromiter(
            (isinstance(x, str) and x.startswith('[') and x.endswith(']')
            for x in arr),
            dtype=bool
        )

        # 3) For those list-entries, parse them and take the min
        def _min_from_list(text):
            try:
                items = ast.literal_eval(text)
            except Exception:
                return np.nan
            clean = []
            for item in items:
                if not item or 'nan' in item:
                    continue
                # strip off any “ mph”/“mph”, take up to first “;”
                num = re.split(r";", item.replace(' mph','').replace('mph','').strip(), 1)[0]
                try:
                    clean.append(float(num))
                except ValueError:
                    pass
            return min(clean) if clean else np.nan

        # apply only on the list entries
        list_vals = [ _min_from_list(x) for x in arr[mask_list] ]
        arr[mask_list] = list_vals

        # 4) Everything else → string → strip units → split on first of ; , - or whitespace
        #    (this runs in C under the hood)
        strs = arr.astype(str)
        strs = np.char.replace(strs, ' mph', '')
        strs = np.char.replace(strs, 'mph', '')
        strs = np.char.strip(strs)

        # vectorized “split on first delimiter”
        def _take_before_delim(s):
            return re.split(r"[;,\-\s]", s, 1)[0]

        vs = np.vectorize(_take_before_delim)(strs)

        # 5) Convert to float, invalid → NaN
        def _to_float(x):
            try:
                return float(x)
            except:
                return np.nan

        result = np.vectorize(_to_float)(vs)

        # 6) restore the explicit 'nan' entries
        result[mask_nan] = np.nan

        return result.tolist()
    
    @staticmethod
    def smooth_speed_limit(speed_limit_mph, window_size=9, nan_max_count=20):
        '''
        if there is anomalous short term fluctuations in speed limit, smooth it out. 
        E.g., if speed_limit = [25, 25, 20, 25, 25], it will be smoothed to [25, 25, 25, 25, 25]
        E.g., if speed_limit = [25, 25, 20, 20, 25], it will be smoothed to [25, 25, 25, 25, 25]
        E.g., if speed_limit = [25, 20, 25, 20, 25], it will be smoothed to [25, 25, 25, 25, 25] or [20, 20, 20, 20, 20] based on frequency of occurrence
        However, since window size is 3, if speed_limit = [25, 20, 20, 20, 25], it will not be changed.
        '''
        smoothed_speed_limit = speed_limit_mph.copy()
        assert window_size % 2 == 1, "Window size should be an odd number." 
        # Fill NaN values if the number of NaNs is less than nan_max_count
        for i in range(1, len(speed_limit_mph) - 1):
            if np.isnan(speed_limit_mph[i]):
                start = i - 1
                while i < len(speed_limit_mph) and np.isnan(speed_limit_mph[i]):
                    i += 1
                end = i
                if end - start - 1 <= nan_max_count and start >= 0 and end < len(speed_limit_mph) and speed_limit_mph[start] == speed_limit_mph[end]:
                    smoothed_speed_limit[start+1:end] = [speed_limit_mph[start]] * (end - start - 1)
        # Smooth the speed limit
        for i in range(len(smoothed_speed_limit)):
            if i < window_size - 1 or i > len(speed_limit_mph) - window_size:
                continue
            else:
                if window_size % 2 == 0:
                    window = smoothed_speed_limit[i - window_size//2:i + window_size//2]
                else:
                    window = smoothed_speed_limit[i - window_size//2:i + window_size//2 + 1]
                if len(set(window)) == 1:
                    smoothed_speed_limit[i] = window[0]
                else:
                    smoothed_speed_limit[i] = max(set(window), key=window.count)

        return smoothed_speed_limit

    def _compute_acceleration(self) -> List[float]:
        """
        Compute acceleration in m/s^2 from speed data.
        """
        if not self.time_s or not self.speed_mps:
            return []
        
        accel_mpss = []
#         for i in range(0, len(self.time_s)-1):
#             time_diff = self.time_s[i+1] - self.time_s[i]
#             speed_diff = self.speed_mps[i+1] - self.speed_mps[i]
#             accel_mpss.append(speed_diff / time_diff)
            
        time_diff = np.diff(self.time_s)
        speed_diff = np.diff(self.speed_mps)
        accel_mpss = speed_diff / time_diff

        accel_mpss = np.append(accel_mpss, 0)  # Assume acceleration at the last point is 0
        return accel_mpss.tolist()

    def _compute_jerk(self) -> List[float]:
        """
        Compute jerk in m/s^3 from acceleration data.
        """
        if not self.time_s or not self.accel_mpss:
            return []
        
        jerk_mpsss = []

        time_diff = np.diff(self.time_s)
        accel_diff = np.diff(self.accel_mpss)
        jerk_mpsss = accel_diff / time_diff

        jerk_mpsss = np.append(jerk_mpsss, 0)
        return jerk_mpsss.tolist()

    def _compute_grade(self) -> List[float]:
        """
        Compute grade in percentage from elevation data.
        TODO: Add functionality to handle NaN values in elevation data
        """
        if not self.elevation_m:
            return []
        
        # Calculate the change in elevation
        delta_elevation = np.diff(self.elevation_m)
        delta_elevation = np.append(delta_elevation, delta_elevation[-1])  # Assign the last value to the second to last value

        # Calculate the horizontal distance using speed and time intervals
        delta_time = np.diff(self.time_s)
        delta_time = np.append(delta_time, delta_time[-1])  
        delta_distance = np.array(self.speed_mps) * delta_time

        # Calculate the grade, handle division by zero
        epsilon = 1e-10
        grade = np.where(delta_distance != 0, delta_elevation / (delta_distance + epsilon), 0)
        return grade.tolist()


    def _convert_time_to_seconds(self, time_list: List[str]) -> List[float]:
        """
        Convert a list of time strings to seconds from the starting point.
        """
        if not time_list:
            return []
        
        # Define the format of the time strings
        time_format = '%Y-%m-%dT%H:%M:%S.%f%z'
        
        start_time = datetime.strptime(time_list[0], time_format)
        return [(datetime.strptime(t, time_format) - start_time).total_seconds() for t in time_list]


    def calculate_distance_mi(self) -> float:
        """
        Calculate distance in miles using trapezoidal integration on speed (mph) over time (s).
        """
        time_s = np.asarray(self.time_s)
        speed_mph = np.asarray(self.speed_mph)

        delta_time = np.diff(time_s)  # seconds
        avg_speed = (speed_mph[:-1] + speed_mph[1:]) / 2  # mph

        # distance = speed (mph) * time (h) => convert sec to hour
        distance = np.sum(avg_speed * (delta_time / 3600))

        return distance
    
    def _convert_distance_to_km(self, distance_mi: float) -> float:
        """
        Convert distance from miles to kilometers.
        """
        return distance_mi * 1.60934
    

    def calculate_ki(self) -> float:
        """
        Calculate the kinetic intensity in 1/m.
        """
        v = self.speed_mps
        t = self.time_s
        h = self.elevation_m
        total_distance_km = self._convert_distance_to_km(self.calculate_distance_mi())
        char_acc = sum([max(0, 0.5 * (v[j+1]**2 - v[j]**2) + ACCEL_GRAVITY * (h[j+1] - h[j])) 
                        for j in range(len(h)-1)]) / (total_distance_km * 1000)
        
        aero_speed = sum([0.25 * (v[j+1]**3 + (v[j+1]**2)*v[j] + v[j+1]*(v[j]**2) + (v[j]**3)) * (t[j+1] - t[j]) for j in range(len(v)-1)]) / (total_distance_km * 1000)
        
        return char_acc / aero_speed
        

    def summary(self):
        """
        Returns a summary of the trajectory with basic statistics.
        """
        return {
            "num_points": len(self.time),
            "duration_seconds": self.time_s[-1] if self.time_s else None,
            "distance_km": self.calculate_distance_mi(),
            "start_time": self.time[0] if self.time else None,
            "end_time": self.time[-1] if self.time else None,
            "start_geohash": self.geohash[0] if self.geohash else None,
            "end_geohash": self.geohash[-1] if self.geohash else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        """
        Creates a Trajectory instance from a dictionary (JSON object).
        """
        return cls(
            time=data.get("Time", []),
            speed_mph=data.get("Speed", []),
            elevation_m=data.get("Elevation", []),
            geohash=data.get("Geohash", None),
            latitude=data.get("Latitude", None),  # Optional
            longitude=data.get("Longitude", None),  # Optional
            speed_limit_mph=data.get("SpeedLimit", None),  # Optional
            road_type=data.get("RoadType", None)  # New field, optional
        )



@dataclass
class TrajectoryModified:
    time_s: List[float] 
    speed_mph: List[float]
    elevation_m: List[float]
    speed_limit_mph: Optional[List[float]] = None
    veh_model: Optional[str] = None
    veh_make: Optional[str] = None
    veh_year: Optional[int] = None
    epa_class: Optional[str] = None
    grade: List[float] = field(init=False, default_factory=list) # Grade in fraction

    def __post_init__(self):
        """
        Perform basic validation to ensure all lists have the same length.
        """
        lengths = [len(self.time_s), len(self.speed_mph), len(self.elevation_m)]

        if self.speed_limit_mph is not None:
            lengths.append(len(self.speed_limit_mph))
        if len(set(lengths)) != 1:
            raise ValueError("All attributes must have the same number of elements.")

        # Convert time to seconds and speed to mps
        self.speed_mps = [speed * 0.44704 for speed in self.speed_mph]

        # Compute grade
        self.grade = self._compute_grade()

        if self.speed_limit_mph is not None:
            self.speed_limit_mph = Trajectory._transform_speed_limit(self.speed_limit_mph)
            self.speed_limit_mph = Trajectory.smooth_speed_limit(self.speed_limit_mph)
        
    def _compute_grade(self) -> List[float]:
        """
        Compute grade in percentage from elevation data.
        TODO: Refactor this method to use the Trajectory class method
        """
        if not self.elevation_m:
            return []
        
        # Calculate the change in elevation
        delta_elevation = np.diff(self.elevation_m)
        delta_elevation = np.append(delta_elevation, delta_elevation[-1])
    
        # Calculate the horizontal distance using speed and time intervals
        delta_time = np.diff(self.time_s)
        delta_time = np.append(delta_time, delta_time[-1])
        delta_distance = np.array(self.speed_mps) * delta_time

        # Calculate the grade, handle division by zero
        epsilon = 1e-10
        grade = np.where(delta_distance != 0, delta_elevation / (delta_distance + epsilon), 0)
        return grade
    
    
    def calculate_distance_mi(self) -> float:
        """
        Calculate distance in miles using trapezoidal integration on speed (mph) over time (s).
        """
        if not self.time_s or not self.speed_mph or len(self.time_s) < 2:
            return 0.0

        time_s = np.asarray(self.time_s)
        speed_mph = np.asarray(self.speed_mph)

        delta_time = np.diff(time_s)  # seconds
        avg_speed = (speed_mph[:-1] + speed_mph[1:]) / 2  # mph

        # distance = speed (mph) * time (h) => convert sec to hour
        distance = np.sum(avg_speed * (delta_time / 3600))

        return distance
    
    def _convert_distance_to_km(self, distance_mi: float) -> float:
        """
        Convert distance from miles to kilometers.
        """
        return distance_mi * 1.60934
    
    def calculate_ki(self) -> float:
        """
        Calculate the kinetic intensity in 1/m.
        """
        if not self.time_s or not self.speed_mph:
            return None

        v = self.speed_mps
        t = self.time_s
        h = self.elevation_m
        total_distance_km = self._convert_distance_to_km(self.calculate_distance_mi())
        char_acc = sum([max(0, 0.5 * (v[j+1]**2 - v[j]**2) + ACCEL_GRAVITY * (h[j+1] - h[j])) 
                        for j in range(len(h)-1)]) / (total_distance_km * 1000)
        
        aero_speed = sum([0.25 * (v[j+1]**3 + (v[j+1]**2)*v[j] + v[j+1]*(v[j]**2) + (v[j]**3)) * (t[j+1] - t[j]) for j in range(len(v)-1)]) / (total_distance_km * 1000)
        
        return char_acc / aero_speed

    @classmethod
    def from_dict(cls, data: Dict):
        """
        Creates a TrajectoryModified instance from a dictionary (JSON object).
        """

        return cls(
            time_s=data.get("Seconds", []),
            speed_mph=data.get("Speed", []),
            elevation_m=data.get("Elevation", []),
            speed_limit_mph=data.get("SpeedLimit")
        )



@dataclass
class TrajectoryAnalysis:
    trajectory: Union[Trajectory, TrajectoryModified]
    epa_class: Optional[str] = None
    pt_type: Optional[str] = None
    vehicle_model: Optional[object] = None
    sim_drive_post: object = field(init=False)
    use_f3: Optional[bool] = False # whether to use fastsim3 instead of fastsim2

    def __post_init__(self):
        # Ensure either epa_class and pt_type are provided or vehicle_model is provided
        if not self.vehicle_model and (not self.epa_class or not self.pt_type):
            raise ValueError("Either provide both epa_class and pt_type, or provide vehicle_model.")
        
        # Initialize the vehicle model using the epa_class if vehicle_model is not provided  
        if not self.vehicle_model:
            self.vehicle_model = self._initialize_vehicle_model(self.epa_class)
        else:
            self.epa_class = self.vehicle_model.scenario_name if not self.use_f3 else self.vehicle_model.to_pydict()['name']
            self.pt_type = self.vehicle_model.veh_pt_type if not self.use_f3 else self.vehicle_model.veh_type()
            if self.pt_type == 'Conv':
                self.pt_type = 'ICEV'

    def _initialize_vehicle_model(self, epa_class: str):
        """
        Initialize the vehicle model based on the epa_class.
        """
        if self.use_f3:
            file_name = f"{epa_class.lower().replace(' ', '_')}.json"

        # Initialize the vehicle model using FASTSim
        if self.pt_type == 'ICEV':
            if self.use_f3:
                model_file = os.path.join(VEHICLE_MODELS_PATH['icev'], file_name)
                return fsim.Vehicle.from_f2_file(model_file)
            tuned_path = ICEV_MODELS_PATH
            tuned_vehicles = pd.read_csv(tuned_path, index_col=0)
        elif self.pt_type == 'BEV':
            if self.use_f3:
                model_file = os.path.join(VEHICLE_MODELS_PATH['bev'], file_name)
                return fsim.Vehicle.from_f2_file(model_file)
            tuned_path = BEV_MODELS_PATH
            tuned_vehicles = pd.read_csv(tuned_path, index_col=0)
        elif self.pt_type == 'PHEV':
            if self.use_f3:
                model_file = os.path.join(VEHICLE_MODELS_PATH['phev'], file_name)
                return fsim.Vehicle.from_f2_file(model_file)
            tuned_path = PHEV_MODELS_PATH
            tuned_vehicles = pd.read_csv(tuned_path, index_col=0)
        elif self.pt_type == 'HEV':
            if self.use_f3:
                model_file = os.path.join(VEHICLE_MODELS_PATH['hev'], file_name)
                return fsim.Vehicle.from_f2_file(model_file)
            tuned_path = HEV_MODELS_PATH
            tuned_vehicles = pd.read_csv(tuned_path, index_col=0)

        vehicle_row = tuned_vehicles.loc[tuned_vehicles['scenario_name'] == self.epa_class]
        veh_dict = {}
        for col in vehicle_row.columns:
            veh_dict[col] = vehicle_row[col].values[0]
        
        return fsim.vehicle.Vehicle.from_dict(veh_dict)

    def run_simdrive(self):
        """
        Run the simulation drive using the FASTSim vehicle model.
        Returns: simdrive.SimDrivePost
        """
        # Initialize the cycle
        if self.use_f3:
            if isinstance(self.trajectory.time_s, np.ndarray):
                self.trajectory.time_s = self.trajectory.time_s.tolist()
            if isinstance(self.trajectory.speed_mps, np.ndarray):
                self.trajectory.speed_mps = self.trajectory.speed_mps.tolist()
            if isinstance(self.trajectory.grade, np.ndarray):
                self.trajectory.grade = self.trajectory.grade.tolist()

            cyc_dict = {'time_seconds': self.trajectory.time_s, 
                        'speed_mps': self.trajectory.speed_mps,
                        'grade': self.trajectory.grade}
            cyc = fsim.Cycle.from_pydict(cyc_dict)
            sim_params_dict = fsim.SimParams.default().to_pydict()
            sim_params_dict['trace_miss_opts'] = "Allow"
            sim_params = fsim.SimParams.from_pydict(sim_params_dict)
            sim_drive = fsim.SimDrive(self.vehicle_model, cyc, sim_params)

            try:
                sim_drive.walk()
                return sim_drive.to_pydict() # returns a python dictionary
            except:
                sim_drive_f2 = sim_drive.to_fastsim2()
                sim_drive_f2.sim_drive()
                return sim_drive_f2 # returns a RustSimDrive object (TODO: probably need to make it consistent)
            
        cyc_dict = {'time_s': self.trajectory.time_s, 
                    'mps': self.trajectory.speed_mps,
                    'grade': self.trajectory.grade}
        cyc = fsim.cycle.Cycle.from_dict(cyc_dict)
        sim_drive = fsim.simdrive.SimDrive(cyc, self.vehicle_model)
        if self.pt_type == 'HEV' or self.pt_type == 'PHEV':
            sim_drive.sim_drive(init_soc=0.5)
        else:
            sim_drive.sim_drive()
        return fsim.simdrive.SimDrivePost(sim_drive) # returns a python SimDrivePost object

    def get_simdrive_post(self):
        """
        Get the simulation drive post object.
        """
        self.sim_drive_post = self.run_simdrive()
        return self.sim_drive_post

    def get_engine_power_out(self):
        """
        Get the engine power output from the simulation drive post object.
        """
        if self.pt_type == 'BEV':
            raise ValueError("Engine power is not applicable for BEVs.")
        if not hasattr(self, 'sim_drive_post'):
            self.get_simdrive_post()
        if self.use_f3:
            if self.pt_type == 'ICEV':
                if isinstance(self.sim_drive_post, dict):
                    return np.array(self.sim_drive_post['veh']['pt_type']['ConventionalVehicle']['fc']['history']['pwr_prop_watts']) / 1e3
            elif self.pt_type == 'HEV':
                if isinstance(self.sim_drive_post, dict):
                    return np.array(self.sim_drive_post['veh']['pt_type']['HybridElectricVehicle']['fc']['history']['pwr_prop_watts']) / 1e3
        return self.sim_drive_post.fc_kw_out_ach.tolist()

    def get_motor_power_out(self):
        """
        Get the motor power output from the simulation drive post object.
        """
        if self.pt_type == 'ICEV':
            raise ValueError("Motor power is not applicable for ICEVs.")
        if not hasattr(self, 'sim_drive_post'):
            self.get_simdrive_post()
        if self.use_f3:
            if self.pt_type == 'BEV':
                if isinstance(self.sim_drive_post, dict):
                    return np.array(self.sim_drive_post['veh']['pt_type']['BatteryElectricVehicle']['res']['history']['pwr_mech_prop_out_watts']) / 1e3
            elif self.pt_type == 'HEV':
                if isinstance(self.sim_drive_post, dict):
                    return np.array(self.sim_drive_post['veh']['pt_type']['HybridElectricVehicle']['em']['history']['pwr_mech_prop_out_watts']) / 1e3

        return self.sim_drive_post.mc_mech_kw_out_ach # TODO: chck if this works for fastsim2
    
    def get_fuel_energy_consumed(self):
        """
        Get the fuel energy consumed from the simulation drive post object.
        """
        if self.pt_type == 'BEV':
            raise ValueError("Fuel energy is not applicable for BEVs.")
        if not hasattr(self, 'sim_drive_post'):
            self.get_simdrive_post()
        if self.use_f3:
            if self.pt_type == 'ICEV':
                if isinstance(self.sim_drive_post, dict): 
                    return np.array(self.sim_drive_post['veh']['pt_type']['ConventionalVehicle']['fc']['history']['energy_fuel_joules'])[-1] / 1e3 / 3600 # returns in kWh
            elif self.pt_type == 'HEV':
                if isinstance(self.sim_drive_post, dict):
                    return np.array(self.sim_drive_post['veh']['pt_type']['HybridElectricVehicle']['fc']['history']['energy_fuel_joules'])[-1] / 1e3 / 3600 # returns in kWh

        return sum(self.sim_drive_post.fs_kwh_out_ach) 

    def get_fuel_consumption(self):
        """
        Get the fuel consumption from the simulation drive post object.
        """
        if not hasattr(self, 'sim_drive_post'):
            self.get_simdrive_post()
        return self.get_fuel_energy_consumed() / KWH_PER_GGE #TODO: need to cross check with fastsim2
    
    def get_CO2_emissions(self):
        """
        Get the CO2 emissions from the amount of fuel consumed.
        TODO: Add a method to get CO2 emissions from the simulation drive post object.
        """
        pass

    def get_battery_energy_consumed(self):
        """
        Get the battery energy consumed from the simulation drive post object.
        """
        if self.pt_type == 'ICEV':
            raise ValueError("Battery energy is not applicable for ICEVs.")
        return self.sim_drive_post.ess_dischg_kj / 3.6e3

    def compute_energy_consumption(self):
        """
        Compute the energy consumption for the given trajectory and vehicle model.
        Returns: float (energy consumed in kWh)
        """
        if self.pt_type == 'BEV':
            return self.get_battery_energy_consumed()
        elif self.pt_type == 'PHEV' or self.pt_type == 'HEV':
            return self.get_fuel_energy_consumed() + self.get_battery_energy_consumed()
        elif self.pt_type == 'Conv' or self.pt_type == 'ICEV':
            return self.get_fuel_energy_consumed()
        
    def get_dist_m(self):
        """
        Get the distance in meters from the simulation drive post object.
        Returns: float (distance in meters)
        """
        if isinstance(self.sim_drive_post, dict):
            return self.sim_drive_post['cyc']['dist_meters'][-1]
        
        return sum(self.sim_drive_post.dist_m)
    
    def get_dist_mi(self):
        """
        Get the distance in miles from the simulation drive post object.
        Returns: float (distance in miles)
        """
        return self.get_dist_m() / 1609.34
        
    def compute_fuel_economy(self):
        """
        Compute the fuel economy for the given trajectory and vehicle model.
        Returns: float (fuel economy in kWh/mi)
        """
        return self.compute_energy_consumption() / (self.get_dist_m() / 1000 * 0.621371)

    def get_overspeeding_duration(self):
        """
        Get the duration of overspeeding by comparing speed with speed limit
        """
        if self.trajectory.speed_limit_mph is None:
            raise ValueError("Speed limit data is not available.")
        
        return sum(1 for i in range(len(self.trajectory.speed_mph)) if ((self.trajectory.speed_mph[i] > self.trajectory.speed_limit_mph[i]) and 
                   (self.trajectory.speed_limit_mph[i] >= 45)))
    
    def get_max_overspeeding(self):
        """
        Get the maximum overspeeding by comparing speed with speed limit
        """
        if self.trajectory.speed_limit_mph is None:
            raise ValueError("Speed limit data is not available.")
        
        return max([0]+[self.trajectory.speed_mph[i] - self.trajectory.speed_limit_mph[i] for i in range(len(self.trajectory.speed_mph)) if 
                        (not pd.isna(self.trajectory.speed_limit_mph[i]) and self.trajectory.speed_limit_mph[i] >= 45)])