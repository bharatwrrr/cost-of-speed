################################################################################

# This file contains PATH variables that need to be set.
# The current setup has 2 drives. The main drive has interpolated data for all days
# but has elevation and speed limit matched data only uptil day 2
# The backup drive has elevation and speed limit matched data for day 3 and day 4. 

# TODO: The drives need to be shuffled to make more sense.

################################################################################

import os

# These are the paths to the drives mounted using sshfs
MAIN_DRIVE_PATH = '/path/to/first/volume'
BACKUP_DRIVE_PATH = '/path/to/second/volume'


def validate_paths(which_data):
    """
    Validate that the configured paths exist.
    """
    assert os.path.exists(MAIN_DRIVE_PATH) or os.path.exists(BACKUP_DRIVE_PATH), f"Both paths ({MAIN_DRIVE_PATH} and {BACKUP_DRIVE_PATH}) do not exist. Config setup might be incorrect. Make sure atleast one of these paths exists."
        
    assert len(os.listdir(MAIN_DRIVE_PATH)) != 0 or (len(os.listdir(BACKUP_DRIVE_PATH)) != 0), f"Both paths ({MAIN_DRIVE_PATH} and {BACKUP_DRIVE_PATH}) are empty. Mounting may not have been successful. Please mount atleast one of the drives again."
    
    if (which_data['day'] == 1) or (which_data['day'] == 2):
        assert len(os.listdir(MAIN_DRIVE_PATH)) != 0, f"data for day {which_data['day']} is found in {MAIN_DRIVE_PATH} but is empty."
    else:
        assert len(os.listdir(BACKUP_DRIVE_PATH)) != 0, f"data for day {which_data['day']} is found in {BACKUP_DRIVE_PATH} but is empty."


ICEV_MODELS_PATH = '/data/vehicles/tuned_vehicles_icev.csv'
BEV_MODELS_PATH = '/data/vehicles/tuned_vehicles_bev.csv'
PHEV_MODELS_PATH = '/data/vehicles/tuned_vehicles_phev.csv'
HEV_MODELS_PATH = '/data/vehicles/tuned_vehicles_hev.csv'

# Paths for individual vehicle models saved as json files
VEHICLE_MODELS_PATH = {
    'icev': '/path/to/vehicle_models/icev/',
    'bev': '/path/to/vehicle_models/bev/',
    'phev': '/path/to/vehicle_models/phev/',
    'hev': '/path/to/vehicle_models/hev/'
}

# Default paths for journey to vehicle model mapping
JOURNEY_TO_VEHICLE_DAY1 = '/path/to/journeys_veh_model_day1.csv'
JOURNEY_TO_VEHICLE_DAY2 = '/path/to/journeys_veh_model_day2.csv'
JOURNEY_TO_VEHICLE_DAY3 = '/path/to/journeys_veh_model_day3.csv'
JOURNEY_TO_VEHICLE_DAY4 = '/path/to/journeys_veh_model_day4.csv'

# Constants for calculations
ACCEL_GRAVITY = 9.81  # m/s^2
KWH_PER_GGE = 33.7  # kWh per gasoline gallon equivalent
