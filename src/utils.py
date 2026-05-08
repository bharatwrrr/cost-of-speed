import pandas as pd
import numpy as np
import json
import os
import geopandas as gpd
from shapely.geometry import Polygon
import requests
import sys
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.affinity import translate

from shapely.errors import TopologicalError
from branca.element import MacroElement
from jinja2 import Template


import matplotlib.pyplot as plt

import pandas as pd
import geohash2
import folium
import branca.colormap as cm

THIS_FILE = os.path.abspath(__file__)
THIS_DIR = os.path.dirname(THIS_FILE)
PLOT_SAVE_PATH = os.path.join(THIS_DIR, '../figures')
ENERGY_DATA_PATH = os.path.join(THIS_DIR, '../data')

def os_patterns(energy_dict):
    ## Collecting overspeeding patterns
    patterns = {}
    for geo in energy_dict.keys():
    #     print(f'Processing geohash {geo}')
        for hour in range(24):
            for jid in energy_dict[geo][str(hour)]:
                if energy_dict[geo][str(hour)][jid]['epa_class'] not in patterns:
                    patterns[energy_dict[geo][str(hour)][jid]['epa_class']] = {}
                    patterns[energy_dict[geo][str(hour)][jid]['epa_class']]['max_os'] = [energy_dict[geo][str(hour)][jid]['max_overspeeding']]
                    patterns[energy_dict[geo][str(hour)][jid]['epa_class']]['time_os'] = [energy_dict[geo][str(hour)][jid]['time_overspeeding']]
                else:
                    patterns[energy_dict[geo][str(hour)][jid]['epa_class']]['max_os'] += [energy_dict[geo][str(hour)][jid]['max_overspeeding']]
                    patterns[energy_dict[geo][str(hour)][jid]['epa_class']]['time_os'] += [energy_dict[geo][str(hour)][jid]['time_overspeeding']]

            print(f'Processed hour {hour}', end='\r')
    for epa_class in patterns:
        patterns[epa_class]['avg_max_os'] = sum(patterns[epa_class]['max_os'])/len(patterns[epa_class]['max_os'])
        patterns[epa_class]['avg_time_os'] = sum(patterns[epa_class]['time_os'])/len(patterns[epa_class]['time_os'])
    return patterns

def get_fuel_saved_data(energy_dict):
    # creating df for choropleth map
    gs_fuel_saved_speed = {}
    gs_fuel_saved_accel = {}
    speed_anomalies = []
    accel_anomalies = []
    accel_normal = []
    for geo in energy_dict:
        count = 0
        if geo not in gs_fuel_saved_speed:
            gs_fuel_saved_speed[geo] = {}
            gs_fuel_saved_accel[geo] = {}
        gs_fuel_saved_speed[geo]['fuel_saving'] = []
        gs_fuel_saved_speed[geo]['dist_mi'] = []
        gs_fuel_saved_speed[geo]['fuel_saving_per_mile'] = []
        gs_fuel_saved_speed[geo]['perc_fuel_saved'] = []
        gs_fuel_saved_speed[geo]['max_perc_fuel_saved'] = 0
        gs_fuel_saved_speed[geo]['total_fuel_used'] = 0
        
        gs_fuel_saved_accel[geo]['fuel_saving'] = []
        gs_fuel_saved_accel[geo]['dist_mi'] = []
        gs_fuel_saved_accel[geo]['fuel_saving_per_mile'] = []
        gs_fuel_saved_accel[geo]['perc_fuel_saved'] = []
        gs_fuel_saved_accel[geo]['max_perc_fuel_saved'] = 0
        gs_fuel_saved_accel[geo]['total_fuel_used'] = 0
        for hour in range(24):
            for jid in energy_dict[geo][str(hour)]:
                fuel_saving_speed = (energy_dict[geo][str(hour)][jid]['og']['fuel_gal'] - 
                                    energy_dict[geo][str(hour)][jid]['adj']['fuel_gal'])
                fuel_saving_accel = (energy_dict[geo][str(hour)][jid]['og']['fuel_gal'] 
                                    - energy_dict[geo][str(hour)][jid]['final']['fuel_gal'])
                fuel_saving_speed_perc = (energy_dict[geo][str(hour)][jid]['og']['fuel_gal'] - 
                                    energy_dict[geo][str(hour)][jid]['adj']['fuel_gal'])/energy_dict[geo][str(hour)][jid]['og']['fuel_gal']*100
                fuel_saving_accel_perc = (energy_dict[geo][str(hour)][jid]['og']['fuel_gal'] 
                                    - energy_dict[geo][str(hour)][jid]['final']['fuel_gal'])/energy_dict[geo][str(hour)][jid]['og']['fuel_gal']*100
                if (fuel_saving_speed>100):
                    speed_anomalies.append((geo, hour, jid, fuel_saving_speed))
                    continue

                if (fuel_saving_accel<-100) or (fuel_saving_accel>30):
                    accel_anomalies.append((geo, hour, jid, fuel_saving_accel))
                    continue
                else:
                    accel_normal.append((geo, hour, jid, fuel_saving_accel))

                count+=1
                gs_fuel_saved_speed[geo]['total_fuel_used'] += energy_dict[geo][str(hour)][jid]['og']['fuel_gal']
                gs_fuel_saved_speed[geo]['fuel_saving'] += [fuel_saving_speed]
                gs_fuel_saved_speed[geo]['dist_mi'] += [energy_dict[geo][str(hour)][jid]['og']['dist_mi']]
                gs_fuel_saved_speed[geo]['perc_fuel_saved'] += [fuel_saving_speed_perc]
                gs_fuel_saved_speed[geo]['max_perc_fuel_saved'] = max(fuel_saving_speed_perc, gs_fuel_saved_speed[geo]['max_perc_fuel_saved'])

                gs_fuel_saved_accel[geo]['total_fuel_used'] += energy_dict[geo][str(hour)][jid]['og']['fuel_gal']
                gs_fuel_saved_accel[geo]['fuel_saving'] += [fuel_saving_accel]
                gs_fuel_saved_accel[geo]['dist_mi'] += [energy_dict[geo][str(hour)][jid]['og']['dist_mi']]
                gs_fuel_saved_accel[geo]['perc_fuel_saved'] += [fuel_saving_accel_perc]
                gs_fuel_saved_accel[geo]['max_perc_fuel_saved'] = max(fuel_saving_accel_perc, gs_fuel_saved_accel[geo]['max_perc_fuel_saved'])
                gs_fuel_saved_accel[geo]['fuel_saving_per_mile'] += []
                
   
        gs_fuel_saved_speed[geo]['perc_fuel_saved'] = np.mean(gs_fuel_saved_speed[geo]['perc_fuel_saved'])
        
        gs_fuel_saved_accel[geo]['perc_fuel_saved'] = np.mean(gs_fuel_saved_accel[geo]['perc_fuel_saved'])
        gs_fuel_saved_speed[geo]['fuel_saving'] = sum(gs_fuel_saved_speed[geo]['fuel_saving'])
        gs_fuel_saved_accel[geo]['fuel_saving'] = sum(gs_fuel_saved_accel[geo]['fuel_saving'])
        gs_fuel_saved_speed[geo]['fuel_saved_per_mile'] = gs_fuel_saved_speed[geo]['fuel_saving']/sum(gs_fuel_saved_speed[geo]['dist_mi'])

        gs_fuel_saved_accel[geo]['fuel_saved_per_mile'] = gs_fuel_saved_accel[geo]['fuel_saving']/sum(gs_fuel_saved_accel[geo]['dist_mi'])
        gs_fuel_saved_speed[geo]['trip_count'] = count
        gs_fuel_saved_accel[geo]['trip_count'] = count
        print(f'processed for geo {geo}', end='\r')

    return gs_fuel_saved_speed, gs_fuel_saved_accel, speed_anomalies, accel_anomalies, accel_normal
    
def get_cost_saved_data(gs_fuel_saved_speed, gs_fuel_saved_accel,
                        gallons_per_year,
                        usd_per_gallon):

    gallons_per_day = gallons_per_year/365

    gs_speed_df = pd.DataFrame.from_dict(gs_fuel_saved_speed, orient='index')
    gs_speed_df['perc_fuel_saved'] = gs_speed_df['perc_fuel_saved'].fillna(0)
    gs_speed_df['fuel_use_frac'] = gs_speed_df['total_fuel_used']/sum(gs_speed_df['total_fuel_used'])

    gs_speed_df = gs_speed_df[['perc_fuel_saved', 'fuel_saved_per_mile', 'fuel_saving', 'fuel_use_frac']]

    gs_speed_df['total_fuel_saved'] = gs_speed_df['perc_fuel_saved']/100*gs_speed_df['fuel_use_frac']*gallons_per_day
    gs_speed_df['total_cost_saved'] = gs_speed_df['total_fuel_saved']*usd_per_gallon

    gs_accel_df = pd.DataFrame.from_dict(gs_fuel_saved_accel, orient='index')
    gs_accel_df['perc_fuel_saved'] = gs_accel_df['perc_fuel_saved'].fillna(0)
    gs_accel_df['fuel_use_frac'] = gs_accel_df['total_fuel_used']/sum(gs_accel_df['total_fuel_used'])
    gs_accel_df = gs_accel_df[['perc_fuel_saved', 'fuel_saved_per_mile', 'fuel_saving', 'fuel_use_frac']]# gs_accel_df['fuel_saved_per_mile'] = sum(gs_accel_df['fuel_saving'])/sum(gs_accel_df['dist_mi'])
    gs_accel_df['total_fuel_saved'] = gs_accel_df['perc_fuel_saved']/100*gs_accel_df['fuel_use_frac']*gallons_per_day
    gs_accel_df['total_cost_saved'] = gs_accel_df['total_fuel_saved']*usd_per_gallon
    
    return gs_speed_df, gs_accel_df


def get_state_max_sl(geo):
    states = gpd.read_file("https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json")
    state = geohash_to_state(geo, states)
    sl_df = pd.read_csv(f'{ENERGY_DATA_PATH}/max_sl.csv', index_col=0)
    if state==None:
        return 70.0
    max_sl = sl_df.loc[state, 'max_sl_mph']
    return max_sl

def fetch_monthly_cons(state, day):
    if day==1:
        month = 'January'
    elif day==2:
        month = 'April'
    elif day==3:
        month = 'August'
    elif day==4:
        month = 'October'

    fuel_cons = pd.read_csv(f'{ENERGY_DATA_PATH}/gvd_usa.csv')
    month_row = fuel_cons[(fuel_cons['STATE'] == state.lower()) & (fuel_cons['YEAR'] == 2024) & (fuel_cons['MONTH'] == month)]
    if month_row.empty:
        print(f"No data found for {state} in {month} 2024")
        return None
    else:
        return month_row.iloc[0]['VOL GASOLINE']

def geohash_to_state(gh3: str, states: gpd.GeoDataFrame) -> str:
    # 1) Decode center +/- error
    try:
        lat, lon, lat_err, lon_err = geohash2.decode_exactly(gh3)
    except Exception as e:
        raise ValueError(f"Invalid geohash '{gh3}': {e}")

    # 2) Center-point check
    center = Point(lon, lat)
    idxs = list(states.geometry.sindex.intersection((lon, lat, lon, lat)))
    for i in idxs:
        if states.geometry.iloc[i].contains(center):
            return states.iloc[i]["name"].lower()

    # 3) Build and clean the cell polygon
    min_lat, max_lat = lat - lat_err, lat + lat_err
    min_lon, max_lon = lon - lon_err, lon + lon_err
    cell_poly = Polygon([
        (min_lon, min_lat),
        (max_lon, min_lat),
        (max_lon, max_lat),
        (min_lon, max_lat),
        (min_lon, min_lat),
    ]).buffer(0)  # clean it

    # 4) Find all states whose bbox overlaps
    bbox_idxs = list(states.geometry.sindex.intersection(cell_poly.bounds))
    if not bbox_idxs:
        raise ValueError(f"Geohash '{gh3}' overlaps no state bboxes.")

    # 5) Try intersections, track max area
    best_state = None
    best_area = 0.0
    for i in bbox_idxs:
        poly = states.geometry.iloc[i]
        # clean the state polygon if it’s invalid
        if not poly.is_valid:
            poly = poly.buffer(0)
        try:
            inter = poly.intersection(cell_poly)
        except TopologicalError:
            # skip any candidate that still errors out
            continue
        if not inter.is_empty:
            area = inter.area
            if area > best_area:
                best_area = area
                best_state = states.iloc[i]

    if best_state is None:
        return None
    return best_state["name"].lower()

def fetch_monthly_gas_price(state, day):
    df_prices = pd.read_csv(f'{ENERGY_DATA_PATH}/prices_usa.csv')
    with open(f'{ENERGY_DATA_PATH}/state_to_padd.json', 'r') as f:
        state_to_padd = json.load(f)
    if day==1:
        period = '2024-01-01'
    elif day==2:
        period = '2024-04-01'
    elif day==3:
        period = '2024-08-01'
    elif day==4:
        period = '2024-10-01'

    if state.lower() not in df_prices['area-name'].unique():
        area_name = state_to_padd[state.lower()]
    else:
        area_name = state.lower()
    return df_prices.loc[(df_prices['area-name']==area_name.upper()) & 
                  (df_prices['product-name']=='Conventional Regular Gasoline') & 
                  (df_prices['period']==period)]['value'].item()






def get_counties(state):
    # Load MN counties
    counties_url = 'https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json'
    county_geojson = requests.get(counties_url).json()
    counties = {
        'type': 'FeatureCollection',
        'features': [f for f in county_geojson['features'] if f['properties']['STATE'] == state]
    }
    return counties

def plot_fuel_saved(gs_df, day, col='total_fuel_saved', vmax=71000):

    # SETTINGS
    column_to_plot = col
    # DAY = '2025-03-28'  # or whatever you're using
    colors = ['darkblue','blue', 'green', 'yellow', 'orange', 'red']
    # quantile_levels = [0, 0.4, 0.8, 0.9, 0.95, 0.98, 1.0]
    quantile_levels = [0, 0.2, 0.4, 0.6, 0.8, 0.99, 1.0]

    # FILTER DATA
    plot_df = gs_df.dropna(subset=[column_to_plot]).copy()

    # Convert geohash to polygon
    def geohash_to_polygon(gh):
        lat_center, lon_center, lat_err, lon_err = geohash2.decode_exactly(gh)
        lat_min = lat_center - lat_err
        lat_max = lat_center + lat_err
        lon_min = lon_center - lon_err
        lon_max = lon_center + lon_err
        return Polygon([
            (lon_min, lat_min),
            (lon_max, lat_min),
            (lon_max, lat_max),
            (lon_min, lat_max)
        ])

    # Load MN state boundary
    states = gpd.read_file("https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json")
    counties_url = 'https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json'
    county_geojson = requests.get(counties_url).json()

    m = folium.Map(location=[46.5, -94.5], zoom_start=6, tiles='cartodbpositron')

    # Define bins based on current data + editable upper bound
    # bins = [0, 100, 500, 1000, 2500, 5000, 10000, 25000, 50000, vmax]
    bins = np.linspace(0, vmax, 10).tolist()
    print(bins)
    # Corresponding perceptual colors from blue to red
    # colors = [
    #     '#313695', '#4575b4', '#74add1', '#abd9e9',
    #     '#ffffbf', '#fdae61', '#f46d43', '#d73027',
    #     '#a50026'
    # ]
    colors = [
        '#084594',  # deep blue
        '#2b8cbe',  # medium blue
        '#41ab5d',  # green
        '#addd8e',  # light green
        '#ffffbf',  # yellow
        '#fdae61',  # orange
        '#f46d43',  # orange-red
        '#d73027',  # red
        '#990000'   # dark red
    ]
    
    # Create the StepColormap
    cmap = cm.StepColormap(
        colors=colors,
        index=bins,
        vmin=bins[0],
        vmax=bins[-1],
        caption="Total Fuel Saved (gallons)"
    )

    # Add it to the map
    cmap.add_to(m)

    # # Add a default color map
    # cmap = cm.linear.YlOrRd_09.scale(0, vmax)
    # cmap.add_to(m)
    
    for state in states['name']:
        if state == 'Puerto Rico':
            continue
        state_gdf = states[states['name'] == state].to_crs("EPSG:4326")
        state_id  = state_gdf['id'].item()
        counties = {
            'type': 'FeatureCollection',
            'features': [f for f in county_geojson['features'] if f['properties']['STATE'] == state_id]
        }


        # Convert geohashes to GeoDataFrame
        geometries = [geohash_to_polygon(gh) for gh in plot_df.index]
        gh_gdf = gpd.GeoDataFrame(plot_df, geometry=geometries, crs="EPSG:4326")
        # Clip geohash polygons to Minnesota
        gh_clipped = gpd.overlay(gh_gdf, state_gdf, how='intersection')

        
        # Then use it like this in your style function
        for _, row in gh_clipped.iterrows():
            val = row[column_to_plot]
            folium.GeoJson(
                data=row['geometry'].__geo_interface__,
                style_function=lambda x, val=val: {
                    'fillColor': cmap(val),
                    'color': None,
                    'fillOpacity': 0.8
                },
                tooltip=f"Value: {val:,.2f} gallons"
            ).add_to(m)


        # Add MN outline
        folium.GeoJson(
            state_gdf.__geo_interface__,
            style_function=lambda x: {'fill': False, 'color': 'black', 'weight': 1}
        ).add_to(m)

        # # Add MN counties
        # folium.GeoJson(
        #     counties,
        #     name=f'{state} Counties',
        #     style_function=lambda feature: {
        #         'fillColor': 'none',
        #         'color': 'black',
        #         'weight': 0.5,
        #     },
        #     tooltip=folium.GeoJsonTooltip(
        #         fields=['NAME'],
        #         aliases=['County:'],
        #         style=(
        #             "background-color: white; "
        #             "color: black; "
        #             "font-size: 14px; "
        #             "font-weight: bold;"
        #         )
        #     )
        # ).add_to(m)

    # Save map
    # m.save(f"{PLOT_SAVE_PATH}/mn_only_fuel_cost_{day}.html")
    m.save(f"{PLOT_SAVE_PATH}/{col}_day{day}.html")
    m

def plot_fuel_saved_for_state(gs_df, day, col='total_fuel_saved', suffix='', vmax=71000, state_name='Minnesota'):
    """
    Plot fuel saved data for a specific state with geohashes clipped to state boundaries.
    
    Args:
        gs_df: DataFrame with geohash indices and fuel data
        day: Day number for the filename
        col: Column name to plot
        suffix: Suffix for the filename
        vmax: Maximum value for color scale
        state_name: Name of the state to plot (default: 'Minnesota')
    """
    # SETTINGS
    column_to_plot = col
    
    # FILTER DATA
    plot_df = gs_df.dropna(subset=[column_to_plot]).copy()

    # Convert geohash to polygon
    def geohash_to_polygon(gh):
        lat_center, lon_center, lat_err, lon_err = geohash2.decode_exactly(gh)
        lat_min = lat_center - lat_err
        lat_max = lat_center + lat_err
        lon_min = lon_center - lon_err
        lon_max = lon_center + lon_err
        return Polygon([
            (lon_min, lat_min),
            (lon_max, lat_min),
            (lon_max, lat_max),
            (lon_min, lat_max)
        ])

    # Load state boundaries
    states = gpd.read_file("https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json")
    
    # Get the specific state
    target_state = states[states['name'] == state_name].to_crs("EPSG:4326")
    if target_state.empty:
        raise ValueError(f"State '{state_name}' not found in the dataset")
    
    # Get state bounds for map centering
    bounds = target_state.bounds
    center_lat = (bounds.miny.iloc[0] + bounds.maxy.iloc[0]) / 2
    center_lon = (bounds.minx.iloc[0] + bounds.maxx.iloc[0]) / 2
    
    # Filter geohashes to only include those that intersect with the target state
    geometries = [geohash_to_polygon(gh) for gh in plot_df.index]
    gh_gdf = gpd.GeoDataFrame(plot_df, geometry=geometries, crs="EPSG:4326")
    
    # Find geohashes that intersect with the target state
    intersecting_mask = gh_gdf.geometry.intersects(target_state.geometry.iloc[0])
    gh_gdf_filtered = gh_gdf[intersecting_mask].copy()
    
    if gh_gdf_filtered.empty:
        print(f"No geohashes found intersecting with {state_name}")
        return None
    
    # Clip geohash polygons to state boundaries
    gh_clipped = gpd.overlay(gh_gdf_filtered, target_state, how='intersection')
    
    # Create map centered on the target state
    m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles=None)
    
    # Define bins and colors
    bins = np.linspace(0, vmax, 10).tolist()
    print(f"Color bins for {state_name}: {bins}")
    
    colors = [
        '#084594',  # deep blue
        '#2b8cbe',  # medium blue
        '#41ab5d',  # green
        '#addd8e',  # light green
        '#ffffbf',  # yellow
        '#fdae61',  # orange
        '#f46d43',  # orange-red
        '#d73027',  # red
        '#990000'   # dark red
    ]
    
    # Create the StepColormap - Update caption based on column type
    if 'energy' in col.lower():
        caption_text = f"Total Energy Saved (kWh) - {state_name}"
        tooltip_unit = "kWh"
    elif 'cost' in col.lower():
        caption_text = f"Total Cost Saved ($) - {state_name}"
        tooltip_unit = "$"
    else:
        caption_text = f"Total Fuel Saved (gallons) - {state_name}"
        tooltip_unit = "gallons"
    
    cmap = cm.StepColormap(
        colors=colors,
        index=bins,
        vmin=bins[0],
        vmax=bins[-1],
        caption=caption_text
    )
    cmap.add_to(m)
    
    # Add white background
    white_background = MacroElement()
    white_background._template = Template("""
    {% macro html(this, kwargs) %}
    <style>
        .folium-map {
            background: white !important;
        }
    </style>
    {% endmacro %}
    """)
    m.get_root().add_child(white_background)
    
    # Add clipped geohash polygons to map
    for _, row in gh_clipped.iterrows():
        val = row[column_to_plot]
        folium.GeoJson(
            data=row['geometry'].__geo_interface__,
            style_function=lambda x, val=val: {
                'fillColor': cmap(val),
                'color': None,
                'fillOpacity': 0.8
            },
            tooltip=f"Value: {val:,.2f} {tooltip_unit}"
        ).add_to(m)

    # Add state outline
    folium.GeoJson(
        target_state.__geo_interface__,
        style_function=lambda x: {'fill': False, 'color': 'black', 'weight': 2}
    ).add_to(m)

    # Save map
    state_abbrev = state_name.lower().replace(' ', '_')
    filename = f"{PLOT_SAVE_PATH}/{col}_{state_abbrev}_day{day}_{suffix}.html"
    m.save(filename)
    print(f"Map saved to: {filename}")
    


def plot_energy_saved_for_state(gs_df, day, col='total_energy_saved', suffix='', vmax=71000, state_name='Minnesota'):
    """
    Plot energy saved data for a specific state with geohashes clipped to state boundaries.
    
    Args:
        gs_df: DataFrame with geohash indices and energy data
        day: Day number for the filename
        col: Column name to plot
        suffix: Suffix for the filename
        vmax: Maximum value for color scale
        state_name: Name of the state to plot (default: 'Minnesota')
    """ 
    # SETTINGS
    column_to_plot = col
    
    # FILTER DATA
    plot_df = gs_df.dropna(subset=[column_to_plot]).copy()

    # Convert geohash to polygon
    def geohash_to_polygon(gh):
        lat_center, lon_center, lat_err, lon_err = geohash2.decode_exactly(gh)
        lat_min = lat_center - lat_err
        lat_max = lat_center + lat_err
        lon_min = lon_center - lon_err
        lon_max = lon_center + lon_err
        return Polygon([
            (lon_min, lat_min),
            (lon_max, lat_min),
            (lon_max, lat_max),
            (lon_min, lat_max)
        ])

    # Load state boundaries
    states = gpd.read_file("https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json")
    
    # Get the specific state
    target_state = states[states['name'] == state_name].to_crs("EPSG:4326")
    if target_state.empty:
        raise ValueError(f"State '{state_name}' not found in the dataset")
    
    # Get state bounds for map centering
    bounds = target_state.bounds
    center_lat = (bounds.miny.iloc[0] + bounds.maxy.iloc[0]) / 2
    center_lon = (bounds.minx.iloc[0] + bounds.maxx.iloc[0]) / 2
    
    # Filter geohashes to only include those that intersect with the target state
    geometries = [geohash_to_polygon(gh) for gh in plot_df.index]
    gh_gdf = gpd.GeoDataFrame(plot_df, geometry=geometries, crs="EPSG:4326")
    
    # Find geohashes that intersect with the target state
    intersecting_mask = gh_gdf.geometry.intersects(target_state.geometry.iloc[0])
    gh_gdf_filtered = gh_gdf[intersecting_mask].copy()
    
    if gh_gdf_filtered.empty:
        print(f"No geohashes found intersecting with {state_name}")
        return None
    
    # Clip geohash polygons to state boundaries
    gh_clipped = gpd.overlay(gh_gdf_filtered, target_state, how='intersection')
    
    # Create map centered on the target state
    m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles=None)
    
    # Define bins and colors
    bins = np.linspace(0, vmax, 10).tolist()
    print(f"Color bins for {state_name}: {bins}")
    
    colors = [
        '#084594',  # deep blue
        '#2b8cbe',  # medium blue
        '#41ab5d',  # green
        '#addd8e',  # light green
        '#ffffbf',  # yellow
        '#fdae61',  # orange
        '#f46d43',  # orange-red
        '#d73027',  # red
        '#990000'   # dark red
    ]
    
    # Create the StepColormap - Update caption based on column type
    if 'energy' in col.lower():
        caption_text = f"Total Energy Saved (kWh) - {state_name}"
        tooltip_unit = "kWh"
    else:
        caption_text = f"Total Fuel Saved (gallons) - {state_name}"
        tooltip_unit = "gallons"
    
    cmap = cm.StepColormap(
        colors=colors,
        index=bins,
        vmin=bins[0],
        vmax=bins[-1],
        caption=caption_text
    )
    cmap.add_to(m)
    
    # Add white background
    white_background = MacroElement()
    white_background._template = Template("""
    {% macro html(this, kwargs) %}
    <style>
        .folium-map {
            background: white !important;
        }
    </style>
    {% endmacro %}
    """)
    m.get_root().add_child(white_background)
    
    # Add clipped geohash polygons to map
    for _, row in gh_clipped.iterrows():
        val = row[column_to_plot]
        folium.GeoJson(
            data=row['geometry'].__geo_interface__,
            style_function=lambda x, val=val: {
                'fillColor': cmap(val),
                'color': None,
                'fillOpacity': 0.8
            },
            tooltip=f"Value: {val:,.2f} {tooltip_unit}"
        ).add_to(m)

    # Add state outline
    folium.GeoJson(
        target_state.__geo_interface__,
        style_function=lambda x: {'fill': False, 'color': 'black', 'weight': 2}
    ).add_to(m)

    # Save map
    state_abbrev = state_name.lower().replace(' ', '_')
    filename = f"{PLOT_SAVE_PATH}/{col}_{state_abbrev}_day{day}_{suffix}.html"
    m.save(filename)
    print(f"Map saved to: {filename}")
    
    
    
def plot_fuel_saved_png_v2(
    gs_df,
    day,
    col='total_fuel_saved',
    title='Total Fuel Saved (gallons)',
    suffix='',
    vmax=71000,
    vmin=0,
    cmap='YlGnBu',
    hawaii_offset=(42, 8)
):
    """
    Plot fuel-saved over the continental US + Hawaii (shifted),
    and save as a PNG.

    Parameters
    ----------
    gs_df : pandas.DataFrame
        Indexed by geohash strings, with a numeric column `col`.
    day : str
        Identifier for the day (used in filename).
    col : str
        Column name in gs_df to color by.
    suffix : str
        Optional filename suffix.
    vmax : float
        Maximum value for the color scale.
    cmap : str or Colormap
        Matplotlib colormap name or instance (e.g. 'YlGnBu', 'viridis', etc.).
    hawaii_offset : tuple(float, float)
        (xoff, yoff) in degrees to shift Hawaii eastward/northward.
    """

    # --- 1) build geohash-polygons GeoDataFrame ---
    def geohash_to_polygon(gh):
        lat_c, lon_c, lat_e, lon_e = geohash2.decode_exactly(gh)
        return Polygon([
            (lon_c - lon_e, lat_c - lat_e),
            (lon_c + lon_e, lat_c - lat_e),
            (lon_c + lon_e, lat_c + lat_e),
            (lon_c - lon_e, lat_c + lat_e),
        ])

    geometries = [geohash_to_polygon(gh) for gh in gs_df.index]
    gh_gdf = gpd.GeoDataFrame(gs_df.copy(), geometry=geometries, crs="EPSG:4326")

    # --- 2) load US states, drop AK & PR ---
    states = gpd.read_file(
        "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
    ).to_crs("EPSG:4326")
    states = states[~states['name'].isin(['Alaska', 'Puerto Rico'])]

    # separate continental vs. HI
    continental = states[states['name'] != 'Hawaii'].copy()
    hawaii     = states[ states['name'] == 'Hawaii'].copy()

    # --- 3) clip geohash grid to each region ---
    usa_union = unary_union(continental.geometry)
    gh_cont  = gh_gdf.clip(usa_union)
    gh_hi    = gh_gdf.clip(hawaii.unary_union)

    # --- 4) translate HI grid & shape ---
    xoff, yoff = hawaii_offset
    gh_hi['geometry']   = gh_hi['geometry'].apply(lambda g: translate(g, xoff=xoff, yoff=yoff))
    hawaii['geometry']  = hawaii['geometry'].apply(lambda g: translate(g, xoff=xoff, yoff=yoff))

    # --- 5) combine back ---
    gh_plot    = pd.concat([gh_cont, gh_hi], ignore_index=True)
    states_plot = pd.concat([continental, hawaii], ignore_index=True)

    # --- 6) plot with Matplotlib ---
    fig, ax = plt.subplots(figsize=(12, 8))
    # state outlines
    states_plot.boundary.plot(ax=ax, linewidth=0.8, edgecolor='black')
    # choropleth, no cell borders
    gh_plot.plot(
        column=col,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        edgecolor='none',
        legend=True,
        legend_kwds={
            'label': title,
            'shrink': 0.5,
            'pad': 0,
            'ticks': np.arange(vmin, vmax + 10, 10)
        }
    )
    
    # get the colorbar axis
    cbar_ax = fig.axes[-1]

    # force ticks every 10
    ticks = np.arange(vmin, vmax + 10, 10)
    cbar_ax.set_yticks(ticks)

    ax.set_axis_off()
    fig.set_facecolor('white')
    fig.subplots_adjust(right=0.95)  # default is ~0.9–0.92; increase to push map rightwards

    plt.tight_layout()

    # --- 7) save PNG ---
    os.makedirs(PLOT_SAVE_PATH, exist_ok=True)
    fname = f"{col}_day{day}" + (f"_{suffix}" if suffix else "") + ".png"
    out_path = os.path.join(PLOT_SAVE_PATH, fname)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved map to {out_path}")
    

def plot_energy_saved_state_png_new(
    gs_df,
    day,
    col='avg_energy_saved_per_100_miles',
    state_name='California',
    title='Average Energy Saved per 100 mile (kWh)',
    suffix='',
    vmax=4.5,
    cmap='viridis',
    plot_outline=True,
    figsize=(6, 8),
    trip_thresh=None,
    thresh_col='num_trips'
):
    """
    Make a standalone PNG choropleth for one U.S. state.
    """

    # 1 ── build geohash polygons
    def gh_to_poly(gh):
        lat_c, lon_c, lat_e, lon_e = geohash2.decode_exactly(gh)
        return Polygon([
            (lon_c - lon_e, lat_c - lat_e),
            (lon_c + lon_e, lat_c - lat_e),
            (lon_c + lon_e, lat_c + lat_e),
            (lon_c - lon_e, lat_c + lat_e),
        ])

    geoms = [gh_to_poly(gh) for gh in gs_df.index]
    gh_gdf = gpd.GeoDataFrame(gs_df.copy(), geometry=geoms, crs='EPSG:4326')

    # 2 ── load state polygon
    states = gpd.read_file(
        "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
    ).to_crs('EPSG:4326')

    target = states[states['name'].str.lower() == state_name.lower()]
    if target.empty:
        raise ValueError(f"State '{state_name}' not found in GeoJSON file.")

    state_poly = unary_union(target.geometry)

    # 3 ── clip geohash grid
    gh_state = gh_gdf.clip(state_poly)
    if gh_state.empty:
        print(f'No geohashes intersecting {state_name}; nothing plotted.')
        return

    # 4 ── apply trip threshold mask (NEW)
    if trip_thresh is not None:
        low_trip_mask = gh_state[thresh_col] < trip_thresh

        gh_low = gh_state[low_trip_mask].copy()
        gh_plot = gh_state[~low_trip_mask].copy()
    else:
        gh_plot = gh_state
        gh_low = None

    # 5 ── plot
    fig, ax = plt.subplots(figsize=figsize)

    # main choropleth (valid cells)
    gh_plot.plot(
        column=col,
        cmap=cmap,
        vmin=0,
        vmax=vmax,
        ax=ax,
        edgecolor='none',
        legend=True,
        legend_kwds={
            'label': title.format(state=state_name),
            'shrink': 0.7,
            'pad': 0.02
        }
    )

    # grey + hatched cells for low-trip geohashes (NEW)
    if gh_low is not None and not gh_low.empty:
        gh_low.plot(
            ax=ax,
            color='lightgrey',
            edgecolor='grey',
            hatch='///',
            linewidth=0.2
        )

    if plot_outline:
        target.boundary.plot(ax=ax, linewidth=1.2, edgecolor='black')

    ax.set_axis_off()
    fig.set_facecolor('white')
    plt.tight_layout()

    # 6 ── save
    os.makedirs(PLOT_SAVE_PATH, exist_ok=True)
    state_tag = state_name.lower().replace(' ', '_')
    fname = f"{col}_{state_tag}_day{day}"
    if suffix:
        fname += f"_{suffix}"
    out_path = os.path.join(PLOT_SAVE_PATH, fname + ".png")

    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved state energy map to {out_path}")
    
def plot_energy_saved_state_png(
    gs_df,
    day,
    col='avg_energy_saved_per_100_miles',              # <- default column
    state_name='California',
    title='Average Energy Saved per 100 mile (kWh)',
    suffix='',
    vmax=4.5,
    cmap='viridis',
    plot_outline=True,
    figsize=(6, 8)
):
    """
    Make a standalone PNG choropleth for one U.S. state.

    Parameters
    ----------
    gs_df : pandas.DataFrame   (index = geohash)
        Must contain a numeric column *col*.
    day : str or int
        Identifier used in the output filename.
    col : str
        Column in *gs_df* to visualise (default 'total_energy_saved').
    state_name : str
        Full state name as it appears in the US-states GeoJSON.
    title : str
        Format string for colour-bar label.  Receives `{state}`.
    suffix : str
        Optional filename suffix appended to the PNG name.
    vmax : float
        Upper bound for the colour scale.
    cmap : str or matplotlib.colors.Colormap
        Colour map to use (e.g. 'viridis', 'YlGnBu').
    plot_outline : bool
        Draw state boundary on top of the fill.
    figsize : tuple
        (width, height) in inches.
    """

    # 1 ── build geohash polygons
    def gh_to_poly(gh):
        lat_c, lon_c, lat_e, lon_e = geohash2.decode_exactly(gh)
        return Polygon([
            (lon_c - lon_e, lat_c - lat_e),
            (lon_c + lon_e, lat_c - lat_e),
            (lon_c + lon_e, lat_c + lat_e),
            (lon_c - lon_e, lat_c + lat_e),
        ])

    geoms = [gh_to_poly(gh) for gh in gs_df.index]
    gh_gdf = gpd.GeoDataFrame(gs_df.copy(), geometry=geoms, crs='EPSG:4326')

    # 2 ── load state polygon
    states = gpd.read_file(
        "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
    ).to_crs('EPSG:4326')

    target = states[states['name'].str.lower() == state_name.lower()]
    if target.empty:
        raise ValueError(f"State '{state_name}' not found in GeoJSON file.")

    state_poly = unary_union(target.geometry)

    # 3 ── clip geohash grid
    gh_state = gh_gdf.clip(state_poly)
    if gh_state.empty:
        print(f'No geohashes intersecting {state_name}; nothing plotted.')
        return

    # 4 ── plot
    fig, ax = plt.subplots(figsize=figsize)
    gh_state.plot(
        column=col,
        cmap=cmap,
        vmin=0,
        vmax=vmax,
        ax=ax,
        edgecolor='none',
        legend=True,
        legend_kwds={
            'label': title.format(state=state_name),
            'shrink': 0.7,
            'pad': 0.02
        }
    )

    if plot_outline:
        target.boundary.plot(ax=ax, linewidth=1.2, edgecolor='black')

    ax.set_axis_off()
    fig.set_facecolor('white')
    plt.tight_layout()

    # 5 ── save
    os.makedirs(PLOT_SAVE_PATH, exist_ok=True)
    state_tag = state_name.lower().replace(' ', '_')
    fname = f"{col}_{state_tag}_day{day}"
    if suffix:
        fname += f"_{suffix}"
    out_path = os.path.join(PLOT_SAVE_PATH, fname + ".png")

    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved state energy map to {out_path}")
    
    

    
def plot_state_speed_png(
    max_sl_df,
    day,
    col='max_sl_mph',
    suffix='',
    colors=None,
    hawaii_offset=(42, 8),
    PLOT_SAVE_PATH='./../figures/'
):
    """
    Create a static PNG choropleth of US states colored by `col`, shifting Hawaii in-set.

    Parameters
    ----------
    max_sl_df : pandas.DataFrame
        Must have lowercase state names as its index and a numeric column `col`.
    day : str
        Day identifier, used in the output filename.
    col : str
        Column in max_sl_df to map (default 'max_sl_mph').
    suffix : str
        Optional extra text for the filename.
    colors : list of str
        Hex or named colors for each category (default five‐color ramp).
    hawaii_offset : tuple(float, float)
        (xoff, yoff) degrees to shift Hawaii toward the continental U.S.
    PLOT_SAVE_PATH : str
        Directory where the PNG will be saved.
    """
    # 1) Prepare bins & colormap
    if colors is None:
        colors = ['#ffea00', '#fed976', '#fc8d59', '#d7301f', '#99000d']
    # unique values, sorted, with a tiny extra bin to close the last
    bins = sorted(max_sl_df[col].dropna().unique().tolist())
    bins.append(bins[-1] + 1e-6)
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bins, len(colors))

    # 2) Load US states outline & index by lowercase name
    states = gpd.read_file(
        "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
    ).to_crs("EPSG:4326")
    states['state'] = states['name'].str.lower()
    states = states.set_index('state')
    # drop AK & PR
    states = states.drop(['alaska','puerto rico'], errors='ignore')

    # 3) Join speed‐limit data
    plot_gdf = states.join(max_sl_df[[col]], how='left')
    # drop states with no data
    plot_gdf = plot_gdf.dropna(subset=[col])

    # 4) Split continental vs HI
    cont = plot_gdf.loc[plot_gdf.index!='hawaii'].copy()
    hi   = plot_gdf.loc[plot_gdf.index=='hawaii'].copy()
    # shift Hawaii
    hi['geometry'] = hi['geometry'].apply(
        lambda geom: translate(geom, xoff=hawaii_offset[0], yoff=hawaii_offset[1])
    )

    # 5) Combine back
    final_gdf = pd.concat([cont, hi], ignore_index=True)
    final_gdf = gpd.GeoDataFrame(final_gdf, geometry='geometry', crs="EPSG:4326")

    # 6) Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    final_gdf.plot(
        column=col,
        cmap=cmap,
        norm=norm,
        ax=ax,
        edgecolor='black',
        linewidth=1
    )
    ax.set_axis_off()
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    
     # --- Build a categorical legend ---
    unique_vals = sorted(max_sl_df[col].dropna().unique().tolist())
    # map each value to its patch color
    legend_handles = []
    for val in unique_vals:
        color = cmap(norm(val))
        label = f"{int(val)} mph"
        patch = mpatches.Patch(facecolor=color, edgecolor='black', label=label)
        legend_handles.append(patch)
    
    # 1) Create the legend, nudged left and up via bbox_to_anchor
    leg = ax.legend(
        handles=legend_handles,
        title='Max Speed Limit',
        loc='lower right',
        bbox_to_anchor=(0.96, 0.06),  # x < 1 moves left, y > 0 moves up
        frameon=False,
        fontsize='large',
        title_fontsize='large',
        ncol=1,
        handlelength=1.5,
        handleheight=1.0,
        columnspacing=0.6
    )

    # 2) Bold the title
    leg.get_title().set_fontweight('bold')


    # 7) Save PNG
    os.makedirs(PLOT_SAVE_PATH, exist_ok=True)
    fname = f"state_map_{col}_day{day}"
    if suffix:
        fname += f"_{suffix}"
    out_path = os.path.join(PLOT_SAVE_PATH, f"{fname}.png")
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved map to {out_path}")
    
    

    
    
def geohash_to_polygon(gh: str) -> Polygon:
    """
    Convert a geohash to a polygon representing its bounding box.
    """
    lat, lon, lat_err, lon_err = geohash2.decode_exactly(gh)
    min_lat, max_lat = lat - lat_err, lat + lat_err
    min_lon, max_lon = lon - lon_err, lon + lon_err
    
    poly = Polygon([
        (min_lon, min_lat),
        (max_lon, min_lat),
        (max_lon, max_lat),
        (min_lon, max_lat),
        (min_lon, min_lat),
    ])
    
    # Clean the polygon if needed
    if not poly.is_valid:
        poly = poly.buffer(0)
    
    return poly


def distribute_geohash_to_states(gh: str, states: gpd.GeoDataFrame, 
                                 total_dist_mi: float, 
                                 total_fuel_used: float,
                                 fuel_saved_metrics: dict) -> pd.DataFrame:
    """
    Distribute a single geohash's metrics across states based on spatial overlap.
    
    Parameters:
    -----------
    gh : str
        Geohash identifier
    states : gpd.GeoDataFrame
        GeoDataFrame with state geometries
    total_dist_mi : float
        Total distance in miles for this geohash
    total_fuel_used : float
        Total fuel used for this geohash
    fuel_saved_metrics : dict
        Dictionary with fuel saved metrics (e.g., {'fuel_saving_per_day': value, ...})
    
    Returns:
    --------
    pd.DataFrame with one row per state, containing distributed portions
    """
    try:
        cell_poly = geohash_to_polygon(gh)
    except Exception as e:
        print(f"Warning: Could not create polygon for geohash {gh}: {e}")
        return pd.DataFrame()
    
    # Find states whose bboxes overlap with this geohash
    bbox_idxs = list(states.geometry.sindex.intersection(cell_poly.bounds))
    
    if not bbox_idxs:
        print(f"Warning: Geohash {gh} overlaps no state bboxes")
        return pd.DataFrame()
    
    # Calculate overlaps
    overlaps = []
    total_geohash_area = cell_poly.area
    
    for i in bbox_idxs:
        state_poly = states.geometry.iloc[i]
        state_name = states.iloc[i]["name"].lower()
        
        # Clean state polygon if needed
        if not state_poly.is_valid:
            state_poly = state_poly.buffer(0)
        
        try:
            intersection = state_poly.intersection(cell_poly)
        except TopologicalError:
            continue
        
        if not intersection.is_empty:
            overlap_area = intersection.area
            area_fraction = overlap_area / total_geohash_area
            
            overlaps.append({
                'geohash': gh,
                'state': state_name,
                'area_fraction': area_fraction,
                'overlap_area': overlap_area,
                'total_dist_mi_portion': total_dist_mi * area_fraction,
                'total_fuel_used_portion': total_fuel_used * area_fraction,
                **{f"{k}_portion": v * area_fraction for k, v in fuel_saved_metrics.items()}
            })
    
    if not overlaps:
        print(f"Warning: Geohash {gh} has no clean intersections with any state")
        return pd.DataFrame()
    
    # Normalize area fractions to sum to 1.0 (handles any floating point errors)
    df = pd.DataFrame(overlaps)
    total_fraction = df['area_fraction'].sum()
    df['area_fraction'] = df['area_fraction'] / total_fraction
    
    # Recalculate portions with normalized fractions
    df['total_dist_mi_portion'] = total_dist_mi * df['area_fraction']
    df['total_fuel_used_portion'] = total_fuel_used * df['area_fraction']
    for k in fuel_saved_metrics.keys():
        df[f"{k}_portion"] = fuel_saved_metrics[k] * df['area_fraction']
    
    return df


def create_state_distribution_table(gs_speed_df: pd.DataFrame, 
                                    states: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Create a distribution table showing how each geohash's metrics are split across states.
    
    Parameters:
    -----------
    gs_speed_df : pd.DataFrame
        DataFrame indexed by geohash with metrics to distribute
    states : gpd.GeoDataFrame
        GeoDataFrame with state geometries
    
    Returns:
    --------
    pd.DataFrame with columns: geohash, state, area_fraction, and distributed portions
    """
    all_distributions = []
    
    for gh in gs_speed_df.index:
        row = gs_speed_df.loc[gh]
        
        # Metrics that need to be distributed
        total_dist_mi = row.get('total_dist_mi', 0)
        total_fuel_used = row.get('total_fuel_used', 0)
        
        # Fuel saved metrics to distribute
        fuel_saved_metrics = {
            'fuel_saving_per_day': row.get('fuel_saving_per_day', 0),
            # Add other fuel saved metrics here as needed
        }
        
        # Distribute this geohash
        dist_df = distribute_geohash_to_states(
            gh, states, total_dist_mi, total_fuel_used, fuel_saved_metrics
        )
        
        if not dist_df.empty:
            all_distributions.append(dist_df)
    
    if not all_distributions:
        return pd.DataFrame()
    
    return pd.concat(all_distributions, ignore_index=True)