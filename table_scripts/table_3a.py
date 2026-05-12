"""
Table A3: BEV vs. ICEV Speeding Statistics for California

Reads per-vehicle-class speeding data for each of the 4 dataset days and
produces the full Table A3: per-day rows (Jan, Apr, Aug, Oct) plus a
weighted average row, for Subcompact Cars (SC) and Small Station Wagons (SSW)
across both BEVs and ICEVs.

Inputs  (relative to repo root):
    figure_data/per_class_veh_os_ca_bev_day{1-4}.csv
    figure_data/per_class_veh_os_ca_icev_day{1-4}.csv

Outputs (relative to repo root):
    table_data/table_a3_bev_summary.csv
    table_data/table_a3_icev_summary.csv
    table_data/table_a3_combined.csv
"""

import os
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURE_DATA_PATH = os.path.join(REPO_ROOT, 'figure_data')
OUTPUT_PATH = os.path.join(REPO_ROOT, 'table_data')

# Day index → month label (matches the 4 dataset days in the paper)
DAY_TO_MONTH = {1: 'Jan', 2: 'Apr', 3: 'Aug', 4: 'Oct'}


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_per_class_csv(pt_type: str, day: int) -> pd.DataFrame:
    """Load the per-vehicle-class speeding CSV for a given powertrain and day."""
    fname = f'per_class_veh_os_ca_{pt_type.lower()}_day{day}.csv'
    fpath = os.path.join(FIGURE_DATA_PATH, fname)
    return pd.read_csv(fpath, index_col=0)


def build_summary(pt_type: str) -> pd.DataFrame:
    """
    Build the full summary table for a given powertrain type ('bev' or 'icev').

    Returns a DataFrame with a MultiIndex (month, vehicle_class) and columns:
        perc_trip_os    – speeding prevalence (% of trips)
        avg_max_os      – mean maximum speed excess (mph), averaged over speeding trips
        avg_dist_mi     – mean distance per trip (miles), averaged over all trips

    Rows cover Jan / Apr / Aug / Oct (individual days) plus 'Average'
    (trip-count-weighted aggregate across all 4 days).
    """
    per_day_rows = []
    # Accumulators for the weighted average row
    raw = {}

    for day in [1, 2, 3, 4]:
        df    = load_per_class_csv(pt_type, day)
        month = DAY_TO_MONTH[day]

        for veh_class in df.index:
            num_trips = df.loc[veh_class, 'num_trips']
            num_os_trips = df.loc[veh_class, 'num_os_trips']
            avg_max_os = df.loc[veh_class, 'avg_max_os']
            dist_mi = df.loc[veh_class, 'dist_mi']
            total_time_os = df.loc[veh_class, 'total_time_os']
            total_trip_time = df.loc[veh_class, 'total_trip_time']
            avg_dist = dist_mi / num_trips
            perc_trip_os = df.loc[veh_class, 'perc_trips_os'] * 100

            per_day_rows.append({
                'month': month,
                'veh_class': veh_class,
                'perc_trip_os': perc_trip_os,
                'avg_max_os': avg_max_os,
                'avg_dist_mi': avg_dist,
            })

            if veh_class not in raw:
                raw[veh_class] = {
                    'avg_max_os': [avg_max_os],
                    'avg_dist': [avg_dist],
                    'num_trips': [num_trips],
                    'num_os_trips': [num_os_trips],
                    'total_time_os': total_time_os,
                    'total_trip_time': total_trip_time,
                }
            else:
                raw[veh_class]['avg_max_os'].append(avg_max_os)
                raw[veh_class]['avg_dist'].append(avg_dist)
                raw[veh_class]['num_trips'].append(num_trips)
                raw[veh_class]['num_os_trips'].append(num_os_trips)
                raw[veh_class]['total_time_os'] += total_time_os
                raw[veh_class]['total_trip_time'] += total_trip_time

    # Build the weighted-average row across all 4 days
    avg_rows = []
    for veh_class, data in raw.items():
        num_trips = np.array(data['num_trips'])
        num_os_trips = np.array(data['num_os_trips'])
        avg_max_os = np.array(data['avg_max_os'])
        avg_dist = np.array(data['avg_dist'])
        total_trips = num_trips.sum()

        avg_rows.append({
            'month': 'Average',
            'veh_class': veh_class,
            'perc_trip_os': num_os_trips.sum() / total_trips * 100,
            'avg_max_os': (avg_max_os * num_trips).sum() / total_trips,
            'avg_dist_mi': (avg_dist * num_trips).sum() / total_trips,
        })

    all_rows = per_day_rows + avg_rows
    summary  = pd.DataFrame(all_rows).set_index(['month', 'veh_class'])
    return summary


def build_combined_table(bev_summary: pd.DataFrame,
                         icev_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Assemble the combined Table A3 in the layout used in the paper:
    months as groups, four vehicle rows per group (SC ICEV, SSW ICEV, SC BEV, SSW BEV).
    """
    month_order = ['Jan', 'Apr', 'Aug', 'Oct', 'Average']
    vehicle_map = [
        ('Subcompact Cars', 'SC ICEV', icev_summary),
        ('Small Station Wagons', 'SSW ICEV', icev_summary),
        ('Subcompact Cars', 'SC BEV', bev_summary),
        ('Small Station Wagons', 'SSW BEV', bev_summary),
    ]

    rows = []
    for month in month_order:
        for veh_class, label, src in vehicle_map:
            row = src.loc[(month, veh_class)]
            rows.append({
                'Month': month,
                'Vehicle': label,
                'Speeding Prevalence (%)': round(row['perc_trip_os'], 2),
                'Mean Maximum Speed Excess (mph)': round(row['avg_max_os'],   2),
                'Distance Travelled per Trip (miles)': round(row['avg_dist_mi'],  2),
            })

    return pd.DataFrame(rows)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    print('Building BEV summary...')
    bev_summary = build_summary('bev')
    bev_summary.to_csv(os.path.join(OUTPUT_PATH, 'table_a3_bev_summary.csv'))

    print('Building ICEV summary...')
    icev_summary = build_summary('icev')
    icev_summary.to_csv(os.path.join(OUTPUT_PATH, 'table_a3_icev_summary.csv'))

    print('Assembling combined Table A3...')
    combined = build_combined_table(bev_summary, icev_summary)
    combined.to_csv(os.path.join(OUTPUT_PATH, 'table_a3_combined.csv'), index=False)

    print(f'\nOutputs written to: {OUTPUT_PATH}')
    print()
    print(combined.to_string(index=False))


if __name__ == '__main__':
    main()