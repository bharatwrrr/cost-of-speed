"""
State-level SC1 Fuel and Cost Savings Summary

For each of the 4 dataset days, joins the state-level fuel saving summary with
the time penalty data, computes derived metrics, and saves a formatted CSV.
Also produces a cross-day aggregate by summing raw columns before recomputing
ratios, which avoids averaging-of-averages bias.

Inputs  (relative to repo root):
    data/state_summary_without_coasting_day{1-4}.csv
        Columns used: num_trips, total_fuel_used, dist_mi, fuel_saving,
                      vmt_2021, scaled_fuel_saving, scaled_cost_saving

    data/state_time_penalty_day{1-4}.csv
        Columns used: dist_mi, og_time, adj_time  →  time_penalty_per_100_miles

Outputs (relative to repo root):
    table_data/statewise_sc1_day{1-4}.csv   – per-day state tables
    table_data/statewise_sc1_average.csv    – cross-day aggregate state table
"""

import os
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(REPO_ROOT, 'data')
OUTPUT_PATH = os.path.join(REPO_ROOT, 'table_data')

DAY_TO_MONTH = {1: 'Jan', 2: 'Apr', 3: 'Aug', 4: 'Oct'}

# Columns summed when aggregating across days
COLS_TO_SUM = [
    'num_trips', 'total_fuel_used', 'dist_mi', 'fuel_saving',
    'vmt_2021', 'scaled_fuel_saving', 'scaled_cost_saving',
]


# ── Per-day processing ────────────────────────────────────────────────────────

def process_day(day: int) -> pd.DataFrame:
    """
    Load and join the state summary and time penalty data for one day.
    Computes avg_fuel_saved_per_100_miles, joins time_penalty_per_100_miles,
    drops Alaska, and sorts by avg_fuel_saved_per_100_miles descending.
    """
    dtime_df = pd.read_csv(
        os.path.join(DATA_PATH, f'state_time_penalty_day{day}.csv'), index_col=0
    )
    state_df = pd.read_csv(
        os.path.join(DATA_PATH, f'state_summary_without_coasting_day{day}.csv'), index_col=0
    )

    state_df['avg_fuel_saved_per_100_miles'] = (
        state_df['fuel_saving'] / state_df['dist_mi'] * 100
    )
    state_df = state_df.join(dtime_df[['time_penalty_per_100_miles']])
    state_df = state_df.drop('AL', errors='ignore')
    state_df.sort_values('avg_fuel_saved_per_100_miles', ascending=False, inplace=True)

    return state_df


# ── Cross-day aggregate ───────────────────────────────────────────────────────

def build_average() -> pd.DataFrame:
    """
    Aggregate all 4 days by summing raw columns first, then recomputing
    ratio-based metrics. This avoids averaging-of-averages bias.

    time_penalty_per_100_miles is recomputed from summed og_time / adj_time / dist_mi
    so that states with more vehicle-miles travelled are weighted appropriately.
    """
    # Time penalty: sum raw time and distance columns, then recompute ratio
    dtime_dfs = [
        pd.read_csv(
            os.path.join(DATA_PATH, f'state_time_penalty_day{day}.csv'), index_col=0
        )
        for day in [1, 2, 3, 4]
    ]
    combined_dtime = pd.concat(dtime_dfs)
    dtime_summed = (
        combined_dtime
        .groupby(combined_dtime.index)[['dist_mi', 'og_time', 'adj_time']]
        .sum()
    )
    dtime_summed['time_penalty_per_100_miles'] = (
        (dtime_summed['adj_time'] - dtime_summed['og_time']) / dtime_summed['dist_mi'] * 100
    )

    # State summary: sum raw columns, then recompute fuel-saving ratio
    state_dfs = [
        pd.read_csv(
            os.path.join(DATA_PATH, f'state_summary_without_coasting_day{day}.csv'), index_col=0
        )
        for day in [1, 2, 3, 4]
    ]
    combined_state = pd.concat(state_dfs)
    state_summed = (
        combined_state
        .groupby(combined_state.index)[COLS_TO_SUM]
        .sum()
    )
    state_summed['avg_fuel_saved_per_100_miles'] = (
        state_summed['fuel_saving'] / state_summed['dist_mi'] * 100
    )

    state_final = state_summed.join(dtime_summed[['time_penalty_per_100_miles']])
    state_final.sort_values('avg_fuel_saved_per_100_miles', ascending=False, inplace=True)

    return state_final


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    for day in [1, 2, 3, 4]:
        month = DAY_TO_MONTH[day]
        print(f'Processing day {day} ({month})...')
        df = process_day(day)
        out_path = os.path.join(OUTPUT_PATH, f'statewise_sc1_day{day}_{month.lower()}.csv')
        df.to_csv(out_path)
        print(f'  Saved: {out_path}  ({len(df)} states)')

    print('Computing cross-day average...')
    avg_df = build_average()
    avg_path = os.path.join(OUTPUT_PATH, 'statewise_sc1_average.csv')
    avg_df.to_csv(avg_path)
    print(f'  Saved: {avg_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()