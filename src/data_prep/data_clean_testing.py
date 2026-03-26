"""Debug helper for running only the first race of selected seasons.

This script is intended for inspection/debugging of the race processing
pipeline. It does not build or save a full multi-race dataset.
"""

from data_clean import *


def debug_first_race_per_season(years, verbose=True):
    """Process only the first non-sprint race for each season.

    Parameters
    ----------
    years : list[int]
        Season years to debug.
    verbose : bool, default=True
        Passed through to ``process_session`` for detailed logging.

    Returns
    -------
    dict[int, pd.DataFrame]
        Mapping of season year to processed race dataframe.

    Raises
    ------
    RuntimeError
        If no usable race data is collected for any requested year.
    """
    results = {}

    for year in years:
        print(f"\n{'=' * 60}")
        print(f"  SEASON {year} (debug first race only)")
        print(f"{'=' * 60}")

        schedule = fastf1.get_event_schedule(year, include_testing=False)
        race_events = (
            schedule[schedule["RoundNumber"] > 0]
            .sort_values("RoundNumber")
            .reset_index(drop=True)
        )

        first_event = None
        for _, event_row in race_events.iterrows():
            if not is_sprint_weekend(event_row):
                first_event = event_row
                break

        if first_event is None:
            print("  No non-sprint race found for this season.")
            continue

        event_name = first_event.get("EventName", "Unknown")
        round_number = int(first_event.get("RoundNumber", 0))
        print(f"  Processing Round {round_number}: {event_name}")

        result = process_session(
            year, event_name, round_number, verbose=verbose
        )

        if result is not None and not result.empty:
            results[year] = result
            print(
                f"  Collected {len(result)} laps from "
                f"{result['Driver'].nunique()} drivers"
            )
            print(f"  Columns: {list(result.columns)}")
            print("  Sample rows:")
            print(result.head(5).to_string(index=False))
        else:
            print("  No usable laps collected for this race.")

    if not results:
        raise RuntimeError("No data collected for first-race debug run.")

    return results

if __name__ == "__main__":
    years = [2024]
    # run this for looking at the drs data
    debug_first_race_per_season(years, verbose=True)

    # now going to look at the weather data.
    #build_dataset(years, verbose=True)
    # currently getting the output Weather: merge failed (Merge keys contain null values on left side), using NaNs for weather data.
    