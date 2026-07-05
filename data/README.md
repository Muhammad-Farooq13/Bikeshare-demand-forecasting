# Dataset

**Source:** UCI Machine Learning Repository — "Bike Sharing Dataset"
**Original authors:** Hadi Fanaee-T and João Gama (2013)
**Citation:** Fanaee-T, H. (2013). Bike Sharing [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5W894
**License:** CC BY 4.0 (Creative Commons Attribution 4.0 International) — sharing and adaptation permitted with attribution, which this document provides
**Fetched via:** a GitHub mirror of the UCI files (`src/ingest.py` downloads `hour.csv` at build time), since this sandbox has no direct network path to `archive.ics.uci.edu`

## What it contains

Hourly bike rental counts from the Capital Bikeshare system (Washington, D.C.)
across **2011–2012** (17,379 hourly records), joined with weather and calendar
context:

| Column | Description |
|---|---|
| `dteday` | Date |
| `season` | 1=winter, 2=spring, 3=summer, 4=fall |
| `yr` | 0=2011, 1=2012 |
| `mnth`, `hr` | Month (1–12), hour of day (0–23) |
| `holiday`, `workingday` | Calendar flags |
| `weekday` | Day of week (0–6) |
| `weathersit` | 1=clear ... 4=heavy rain/snow |
| `temp`, `atemp`, `hum`, `windspeed` | Normalized weather readings |
| `casual`, `registered`, `cnt` | Rental counts (`cnt` = casual + registered, the forecasting target) |

## Why this dataset

Bike-share and micromobility operators (Citi Bike, Capital Bikeshare, Lyft's
bike/scooter fleets) run demand forecasting continuously to decide **fleet
rebalancing** — trucking bikes between stations overnight so popular stations
aren't empty at 8am and unpopular ones aren't full. Getting this wrong either
strands riders or wastes rebalancing-truck operating costs. This dataset is
real (not synthetic), has genuine weather-driven and calendar-driven demand
patterns, and is large enough (17k+ hourly points across 2 full years) to
require honest time-series validation instead of a single lucky train/test
split.

## Re-fetching

```bash
python -m src.ingest
```
