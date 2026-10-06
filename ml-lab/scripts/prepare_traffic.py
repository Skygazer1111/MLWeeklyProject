"""Prepare an auditable, continuous six-week transport teaching dataset."""
from pathlib import Path
import hashlib
import json
import pandas as pd


def prepare(data_dir):
    data_dir = Path(data_dir)
    original = data_dir / 'traffic_original.csv.gz'
    raw = pd.read_csv(original, parse_dates=['date_time'])
    assert len(raw) == 48204
    # Multiple weather descriptions can refer to the same measured traffic hour.
    assert raw.groupby('date_time')['traffic_volume'].nunique().max() == 1
    hourly = raw.drop_duplicates('date_time').sort_values('date_time')
    run_id = hourly['date_time'].diff().ne(pd.Timedelta(hours=1)).cumsum()
    longest = max(hourly.groupby(run_id), key=lambda item: len(item[1]))[1]
    selected = longest.iloc[:1008].copy()  # 42 days; selected before modelling.
    assert len(selected) == 1008
    assert selected['date_time'].diff().iloc[1:].eq(pd.Timedelta(hours=1)).all()
    df = pd.DataFrame({
        'date_time': selected['date_time'],
        'traffic_volume': selected['traffic_volume'],
        'temperature_c': selected['temp'] - 273.15,
    }).reset_index(drop=True)
    assert not df.isna().any().any()
    assert df['temperature_c'].between(-50, 60).all()
    df.to_csv(data_dir / 'traffic.csv', index=False)
    metadata = {
        'name': 'Metro Interstate Traffic Volume — six-week teaching window',
        'original_rows': len(raw), 'unique_hours': len(hourly), 'rows': len(df),
        'start': str(df['date_time'].iloc[0]), 'end': str(df['date_time'].iloc[-1]),
        'columns': {'date_time': 'Source local CST timestamp',
                    'traffic_volume': 'Vehicles counted during this hour',
                    'temperature_c': 'Air temperature in degrees Celsius'},
        'source': 'https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume',
        'download': 'https://archive.ics.uci.edu/static/public/492/metro%2Binterstate%2Btraffic%2Bvolume.zip',
        'citation': 'Hogue, J. (2019). Metro Interstate Traffic Volume. UCI Machine Learning Repository. DOI: 10.24432/C5X60B.',
        'license': 'CC BY 4.0',
        'preparation': 'Validate identical traffic counts at repeated timestamps; retain first weather entry per hour; sort timestamps; select first 1008 hours of longest continuous run; convert Kelvin to Celsius. No interpolation or invented readings.',
        'limitations': 'One road sensor and one six-week window. High volume is not proof of congestion: vehicle speed and road capacity are not included.',
        'sha256_original': hashlib.sha256(original.read_bytes()).hexdigest(),
    }
    (data_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return df


if __name__ == '__main__':
    result = prepare(Path(__file__).resolve().parents[1] / 'data')
    print(result.head().to_string(index=False))
    print('Prepared', len(result), 'continuous hourly observations.')
