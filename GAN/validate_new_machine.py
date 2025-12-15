import pandas as pd
import sys
from pathlib import Path

# Get machine ID from command line or use default
machine = sys.argv[1] if len(sys.argv) > 1 else "cnc_fanuc_robodrill_001"

# Try multiple directories
data_path = None
for dir_name in ['synthetic', 'synthetic_fixed']:
    path = Path(f'data/{dir_name}/{machine}/train.parquet')
    if path.exists():
        data_path = path
        break

if not data_path:
    print(f'❌ ERROR: No synthetic data found for {machine}')
    print(f'   Searched in: data/synthetic/{machine}/ and data/synthetic_fixed/{machine}/')
    sys.exit(1)

df = pd.read_parquet(data_path)

print('='*70)
print(f'VALIDATION: {machine}')
print(f'Data source: {data_path}')
print('='*70)
print(f'Samples: {len(df)}')
print(f'Features: {len(df.columns)}')
print(f'Columns: {list(df.columns)}')
print()

ts_sorted = df['timestamp'].is_monotonic_increasing if 'timestamp' in df.columns else False
print(f'Timestamp sorted: {ts_sorted} {"✅" if ts_sorted else "❌"}')

if 'rul' in df.columns:
    rul_dec = (df['rul'].diff()[1:] <= 0).sum()
    rul_pct = rul_dec/(len(df)-1)*100
    print(f'RUL decreasing: {rul_dec}/{len(df)-1} ({rul_pct:.1f}%) {"✅" if rul_pct > 90 else "❌"}')
    print(f'RUL range: {df["rul"].max():.2f} → {df["rul"].min():.2f}')
else:
    rul_pct = 0
    print('RUL decreasing: ❌ (RUL column not found)')

if 'timestamp' in df.columns:
    print(f'Time range: {df["timestamp"].min()} to {df["timestamp"].max()}')
print()

status = "✅ PASS" if ts_sorted and rul_pct > 90 else "❌ FAIL"
print(f'Status: {status}')
