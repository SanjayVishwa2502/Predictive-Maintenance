import pandas as pd

machine = "cnc_fanuc_robodrill_001"
df = pd.read_parquet(f'data/synthetic_fixed/{machine}/train.parquet')

print('='*70)
print(f'VALIDATION: {machine}')
print('='*70)
print(f'Samples: {len(df)}')
print(f'Features: {len(df.columns)}')
print(f'Columns: {list(df.columns)}')
print()

ts_sorted = df['timestamp'].is_monotonic_increasing
print(f'Timestamp sorted: {ts_sorted} {"✅" if ts_sorted else "❌"}')

rul_dec = (df['rul'].diff()[1:] <= 0).sum()
rul_pct = rul_dec/(len(df)-1)*100
print(f'RUL decreasing: {rul_dec}/{len(df)-1} ({rul_pct:.1f}%) {"✅" if rul_pct > 90 else "❌"}')

print(f'RUL range: {df["rul"].max():.2f} → {df["rul"].min():.2f}')
print(f'Time range: {df["timestamp"].min()} to {df["timestamp"].max()}')
print()

status = "✅ PASS" if ts_sorted and rul_pct > 90 else "❌ FAIL"
print(f'Status: {status}')
