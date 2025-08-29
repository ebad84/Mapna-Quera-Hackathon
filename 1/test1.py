import pandas as pd
import numpy as np
import zipfile
import os

# بارگذاری داده و آماده‌سازی
df = pd.read_csv('statistical_analysis_data.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

SENSORS = ['Temperature_C','Pressure_kPa','VibAccel_m_s2','VibVelocity_mm_s']

def winsorize(s: pd.Series) -> pd.Series:
    q1 = s.quantile(0.01, interpolation='linear')
    q99 = s.quantile(0.99, interpolation='linear')
    return s.clip(lower=q1, upper=q99)

wdf = df.copy()
for c in SENSORS:
    wdf[c] = winsorize(wdf[c])

# مقداردهی اولیه متغیرهای پاسخ
value_01 = None
value_02 = None
value_03 = None
value_04 = None
value_05 = None
value_06 = None
value_07 = None
value_08 = None
value_09 = None
value_10 = None
value_11 = None
value_12 = None
value_13 = None
value_14 = None
value_15 = None
value_16 = None
value_17 = None
value_18 = None
value_19 = None
value_20 = None
value_21 = None
value_22 = None
value_23 = None
value_24 = None
value_25 = None
value_26 = None
value_27 = None
value_28 = None
value_29 = None
value_30 = None
value_31 = None
value_32 = None

# میانگین (وینسورشده)
value_01 = wdf['Temperature_C'].mean()
value_02 = wdf['Pressure_kPa'].mean()
value_03 = wdf['VibAccel_m_s2'].mean()
value_04 = wdf['VibVelocity_mm_s'].mean()

# انحراف معیار نمونه‌ای (ddof=1) 
value_05 = wdf['Temperature_C'].std(ddof=1)
value_06 = wdf['Pressure_kPa'].std(ddof=1)
value_07 = wdf['VibAccel_m_s2'].std(ddof=1)
value_08 = wdf['VibVelocity_mm_s'].std(ddof=1)

# کمینه و بیشینه خام (بدون وینسور)
value_09 = df['Temperature_C'].min()
value_10 = df['Temperature_C'].max()
value_11 = df['Pressure_kPa'].min()
value_12 = df['Pressure_kPa'].max()
value_13 = df['VibAccel_m_s2'].min()
value_14 = df['VibAccel_m_s2'].max()
value_15 = df['VibVelocity_mm_s'].min()
value_16 = df['VibVelocity_mm_s'].max()

# همبستگی پیرسون (وینسورشده، pairwise)
value_17 = wdf[['Temperature_C', 'Pressure_kPa']].corr().iloc[0,1]
value_18 = wdf[['Temperature_C', 'VibAccel_m_s2']].corr().iloc[0,1]
value_19 = wdf[['Temperature_C', 'VibVelocity_mm_s']].corr().iloc[0,1]
value_20 = wdf[['Pressure_kPa', 'VibAccel_m_s2']].corr().iloc[0,1]
value_21 = wdf[['Pressure_kPa', 'VibVelocity_mm_s']].corr().iloc[0,1]
value_22 = wdf[['VibAccel_m_s2', 'VibVelocity_mm_s']].corr().iloc[0,1]

# میانه (وینسورشده)
value_23 = wdf['Temperature_C'].median()
value_24 = wdf['Pressure_kPa'].median()
value_25 = wdf['VibAccel_m_s2'].median()
value_26 = wdf['VibVelocity_mm_s'].median()

# خودهمبستگی lag=1 (وینسورشده)
value_27 = wdf['Temperature_C'].autocorr(lag=1)
value_28 = wdf['Pressure_kPa'].autocorr(lag=1)
value_29 = wdf['VibAccel_m_s2'].autocorr(lag=1)
value_30 = wdf['VibVelocity_mm_s'].autocorr(lag=1)

# نرخ NaN
nan_mask = df[SENSORS].isna().any(axis=1)
value_31 = nan_mask.mean()

# میانه فاصله نمونه‌برداری
df_sorted = df.sort_values('timestamp')
time_diffs = df_sorted['timestamp'].diff().dt.total_seconds()
value_32 = time_diffs.median()

# ساخت خروجی
if not os.path.exists(os.path.join(os.getcwd(), 'notebook.ipynb')):
    %notebook -e notebook.ipynb

def compress(file_names):
    print("File Paths:")
    print(file_names)
    compression = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile("result.zip", mode="w") as zf:
        for file_name in file_names:
            zf.write('./' + file_name, file_name, compress_type=compression)

answers = [globals()[f'value_{i:02d}'] for i in range(1, 33)]
s = pd.Series(answers, dtype='float64', name="prediction")
s.to_csv("submission.csv", index=False, float_format="%.6f")

file_names = ['notebook.ipynb', 'submission.csv']
compress(file_names)