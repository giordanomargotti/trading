"""Converte gli archivi 7z SPX EOD in Parquet mensili (colonne essenziali)."""
import sys, os, glob, shutil, py7zr
import pandas as pd, numpy as np
src, out, tmp = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
KEEP = ['QUOTE_DATE','UNDERLYING_LAST','EXPIRE_DATE','DTE','STRIKE','C_BID','C_ASK','C_DELTA','C_IV','C_VEGA','C_GAMMA','C_THETA','C_VOLUME',
        'P_BID','P_ASK','P_DELTA','P_IV','P_VEGA','P_GAMMA','P_THETA','P_VOLUME']
for z in sorted(glob.glob(os.path.join(src, '*.7z'))):
    d = os.path.join(tmp, os.path.basename(z)[:-3])
    shutil.rmtree(d, ignore_errors=True)
    with py7zr.SevenZipFile(z) as a:
        a.extractall(d)
    for f in sorted(glob.glob(os.path.join(d, '**', '*.txt'), recursive=True)):
        name = os.path.basename(f)[:-4]
        dst = os.path.join(out, name + '.parquet')
        df = pd.read_csv(f, skipinitialspace=True, low_memory=False)
        df.columns = df.columns.str.strip().str.replace(r'[\[\]]', '', regex=True).str.upper()
        df = df[KEEP].copy()
        for c in ('QUOTE_DATE', 'EXPIRE_DATE'):
            df[c] = pd.to_datetime(df[c].astype(str).str.strip(), errors='coerce')
        for c in KEEP:
            if c not in ('QUOTE_DATE', 'EXPIRE_DATE'):
                df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32' if c != 'UNDERLYING_LAST' else 'float64')
        df = df.dropna(subset=['QUOTE_DATE', 'EXPIRE_DATE', 'STRIKE'])
        if os.path.exists(dst):   # archivi sovrapposti (2021 annuale e trimestrali)
            old = pd.read_parquet(dst)
            df = pd.concat([old, df]).drop_duplicates(['QUOTE_DATE', 'EXPIRE_DATE', 'STRIKE'])
        df.to_parquet(dst, index=False)
        print(name, len(df), df.QUOTE_DATE.min().date(), df.QUOTE_DATE.max().date(), flush=True)
    shutil.rmtree(d, ignore_errors=True)
