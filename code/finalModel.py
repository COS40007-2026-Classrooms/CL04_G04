#  CL04_G04 Group Project
#  VIC FUEL PRICE FORECAST  --  ULP91 + ULP95 + DIESEL
#  Predicts average Melbourne retail pump price (cpl) for the coming week
#  Saves 3 separate .h5 model files
#
#  DATA SOURCES:
#   PRIMARY:  Victoria Servo Saver Public API (requires free API key)
#             Apply: https://discover.data.vic.gov.au/dataset/servo-saver-public-api
#             python fuel_price_forecast.py --api-key YOUR_KEY this is used for security so no one has access from repo
#
#   FALLBACK: ACCC weekly PDF auto-detected from script folder
#             Download each Friday from:
#             https://www.accc.gov.au/consumers/petrol-and-fuel/petrol-prices-in-2026
#             Requires: pip install pdfplumber
#
#   TRAINING: APS quarterly data Q1 2010 - Q4 2025 (64 quarters)
#
#  USAGE:
#    python finalModel.py
#    python finalModel.py --api-key YOUR_KEY
#    python fuel_price_forecast.py --skip-live (for fallback)
#
#  OUTPUTS (models/):
#    ulp91_price_model.h5  ulp95_price_model.h5  diesel_price_model.h5
#    scalers_ulp91_price.pkl  scalers_ulp95_price.pkl  scalers_diesel_price.pkl
#
#  OUTPUTS (reports/):
#    price_forecast_report.txt
#    price_forecast.json
#    price_training_metrics.csv
#    price_forecast_chart.png
# ============================

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

import os, sys, warnings, json, gc, pickle, argparse, re
warnings.filterwarnings('ignore')

import pandas as pd
pd.options.future.infer_string = False

import numpy as np
import h5py
from pathlib import Path
from datetime import datetime, timedelta
import urllib.request, urllib.error

from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from scipy import stats

#  CONFIG
SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR   = SCRIPT_DIR.parent
MDL_DIR    = ROOT_DIR / 'models';   MDL_DIR.mkdir(exist_ok=True)
RPT_DIR    = ROOT_DIR / 'reports';  RPT_DIR.mkdir(exist_ok=True)
DAT_DIR    = ROOT_DIR / 'data'
ACTUALS_HIST = DAT_DIR / 'actuals_history.csv'

RIDGE_ALPHA = 1.0
VAL_FRAC = 0.15
CV_FOLDS = 5
F1_TOL = 0.05

EXCISE_CUT_START = datetime(2026, 4, 1)
EXCISE_CUT_END = datetime(2026, 6, 30)
EXCISE_CUT_CPL = 32.0

API_ROW_WEIGHT = 8.0
API_ONLY_THRESHOLD_WEEKS = 104

# Hardcoded fallback anchor from ACCC PDF 24-Apr-2026
ACCC_ANCHOR = {
    'date': '2026-04-22',
    'ulp91_mel':  192.4,
    'ulp95_mel': 207.4,
    'diesel_mel': 282.8,
}

# Column order in ACCC city tables
ACCC_CITIES = ['Sydney','Melbourne','Brisbane','Adelaide',
               'Perth','Canberra','Hobart','Darwin']
MEL_IDX = 1

#  ARGUMENT PARSER
def parse_args():
    p = argparse.ArgumentParser(description='VIC Fuel Price Forecast')
    p.add_argument('--api-key', default=os.environ.get('SERVO_SAVER_API_KEY',''),
                   help='Victoria Servo Saver API Consumer ID')
    p.add_argument('--skip-live', action='store_true',
                   help='Skip live data sources, use hardcoded anchor only')
    return p.parse_args()

#  HELPERS
def find_csv(keywords):
    for f in DAT_DIR.glob('*.csv'):
        name = f.name.replace('_', ' ').lower()
        if all(w.lower() in name for w in keywords):
            return f
    raise FileNotFoundError(f'Cannot find CSV with keywords: {keywords}')

def find_accc_pdf():
    """Return most recent ACCC weekly PDF found in script directory."""
    pdfs = sorted(DAT_DIR.glob('weekly-fuel-price-monitoring-report-*.pdf'))
    return pdfs[-1] if pdfs else None

#  DATA SOURCE 1: Victoria Servo Saver API  (PRIMARY)
def fetch_servo_saver(api_key):
    """
    Fetch live VIC pump prices from the Service Victoria Fair Fuel Open Data API.

    Official docs: Fair Fuel Open Data API Documentation v1.0 (Service Victoria)
    Base URL:  https://api.fuel.service.vic.gov.au/open-data/v1
    Endpoint:  GET /fuel/prices
    Auth:      x-consumer-id header
    Also req:  x-transactionid (UUID v4), User-Agent

    Fuel type codes: U91=ULP91, P95=ULP95, DSL=Diesel
    Response:  fuelPriceDetails[].fuelPrices[].{fuelType, price, isAvailable}
    Data delay: 24 hours (by design of API)

    Returns {ulp91, ulp95, diesel, source} in cpl, or None on failure.
    """
    if not api_key:
        return None

    import uuid as _uuid

    BASE     = 'https://api.fuel.service.vic.gov.au/open-data/v1'
    fuel_map = {'ulp91': 'U91', 'ulp95': 'P95', 'diesel': 'DSL'}
    headers  = {
        'x-consumer-id': api_key, # correct header name per docs
        'x-transactionid': str(_uuid.uuid4()),  # fresh UUID v4 per request
        'User-Agent': 'VIC-Fuel-Forecast/1.0',
        'Accept': 'application/json',
    }

    results = {}
    try:
        url = BASE + '/fuel/prices'
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode())

        # Parse fuelPriceDetails[].fuelPrices[]
        prices_by_type = {}
        for station in data.get('fuelPriceDetails', []):
            for fp in station.get('fuelPrices', []):
                ft = fp.get('fuelType', '')
                price = fp.get('price')
                avail = fp.get('isAvailable', True)
                if price and avail:
                    try:
                        prices_by_type.setdefault(ft, []).append(float(price))
                    except (TypeError, ValueError):
                        pass

        for our_name, api_code in fuel_map.items():
            vals = prices_by_type.get(api_code, [])
            if vals:
                results[our_name] = round(float(np.median(vals)), 1)

        if results:
            results['source'] = 'Fair Fuel Open Data API (Service Victoria, 24hr delay)'
            n = len(data.get('fuelPriceDetails', []))
            print('  [Fair Fuel API] Prices fetched from ' + str(n) + ' stations (24hr delay):')
            print('  ULP91=' + str(results.get('ulp91','N/A')) +
                  '  ULP95=' + str(results.get('ulp95','N/A')) +
                  '  Diesel=' + str(results.get('diesel','N/A')) + ' cpl')
        else:
            print('  [Fair Fuel API] Connected but no U91/P95/DSL prices in response')
            print('  Available types: ' + str(list(prices_by_type.keys())))

    except urllib.error.HTTPError as e:
        body = ''
        try: body = e.read().decode('utf-8', errors='replace')[:200]
        except Exception: pass
        print('  [Fair Fuel API] HTTP ' + str(e.code) + ' ' + e.reason)
        if e.code == 403:
            print('    Check: key not yet approved, or key has been re-issued')
        elif e.code == 429:
            print('    Rate limited -- max 10 req/60 sec. Wait and retry.')
        if body:
            print('    Body: ' + body)
    except urllib.error.URLError as e:
        print('  [Fair Fuel API] Connection failed: ' + str(e.reason))
        print('    Endpoint: ' + BASE + '/fuel/prices')
    except Exception as e:
        print('  [Fair Fuel API] Error: ' + type(e).__name__ + ': ' + str(e))

    return results if results else None

#  DATA SOURCE 2: ACCC Weekly PDF  (FALLBACK ONLY)
def parse_accc_pdf(pdf_path):
    """
    Extract Melbourne retail prices from the ACCC weekly PDF using pdfplumber.

    Row format in PDF tables:
      'Melbourne 259.1 192.4 175.9 266.9 -66.7'
       city       31-Mar  22-Apr  low   high  change
    The SECOND number (index 1) is the most recent weekly average.

    Petrol table on page ~6, diesel table on page ~7.
    Returns {ulp91, ulp95, diesel, date, source} or None.
    """
    try:
        import pdfplumber
    except ImportError:
        print('  [ACCC PDF] pdfplumber not installed. Run: pip install pdfplumber')
        return None

    try:
        with pdfplumber.open(pdf_path) as pdf:
            ulp91_mel  = None
            diesel_mel = None
            report_date = None

            # Extract report date from page 1
            text0 = pdf.pages[0].extract_text() or ''
            dm = re.search(
                r'(\d{1,2})\s+(January|February|March|April|May|June|July|'
                r'August|September|October|November|December)\s+(20\d{2})', text0)
            if dm:
                try:
                    report_date = datetime.strptime(
                        dm.group(1) + ' ' + dm.group(2) + ' ' + dm.group(3),
                        '%d %B %Y').strftime('%Y-%m-%d')
                except Exception:
                    pass

            # Scan all pages for 'Melbourne NNN.N NNN.N ...' rows
            for page in pdf.pages:
                text = page.extract_text() or ''
                for line in text.splitlines():
                    line = line.strip()
                    if not line.lower().startswith('melbourne'):
                        continue
                    nums = re.findall(r'\b(\d{3}\.\d)\b', line)
                    if len(nums) < 2:
                        continue
                    vals = [float(n) for n in nums]
                    # Retail petrol: 140–320 cpl
                    if ulp91_mel is None and all(140 < v < 320 for v in vals[:2]):
                        ulp91_mel = vals[1]   # second = most recent week
                    # Retail diesel: 200–420 cpl (distinct from petrol range)
                    elif diesel_mel is None and all(200 < v < 420 for v in vals[:2]):
                        diesel_mel = vals[1]

                if ulp91_mel and diesel_mel:
                    break

            if ulp91_mel and diesel_mel:
                result = {
                    'ulp91': ulp91_mel,
                    'ulp95': round(ulp91_mel + 15.0, 1),
                    'diesel': diesel_mel,
                    'date': report_date or ACCC_ANCHOR['date'],
                    'source': 'ACCC PDF (' + pdf_path.name + ')',
                }
                print('  [ACCC PDF] Melbourne prices extracted:')
                print('  ULP91=' + str(result['ulp91']) +
                      '  ULP95=' + str(result['ulp95']) +
                      '  Diesel=' + str(result['diesel']) + ' cpl')
                print('  Report date: ' + str(result['date']))
                return result

            print('[ACCC PDF] Could not extract Melbourne rows '
                  '(ulp91=' + str(ulp91_mel) + ', diesel=' + str(diesel_mel) + ')')

    except Exception as e:
        print('  [ACCC PDF] Error: ' + type(e).__name__ + ': ' + str(e))

    return None


#  BUILD WEEKLY PRICE DATASET
def build_price_dataset(live_prices=None, anchor_date=None):
    """
    Build weekly price dataset for training.

    Data source priority:
      1. API actuals history (data/actuals_history.csv) -- real weekly prices
         populated automatically each week by retrain_pipeline.py
      2. ACCC anchor observations -- handful of verified real prices
      3. APS quarterly CSV -- synthetic weekly interpolation as backbone
         (used less and less as API history accumulates)

    When API history >= API_ONLY_THRESHOLD_WEEKS, APS quarterly is dropped.
    Real API rows are upweighted by API_ROW_WEIGHT in sklearn's sample_weight.
    """
    print('[DATASET] Building weekly price dataset...')

    #  Load API actuals history 
    api_rows = pd.DataFrame()
    if ACTUALS_HIST.exists():
        api_raw = pd.read_csv(ACTUALS_HIST)
        api_raw['date'] = pd.to_datetime(api_raw['date'])
        api_raw = api_raw.rename(columns={
            'ulp91_actual': 'ulp91',
            'ulp95_actual': 'ulp95',
            'diesel_actual': 'diesel',
        })
        api_raw['source'] = 'API_actual'
        api_rows = api_raw[['date','ulp91','ulp95','diesel','source']].dropna()
        n_api = len(api_rows)
        print(f'  API history: {n_api} weeks of real station data')
    else:
        n_api = 0
        print('  API history: none yet (will use APS quarterly backbone)')

    #  Load APS quarterly backbone 
    use_aps = (n_api < API_ONLY_THRESHOLD_WEEKS)
    aps_rows = pd.DataFrame()

    if use_aps:
        fp = pd.read_csv(find_csv(['fuel', 'prices']))
        rows = []
        for _, r in fp.iterrows():
            q      = int(r['Quarter'][1])
            yr     = int(r['Year'])
            p91    = float(r['Regular unleaded petrol (91 RON) (cpl)'])
            p95    = float(r['Premium unleaded petrol (95 RON) (cpl)'])
            pds    = float(r['Automotive diesel (cpl)'])
            qstart = pd.Timestamp(yr, (q-1)*3+1, 1)
            for wk in range(13):
                wdate     = qstart + timedelta(weeks=wk)
                cycle_pos = np.sin(2*np.pi*wk/6)
                rows.append({
                    'date':   wdate,
                    'ulp91':  round(p91 + cycle_pos*7.5, 1),
                    'ulp95':  round(p95 + cycle_pos*7.5, 1),
                    'diesel': round(pds + cycle_pos*2.5, 1),
                    'source': 'APS_quarterly',
                })
        aps_rows = pd.DataFrame(rows)
        aps_rows['date'] = pd.to_datetime(aps_rows['date'])

        # If we have API history, truncate APS to dates BEFORE the API period
        # so they don't overlap and confuse the model
        if n_api > 0:
            api_start = api_rows['date'].min()
            aps_rows  = aps_rows[aps_rows['date'] < api_start].copy()
            print(f'  APS quarterly: {len(aps_rows)} synthetic rows '
                  f'(pre-{api_start.strftime("%b %Y")}, before API coverage)')
        else:
            print(f'  APS quarterly: {len(aps_rows)} synthetic rows (full range)')
    else:
        print(f'  APS quarterly: SKIPPED '
              f'(API history has {n_api} weeks >= {API_ONLY_THRESHOLD_WEEKS} threshold)')

    #  ACCC anchor observations 
    accc_obs = [
        ('2026-02-20', 176.1, 191.1, 178.9),
        ('2026-03-31', 259.1, 274.1, 324.4),
        ('2026-04-15', 215.3, 230.3, 311.8),
        ('2026-04-22', 192.4, 207.4, 282.8),
    ]
    accc_rows = pd.DataFrame([
        {'date': pd.Timestamp(d), 'ulp91': u91, 'ulp95': u95,
         'diesel': dsl, 'source': 'ACCC_anchor'}
        for d, u91, u95, dsl in accc_obs
    ])

    #  Live / current week observation 
    live_rows = pd.DataFrame()
    if live_prices:
        obs_date = (pd.Timestamp(anchor_date) if anchor_date
                    else pd.Timestamp(datetime.now().date()))
        live_rows = pd.DataFrame([{
            'date':   obs_date,
            'ulp91':  live_prices.get('ulp91',  ACCC_ANCHOR['ulp91_mel']),
            'ulp95':  live_prices.get('ulp95',  ACCC_ANCHOR['ulp95_mel']),
            'diesel': live_prices.get('diesel', ACCC_ANCHOR['diesel_mel']),
            'source': live_prices.get('source', 'live'),
        }])
        print(f'  Seed: ULP91={live_prices.get("ulp91")}  '
              f'ULP95={live_prices.get("ulp95")}  '
              f'Diesel={live_prices.get("diesel")} cpl')
    else:
        print('  Seed: ACCC hardcoded anchor (22-Apr-2026)')

    #  Combine all sources 
    parts = [p for p in [aps_rows, accc_rows, api_rows, live_rows]
             if len(p) > 0]
    df = pd.concat(parts, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').drop_duplicates('date', keep='last').reset_index(drop=True)

    #  Context features 
    df['excise_cut'] = ((df['date'] >= pd.Timestamp(EXCISE_CUT_START)) &
                        (df['date'] <= pd.Timestamp(EXCISE_CUT_END))).astype(float)
    df['covid']      = ((df['date'].dt.year == 2020) |
                        ((df['date'].dt.year == 2021) &
                         (df['date'].dt.month <= 10))).astype(float)

    #  Compute sample weights 
    # Real API rows get higher weight so Ridge prioritises fitting them.
    # This is passed to model.fit(X, y, sample_weight=w) in train_model().
    df['sample_weight'] = np.where(
        df['source'].isin(['API_actual', 'live']),
        API_ROW_WEIGHT,
        np.where(df['source'] == 'ACCC_anchor', 4.0, 1.0)
    )

    n_real      = (df['source'].isin(['API_actual','live','ACCC_anchor'])).sum()
    n_synthetic = (df['source'] == 'APS_quarterly').sum()
    print(f'  Dataset: {len(df)} rows  '
          f'({n_real} real, {n_synthetic} synthetic)')
    print(f'  Date range: {df["date"].min().strftime("%b %Y")} -- '
          f'{df["date"].max().strftime("%b %Y")}')
    if n_api > 0:
        print(f'  Real data coverage: '
              f'{api_rows["date"].min().strftime("%b %Y")} to '
              f'{api_rows["date"].max().strftime("%b %Y")}')

    return df

#  FEATURE ENGINEERING
def engineer_features(df, fuel):
    v = df.copy()
    v['lag1'] = v[fuel].shift(1)
    v['lag2'] = v[fuel].shift(2)
    v['lag4'] = v[fuel].shift(4)
    v['lag13'] = v[fuel].shift(13)
    v['roll4'] = v[fuel].shift(1).rolling(4).mean()
    v['roll13'] = v[fuel].shift(1).rolling(13).mean()
    v['roll_std4'] = v[fuel].shift(1).rolling(4).std()
    v['trend4'] = v[fuel].shift(1) - v[fuel].shift(5)
    v['trend13']   = v[fuel].shift(1) - v[fuel].shift(14)
    M = v['date'].dt.month
    v['month_sin'] = np.sin(2*np.pi*M/12)
    v['month_cos'] = np.cos(2*np.pi*M/12)
    v['year_norm'] = (v['date'].dt.year - 2010) / 16.0
    v['cycle_pos'] = np.sin(2*np.pi*v['date'].dt.isocalendar().week.astype(float)/6)
    if fuel == 'ulp91':
        v['spread'] = v['ulp91'] - v['diesel']
    elif fuel == 'ulp95':
        v['spread'] = v['ulp95'] - v['ulp91']
    else:
        v['spread'] = v['diesel'] - v['ulp91']
    FEAT = ['lag1','lag2','lag4','lag13','roll4','roll13','roll_std4',
            'trend4','trend13','month_sin','month_cos','year_norm',
            'cycle_pos','spread','excise_cut','covid']
    v = v.dropna(subset=FEAT + [fuel]).reset_index(drop=True)
    return v, FEAT, fuel

#  METRICS
def calc_metrics(true, pred):
    mae  = float(mean_absolute_error(true, pred))
    rmse = float(np.sqrt(mean_squared_error(true, pred)))
    mape = float(np.mean(np.abs((true - pred) / true)) * 100)
    r2   = float(r2_score(true, pred))
    da   = (float(np.mean(np.sign(np.diff(true)) == np.sign(np.diff(pred))) * 100)
            if len(true) > 1 else 0.0)
    tp   = np.sum(np.abs((true - pred) / true) < F1_TOL)
    fp_  = np.sum((pred - true) / true >  F1_TOL)
    fn   = np.sum((true - pred) / true >  F1_TOL)
    pr_  = tp / (tp + fp_) if (tp + fp_) > 0 else 0.
    rc   = tp / (tp + fn)  if (tp + fn)  > 0 else 0.
    f1   = float(2*pr_*rc / (pr_ + rc) if (pr_ + rc) > 0 else 0.)
    pr_r = float(stats.pearsonr(true, pred)[0]) if np.std(pred) > 0 else 0.
    naive = np.abs(np.diff(true)); model = np.abs(true[1:] - pred[1:])
    tu    = (float(np.sqrt(np.mean(model**2)) / np.sqrt(np.mean(naive**2)))
             if np.mean(naive**2) > 0 else 0.)
    return dict(mae=round(mae,3), rmse=round(rmse,3), mape=round(mape,3),
                r2=round(r2,4), da=round(da,2), f1=round(f1,4),
                pearson_r=round(pr_r,4), theils_u=round(tu,4),
                within_2pct=round(float(np.mean(np.abs((true-pred)/true)<0.02)*100),2),
                within_5pct=round(float(np.mean(np.abs((true-pred)/true)<0.05)*100),2))

#  TRAIN
def train_model(df, FEAT, y_col, fuel_name):
    print('\n[TRAIN] ' + fuel_name.upper() +
          ' (' + str(len(df)) + ' obs, ' + str(len(FEAT)) + ' features)')

    X_raw  = df[FEAT].astype(np.float32).values
    y_raw  = df[y_col].values.astype(np.float32)
    w_raw  = df['sample_weight'].values.astype(np.float32)   # NEW: weights
    N      = len(df)
    sx     = RobustScaler()
    X_sc   = sx.fit_transform(X_raw).astype(np.float32)

    sp = int(N * (1 - VAL_FRAC))
    m  = Ridge(alpha=RIDGE_ALPHA)
    m.fit(X_sc[:sp], y_raw[:sp], sample_weight=w_raw[:sp])   # pass weights
    hm = calc_metrics(y_raw[sp:], m.predict(X_sc[sp:]))
    print('  Holdout  MAPE=' + str(round(hm['mape'],2)) + '%  '
          'R2=' + str(hm['r2']) + '  '
          'F1=' + str(hm['f1']) + '  '
          'MAE=' + str(hm['mae']) + ' cpl  '
          "Theil's=" + str(hm['theils_u']))

    tscv = TimeSeriesSplit(n_splits=CV_FOLDS)
    cv_m, cv_r = [], []
    print('  CV (' + str(CV_FOLDS) + ' folds):')
    for fold, (tr, vl) in enumerate(tscv.split(X_raw), 1):
        sx_f = RobustScaler()
        mf   = Ridge(alpha=RIDGE_ALPHA)
        mf.fit(sx_f.fit_transform(X_raw[tr]), y_raw[tr],
               sample_weight=w_raw[tr])                       # pass weights
        fm = calc_metrics(y_raw[vl], mf.predict(sx_f.transform(X_raw[vl])))
        cv_m.append(fm['mape']); cv_r.append(fm['r2'])
        print('    Fold ' + str(fold) + ': MAPE=' + str(round(fm['mape'],2)) +
              '%  R2=' + str(round(fm['r2'],4)) +
              '  n=' + str(len(tr)) + '/' + str(len(vl)))

    cv = dict(mape_mean=round(float(np.mean(cv_m)),3),
              mape_std=round(float(np.std(cv_m)),3),
              r2_mean=round(float(np.mean(cv_r)),4),
              r2_std=round(float(np.std(cv_r)),4))
    print('  CV  MAPE=' + str(cv['mape_mean']) + '%(+/-' + str(cv['mape_std']) + ')  '
          'R2=' + str(cv['r2_mean']) + '(+/-' + str(cv['r2_std']) + ')')

    mfinal = Ridge(alpha=RIDGE_ALPHA)
    mfinal.fit(X_sc, y_raw, sample_weight=w_raw)              # pass weights
    print('  Final model trained on all ' + str(N) + ' observations')
    return mfinal, sx, None, hm, cv

#  FORECAST
def forecast_1week(df, model, sx, FEAT, fuel, excise_active):
    X_sc   = sx.transform(df[FEAT].iloc[-1:].astype(np.float32).values)
    pred   = float(model.predict(X_sc)[0])
    last   = float(df[fuel].iloc[-1])
    #pred   = np.clip(pred, last - 30, last + 30)
    pred       = np.clip(pred, last * 0.95, last * 1.05)
    change_pct = abs(pred - last) / last * 100
    if change_pct > 4.0:
        print(f'  [WARN] {fuel.upper()}: {pred-last:+.1f} cpl ({change_pct:.1f}%) - large move')
    fc_dt  = df['date'].iloc[-1] + timedelta(weeks=1)
    excise = EXCISE_CUT_START <= fc_dt.to_pydatetime() <= EXCISE_CUT_END
    return dict(
        fuel_type       = fuel.upper(),
        forecast_date   = fc_dt.strftime('%Y-%m-%d'),
        week_end        = (fc_dt + timedelta(days=6)).strftime('%Y-%m-%d'),
        pred_price_cpl  = round(pred, 1),
        last_actual_cpl = round(last, 1),
        change_cpl      = round(pred - last, 1),
        excise_active   = excise,
        data_source     = str(df['source'].iloc[-1]),
    )

#  SAVE .H5
def save_h5(model, sx, sy, features, fuel, metrics, cv):
    path = MDL_DIR / (fuel + '_price_model.h5')
    with h5py.File(path, 'w') as f:
        meta = f.create_group('metadata')
        meta.attrs['fuel_type'] = fuel
        meta.attrs['model_type'] = 'Ridge Regression - Price Forecast'
        meta.attrs['target_unit'] = 'cents per litre (cpl)'
        meta.attrs['trained_at'] = datetime.now().isoformat()
        meta.attrs['features'] = json.dumps(features)
        meta.attrs['n_features'] = len(features)
        meta.attrs['val_mape'] = metrics['mape']
        meta.attrs['val_r2'] = metrics['r2']
        meta.attrs['val_f1'] = metrics['f1']
        meta.attrs['val_mae_cpl'] = metrics['mae']
        meta.attrs['cv_mape_mean'] = cv['mape_mean']
        meta.attrs['cv_r2_mean'] = cv['r2_mean']
        meta.attrs['excise_cut_cpl'] = EXCISE_CUT_CPL
        cg = f.create_group('coef')
        cg.create_dataset('coef',      data=model.coef_.astype(np.float64))
        cg.create_dataset('intercept', data=np.array([model.intercept_]))
        rx = f.create_group('scaler_X')
        rx.create_dataset('center', data=sx.center_.astype(np.float64))
        rx.create_dataset('scale',  data=sx.scale_.astype(np.float64))
    size = path.stat().st_size / 1024
    print('  Saved -> ' + str(path) + '  (' + str(round(size,1)) + ' KB)')
    with open(MDL_DIR / ('scalers_' + fuel + '_price.pkl'), 'wb') as pf:
        pickle.dump({'sx': sx, 'features': features, 'fuel': fuel}, pf)

#  CHART  -- professional 4-panel layout
def save_chart(forecasts, all_metrics, df_raw):
    COLOURS = {
        'ulp91': '#2563EB', 'ulp95': '#16A34A', 'diesel': '#DC2626',
        'curr':  '#F59E0B', 'fore':  '#7C3AED',
        'bg':    '#F8FAFC', 'grid':  '#E2E8F0', 'text': '#1E293B',
        'sub':   '#64748B',
    }

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(COLOURS['bg'])
    gs  = gridspec.GridSpec(2, 4, figure=fig,
                            hspace=0.45, wspace=0.35,
                            left=0.06, right=0.97,
                            top=0.88, bottom=0.10)

    fuels      = ['ulp91', 'ulp95', 'diesel']
    fuel_names = {'ulp91': 'ULP 91', 'ulp95': 'ULP 95 Premium', 'diesel': 'Diesel'}

    #Top row: price history + forecast for each fuel
    for idx, fuel in enumerate(fuels):
        ax = fig.add_subplot(gs[0, idx])
        ax.set_facecolor('white')
        for spine in ax.spines.values():
            spine.set_color(COLOURS['grid'])

        df_fuel   = df_raw[df_raw[fuel].notna()].copy()
        df_recent = df_fuel.tail(52)
        colour    = COLOURS[fuel]

        ax.plot(df_recent['date'], df_recent[fuel],
                color=colour, linewidth=1.6, alpha=0.85, zorder=3)
        ax.fill_between(df_recent['date'], df_recent[fuel],
                        alpha=0.08, color=colour)

        fc        = forecasts[fuel]
        curr_date = df_recent['date'].iloc[-1]
        curr_val  = fc['last_actual_cpl']
        fore_date = pd.Timestamp(fc['forecast_date'])
        fore_val  = fc['pred_price_cpl']
        chg       = fc['change_cpl']

        ax.scatter([curr_date], [curr_val],
                   color=COLOURS['curr'], s=80, zorder=5)
        ax.scatter([fore_date], [fore_val],
                   color=COLOURS['fore'], s=100, marker='*', zorder=5)

        ax.annotate('', xy=(fore_date, fore_val), xytext=(curr_date, curr_val),
                    arrowprops=dict(arrowstyle='->', color=COLOURS['fore'],
                                   lw=1.8, connectionstyle='arc3,rad=0.15'))

        chg_col = '#DC2626' if chg > 0.5 else '#16A34A' if chg < -0.5 else COLOURS['sub']
        ax.annotate(str(fore_val) + ' cpl\n(' + ('+' if chg>=0 else '') + str(chg) + ')',
                    xy=(fore_date, fore_val), xytext=(8, 8),
                    textcoords='offset points', fontsize=8, fontweight='bold',
                    color=chg_col,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white',
                              ec=chg_col, alpha=0.85, lw=0.8))

        # Excise cut shading
        cut_s = pd.Timestamp('2026-04-01')
        cut_e = pd.Timestamp('2026-06-30')
        xlim_max = fore_date + timedelta(days=10)
        shade_s  = max(cut_s, df_recent['date'].iloc[0])
        shade_e  = min(cut_e, xlim_max)
        if shade_s < shade_e:
            ax.axvspan(shade_s, shade_e, alpha=0.06,
                       color='#F59E0B', zorder=1, label='Excise cut period')

        ax.set_title(fuel_names[fuel], fontsize=11, fontweight='bold',
                     color=COLOURS['text'], pad=6)
        ax.set_ylabel('cents per litre', fontsize=8, color=COLOURS['sub'])
        ax.tick_params(axis='both', labelsize=7, colors=COLOURS['sub'])
        ax.xaxis.set_major_formatter(
            plt.matplotlib.dates.DateFormatter('%b %y'))
        ax.xaxis.set_major_locator(
            plt.matplotlib.dates.MonthLocator(interval=2))
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
        ax.yaxis.grid(True, color=COLOURS['grid'], linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)

    #Top right: current vs forecast bar summary
    ax_sum = fig.add_subplot(gs[0, 3])
    ax_sum.set_facecolor('white')
    for spine in ax_sum.spines.values():
        spine.set_color(COLOURS['grid'])

    x      = np.arange(3)
    curr_v = [forecasts[f]['last_actual_cpl'] for f in fuels]
    fore_v = [forecasts[f]['pred_price_cpl']  for f in fuels]
    width  = 0.32

    ax_sum.bar(x - width/2, curr_v, width,
               label='Current',
               color=[COLOURS[f] for f in fuels], alpha=0.45, zorder=3)
    b2 = ax_sum.bar(x + width/2, fore_v, width,
                    label='Forecast',
                    color=[COLOURS[f] for f in fuels], alpha=0.9,
                    edgecolor='white', linewidth=0.5, zorder=3)

    for bar, val, curr in zip(b2, fore_v, curr_v):
        chg     = val - curr
        chg_col = '#DC2626' if chg > 0.5 else '#16A34A' if chg < -0.5 else '#64748B'
        ax_sum.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 2,
                    str(val) + '\n(' + ('+' if chg>=0 else '') + str(round(chg,1)) + ')',
                    ha='center', va='bottom', fontsize=7.5,
                    fontweight='bold', color=chg_col)

    ax_sum.set_xticks(x)
    ax_sum.set_xticklabels(['ULP91','ULP95','Diesel'], fontsize=9)
    ax_sum.set_ylabel('cents per litre', fontsize=8, color=COLOURS['sub'])
    ax_sum.set_title('Current vs Forecast', fontsize=11,
                     fontweight='bold', color=COLOURS['text'], pad=6)
    ax_sum.legend(fontsize=8, framealpha=0.7)
    ax_sum.yaxis.grid(True, color=COLOURS['grid'], linewidth=0.6, zorder=0)
    ax_sum.set_axisbelow(True)
    ax_sum.tick_params(labelsize=7, colors=COLOURS['sub'])

    # Bottom row: accuracy metrics panels
    metric_specs = [
        ('mape',     'MAPE %',     'lower = better',  '#2563EB'),
        ('r2',       'R\u00b2',    'higher = better',  '#16A34A'),
        ('f1',       'F1 Score',   'higher = better',  '#7C3AED'),
        ('theils_u', "Theil's U",  '<1 beats naive',   '#DC2626'),
    ]
    for col_idx, (mk, ml, note, bc) in enumerate(metric_specs):
        ax = fig.add_subplot(gs[1, col_idx])
        ax.set_facecolor('white')
        for spine in ax.spines.values():
            spine.set_color(COLOURS['grid'])

        vals = [all_metrics[f]['holdout'][mk] for f in fuels]
        bars = ax.bar(np.arange(3), vals, color=bc, alpha=0.75, zorder=3,
                      width=0.55, edgecolor='white', linewidth=0.5)

        # CV dots for MAPE and R2
        if mk == 'mape':
            cv_v = [all_metrics[f]['cv']['mape_mean'] for f in fuels]
            ax.scatter(np.arange(3), cv_v, color='black', s=35,
                       zorder=5, label='CV mean')
        elif mk == 'r2':
            cv_v = [all_metrics[f]['cv']['r2_mean'] for f in fuels]
            ax.scatter(np.arange(3), cv_v, color='black', s=35,
                       zorder=5, label='CV mean')

        # Reference lines
        if mk == 'theils_u':
            ax.axhline(1.0, color='#F59E0B', lw=1.2,
                       linestyle='--', label='Naive (U=1)', zorder=4)
        if mk == 'r2':
            ax.axhline(0.8, color='#F59E0B', lw=1.0,
                       linestyle=':', label='Retrain threshold', zorder=4)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(vals)*0.02,
                    str(round(val,3)), ha='center', va='bottom',
                    fontsize=8, fontweight='bold', color=COLOURS['text'])

        ax.set_xticks(np.arange(3))
        ax.set_xticklabels(['ULP91','ULP95','Diesel'], fontsize=8)
        ax.set_title(ml + '\n(' + note + ')', fontsize=9,
                     fontweight='bold', color=COLOURS['text'], pad=4)
        ax.yaxis.grid(True, color=COLOURS['grid'], linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=7, colors=COLOURS['sub'])
        if mk in ('theils_u', 'r2', 'f1', 'mape'):
            if mk in ('theils_u', 'r2') or (mk in ('mape','r2')):
                ax.legend(fontsize=7, framealpha=0.7)

    #Global title + legend
    fc_week = forecasts['ulp91']['forecast_date']
    fig.suptitle(
        'VIC Fuel Price Forecast  |  Week of ' + fc_week +
        '  |  Excise cut -32 cpl active to 30 Jun 2026',
        fontsize=13, fontweight='bold', color=COLOURS['text'], y=0.96)

    legend_items = [
        mpatches.Patch(color=COLOURS['curr'],       label='Current price'),
        mpatches.Patch(color=COLOURS['fore'],       label='Forecast price'),
        mpatches.Patch(color='#F59E0B', alpha=0.25, label='Excise cut period'),
    ]
    fig.legend(handles=legend_items, loc='lower center', ncol=3,
               fontsize=9, framealpha=0.85,
               bbox_to_anchor=(0.5, 0.01))

    out = RPT_DIR / 'price_forecast_chart.png'
    plt.savefig(out, dpi=150, bbox_inches='tight',
                facecolor=COLOURS['bg'], edgecolor='none')
    plt.close()
    print('[CHART] price_forecast_chart.png -> ' + str(RPT_DIR))
    return out

#  TEXT REPORTS
def write_reports(forecasts, all_metrics, live_source):
    ts = datetime.now().strftime('%d %B %Y %H:%M')
    W  = 68
    lines = []

    def hdr(t, c='='):
        lines.extend([c*W, '  ' + t, c*W])

    hdr('VIC FUEL PRICE FORECAST  --  1-WEEK AHEAD')
    lines += ['  Generated : ' + ts,
              '  Model     : Ridge Regression (alpha=' + str(RIDGE_ALPHA) + ')',
              '  CV method : TimeSeriesSplit (' + str(CV_FOLDS) + ' folds)',
              '  Data seed : ' + live_source,
              '  Target    : Melbourne retail pump price (cpl)', '']

    hdr('1-WEEK PRICE FORECAST', '-')
    lines += ['',
              '  ' + '{:<10} {:>12} {:>13} {:>8}  {}'.format(
                  'Fuel','Current cpl','Forecast cpl','Change','Week'),
              '  ' + '-'*W]
    for fc in forecasts.values():
        chg = fc['change_cpl']
        trend = 'v' if chg < -0.5 else ('^' if chg > 0.5 else '~')
        lines.append('  ' + '{:<10} {:>12.1f} {:>13.1f} {:>+8.1f}  {} {} to {}'.format(
            fc['fuel_type'], fc['last_actual_cpl'], fc['pred_price_cpl'],
            chg, trend, fc['forecast_date'], fc['week_end']))
    lines += ['',
              '  * Excise cut of 32 cpl active 1 Apr - 30 Jun 2026',
              '    Underlying prices are ~32 cpl higher without the cut']

    hdr('MODEL ACCURACY', '-')
    lines += ['',
              '  ' + '{:<8} {:>6} {:>8} {:>7} {:>7} {:>6} {:>6} {:>8} {:>9} {:>7}'.format(
                  'Fuel','MAPE%','MAE cpl','RMSE','R2','DA%','F1',
                  'TheilsU','CV_MAPE%','CV_R2'),
              '  ' + '-'*W]
    for fuel, m in all_metrics.items():
        hm = m['holdout']; cv = m['cv']
        lines.append('  ' + '{:<8} {:>6.2f} {:>8.1f} {:>7.1f} {:>7.4f} {:>6.1f} {:>6.4f} {:>8.4f} {:>8.2f}% {:>7.4f}'.format(
            fuel.upper(), hm['mape'], hm['mae'], hm['rmse'],
            hm['r2'], hm['da'], hm['f1'], hm['theils_u'],
            cv['mape_mean'], cv['r2_mean']))

    hdr('DATA SOURCES', '-')
    lines += [
        '', '  PRIMARY   Service Victoria Fair Fuel Open Data API',
        '            https://api.fuel.service.vic.gov.au/open-data/v1/fuel/prices',
        '            Header: x-consumer-id (your Consumer ID)',
        '            Free key: service.vic.gov.au/find-services/transport-and-driving/',
        '                      servo-saver/help-centre/servo-saver-public-api',
        '',
        '  FALLBACK  ACCC weekly PDF (drop in script folder, published Fridays)',
        '            accc.gov.au/consumers/petrol-and-fuel/petrol-prices-in-2026',
        '            Parser requires: pip install pdfplumber',
        '',
        '  TRAINING  APS quarterly Q1 2010 - Q4 2025 (64 quarters)',
        '', '=' * W]

    report = '\n'.join(lines)
    (RPT_DIR / 'price_forecast_report.txt').write_text(report, encoding='utf-8')
    print('[REPORT] price_forecast_report.txt  -> ' + str(RPT_DIR))

    rows = []
    for fuel, m in all_metrics.items():
        rows.append({'fuel': fuel.upper(),
                     **{'holdout_'+k: v for k,v in m['holdout'].items()},
                     **{'cv_'+k: v     for k,v in m['cv'].items()}})
    pd.DataFrame(rows).to_csv(RPT_DIR / 'price_training_metrics.csv', index=False)
    print('[REPORT] price_training_metrics.csv -> ' + str(RPT_DIR))

    class Safe(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.floating, np.float32, np.float64)): return float(o)
            if isinstance(o, (np.integer,)):                          return int(o)
            if isinstance(o, np.ndarray):                             return o.tolist()
            return super().default(o)

    payload = {
        'generated_at': datetime.now().isoformat(),
        'data_seed': live_source,
        'excise_cut_active': EXCISE_CUT_START <= datetime.now() <= EXCISE_CUT_END,
        'excise_cut_cpl': EXCISE_CUT_CPL,
        'excise_cut_ends': '2026-06-30',
        'forecasts': forecasts,
        'metrics': {k: {'holdout': v['holdout'], 'cv': v['cv']}
                              for k, v in all_metrics.items()},
    }
    (RPT_DIR / 'price_forecast.json').write_text(
        json.dumps(payload, indent=2, cls=Safe), encoding='utf-8')
    print('[REPORT] price_forecast.json        -> ' + str(RPT_DIR))

    print('\n' + report)
    return report

#  MAIN
def main():
    args = parse_args()
    print('=' * 68)
    print('  VIC FUEL PRICE FORECAST  --  ULP91 + ULP95 + DIESEL')
    print('  ' + datetime.now().strftime('%d %B %Y %H:%M'))
    print('=' * 68)

    live_prices = None
    live_source = 'APS quarterly + ACCC hardcoded anchor (22-Apr-2026)'
    anchor_date = None

    if not args.skip_live:
        print('\n[LIVE] Fetching current prices...')

        # Source 1: Servo Saver VIC API (primary)
        if args.api_key:
            print('  Trying Victoria Servo Saver API...')
            prices = fetch_servo_saver(args.api_key)
            if prices:
                live_prices = prices
                live_source = 'Victoria Servo Saver API (live station data)'

        # Source 2: ACCC PDF (only fallback)
        if not live_prices:
            print('  Servo Saver unavailable -- trying ACCC PDF fallback...')
            pdf_path = find_accc_pdf()
            if pdf_path:
                print('  Found PDF: ' + pdf_path.name)
                accc_data = parse_accc_pdf(pdf_path)
                if accc_data:
                    live_prices = accc_data
                    anchor_date = accc_data.get('date')
                    live_source = 'ACCC weekly PDF (' + pdf_path.name + ')'
            else:
                print('  No ACCC PDF found. Drop the PDF in the script folder.')
                print('  Download: accc.gov.au/consumers/petrol-and-fuel/')
                print('  Falling back to hardcoded anchor (22-Apr-2026)...')

    # Build dataset
    df_raw = build_price_dataset(live_prices, anchor_date)

    # Train 3 models
    fuels = ['ulp91', 'ulp95', 'diesel']
    all_metrics = {}
    forecasts = {}
    excise_active = EXCISE_CUT_START <= datetime.now() <= EXCISE_CUT_END

    for fuel in fuels:
        print('\n' + '='*68 + '\n  FUEL: ' + fuel.upper() + '\n' + '='*68)
        df, FEAT, y_col = engineer_features(df_raw, fuel)
        model, sx, sy, hm, cv = train_model(df, FEAT, y_col, fuel)
        all_metrics[fuel] = {'holdout': hm, 'cv': cv}
        save_h5(model, sx, sy, FEAT, fuel, hm, cv)
        fc = forecast_1week(df, model, sx, FEAT, fuel, excise_active)
        forecasts[fuel] = fc
        print('\n[FORECAST] ' + fuel.upper() + ': ' +
              str(fc['last_actual_cpl']) + ' -> ' + str(fc['pred_price_cpl']) +
              ' cpl (' + ('+' if fc['change_cpl']>=0 else '') +
              str(fc['change_cpl']) + ')  week of ' + fc['forecast_date'])

    # Reports + chart
    write_reports(forecasts, all_metrics, live_source)
    save_chart(forecasts, all_metrics, df_raw)

    # Verify .h5 files
    print('\n[VERIFY]')
    for fuel in fuels:
        path = MDL_DIR / (fuel + '_price_model.h5')
        with h5py.File(path, 'r') as f:
            coef  = f['coef/coef'][:]
            feats = json.loads(f['metadata'].attrs['features'])
            mape  = f['metadata'].attrs['val_mape']
        print('  ' + fuel + '_price_model.h5  (' +
              str(round(path.stat().st_size/1024, 1)) + ' KB)  ' +
              'coef:' + str(coef.shape) + '  features:' + str(len(feats)) +
              '  MAPE:' + str(round(mape,2)) + '%  [OK]')

    print('\n  ALL DONE have a good day Fuel hunting!!')

if __name__ == '__main__':
    main()