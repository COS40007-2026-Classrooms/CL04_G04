#  VIC FUEL FORECAST  --  RETRAINING PIPELINE
#  Runs weekly (Wednesday morning via GitHub Actions cron)
#  FLOW:
#    1. Fetch live prices from Fair Fuel API
#    2. Compare last week's forecast against this week's actuals -> accuracy check
#    3. If accuracy >= threshold: retrain model, save new weights
#    4. Generate suburb forecasts
#    5. Write metrics to JSON (for GitHub Actions to inspect and badge)
#    6. Exit code 0 = success/passed threshold, 1 = below threshold
#
#  THRESHOLD LOGIC:
#    MAPE <= MAPE_THRESHOLD (default 3.0%) -> model passes -> retrain + deploy
#    MAPE >  MAPE_THRESHOLD                -> model fails  -> keep old weights,
#                                             alert, do NOT update forecasts
#
#  USAGE:
#    python retrain_pipeline.py --api-key YOUR_KEY
#    python retrain_pipeline.py --api-key YOUR_KEY --mape-threshold 2.5
#    python retrain_pipeline.py --api-key YOUR_KEY --force-retrain   # skip check

import os, sys, json, argparse, warnings
warnings.filterwarnings('ignore')

import pandas as pd
pd.options.future.infer_string = False
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR   = SCRIPT_DIR.parent
MDL_DIR = ROOT_DIR / 'models'; MDL_DIR.mkdir(exist_ok=True)
RPT_DIR = ROOT_DIR / 'reports';  RPT_DIR.mkdir(exist_ok=True)
DAT_DIR = ROOT_DIR / 'data'; DAT_DIR.mkdir(exist_ok=True)

# Accuracy thresholds -- model must beat ALL of these to pass
DEFAULT_THRESHOLDS = {
    'mape_max': 3.0,   # MAPE must be <= this (%)
    'da_min': 55.0,  # Directional accuracy must be >= this (%)
    'mae_max_cpl': 6.0,   # MAE must be <= this (cents per litre)
}

METRICS_FILE = DAT_DIR / 'pipeline_metrics.json'
FORECAST_HIST = DAT_DIR / 'forecast_history.csv'   # stores weekly forecasts for backtesting
ACTUALS_HIST = DAT_DIR / 'actuals_history.csv'     # stores weekly actuals for backtesting


def parse_args():
    p = argparse.ArgumentParser(description='VIC Fuel Forecast Retraining Pipeline')
    p.add_argument('--api-key', default=os.environ.get('SERVO_SAVER_API_KEY',''),
                   help='Service Victoria Fair Fuel API Consumer ID')
    p.add_argument('--mape-threshold', type=float, default=DEFAULT_THRESHOLDS['mape_max'],
                   help=f'MAPE threshold %% (default {DEFAULT_THRESHOLDS["mape_max"]})')
    p.add_argument('--da-threshold', type=float, default=DEFAULT_THRESHOLDS['da_min'],
                   help=f'Directional accuracy threshold %% (default {DEFAULT_THRESHOLDS["da_min"]})')
    p.add_argument('--mae-threshold', type=float, default=DEFAULT_THRESHOLDS['mae_max_cpl'],
                   help=f'MAE threshold cpl (default {DEFAULT_THRESHOLDS["mae_max_cpl"]})')
    p.add_argument('--force-retrain', action='store_true',
                   help='Skip accuracy check and retrain unconditionally')
    p.add_argument('--dry-run', action='store_true',
                   help='Run checks only, do not write model files')
    return p.parse_args()

#  STEP 1: FETCH LIVE PRICES
def fetch_live_prices(api_key):
    """Fetch current week's actual prices from the API."""
    import uuid, urllib.request, urllib.error
    if not api_key:
        print('  [WARN] No API key -- cannot fetch live prices')
        return None

    headers = {
        'x-consumer-id':   api_key,
        'x-transactionid': str(uuid.uuid4()),
        'User-Agent': 'VIC-Fuel-Forecast/1.0',
        'Accept': 'application/json',
    }
    try:
        url = 'https://api.fuel.service.vic.gov.au/open-data/v1/fuel/prices'
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())

        prices = {}
        by_type = {}
        for station in data.get('fuelPriceDetails', []):
            for fp in station.get('fuelPrices', []):
                ft    = fp.get('fuelType','')
                price = fp.get('price')
                avail = fp.get('isAvailable', True)
                if price and avail:
                    try:
                        by_type.setdefault(ft, []).append(float(price))
                    except (TypeError, ValueError):
                        pass

        fuel_map = {'U91':'ulp91','P95':'ulp95','DSL':'diesel'}
        outlier  = {'ulp91':(120,350),'ulp95':(130,370),'diesel':(150,420)}

        for code, name in fuel_map.items():
            vals = [v for v in by_type.get(code,[]) if outlier[name][0] <= v <= outlier[name][1]]
            if vals:
                prices[name] = round(float(np.median(vals)), 1)

        if prices:
            print(f'  Live prices: ULP91={prices.get("ulp91","N/A")}  '
                  f'ULP95={prices.get("ulp95","N/A")}  '
                  f'Diesel={prices.get("diesel","N/A")} cpl')
        return prices if prices else None

    except Exception as e:
        print(f'  [ERROR] fetch_live_prices: {e}')
        return None

#  STEP 2: ACCURACY CHECK -- compare last forecast vs this week's actuals
def check_accuracy(live_prices, thresholds):
    """
    Load last week's forecast from forecast_history.csv.
    Compare against this week's live prices (the 'actuals').
    Returns (passed: bool, metrics: dict).
    """
    if not FORECAST_HIST.exists():
        print('  No forecast history yet -- skipping accuracy check (first run)')
        return True, {}

    hist = pd.read_csv(FORECAST_HIST)
    hist['forecast_date'] = pd.to_datetime(hist['forecast_date'])

    # Get the most recent forecast that should now be verifiable
    # A forecast made last Wednesday predicts the price for THIS week
    today = datetime.now()
    cutoff = today - timedelta(days=6)
    verifiable = hist[hist['forecast_date'] <= cutoff].copy()

    if len(verifiable) == 0:
        print('  No verifiable forecasts yet -- skipping accuracy check')
        return True, {}

    # Use the most recent verifiable forecast
    last = verifiable.sort_values('forecast_date').iloc[-1]

    metrics = {'forecast_date': str(last['forecast_date'].date()), 'fuels': {}}
    all_passed = True

    print(f'\n  Backtesting forecast from {last["forecast_date"].date()}:')
    print(f'  {"Fuel":<10} {"Forecast":>10} {"Actual":>10} {"Error cpl":>10} {"APE%":>8} {"Dir":>5}')
    print('  ' + '-' * 55)

    for fuel in ['ulp91', 'ulp95', 'diesel']:
        fc_col = f'{fuel}_forecast_cpl'
        if fc_col not in last or fuel not in (live_prices or {}):
            continue

        forecast = float(last[fc_col])
        actual   = float(live_prices[fuel])
        error    = actual - forecast
        ape      = abs(error) / actual * 100

        # Directional accuracy: did we predict the right direction of change?
        prev_col = f'{fuel}_current_cpl'
        if prev_col in last:
            prev_price = float(last[prev_col])
            pred_dir   = 'up' if forecast > prev_price else 'down'
            act_dir    = 'up' if actual > prev_price   else 'down'
            da_correct = pred_dir == act_dir
        else:
            da_correct = None

        da_str = ('✓' if da_correct else '✗') if da_correct is not None else '?'
        print(f'  {fuel.upper():<10} {forecast:>10.1f} {actual:>10.1f} '
              f'{error:>+10.1f} {ape:>8.2f}% {da_str:>5}')

        metrics['fuels'][fuel] = {
            'forecast_cpl':  forecast,
            'actual_cpl':    actual,
            'error_cpl':     round(error, 2),
            'ape_pct':       round(ape, 3),
            'da_correct':    da_correct,
        }

    # Compute aggregate metrics across all fuels
    apes  = [m['ape_pct']   for m in metrics['fuels'].values()]
    das   = [m['da_correct'] for m in metrics['fuels'].values() if m['da_correct'] is not None]
    maes  = [abs(m['error_cpl']) for m in metrics['fuels'].values()]

    if not apes:
        return True, metrics

    mape   = float(np.mean(apes))
    da_pct = float(np.mean(das) * 100) if das else 100.0
    mae    = float(np.mean(maes))

    metrics['aggregate'] = {
        'mape_pct':    round(mape, 3),
        'da_pct':      round(da_pct, 1),
        'mae_cpl':     round(mae, 2),
    }

    print(f'\n  Aggregate: MAPE={mape:.2f}%  DA={da_pct:.1f}%  MAE={mae:.2f} cpl')
    print(f'  Thresholds: MAPE<={thresholds["mape_max"]}%  DA>={thresholds["da_min"]}%  MAE<={thresholds["mae_max_cpl"]} cpl')

    # Evaluate against thresholds
    checks = {
        'mape': (mape <= thresholds['mape_max'],
                 f'MAPE {mape:.2f}% <= {thresholds["mape_max"]}%'),
        'da':   (da_pct >= thresholds['da_min'],
                 f'DA {da_pct:.1f}% >= {thresholds["da_min"]}%'),
        'mae':  (mae <= thresholds['mae_max_cpl'],
                 f'MAE {mae:.2f} <= {thresholds["mae_max_cpl"]} cpl'),
    }

    print('\n  Threshold checks:')
    for name, (passed, msg) in checks.items():
        icon = '[PASS]' if passed else '[FAIL]'
        print(f'    {icon} {msg}')
        if not passed:
            all_passed = False

    metrics['threshold_checks'] = {k: v[0] for k, v in checks.items()}
    metrics['all_passed'] = all_passed
    return all_passed, metrics

#  STEP 3: RETRAIN MODELS
def retrain_models(live_prices, dry_run=False, api_key=''):
    """
    Re-run fuel_price_forecast.py with fresh live prices.
    Imports and calls the training function directly.
    Returns True on success.
    """
    print('\n[STEP 3] Retraining models...')
    try:
        # Import the main forecast module (it must be in the same directory)
        import importlib.util
        spec = importlib.util.spec_from_file_location('finalModel',SCRIPT_DIR / 'finalModel.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        if dry_run:
            print('  [DRY RUN] Skipping actual model write')
            return True

        # The module's main() will use the live prices already fetched
        original_argv = sys.argv[:]
        sys.argv = ['finalModel.py', '--api-key', api_key]
        mod.main()
        sys.argv = original_argv
        print('  Models retrained successfully')
        return True

    except Exception as e:
        print(f'  [ERROR] Retrain failed: {e}')
        import traceback; traceback.print_exc()
        return False

# STEP 4: SAVE FORECAST TO HISTORY
def save_forecast_to_history(live_prices):
    """Append this week's forecast to forecast_history.csv for future backtesting."""
    forecast_json = RPT_DIR / 'price_forecast.json'
    if not forecast_json.exists():
        print('  [WARN] price_forecast.json not found -- skipping history save')
        return

    with open(forecast_json) as f:
        payload = json.load(f)

    row = {'forecast_date': datetime.now().strftime('%Y-%m-%d'), 'run_timestamp': datetime.now().isoformat()}

    for fuel in ['ulp91', 'ulp95', 'diesel']:
        fc = payload.get('forecasts', {}).get(fuel, {})
        row[f'{fuel}_current_cpl']  = fc.get('last_actual_cpl')
        row[f'{fuel}_forecast_cpl'] = fc.get('pred_price_cpl')
        row[f'{fuel}_change_cpl']   = fc.get('change_cpl')

    new_row = pd.DataFrame([row])

    if FORECAST_HIST.exists():
        hist = pd.read_csv(FORECAST_HIST)
        # Avoid duplicate runs on the same date
        hist = hist[hist['forecast_date'] != row['forecast_date']]
        hist = pd.concat([hist, new_row], ignore_index=True)
    else:
        hist = new_row

    hist.to_csv(FORECAST_HIST, index=False)
    print(f'  Forecast saved to history ({len(hist)} weeks on record)')

# STEP 5: SAVE ACTUALS TO HISTORY
def save_actuals_to_history(live_prices):
    """Append this week's actual prices to actuals_history.csv."""
    if not live_prices:
        return

    row = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'ulp91_actual': live_prices.get('ulp91'),
        'ulp95_actual': live_prices.get('ulp95'),
        'diesel_actual': live_prices.get('diesel'),
    }
    new_row = pd.DataFrame([row])

    if ACTUALS_HIST.exists():
        hist = pd.read_csv(ACTUALS_HIST)
        hist = hist[hist['date'] != row['date']]
        hist = pd.concat([hist, new_row], ignore_index=True)
    else:
        hist = new_row

    hist.to_csv(ACTUALS_HIST, index=False)
    print(f'  Actuals saved to history ({len(hist)} weeks on record)')




# STEP 6: WRITE PIPELINE METRICS JSON (GitHub Actions)
def write_pipeline_metrics(accuracy_metrics, retrain_attempted, retrain_success,
                            threshold_passed, thresholds, dry_run):

    payload = {
        'run_timestamp': datetime.now().isoformat(),
        'threshold_passed': threshold_passed,
        'retrain_attempted':  retrain_attempted,
        'retrain_success': retrain_success,
        'dry_run': dry_run,
        'thresholds': thresholds,
        'accuracy': accuracy_metrics,
    }

    METRICS_FILE.write_text(json.dumps(payload, indent=2, default=str))
    print(f'\n  Pipeline metrics -> {METRICS_FILE}')

    # Also write a compact badge-ready summary
    if accuracy_metrics.get('aggregate'):
        agg = accuracy_metrics['aggregate']
        badge = {
            'mape': f'{agg["mape_pct"]:.2f}%',
            'da': f'{agg["da_pct"]:.1f}%',
            'mae':f'{agg["mae_cpl"]:.2f} cpl',
            'status': 'passing' if threshold_passed else 'failing',
        }
        (DAT_DIR / 'badge_summary.json').write_text(json.dumps(badge, indent=2))

# MAIN
def main():
    args = parse_args()
    W = 67 #:)
    thresholds = {
        'mape_max': args.mape_threshold,
        'da_min': args.da_threshold,
        'mae_max_cpl': args.mae_threshold,
    }

    print('=' * W)
    print('  VIC FUEL FORECAST  --  RETRAINING PIPELINE')
    print(f' {datetime.now().strftime("%d %B %Y %H:%M")}')
    print('=' * W)
    print(f' Thresholds: MAPE<={thresholds["mape_max"]}%  '
          f'DA>={thresholds["da_min"]}%  MAE<={thresholds["mae_max_cpl"]} cpl')
    if args.force_retrain: print('  Mode: FORCE RETRAIN (skip accuracy check)')
    if args.dry_run: print('  Mode: DRY RUN (no model writes)')

    # Step 1: Fetch live prices (this week's actuals)
    print(f'\n[STEP 1] Fetching live prices...')
    live_prices = fetch_live_prices(args.api_key)

    if live_prices:
        save_actuals_to_history(live_prices)

    # Step 2: Accuracy check against last week's forecast
    print(f'\n[STEP 2] Accuracy check...')
    if args.force_retrain:
        threshold_passed = True
        accuracy_metrics = {'note': 'force_retrain -- check skipped'}
        print('  Skipped (--force-retrain)')
    else:
        threshold_passed, accuracy_metrics = check_accuracy(live_prices, thresholds)

    # Step 3: Retrain if threshold passed
    retrain_attempted = False
    retrain_success   = False

    if threshold_passed:
        print(f'\n[STEP 3] Threshold PASSED -- retraining...')
        retrain_attempted = True
        retrain_success = retrain_models(live_prices, dry_run=args.dry_run, api_key=args.api_key)

        if retrain_success and not args.dry_run:
            # Step 4: Save new forecast to history for next week's check
            print('\n[STEP 4] Saving forecast to history...')
            save_forecast_to_history(live_prices)
    else:
        print(f'\n[STEP 3] Threshold FAILED -- keeping existing models')
        print('The current model weights are preserved.')
        print('Investigate accuracy degradation before retraining.')

    # Step 5: Write pipeline metrics for GitHub Actions
    write_pipeline_metrics(
        accuracy_metrics, retrain_attempted, retrain_success,
        threshold_passed, thresholds, args.dry_run)

    # Exit code for GitHub Actions
    # 0 = everything OK (pass or first run)
    # 1 = threshold failed (CI will flag the run as failed/warning)
    print('\n' + '=' * W)
    if threshold_passed:
        print(' PIPELINE STATUS: PASSED')
    else:
        print(' PIPELINE STATUS: THRESHOLD NOT MET -- models NOT updated')
    print('=' * W)

    sys.exit(0 if threshold_passed else 1)

if __name__ == '__main__':
    main()
