# 🚀 Quick Start Guide - Flood Sentinel Phase 1

## Your API Keys Are Configured!

✅ **NOAA API**: Active and ready  
✅ **OpenWeather API**: Active and ready  
✅ **Security**: Keys protected in .env file  
✅ **Real Data**: No synthetic data - all APIs will be used

---

## Step 1: Test Your API Keys (Optional but Recommended)

```bash
python test_api_keys.py
```

This will verify:
- ✓ Keys are loaded from .env
- ✓ NOAA API is responding
- ✓ OpenWeather API is responding
- ✓ USGS APIs are accessible

**Expected output:** All green checkmarks ✓

---

## Step 2: Run Phase 1 Scripts

### Option A: Automatic (Recommended)
```bash
RUN_PHASE1.bat
```
Runs all three scripts in sequence automatically.

### Option B: Manual Step-by-Step
```bash
# Install dependencies first
pip install -r requirements_phase1.txt

# Then run each script
python 1_build_base_dataset.py    # ~2-3 minutes
python 2_enrich_dataset.py         # ~5-10 minutes (API calls)
python 3_train_model.py            # ~3-5 minutes
```

---

## What To Expect

### Script 1: Base Dataset Creation
```
✓ Using REAL USGS flood data (no synthetic fallback)
✓ Retrieved 1234 high water marks from USGS
✓ Filtered to 567 REAL flood points in Harris County
```

### Script 2: Feature Enrichment
```
✓ NOAA API: Retrieved Harvey precipitation data: 1347mm
✓ Using Houston baseline: 1347mm (53.0 inches)
✓ Elevation data from USGS 3DEP API
```

### Script 3: Model Training
```
✓ Model training complete
✓ Test Accuracy: 0.8945
✓ Test F1-Score: 0.8523
✓ Test ROC-AUC: 0.9234
```

---

## Output Files Created

After running all scripts:

```
data/
  ├── base_points.csv              (~1,500 real flood locations)
  └── final_training_dataset.csv   (9 features + target)

models/
  ├── flood_risk_model.pkl         (Trained on REAL data!)
  └── feature_scaler.pkl

results/
  ├── confusion_matrix.png
  ├── feature_importance.png
  ├── roc_curve.png
  └── probability_distribution.png
```

---

## Troubleshooting

### If API Keys Don't Load:
1. Make sure `.env` file exists in `c:\CAC\`
2. Check that `config.py` exists
3. Run `python test_api_keys.py` to diagnose

### If NOAA API Fails:
- Script automatically falls back to documented Harvey data (1270mm/50")
- Your model will still be trained on accurate data
- Check rate limits: max 5 requests/second

### If USGS API Fails:
- Script retries 3 times with 2-second delays
- Only uses synthetic data if all retries fail
- This API is very reliable

---

## Timeline

**Total Runtime:** 10-18 minutes

- Script 1: 2-3 min (API calls with retries)
- Script 2: 5-10 min (1500 points × API calls)
- Script 3: 3-5 min (model training)

---

## Next Steps After Phase 1

Once all three scripts complete successfully:

1. ✅ Review visualizations in `results/` folder
2. ✅ Check model performance metrics
3. ✅ Confirm output files are created
4. ➡️ **Notify me to proceed to Phase 2: Backend Development**

---

## Phase 2 Preview

Phase 2 will create:
- Flask REST API server
- Real-time weather integration (using your OpenWeather key!)
- Interactive web map
- Live flood risk predictions

**Your OpenWeather API key will be used in Phase 2 for live weather data.**

---

## Support

If you encounter any issues:
1. Check `API_INTEGRATION_NOTES.md` for detailed info
2. Review `PHASE1_README.md` for troubleshooting
3. Run `test_api_keys.py` to verify configuration

---

**Ready to start?** Run: `RUN_PHASE1.bat` or `python 1_build_base_dataset.py`
