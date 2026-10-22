# Streamlit app

Interactive front end for the project: overview, process audit, customer risk (with a campaign
simulator), customer segments, customer lookup, model evidence and method.

## Run locally
```bash
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

## Deploy on Streamlit Community Cloud (free)
1. Go to https://share.streamlit.io and sign in with GitHub.
2. **Create app** → *Deploy a public app from GitHub*.
3. Repository `ronitnoronha/retail-revenue-leakage-analytics`, branch `main`,
   main file path `app/streamlit_app.py`.
4. Optional: choose a custom subdomain (e.g. `retail-leakage-ronit`) → **Deploy**.

Dependencies come from `app/requirements.txt` (next to the app file).

## Refreshing the data
The app reads only the small CSVs in `app/data/`. To rebuild them after re-running the pipeline:
```bash
python dashboard/build_data.py
python app/build_app_data.py
```
