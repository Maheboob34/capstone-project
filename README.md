# Rail Cascade Monitor

Interactive dashboard for **delay cascades in the German regional and long-distance rail network**
(Berlin-centred). It replays the July to September 2026 test period and shows, for every primary delay, the
predicted probability that other trains will also be delayed at the same station within 30 minutes.

Authors: Maheboob Patel, Mohammad Akbar Ali. Supervisor: Prof. Humera Noor Minhas.
M.Sc. Data Science, University of Europe for Applied Sciences, Potsdam.

## Pages
- **Risk map:** pick a prediction target, a day and hours; see station bubbles coloured by predicted risk, alert
  statistics (precision and recall for a chosen threshold), and the highest-risk events.
- **Hub amplification:** stations where delays are followed by more knock-on delays than chance predicts.
- **Model performance:** comparison with baselines (with 95% intervals), reliability of the probabilities,
  and which feature groups matter.
- **About:** definitions, method and limitations.

## Run it
1. Create the data files in Colab with `12_export_dashboard_data.py`, download `rail-cascade-monitor-data.zip`
   and unzip it into the `data/` folder of this project. You should have:
   `data/events.parquet`, `data/hubs.csv`, `data/metrics.csv`, `data/ablation.csv`.
2. Install and start:
   ```
   pip install -r requirements.txt
   streamlit run app.py
   ```

## Deploy for free (public link)
**Streamlit Community Cloud:** push this folder (including `data/`) to a GitHub repository, then create a new app
at share.streamlit.io, choose the repository and set the main file to `app.py`.

**Hugging Face Spaces:** create a Space with the Streamlit SDK and upload the same files.

The data files are small (a few MB), so they can live in the repository.

## Files
- `app.py` the dashboard
- `logic.py` data logic without Streamlit (easy to test)
- `.streamlit/config.toml` dark theme
- `data/` exported results (not included until you run the export script)

## Data and credit
Train timetable and delay data: Deutsche Bahn, published under CC BY 4.0 (via a public Hugging Face dataset).
Station coordinates: the db-stations collection, built from Deutsche Bahn open data (CC BY 4.0).
Please keep this credit in any public version of the dashboard.

## Limitations
Cascades are inferred from delays that co-occur in space and time (association, not proven causation).
Probabilities run low on unusually disrupted days. There is no weather or track-works data.
This is a decision-support demonstration, not an operational system.
