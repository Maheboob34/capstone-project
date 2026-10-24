"""Rail Cascade Monitor: an interactive dashboard for delay cascades in the German rail network.

Run locally:  streamlit run app.py
Needs the files in ./data (created by 12_export_dashboard_data.py).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from logic import (
    TARGETS,
    alert_stats,
    busiest_day,
    hourly_summary,
    pct,
    reliability,
    station_summary,
    window_events,
)

DATA = Path(__file__).parent / "data"
ACCENT = "#2dd4bf"
GREY = "#64748b"
MAP_STYLE = "carto-darkmatter"

st.set_page_config(page_title="Rail Cascade Monitor", page_icon="🚆", layout="wide")
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; }
    h1 { letter-spacing: -0.02em; }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.09);
        padding: 14px 18px; border-radius: 14px;
    }
    div[data-testid="stMetricLabel"] p { font-size: 0.85rem; opacity: 0.75; }
    .note { font-size: 0.85rem; opacity: 0.7; }
    </style>
    """,
    unsafe_allow_html=True,
)

REQUIRED = ["events.parquet", "hubs.csv", "metrics.csv", "ablation.csv"]
missing = [f for f in REQUIRED if not (DATA / f).exists()]
if missing:
    st.error(
        f"Missing data files in {DATA}: {', '.join(missing)}. "
        "Run 12_export_dashboard_data.py and copy its output into the data folder."
    )
    st.stop()


@st.cache_data
def load_events() -> pd.DataFrame:
    df = pd.read_parquet(DATA / "events.parquet")
    df["actual_time"] = pd.to_datetime(df["actual_time"])
    df["day"] = df["actual_time"].dt.date
    return df


@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name)


def make_map(df, size, color, color_label, hover, range_color=None, scale="YlOrRd", height=580):
    """Dark map with one bubble per station (works with old and new Plotly map APIs)."""
    new_api = hasattr(px, "scatter_map")
    fn = px.scatter_map if new_api else px.scatter_mapbox
    fig = fn(
        df,
        lat="lat",
        lon="lon",
        size=size,
        color=color,
        hover_name="station_name",
        hover_data=hover,
        color_continuous_scale=scale,
        range_color=range_color,
        size_max=26,
        zoom=4.6,
        center={"lat": 51.2, "lon": 10.3},
        height=height,
    )
    fig.update_layout(**{("map_style" if new_api else "mapbox_style"): MAP_STYLE})
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), coloraxis_colorbar=dict(title=color_label))
    return fig


events = load_events()

st.title("🚆 Rail Cascade Monitor")
st.caption(
    "Predicting how train delays spread through the German regional and long-distance rail network "
    "(Berlin-centred). Data: Deutsche Bahn, CC BY 4.0. Replay of the test period, July to September 2026."
)

tab_map, tab_hubs, tab_perf, tab_about = st.tabs(
    ["🗺️ Risk map", "🏛️ Hub amplification", "📊 Model performance", "ℹ️ About"]
)

# ---------------------------------------------------------------- Risk map
with tab_map:
    c1, c2, c3 = st.columns([2, 1.4, 2])
    target_label = c1.selectbox("What to predict", list(TARGETS))
    days = sorted(events["day"].unique())
    default_day = busiest_day(events)
    day = c2.date_input("Day", value=default_day, min_value=days[0], max_value=days[-1])
    if isinstance(day, (list, tuple)):
        day = day[0] if day else default_day
    hours = c3.slider("Hours of the day", 0, 23, (6, 22))
    thr = st.slider(
        "Alert threshold: raise an alert when the predicted probability is at least",
        0.05, 0.95, 0.50, 0.05,
    )

    p_col, y_col = TARGETS[target_label]["p"], TARGETS[target_label]["y"]
    win = window_events(events, day, hours)

    if win.empty:
        st.info("No primary delays in this window. Try another day or a wider range of hours.")
    else:
        stats = alert_stats(win, p_col, y_col, thr)
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Primary delays", f"{stats['events']:,}", help="Delays of 10+ minutes that appeared at a stop")
        k2.metric("Alerts raised", f"{stats['alerts']:,}", help=f"Events with predicted risk of at least {thr:.0%}")
        k3.metric("Alert precision", pct(stats["precision"]), help="Share of alerts that became real cascades")
        k4.metric("Cascades caught", pct(stats["recall"]), help="Share of real cascades that had raised an alert")
        k5.metric("Base rate", pct(stats["base_rate"]), help="Share of all primary delays that became cascades")

        summ = station_summary(win, p_col, y_col)
        if summ.empty:
            st.info("No stations with known coordinates in this window.")
        else:
            fig = make_map(
                summ,
                size="events",
                color="mean_risk",
                color_label="Predicted<br>risk",
                hover={"events": True, "mean_risk": ":.0%", "observed": ":.0%", "lat": False, "lon": False},
                range_color=(0, 1),
            )
            fig.update_layout(coloraxis_colorbar=dict(title="Predicted<br>risk", tickformat=".0%"))
            st.plotly_chart(fig)
            st.markdown(
                "<p class='note'>Bubble size = number of primary delays at the station in the window; "
                "colour = average predicted probability that a delay there becomes a cascade; "
                "hover to see the share that really did.</p>",
                unsafe_allow_html=True,
            )

        left, right = st.columns([1.3, 1])
        with left:
            hourly = hourly_summary(win, y_col)
            figh = go.Figure()
            figh.add_bar(x=hourly["hour"], y=hourly["events"], name="Primary delays", marker_color=GREY)
            figh.add_bar(x=hourly["hour"], y=hourly["cascades"], name="Became cascades", marker_color=ACCENT)
            figh.update_layout(
                barmode="overlay", height=300, margin=dict(l=0, r=0, t=40, b=0),
                title="Primary delays by hour", xaxis_title="Hour of day", yaxis_title="Events",
                legend=dict(orientation="h", y=1.18),
            )
            st.plotly_chart(figh)
        with right:
            st.markdown("#### How to read this")
            st.markdown(
                "Each primary delay gets a predicted probability that **other trains will also be delayed "
                "at the same station within 30 minutes**. Lower the threshold to catch more cascades "
                "(more false alarms); raise it for fewer, more reliable alerts."
            )

        st.markdown("#### Highest-risk events in the window")
        feed = win.sort_values(p_col, ascending=False).head(300).copy()
        feed["risk"] = (feed[p_col] * 100).round(0).astype(int)
        feed["outcome"] = np.where(feed[y_col], "🔴 cascade", "⚪ no cascade")
        feed = feed[["actual_time", "station_name", "train_type", "delay_min", "risk", "outcome", "knockon_real"]]
        feed.columns = ["Time", "Station", "Train type", "Delay (min)", "Predicted risk (%)", "Outcome", "Other trains delayed"]
        st.dataframe(
            feed,
            hide_index=True,
            column_config={
                "Time": st.column_config.DatetimeColumn("Time", format="HH:mm"),
                "Predicted risk (%)": st.column_config.ProgressColumn(
                    "Predicted risk (%)", min_value=0, max_value=100, format="%d"
                ),
            },
        )

# ---------------------------------------------------------------- Hubs
with tab_hubs:
    hubs = load_csv("hubs.csv")
    st.subheader("Which stations amplify delays?")
    st.markdown(
        "After a primary delay, we count how many other trains are newly delayed at the same station within "
        "30 minutes and compare that with the number **expected by chance** (the same station and time of week, "
        "one week earlier and later). The difference is the *excess*."
    )
    mode = st.radio(
        "Rank by",
        ["Total excess knock-on trains", "Excess per primary delay (with 95% interval)"],
        horizontal=True,
    )
    per_event = mode.startswith("Excess per")
    col = "excess_per_event" if per_event else "total_excess"
    n_show = st.slider("Stations shown", 5, 30, 15)
    top = hubs.sort_values(col, ascending=False).head(n_show).sort_values(col)
    err = None
    if per_event:
        err = dict(
            type="data",
            symmetric=False,
            array=(top["excess_per_event_high"] - top[col]).clip(lower=0),
            arrayminus=(top[col] - top["excess_per_event_low"]).clip(lower=0),
        )
    figb = go.Figure(go.Bar(x=top[col], y=top["station_name"], orientation="h", marker_color=ACCENT, error_x=err))
    figb.update_layout(
        height=max(360, 26 * n_show + 100), margin=dict(l=0, r=10, t=10, b=0),
        xaxis_title="Excess knock-on trains per primary delay" if per_event else "Total excess knock-on trains",
    )
    st.plotly_chart(figb)

    geo_hubs = hubs.dropna(subset=["lat", "lon"]).copy()
    geo_hubs["size"] = geo_hubs["total_excess"].clip(lower=1)
    st.plotly_chart(
        make_map(
            geo_hubs, size="size", color="excess_per_event", color_label="Excess per<br>delay",
            hover={"events": True, "total_excess": ":.0f", "excess_per_event": ":.2f",
                   "lat": False, "lon": False, "size": False},
            height=460,
        )
    )
    st.markdown(
        "<p class='note'>Caveats: the data is Berlin-centred, so Berlin stations are over-represented. "
        "Stations were selected because they scored high, so the intervals are somewhat optimistic. "
        "Excess means <i>association</i> (it can reflect track or platform capacity), not proven causation. "
        "Ratios to the expected value are unstable when the expected value is small, so we rank by excess.</p>",
        unsafe_allow_html=True,
    )
    table = hubs[["station_name", "events", "mean_real", "mean_expected", "excess_per_event",
                  "excess_per_event_low", "excess_per_event_high"]].copy()
    table.columns = ["Station", "Primary delays", "Knock-on trains (real)", "Knock-on trains (expected)",
                     "Excess per delay", "95% low", "95% high"]
    st.dataframe(table, hide_index=True)

# ---------------------------------------------------------------- Performance
with tab_perf:
    st.subheader("How good are the predictions?")
    st.markdown(
        "Test period: July to September 2026, never used for training. Intervals come from a bootstrap that "
        "resamples whole days, because events on the same day are not independent."
    )
    metrics = load_csv("metrics.csv")
    tgt = st.radio("Target", sorted(metrics["target"].unique()), horizontal=True)
    m = metrics[metrics["target"] == tgt]

    def ci(row, key):
        return f"{row[key]:.3f} [{row[key + '_lo']:.3f}, {row[key + '_hi']:.3f}]"

    st.dataframe(
        pd.DataFrame(
            {
                "Model": m["model"].values,
                "PR-AUC [95% CI]": [ci(r, "pr_auc") for _, r in m.iterrows()],
                "ROC-AUC [95% CI]": [ci(r, "roc_auc") for _, r in m.iterrows()],
                "Brier score [95% CI] (lower is better)": [ci(r, "brier") for _, r in m.iterrows()],
            }
        ),
        hide_index=True,
    )
    st.markdown(
        "<p class='note'>The station-and-hour baseline knows only where and when. Most of the predictive signal "
        "is structural (busy stations at busy times); live delay information adds a smaller part.</p>",
        unsafe_allow_html=True,
    )

    st.markdown("#### Are the probabilities trustworthy?")
    lab = st.selectbox("Reliability for", list(TARGETS), key="rel")
    pc, pr_, yc = TARGETS[lab]["p"], TARGETS[lab]["p_raw"], TARGETS[lab]["y"]
    rel_cal, rel_raw = reliability(events, pc, yc), reliability(events, pr_, yc)
    figr = go.Figure()
    figr.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect", line=dict(dash="dash", color=GREY)))
    figr.add_trace(go.Scatter(x=rel_raw["predicted"], y=rel_raw["actual"], mode="lines+markers", name="Raw model"))
    figr.add_trace(go.Scatter(x=rel_cal["predicted"], y=rel_cal["actual"], mode="lines+markers",
                              name="Calibrated (shown in the map)", line=dict(color=ACCENT)))
    figr.update_layout(height=420, xaxis_title="Predicted probability", yaxis_title="Observed cascade rate",
                       margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(figr)

    st.markdown("#### What do the predictions rely on?")
    abl = load_csv("ablation.csv")
    abl = abl[abl["features removed"] != "none (full model)"].sort_values("PR-AUC change")
    figa = go.Figure(go.Bar(x=abl["PR-AUC change"], y=abl["features removed"], orientation="h", marker_color=ACCENT))
    figa.update_layout(height=380, margin=dict(l=0, r=10, t=10, b=0),
                       xaxis_title="Change in PR-AUC when this feature group is removed (2+ trains target)")
    st.plotly_chart(figa)
    st.markdown(
        "<p class='note'>Single runs without intervals: only the larger drops are credible. Station size and "
        "history matter most; recent delays at the station add a little; neighbouring stations add nothing.</p>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------- About
with tab_about:
    st.subheader("About this project")
    st.markdown(
        """
**Question.** When one train gets delayed, how likely is it that other trains are delayed too, and can we
see it coming?

**Data.** Public Deutsche Bahn timetable and delay records (CC BY 4.0), 14 December 2025 to 30 September 2026,
regional and long-distance trains only, in a Berlin-centred network. S-Bahn, U-Bahn and buses are excluded
because their short headways absorb delays differently.

**Definitions.**
- *Primary delay:* a delay of at least 10 minutes that appears at a stop.
- *Cascade:* at least 2 other trains are newly delayed (5+ minutes) at the same station within 30 minutes.
  *Severe cascade:* at least 4.
- Placebo tests (the same station one week and three hours earlier and later) show that knock-on delays are
  about twice as frequent after a primary delay as by chance at the same station, while definitions that
  include neighbouring stations are close to chance.

**Model.** Gradient boosting on information available at the moment of the delay (timetable, recent delays,
station and network context), calibrated on a separate validation period. Training: until 30 April 2026;
validation: May and June; test: July to September.

**Limitations.**
- Cascades are inferred from delays that co-occur, which shows association, not proven causation.
- Probabilities run low on unusually disrupted days, because the calibration period was calmer.
- No weather, track works or rolling-stock data. The network is Berlin-centred.
- A decision-support demonstration, not an operational system.

**Team.** Maheboob Patel and Mohammad Akbar Ali. Supervisor: Prof. Humera Noor Minhas.
M.Sc. Data Science, University of Europe for Applied Sciences, Potsdam.
        """
    )
