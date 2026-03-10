"""
Crop Diversity Interactive Dashboard
=====================================

Interactive visualization tool for exploring crop diversity indices across
Indian districts, states, and irrigation regimes.

USAGE:
    python scripts/58_crop_diversity_dashboard.py

Then open http://127.0.0.1:8050 in your browser.

REQUIREMENTS:
    pip install dash plotly pandas geopandas dash-bootstrap-components

DATA:
    All data from outputs/crop_diversity_analysis/
    Shapefile from Package_Maps_Share_20251120_FINAL/shapefiles/in_district.shp
"""

import os
import io
import json
import warnings
import pandas as pd
import geopandas as gpd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, html, dcc, Input, Output, State, callback, no_update, dash_table
import dash
import dash_bootstrap_components as dbc

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "outputs", "crop_diversity_analysis")
SHAPEFILE = os.path.join(
    BASE, "Package_Maps_Share_20251120_FINAL", "shapefiles", "in_district.shp"
)

# ---------------------------------------------------------------------------
# Load & Prepare Data
# ---------------------------------------------------------------------------
print("Loading data...")

df_main = pd.read_csv(os.path.join(DATA_DIR, "district_diversity_indices.csv"))
df_panel = pd.read_csv(os.path.join(DATA_DIR, "district_year_diversity_panel.csv"))
df_change = pd.read_csv(os.path.join(DATA_DIR, "district_diversity_change.csv"))
df_state = pd.read_csv(os.path.join(DATA_DIR, "state_diversity_summary.csv"))

print("Loading shapefile...")
gdf = gpd.read_file(SHAPEFILE)

# Build matching key from shapefile
gdf["match_key"] = gdf["stname"].str.upper().str.strip() + "|" + gdf["dtname"].str.upper().str.strip()


# Normalize names for fuzzy matching
def normalize(s):
    """Normalize district/state names for matching."""
    return (
        s.upper()
        .replace("&", "AND")
        .replace("-", " ")
        .replace("  ", " ")
        .strip()
    )


gdf["norm_key"] = gdf["stname"].apply(normalize) + "|" + gdf["dtname"].apply(normalize)
df_main["norm_key"] = df_main["district_key"].apply(normalize)

# Merge shapefile with diversity data
merged = gdf.merge(df_main, on="norm_key", how="left")

# Convert to GeoJSON for Plotly
print("Converting to GeoJSON...")
merged_json = json.loads(merged.to_json())

# Add id to each feature for choropleth mapping
for i, feature in enumerate(merged_json["features"]):
    feature["id"] = i

# Build a mapping from index to properties for hover
merged = merged.reset_index(drop=True)

# ---------------------------------------------------------------------------
# Precompute some aggregations
# ---------------------------------------------------------------------------

# State-level time series
state_yearly = (
    df_panel.groupby(["state_name", "year_start"])
    .agg(
        shannon_index=("shannon_index", "mean"),
        simpson_index=("simpson_index", "mean"),
        crop_richness=("crop_richness", "mean"),
    )
    .reset_index()
)

# State-level std for confidence bands
state_yearly_std = (
    df_panel.groupby(["state_name", "year_start"])
    .agg(
        shannon_std=("shannon_index", "std"),
        simpson_std=("simpson_index", "std"),
        richness_std=("crop_richness", "std"),
    )
    .reset_index()
)
state_yearly = state_yearly.merge(state_yearly_std, on=["state_name", "year_start"], how="left")

# Add irrigation regime to panel data via merge
irr_map = df_main[["district_key", "irrigation_regime"]].dropna().set_index("district_key")["irrigation_regime"]
df_panel["irrigation_regime"] = df_panel["district_key"].map(irr_map)

# Irrigation regime time series
irr_yearly = (
    df_panel.dropna(subset=["irrigation_regime"])
    .groupby(["irrigation_regime", "year_start"])
    .agg(
        shannon_index=("shannon_index", "mean"),
        simpson_index=("simpson_index", "mean"),
        crop_richness=("crop_richness", "mean"),
    )
    .reset_index()
)

# Crop share columns
CROP_SHARE_COLS = [c for c in df_main.columns if c.startswith("share_")]
CROP_SHARE_LABELS = {c: c.replace("share_", "").replace("_", " ").title() for c in CROP_SHARE_COLS}

# Index options
INDEX_OPTIONS = [
    {"label": "Shannon Index", "value": "shannon_index"},
    {"label": "Simpson Index", "value": "simpson_index"},
    {"label": "Crop Richness", "value": "crop_richness"},
    {"label": "Agro-Biodiversity Index (ABI)", "value": "agro_biodiversity_index"},
]

INDEX_LABELS = {o["value"]: o["label"] for o in INDEX_OPTIONS}

# States and irrigation regimes for dropdowns
ALL_STATES = sorted(df_main["state_name"].dropna().unique().tolist())
ALL_REGIMES = sorted(df_main["irrigation_regime"].dropna().unique().tolist())
ALL_YEARS = sorted(df_panel["year_start"].unique().tolist())

# Color scales
COLOR_SCALES = {
    "shannon_index": "YlGnBu",
    "simpson_index": "YlOrRd",
    "crop_richness": "Viridis",
    "agro_biodiversity_index": "RdYlGn",
}

REGIME_COLORS = {
    "Rainfed (<40%)": "#8B4513",
    "Semi-Irrigated (40-60%)": "#DAA520",
    "Irrigated (>60%)": "#4682B4",
}

# Precompute summary stats
TOTAL_DISTRICTS = int(df_main["district_name"].notna().sum())
AVG_SHANNON = float(df_main["shannon_index"].mean())
AVG_ABI = float(df_main["agro_biodiversity_index"].mean())
PCT_DECLINING = float((df_change["shannon_change"] < 0).sum() / len(df_change) * 100) if len(df_change) > 0 else 0
MOST_DIVERSE_STATE = df_state.loc[df_state["agro_biodiversity_index"].idxmax(), "state_name"] if len(df_state) > 0 else "N/A"
LEAST_DIVERSE_STATE = df_state.loc[df_state["agro_biodiversity_index"].idxmin(), "state_name"] if len(df_state) > 0 else "N/A"

# ---------------------------------------------------------------------------
# Styling Constants
# ---------------------------------------------------------------------------
FONT_FAMILY = "Georgia, 'Palatino Linotype', 'Book Antiqua', Palatino, serif"
FONT_SANS = "'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"

HEADER_BG = "#1B2A4A"
HEADER_ACCENT = "#C9A961"
CARD_SHADOW = "0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.08)"
CARD_STYLE = {
    "backgroundColor": "white",
    "borderRadius": "4px",
    "padding": "12px",
    "boxShadow": CARD_SHADOW,
    "border": "1px solid #e8e8e8",
}
SECTION_BG = "#FAFAFA"
PLOT_BG = "white"
PAPER_BG = "#FAFAFA"
GRID_COLOR = "#E8E8E8"
TITLE_SIZE = 18
SUBTITLE_SIZE = 14
BODY_SIZE = 12
MUTED_TEXT = "#6c757d"

PLOT_LAYOUT_DEFAULTS = dict(
    paper_bgcolor=PAPER_BG,
    plot_bgcolor=PLOT_BG,
    font=dict(family=FONT_SANS, size=BODY_SIZE, color="#333"),
    xaxis=dict(gridcolor=GRID_COLOR, linecolor="#ccc"),
    yaxis=dict(gridcolor=GRID_COLOR, linecolor="#ccc"),
    margin=dict(l=50, r=20, t=45, b=40),
)

TABLE_STYLE_HEADER = {
    "backgroundColor": "#f0f0f0",
    "fontWeight": "bold",
    "fontSize": "11px",
    "fontFamily": FONT_SANS,
    "border": "1px solid #ddd",
    "textAlign": "center",
    "padding": "6px 8px",
}

TABLE_STYLE_DATA = {
    "fontSize": "11px",
    "fontFamily": FONT_SANS,
    "border": "1px solid #eee",
    "padding": "4px 8px",
}

TABLE_STYLE_DATA_CONDITIONAL = [
    {"if": {"row_index": "odd"}, "backgroundColor": "#f9f9f9"},
    {"if": {"state": "active"}, "backgroundColor": "#e8f4fd", "border": "1px solid #4682B4"},
]

# Tab styling
TAB_STYLE = {
    "padding": "8px 16px",
    "fontFamily": FONT_SANS,
    "fontSize": "13px",
    "fontWeight": "500",
    "borderBottom": "2px solid transparent",
    "backgroundColor": "transparent",
    "color": "#555",
}

TAB_SELECTED_STYLE = {
    **TAB_STYLE,
    "borderBottom": f"2px solid {HEADER_BG}",
    "color": HEADER_BG,
    "fontWeight": "700",
}

# ---------------------------------------------------------------------------
# Helper: stat card
# ---------------------------------------------------------------------------
def stat_card(title, value, subtitle="", color=HEADER_BG):
    return dbc.Card(
        dbc.CardBody(
            [
                html.P(title, style={
                    "fontSize": "11px", "color": MUTED_TEXT, "marginBottom": "2px",
                    "fontFamily": FONT_SANS, "textTransform": "uppercase", "letterSpacing": "0.5px",
                }),
                html.H4(value, style={
                    "fontSize": "20px", "fontWeight": "700", "color": color,
                    "marginBottom": "2px", "fontFamily": FONT_SANS,
                }),
                html.P(subtitle, style={
                    "fontSize": "10px", "color": MUTED_TEXT, "marginBottom": "0",
                }) if subtitle else html.Span(),
            ],
            style={"padding": "10px 12px"},
        ),
        style={"border": "1px solid #e0e0e0", "borderTop": f"3px solid {color}"},
        className="shadow-sm",
    )


# ---------------------------------------------------------------------------
# Dash App
# ---------------------------------------------------------------------------
app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.FLATLY],
)
app.title = "Crop Diversity & Agro-Biodiversity in India"

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
app.layout = html.Div(
    style={"fontFamily": FONT_SANS, "backgroundColor": SECTION_BG, "minHeight": "100vh"},
    children=[
        # ---- HEADER ----
        html.Div(
            style={
                "backgroundColor": HEADER_BG,
                "color": "white",
                "padding": "16px 30px",
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "center",
            },
            children=[
                html.Div([
                    html.H1(
                        "Crop Diversity & Agro-Biodiversity in India (1997\u20132021)",
                        style={
                            "margin": "0", "fontSize": "22px", "fontWeight": "700",
                            "fontFamily": FONT_FAMILY, "letterSpacing": "0.3px",
                        },
                    ),
                    html.P(
                        "District-level analysis across 755 districts, 54 crops, 3 irrigation regimes",
                        style={
                            "margin": "3px 0 0 0", "opacity": "0.75", "fontSize": "12px",
                            "fontFamily": FONT_SANS,
                        },
                    ),
                ]),
                html.Div([
                    html.Span("CEEW", style={
                        "fontSize": "16px", "fontWeight": "700", "color": HEADER_ACCENT,
                        "fontFamily": FONT_FAMILY, "letterSpacing": "2px",
                    }),
                    html.Br(),
                    html.Span("Council on Energy, Environment & Water", style={
                        "fontSize": "9px", "color": "#aaa", "fontFamily": FONT_SANS,
                    }),
                ], style={"textAlign": "right"}),
            ],
        ),

        # ---- TABS ----
        dcc.Tabs(
            id="main-tabs",
            value="tab-overview",
            style={"margin": "0 15px", "borderBottom": "1px solid #ddd", "backgroundColor": "white"},
            children=[
                dcc.Tab(label="Overview Dashboard", value="tab-overview",
                        style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                dcc.Tab(label="Irrigation Analysis", value="tab-irrigation",
                        style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                dcc.Tab(label="Temporal Trends", value="tab-timeseries",
                        style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                dcc.Tab(label="Build-Up Explorer", value="tab-buildup",
                        style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                dcc.Tab(label="Change Analysis", value="tab-change",
                        style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
            ],
        ),

        # ---- TAB CONTENT ----
        html.Div(id="tab-content", style={"padding": "12px 15px"}),

        # ---- FOOTER ----
        html.Div(
            style={
                "backgroundColor": "#f0f0f0", "borderTop": "1px solid #ddd",
                "padding": "10px 30px", "marginTop": "20px",
                "display": "flex", "justifyContent": "space-between", "alignItems": "center",
            },
            children=[
                html.Span(
                    "Data Source: District-level crop area statistics, Land Use Statistics (LUS), "
                    "Ministry of Agriculture & Farmers' Welfare, Government of India",
                    style={"fontSize": "10px", "color": MUTED_TEXT, "fontFamily": FONT_SANS},
                ),
                html.Span(
                    "CEEW \u00b7 Crop Diversity Dashboard v2.0",
                    style={"fontSize": "10px", "color": MUTED_TEXT, "fontFamily": FONT_SANS},
                ),
            ],
        ),
    ],
)


# Tab routing
@callback(Output("tab-content", "children"), Input("main-tabs", "value"))
def render_tab(tab):
    if tab == "tab-overview":
        return render_overview_tab()
    elif tab == "tab-irrigation":
        return render_irrigation_tab()
    elif tab == "tab-timeseries":
        return render_timeseries_tab()
    elif tab == "tab-buildup":
        return render_buildup_tab()
    elif tab == "tab-change":
        return render_change_tab()
    return html.Div("Select a tab")


# =========================================================================
# TAB 1: OVERVIEW DASHBOARD
# =========================================================================
def render_overview_tab():
    return html.Div([
        # Controls row
        dbc.Row([
            dbc.Col([
                html.Label("Diversity Index", style={"fontWeight": "600", "fontSize": "11px", "color": "#555"}),
                dcc.Dropdown(
                    id="map-index", options=INDEX_OPTIONS,
                    value="agro_biodiversity_index", clearable=False,
                    style={"fontSize": "12px"},
                ),
            ], width=2),
            dbc.Col([
                html.Label("Filter by State", style={"fontWeight": "600", "fontSize": "11px", "color": "#555"}),
                dcc.Dropdown(
                    id="map-state-filter",
                    options=[{"label": s, "value": s} for s in ALL_STATES],
                    multi=True, placeholder="All States",
                    style={"fontSize": "12px"},
                ),
            ], width=3),
            dbc.Col([
                html.Label("Irrigation Regime", style={"fontWeight": "600", "fontSize": "11px", "color": "#555"}),
                dcc.Dropdown(
                    id="map-regime-filter",
                    options=[{"label": r, "value": r} for r in ALL_REGIMES],
                    multi=True, placeholder="All Regimes",
                    style={"fontSize": "12px"},
                ),
            ], width=2),
            dbc.Col([
                html.Label("Overlay", style={"fontWeight": "600", "fontSize": "11px", "color": "#555"}),
                dcc.Checklist(
                    id="map-regime-overlay",
                    options=[{"label": " Color by irrigation regime", "value": "yes"}],
                    value=[], style={"fontSize": "12px", "marginTop": "6px"},
                ),
            ], width=2),
        ], className="mb-2 g-2"),

        # Main content: Map (60%) + Side panel (40%)
        dbc.Row([
            # LEFT: Choropleth map
            dbc.Col([
                dbc.Card([
                    dcc.Graph(id="choropleth-map", style={"height": "580px"}),
                ], style={"border": "1px solid #e0e0e0"}, className="shadow-sm"),
                # Below map: distribution histogram
                dbc.Card([
                    dcc.Graph(id="overview-histogram", style={"height": "160px"}),
                ], style={"border": "1px solid #e0e0e0", "marginTop": "8px"}, className="shadow-sm"),
            ], width=7),

            # RIGHT: Summary + mini charts + detail
            dbc.Col([
                # Summary stat cards
                html.Div([
                    html.H6("Summary Statistics", style={
                        "fontSize": "13px", "fontWeight": "700", "color": HEADER_BG,
                        "marginBottom": "8px", "fontFamily": FONT_FAMILY,
                        "borderBottom": f"2px solid {HEADER_ACCENT}", "paddingBottom": "4px",
                    }),
                    dbc.Row([
                        dbc.Col(stat_card("Districts", str(TOTAL_DISTRICTS), "analyzed"), width=6),
                        dbc.Col(stat_card("Avg Shannon", f"{AVG_SHANNON:.3f}", "mean index"), width=6),
                    ], className="g-2 mb-2"),
                    dbc.Row([
                        dbc.Col(stat_card("Avg ABI", f"{AVG_ABI:.3f}", "agro-biodiversity"), width=6),
                        dbc.Col(stat_card("Declining", f"{PCT_DECLINING:.1f}%", "Shannon decrease", color="#c0392b"), width=6),
                    ], className="g-2 mb-2"),
                ]),

                # Mini bar chart: top 10 states
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Top 10 States", style={
                            "fontSize": "12px", "fontWeight": "700", "color": "#333",
                            "marginBottom": "4px",
                        }),
                        dcc.Graph(id="overview-top-states", style={"height": "200px"},
                                  config={"displayModeBar": False}),
                    ], style={"padding": "8px"}),
                ], style={"border": "1px solid #e0e0e0", "marginBottom": "8px"}, className="shadow-sm"),

                # District detail panel (click-driven)
                dbc.Card([
                    dbc.CardBody(
                        id="district-detail-panel",
                        children=[
                            html.H6("District Details", style={
                                "fontSize": "12px", "fontWeight": "700", "color": "#333",
                                "marginBottom": "4px",
                            }),
                            html.P("Click a district on the map.", style={"color": MUTED_TEXT, "fontSize": "11px"}),
                        ],
                        style={"padding": "8px", "maxHeight": "260px", "overflowY": "auto"},
                    ),
                ], style={"border": "1px solid #e0e0e0"}, className="shadow-sm"),
            ], width=5),
        ], className="g-2"),
    ])


@callback(
    Output("choropleth-map", "figure"),
    Output("overview-histogram", "figure"),
    Output("overview-top-states", "figure"),
    Input("map-index", "value"),
    Input("map-state-filter", "value"),
    Input("map-regime-filter", "value"),
    Input("map-regime-overlay", "value"),
)
def update_overview(index_col, states, regimes, overlay):
    dff = merged.copy()

    if states:
        dff = dff[dff["state_name"].isin(states)]
    if regimes:
        dff = dff[dff["irrigation_regime"].isin(regimes)]

    valid_ids = set(dff.index.tolist())
    filtered_geojson = {
        "type": "FeatureCollection",
        "features": [f for f in merged_json["features"] if f["id"] in valid_ids],
    }

    use_overlay = "yes" in (overlay or [])

    if use_overlay and "irrigation_regime" in dff.columns:
        fig_map = px.choropleth_mapbox(
            dff, geojson=filtered_geojson, locations=dff.index,
            color="irrigation_regime", color_discrete_map=REGIME_COLORS,
            hover_name="district_name",
            hover_data={
                "state_name": True, index_col: ":.3f",
                "irrigation_regime": True, "dominant_crop": True,
                "dominant_crop_share": ":.1%", "crop_richness": True,
            },
            mapbox_style="open-street-map",
            center={"lat": 22, "lon": 82}, zoom=3.5, opacity=0.65,
        )
    else:
        fig_map = px.choropleth_mapbox(
            dff, geojson=filtered_geojson, locations=dff.index,
            color=index_col,
            color_continuous_scale=COLOR_SCALES.get(index_col, "YlGnBu"),
            hover_name="district_name",
            hover_data={
                "state_name": True, "shannon_index": ":.3f",
                "simpson_index": ":.3f", "crop_richness": True,
                "agro_biodiversity_index": ":.3f",
                "irrigation_regime": True, "dominant_crop": True,
                "dominant_crop_share": ":.1%",
            },
            labels={index_col: INDEX_LABELS.get(index_col, index_col)},
            mapbox_style="open-street-map",
            center={"lat": 22, "lon": 82}, zoom=3.5, opacity=0.65,
        )

    fig_map.update_layout(
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
        title=dict(text=f"{INDEX_LABELS.get(index_col, index_col)} by District",
                   font=dict(size=14, family=FONT_FAMILY)),
        paper_bgcolor=PAPER_BG,
        font=dict(family=FONT_SANS, size=11),
    )

    # Histogram
    hist_data = dff[index_col].dropna()
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=hist_data, nbinsx=50, marker_color=HEADER_BG, opacity=0.8,
        name=INDEX_LABELS.get(index_col, index_col),
    ))
    fig_hist.add_vline(x=hist_data.mean(), line_dash="dash", line_color="#c0392b", line_width=1.5)
    fig_hist.update_layout(
        **PLOT_LAYOUT_DEFAULTS,
        title=dict(text=f"Distribution of {INDEX_LABELS.get(index_col, index_col)}",
                   font=dict(size=12)),
        margin=dict(l=40, r=10, t=30, b=25),
        showlegend=False,
        xaxis_title=None, yaxis_title="Count",
    )

    # Top 10 states bar chart
    state_means = dff.dropna(subset=[index_col]).groupby("state_name")[index_col].mean().nlargest(10)
    fig_states = go.Figure()
    fig_states.add_trace(go.Bar(
        x=state_means.values, y=state_means.index, orientation="h",
        marker_color=HEADER_BG, opacity=0.85,
    ))
    fig_states.update_layout(
        **PLOT_LAYOUT_DEFAULTS,
        margin=dict(l=100, r=10, t=8, b=8),
        yaxis=dict(autorange="reversed", tickfont=dict(size=10), gridcolor=GRID_COLOR),
        xaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(size=9)),
        showlegend=False,
    )

    return fig_map, fig_hist, fig_states


@callback(
    Output("district-detail-panel", "children"),
    Input("choropleth-map", "clickData"),
    State("map-index", "value"),
)
def update_district_panel(click_data, index_col):
    if not click_data:
        return [
            html.H6("District Details", style={
                "fontSize": "12px", "fontWeight": "700", "color": "#333", "marginBottom": "4px",
            }),
            html.P("Click a district on the map.", style={"color": MUTED_TEXT, "fontSize": "11px"}),
        ]

    point = click_data["points"][0]
    loc = point.get("location")
    if loc is None or loc >= len(merged):
        return html.P("No data available.", style={"fontSize": "11px"})

    row = merged.iloc[loc]

    children = [
        html.H6(
            f"{row.get('district_name', 'N/A')}, {row.get('state_name', 'N/A')}",
            style={"fontSize": "13px", "fontWeight": "700", "color": HEADER_BG, "marginBottom": "4px"},
        ),
        html.Hr(style={"margin": "4px 0"}),
    ]

    # Index values as compact list
    for idx, label in INDEX_LABELS.items():
        val = row.get(idx)
        if pd.notna(val):
            children.append(html.Div([
                html.Span(f"{label}: ", style={"fontWeight": "600", "fontSize": "11px", "color": "#555"}),
                html.Span(f"{val:.4f}", style={"fontSize": "11px"}),
            ], style={"marginBottom": "1px"}))

    # ABI category badge
    abi_cat = row.get("abi_category")
    if pd.notna(abi_cat):
        cat_colors = {"Very Low": "#c0392b", "Low": "#e67e22", "Moderate": "#f39c12", "High": "#27ae60", "Very High": "#1a5e20"}
        children.append(html.Span(abi_cat, style={
            "backgroundColor": cat_colors.get(abi_cat, "#999"), "color": "white",
            "padding": "1px 6px", "borderRadius": "3px", "fontSize": "10px",
            "display": "inline-block", "marginTop": "3px",
        }))

    children.append(html.Hr(style={"margin": "4px 0"}))

    # Irrigation + crop info compact
    irr_pct = row.get("irrigation_pct")
    irr_regime = row.get("irrigation_regime")
    dom = row.get("dominant_crop")
    dom_share = row.get("dominant_crop_share")
    richness = row.get("crop_richness")

    info_items = []
    if pd.notna(irr_pct):
        info_items.append(f"Irrigation: {irr_pct:.1f}%")
    if pd.notna(irr_regime):
        info_items.append(f"Regime: {irr_regime}")
    if pd.notna(dom):
        crop_str = f"Dominant: {dom}"
        if pd.notna(dom_share):
            crop_str += f" ({dom_share:.1%})"
        info_items.append(crop_str)
    if pd.notna(richness):
        info_items.append(f"Richness: {int(richness)}")

    for item in info_items:
        children.append(html.P(item, style={"fontSize": "10px", "margin": "1px 0", "color": "#444"}))

    # Mini sparkline
    dk = row.get("district_key")
    if pd.notna(dk):
        panel_d = df_panel[df_panel["district_key"] == dk].sort_values("year_start")
        if len(panel_d) > 0:
            children.append(html.Hr(style={"margin": "4px 0"}))
            spark = go.Figure()
            spark.add_trace(go.Scatter(
                x=panel_d["year_start"], y=panel_d["shannon_index"],
                mode="lines", line=dict(color=HEADER_BG, width=1.5),
                fill="tozeroy", fillcolor="rgba(27,42,74,0.08)",
            ))
            spark.update_layout(
                height=80, margin=dict(l=25, r=5, t=5, b=20),
                xaxis=dict(showgrid=False, dtick=5, tickfont=dict(size=8)),
                yaxis=dict(showgrid=True, gridcolor="#eee", tickfont=dict(size=8)),
                paper_bgcolor="white", plot_bgcolor="white",
            )
            children.append(dcc.Graph(figure=spark, config={"displayModeBar": False}))

    return children


# =========================================================================
# TAB 2: IRRIGATION ANALYSIS
# =========================================================================
def render_irrigation_tab():
    return html.Div([
        # Controls
        dbc.Row([
            dbc.Col([
                html.Label("Diversity Index", style={"fontWeight": "600", "fontSize": "11px", "color": "#555"}),
                dcc.Dropdown(
                    id="irr-index", options=INDEX_OPTIONS,
                    value="agro_biodiversity_index", clearable=False,
                    style={"fontSize": "12px"},
                ),
            ], width=3),
        ], className="mb-2"),

        # Three maps side by side
        html.H6("Spatial Distribution by Irrigation Regime", style={
            "fontSize": "14px", "fontWeight": "700", "color": HEADER_BG,
            "fontFamily": FONT_FAMILY, "borderBottom": f"1px solid {GRID_COLOR}",
            "paddingBottom": "4px", "marginBottom": "8px",
        }),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    html.Div("Rainfed (<40%)", style={
                        "backgroundColor": "#8B4513", "color": "white",
                        "padding": "4px 8px", "fontSize": "11px", "fontWeight": "600",
                    }),
                    dcc.Graph(id="irr-map-rainfed", style={"height": "340px"},
                              config={"displayModeBar": False}),
                ], className="shadow-sm"),
            ], width=4),
            dbc.Col([
                dbc.Card([
                    html.Div("Semi-Irrigated (40-60%)", style={
                        "backgroundColor": "#DAA520", "color": "white",
                        "padding": "4px 8px", "fontSize": "11px", "fontWeight": "600",
                    }),
                    dcc.Graph(id="irr-map-semi", style={"height": "340px"},
                              config={"displayModeBar": False}),
                ], className="shadow-sm"),
            ], width=4),
            dbc.Col([
                dbc.Card([
                    html.Div("Irrigated (>60%)", style={
                        "backgroundColor": "#4682B4", "color": "white",
                        "padding": "4px 8px", "fontSize": "11px", "fontWeight": "600",
                    }),
                    dcc.Graph(id="irr-map-irrigated", style={"height": "340px"},
                              config={"displayModeBar": False}),
                ], className="shadow-sm"),
            ], width=4),
        ], className="g-2 mb-3"),

        # Crop composition + box plots + stats table
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Crop Composition by Regime", style={
                            "fontSize": "12px", "fontWeight": "700", "marginBottom": "4px",
                        }),
                        dcc.Graph(id="irr-crop-composition", style={"height": "280px"},
                                  config={"displayModeBar": False}),
                    ], style={"padding": "8px"}),
                ], className="shadow-sm"),
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Index Distribution by Regime", style={
                            "fontSize": "12px", "fontWeight": "700", "marginBottom": "4px",
                        }),
                        dcc.Graph(id="irr-boxplots", style={"height": "280px"},
                                  config={"displayModeBar": False}),
                    ], style={"padding": "8px"}),
                ], className="shadow-sm"),
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Key Statistics by Regime", style={
                            "fontSize": "12px", "fontWeight": "700", "marginBottom": "4px",
                        }),
                        html.Div(id="irr-stats-table"),
                    ], style={"padding": "8px"}),
                ], className="shadow-sm"),
            ], width=4),
        ], className="g-2"),
    ])


@callback(
    Output("irr-map-rainfed", "figure"),
    Output("irr-map-semi", "figure"),
    Output("irr-map-irrigated", "figure"),
    Output("irr-crop-composition", "figure"),
    Output("irr-boxplots", "figure"),
    Output("irr-stats-table", "children"),
    Input("irr-index", "value"),
)
def update_irrigation_tab(index_col):
    regime_order = ["Rainfed (<40%)", "Semi-Irrigated (40-60%)", "Irrigated (>60%)"]
    regime_label_map = {
        "Rainfed (<40%)": "rainfed",
        "Semi-Irrigated (40-60%)": "semi",
        "Irrigated (>60%)": "irrigated",
    }

    # Get global color range for consistent scaling
    vmin = merged[index_col].quantile(0.02) if merged[index_col].notna().any() else 0
    vmax = merged[index_col].quantile(0.98) if merged[index_col].notna().any() else 1

    figs = []
    for regime in regime_order:
        dff = merged[merged["irrigation_regime"] == regime].copy()
        valid_ids = set(dff.index.tolist())
        filtered_gj = {
            "type": "FeatureCollection",
            "features": [f for f in merged_json["features"] if f["id"] in valid_ids],
        }

        fig = px.choropleth_mapbox(
            dff, geojson=filtered_gj, locations=dff.index,
            color=index_col,
            color_continuous_scale=COLOR_SCALES.get(index_col, "YlGnBu"),
            range_color=[vmin, vmax],
            hover_name="district_name",
            hover_data={"state_name": True, index_col: ":.3f"},
            mapbox_style="open-street-map",
            center={"lat": 22, "lon": 82}, zoom=3.2, opacity=0.65,
        )
        fig.update_layout(
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            paper_bgcolor=PAPER_BG,
            coloraxis_showscale=False,
            font=dict(family=FONT_SANS, size=10),
        )
        figs.append(fig)

    # Crop composition stacked bar
    regime_shares = df_main.dropna(subset=["irrigation_regime"]).groupby("irrigation_regime")[CROP_SHARE_COLS].mean()
    fig_comp = go.Figure()
    palette = ["#8B4513", "#DAA520", "#4682B4", "#6B8E23", "#CD853F",
               "#708090", "#BC8F8F", "#5F9EA0", "#D2691E", "#A0522D"]
    for i, col in enumerate(CROP_SHARE_COLS):
        fig_comp.add_trace(go.Bar(
            name=CROP_SHARE_LABELS[col],
            x=regime_shares.index, y=regime_shares[col],
            marker_color=palette[i % len(palette)],
        ))
    fig_comp.update_layout(
        **PLOT_LAYOUT_DEFAULTS,
        barmode="stack",
        yaxis_title="Share", yaxis_tickformat=".0%",
        legend=dict(orientation="h", yanchor="top", y=-0.25, font=dict(size=9)),
        margin=dict(l=40, r=10, t=10, b=80),
    )

    # Box plots
    dff_box = df_main.dropna(subset=["irrigation_regime", index_col])
    fig_box = go.Figure()
    for regime in regime_order:
        regime_data = dff_box[dff_box["irrigation_regime"] == regime][index_col]
        fig_box.add_trace(go.Box(
            y=regime_data, name=regime.split("(")[0].strip(),
            marker_color=REGIME_COLORS.get(regime, "#999"),
            boxmean="sd",
        ))
    fig_box.update_layout(
        **PLOT_LAYOUT_DEFAULTS,
        yaxis_title=INDEX_LABELS.get(index_col, index_col),
        showlegend=False,
        margin=dict(l=50, r=10, t=10, b=40),
    )

    # Stats table
    stats_rows = []
    for regime in regime_order:
        rd = df_main[df_main["irrigation_regime"] == regime]
        stats_rows.append({
            "Regime": regime.split("(")[0].strip(),
            "N": len(rd),
            "Mean": f"{rd[index_col].mean():.3f}" if rd[index_col].notna().any() else "N/A",
            "Median": f"{rd[index_col].median():.3f}" if rd[index_col].notna().any() else "N/A",
            "Std": f"{rd[index_col].std():.3f}" if rd[index_col].notna().any() else "N/A",
            "Min": f"{rd[index_col].min():.3f}" if rd[index_col].notna().any() else "N/A",
            "Max": f"{rd[index_col].max():.3f}" if rd[index_col].notna().any() else "N/A",
        })
    stats_df = pd.DataFrame(stats_rows)
    stats_table = dash_table.DataTable(
        data=stats_df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in stats_df.columns],
        style_header=TABLE_STYLE_HEADER,
        style_data=TABLE_STYLE_DATA,
        style_data_conditional=TABLE_STYLE_DATA_CONDITIONAL,
        style_table={"overflowX": "auto"},
    )

    return figs[0], figs[1], figs[2], fig_comp, fig_box, stats_table


# =========================================================================
# TAB 3: TEMPORAL TRENDS
# =========================================================================
def render_timeseries_tab():
    return html.Div([
        # Controls
        dbc.Row([
            dbc.Col([
                html.Label("Diversity Index", style={"fontWeight": "600", "fontSize": "11px", "color": "#555"}),
                dcc.Dropdown(
                    id="ts-index", options=INDEX_OPTIONS[:3],
                    value="shannon_index", clearable=False,
                    style={"fontSize": "12px"},
                ),
            ], width=2),
            dbc.Col([
                html.Label("Group By", style={"fontWeight": "600", "fontSize": "11px", "color": "#555"}),
                dcc.RadioItems(
                    id="ts-groupby",
                    options=[
                        {"label": "State", "value": "state"},
                        {"label": "Irrigation Regime", "value": "irrigation"},
                    ],
                    value="state", inline=True,
                    style={"fontSize": "12px", "marginTop": "6px"},
                ),
            ], width=2),
            dbc.Col([
                html.Label("Select States", style={"fontWeight": "600", "fontSize": "11px", "color": "#555"}),
                dcc.Dropdown(
                    id="ts-states",
                    options=[{"label": s, "value": s} for s in ALL_STATES],
                    multi=True,
                    value=["Karnataka", "Punjab", "Madhya Pradesh", "Uttar Pradesh", "Maharashtra"],
                    placeholder="Select states...",
                    style={"fontSize": "12px"},
                ),
            ], width=4),
        ], className="mb-2 g-2"),

        # Line chart with confidence bands
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dcc.Graph(id="timeseries-chart", style={"height": "380px"}),
                ], className="shadow-sm", style={"border": "1px solid #e0e0e0"}),
            ], width=12),
        ], className="mb-3"),

        # Animated map + sparklines grid
        html.H6("Spatial-Temporal Explorer", style={
            "fontSize": "14px", "fontWeight": "700", "color": HEADER_BG,
            "fontFamily": FONT_FAMILY, "borderBottom": f"1px solid {GRID_COLOR}",
            "paddingBottom": "4px", "marginBottom": "8px",
        }),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Label("Year", style={"fontWeight": "600", "fontSize": "11px"}),
                        dcc.Slider(
                            id="anim-year-slider",
                            min=min(ALL_YEARS), max=max(ALL_YEARS), step=1,
                            value=min(ALL_YEARS),
                            marks={y: str(y) for y in ALL_YEARS if y % 5 == 0},
                            tooltip={"placement": "bottom"},
                        ),
                        html.Label("Index", style={"fontWeight": "600", "fontSize": "11px", "marginTop": "8px"}),
                        dcc.Dropdown(
                            id="anim-index", options=INDEX_OPTIONS[:3],
                            value="shannon_index", clearable=False,
                            style={"fontSize": "12px", "width": "200px"},
                        ),
                        dcc.Graph(id="animated-map", style={"height": "420px"}),
                    ], style={"padding": "8px"}),
                ], className="shadow-sm", style={"border": "1px solid #e0e0e0"}),
            ], width=7),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("State Sparklines (Top 20)", style={
                            "fontSize": "12px", "fontWeight": "700", "marginBottom": "4px",
                        }),
                        dcc.Graph(id="sparklines-grid", style={"height": "480px"},
                                  config={"displayModeBar": False}),
                    ], style={"padding": "8px"}),
                ], className="shadow-sm", style={"border": "1px solid #e0e0e0"}),
            ], width=5),
        ], className="g-2"),
    ])


@callback(
    Output("timeseries-chart", "figure"),
    Input("ts-index", "value"),
    Input("ts-groupby", "value"),
    Input("ts-states", "value"),
)
def update_timeseries(index_col, groupby, states):
    if groupby == "state":
        data = state_yearly.copy()
        if states:
            data = data[data["state_name"].isin(states)]

        fig = go.Figure()
        colors = px.colors.qualitative.D3
        std_col = {"shannon_index": "shannon_std", "simpson_index": "simpson_std", "crop_richness": "richness_std"}.get(index_col, "shannon_std")

        for i, st in enumerate(data["state_name"].unique()):
            sd = data[data["state_name"] == st].sort_values("year_start")
            color = colors[i % len(colors)]

            # Confidence band
            if std_col in sd.columns and sd[std_col].notna().any():
                upper = sd[index_col] + sd[std_col]
                lower = sd[index_col] - sd[std_col]
                fig.add_trace(go.Scatter(
                    x=pd.concat([sd["year_start"], sd["year_start"][::-1]]),
                    y=pd.concat([upper, lower[::-1]]),
                    fill="toself", fillcolor=f"rgba{tuple(list(px.colors.hex_to_rgb(color)) + [0.1])}",
                    line=dict(width=0), showlegend=False, hoverinfo="skip",
                ))

            fig.add_trace(go.Scatter(
                x=sd["year_start"], y=sd[index_col],
                mode="lines+markers", name=st,
                line=dict(color=color, width=2),
                marker=dict(size=4),
            ))

        fig.update_layout(
            **PLOT_LAYOUT_DEFAULTS,
            title=dict(text=f"{INDEX_LABELS.get(index_col, index_col)} Trends by State",
                       font=dict(size=14, family=FONT_FAMILY)),
            xaxis_title="Year",
            yaxis_title=INDEX_LABELS.get(index_col, index_col),
            legend=dict(font=dict(size=10)),
        )
    else:
        data = irr_yearly.copy()
        fig = px.line(
            data, x="year_start", y=index_col, color="irrigation_regime",
            color_discrete_map=REGIME_COLORS,
            labels={
                "year_start": "Year",
                index_col: INDEX_LABELS.get(index_col, index_col),
                "irrigation_regime": "Irrigation Regime",
            },
            title=f"{INDEX_LABELS.get(index_col, index_col)} Trends by Irrigation Regime",
            markers=True,
        )
        fig.update_layout(**PLOT_LAYOUT_DEFAULTS)
        fig.update_traces(line=dict(width=2), marker=dict(size=5))

    return fig


@callback(
    Output("animated-map", "figure"),
    Input("anim-year-slider", "value"),
    Input("anim-index", "value"),
)
def update_animated_map(year, index_col):
    year_data = df_panel[df_panel["year_start"] == year].copy()
    year_data["norm_key"] = year_data["district_key"].apply(normalize)

    year_merged = gdf[["norm_key", "geometry"]].merge(year_data, on="norm_key", how="left")

    year_json = json.loads(year_merged.to_json())
    for i, f in enumerate(year_json["features"]):
        f["id"] = i

    fig = px.choropleth_mapbox(
        year_merged, geojson=year_json, locations=year_merged.index,
        color=index_col,
        color_continuous_scale=COLOR_SCALES.get(index_col, "YlGnBu"),
        hover_name="district_name",
        hover_data={"state_name": True, index_col: ":.3f", "crop_richness": True},
        labels={index_col: INDEX_LABELS.get(index_col, index_col)},
        title=f"{INDEX_LABELS.get(index_col, index_col)} \u2014 {year}",
        mapbox_style="open-street-map",
        center={"lat": 22, "lon": 82}, zoom=3.2, opacity=0.65,
    )
    fig.update_layout(
        margin={"r": 0, "t": 35, "l": 0, "b": 0},
        paper_bgcolor=PAPER_BG,
        font=dict(family=FONT_SANS, size=11),
    )
    return fig


@callback(
    Output("sparklines-grid", "figure"),
    Input("ts-index", "value"),
)
def update_sparklines(index_col):
    # Top 20 states by number of districts
    top_states = df_panel["state_name"].value_counts().head(20).index.tolist()
    state_ts = state_yearly[state_yearly["state_name"].isin(top_states)]

    n_cols = 4
    n_rows = 5
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=top_states[:20],
        vertical_spacing=0.06, horizontal_spacing=0.06,
    )

    for i, st in enumerate(top_states[:20]):
        r = i // n_cols + 1
        c = i % n_cols + 1
        sd = state_ts[state_ts["state_name"] == st].sort_values("year_start")
        fig.add_trace(go.Scatter(
            x=sd["year_start"], y=sd[index_col],
            mode="lines", line=dict(color=HEADER_BG, width=1.5),
            fill="tozeroy", fillcolor="rgba(27,42,74,0.06)",
            showlegend=False,
        ), row=r, col=c)

    fig.update_layout(
        paper_bgcolor=PAPER_BG, plot_bgcolor="white",
        margin=dict(l=20, r=10, t=30, b=10),
        font=dict(family=FONT_SANS, size=8),
        height=480,
    )
    fig.update_xaxes(showticklabels=False, showgrid=False)
    fig.update_yaxes(showticklabels=False, showgrid=True, gridcolor="#f0f0f0")
    fig.update_annotations(font_size=9)

    return fig


# =========================================================================
# TAB 4: BUILD-UP EXPLORER
# =========================================================================
def render_buildup_tab():
    return html.Div([
        html.H6("Interactive Build-Up Tool", style={
            "fontSize": "14px", "fontWeight": "700", "color": HEADER_BG,
            "fontFamily": FONT_FAMILY, "borderBottom": f"1px solid {GRID_COLOR}",
            "paddingBottom": "4px", "marginBottom": "4px",
        }),
        html.P(
            "Start with all districts. Layer criteria to narrow down. "
            "Matching districts are highlighted on the map.",
            style={"color": MUTED_TEXT, "fontSize": "11px", "marginBottom": "8px"},
        ),

        # Controls
        dbc.Row([
            dbc.Col([
                html.Label("States", style={"fontWeight": "600", "fontSize": "11px", "color": "#555"}),
                dcc.Dropdown(
                    id="bu-states",
                    options=[{"label": s, "value": s} for s in ALL_STATES],
                    multi=True, placeholder="Any state",
                    style={"fontSize": "12px"},
                ),
            ], width=3),
            dbc.Col([
                html.Label("Irrigation Regime", style={"fontWeight": "600", "fontSize": "11px", "color": "#555"}),
                dcc.Dropdown(
                    id="bu-regime",
                    options=[{"label": r, "value": r} for r in ALL_REGIMES],
                    multi=True, placeholder="Any regime",
                    style={"fontSize": "12px"},
                ),
            ], width=2),
            dbc.Col([
                html.Label("Index", style={"fontWeight": "600", "fontSize": "11px", "color": "#555"}),
                dcc.Dropdown(
                    id="bu-index", options=INDEX_OPTIONS,
                    value="agro_biodiversity_index", clearable=False,
                    style={"fontSize": "12px"},
                ),
            ], width=2),
            dbc.Col([
                html.Label("Min", style={"fontWeight": "600", "fontSize": "11px", "color": "#555"}),
                dcc.Input(id="bu-min", type="number", value=0, step=0.05,
                          style={"width": "100%", "fontSize": "12px"}),
            ], width=1),
            dbc.Col([
                html.Label("Max", style={"fontWeight": "600", "fontSize": "11px", "color": "#555"}),
                dcc.Input(id="bu-max", type="number", value=1, step=0.05,
                          style={"width": "100%", "fontSize": "12px"}),
            ], width=1),
            dbc.Col([
                html.Label("Dominant Crop", style={"fontWeight": "600", "fontSize": "11px", "color": "#555"}),
                dcc.Input(id="bu-crop", type="text", placeholder="e.g. Rice",
                          style={"width": "100%", "fontSize": "12px"}),
            ], width=2),
        ], className="mb-2 g-2"),

        # Summary bar
        html.Div(
            id="bu-summary",
            style={
                "backgroundColor": "#EBF5FB", "padding": "6px 12px",
                "borderRadius": "4px", "marginBottom": "8px",
                "fontSize": "12px", "border": "1px solid #AED6F1",
                "fontFamily": FONT_SANS,
            },
        ),

        # Map + results table
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dcc.Graph(id="buildup-map", style={"height": "480px"}),
                ], className="shadow-sm", style={"border": "1px solid #e0e0e0"}),
            ], width=7),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Matched Districts", style={
                            "fontSize": "12px", "fontWeight": "700", "marginBottom": "4px",
                        }),
                        html.Div(id="bu-results-table", style={"maxHeight": "420px", "overflowY": "auto"}),
                    ], style={"padding": "8px"}),
                ], className="shadow-sm", style={"border": "1px solid #e0e0e0"}),
            ], width=5),
        ], className="g-2 mb-2"),

        # Download button
        dbc.Row([
            dbc.Col([
                dbc.Button("Download Filtered Data (CSV)", id="bu-download-btn",
                           color="secondary", size="sm", outline=True,
                           style={"fontSize": "11px"}),
                dcc.Download(id="bu-download"),
            ], width="auto"),
        ]),
    ])


@callback(
    Output("buildup-map", "figure"),
    Output("bu-summary", "children"),
    Output("bu-results-table", "children"),
    Input("bu-states", "value"),
    Input("bu-regime", "value"),
    Input("bu-index", "value"),
    Input("bu-min", "value"),
    Input("bu-max", "value"),
    Input("bu-crop", "value"),
)
def update_buildup(states, regimes, index_col, min_val, max_val, crop_text):
    dff = merged.copy()
    mask = pd.Series(True, index=dff.index)

    criteria_text = []
    if states:
        mask &= dff["state_name"].isin(states)
        criteria_text.append(f"States: {', '.join(states)}")
    if regimes:
        mask &= dff["irrigation_regime"].isin(regimes)
        criteria_text.append(f"Regimes: {', '.join(regimes)}")
    if min_val is not None:
        mask &= dff[index_col].fillna(-999) >= min_val
        criteria_text.append(f"{INDEX_LABELS[index_col]} \u2265 {min_val}")
    if max_val is not None:
        mask &= dff[index_col].fillna(999) <= max_val
        criteria_text.append(f"{INDEX_LABELS[index_col]} \u2264 {max_val}")
    if crop_text:
        mask &= dff["dominant_crop"].fillna("").str.contains(crop_text, case=False)
        criteria_text.append(f"Dominant crop: '{crop_text}'")

    dff["selected"] = mask.map({True: "Matches", False: "Other"})
    n_matched = mask.sum()

    fig = px.choropleth_mapbox(
        dff, geojson=merged_json, locations=dff.index,
        color="selected",
        color_discrete_map={"Matches": "#27ae60", "Other": "#e0e0e0"},
        hover_name="district_name",
        hover_data={
            "state_name": True, index_col: ":.3f",
            "irrigation_regime": True, "dominant_crop": True, "selected": False,
        },
        category_orders={"selected": ["Other", "Matches"]},
        mapbox_style="open-street-map",
        center={"lat": 22, "lon": 82}, zoom=3.5, opacity=0.65,
    )
    fig.update_layout(
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
        paper_bgcolor=PAPER_BG,
        title=dict(text=f"{n_matched} districts match", font=dict(size=13)),
        font=dict(family=FONT_SANS, size=11),
    )

    summary = f"Matching: {n_matched} / {len(dff)} districts"
    if criteria_text:
        summary += " \u2014 " + " AND ".join(criteria_text)
    else:
        summary = "No filters applied. All districts shown."

    # Results table
    matched = dff[mask][["district_name", "state_name", index_col, "irrigation_regime", "dominant_crop", "crop_richness"]].copy()
    matched = matched.dropna(subset=["district_name"]).sort_values(index_col, ascending=False)
    matched.columns = ["District", "State", INDEX_LABELS.get(index_col, index_col), "Regime", "Dominant Crop", "Richness"]

    # Format numeric column
    idx_label = INDEX_LABELS.get(index_col, index_col)
    matched[idx_label] = matched[idx_label].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    matched["Richness"] = matched["Richness"].apply(lambda x: f"{int(x)}" if pd.notna(x) else "")

    table = dash_table.DataTable(
        data=matched.head(200).to_dict("records"),
        columns=[{"name": c, "id": c} for c in matched.columns],
        style_header=TABLE_STYLE_HEADER,
        style_data=TABLE_STYLE_DATA,
        style_data_conditional=TABLE_STYLE_DATA_CONDITIONAL,
        style_table={"overflowX": "auto"},
        sort_action="native",
        filter_action="native",
        page_size=50,
        export_format="csv",
        style_cell={"textAlign": "left", "maxWidth": "120px", "overflow": "hidden", "textOverflow": "ellipsis"},
    )

    return fig, summary, table


@callback(
    Output("bu-download", "data"),
    Input("bu-download-btn", "n_clicks"),
    State("bu-states", "value"),
    State("bu-regime", "value"),
    State("bu-index", "value"),
    State("bu-min", "value"),
    State("bu-max", "value"),
    State("bu-crop", "value"),
    prevent_initial_call=True,
)
def download_buildup(n_clicks, states, regimes, index_col, min_val, max_val, crop_text):
    if not n_clicks:
        return no_update

    dff = merged.copy()
    mask = pd.Series(True, index=dff.index)
    if states:
        mask &= dff["state_name"].isin(states)
    if regimes:
        mask &= dff["irrigation_regime"].isin(regimes)
    if min_val is not None:
        mask &= dff[index_col].fillna(-999) >= min_val
    if max_val is not None:
        mask &= dff[index_col].fillna(999) <= max_val
    if crop_text:
        mask &= dff["dominant_crop"].fillna("").str.contains(crop_text, case=False)

    export_cols = ["district_name", "state_name", "shannon_index", "simpson_index",
                   "crop_richness", "agro_biodiversity_index", "irrigation_regime",
                   "irrigation_pct", "dominant_crop", "dominant_crop_share"]
    export_df = dff[mask][[c for c in export_cols if c in dff.columns]]

    return dcc.send_data_frame(export_df.to_csv, "filtered_districts.csv", index=False)


# =========================================================================
# TAB 5: CHANGE ANALYSIS
# =========================================================================
def render_change_tab():
    return html.Div([
        # Controls
        dbc.Row([
            dbc.Col([
                html.Label("Change Metric", style={"fontWeight": "600", "fontSize": "11px", "color": "#555"}),
                dcc.Dropdown(
                    id="change-metric",
                    options=[
                        {"label": "Shannon Index Change", "value": "shannon_change"},
                        {"label": "Simpson Index Change", "value": "simpson_change"},
                        {"label": "Crop Richness Change", "value": "richness_change"},
                    ],
                    value="shannon_change", clearable=False,
                    style={"fontSize": "12px"},
                ),
            ], width=3),
        ], className="mb-2"),

        # Section header
        html.H6("Early Period (\u22642005) vs Late Period (\u22652015)", style={
            "fontSize": "14px", "fontWeight": "700", "color": HEADER_BG,
            "fontFamily": FONT_FAMILY, "borderBottom": f"1px solid {GRID_COLOR}",
            "paddingBottom": "4px", "marginBottom": "8px",
        }),

        # Three maps: Early | Change | Late
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    html.Div("Early Period (\u22642005)", style={
                        "backgroundColor": "#5B7DB1", "color": "white",
                        "padding": "4px 8px", "fontSize": "11px", "fontWeight": "600",
                    }),
                    dcc.Graph(id="change-map-early", style={"height": "350px"},
                              config={"displayModeBar": False}),
                ], className="shadow-sm"),
            ], width=4),
            dbc.Col([
                dbc.Card([
                    html.Div("Change (Late \u2212 Early)", style={
                        "backgroundColor": HEADER_BG, "color": "white",
                        "padding": "4px 8px", "fontSize": "11px", "fontWeight": "600",
                    }),
                    dcc.Graph(id="change-map", style={"height": "350px"},
                              config={"displayModeBar": False}),
                ], className="shadow-sm"),
            ], width=4),
            dbc.Col([
                dbc.Card([
                    html.Div("Late Period (\u22652015)", style={
                        "backgroundColor": "#2E86C1", "color": "white",
                        "padding": "4px 8px", "fontSize": "11px", "fontWeight": "600",
                    }),
                    dcc.Graph(id="change-map-late", style={"height": "350px"},
                              config={"displayModeBar": False}),
                ], className="shadow-sm"),
            ], width=4),
        ], className="g-2 mb-3"),

        # Histogram + regime bar + distribution
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Distribution of Change", style={"fontSize": "12px", "fontWeight": "700"}),
                        dcc.Graph(id="change-histogram", style={"height": "230px"},
                                  config={"displayModeBar": False}),
                    ], style={"padding": "8px"}),
                ], className="shadow-sm"),
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Mean Change by Regime", style={"fontSize": "12px", "fontWeight": "700"}),
                        dcc.Graph(id="change-by-regime", style={"height": "230px"},
                                  config={"displayModeBar": False}),
                    ], style={"padding": "8px"}),
                ], className="shadow-sm"),
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Change Summary", style={"fontSize": "12px", "fontWeight": "700", "marginBottom": "6px"}),
                        html.Div(id="change-summary-stats"),
                    ], style={"padding": "8px"}),
                ], className="shadow-sm"),
            ], width=4),
        ], className="g-2 mb-3"),

        # Gainers and losers tables
        html.H6("Top Gainers & Losers", style={
            "fontSize": "14px", "fontWeight": "700", "color": HEADER_BG,
            "fontFamily": FONT_FAMILY, "borderBottom": f"1px solid {GRID_COLOR}",
            "paddingBottom": "4px", "marginBottom": "8px",
        }),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    html.Div("Top 15 Gainers", style={
                        "backgroundColor": "#27ae60", "color": "white",
                        "padding": "4px 8px", "fontSize": "11px", "fontWeight": "600",
                    }),
                    dbc.CardBody([
                        html.Div(id="change-gainers-table"),
                    ], style={"padding": "6px"}),
                ], className="shadow-sm"),
            ], width=6),
            dbc.Col([
                dbc.Card([
                    html.Div("Top 15 Losers", style={
                        "backgroundColor": "#c0392b", "color": "white",
                        "padding": "4px 8px", "fontSize": "11px", "fontWeight": "600",
                    }),
                    dbc.CardBody([
                        html.Div(id="change-losers-table"),
                    ], style={"padding": "6px"}),
                ], className="shadow-sm"),
            ], width=6),
        ], className="g-2"),
    ])


@callback(
    Output("change-map-early", "figure"),
    Output("change-map", "figure"),
    Output("change-map-late", "figure"),
    Output("change-histogram", "figure"),
    Output("change-by-regime", "figure"),
    Output("change-summary-stats", "children"),
    Output("change-gainers-table", "children"),
    Output("change-losers-table", "children"),
    Input("change-metric", "value"),
)
def update_change_tab(metric):
    df_c = df_change.copy()
    df_c["norm_key"] = df_c["district_key"].apply(normalize)
    change_merged = gdf[["norm_key", "geometry"]].merge(df_c, on="norm_key", how="left")

    change_json = json.loads(change_merged.to_json())
    for i, f in enumerate(change_json["features"]):
        f["id"] = i

    metric_label = {
        "shannon_change": "Shannon Change",
        "simpson_change": "Simpson Change",
        "richness_change": "Richness Change",
    }[metric]

    # Early/late column names
    base_name = metric.replace("_change", "")
    early_col = f"{base_name}_early"
    late_col = f"{base_name}_late"

    # --- Early period map ---
    vmin_e = change_merged[early_col].quantile(0.02) if change_merged[early_col].notna().any() else 0
    vmax_e = change_merged[late_col].quantile(0.98) if change_merged[late_col].notna().any() else 1
    # Use same range for both early and late
    global_vmin = min(vmin_e, change_merged[late_col].quantile(0.02) if change_merged[late_col].notna().any() else 0)
    global_vmax = max(vmax_e, change_merged[early_col].quantile(0.98) if change_merged[early_col].notna().any() else 1)

    fig_early = px.choropleth_mapbox(
        change_merged, geojson=change_json, locations=change_merged.index,
        color=early_col,
        color_continuous_scale="YlGnBu",
        range_color=[global_vmin, global_vmax],
        hover_name="district_name",
        hover_data={"state_name": True, early_col: ":.3f"},
        mapbox_style="open-street-map",
        center={"lat": 22, "lon": 82}, zoom=3.0, opacity=0.65,
    )
    fig_early.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        paper_bgcolor=PAPER_BG, coloraxis_showscale=False,
        font=dict(family=FONT_SANS, size=10),
    )

    # --- Late period map ---
    fig_late = px.choropleth_mapbox(
        change_merged, geojson=change_json, locations=change_merged.index,
        color=late_col,
        color_continuous_scale="YlGnBu",
        range_color=[global_vmin, global_vmax],
        hover_name="district_name",
        hover_data={"state_name": True, late_col: ":.3f"},
        mapbox_style="open-street-map",
        center={"lat": 22, "lon": 82}, zoom=3.0, opacity=0.65,
    )
    fig_late.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        paper_bgcolor=PAPER_BG, coloraxis_showscale=False,
        font=dict(family=FONT_SANS, size=10),
    )

    # --- Change map ---
    vmax_c = change_merged[metric].abs().quantile(0.95) if change_merged[metric].notna().any() else 1
    fig_change = px.choropleth_mapbox(
        change_merged, geojson=change_json, locations=change_merged.index,
        color=metric,
        color_continuous_scale="RdBu", range_color=[-vmax_c, vmax_c],
        color_continuous_midpoint=0,
        hover_name="district_name",
        hover_data={"state_name": True, metric: ":.3f"},
        labels={metric: metric_label},
        mapbox_style="open-street-map",
        center={"lat": 22, "lon": 82}, zoom=3.0, opacity=0.65,
    )
    fig_change.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        paper_bgcolor=PAPER_BG,
        font=dict(family=FONT_SANS, size=10),
    )

    # --- Histogram ---
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=df_c[metric].dropna(), nbinsx=50,
        marker_color=HEADER_BG, opacity=0.8,
    ))
    n_gain = (df_c[metric] > 0).sum()
    n_lose = (df_c[metric] < 0).sum()
    fig_hist.add_vline(x=0, line_dash="dash", line_color="#c0392b", line_width=1.5)
    fig_hist.add_annotation(
        x=0.5, y=0.92, xref="paper", yref="paper",
        text=f"Gaining: {n_gain} | Losing: {n_lose}",
        showarrow=False, font=dict(size=10, color="#333"),
    )
    fig_hist.update_layout(
        **PLOT_LAYOUT_DEFAULTS,
        margin=dict(l=40, r=10, t=10, b=30),
        showlegend=False,
    )

    # --- Change by regime ---
    df_c_irr = df_c.copy()
    irr_map_dict = df_main.set_index("district_key")["irrigation_regime"].to_dict()
    df_c_irr["irrigation_regime"] = df_c_irr["district_key"].map(irr_map_dict)
    regime_change = df_c_irr.dropna(subset=["irrigation_regime"]).groupby("irrigation_regime")[metric].agg(["mean", "median"]).reset_index()

    fig_regime = go.Figure()
    fig_regime.add_trace(go.Bar(
        x=regime_change["irrigation_regime"].apply(lambda x: x.split("(")[0].strip()),
        y=regime_change["mean"],
        marker_color=[REGIME_COLORS.get(r, "#999") for r in regime_change["irrigation_regime"]],
    ))
    fig_regime.update_layout(
        **PLOT_LAYOUT_DEFAULTS,
        yaxis_title=metric_label,
        margin=dict(l=50, r=10, t=10, b=40),
        showlegend=False,
    )

    # --- Summary stats ---
    change_vals = df_c[metric].dropna()
    summary_children = [
        html.Div([
            html.Span("Mean change: ", style={"fontWeight": "600", "fontSize": "11px"}),
            html.Span(f"{change_vals.mean():.4f}", style={"fontSize": "11px"}),
        ], style={"marginBottom": "3px"}),
        html.Div([
            html.Span("Median change: ", style={"fontWeight": "600", "fontSize": "11px"}),
            html.Span(f"{change_vals.median():.4f}", style={"fontSize": "11px"}),
        ], style={"marginBottom": "3px"}),
        html.Div([
            html.Span("Std dev: ", style={"fontWeight": "600", "fontSize": "11px"}),
            html.Span(f"{change_vals.std():.4f}", style={"fontSize": "11px"}),
        ], style={"marginBottom": "3px"}),
        html.Hr(style={"margin": "6px 0"}),
        html.Div([
            html.Span(f"{n_gain}", style={"color": "#27ae60", "fontWeight": "700", "fontSize": "14px"}),
            html.Span(" districts gained", style={"fontSize": "11px"}),
        ], style={"marginBottom": "2px"}),
        html.Div([
            html.Span(f"{n_lose}", style={"color": "#c0392b", "fontWeight": "700", "fontSize": "14px"}),
            html.Span(" districts declined", style={"fontSize": "11px"}),
        ], style={"marginBottom": "2px"}),
        html.Div([
            html.Span(f"{(df_c[metric] == 0).sum()}", style={"color": "#7f8c8d", "fontWeight": "700", "fontSize": "14px"}),
            html.Span(" no change", style={"fontSize": "11px"}),
        ]),
    ]

    # --- Gainers table ---
    top_gain = df_c.nlargest(15, metric)[["district_name", "state_name", early_col, late_col, metric]].copy()
    top_gain.columns = ["District", "State", "Early", "Late", "Change"]
    for c in ["Early", "Late", "Change"]:
        top_gain[c] = top_gain[c].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "")

    gainers_table = dash_table.DataTable(
        data=top_gain.to_dict("records"),
        columns=[{"name": c, "id": c} for c in top_gain.columns],
        style_header=TABLE_STYLE_HEADER,
        style_data=TABLE_STYLE_DATA,
        style_data_conditional=[
            *TABLE_STYLE_DATA_CONDITIONAL,
            {"if": {"column_id": "Change"}, "color": "#27ae60", "fontWeight": "600"},
        ],
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "fontSize": "11px"},
    )

    # --- Losers table ---
    top_lose = df_c.nsmallest(15, metric)[["district_name", "state_name", early_col, late_col, metric]].copy()
    top_lose.columns = ["District", "State", "Early", "Late", "Change"]
    for c in ["Early", "Late", "Change"]:
        top_lose[c] = top_lose[c].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "")

    losers_table = dash_table.DataTable(
        data=top_lose.to_dict("records"),
        columns=[{"name": c, "id": c} for c in top_lose.columns],
        style_header=TABLE_STYLE_HEADER,
        style_data=TABLE_STYLE_DATA,
        style_data_conditional=[
            *TABLE_STYLE_DATA_CONDITIONAL,
            {"if": {"column_id": "Change"}, "color": "#c0392b", "fontWeight": "600"},
        ],
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "fontSize": "11px"},
    )

    return fig_early, fig_change, fig_late, fig_hist, fig_regime, summary_children, gainers_table, losers_table


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Crop Diversity & Agro-Biodiversity Dashboard")
    print("  Open http://127.0.0.1:8050 in your browser")
    print("=" * 60 + "\n")
    app.run(debug=False, port=8050)
