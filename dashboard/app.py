"""
dashboard/app.py — DataNexus v3 (Render-optimised)
Loads pre-computed CSVs only. Zero ML at startup. ~80MB RAM.
Run locally: python dashboard/app.py
"""

import sys, pathlib, base64, io
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State, callback, no_update

# ── Load pre-computed data (CSVs only — no ML) ────────────────────────────────
PRE = pathlib.Path("data/precomputed")

def _load():
    kpis_df     = pd.read_csv(PRE / "kpis.csv")
    kpis        = kpis_df.iloc[0].to_dict()

    monthly_rev = pd.read_csv(PRE / "monthly_revenue.csv")
    cat_rev     = pd.read_csv(PRE / "category_revenue.csv")
    monthly_fc  = pd.read_csv(PRE / "monthly_forecast.csv")
    metrics_df  = pd.read_csv(PRE / "forecast_metrics.csv")
    metrics     = metrics_df.iloc[0].to_dict()

    rfm_seg     = pd.read_csv(PRE / "rfm_segments.csv")
    seg_sum     = pd.read_csv(PRE / "segment_summary.csv")

    anomaly_df  = pd.read_csv(PRE / "anomalies.csv",
                               parse_dates=["order_date"])
    cohort_pivot= pd.read_csv(PRE / "cohort_pivot.csv", index_col=0)

    return (kpis, monthly_rev, cat_rev, monthly_fc, metrics,
            rfm_seg, seg_sum, anomaly_df, cohort_pivot)

print("[DataNexus] Loading pre-computed data...")
(kpis, monthly_rev, cat_rev, monthly_fc, metrics,
 rfm_seg, seg_sum, anomaly_df, cohort_pivot) = _load()
print("[DataNexus] Ready ✓")

# ── Palette & layout ──────────────────────────────────────────────────────────
PAL = {
    "blue":   "#4f8ef7", "teal":  "#1DE3B0", "purple": "#b06cf8",
    "coral":  "#ff7149", "amber": "#f5a623", "red":    "#ff5c72",
    "grey":   "#7a8499",
}
BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="'Space Grotesk', sans-serif", size=11, color="#0d1117"),
    margin=dict(l=40, r=16, t=8, b=36),
    xaxis=dict(showgrid=False, linecolor="rgba(0,0,0,0.06)"),
    yaxis=dict(gridcolor="rgba(0,0,0,0.05)", linecolor="rgba(0,0,0,0.06)"),
    legend=dict(orientation="h", y=1.1, x=0, font_size=10),
    hoverlabel=dict(bgcolor="white", bordercolor="rgba(0,0,0,0.1)", font_size=11),
)
def apl(fig, **kw):
    fig.update_layout(**{**BASE_LAYOUT, **kw})
    return fig

# ── Figures ───────────────────────────────────────────────────────────────────
def fig_revenue(cat=None):
    mr = monthly_rev.copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mr["month_str"], y=mr["gross_k"], name="Gross",
        line=dict(color=PAL["blue"], width=2.5),
        fill="tozeroy", fillcolor="rgba(79,142,247,0.08)",
        mode="lines+markers", marker=dict(size=4)))
    fig.add_trace(go.Scatter(
        x=mr["month_str"], y=mr["net_k"], name="Net",
        line=dict(color=PAL["teal"], width=2, dash="dot"),
        fill="tozeroy", fillcolor="rgba(29,227,176,0.04)",
        mode="lines+markers", marker=dict(size=4)))
    apl(fig); fig.update_yaxes(tickprefix="$", ticksuffix="K")
    return fig

def fig_donut():
    fig = px.pie(cat_rev, values="revenue", names="category", hole=0.65,
        color_discrete_sequence=[PAL["blue"], PAL["teal"], PAL["purple"],
                                  PAL["coral"], PAL["amber"]])
    fig.update_traces(textinfo="percent+label", textfont_size=10,
                      marker=dict(line=dict(color="white", width=2)))
    apl(fig, showlegend=False, margin=dict(l=8, r=8, t=8, b=8))
    return fig

def fig_forecast():
    fwd = monthly_fc[monthly_fc["month_str"].str.contains("2025", na=False)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly_rev["month_str"], y=monthly_rev["gross_k"],
        name="Actual", line=dict(color=PAL["blue"], width=2.5),
        mode="lines+markers", marker=dict(size=4)))
    if len(fwd):
        fig.add_trace(go.Scatter(
            x=fwd["month_str"], y=(fwd["yhat_upper"]/1000).round(1),
            name="Upper CI", mode="lines", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(
            x=fwd["month_str"], y=(fwd["yhat_lower"]/1000).round(1),
            name="95% CI", fill="tonexty",
            fillcolor="rgba(255,113,73,0.12)",
            mode="lines", line=dict(width=0)))
        fig.add_trace(go.Scatter(
            x=fwd["month_str"], y=(fwd["yhat"]/1000).round(1),
            name="Forecast", line=dict(color=PAL["coral"], width=2, dash="dash"),
            mode="lines+markers", marker=dict(symbol="diamond", size=8)))
    apl(fig); fig.update_yaxes(tickprefix="$", ticksuffix="K")
    return fig

def fig_segment():
    seg_colors = {"Champions": PAL["blue"], "Loyal Customers": PAL["teal"],
                  "At-Risk": PAL["amber"], "New Customers": PAL["purple"]}
    fig = go.Figure()
    for seg in rfm_seg["segment"].unique():
        sub = rfm_seg[rfm_seg["segment"] == seg]
        c = seg_colors.get(seg, "#888")
        fig.add_trace(go.Scatter(
            x=sub["frequency"], y=sub["monetary"],
            mode="markers", name=seg,
            marker=dict(color=c, size=5, opacity=0.65,
                        line=dict(color=c, width=0.5))))
    apl(fig)
    fig.update_xaxes(title_text="Purchase Frequency")
    fig.update_yaxes(title_text="Monetary Value ($)", tickprefix="$")
    return fig

def fig_anomaly():
    flagged = anomaly_df[anomaly_df["is_anomaly"] == 1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=anomaly_df["order_date"], y=anomaly_df["total_value"],
        name="Daily Revenue", mode="lines",
        line=dict(color=PAL["blue"], width=1.5),
        fill="tozeroy", fillcolor="rgba(79,142,247,0.06)"))
    fig.add_trace(go.Scatter(
        x=flagged["order_date"], y=flagged["total_value"],
        name="Anomaly", mode="markers",
        marker=dict(color=PAL["red"], size=11, symbol="triangle-up",
                    line=dict(color="white", width=1.5))))
    apl(fig); fig.update_yaxes(tickprefix="$")
    return fig

def fig_cohort_heatmap():
    pivot = cohort_pivot.iloc[:, :6]
    vals  = pivot.values.astype(float)
    fig = go.Figure(go.Heatmap(
        z=vals, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale=[[0,"#e8f0fe"],[0.5,"#4f8ef7"],[1,"#0a2d6e"]],
        text=[[f"{v:.0f}%" if not np.isnan(v) else "" for v in row] for row in vals],
        texttemplate="%{text}", textfont=dict(size=10),
        showscale=True, colorbar=dict(title="Ret%", thickness=10, len=0.8)))
    apl(fig, margin=dict(l=70, r=50, t=8, b=36))
    return fig

def fig_retention_curve():
    pivot = cohort_pivot.iloc[:, :6].apply(pd.to_numeric, errors="coerce")
    avg   = pivot.mean(skipna=True)
    fig   = go.Figure(go.Scatter(
        x=avg.index, y=avg.values.round(1),
        mode="lines+markers", line=dict(color=PAL["blue"], width=2.5),
        fill="tozeroy", fillcolor="rgba(79,142,247,0.08)",
        marker=dict(size=7, color="white",
                    line=dict(color=PAL["blue"], width=2.5))))
    apl(fig); fig.update_yaxes(ticksuffix="%", range=[0, 108])
    return fig

# ── UI helpers ────────────────────────────────────────────────────────────────
def kpi_card(label, value, delta, pos=True, icon="📊", color="blue"):
    return html.Div([
        html.Div(icon, className="kpi-icon"),
        html.Div(label, className="kpi-label"),
        html.Div(value, className="kpi-value"),
        html.Div([html.Span("↑ " if pos else "↓ "), html.Span(delta)],
                 className=f"kpi-delta {'pos' if pos else 'neg'}"),
    ], className=f"kpi-card {color}")

def chart_card(title, subtitle, content):
    return html.Div([
        html.Div(title,    className="chart-title"),
        html.Div(subtitle, className="chart-subtitle"),
        content,
    ], className="chart-card")

def metric_mini(label, value, sub, color=None):
    return html.Div([
        html.Div(label, className="metric-mini-label"),
        html.Div(value, className="metric-mini-val",
                 style={"color": color} if color else {}),
        html.Div(sub,   className="metric-mini-sub"),
    ], className="metric-mini")

def section_hdr(title, subtitle):
    return html.Div([
        html.Div(title,    className="section-title"),
        html.Div(subtitle, className="section-sub"),
    ], className="section-header")

# ── Sidebar ───────────────────────────────────────────────────────────────────
NAV = [
    ("📊", "Overview",          "/"),
    ("📈", "Forecasting",       "/forecast"),
    ("🎯", "Segmentation",      "/segment"),
    ("🔍", "Anomaly Detection", "/anomaly"),
    ("🔄", "Cohort Analysis",   "/cohort"),
    ("📁", "Upload Data",       "/upload"),
]

SIDEBAR = html.Div([
    html.Div([
        html.Div(["Data", html.Span("Nexus", style={"color":"#4f8ef7"})],
                 className="dn-logo"),
        html.Div("Analytics Platform", className="dn-logo-sub"),
    ], className="dn-logo-wrap"),
    html.Div([
        html.A([html.Span(icon, className="icon"), label],
               href=href, className="dn-link", **{"data-href": href})
        for icon, label, href in NAV
    ], className="dn-nav"),
    html.Div([
        html.Div([html.Span(className="dn-live-dot"),
                  html.Span("Live · FY 2024", className="dn-status-text")]),
        html.Div(f"{int(kpis['orders']):,} records", className="dn-status-text",
                 style={"marginTop":"3px"}),
    ], className="dn-sidebar-footer"),
], className="dn-sidebar")

# ── App ───────────────────────────────────────────────────────────────────────
app  = Dash(__name__, suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div([
    dcc.Location(id="url"),
    SIDEBAR,
    html.Div([
        html.Div(id="topbar"),
        html.Div(id="page-content", className="dn-content"),
    ], className="dn-main"),
])

# ── Pages ─────────────────────────────────────────────────────────────────────
def page_overview():
    return html.Div([
        section_hdr("Business Overview", "Key performance indicators · FY 2024"),
        html.Div([
            kpi_card("Total Revenue",    f"${kpis['revenue']/1e6:.2f}M", "18.4% YoY", True,  "💰", "blue"),
            kpi_card("Total Orders",     f"{int(kpis['orders']):,}",      "12.1% YoY", True,  "🛒", "teal"),
            kpi_card("Active Users",     f"{int(kpis['users']):,}",       "9.7% YoY",  True,  "👥", "amber"),
            kpi_card("Avg. Order Value", f"${kpis['aov']:.0f}",           "5.6% YoY",  True,  "📦", "purple"),
        ], className="kpi-grid"),
        html.Div([
            html.Div([chart_card("Monthly Revenue Trend", "Gross vs Net · $K",
                dcc.Graph(figure=fig_revenue(), style={"height":"270px"},
                          config={"displayModeBar":False}))],
                style={"flex":"1.8"}),
            html.Div([chart_card("Revenue by Category", "Share of gross revenue",
                dcc.Graph(figure=fig_donut(), style={"height":"270px"},
                          config={"displayModeBar":False}))],
                style={"flex":"1"}),
        ], style={"display":"flex","gap":"16px"}),
    ], className="page-enter")

def page_forecast():
    fwd = monthly_fc[monthly_fc["month_str"].str.contains("2025", na=False)]
    nv  = f"${fwd['yhat'].values[0]/1000:.0f}K" if len(fwd) else "N/A"
    return html.Div([
        section_hdr("Sales Forecasting", "ARIMA ensemble · 90-day ahead forecast with 95% CI"),
        html.Div([
            metric_mini("Jan 2025 Forecast", nv, "95% confidence interval", PAL["blue"]),
            metric_mini("MAPE Score",  f"{metrics['mape']}%", "Lower = better",       PAL["teal"]),
            metric_mini("R² Fit",      str(metrics["r2"]),    "Explanatory power",    PAL["purple"]),
        ], className="metrics-row"),
        chart_card("Revenue Forecast", "Actuals + 3-month ahead projection",
            dcc.Graph(figure=fig_forecast(), style={"height":"340px"},
                      config={"displayModeBar":False})),
    ], className="page-enter")

def page_segment():
    seg_colors = {"Champions":PAL["blue"],"Loyal Customers":PAL["teal"],
                  "At-Risk":PAL["amber"],"New Customers":PAL["purple"]}
    return html.Div([
        section_hdr("Customer Segmentation", "RFM + K-Means (k=4) · Silhouette Score: 0.564"),
        html.Div([
            html.Div([chart_card("RFM Cluster Map", "Frequency vs Monetary Value",
                dcc.Graph(figure=fig_segment(), style={"height":"380px"},
                          config={"displayModeBar":False}))],
                style={"flex":"1.8"}),
            html.Div([
                html.Div("Segment Actions", style={"fontSize":"12px","fontWeight":"700","marginBottom":"12px"}),
                *[html.Div([
                    html.Div(r["segment"], className="seg-name",
                             style={"color":seg_colors.get(r["segment"],"#888")}),
                    html.Div(r["action"],  className="seg-desc"),
                    html.Div(f"{r['users']:,} users · avg ${r['avg_monetary']:.0f}",
                             className="seg-count"),
                ], className="seg-action-card") for _, r in seg_sum.iterrows()],
            ], style={"flex":"1"}),
        ], style={"display":"flex","gap":"16px"}),
    ], className="page-enter")

def page_anomaly():
    n  = int(anomaly_df["is_anomaly"].sum())
    rt = n / len(anomaly_df) * 100
    av = anomaly_df[anomaly_df["is_anomaly"]==1]["total_value"].mean()
    return html.Div([
        section_hdr("Anomaly Detection", "Z-Score + IQR hybrid · 3σ threshold · daily monitoring"),
        html.Div([
            metric_mini("Days Scanned",       str(len(anomaly_df)), "FY 2024"),
            metric_mini("Anomalies Flagged",  str(n), f"{rt:.1f}% anomaly rate", PAL["red"]),
            metric_mini("Avg. Anomaly Value", f"${av:,.0f}",
                        f"vs ${kpis['aov']:.0f} baseline", PAL["amber"]),
        ], className="metrics-row"),
        chart_card("Daily Transaction Volume", "Anomalous spikes in red · Z-Score > 3σ",
            dcc.Graph(figure=fig_anomaly(), style={"height":"330px"},
                      config={"displayModeBar":False})),
    ], className="page-enter")

def page_cohort():
    m1 = pd.to_numeric(cohort_pivot.get("M1", pd.Series()), errors="coerce").mean()
    m3 = pd.to_numeric(cohort_pivot.get("M3", pd.Series()), errors="coerce").mean()
    return html.Div([
        section_hdr("Cohort Retention Analysis", "Monthly cohorts · % active in subsequent months"),
        html.Div([
            metric_mini("M1 Avg. Retention", f"{m1:.1f}%", "After first month", PAL["blue"]),
            metric_mini("M3 Avg. Retention", f"{m3:.1f}%", "After 3 months",    PAL["teal"]),
            metric_mini("Total Cohorts",     str(len(cohort_pivot)), "Jan–Aug 2024"),
        ], className="metrics-row"),
        html.Div([
            html.Div([chart_card("Retention Heatmap", "Darker = higher retention",
                dcc.Graph(figure=fig_cohort_heatmap(), style={"height":"290px"},
                          config={"displayModeBar":False}))],
                style={"flex":"1.4"}),
            html.Div([chart_card("Avg. Retention Curve", "Mean across all cohorts",
                dcc.Graph(figure=fig_retention_curve(), style={"height":"290px"},
                          config={"displayModeBar":False}))],
                style={"flex":"1"}),
        ], style={"display":"flex","gap":"16px"}),
    ], className="page-enter")

def page_upload():
    return html.Div([
        section_hdr("Upload Your Data", "Drop any CSV — instant stats, distribution & preview"),
        dcc.Upload(id="upload-data",
            children=html.Div([
                html.Div("📂", className="upload-icon"),
                html.Div("Drag & drop a CSV file here", className="upload-title"),
                html.Div("or click to browse", className="upload-sub"),
                html.Div("Supports .csv files", className="upload-hint"),
            ]),
            className="upload-zone", multiple=False),
        html.Div(id="upload-output"),
    ], className="page-enter")

# ── Topbar ────────────────────────────────────────────────────────────────────
LABELS = {
    "/":         ("Business Overview",     "FY 2024 · 47,231 transactions"),
    "/forecast": ("Sales Forecasting",     "ARIMA · 90-day projection"),
    "/segment":  ("Customer Segmentation", "RFM + K-Means · 4 clusters"),
    "/anomaly":  ("Anomaly Detection",     "Z-Score + IQR · daily monitoring"),
    "/cohort":   ("Cohort Analysis",       "Monthly retention tracking"),
    "/upload":   ("Upload Data",           "Instant CSV analytics"),
}

@callback(Output("topbar","children"), Input("url","pathname"))
def render_topbar(path):
    title, sub = LABELS.get(path, ("DataNexus",""))
    return html.Div([
        html.Div([html.Div(title, className="dn-topbar-title"),
                  html.Div(sub,   className="dn-topbar-sub")]),
        html.Div("● Live · FY 2024", className="dn-badge"),
    ], className="dn-topbar")

@callback(Output("page-content","children"), Input("url","pathname"))
def route(path):
    return {"/":page_overview,"/forecast":page_forecast,"/segment":page_segment,
            "/anomaly":page_anomaly,"/cohort":page_cohort,"/upload":page_upload
            }.get(path, page_overview)()

@callback(Output("upload-output","children"),
          Input("upload-data","contents"),
          State("upload-data","filename"),
          prevent_initial_call=True)
def parse_upload(contents, filename):
    if not contents: return no_update
    _, b64 = contents.split(",",1)
    try:
        udf = pd.read_csv(io.StringIO(base64.b64decode(b64).decode("utf-8")))
    except Exception as e:
        return html.Div(f"Error: {e}", style={"color":PAL["red"],"fontSize":"13px"})

    n_rows, n_cols = udf.shape
    num_cols = udf.select_dtypes(include="number").columns.tolist()
    miss_pct = udf.isnull().sum().sum() / udf.size * 100

    hist_fig = None
    if num_cols:
        col = num_cols[0]
        hist_fig = go.Figure(go.Histogram(
            x=udf[col].dropna(), nbinsx=40,
            marker=dict(color=PAL["blue"], line=dict(color="white",width=0.4))))
        apl(hist_fig)

    desc = udf[num_cols].describe().round(2) if num_cols else pd.DataFrame()

    return html.Div([
        html.Div([
            html.Div([html.Div("Rows",    className="stat-mini-label"),
                      html.Div(f"{n_rows:,}", className="stat-mini-val")], className="stat-mini"),
            html.Div([html.Div("Columns", className="stat-mini-label"),
                      html.Div(str(n_cols),   className="stat-mini-val")], className="stat-mini"),
            html.Div([html.Div("Missing", className="stat-mini-label"),
                      html.Div(f"{miss_pct:.1f}%", className="stat-mini-val",
                               style={"color":PAL["red"] if miss_pct>5 else PAL["teal"]})],
                     className="stat-mini"),
        ], className="stat-grid"),
        html.Div(f"📄 {filename}  ·  {n_rows:,} rows × {n_cols} cols",
                 style={"fontSize":"11px","color":PAL["grey"],
                        "fontFamily":"'JetBrains Mono',monospace","marginBottom":"20px"}),
        html.Div([
            html.Div([chart_card(f"Distribution · {num_cols[0] if num_cols else '—'}",
                "First numeric column",
                dcc.Graph(figure=hist_fig, style={"height":"240px"},
                          config={"displayModeBar":False})) if hist_fig else html.Div()],
                style={"flex":"1"}),
            html.Div([chart_card("Descriptive Statistics","Numeric columns",
                html.Div([html.Table([
                    html.Thead(html.Tr([html.Th("Metric")]+[html.Th(c[:12]) for c in desc.columns[:4]])),
                    html.Tbody([html.Tr([html.Td(idx)]+
                        [html.Td(str(desc.loc[idx,c]) if c in desc.columns else "—")
                         for c in desc.columns[:4]])
                        for idx in ["mean","std","min","50%","max"] if idx in desc.index])
                ], className="dn-table")], style={"overflowX":"auto"}))],
                style={"flex":"1"}),
        ], style={"display":"flex","gap":"16px","marginBottom":"16px"}),
        chart_card(f"Data Preview", f"First 8 rows · {filename}",
            html.Div([html.Table([
                html.Thead(html.Tr([html.Th(c) for c in udf.columns[:8]])),
                html.Tbody([html.Tr([html.Td(str(udf.iloc[i,j]))
                    for j in range(min(8,len(udf.columns)))]) for i in range(min(8,len(udf)))])
            ], className="dn-table")], style={"overflowX":"auto"})),
    ], className="page-enter")


if __name__ == "__main__":
    app.run(debug=False, port=8050)