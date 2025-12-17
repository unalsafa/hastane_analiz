# hastane_analiz/dashboard/dashboard_dogum.py

import pandas as pd
import streamlit as st
import altair as alt
import plotly.express as px
import plotly.graph_objects as go

from hastane_analiz.db.connection import get_connection

# =========================================================
# TEMA / RENK PALETİ (Sağlık Bakanlığı hissi)
# =========================================================
MINISTRY_TURQUOISE = "#00A3B4"   # turkuaz
MINISTRY_TURQUOISE_DARK = "#007C8A"
PRIMARY_GREEN = "#0B6E4F"
ACCENT_BLUE = "#0EA5E9"
WARN_COLOR = "#F59E0B"
DANGER_COLOR = "#EF4444"

BG_APP = "#EEF2F6"
BG_CARD = "#FFFFFF"
BORDER = "#D8DEE9"
TEXT = "#0F172A"
MUTED = "#64748B"

PLOT_TEMPLATE = "plotly_white"


# =========================================================
# CSS / UI
# =========================================================
def inject_css():
    TOPBAR_H = 86  # topbar yüksekliği (px) — gerekirse 78/90 oynarsın

    st.markdown(
        f"""
        <style>
        /* =========================
           APP BACKGROUND / LAYOUT
        ========================== */
        .stApp {{
            background: {BG_APP};
        }}

        /* Topbar fixed olduğu için içerik topbar altından başlasın */
        div.block-container {{
            padding-top: {TOPBAR_H + 18}px !important;
            padding-bottom: 2rem;
        }}

        /* =========================
           DEPLOY GİZLE (DOM'una göre)
        ========================== */
        div[data-testid="stAppDeployButton"] {{
            display: none !important;
        }}
        button[aria-label="Deploy"], button[title="Deploy"] {{
            display: none !important;
        }}

        /* =========================
           STREAMLIT CHROME
           (DOM KALSIN -> OK ÇALIŞIR)
        ========================== */
        header[data-testid="stHeader"] {{
            background: transparent !important;
            height: 0px !important;
            border: 0 !important;
        }}
        [data-testid="stToolbar"] {{
            background: transparent !important;
            height: 0px !important;
        }}
        #MainMenu {{ visibility: hidden !important; }}
        footer {{ visibility: hidden !important; }}

        /* =========================
           SIDEBAR (Bakanlık turkuazı)
        ========================== */
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {MINISTRY_TURQUOISE_DARK} 0%, {MINISTRY_TURQUOISE} 100%);
            border-right: 0px;
        }}
        section[data-testid="stSidebar"] * {{
            color: white !important;
        }}
        section[data-testid="stSidebar"] .stMarkdown,
        section[data-testid="stSidebar"] label {{
            color: white !important;
        }}

        /* Sidebar inputs */
        section[data-testid="stSidebar"] .stSelectbox > div,
        section[data-testid="stSidebar"] .stMultiSelect > div,
        section[data-testid="stSidebar"] .stTextInput > div {{
            background: rgba(255,255,255,0.12) !important;
            border-radius: 10px !important;
            border: 1px solid rgba(255,255,255,0.25) !important;
        }}
        section[data-testid="stSidebar"] input {{
            color: white !important;
        }}

        /* =========================
           TOPBAR (FIXED NAVBAR)
        ========================== */
        .topbar {{
            position: fixed;
            top: 10px;
            left: 200px;
            right: 5px;
            z-index: 9999;

            background: {BG_CARD};
            border: 1px solid {BORDER};
            border-radius: 16px;
            padding: 14px 16px;
            box-shadow: 0 14px 34px rgba(2, 6, 23, 0.10);

            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
        }}

        /* Sol tarafta sidebar oku için boşluk */
        .topbar-left {{
            display: flex;
            align-items: center;
            gap: 12px;
            min-width: 280px;
            margin-left: 56px;
        }}

        .logo-badge {{
            width: 34px;
            height: 34px;
            border-radius: 10px;
            background: rgba(11,110,79,0.10);
            border: 1px solid rgba(11,110,79,0.20);
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            color: {PRIMARY_GREEN};
        }}

        .topbar-title {{
            font-weight: 800;
            color: {TEXT};
            margin: 0;
            line-height: 1.1;
        }}
        .topbar-sub {{
            font-size: 12px;
            color: {MUTED};
            margin-top: 2px;
        }}

        .topbar-center {{
            flex: 1;
            display: flex;
            justify-content: center;
        }}
        .search-pill {{
            width: min(520px, 100%);
            background: #F1F5F9;
            border: 1px solid {BORDER};
            border-radius: 999px;
            padding: 10px 14px;
            color: {MUTED};
            font-size: 13px;
        }}

        .topbar-right {{
            display: flex;
            align-items: center;
            gap: 10px;
            min-width: 120px;
            justify-content: flex-end;
        }}
        .lang-pill {{
            width: 34px;
            height: 34px;
            border-radius: 999px;
            background: rgba(22,163,74,0.12);
            border: 1px solid rgba(22,163,74,0.25);
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            color: {PRIMARY_GREEN};
        }}

        /* =========================
           SIDEBAR COLLAPSE/EXPAND BUTTON
           TOPBAR ÜSTÜNE SABİTLE
        ========================== */
        button[data-testid="stSidebarCollapseButton"],
        button[data-testid="stSidebarExpandButton"] {{
            position: fixed !important;
            top: 22px !important;   /* topbar içine hizalı */
            left: 28px !important;
            z-index: 10000 !important;

            background: #FFFFFF !important;
            border: 1px solid {BORDER} !important;
            border-radius: 12px !important;
            box-shadow: 0 10px 26px rgba(2, 6, 23, 0.12) !important;

            width: 44px !important;
            height: 44px !important;

            display: hidden !important;
            align-items: center !important;
            justify-content: center !important;
        }}

        button[data-testid="stSidebarCollapseButton"] span[data-testid="stIconMaterial"],
        button[data-testid="stSidebarExpandButton"] span[data-testid="stIconMaterial"] {{
            color: rgba(49, 51, 63, 0.75) !important;
        }}

        /* =========================
           CARDS + SECTION TITLE BAR
        ========================== */
        .card {{
            background: {BG_CARD};
            border: 1px solid {BORDER};
            border-radius: 16px;
            padding: 14px 16px;
            box-shadow: 0 10px 26px rgba(2, 6, 23, 0.06);
        }}
        .section-title {{
            background: linear-gradient(90deg, {MINISTRY_TURQUOISE_DARK} 0%, {MINISTRY_TURQUOISE} 100%);
            color: white;
            font-weight: 800;
            border-radius: 12px 12px 0 0;
            padding: 10px 14px;
            margin: -14px -16px 12px -16px;
            font-size: 14px;
        }}
        .section-gap {{
            height: 12px;
        }}

        .stDataFrame {{
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid {BORDER};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )





def topbar():
    st.markdown(
        f"""
        <div class="topbar">
            <div class="topbar-left">
                <div class="logo-badge">SB</div>
                <div>
                    <div class="topbar-title">Doğum Dashboard</div>
                    <div class="topbar-sub">Halk Sağlığı • Doğum • Analiz</div>
                </div>
            </div>
            <div class="topbar-center">
                <div class="search-pill">🔍  Hastane / İlçe / Başkanlık ara</div>
            </div>
            <div class="topbar-right">
                <div class="lang-pill">TR</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# DB
# =========================================================
def read_sql(query: str, params=None) -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql(query, conn, params=params)


@st.cache_data(ttl=600)
def load_dogum_data() -> pd.DataFrame:
    return read_sql("SELECT * FROM hastane_analiz.v_dogum_oran;")


# =========================================================
# DATA HELPERS
# =========================================================
def _ensure_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    return df


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = _ensure_cols(
        df,
        [
            "yil",
            "ay",
            "baskanlik_adi",
            "ilce_adi",
            "birim_adi",
            "kurum_rol_adi",
            "normal_dogum",
            "mudahaleli_dogum",
            "sezaryen_dogum",
            "primer_sezaryen",
            "canli_bebek",
            "olu_bebek",
            "dogum_salon_sayisi",
            "dogum_masa_sayisi",
            "tdl_oda_sayisi",
        ],
    )

    df["toplam_dogum"] = (
        df["normal_dogum"].fillna(0)
        + df["mudahaleli_dogum"].fillna(0)
        + df["sezaryen_dogum"].fillna(0)
    )

    df["toplam_dogum_safe"] = df["toplam_dogum"].replace({0: pd.NA})

    df["sez_oran"] = df["sezaryen_dogum"] / df["toplam_dogum_safe"] * 100
    df["primer_oran"] = df["primer_sezaryen"] / df["sezaryen_dogum"].replace({0: pd.NA}) * 100
    df["olu_oran"] = (
        df["olu_bebek"] / (df["canli_bebek"] + df["olu_bebek"]).replace({0: pd.NA})
    ) * 100

    # capacity normalized (optional)
    for col in ["dogum_salon_sayisi", "dogum_masa_sayisi", "tdl_oda_sayisi"]:
        denom = df[col].replace({0: pd.NA})
        df[f"{col}_basina_dogum"] = df["toplam_dogum"] / denom

    return df


def fmt_int(x) -> str:
    try:
        return f"{int(round(float(x))):,}".replace(",", ".")
    except Exception:
        return "0"


def fmt_pct(x, digits=1) -> str:
    if x is None or pd.isna(x):
        return "-"
    try:
        return f"{float(x):.{digits}f} %"
    except Exception:
        return "-"


def card_kpi(title: str, value: str, sub: str = ""):
    st.markdown(
        f"""
        <div class="card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def card_role(title: str, value: str, sub: str, pill: str = "", pill_kind: str = ""):
    pill_html = ""
    if pill:
        cls = "pill"
        if pill_kind == "green":
            cls = "pill pill-green"
        elif pill_kind == "red":
            cls = "pill pill-red"
        pill_html = f"""<span class="{cls}">{pill}</span>"""

    st.markdown(
        f"""
        <div class="card">
            <div style="display:flex; justify-content:space-between; align-items:center; gap:10px;">
                <div class="kpi-title" style="margin:0">{title}</div>
                <div>{pill_html}</div>
            </div>
            <div class="kpi-value" style="font-size:20px">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# FILTERS
# =========================================================
def filter_block(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.markdown("## Filtreler")

    yil_list = sorted(df["yil"].dropna().unique().tolist())
    yil = st.sidebar.selectbox("Yıl", yil_list, index=len(yil_list) - 1 if yil_list else 0)

    ay_list = sorted(df.loc[df["yil"] == yil, "ay"].dropna().unique().tolist())
    ay = st.sidebar.multiselect("Ay", ay_list, default=ay_list)

    baskanlik = st.sidebar.multiselect(
        "Başkanlık",
        sorted(df["baskanlik_adi"].dropna().unique().tolist()),
        default=[],
    )
    ilce = st.sidebar.multiselect(
        "İlçe",
        sorted(df["ilce_adi"].dropna().unique().tolist()),
        default=[],
    )
    kurum_rol = st.sidebar.multiselect(
        "Kurum Rolü (A1/A2/B/C...)",
        sorted(df["kurum_rol_adi"].dropna().unique().tolist()),
        default=[],
    )

    f = df[df["yil"] == yil].copy()
    f = f[f["ay"].isin(ay)]

    if baskanlik:
        f = f[f["baskanlik_adi"].isin(baskanlik)]
    if ilce:
        f = f[f["ilce_adi"].isin(ilce)]
    if kurum_rol:
        f = f[f["kurum_rol_adi"].isin(kurum_rol)]

    return f


# =========================================================
# BLOCKS
# =========================================================
def kpi_row(df: pd.DataFrame):
    toplam = df["toplam_dogum"].sum()
    normal = df["normal_dogum"].sum()
    sez = df["sezaryen_dogum"].sum()
    primer = df["primer_sezaryen"].sum()
    canli = df["canli_bebek"].sum()
    olu = df["olu_bebek"].sum()

    sez_oran = (sez / toplam * 100) if toplam else pd.NA
    primer_oran = (primer / sez * 100) if sez else pd.NA
    olu_oran = (olu / (canli + olu) * 100) if (canli + olu) else pd.NA

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        card_kpi("Toplam Doğum", fmt_int(toplam), "Seçili dönem")
    with c2:
        card_kpi("Normal Doğum", fmt_int(normal), "Seçili dönem")
    with c3:
        card_kpi("Sezaryen", fmt_int(sez), "Seçili dönem")
    with c4:
        card_kpi("Sezaryen Oranı", fmt_pct(sez_oran, 1), "Seçili dönem")
    with c5:
        card_kpi("Primer Oranı", fmt_pct(primer_oran, 1), "Primer / Sezaryen")
    with c6:
        card_kpi("Ölü Doğum Oranı", fmt_pct(olu_oran, 2), "Ölü / (Canlı+Ölü)")


def trend_block(df: pd.DataFrame):
    # Trend: toplam + normal + sezaryen (kullanıcının isteği)
    trend = (
        df.groupby(["yil", "ay"], as_index=False)
        .agg(
            toplam=("toplam_dogum", "sum"),
            normal=("normal_dogum", "sum"),
            sezaryen=("sezaryen_dogum", "sum"),
        )
        .sort_values(["yil", "ay"])
    )
    trend["tarih"] = pd.to_datetime(dict(year=trend["yil"], month=trend["ay"], day=1))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=trend["tarih"],
            y=trend["toplam"],
            mode="lines+markers",
            name="Toplam Doğum",
            line=dict(color=PRIMARY_GREEN, width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=trend["tarih"],
            y=trend["normal"],
            mode="lines+markers",
            name="Normal Doğum",
            line=dict(color=MINISTRY_TURQUOISE, width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=trend["tarih"],
            y=trend["sezaryen"],
            mode="lines+markers",
            name="Sezaryen Doğum",
            line=dict(color=ACCENT_BLUE, width=3),
        )
    )

    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=380,
        margin=dict(l=10, r=10, t=10, b=10),
        legend_title_text="",
        yaxis_title=None,
        xaxis_title=None,
    )
    st.plotly_chart(fig, use_container_width=True)


def baskanlik_sezaryen_bar_vertical(df: pd.DataFrame):
    # Dikey grouped bar (kullanıcının isteği)
    g = (
        df.groupby("baskanlik_adi", as_index=False)
        .agg(
            toplam_dogum=("toplam_dogum", "sum"),
            sez=("sezaryen_dogum", "sum"),
            primer=("primer_sezaryen", "sum"),
        )
    )
    g["sez_oran"] = g["sez"] / g["toplam_dogum"].replace({0: pd.NA}) * 100
    g["primer_oran"] = g["primer"] / g["sez"].replace({0: pd.NA}) * 100

    # Sırala: sezaryen oranına göre (yüksekten düşüğe)
    g = g.sort_values("sez_oran", ascending=False)

    long = g.melt(
        id_vars=["baskanlik_adi"],
        value_vars=["sez_oran", "primer_oran"],
        var_name="Gösterge",
        value_name="Oran",
    )
    long["Gösterge"] = long["Gösterge"].map(
        {"sez_oran": "Sezaryen Oranı", "primer_oran": "Primer Sezaryen Oranı"}
    )

    fig = px.bar(
        long,
        x="baskanlik_adi",
        y="Oran",
        color="Gösterge",
        barmode="group",
        template=PLOT_TEMPLATE,
        labels={"baskanlik_adi": "Başkanlık", "Oran": "Oran (%)"},
    )

    # kurumsal renkler
    fig.for_each_trace(
        lambda t: t.update(
            marker_color=PRIMARY_GREEN if t.name == "Sezaryen Oranı" else ACCENT_BLUE
        )
    )

    # Referans çizgisi (örn %30)
    ref = 30
    fig.add_hline(
        y=ref,
        line_width=2,
        line_dash="dash",
        line_color="rgba(239,68,68,0.85)",
        annotation_text="Referans %30",
        annotation_position="top left",
    )

    fig.update_layout(
        height=420,
        legend_title_text="",
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_tickangle=-25,
    )

    st.plotly_chart(fig, use_container_width=True)


def baskanlik_olu_dogum_bar(df: pd.DataFrame):
    g = (
        df.groupby("baskanlik_adi", as_index=False)
        .agg(canli=("canli_bebek", "sum"), olu=("olu_bebek", "sum"))
    )
    g["olu_oran"] = g["olu"] / (g["canli"] + g["olu"]).replace({0: pd.NA}) * 100
    g = g.sort_values("olu_oran", ascending=False)

    fig = px.bar(
        g,
        x="baskanlik_adi",
        y="olu_oran",
        color="olu_oran",
        color_continuous_scale="Reds",
        template=PLOT_TEMPLATE,
        labels={"baskanlik_adi": "Başkanlık", "olu_oran": "Ölü Doğum Oranı (%)"},
    )
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), xaxis_tickangle=-25)
    st.plotly_chart(fig, use_container_width=True)


def kurum_rol_karsilastirma_cards(df: pd.DataFrame):
    overall_toplam = df["toplam_dogum"].sum()
    overall_sez_oran = (df["sezaryen_dogum"].sum() / overall_toplam * 100) if overall_toplam else pd.NA

    g = (
        df.groupby("kurum_rol_adi", as_index=False)
        .agg(
            toplam=("toplam_dogum", "sum"),
            sez=("sezaryen_dogum", "sum"),
            primer=("primer_sezaryen", "sum"),
            canli=("canli_bebek", "sum"),
            olu=("olu_bebek", "sum"),
            kurum_sayisi=("birim_adi", "nunique"),
        )
        .sort_values("toplam", ascending=False)
    )
    g["sez_oran"] = g["sez"] / g["toplam"].replace({0: pd.NA}) * 100
    g["primer_oran"] = g["primer"] / g["sez"].replace({0: pd.NA}) * 100
    g["olu_oran"] = g["olu"] / (g["canli"] + g["olu"]).replace({0: pd.NA}) * 100

    cols = st.columns(min(4, max(1, len(g))))
    for i, row in enumerate(g.itertuples(index=False)):
        delta = None
        if pd.notna(row.sez_oran) and pd.notna(overall_sez_oran):
            delta = row.sez_oran - overall_sez_oran

        pill = ""
        pill_kind = ""
        if delta is not None:
            pill = f"Sez Δ {delta:+.1f}"
            pill_kind = "red" if delta > 0 else "green"

        title = f"{row.kurum_rol_adi} • {int(row.kurum_sayisi)} kurum"
        value = f"Sez: {fmt_pct(row.sez_oran, 1)}"
        # ✅ toplam doğum sayısını ekledik
        sub = f"Toplam: {fmt_int(row.toplam)} • Primer: {fmt_pct(row.primer_oran, 1)} • Ölü: {fmt_pct(row.olu_oran, 2)}"

        with cols[i % len(cols)]:
            card_role(title=title, value=value, sub=sub, pill=pill, pill_kind=pill_kind)

    st.caption("Δ: seçili filtre kapsamındaki genel sezaryen ortalamasına göre farkı gösterir.")


def ranking_table(df: pd.DataFrame):
    grp = (
        df.groupby(["baskanlik_adi", "ilce_adi", "birim_adi"], as_index=False)
        .agg(
            toplam_dogum=("toplam_dogum", "sum"),
            sezaryen_dogum=("sezaryen_dogum", "sum"),
            primer_sezaryen=("primer_sezaryen", "sum"),
            canli_bebek=("canli_bebek", "sum"),
            olu_bebek=("olu_bebek", "sum"),
        )
    )

    grp["sez_oran"] = grp["sezaryen_dogum"] / grp["toplam_dogum"].replace({0: pd.NA}) * 100
    grp["primer_oran"] = grp["primer_sezaryen"] / grp["sezaryen_dogum"].replace({0: pd.NA}) * 100
    grp["olu_oran"] = grp["olu_bebek"] / (grp["canli_bebek"] + grp["olu_bebek"]).replace({0: pd.NA}) * 100

    sort_by = st.selectbox(
        "Sıralama kriteri",
        options=["sez_oran", "primer_oran", "olu_oran", "toplam_dogum"],
        index=0,
    )
    grp = grp.sort_values(by=sort_by, ascending=False)

    out = grp.copy()
    # ✅ oranları "42.6 %" formatına çevir
    out["sez_oran"] = out["sez_oran"].map(lambda x: fmt_pct(x, 1))
    out["primer_oran"] = out["primer_oran"].map(lambda x: fmt_pct(x, 1))
    out["olu_oran"] = out["olu_oran"].map(lambda x: fmt_pct(x, 2))
    out["toplam_dogum"] = out["toplam_dogum"].map(fmt_int)

    show_cols = [
        "baskanlik_adi",
        "ilce_adi",
        "birim_adi",
        "toplam_dogum",
        "sez_oran",
        "primer_oran",
        "olu_oran",
    ]

    st.dataframe(out[show_cols], use_container_width=True, hide_index=True)


def baskanlik_heatmap(df: pd.DataFrame):
    grp = (
        df.groupby("baskanlik_adi", as_index=False)
        .agg(
            toplam=("toplam_dogum", "sum"),
            sez=("sezaryen_dogum", "sum"),
            primer=("primer_sezaryen", "sum"),
            canli=("canli_bebek", "sum"),
            olu=("olu_bebek", "sum"),
        )
    )
    grp["sez_oran"] = grp["sez"] / grp["toplam"].replace({0: pd.NA}) * 100
    grp["primer_oran"] = grp["primer"] / grp["sez"].replace({0: pd.NA}) * 100
    grp["olu_oran"] = grp["olu"] / (grp["canli"] + grp["olu"]).replace({0: pd.NA}) * 100

    heat = grp.melt(
        id_vars=["baskanlik_adi"],
        value_vars=["sez_oran", "primer_oran", "olu_oran"],
        var_name="Gösterge",
        value_name="Değer",
    )

    heat["Gösterge"] = heat["Gösterge"].map(
        {"sez_oran": "Sezaryen", "primer_oran": "Primer", "olu_oran": "Ölü Doğum"}
    )

    chart = (
        alt.Chart(heat)
        .mark_rect()
        .encode(
            x=alt.X("Gösterge:N"),
            y=alt.Y("baskanlik_adi:N", title="Başkanlık"),
            color=alt.Color("Değer:Q", scale=alt.Scale(scheme="redyellowgreen")),
            tooltip=["baskanlik_adi", "Gösterge", alt.Tooltip("Değer:Q", format=".2f")],
        )
        .properties(height=320)
    )

    st.altair_chart(chart, use_container_width=True)


def kapasite_block(df: pd.DataFrame):
    kap_cols = [
        "dogum_salon_sayisi_basina_dogum",
        "dogum_masa_sayisi_basina_dogum",
        "tdl_oda_sayisi_basina_dogum",
    ]
    kap_cols = [c for c in kap_cols if c in df.columns]

    if not kap_cols:
        st.info("Kapasite analizi için gerekli kolonlar bulunamadı.")
        return

    g = (
        df.groupby("birim_adi", as_index=False)
        .agg(
            ilce=("ilce_adi", "first"),
            baskanlik=("baskanlik_adi", "first"),
            toplam_dogum=("toplam_dogum", "sum"),
            **{col: (col, "mean") for col in kap_cols},
        )
    ).sort_values(kap_cols[0], ascending=False).head(25)

    # tabloyu biraz daha okunur yap
    show = g[["baskanlik", "ilce", "birim_adi", "toplam_dogum"] + kap_cols].copy()
    show["toplam_dogum"] = show["toplam_dogum"].map(fmt_int)
    for c in kap_cols:
        # önce sayıya çevir (Türkçe virgül vs. dahil), sonra round
        s = show[c].astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
        show[c] = pd.to_numeric(s, errors="coerce").round(2)

    st.dataframe(show, use_container_width=True, hide_index=True)
    st.caption("Not: Değerler aylık ortalama olarak hesaplanır; 0 kapasite bildirilen yerlerde oran üretilmez.")


def hastane_karsilastirma_block(df: pd.DataFrame):
    hastaneler = sorted(df["birim_adi"].dropna().unique().tolist())
    if not hastaneler:
        st.info("Karşılaştırma için birim_adi bulunamadı.")
        return

    col1, col2 = st.columns(2)
    with col1:
        a = st.selectbox("Hastane A", hastaneler, index=0)
    with col2:
        b = st.selectbox("Hastane B", hastaneler, index=min(1, len(hastaneler) - 1))

    def _ozet(one: str) -> pd.DataFrame:
        d = df[df["birim_adi"] == one]
        toplam = d["toplam_dogum"].sum()
        sez = d["sezaryen_dogum"].sum()
        primer = d["primer_sezaryen"].sum()
        canli = d["canli_bebek"].sum()
        olu = d["olu_bebek"].sum()

        sez_oran = (sez / toplam * 100) if toplam else pd.NA
        primer_oran = (primer / sez * 100) if sez else pd.NA
        olu_oran = (olu / (canli + olu) * 100) if (canli + olu) else pd.NA

        return pd.DataFrame(
            {
                "Gösterge": ["Toplam Doğum", "Sezaryen Oranı", "Primer Sezaryen Oranı", "Ölü Doğum Oranı"],
                "Değer": [fmt_int(toplam), fmt_pct(sez_oran, 1), fmt_pct(primer_oran, 1), fmt_pct(olu_oran, 2)],
            }
        )

    da = _ozet(a)
    db = _ozet(b)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"<div class='card'><div class='section-title'>{a}</div></div>", unsafe_allow_html=True)
        st.dataframe(da, use_container_width=True, hide_index=True)
    with c2:
        st.markdown(f"<div class='card'><div class='section-title'>{b}</div></div>", unsafe_allow_html=True)
        st.dataframe(db, use_container_width=True, hide_index=True)


# =========================================================
# MAIN
# =========================================================
def main():
    st.set_page_config(page_title="Doğum Dashboard", layout="wide",initial_sidebar_state="expanded")
    st.markdown("""
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:hidden;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    </style>
    """, unsafe_allow_html=True)
    inject_css()
    topbar()

    df_raw = load_dogum_data()
    if df_raw.empty:
        st.warning("View boş: hastane_analiz.v_dogum_oran veri döndürmüyor.")
        return

    df_raw = add_derived_metrics(df_raw)
    df = filter_block(df_raw)

    if df.empty:
        st.warning("Seçili filtrelerde veri bulunamadı.")
        return

    tab_ozet, tab_detay = st.tabs(["📌 Özet", "🔎 Detay"])

    with tab_ozet:
        kpi_row(df)

        st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)

        st.markdown("<div class='card'><div class='section-title'>Aylara Göre Trend (Toplam • Normal • Sezaryen)</div>", unsafe_allow_html=True)
        trend_block(df)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)

        st.markdown("<div class='card'><div class='section-title'>Başkanlık Bazında Sezaryen ve Primer Sezaryen Oranları</div>", unsafe_allow_html=True)
        baskanlik_sezaryen_bar_vertical(df)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)

        st.markdown("<div class='card'><div class='section-title'>Başkanlık Bazında Ölü Doğum Oranı</div>", unsafe_allow_html=True)
        baskanlik_olu_dogum_bar(df)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)

        st.markdown("<div class='card'><div class='section-title'>Kurum Rolüne Göre Kıyas (A1/A2/B/C...)</div>", unsafe_allow_html=True)
        kurum_rol_karsilastirma_cards(df)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)

        st.markdown("<div class='card'><div class='section-title'>Hastane Sıralama Tablosu</div>", unsafe_allow_html=True)
        ranking_table(df)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_detay:
        st.markdown("<div class='card'><div class='section-title'>Detay: Başkanlık Bazında Oran Isı Haritası</div>", unsafe_allow_html=True)
        baskanlik_heatmap(df)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)

        st.markdown("<div class='card'><div class='section-title'>Kapasite Kullanım Analizi (Doğumhane)</div>", unsafe_allow_html=True)
        kapasite_block(df)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)

        st.markdown("<div class='card'><div class='section-title'>Hastane Karşılaştırma</div>", unsafe_allow_html=True)
        hastane_karsilastirma_block(df)
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
