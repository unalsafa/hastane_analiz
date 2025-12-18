import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from io import BytesIO
from hastane_analiz.db.connection import get_connection

# =========================================================
# TEMA / RENK PALETİ
# =========================================================
MINISTRY_TURQUOISE = "#00A3B4"
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
# CSS / UI (DOGUM’dan aynı mantık)
# =========================================================
def inject_css():
    TOPBAR_H = 86  # topbar yüksekliği (px)

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
           (Sidebar okunu bozmaz)
        ========================== */
        div[data-testid="stAppDeployButton"] {{
            display: none !important;
        }}
        button[aria-label="Deploy"], button[title="Deploy"] {{
            display: none !important;
        }}

        /* =========================
           STREAMLIT CHROME
           (DOM kalsın, sadece görünmesin)
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
           SIDEBAR
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
           TOPBAR (FIXED)
        ========================== */
        .topbar {{
            position: fixed;
            top: 10px;
            left: 200px;    /* sidebar açıkken hizalı */
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
           (ok butonu topbar üstünde dursun)
        ========================== */
        button[data-testid="stSidebarCollapseButton"],
        button[data-testid="stSidebarExpandButton"] {{
            position: fixed !important;
            top: 22px !important;
            left: 28px !important;
            z-index: 10000 !important;

            background: #FFFFFF !important;
            border: 1px solid {BORDER} !important;
            border-radius: 12px !important;
            box-shadow: 0 10px 26px rgba(2, 6, 23, 0.12) !important;

            width: 44px !important;
            height: 44px !important;

            display: flex !important;
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

        .kpi-title {{
            font-size: 11px;
            color: {MUTED};
            text-transform: uppercase;
            letter-spacing: .08em;
            margin-bottom: 6px;
        }}
        .kpi-value {{
            font-size: 22px;
            font-weight: 900;
            color: {TEXT};
            line-height: 1.1;
        }}
        .kpi-sub {{
            margin-top: 6px;
            font-size: 12px;
            color: {MUTED};
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
        """
        <div class="topbar">
            <div class="topbar-left">
                <div class="logo-badge">SB</div>
                <div>
                    <div class="topbar-title">Acil Dashboard</div>
                    <div class="topbar-sub">Halk Sağlığı • Acil • Analiz</div>
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
def load_acil_data() -> pd.DataFrame:
    return read_sql("SELECT * FROM hastane_analiz.v_acil_oran;")


@st.cache_data(ttl=300)
def load_validation_issues(kategori: str) -> pd.DataFrame:
    sql = """
    WITH last_run AS (
        SELECT DISTINCT ON (kategori, yil)
            kategori, yil, run_id
        FROM hastane_analiz.validation_run
        WHERE kategori = %s
        ORDER BY kategori, yil, run_id DESC
    )
    SELECT
        vi.yil,
        vi.ay,
        vi.kurum_kodu,
        bd.birim_adi,
        bd.ilce_adi,
        bd.baskanlik_adi,
        bd.kurum_rol_adi,
        vi.metrik_adi AS hatali_veri,
        vi.severity,
        vi.rule_code,
        vi.message AS hata_sebebi,
        vi.oran,
        vi.diger_ay_ort
    FROM hastane_analiz.validation_issue vi
    JOIN last_run lr
      ON lr.run_id = vi.run_id
    LEFT JOIN hastane_analiz.birim_def bd
      ON bd.kurum_kodu = vi.kurum_kodu
    WHERE vi.kategori = %s
      AND vi.rule_code IN ('MONTH_OUTLIER','MONTH_MISSING')
    ORDER BY vi.yil DESC, vi.ay DESC,
             bd.baskanlik_adi NULLS LAST, bd.ilce_adi NULLS LAST, bd.birim_adi NULLS LAST;
    """
    return read_sql(sql, params=(kategori, kategori))


# =========================================================
# HELPERS
# =========================================================
def fmt_int(x) -> str:
    try:
        return f"{int(round(float(x))):,}".replace(",", ".")
    except Exception:
        return "0"


def fmt_pct(x, digits=2) -> str:
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


def df_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Validation") -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name[:31])
    return output.getvalue()


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
    toplam = df["toplam_triaj_hasta"].sum() if "toplam_triaj_hasta" in df.columns else 0
    amb = df["ambulans_ile_gelen"].sum() if "ambulans_ile_gelen" in df.columns else 0
    tekrar = df["tekrar_24s"].sum() if "tekrar_24s" in df.columns else 0
    yatis = df["yatis_sayisi"].sum() if "yatis_sayisi" in df.columns else 0

    amb_oran = (amb / toplam * 100) if toplam else pd.NA
    tekrar_oran = (tekrar / toplam * 100) if toplam else pd.NA
    yatis_oran = (yatis / toplam * 100) if toplam else pd.NA

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        card_kpi("Toplam Triaj Hasta", fmt_int(toplam), "Seçili dönem")
    with c2:
        card_kpi("Ambulans ile Gelen", fmt_int(amb), "Seçili dönem")
    with c3:
        card_kpi("Ambulans Oranı", fmt_pct(amb_oran, 2), "Ambulans / Toplam")
    with c4:
        card_kpi("24s Tekrar", fmt_int(tekrar), "Seçili dönem")
    with c5:
        card_kpi("Tekrar Oranı", fmt_pct(tekrar_oran, 2), "Tekrar / Toplam")
    with c6:
        card_kpi("Yatış Oranı", fmt_pct(yatis_oran, 2), "Yatış / Toplam")


def trend_block(df: pd.DataFrame):
    g = (
        df.groupby(["yil", "ay"], as_index=False)
        .agg(
            toplam=("toplam_triaj_hasta", "sum"),
            kirmizi=("kirmizi_hasta", "sum") if "kirmizi_hasta" in df.columns else ("toplam_triaj_hasta", "sum"),
            sari=("sari_hasta", "sum") if "sari_hasta" in df.columns else ("toplam_triaj_hasta", "sum"),
            yesil=("yesil_muayene", "sum") if "yesil_muayene" in df.columns else ("toplam_triaj_hasta", "sum"),
        )
        .sort_values(["yil", "ay"])
    )
    g["tarih"] = pd.to_datetime(dict(year=g["yil"], month=g["ay"], day=1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g["tarih"], y=g["toplam"], mode="lines+markers", name="Toplam Triaj", line=dict(color=PRIMARY_GREEN, width=3)))
    fig.add_trace(go.Scatter(x=g["tarih"], y=g["kirmizi"], mode="lines+markers", name="Kırmızı", line=dict(color=DANGER_COLOR, width=2)))
    fig.add_trace(go.Scatter(x=g["tarih"], y=g["sari"], mode="lines+markers", name="Sarı", line=dict(color=WARN_COLOR, width=2)))
    fig.add_trace(go.Scatter(x=g["tarih"], y=g["yesil"], mode="lines+markers", name="Yeşil (muayene)", line=dict(color=ACCENT_BLUE, width=2)))

    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=380,
        margin=dict(l=10, r=10, t=10, b=10),
        legend_title_text="",
        yaxis_title=None,
        xaxis_title=None,
    )
    st.plotly_chart(fig, use_container_width=True)


def ranking_table(df: pd.DataFrame):
    grp = (
        df.groupby(["baskanlik_adi", "ilce_adi", "birim_adi"], as_index=False)
        .agg(
            toplam=("toplam_triaj_hasta", "sum"),
            kirmizi_oran=("kirmizi_oran", "mean"),
            sari_oran=("sari_oran", "mean"),
            yesil_oran=("yesil_oran", "mean"),
            ambulans_oran=("ambulans_oran", "mean") if "ambulans_oran" in df.columns else ("kirmizi_oran", "mean"),
        )
    )

    sort_by = st.selectbox(
        "Sıralama kriteri",
        options=["toplam", "ambulans_oran", "kirmizi_oran", "sari_oran", "yesil_oran"],
        index=0,
    )
    grp = grp.sort_values(by=sort_by, ascending=False)

    out = grp.copy()
    out["toplam"] = out["toplam"].map(fmt_int)
    for c in ["ambulans_oran", "kirmizi_oran", "sari_oran", "yesil_oran"]:
        if c in out.columns:
            out[c] = out[c].map(lambda x: fmt_pct(x, 2))

    show_cols = ["baskanlik_adi", "ilce_adi", "birim_adi", "toplam", "ambulans_oran", "kirmizi_oran", "sari_oran", "yesil_oran"]
    show_cols = [c for c in show_cols if c in out.columns]
    st.dataframe(out[show_cols], use_container_width=True, hide_index=True)


def validation_table_block(df_filtered: pd.DataFrame, kategori: str = "ACIL"):
    issues = load_validation_issues(kategori)
    if issues.empty:
        st.success("Validation issue bulunamadı (MONTH_OUTLIER / MONTH_MISSING).")
        return

    # Dashboard filtreleriyle uyumlu hale getir (yıl/ay/baskanlik/ilce/rol)
    yil = int(df_filtered["yil"].iloc[0]) if "yil" in df_filtered.columns and not df_filtered.empty else None
    ay_list = sorted(df_filtered["ay"].dropna().unique().tolist()) if "ay" in df_filtered.columns else []

    if yil is not None and "yil" in issues.columns:
        issues = issues[issues["yil"] == yil]
    if ay_list and "ay" in issues.columns:
        issues = issues[issues["ay"].isin(ay_list)]

    for col in ["baskanlik_adi", "ilce_adi", "kurum_rol_adi", "birim_adi"]:
        if col in df_filtered.columns and col in issues.columns:
            vals = sorted(df_filtered[col].dropna().unique().tolist())
            if vals:
                issues = issues[issues[col].isin(vals)]

    if issues.empty:
        st.info("Seçili filtrelerde validation issue yok.")
        return

    st.markdown("<div class='card'><div class='section-title'>Kalite Uyarıları (MONTH_OUTLIER / MONTH_MISSING)</div>", unsafe_allow_html=True)

    # küçük temizlik/format
    out = issues.copy()
    if "oran" in out.columns:
        out["oran"] = pd.to_numeric(out["oran"], errors="coerce").round(4)
    if "diger_ay_ort" in out.columns:
        out["diger_ay_ort"] = pd.to_numeric(out["diger_ay_ort"], errors="coerce").round(1)

    show_cols = [
        "yil", "ay", "kurum_kodu", "birim_adi", "hatali_veri",
        "severity", "rule_code", "hata_sebebi", "oran", "diger_ay_ort"
    ]
    show_cols = [c for c in show_cols if c in out.columns]

    export_df = out[show_cols].copy()

    # ✅ Tek buton: filtrelenmiş sonucu Excel'e aktar
    if not export_df.empty:
        st.download_button(
            label="📥 Validation Excel indir",
            data=df_to_excel_bytes(export_df, sheet_name="Validation"),
            file_name=f"{kategori}_VALIDATION_{yil}.xlsx" if yil else f"{kategori}_VALIDATION.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"dl_validation_single_{kategori}_{yil}",
        )

    st.dataframe(export_df, use_container_width=True, hide_index=True)
    st.caption("Not: Bu tablo, ilgili kategori için en son validation_run çıktısından gelir.")
    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# MAIN
# =========================================================
def main():
    st.set_page_config(page_title="Acil Dashboard", layout="wide", initial_sidebar_state="expanded")

    inject_css()
    topbar()

    df_raw = load_acil_data()
    if df_raw.empty:
        st.warning("View boş: hastane_analiz.v_acil_oran veri döndürmüyor.")
        return

    df = filter_block(df_raw)
    if df.empty:
        st.warning("Seçili filtrelerde veri bulunamadı.")
        return

    tab_ozet, tab_detay = st.tabs(["📌 Özet", "🔎 Detay"])

    with tab_ozet:
        kpi_row(df)
        st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)

        st.markdown("<div class='card'><div class='section-title'>Aylara Göre Trend (Toplam • Kırmızı • Sarı • Yeşil)</div>", unsafe_allow_html=True)
        trend_block(df)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)

        st.markdown("<div class='card'><div class='section-title'>Hastane Sıralama Tablosu</div>", unsafe_allow_html=True)
        ranking_table(df)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)

        # ✅ Validation tablo + export burada
        validation_table_block(df, kategori="ACIL")

    with tab_detay:
        st.info("Detay sekmesini istersen DOGUM’daki gibi heatmap / kapasite / karşılaştırma bloklarıyla büyütürüz.")


if __name__ == "__main__":
    main()
