import streamlit as st
import pandas as pd
from psycopg2 import connect
from psycopg2.extras import DictCursor
import io  # dosyanın başındaki importlara ekle


# ============================================================
# DB AYARLARI
# ============================================================

DB_CONFIG = {
    "dbname": "hastane_analiz",
    "user": "postgres",
    "password": "deneme",
    "host": "localhost",
    "port": 5432,
}


def get_connection():
    return connect(**DB_CONFIG, cursor_factory=DictCursor)


# Küçük bir yardımcı: her yerde tekrar yazmamak için
def read_sql(query: str, params: dict | None = None) -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql(query, conn, params=params)


# ============================================================
#  ACIL DASHBOARD İÇİN CACHE'LENEN YARDIMCI FONKSIYONLAR
# ============================================================

@st.cache_data
def load_filter_data():
    """
    Yıl/Ay, metrik listesi ve ilçe listesini getirir.
    """
    sql_periods = """
        SELECT DISTINCT yil, ay
        FROM hastane_analiz.v_fact_metrik
        WHERE kategori = 'ACIL'
        ORDER BY yil DESC, ay DESC;
    """

    sql_metrics = """
        SELECT metrik_kodu, metrik_adi, veri_tipi
        FROM hastane_analiz.metric_def
        WHERE kategori = 'ACIL'
          AND aktif_mi = true
          AND metrik_kodu IS NOT NULL
        ORDER BY sayfa_adi, metrik_adi;
    """

    sql_districts = """
        SELECT DISTINCT ilce_adi
        FROM hastane_analiz.v_fact_metrik
        WHERE kategori = 'ACIL'
        ORDER BY ilce_adi;
    """

    df_periods = read_sql(sql_periods)
    df_metrics = read_sql(sql_metrics)
    df_districts = read_sql(sql_districts)

    return df_periods, df_metrics, df_districts


@st.cache_data
def load_metric_data(yil: int, ay: int, metrik_kodu: str, ilceler: list[str] | None):
    """
    Seçilen yıl/ay + metrik + (opsiyonel) ilçe filtresi için detay veri.
    Her satır: hastane bazlı.
    """
    base_sql = """
        SELECT
            yil,
            ay,
            ilce_adi,
            birim_adi,
            metrik_kodu,
            toplam_deger
        FROM hastane_analiz.v_fact_metrik
        WHERE kategori = 'ACIL'
          AND yil = %(yil)s
          AND ay  = %(ay)s
          AND metrik_kodu = %(metrik_kodu)s
    """

    params: dict = {"yil": int(yil), "ay": int(ay), "metrik_kodu": metrik_kodu}

    if ilceler:
        base_sql += " AND ilce_adi IN %(ilceler)s"
        params["ilceler"] = tuple(ilceler)

    df = read_sql(base_sql, params=params)
    return df


@st.cache_data
def load_trend_data(metrik_kodu: str, ilceler: list[str] | None):
    """
    Seçilen metrik için TÜM yıllar/aylar bazında trend serisi.
    İsteğe bağlı ilçe filtresi.
    """
    sql = """
        SELECT
            yil,
            ay,
            SUM(toplam_deger) AS toplam_deger
        FROM hastane_analiz.v_fact_metrik
        WHERE kategori = 'ACIL'
          AND metrik_kodu = %(metrik_kodu)s
    """

    params: dict = {"metrik_kodu": metrik_kodu}

    if ilceler:
        sql += " AND ilce_adi IN %(ilceler)s"
        params["ilceler"] = tuple(ilceler)

    sql += """
        GROUP BY yil, ay
        ORDER BY yil, ay;
    """

    df = read_sql(sql, params=params)

    if not df.empty:
        df["ay"] = df["ay"].astype(int)
        df["period"] = (
            df["yil"].astype(int).astype(str) + "-" +
            df["ay"].astype(str).str.zfill(2)
        )

    return df


# ============================================================
#  KALİTE DASHBOARD İÇİN YARDIMCI FONKSIYONLAR
# ============================================================

@st.cache_data
def load_quality_summary():
    sql = """
        SELECT *
        FROM hastane_analiz.v_kalite_dosya_ozet
        ORDER BY fatal_sayisi DESC, warn_sayisi DESC, son_kayit_zamani DESC;
    """
    return read_sql(sql)


@st.cache_data
def load_quality_fatal():
    sql = """
        SELECT *
        FROM hastane_analiz.v_kalite_son_fatal
        ORDER BY olusturma_zamani DESC;
    """
    return read_sql(sql)


@st.cache_data
def load_quality_negative():
    sql = """
        SELECT *
        FROM hastane_analiz.v_kalite_raw_negatif
        ORDER BY yukleme_zamani DESC
        LIMIT 500;
    """
    return read_sql(sql)

@st.cache_data
def load_quality_issues():
    sql = """
        SELECT
            kalite_id,
            seviye,
            kural_kodu,
            mesaj,
            kaynak_dosya,
            kategori,
            sayfa_adi,
            row_index,
            context_json,
            olusturma_zamani
        FROM hastane_analiz.etl_kalite_sonuc
        WHERE seviye IN ('WARN','FATAL')
        ORDER BY olusturma_zamani DESC, kalite_id DESC;
    """
    return read_sql(sql)

@st.cache_data
def load_anomaly_details():
    sql = """
        SELECT
            kalite_id,
            seviye,
            kural_kodu,
            kategori,
            sayfa_adi,
            yil,
            ay,
            kurum_kodu,
            birim_adi,
            ilce_adi,
            metrik_adi,
            metrik_deger,
            q1,
            q3,
            median,
            threshold,
            mesaj,
            kaynak_dosya,
            olusturma_zamani
        FROM hastane_analiz.v_kalite_anomali_aktif
        ORDER BY olusturma_zamani DESC, kalite_id DESC;
    """
    return read_sql(sql)



# ---------------------------------------------------------
#  UI
# ---------------------------------------------------------

def main():
    st.set_page_config(page_title="Acil Hizmetler & Veri Kalitesi", layout="wide")

    st.title("🏥 Hastane Analiz Dashboard")

    tab_acil, tab_kalite = st.tabs(["🚑 Acil Hizmetler", "🧪 Veri Kalitesi"])

    # =======================================================
    #  TAB 1: ACİL DASHBOARD
    # =======================================================
    with tab_acil:
        st.subheader("Acil Hizmetler Dashboard (Pilot)")

        # ---------------- Filtre verilerini çek ----------------
        df_periods, df_metrics, df_districts = load_filter_data()

        if df_periods.empty or df_metrics.empty:
            st.warning("ACİL kategorisi için henüz veri veya tanımlı metrik bulunamadı.")
            return

        # ---------------- Sol panel: Filtreler -----------------
        with st.sidebar:
            st.header("Filtreler")

            # Yıl & Ay
            yil_list = sorted(df_periods["yil"].unique(), reverse=True)
            selected_yil = st.selectbox("Yıl", yil_list, index=0)

            ay_list = sorted(
                df_periods.loc[df_periods["yil"] == selected_yil, "ay"].unique()
            )
            selected_ay = st.selectbox("Ay", ay_list)

            # Metrik seçimi (metrik_adi göster, metrik_kodu ile çalış)
            metric_options = {
                row["metrik_adi"]: (row["metrik_kodu"], row["veri_tipi"])
                for _, row in df_metrics.iterrows()
            }

            metric_name = st.selectbox("Metrik", list(metric_options.keys()))
            selected_metric_code, selected_metric_type = metric_options[metric_name]

            # İlçe çoklu seçim
            ilce_list = df_districts["ilce_adi"].tolist()
            selected_districts = st.multiselect(
                "İlçe (boş bırakılırsa hepsi)",
                ilce_list,
                default=[]
            )

        # ---------------- Veri çek -----------------
        df = load_metric_data(selected_yil, selected_ay,
                              selected_metric_code,
                              selected_districts or None)

        if df.empty:
            st.info("Seçilen filtrelere göre veri bulunamadı.")
        else:
            # ---------------- Üst metrik kartları -----------------
            col1, col2, col3 = st.columns(3)

            if selected_metric_type.upper() == "BOOLEAN":
                # Var/Yok metrikleri için: değer > 0 olan hastane sayısı
                var_hastane_sayisi = (df["toplam_deger"] > 0).sum()
                toplam_hastane_sayisi = df["birim_adi"].nunique()

                with col1:
                    st.metric("Var olan hastane sayısı", f"{var_hastane_sayisi}")
                with col2:
                    st.metric("Toplam hastane sayısı", f"{toplam_hastane_sayisi}")
                with col3:
                    oran = (
                        100 * var_hastane_sayisi / toplam_hastane_sayisi
                        if toplam_hastane_sayisi > 0
                        else 0
                    )
                    st.metric("Oran (%)", f"{oran:.1f}")
            else:
                # Numerik metrikler için
                total_value = float(df["toplam_deger"].sum())
                hospital_count = df["birim_adi"].nunique()
                district_count = df["ilce_adi"].nunique()

                with col1:
                    st.metric("Toplam Değer", f"{total_value:,.0f}".replace(",", "."))
                with col2:
                    st.metric("Hastane Sayısı", f"{hospital_count}")
                with col3:
                    st.metric("İlçe Sayısı", f"{district_count}")

            st.markdown("---")

            # ---------------- İlçe bazlı özet + grafik -----------------
            st.subheader("İlçe Bazlı Toplam Değer")

            df_ilce = (
                df.groupby("ilce_adi", as_index=False)["toplam_deger"]
                .sum()
                .sort_values("toplam_deger", ascending=False)
            )

            col_table, col_chart = st.columns((1, 2))

            with col_table:
                st.dataframe(df_ilce, use_container_width=True)

            with col_chart:
                # Bar chart için index'e ilçe_adi koy
                chart_data = df_ilce.set_index("ilce_adi")["toplam_deger"]
                st.bar_chart(chart_data)

            st.markdown("---")

            # ---------------- Yıllık trend (tüm yıllar) -----------------
            st.subheader("Yıllara Göre Trend (Tüm Dönemler)")

            df_trend = load_trend_data(selected_metric_code,
                                       selected_districts or None)

            if df_trend.empty:
                st.info("Trend grafiği için yeterli veri bulunamadı.")
            else:
                trend_series = df_trend.set_index("period")["toplam_deger"]
                st.line_chart(trend_series)

            st.markdown("---")

            # ---------------- Hastane detay tablosu -----------------
            st.subheader("Hastane Detayları")

            df_detail = df[["yil", "ay", "ilce_adi", "birim_adi",
                            "metrik_kodu", "toplam_deger"]].copy()
            df_detail = df_detail.sort_values(["ilce_adi", "birim_adi"])

            st.dataframe(df_detail, use_container_width=True)

    # =======================================================
    #  TAB 2: VERİ KALİTESİ
    # =======================================================
    with tab_kalite:
        st.subheader("Veri Kalitesi")

        tab1, tab2, tab3, tab4 = st.tabs([
            "📁 Dosya Özeti",
            "❌ Son FATAL Kayıtlar",
            "📉 Negatif Değerler",
            "📊 Anomali Detayları",
        ])


        # ---- Dosya Özeti ----
        with tab1:
            df_ozet = load_quality_summary()
            if df_ozet.empty:
                st.info("Şu ana kadar kalite kaydı bulunamadı.")
            else:
                kategori_sec = st.multiselect(
                    "Kategori filtrele",
                    sorted(df_ozet["kategori"].dropna().unique()),
                    default=list(sorted(df_ozet["kategori"].dropna().unique())),
                )

                df_filtre = df_ozet.copy()
                if kategori_sec:
                    df_filtre = df_filtre[df_filtre["kategori"].isin(kategori_sec)]

                st.dataframe(df_filtre, use_container_width=True)

        # ---- Son FATAL Kayıtlar ----
        with tab2:
            df_fatal = load_quality_fatal()
            if df_fatal.empty:
                st.success("FATAL hata kaydı bulunmuyor 🎉")
            else:
                st.dataframe(df_fatal, use_container_width=True)

        # ---- Negatif Değerler ----
        with tab3:
            df_neg = load_quality_negative()
            if df_neg.empty:
                st.success("Negatif metrik değeri bulunmuyor.")
            else:
                st.dataframe(df_neg, use_container_width=True)

                # ---- TAB 4: Anomali Detayları ----
        with tab4:
            st.subheader("Anomali Detayları (WARN / FATAL)")

            df_anom = load_anomaly_details()

            if df_anom.empty:
                st.success("Şu an kayıtlı anomali (WARN/FATAL) bulunmuyor 🎉")
            else:
                # --- Filtreler ---
                col_f1, col_f2, col_f3, col_f4 = st.columns(4)

                with col_f1:
                    sev_list = sorted(df_anom["seviye"].unique())
                    sev_filter = st.multiselect(
                        "Seviye",
                        sev_list,
                        default=sev_list,
                    )

                with col_f2:
                    kural_list = sorted(df_anom["kural_kodu"].unique())
                    kural_filter = st.multiselect(
                        "Kural Kodu",
                        kural_list,
                        default=kural_list,
                    )

                with col_f3:
                    metrik_list = sorted(
                        [m for m in df_anom["metrik_adi"].dropna().unique()]
                    )
                    metrik_filter = st.multiselect(
                        "Metrik",
                        metrik_list,
                        default=[],
                    )

                with col_f4:
                    birim_list = sorted(
                        [b for b in df_anom["birim_adi"].dropna().unique()]
                    )
                    birim_filter = st.multiselect(
                        "Hastane",
                        birim_list,
                        default=[],
                    )

                df_f = df_anom.copy()
                if sev_filter:
                    df_f = df_f[df_f["seviye"].isin(sev_filter)]
                if kural_filter:
                    df_f = df_f[df_f["kural_kodu"].isin(kural_filter)]
                if metrik_filter:
                    df_f = df_f[df_f["metrik_adi"].isin(metrik_filter)]
                if birim_filter:
                    df_f = df_f[df_f["birim_adi"].isin(birim_filter)]

                # --- Üstte özet kartlar ---
                col_m1, col_m2, col_m3 = st.columns(3)
                toplam = len(df_f)
                warn_say = (df_f["seviye"] == "WARN").sum()
                fatal_say = (df_f["seviye"] == "FATAL").sum()

                with col_m1:
                    st.metric("Toplam Anomali", f"{toplam}")
                with col_m2:
                    st.metric("WARN Sayısı", f"{warn_say}")
                with col_m3:
                    st.metric("FATAL Sayısı", f"{fatal_say}")

                st.markdown("---")

                # --- Tablo için birkaç hesaplanmış kolon ---
                df_show = df_f.copy()
                # median > 0 ise "kaç katı" bilgisi
                def oran(row):
                    med = row.get("median")
                    val = row.get("metrik_deger")
                    if med is None or med == 0 or val is None:
                        return None
                    return round(float(val) / float(med), 2)

                df_show["median_kati"] = df_show.apply(oran, axis=1)

                df_show = df_show[[
                    "olusturma_zamani",
                    "seviye",
                    "kural_kodu",
                    "kategori",
                    "yil",
                    "ay",
                    "ilce_adi",
                    "birim_adi",
                    "metrik_adi",
                    "metrik_deger",
                    "median",
                    "threshold",
                    "median_kati",
                    "mesaj",
                    "kaynak_dosya",
                ]].sort_values("olusturma_zamani", ascending=False)

                st.dataframe(df_show, use_container_width=True)

                # ---- Excel export: WARN/FATAL kayıtları ----
        st.markdown("---")
        st.subheader("Hatalı Kayıtları Excel Olarak İndir")

        df_issues = load_quality_issues()

        if df_issues.empty:
            st.info("Şu an WARN veya FATAL kayıt bulunmuyor, Excel oluşturulacak veri yok.")
        else:
            # context_json'u kolonlara aç
            ctx = pd.json_normalize(df_issues["context_json"].fillna({}), max_level=1)
            ctx.columns = [f"ctx_{c}" for c in ctx.columns]

            df_export = pd.concat(
                [df_issues.drop(columns=["context_json"]), ctx],
                axis=1,
            )

            dt_cols = df_export.select_dtypes(include=["datetimetz"]).columns
            if len(dt_cols) > 0:
                df_export[dt_cols] = df_export[dt_cols].apply(
                    lambda s: s.dt.tz_localize(None)
                 )


            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                df_export.to_excel(writer, index=False, sheet_name="Hatalar")

            st.download_button(
                "Hatalı Satırları Excel Olarak İndir",
                data=buffer.getvalue(),
                file_name="veri_kalitesi_hatalar.xlsx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                ),
            )


if __name__ == "__main__":
    main()
