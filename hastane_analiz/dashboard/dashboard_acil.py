import streamlit as st
import pandas as pd
from psycopg2 import connect
from psycopg2.extras import DictCursor

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


# ============================================================
# CACHE'LENEN YARDIMCI FONKSIYONLAR
# ============================================================

@st.cache_data(show_spinner=False)
def load_filter_base():
    """
    ACIL kategorisi için:
      - hangi yıl/ay var
      - hangi metrikler var (metric_def)
      - hangi ilçeler var
    hepsini tek seferde çeker.
    """
    with get_connection() as conn:
        # Yıl / Ay
        sql_periods = """
            SELECT DISTINCT yil, ay
            FROM hastane_analiz.v_fact_metrik
            WHERE kategori = 'ACIL'
            ORDER BY yil, ay;
        """
        df_periods = pd.read_sql(sql_periods, conn)

        # Metrikler
        sql_metrics = """
            SELECT metrik_kodu, metrik_adi
            FROM hastane_analiz.metric_def
            WHERE kategori = 'ACIL'
            ORDER BY metrik_adi;
        """
        df_metrics = pd.read_sql(sql_metrics, conn)

        # İlçeler
        sql_districts = """
            SELECT DISTINCT ilce_adi
            FROM hastane_analiz.v_fact_metrik
            WHERE kategori = 'ACIL'
            ORDER BY ilce_adi;
        """
        df_districts = pd.read_sql(sql_districts, conn)

    return df_periods, df_metrics, df_districts


@st.cache_data(show_spinner=False)
def load_metric_data(yil: int, ay: int, metrik_kodu: str, ilceler: list[str] | None):
    """
    Seçilen yıl/ay/metrik (ve varsa ilçe filtresi) için v_fact_metrik’ten detaylı veri çeker.
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

    params: dict = {
        "yil": int(yil),
        "ay": int(ay),
        "metrik_kodu": metrik_kodu,
    }

    if ilceler:
        # ilce_adi = ANY(ARRAY[...]) kullan
        base_sql += " AND ilce_adi = ANY(%(ilceler)s)"
        params["ilceler"] = ilceler

    with get_connection() as conn:
        df = pd.read_sql(base_sql, conn, params=params)

    # toplami string'e dönmüş olma ihtimaline karşı
    if "toplam_deger" in df.columns:
        df["toplam_deger"] = pd.to_numeric(df["toplam_deger"], errors="coerce")

    return df


# ============================================================
# UI
# ============================================================

def main():
    st.set_page_config(
        page_title="Acil Hizmetler Dashboard (Pilot)",
        layout="wide",
    )

    st.title("🧑‍⚕️ Acil Hizmetler Dashboard (Pilot)")

    # --------------------------
    # Filtre verilerini yükle
    # --------------------------
    df_periods, df_metrics, df_districts = load_filter_base()

    if df_periods.empty or df_metrics.empty:
        st.warning("ACIL kategorisi için veri veya metrik tanımı bulunamadı.")
        return

    with st.sidebar:
        st.header("Filtreler")

        # YIL
        years = sorted(df_periods["yil"].unique())
        default_yil_index = len(years) - 1  # en son yıl
        yil = st.selectbox("Yıl", options=years, index=default_yil_index)

        # AY (seçili yıla göre filtrele)
        months_for_year = sorted(
            df_periods.loc[df_periods["yil"] == yil, "ay"].unique()
        )
        default_ay_index = len(months_for_year) - 1
        ay = st.selectbox("Ay", options=months_for_year, index=default_ay_index)

        # METRİK
                # METRİK
        metric_options = {
            f"{row.metrik_adi} ({row.metrik_kodu})": row.metrik_kodu
            for _, row in df_metrics.iterrows()
        }

        # label -> kod sözlüğü
        metric_labels = list(metric_options.keys())

        selected_metric_label = st.selectbox(
            "Metrik",
            options=metric_labels,
            index=0,  # varsayılan ilk metrik
        )

        # Kullanıcının seçtiği label'dan gerçek kodu al
        selected_metric_code = metric_options[selected_metric_label]


        # İLÇE
        all_districts = df_districts["ilce_adi"].tolist()
        selected_districts = st.multiselect(
            "İlçe (boş bırakılırsa hepsi)",
            options=all_districts,
            default=[],
        )

    # --------------------------
    # Veriyi yükle
    # --------------------------
    df = load_metric_data(yil, ay, selected_metric_code, selected_districts or None)

    if df.empty:
        st.info("Seçilen filtrelere göre veri bulunamadı.")
        return

    # Her ihtimale karşı tekrar numerik'e zorluyoruz
    df["toplam_deger"] = pd.to_numeric(df["toplam_deger"], errors="coerce")
    df = df.dropna(subset=["toplam_deger"])

    if df.empty:
        st.info("Toplam değeri hesaplanabilir satır bulunamadı.")
        return

    # --------------------------
    # Özet kutuları
    # --------------------------
    toplam_deger = float(df["toplam_deger"].sum())
    hastane_sayisi = df["birim_adi"].nunique()
    ilce_sayisi = df["ilce_adi"].nunique()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        label="Toplam Değer",
        value=f"{toplam_deger:,.0f}".replace(",", "."),
    )
    col2.metric("Hastane Sayısı", hastane_sayisi)
    col3.metric("İlçe Sayısı", ilce_sayisi)

    st.markdown("---")

    # --------------------------
    # İlçe bazlı özet
    # --------------------------
    df_ilce = (
        df.groupby("ilce_adi", as_index=False)["toplam_deger"]
        .sum()
        .sort_values("toplam_deger", ascending=False)
    )
    st.subheader("İlçe Bazlı Toplam Değer")
    st.dataframe(df_ilce, use_container_width=True)

    # --------------------------
    # Detay tablo
    # --------------------------
    st.subheader("Hastane Detayları")
    st.dataframe(
        df.sort_values(["ilce_adi", "birim_adi"]),
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
