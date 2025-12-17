# hastane_analiz/dashboard/dashboard_etl_validation.py

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st

from hastane_analiz.db.connection import get_connection
from hastane_analiz.etl.runner import run_etl_for_folder
from hastane_analiz.validation.yearly_outliers import run_yearly_outlier_scan
from hastane_analiz.validation.promote_to_fact import promote_month_to_fact
from hastane_analiz.validation.promote_to_fact import get_issue_summary



# Hangi rule_code'lar hangi gruba giriyor?
BASE_RULE_CODES = [
    "NEGATIVE_VALUE",
    "METRIC_NOT_NUMERIC",
    "RATIO_MISSING",
    "RATIO_OUT_OF_RANGE",
    "REQUIRED_MISSING",
    # sende başka temel kural kodları varsa buraya ekle
]

OUTLIER_RULE_CODES = [
    "MONTH_OUTLIER",
    "MONTH_MISSING",
]


# ==============================
# Yardımcı fonksiyonlar
# ==============================

def load_etl_kalite_for_period(kategori: str, yil: int, ay: int) -> pd.DataFrame:
    """
    etl_kalite_sonuc tablosundan, seçilen kategori + yil/ay için
    (hem yıl/ay kolonuna hem de dosya adına göre) ETL validation kayıtlarını çeker.
    """

    like_pattern = f"%{yil}-{ay:02d}%"

    sql = """
        SELECT
            e.yil,
            e.ay,
            e.kurum_kodu,
            b.birim_adi AS kurum_adi,
            e.metrik_adi,
            e.deger,
            e.prev_mean,
            e.ratio,
            e.seviye,
            e.kural_kodu,
            e.mesaj,
            e.kategori,
            e.sayfa_adi,
            e.kaynak_dosya,
            e.row_index
        FROM hastane_analiz.etl_kalite_sonuc e
        LEFT JOIN hastane_analiz.birim_def b
          ON b.kurum_kodu = e.kurum_kodu
        WHERE e.kategori = %s
          AND (
                (e.yil = %s AND e.ay = %s)
             OR e.kaynak_dosya LIKE %s
          )
        ORDER BY
            e.seviye DESC,
            e.yil NULLS LAST,
            e.ay NULLS LAST,
            b.birim_adi NULLS LAST,
            e.kural_kodu,
            e.kaynak_dosya,
            e.row_index;
    """

    with get_connection() as conn:
        df = pd.read_sql(sql, conn, params=[kategori.upper(), yil, ay, like_pattern])

    return df


def _get_latest_validation_run_id(kategori: str, yil: int) -> int | None:
    sql = """
        SELECT run_id
        FROM hastane_analiz.validation_run
        WHERE kategori = %s
          AND yil = %s
        ORDER BY run_id DESC
        LIMIT 1;
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, (kategori, yil))
        row = cur.fetchone()
        return row[0] if row else None


def load_issue_details(kategori: str, yil: int, ay: int | None = None) -> pd.DataFrame:
    """
    En son validation_run (kategori+yıl) için issue detaylarını getirir.
    """
    run_id = _get_latest_validation_run_id(kategori, yil)
    if run_id is None:
        return pd.DataFrame()

    sql = """
        SELECT
            vi.yil,
            vi.ay,
            vi.kurum_kodu,
            bd.birim_adi AS kurum_adi,
            vi.metrik_adi AS hatali_veri,
            vi.severity,
            vi.rule_code,
            vi.message AS hata_sebebi,
            vi.oran,
            vi.diger_ay_ort
        FROM hastane_analiz.validation_issue vi
        JOIN hastane_analiz.validation_run vr ON vr.run_id = vi.run_id
        LEFT JOIN hastane_analiz.birim_def bd ON bd.kurum_kodu = vi.kurum_kodu
        WHERE vi.run_id = %s
    """
    params: list = [run_id]
    if ay is not None:
        sql += " AND vi.ay = %s"
        params.append(ay)

    with get_connection() as conn:
        df = pd.read_sql(sql, conn, params=params)

    return df


def load_coverage_table(kategori: str, yil: int) -> pd.DataFrame:
    """
    Kategori + yıl için:
      - Beklenen metrik sayısı (acil_kural_def)
      - Gerçekte gelen metrik sayısı
      - Eksik başlık sayısı ve listesi
    """
    if kategori != "ACIL":
        # Şimdilik sadece ACIL için kural tablomuz olduğunu varsayalım
        return pd.DataFrame()

    with get_connection() as conn:
        # Beklenen metrikler (acil_kural_def)
        df_expected = pd.read_sql(
            """
            SELECT DISTINCT metrik_yolu AS metrik_adi
            FROM hastane_analiz.acil_kural_def
            WHERE kategori = %s
              AND aktif_mi = TRUE;
            """,
            conn,
            params=[kategori],
        )
        if df_expected.empty:
            return pd.DataFrame()

        expected_set = set(df_expected["metrik_adi"].astype(str))
        expected_count = len(expected_set)

        # Gerçek gelen metrikler (raw_veri/v_metrik_aylik)
        df_actual = pd.read_sql(
            """
            SELECT
                yil,
                ay,
                kurum_kodu,
                metrik_adi
            FROM hastane_analiz.v_metrik_aylik
            WHERE kategori = %s
              AND yil = %s;
            """,
            conn,
            params=[kategori, yil],
        )

        # Kurum adları
        df_birim = pd.read_sql(
            """
            SELECT DISTINCT
                kurum_kodu,
                birim_adi
            FROM hastane_analiz.birim_def;
            """,
            conn,
        )

    if df_actual.empty:
        return pd.DataFrame()

    rows = []
    for (yil_g, ay_g, kurum_kodu), group in df_actual.groupby(["yil", "ay", "kurum_kodu"]):
        actual_set = set(group["metrik_adi"].astype(str))
        missing = sorted(expected_set - actual_set)
        actual_count = len(actual_set)
        eksik_sayi = len(missing)

        if missing:
            if len(missing) > 15:
                eksik_basliklar = ", ".join(missing[:15]) + " ..."
            else:
                eksik_basliklar = ", ".join(missing)
        else:
            eksik_basliklar = ""

        rows.append(
            {
                "yil": int(yil_g),
                "ay": int(ay_g),
                "kurum_kodu": int(kurum_kodu),
                "beklenen_kategori_veri_sayisi": expected_count,
                "eklenmek_istenen_veri_sayisi": actual_count,
                "eksik_baslik_sayisi": eksik_sayi,
                "eksik_basliklar": eksik_basliklar,
            }
        )

    df_cov = pd.DataFrame(rows)

    # Kurum adlarını ekle
    df_cov = df_cov.merge(df_birim, how="left", on="kurum_kodu")
    df_cov = df_cov[
        [
            "yil",
            "ay",
            "kurum_kodu",
            "birim_adi",
            "beklenen_kategori_veri_sayisi",
            "eklenmek_istenen_veri_sayisi",
            "eksik_baslik_sayisi",
            "eksik_basliklar",
        ]
    ].rename(columns={"birim_adi": "kurum_adi"})

    return df_cov.sort_values(["yil", "ay", "kurum_adi"])


def df_to_excel_download(df: pd.DataFrame, sheet_name: str, file_name: str):
    """
    Verilen DataFrame'i excel'e çevirip Streamlit download_button için
    (data, file_name, mime) döner.
    """
    if df.empty:
        return None, None, None

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    data = buffer.getvalue()
    mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    return data, file_name, mime


# ==============================
# Streamlit UI
# ==============================

def main():
    st.set_page_config(page_title="ETL & Validasyon", layout="wide")

    st.title("ETL & Validasyon")

    # Üst filtreler
    col1, col2, col3 = st.columns(3)
    with col1:
        kategori = st.selectbox("Kategori", ["ACIL", "DOGUM"], index=0)
    with col2:
        yil = st.number_input("Yıl", min_value=2020, max_value=2100, value=2025, step=1)
    with col3:
        ay = st.number_input("Ay", min_value=1, max_value=12, value=1, step=1)

    st.markdown("---")

    # 1) Veri yükleme bloğu
    st.header("1. Veri Yükleme")

    st.write(
        "Bu bölümde seçtiğiniz klasördeki Excel dosyaları `raw_veri` tablosuna yüklenir. "
        "Genelde klasör, tek bir aya ait dosyaları içerir (örn. `C:\\veri\\2025_02`)."
    )

    default_folder = ""
    folder_input = st.text_input("Kaynak klasör yolu", value=default_folder)

    if st.button("Klasörden veriyi yükle (raw_veri)"):
        if not folder_input:
            st.error("Lütfen bir klasör yolu girin.")
        else:
            folder_path = Path(folder_input)
            if not folder_path.exists():
                st.error(f"Klasör bulunamadı: {folder_path}")
            else:
                st.info(f"ETL çalışıyor... Klasör: {folder_path}")
                try:
                    run_etl_for_folder(str(folder_path))
                    st.success("ETL tamamlandı, veriler raw_veri tablosuna yazıldı.")
                except Exception as e:
                    st.error("ETL sırasında hata oluştu.")
                    st.exception(e)

    st.markdown("---")

    # 2) Yıllık sapma taraması
    st.header("2. Yıllık Sapma Taraması (Outlier Ayları Bul)")

    st.write(
        "Seçtiğiniz kategori + yıl için tüm 12 ay taranır; "
        "anormal derecede yüksek/düşük aylar ve eksik aylar validation_issue tablosuna yazılır."
    )

    col_run1, col_run2 = st.columns([1, 2])
    with col_run1:
        if st.button("Bu yıl için sapma taramasını çalıştır"):
            try:
                run_id = run_yearly_outlier_scan(kategori=kategori, yil=int(yil))
                st.success(f"Sapma taraması tamamlandı. run_id = {run_id}")
            except Exception as e:
                st.error("Sapma taraması sırasında hata oluştu.")
                st.exception(e)

    latest_run_id = _get_latest_validation_run_id(kategori, int(yil))
    with col_run2:
        if latest_run_id is not None:
            st.info(f"Bu kategori+yıl için son validation_run_id: {latest_run_id}")
        else:
            st.info("Bu kategori+yıl için henüz validation_run kaydı bulunamadı.")

    st.markdown("---")

    # 3) Detay hata listesi
    st.header("3. Hata Detayları (Kurum Kodu / Kurum Adı / Hatalı Veri)")

    df_issues = load_issue_details(kategori, int(yil), ay=None)

    if df_issues.empty:
        st.info("Bu kategori+yıl için kayıtlı hata bulunamadı (veya henüz validation_run yok).")
    else:
        col_a, col_b, col_c, col_d = st.columns(4)
        total_issues = len(df_issues)
        warn_count = (df_issues["severity"] == "WARN").sum()
        error_count = (df_issues["severity"] == "ERROR").sum()
        info_count = (df_issues["severity"] == "INFO").sum()

        with col_a:
            st.metric("Toplam Issue", total_issues)
        with col_b:
            st.metric("WARN", int(warn_count))
        with col_c:
            st.metric("ERROR", int(error_count))
        with col_d:
            st.metric("INFO", int(info_count))

        st.subheader("Filtreler")

        mode = st.radio(
            "Kural grubu",
            ["Hepsi", "Temel validation", "Sapma / Eksik ay"],
            horizontal=True,
        )

        df_mode = df_issues.copy()
        if mode == "Temel validation":
            df_mode = df_mode[df_mode["rule_code"].isin(BASE_RULE_CODES)]
        elif mode == "Sapma / Eksik ay":
            df_mode = df_mode[df_mode["rule_code"].isin(OUTLIER_RULE_CODES)]

        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            sev_opts = sorted(df_mode["severity"].dropna().unique())
            sev_sel = st.multiselect("Severity", options=sev_opts, default=sev_opts)
        with col_f2:
            rule_opts = sorted(df_mode["rule_code"].dropna().unique())
            rule_sel = st.multiselect("Kural Kodu", options=rule_opts, default=rule_opts)
        with col_f3:
            ay_opts = sorted(df_mode["ay"].dropna().unique())
            ay_sel = st.multiselect("Ay", options=ay_opts, default=ay_opts)

        mask = (
            df_mode["severity"].isin(sev_sel)
            & df_mode["rule_code"].isin(rule_sel)
            & df_mode["ay"].isin(ay_sel)
        )
        df_filtered = df_mode[mask].copy()

        st.subheader("Hata Tablosu")
        st.dataframe(df_filtered, use_container_width=True)

        excel_all, name_all, mime_all = df_to_excel_download(
            df_filtered,
            sheet_name="Hatalar",
            file_name=f"hatalar_{kategori}_{yil}_filtreli.xlsx",
        )

        df_base = df_filtered[df_filtered["rule_code"].isin(BASE_RULE_CODES)]
        excel_base, name_base, mime_base = df_to_excel_download(
            df_base,
            sheet_name="TemelValidation",
            file_name=f"hatalar_{kategori}_{yil}_temel.xlsx",
        )

        df_out = df_filtered[df_filtered["rule_code"].isin(OUTLIER_RULE_CODES)]
        excel_out, name_out, mime_out = df_to_excel_download(
            df_out,
            sheet_name="SapmaEksikAy",
            file_name=f"hatalar_{kategori}_{yil}_sapma.xlsx",
        )

        col_x1, col_x2, col_x3 = st.columns(3)
        with col_x1:
            if excel_all:
                st.download_button(
                    "Görünen (filtreli) hataları indir",
                    data=excel_all,
                    file_name=name_all,
                    mime=mime_all,
                )
        with col_x2:
            if excel_base:
                st.download_button(
                    "Sadece TEMEL validation hataları",
                    data=excel_base,
                    file_name=name_base,
                    mime=mime_base,
                )
        with col_x3:
            if excel_out:
                st.download_button(
                    "Sadece SAPMA / EKSİK AY hataları",
                    data=excel_out,
                    file_name=name_out,
                    mime=mime_out,
                )

    st.markdown("---")

    # 4) Kapsama / Eksik başlıklar
    st.header("4. Kapsama ve Eksik Başlıklar")

    df_cov = load_coverage_table(kategori, int(yil))

    if df_cov.empty:
        st.info("Bu kategori+yıl için kapsama verisi bulunamadı (veya ACIL dışı kategori seçili).")
    else:
        st.dataframe(df_cov, use_container_width=True)

        excel_data2, excel_name2, excel_mime2 = df_to_excel_download(
            df_cov,
            sheet_name="Kapsama",
            file_name=f"kapsama_{kategori}_{yil}.xlsx",
        )
        if excel_data2:
            st.download_button(
                "Kapsama tablosunu Excel olarak indir",
                data=excel_data2,
                file_name=excel_name2,
                mime=excel_mime2,
            )

    st.markdown("---")

    


    # 5) Ana veriye (fact) aktarma
    st.header("5. Ana Veriye Aktarım (fact_metrik_aylik)")

    # 5 için özet
    summary = get_issue_summary(kategori, int(yil), int(ay))
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.metric("FATAL (ETL)", summary.get("FATAL", 0))
    with col_s2:
        st.metric("WARN (ETL)", summary.get("WARN", 0))
    with col_s3:
        st.metric("INFO (ETL)", summary.get("INFO", 0))
        
    st.write(
        "Seçili yıl ve ay için, validation sonuçlarına göre bu ayı ana veri tablosu "
        "`fact_metrik_aylik` içine alabilirsiniz. `promote_to_fact` fonksiyonu içinde "
        "kritik hatalar varsa ayı almama kontrolü zaten mevcut."
    )

    col_fact1, col_fact2 = st.columns(2)
    with col_fact1:
        if st.button("Bu ayı ana veriye AL (fact'e yaz)"):
            try:
                promote_month_to_fact(kategori=kategori, yil=int(yil), ay=int(ay))
                st.success(
                    f"{yil}-{int(ay):02d} için fact_metrik_aylik güncellendi."
                )
            except RuntimeError as e:
                # Bizim bilinçli fırlattığımız hata (FATAL varken geçirme)
                st.error(str(e))
            except Exception as e:
                # Beklenmeyen başka bir hata olursa
                st.error("Ana veriye aktarım sırasında beklenmeyen bir hata oluştu.")
                st.exception(e)

    with col_fact2:
        st.info(
            "Eğer bu ayı ana veriye ALMAMAK istiyorsanız, hiçbir şey yapmayın; "
            "veri sadece raw_veri ve validation_issue'da kalır. Hataları "
            "yukarıdan Excel olarak indirebilirsiniz."
        )


    # 6) ETL temel validation sonuçları (etl_kalite_sonuc)
    st.header("6. ETL Temel Validation Sonuçları (etl_kalite_sonuc)")

    st.write(
        "Bu bölümde, ETL sırasında çalışan kuralların (RANGE, TS_MEAN, "
        "SUM_EQ, BOOLEAN_CHANGE, ZERO_WHILE_OTHERS_POSITIVE, METRIC_OUTLIER_HIGH vb.) "
        "ürettiği kayıtları görürsünüz. Kayıtlar `etl_kalite_sonuc` tablosundan çekilir."
    )

    df_etl = load_etl_kalite_for_period(kategori, int(yil), int(ay))

    if df_etl.empty:
        st.info(
            f"{kategori} / {yil}-{int(ay):02d} için etl_kalite_sonuc kaydı bulunamadı. "
            "Bu ya hiç hata olmadığı, ya da ETL'nin bu ay için çalışmadığı anlamına gelir."
        )
    else:
        sev_counts = df_etl["seviye"].value_counts()
        col_k1, col_k2, col_k3 = st.columns(3)
        with col_k1:
            st.metric("Toplam kayıt", len(df_etl))
        with col_k2:
            st.metric("FATAL", int(sev_counts.get("FATAL", 0)))
        with col_k3:
            st.metric("WARN", int(sev_counts.get("WARN", 0)))

        st.subheader("ETL Validation Filtreleri")
        col_ek1, col_ek2, col_ek3 = st.columns(3)

        with col_ek1:
            sev_opts = sorted(df_etl["seviye"].dropna().unique()) if "seviye" in df_etl.columns else []
            sev_sel = st.multiselect("Seviye", options=sev_opts, default=sev_opts)

        with col_ek2:
            rule_opts = sorted(df_etl["kural_kodu"].dropna().unique()) if "kural_kodu" in df_etl.columns else []
            rule_sel = st.multiselect("Kural Kodu", options=rule_opts, default=rule_sel if (rule_sel := rule_opts) else [])

        with col_ek3:
            if "kurum_adi" in df_etl.columns:
                kurum_opts = sorted(df_etl["kurum_adi"].dropna().unique())
                kurum_sel = st.multiselect("Kurum", options=kurum_opts, default=kurum_opts)
            else:
                st.write("Kurum filtresi devre dışı (kurum_adi kolonu yok).")
                kurum_sel = []

        mask = pd.Series(True, index=df_etl.index)

        if "seviye" in df_etl.columns and sev_sel:
            mask &= df_etl["seviye"].isin(sev_sel)

        if "kural_kodu" in df_etl.columns and rule_sel:
            mask &= df_etl["kural_kodu"].isin(rule_sel)

        if "kurum_adi" in df_etl.columns and kurum_sel:
            mask &= df_etl["kurum_adi"].isin(kurum_sel)

        df_etl_f = df_etl[mask].copy()

        st.subheader("ETL Temel Validation Tablosu")
        st.dataframe(df_etl_f, use_container_width=True)

        excel_etl, name_etl, mime_etl = df_to_excel_download(
            df_etl_f,
            sheet_name="ETL_Validation",
            file_name=f"etl_validation_{kategori}_{yil}_{int(ay):02d}.xlsx",
        )
        if excel_etl:
            st.download_button(
                "ETL temel validation sonuçlarını Excel olarak indir",
                data=excel_etl,
                file_name=name_etl,
                mime=mime_etl,
            )


if __name__ == "__main__":
    main()
