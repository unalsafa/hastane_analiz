# hastane_analiz/etl/schema_registry.py

RAW_VERI_COLUMNS = [
    "yil",
    "ay",
    "kurum_kodu",
    "kategori",
    "sayfa_adi",
    "metrik_adi",
    "metrik_deger_raw",
    "metrik_deger_numeric",
    "kaynak_dosya",
]


def get_raw_veri_insert_sql() -> str:
    """
    raw_veri tablosu icin INSERT SQL uretilir.
    Kolon sirasi RAW_VERI_COLUMNS ile bire bir uyumlu.
    """
    cols = ",\n            ".join(RAW_VERI_COLUMNS)

    placeholders = ", ".join(["%s"] * len(RAW_VERI_COLUMNS))

    sql = f"""
        INSERT INTO hastane_analiz.raw_veri (
            {cols}
        )
        VALUES (
            {placeholders}
        )
    """
    return sql.strip()
