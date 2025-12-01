# Kimlik kolonları (transformer'lar bunları kullanacak)
ID_COLS = [
    "yil",
    "ay",
    "birim_adi",
    "ilce_adi",
    "baskanlik_adi",
    "kurum_rol_adi",
]

# raw_veri tablosuna yazılan kolonlar
RAW_VERI_COLUMNS = [
    "yil",
    "ay",
    "birim_adi",
    "ilce_adi",
    "baskanlik_adi",
    "kurum_rol_adi",
    "kategori",
    "sayfa_adi",
    "metrik_adi",
    "metrik_deger_raw",
    "metrik_deger_numeric",
    "kaynak_dosya",
]

def get_raw_veri_insert_sql() -> str:
    """
    raw_veri INSERT SQL'ini otomatik oluşturur.
    Buraya kolon ekleyip cikarmak yeterli olacak.
    """
    columns = ", ".join(RAW_VERI_COLUMNS)
    placeholders = ", ".join(["%s"] * len(RAW_VERI_COLUMNS))

    return f"""
        INSERT INTO hastane_analiz.raw_veri (
            {columns}
        )
        VALUES ({placeholders})
    """
