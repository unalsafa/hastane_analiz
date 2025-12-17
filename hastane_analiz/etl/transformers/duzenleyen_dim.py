import pandas as pd

# Bu fonksiyon kolon adlarındaki Türkçe/küçük-büyük harf farklarını tolere etmek için
def _find_first_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols_upper = {c.upper(): c for c in df.columns}
    for cand in candidates:
        cand_u = cand.upper()
        if cand_u in cols_upper:
            return cols_upper[cand_u]
    return None


def transform_duzenleyen_dim(df: pd.DataFrame) -> pd.DataFrame:
    """
    DUZENLEYEN.xlsx dosyasini birim_def'e uygun hale getirir.

    Cikis kolonlari:
      - kurum_kodu
      - birim_adi
      - ilce_adi
      - baskanlik_adi
      - kurum_rol_adi
    """

    df = df.copy()

    # Buradaki candidate listelerini kendi Excel basliklarina gore duzenleyebilirsin
    col_kurum_kodu = _find_first_column(df, ["Kurum Kodu", "KURUM KODU", "KurumKodu"])
    col_birim_adi  = _find_first_column(df, ["Kurum Adi", "Kurum Adı", "Birim Adi", "Birim Adı"])
    col_ilce_adi   = _find_first_column(df, ["Ilce Adi", "İlçe Adı", "Ilce", "İlçe"])
    col_baskanlik  = _find_first_column(df, ["Baskanlik Adi", "Başkanlık Adı", "Baskanlik", "Başkanlık"])
    col_kurum_rol  = _find_first_column(df, ["Kurum Rol Adi", "Kurum Rolü", "Rol", "Kurum Rol Adı"])
    col_kurum_tip  = _find_first_column(df, ["KurumTipi"])
    if col_kurum_kodu is None:
        raise ValueError("DUZENLEYEN dosyasinda 'Kurum Kodu' kolonu bulunamadi. Kolon adlarini kontrol et.")

    # Yeni DataFrame'i standart kolon isimleriyle kur
    out = pd.DataFrame()
    out["kurum_kodu"] = pd.to_numeric(df[col_kurum_kodu], errors="coerce")

    if col_birim_adi is not None:
        out["birim_adi"] = df[col_birim_adi].astype(str).str.strip()
    else:
        out["birim_adi"] = None

    if col_ilce_adi is not None:
        out["ilce_adi"] = df[col_ilce_adi].astype(str).str.strip()
    else:
        out["ilce_adi"] = None

    if col_baskanlik is not None:
        out["baskanlik_adi"] = df[col_baskanlik].astype(str).str.strip()
    else:
        out["baskanlik_adi"] = None

    if col_kurum_rol is not None:
        out["kurum_rol_adi"] = df[col_kurum_rol].astype(str).str.strip()
    else:
        out["kurum_rol_adi"] = None
    if col_kurum_tip is not None:
        out["kurum_tipi"] = df[col_kurum_tip].astype(str).str.strip()
    else:
        out["kurum_tipi"] = None
    # Kurum kodu olmayan satirlari at
    out = out.dropna(subset=["kurum_kodu"])
    out["kurum_kodu"] = out["kurum_kodu"].astype("int64")

    # Tekrarlari azaltmak icin kurum_kodu bazinda uniq yapalim
    out = out.drop_duplicates(subset=["kurum_kodu"])

    return out
