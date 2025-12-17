from pathlib import Path
import pandas as pd
from hastane_analiz.etl.category_registry import CATEGORY_RULES, DEFAULT_CATEGORY

def read_excel_file(file_path: Path, sheet_name: str | int | None = 0) -> pd.DataFrame:
    """
    Excel dosyasini okur.
    - sheet_name None olursa: ilk sheet (0) kullanilir.
    - Birden fazla sheet varsa ve sheet_name=None verilse bile
      ilk sheet DataFrame olarak dondurulur.
    """
    if sheet_name is None:
        sheet_name = 0  # ilk sayfa

    df = pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")

    # Eger dict donerse (coklu sheet ve sheet_name=None durumu icin emniyet)
    if isinstance(df, dict):
        # Ilk sheet'i al
        first_key = next(iter(df))
        df = df[first_key]

    return df


def detect_category_from_filename(file_path: Path) -> str:
    """
    Dosya adindan kategori belirler.
    Tüm karar mekanizmasi category_registry icindedir.
    """
    name = file_path.stem.upper()

    for rule in CATEGORY_RULES:
        for keyword in rule.keywords:
            if keyword.upper() in name:
                return rule.name

    return DEFAULT_CATEGORY.name
