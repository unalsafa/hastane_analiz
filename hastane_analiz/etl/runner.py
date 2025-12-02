# hastane_analiz/etl/runner.py


from pathlib import Path
import pandas as pd

from hastane_analiz.config.settings import INPUT_FOLDER
from hastane_analiz.etl.excel_reader import read_excel_file, detect_category_from_filename
from hastane_analiz.etl.transformers.acil import transform_acil
from hastane_analiz.etl.loaders import insert_long_df_to_raw_veri


def run_etl_for_folder(folder: str | None = None) -> None:
    base = Path(folder or INPUT_FOLDER)
    excel_files = list(base.glob("*.xlsx"))
    print(f"[ETL] Klasor: {base} | Bulunan dosya: {len(excel_files)}")

    for file_path in excel_files:
        kategori = detect_category_from_filename(file_path)
        print(f"[ETL] Isleniyor: {file_path.name} | Kategori: {kategori}")

        df = read_excel_file(file_path)

        # 1) DUZENLEYEN -> fact ETL'de tamamen atla
        if kategori == "DUZENLEYEN":
            print(
                "  -> DUZENLEYEN dosyasi fact ETL'de atlandi. "
                "Bu dosya icin ayri dim ETL calistiriliyor (birim_def)."
            )
            continue

        # 2) ACIL transformer
        if kategori == "ACIL":
            long_df = transform_acil(df, sheet_name="ACIL")
            print(f"  -> [ACIL] Long satir sayisi: {len(long_df)}")

            if len(long_df) > 0:
                insert_long_df_to_raw_veri(
                    long_df=long_df,
                    kategori=kategori,
                    file_path=str(file_path),
                    sayfa_adi="ACIL",
                )
                print("  -> [ACIL] raw_veri'ye insert edildi.")
            else:
                print("  -> [ACIL] Long DF bos, insert yapilmadi.")
        else:
            print(f"  -> Bu kategori icin henuz transformer yok: {kategori}")

