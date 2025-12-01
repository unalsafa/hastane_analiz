# hastane_analiz/etl/runner.py

from pathlib import Path
from typing import Optional

from hastane_analiz.etl.excel_reader import read_excel_file, detect_category_from_filename
from hastane_analiz.etl.transformers.duzenleyen import transform_duzenleyen
from hastane_analiz.etl.loaders import insert_long_df_to_raw_veri
from hastane_analiz.config.settings import INPUT_FOLDER

def run_etl_for_folder(folder: Optional[str] = None):
    base = Path(folder or INPUT_FOLDER)
    excel_files = list(base.glob("*.xlsx"))

    print(f"[ETL] Klasor: {base} | Bulunan dosya: {len(excel_files)}")

    for file_path in excel_files:
        kategori = detect_category_from_filename(file_path)
        print(f"[ETL] Isleniyor: {file_path.name} | Kategori: {kategori}")

        df = read_excel_file(file_path)

        # TODO: Dosya adindan yil/ay parse edecegiz.
        # Simdilik sabit bir deger kullanalim.
        yil = 2024
        ay = 2

        if kategori == "DUZENLEYEN":
            long_df = transform_duzenleyen(df, yil=yil, ay=ay)
            print(f"  -> Long satir sayisi: {len(long_df)}")
            if len(long_df) > 0:
                insert_long_df_to_raw_veri(
                    long_df=long_df,
                    kategori=kategori,
                    file_path=str(file_path),
                )
                print("  -> raw_veri'ye insert edildi.")
            else:
                print("  -> Long DF bos, insert yapilmadi.")
        else:
            print(f"  -> Bu kategori icin henuz transformer yok: {kategori}")
