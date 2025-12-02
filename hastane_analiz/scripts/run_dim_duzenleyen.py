from pathlib import Path

from hastane_analiz.config.settings import INPUT_FOLDER
from hastane_analiz.etl.excel_reader import read_excel_file, detect_category_from_filename
from hastane_analiz.etl.transformers.duzenleyen_dim import transform_duzenleyen_dim
from hastane_analiz.etl.loaders_birim_def import upsert_birim_def

def main():
    base = Path(INPUT_FOLDER)
    excel_files = list(base.glob("*.xlsx"))

    print(f"[DIM-DUZENLEYEN] Klasor: {base} | Dosya sayisi: {len(excel_files)}")

    for file_path in excel_files:
        kategori = detect_category_from_filename(file_path)
        if kategori != "DUZENLEYEN":
            continue

        print(f"[DIM-DUZENLEYEN] Isleniyor: {file_path.name}")

        df_raw = read_excel_file(file_path)
        dim_df = transform_duzenleyen_dim(df_raw)
        print(f"  -> Donusen satir sayisi: {len(dim_df)}")

        upsert_birim_def(dim_df)
        print("  -> birim_def'e upsert edildi.")

if __name__ == "__main__":
    main()
