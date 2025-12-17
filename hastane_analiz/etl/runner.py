# hastane_analiz/etl/runner.py
from pathlib import Path
import pandas as pd

from hastane_analiz.config.settings import INPUT_FOLDER
from hastane_analiz.etl.excel_reader import detect_category_from_filename
from hastane_analiz.etl.transformer_registry import TRANSFORMER_REGISTRY, TransformerEntry
from hastane_analiz.etl.loaders import insert_long_df_to_raw_veri
from hastane_analiz.etl.validation import run_validations, save_issues_to_db, Severity, save_issues_snapshot, infer_period_from_df



def run_etl_for_folder(folder: str | None = None) -> None:
    base = Path(folder or INPUT_FOLDER)
    excel_files = list(base.glob("*.xlsx"))
    print(f"[ETL] Klasor: {base} | Bulunan dosya: {len(excel_files)}")

    for file_path in excel_files:
        kategori = detect_category_from_filename(file_path)
        print(f"[ETL] Isleniyor: {file_path.name} | Kategori: {kategori}")

        # DUZENLEYEN -> fact ETL'de tamamen atla
        if kategori == "DUZENLEYEN":
            print(
                "  -> DUZENLEYEN dosyasi fact ETL'de atlandi. "
                "Bu dosya icin ayri dim ETL calistiriliyor (birim_def)."
            )
            continue

        entries = TRANSFORMER_REGISTRY.get(kategori, [])
        if not entries:
            print(f"  -> Bu kategori icin henuz transformer yok: {kategori}")
            continue

        try:
            with pd.ExcelFile(file_path) as xls:
                available_sheets = set(xls.sheet_names)

                for entry in entries:
                    sheet = entry.sayfa_adi
                    if sheet and sheet not in available_sheets:
                        print(f"  -> Sayfa bulunamadi, atlandi: {sheet}")
                        continue

                    # Sayfa belirtilmemisse ilk sayfa
                    target_sheet = sheet or 0
                    df = xls.parse(sheet_name=target_sheet)

                    # Transformer'i calistir
                    try:
                        long_df = entry.fn(df, kategori=kategori, sayfa_adi=sheet)
                    except TypeError:
                        # Geriye dönük imza icin sadece df + sheet_name gonder
                        long_df = entry.fn(df, sheet_name=sheet or "Sheet1")

                    print("[DEBUG] long_df kolonlar:", long_df.columns.tolist())
                    print(f"  -> [{kategori}]/{sheet or 'DEFAULT'} Long satir sayisi: {len(long_df)}")

                    if len(long_df) == 0:
                        print("  -> Long DF bos, insert yapilmadi.")
                        continue

                    issues = run_validations(long_df, file_path=str(file_path), kategori=kategori, sayfa_adi=sheet)

                    # Dosyanın dönemi (yil, ay) – her ACIL dosyası için tek dönem bekliyoruz
                    period = infer_period_from_df(long_df)

                    if period is not None:
                        yil, ay = period
                        save_issues_snapshot(issues, kategori=kategori, yil=yil, ay=ay)
                    else:
                        # Güvenli fallback: dönem bulunamazsa sadece insert (snapshot yapma)
                        save_issues_to_db(issues)


                    fatal_count = sum(1 for i in issues if i.severity == Severity.FATAL)
                    if fatal_count > 0:
                        print(f"  -> [VALIDATION] {fatal_count} FATAL hata var, sayfa yuklenmedi.")
                        continue

                    insert_long_df_to_raw_veri(
                        long_df=long_df,
                        kategori=kategori,
                        file_path=str(file_path),
                        sayfa_adi=sheet,
                    )
                    print(f"  -> [{kategori}]/{sheet or 'DEFAULT'} raw_veri'ye insert/upssert edildi.")
        except Exception as exc:
            print(f"  -> Dosya islenirken hata olustu: {exc}")
