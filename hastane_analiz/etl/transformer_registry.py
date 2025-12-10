from dataclasses import dataclass
from typing import Callable, Optional, List
import pandas as pd

from hastane_analiz.etl.transformers.wide_to_long import transform_wide_to_long


@dataclass
class TransformerEntry:
    fn: Callable[..., pd.DataFrame]
    sayfa_adi: Optional[str] = None


# kategori -> list of TransformerEntry. Aynı kategori içinde birden çok sayfa desteklenir
TRANSFORMER_REGISTRY: dict[str, List[TransformerEntry]] = {
    "ACIL": [TransformerEntry(fn=transform_wide_to_long, sayfa_adi="ACIL")],
    "DOGUM": [TransformerEntry(fn=transform_wide_to_long, sayfa_adi="DOĞUM")],
    "AMELIYATHANE": [TransformerEntry(fn=transform_wide_to_long, sayfa_adi="AMELİYATHANE")],
    "ANADAL_YANDAL": [TransformerEntry(fn=transform_wide_to_long, sayfa_adi="ANADAL YANDAL")],
    "BINA": [TransformerEntry(fn=transform_wide_to_long, sayfa_adi="BİNA")],
    "GORUNTULEME": [
        TransformerEntry(fn=transform_wide_to_long, sayfa_adi="GÖRÜNTÜLEME"),
        TransformerEntry(fn=transform_wide_to_long, sayfa_adi="DİĞER TIBBİ CİHAZ"),
        TransformerEntry(fn=transform_wide_to_long, sayfa_adi="ENDOSKOPİK CİHAZLAR"),
        TransformerEntry(fn=transform_wide_to_long, sayfa_adi="RADYASYON ONKOLOJİSİ"),
        TransformerEntry(fn=transform_wide_to_long, sayfa_adi="NÜKLEER TIP"),
    ],
    "HIZMETLER_1": [
        TransformerEntry(fn=transform_wide_to_long, sayfa_adi="HİZMETLER-1"),
        TransformerEntry(fn=transform_wide_to_long, sayfa_adi="GETAT"),
        TransformerEntry(fn=transform_wide_to_long, sayfa_adi="MERKEZ VE LABORATUVARLAR"),
    ],
    "HIZMETLER_2": [TransformerEntry(fn=transform_wide_to_long, sayfa_adi="HİZMETLER-2")],
    "PERSONEL": [
        TransformerEntry(fn=transform_wide_to_long, sayfa_adi="Personel Form Verisi"),
        TransformerEntry(fn=transform_wide_to_long, sayfa_adi="Diğer Personel Verileri"),
    ],
    "YOGUNBAKIM": [TransformerEntry(fn=transform_wide_to_long, sayfa_adi="YOĞUN BAKIM")],
    # Yeni kategoriler buraya eklenecek
}
