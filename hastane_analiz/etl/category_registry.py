from dataclasses import dataclass
from typing import List


@dataclass
class CategoryRule:
    name: str
    keywords: List[str]   # Dosya adinda aranacak kelimeler (case-insensitive)
    id: int               # Ileride dimension tablo ile eslestirmek icin


# Tüm kategoriler burada tek noktadan yönetilecek
CATEGORY_RULES = [
    CategoryRule(name="ACIL", keywords=["ACIL", "AÇIL", "ACİL"], id=1),
    CategoryRule(name="AMELIYATHANE", keywords=["AMELIYATHANE", "AMELİYATHANE", "AMELIYAT", "AMELİYAT"], id=2),
    CategoryRule(name="DOGUM", keywords=["DOGUM", "DOĞUM"], id=3),
    CategoryRule(name="DUZENLEYEN", keywords=["DUZENLEYEN", "DÜZENLEYEN"], id=4),
    CategoryRule(name="YOGUNBAKIM", keywords=["YOGUN", "YOĞUN", "YOGUN BAKIM", "YOĞUN BAKIM"], id=5),
    CategoryRule(name="ANADAL_YANDAL", keywords=["ANADAL YANDAL"], id=6),
    CategoryRule(name="BINA", keywords=["BINA", "BİNA"], id=7),
    CategoryRule(name="GORUNTULEME", keywords=["GORUNTULEME", "GÖRÜNTÜLEME"], id=8),
    CategoryRule(name="HIZMETLER_1", keywords=["HIZMETLER-1", "HİZMETLER-1"], id=9),
    CategoryRule(name="HIZMETLER_2", keywords=["HIZMETLER-2", "HİZMETLER-2"], id=10),
    CategoryRule(name="PERSONEL", keywords=["PERSONEL", "PERSONELLER"], id=11),
    # ihtiyac oldukça buraya yeni CategoryRule ekleyeceksin
]


DEFAULT_CATEGORY = CategoryRule(name="GENEL", keywords=[], id=999)
