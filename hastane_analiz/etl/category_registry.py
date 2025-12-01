from dataclasses import dataclass
from typing import List

@dataclass
class CategoryRule:
    name: str
    keywords: List[str]   # Dosya adında aranacak kelimeler (case-insensitive)
    id: int               # Ileride dimension tablo ile eslestirmek icin

# Tüm kategoriler burada tek noktadan yönetilecek
CATEGORY_RULES = [
    CategoryRule(name="ACIL",         keywords=["ACIL", "AÇIL"],           id=1),
    CategoryRule(name="AMELIYATHANE", keywords=["AMELIYAT", "AMELİYAT"],   id=2),
    CategoryRule(name="DOGUM",        keywords=["DOGUM", "DOĞUM"],         id=3),
    CategoryRule(name="DUZENLEYEN",   keywords=["DUZENLEYEN", "DÜZENLEYEN"], id=4),
    CategoryRule(name="YOGUNBAKIM",   keywords=["YOGUN", "YOĞUN"],         id=5),
    # ihtiyac oldukça buraya yeni CategoryRule ekleyeceksin
]

DEFAULT_CATEGORY = CategoryRule(name="GENEL", keywords=[], id=999)
