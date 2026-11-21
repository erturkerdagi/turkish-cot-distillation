# -*- coding: utf-8 -*-
"""
rev_J_sozcuk.py — Tablo 3'ün "Word" sütununu yeniden ölçer.

Sadece CPU. GPU yok, model yüklenmez, eğitim yoktur. ~1-2 dakika.
Hiçbir mevcut dosyayı DEĞİŞTİRMEZ; tek çıktı revizyon_ciktilari/revJ_sozcuk.json.

Ne yapar: degerlendirme1000 klasöründeki 12 ana koşu dosyasını okur ve her
koşu için üretilen cevabın ortalama sözcük sayısını hesaplar. Kayıtlarda
hazır "kelime" alanı varsa onu kullanır; yoksa üretilen metinden sayar.
Her iki yolu da ayrı ayrı raporlar, böylece tutarlılık görülür.
"""

import os, re, io, json, glob

ANA_DIZIN = os.environ.get(
    "PROJE_DIZINI",
    "/content/drive/MyDrive/turkish-cot-distillation/Project",
)
ANA_KLASOR_ADI = "degerlendirme1000"
CIKTI_KLASORU = "revizyon_ciktilari"

AD = {"qwen05": "Qwen2.5-0.5B", "qwen15": "Qwen2.5-1.5B",
      "qwen3": "Qwen2.5-3B", "llama3": "Llama-3.2-3B"}
SURUM = {"orijinal": "Original", "sade_morfem": "Morpheme-simplified",
         "sade_kelime": "Word-simplified"}
METIN_ALANLARI = ("metin", "cikti", "uretilen", "cevap_metni", "yanit", "output", "text")


def klasor_bul():
    """degerlendirme1000'i ANA_DIZIN'den yukarı doğru ve Drive genelinde arar."""
    adaylar = []
    d = ANA_DIZIN
    for _ in range(6):
        adaylar.append(os.path.join(d, ANA_KLASOR_ADI))
        ust = os.path.dirname(d)
        if ust == d:
            break
        d = ust
    for a in adaylar:
        if os.path.isdir(a):
            return a
    for kok in ("/content/drive/MyDrive", "/content/drive"):
        if os.path.isdir(kok):
            for y in glob.glob(os.path.join(kok, "**", ANA_KLASOR_ADI), recursive=True):
                if os.path.isdir(y):
                    return y
    return None


def sozcuk_say(s):
    return len([p for p in re.split(r"\s+", (s or "").strip()) if p])


def main():
    klasor = klasor_bul()
    if not klasor:
        print(f"!! '{ANA_KLASOR_ADI}' klasörü bulunamadı. ANA_DIZIN'i kontrol edin.")
        return
    print(f"Klasör: {klasor}\n")

    sonuc, uyari = [], []
    for m in ("qwen05", "qwen15", "qwen3", "llama3"):
        for v in ("orijinal", "sade_morfem", "sade_kelime"):
            esler = [y for y in glob.glob(os.path.join(klasor, f"{m}_{v}*.jsonl"))
                     if "checkpoint-" not in y]
            if not esler:
                uyari.append(f"{m}_{v}: dosya yok")
                continue
            yol = max(esler, key=os.path.getsize)

            alan_top = alan_n = metin_top = metin_n = 0
            with io.open(yol, encoding="utf-8") as f:
                for satir in f:
                    satir = satir.strip()
                    if not satir:
                        continue
                    try:
                        k = json.loads(satir)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(k.get("kelime"), (int, float)):
                        alan_top += float(k["kelime"]); alan_n += 1
                    for ad in METIN_ALANLARI:
                        if isinstance(k.get(ad), str):
                            metin_top += sozcuk_say(k[ad]); metin_n += 1
                            break

            kayit = {"model": AD[m], "surum": SURUM[v], "dosya": os.path.basename(yol),
                     "alan_ortalama": round(alan_top / alan_n, 1) if alan_n else None,
                     "alan_n": alan_n,
                     "metin_ortalama": round(metin_top / metin_n, 1) if metin_n else None,
                     "metin_n": metin_n}
            sonuc.append(kayit)
            a = f"{kayit['alan_ortalama']}" if alan_n else "—"
            t = f"{kayit['metin_ortalama']}" if metin_n else "—"
            print(f"  {AD[m]:14} {SURUM[v]:20} kelime alanı={a:>7} (n={alan_n:3})   "
                  f"metinden={t:>7} (n={metin_n:3})")
            if alan_n and metin_n and abs(kayit["alan_ortalama"] - kayit["metin_ortalama"]) > 1.0:
                uyari.append(f"{m}_{v}: alan ve metin sayımı farklı "
                             f"({kayit['alan_ortalama']} vs {kayit['metin_ortalama']})")

    if uyari:
        print("\nUYARILAR:")
        for u in uyari:
            print("  !!", u)
    if not any(r["alan_n"] or r["metin_n"] for r in sonuc):
        print("\n!! Hiçbir dosyada sözcük sayısı ya da metin alanı bulunamadı.")
        print("   Tablo 3'ün Word sütunu eski değerleriyle bırakılmalıdır.")

    hedef = os.path.join(os.path.dirname(klasor), CIKTI_KLASORU)
    os.makedirs(hedef, exist_ok=True)
    cikti = os.path.join(hedef, "revJ_sozcuk.json")
    with io.open(cikti, "w", encoding="utf-8") as f:
        json.dump({"sonuclar": sonuc, "uyarilar": uyari}, f, ensure_ascii=False, indent=2)
    print(f"\nYazıldı: {cikti}")
    print("Bu dosyayı bana verin; Tablo 3'ün Word sütununu ona göre dolduracağım.")


if __name__ == "__main__":
    main()
