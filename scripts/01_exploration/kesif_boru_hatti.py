"""
KEŞİF AŞAMASI BORU HATTI
========================
Amaç: Aynı problemleri bir modele iki dilde (EN/TR) çözdürüp, üretilen
muhakeme zincirlerinin token/kelime/morfem yükünü ve doğruluğunu ölçmek.

Bu, makalenin "gözlem" bölümünün verisini üretir.

Colab'da çalıştırmak için:
  1. pip install zeyrek datasets nltk
  2. Model çağırma kısmını (cozum_uret) doldur (API veya yerel model)
  3. Çalıştır.
"""

import re
import json
import zeyrek

# ============================================================
# AŞAMA 1: VERİ HAZIRLAMA
# ============================================================
# Gerçek kullanımda: eşleşen EN/TR problemleri yükle.
#   from datasets import load_dataset
#   en = load_dataset("openai/gsm8k", "main", split="test")
#   tr = load_dataset("malhajar/gsm8k-tr", split="test")
# Burada iskeleti göstermek için elle birkaç eşleşen çift koyuyorum.

def veri_yukle(n=100):
    """Eşleşen EN/TR problem çiftlerini döndürür.
    Gerçekte HF'den yüklenecek; burada örnek."""
    ornekler = [
        {
            "en_q": "Natalia sold clips to 48 friends in April, then half as many in May. How many total?",
            "tr_q": "Natalia Nisan'da 48 arkadaşına toka sattı, Mayıs'ta bunun yarısı kadar sattı. Toplam kaç?",
            "cevap": "72",
        },
        {
            "en_q": "Weng earns 12 dollars an hour. She babysat 50 minutes. How much did she earn?",
            "tr_q": "Weng saatte 12 dolar kazanıyor. 50 dakika bakıcılık yaptı. Ne kadar kazandı?",
            "cevap": "10",
        },
    ]
    return ornekler[:n]


# ============================================================
# AŞAMA 2: ÜRETİM  (SEN DOLDURACAKSIN)
# ============================================================
def cozum_uret(soru, dil):
    """
    Modele bir soruyu verip adım adım çözüm (muhakeme zinciri) ürettirir.

    SEN BURAYI DOLDURACAKSIN. İki seçenek:

    --- SEÇENEK A: API (kolay) ---
    # from openai import OpenAI  (ya da başka sağlayıcı)
    # client = ...
    # prompt = f"Bu problemi adım adım çöz:\n{soru}" if dil=="tr" else f"Solve step by step:\n{soru}"
    # yanit = client.chat.completions.create(...)
    # return yanit.choices[0].message.content

    --- SEÇENEK B: Yerel model (ücretsiz, GPU gerekir) ---
    # from transformers import pipeline
    # global pipe  (bir kez yüklenir)
    # return pipe(prompt)[0]["generated_text"]

    Şimdilik sahte (placeholder) bir çıktı döndürüyorum ki boru hattı test edilebilsin.
    """
    if dil == "tr":
        return ("Öncelikle problemi dikkatlice inceleyelim. Natalia Nisan ayında "
                "48 arkadaşına toka sattı. Mayıs ayında bunun yarısı kadar sattığına göre "
                "48'in yarısını hesaplamamız gerekiyor, bu da 24 eder. Dolayısıyla toplam "
                "satış miktarı 48 artı 24'tür. Sonuç olarak 72 toka satmıştır. Cevap: 72")
    else:
        return ("First, Natalia sold 48 clips in April. In May she sold half, so 48/2 = 24. "
                "Total is 48 + 24 = 72. Answer: 72")


# ============================================================
# AŞAMA 3: DOĞRULUK KONTROLÜ
# ============================================================
def cevap_ayikla(metin):
    """Çözüm metninden nihai sayısal cevabı çıkarır."""
    # "Cevap: X" / "Answer: X" kalıbını ara, yoksa son sayıyı al
    m = re.search(r"(?:cevap|answer)\s*[:=]?\s*\$?(-?\d[\d,\.]*)", metin, re.IGNORECASE)
    if m:
        return m.group(1).replace(",", "").rstrip(".")
    sayilar = re.findall(r"-?\d[\d,\.]*", metin)
    return sayilar[-1].replace(",", "").rstrip(".") if sayilar else None

def dogru_mu(uretilen_metin, gercek_cevap):
    tahmin = cevap_ayikla(uretilen_metin)
    return tahmin is not None and tahmin == gercek_cevap.strip()


# ============================================================
# AŞAMA 4: ÖLÇÜM
# ============================================================
_analyzer = zeyrek.MorphAnalyzer()

def kelime_say(metin):
    return len(re.findall(r"\w+", metin, re.UNICODE))

def morfem_yuku(metin):
    """Türkçe metnin toplam morfem yükünü döndürür."""
    kelimeler = re.findall(r"\w+", metin, re.UNICODE)
    toplam = 0
    for k in kelimeler:
        if k.isdigit():
            toplam += 1
            continue
        try:
            s = _analyzer.analyze(k)
            if s and s[0]:
                toplam += len(s[0][0].morphemes)
        except Exception:
            pass
    return toplam

def olc(metin, dil):
    """Bir çözüm zincirinin metriklerini döndürür."""
    sonuc = {"kelime": kelime_say(metin)}
    if dil == "tr":
        sonuc["morfem"] = morfem_yuku(metin)
        sonuc["morfem_kelime_orani"] = round(sonuc["morfem"] / max(sonuc["kelime"], 1), 2)
    return sonuc


# ============================================================
# AŞAMA 5: ANA DÖNGÜ + ANALİZ
# ============================================================
def kesif_calistir(n=100):
    veri = veri_yukle(n)
    kayitlar = []

    for ornek in veri:
        # İki dilde çözüm üret
        en_cozum = cozum_uret(ornek["en_q"], "en")
        tr_cozum = cozum_uret(ornek["tr_q"], "tr")

        kayit = {
            "cevap": ornek["cevap"],
            "en_dogru": dogru_mu(en_cozum, ornek["cevap"]),
            "tr_dogru": dogru_mu(tr_cozum, ornek["cevap"]),
            "en_olcum": olc(en_cozum, "en"),
            "tr_olcum": olc(tr_cozum, "tr"),
        }
        kayitlar.append(kayit)

    # Toplu analiz
    n_ornek = len(kayitlar)
    en_kelime = sum(k["en_olcum"]["kelime"] for k in kayitlar) / n_ornek
    tr_kelime = sum(k["tr_olcum"]["kelime"] for k in kayitlar) / n_ornek
    tr_morfem = sum(k["tr_olcum"]["morfem"] for k in kayitlar) / n_ornek
    en_dogruluk = sum(k["en_dogru"] for k in kayitlar) / n_ornek
    tr_dogruluk = sum(k["tr_dogru"] for k in kayitlar) / n_ornek

    print("="*60)
    print(f"KEŞİF SONUÇLARI ({n_ornek} problem)")
    print("="*60)
    print(f"Ort. EN çözüm uzunluğu (kelime) : {en_kelime:.1f}")
    print(f"Ort. TR çözüm uzunluğu (kelime) : {tr_kelime:.1f}")
    print(f"Ort. TR morfem yükü             : {tr_morfem:.1f}")
    print(f"  -> TR morfem / EN kelime      : {tr_morfem/en_kelime:.2f}x  (toplam morfolojik yük)")
    print(f"  -> TR morfem / TR kelime      : {tr_morfem/tr_kelime:.2f}   (kelime başına yoğunluk)")
    print(f"Doğruluk EN / TR                : {en_dogruluk:.0%} / {tr_dogruluk:.0%}")
    print()

    # Ham kayıtları da diske yaz (sonra derin analiz için)
    with open("kesif_kayitlar.json", "w", encoding="utf-8") as f:
        json.dump(kayitlar, f, ensure_ascii=False, indent=2)
    print("Ham kayıtlar 'kesif_kayitlar.json' dosyasına yazıldı.")
    return kayitlar


if __name__ == "__main__":
    kesif_calistir(n=100)
