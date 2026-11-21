# ============================================================
#  KNOSYS REVİZYONU — TÜM CPU ANALİZLERİ, TEK ÇALIŞTIRMA
# ------------------------------------------------------------
#  GÜVENLİK GARANTİSİ
#    * Mevcut hiçbir dosya DEĞİŞTİRİLMEZ veya SİLİNMEZ.
#    * Girdi dosyaları yalnızca okunur (salt-okunur açılır).
#    * Her çıktı tek bir YENİ klasöre yazılır: <ANA_DIZIN>/revizyon_ciktilari/
#    * Sonunda bu klasörün .zip arşivi üretilir (paylaşmak için).
#
#  KARŞILANAN HAKEM MADDELERİ
#    M1.1  M1.2  M1.3  M2.1  M2.2  M2.3  M2.4  M3.2  M3.3
#    M4.1  M4.2  M4.3  M4.4  M6.1  M6.2  M6.3  M6.4  M6.6
#    M7.2  M7.3  R3-1  R3-2  R3-3  R3-6
#
#  GPU GEREKMEZ.  Tahmini süre: 20-40 dakika (Zeyrek en yavaş kısım).
# ============================================================
#  COLAB KULLANIMI
#    from google.colab import drive
#    drive.mount('/content/drive')
#    %run rev_TUMU.py
# ============================================================

import subprocess, sys
print("Paketler kuruluyor...")
for paket in ["zeyrek", "nltk", "transformers", "datasets", "morfessor",
              "statsmodels", "pandas", "numpy", "scipy", "tqdm"]:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", paket],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Zeyrek içeride NLTK'nın cümle bölücüsünü kullanır; veri paketleri şart.
print("NLTK veri paketleri indiriliyor...")
import nltk
for _p in ["punkt_tab", "punkt"]:
    try:
        nltk.download(_p, quiet=True)
    except Exception as _e:
        print(f"  uyarı: {_p} indirilemedi ({_e})")

import os, re, io, json, glob, random, shutil, logging, traceback
import statistics as ist
import itertools
from collections import Counter
from datetime import datetime

# ============================================================
#  AYARLAR  —  tek değiştirmeniz gereken yer ANA_DIZIN
# ============================================================
ANA_DIZIN = os.environ.get(
    "PROJE_DIZINI",
    "/content/drive/MyDrive/turkish-cot-distillation/Project",
)

CIKTI_KLASOR_ADI = "revizyon_ciktilari"

OGRETMEN_ADI = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
OGRENCI_ADI  = "Qwen/Qwen2.5-1.5B-Instruct"

KESIF_TOKEN_SINIRI  = 1024     # colab_kesif_v4.py
EGITIM_TOKEN_SINIRI = 1024     # asama3_tam_egitim_v2.py
BENZERLIK_ESIGI     = 0.75     # asama2_sadelestirme.py
DENKLIK_MARJI       = 0.03     # ±3 puan (M4.3)
BOOTSTRAP_N         = 10000
TOHUM               = 42

MODELLER    = ["qwen05", "qwen15", "qwen3", "llama3"]
VERSIYONLAR = ["orijinal", "sade_morfem", "sade_kelime"]
# ============================================================

CIKTI_DIR = os.path.join(ANA_DIZIN, CIKTI_KLASOR_ADI)
os.makedirs(CIKTI_DIR, exist_ok=True)
random.seed(TOHUM)
SONUC = {"calisma_zamani": datetime.now().isoformat(timespec="seconds"),
         "ana_dizin": ANA_DIZIN}


# ------------------------------------------------------------
#  Ekrana ve dosyaya aynı anda yazan kayıt tutucu
# ------------------------------------------------------------
class Kayitci(io.TextIOBase):
    def __init__(self, dosya_yolu):
        self.terminal = sys.__stdout__
        self.dosya = open(dosya_yolu, "w", encoding="utf-8")
    def write(self, s):
        self.terminal.write(s); self.dosya.write(s); self.dosya.flush()
        return len(s)
    def flush(self):
        self.terminal.flush(); self.dosya.flush()

KAYIT_YOLU = os.path.join(CIKTI_DIR, "00_calisma_kaydi.txt")
sys.stdout = Kayitci(KAYIT_YOLU)


def baslik(s, seviye=1):
    c = "=" if seviye == 1 else "-"
    print("\n" + c * 76); print(s); print(c * 76)

def alt(s):
    print(f"\n  >> {s}")


# ------------------------------------------------------------
#  Girdi dosyalarını ADIYLA bul (yol değişse de çalışsın)
# ------------------------------------------------------------
DRIVE_KOKU = "/content/drive/MyDrive"

def arama_kokleri():
    """ANA_DIZIN'den başlayıp MyDrive'a kadar üst klasörleri sırayla döndürür.
    Dosyalar taşınmış olsa da bulunur; çıktı yine ANA_DIZIN'e yazılır."""
    kokler, p = [os.path.abspath(ANA_DIZIN)], os.path.abspath(ANA_DIZIN)
    sinir = os.path.abspath(DRIVE_KOKU)
    while os.path.normpath(p) != sinir:
        ust = os.path.dirname(p)
        if not ust or ust == p or not ust.startswith(sinir):
            break
        kokler.append(ust)
        p = ust
    return kokler

KOKLER = arama_kokleri()

def _ele(yollar):
    return [y for y in yollar if CIKTI_KLASOR_ADI not in y]

def dosya_bul(desen, zorunlu=True):
    """Dosyayı ADIYLA değil DESENİYLE arar (asama2_sadelesmis*.jsonl gibi),
    böylece v1000 benzeri sürüm ekleri sorun çıkarmaz. Birden çok eşleşmede
    EN BÜYÜK dosya seçilir — yani en kapsamlı koşu."""
    for kok in KOKLER:
        bulunan = _ele(glob.glob(os.path.join(kok, desen)) +
                       glob.glob(os.path.join(kok, "**", desen), recursive=True))
        bulunan = sorted(set(bulunan), key=lambda y: -os.path.getsize(y))
        if bulunan:
            if len(bulunan) > 1:
                print(f"  ~ {desen}: {len(bulunan)} eşleşme, en büyüğü seçildi")
                for b in bulunan[:4]:
                    print(f"      {os.path.getsize(b)/1024:9.0f} KB  {b}")
            return bulunan[0]
    if zorunlu:
        print(f"  !! BULUNAMADI: {desen}")
    return None

def klasor_bul(ad):
    for kok in KOKLER:
        dogrudan = os.path.join(kok, ad)
        if os.path.isdir(dogrudan):
            return dogrudan
        for k in _ele(glob.glob(os.path.join(kok, "**", ad), recursive=True)):
            if os.path.isdir(k):
                return k
    return None

def jsonl_oku(yol):
    with open(yol, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def json_oku(yol):
    with open(yol, encoding="utf-8") as f:
        return json.load(f)

def kaydet(ad, nesne):
    yol = os.path.join(CIKTI_DIR, ad)
    with open(yol, "w", encoding="utf-8") as f:
        json.dump(nesne, f, ensure_ascii=False, indent=2, default=str)
    return yol


_REF_ONBELLEK = {}
def referans_sozlugu():
    """{soru: referans_cozum} döndürür.
    Önce yerel asama1_ogretmen_verisi dosyasını dener; yoksa gsm8k-tr veri
    kümesini doğrudan indirir (referans çözümler zaten veri kümesinde var)."""
    if _REF_ONBELLEK:
        return _REF_ONBELLEK
    if YOL.get("referans"):
        try:
            for r in jsonl_oku(YOL["referans"]):
                _REF_ONBELLEK[r["soru"]] = r.get("ogretmen_cozum", "")
            print(f"    Referans çözümler yerel dosyadan: {len(_REF_ONBELLEK)}")
            return _REF_ONBELLEK
        except Exception as e:
            print(f"    Yerel referans okunamadı ({e}); veri kümesine geçiliyor.")
    try:
        from datasets import load_dataset
        for bolum in ["train", "test"]:
            try:
                ds = load_dataset("malhajar/gsm8k-tr", "main", split=bolum)
            except Exception:
                continue
            for i in range(len(ds)):
                _REF_ONBELLEK[ds[i]["question"]] = ds[i]["answer"]
        print(f"    Referans çözümler gsm8k-tr veri kümesinden: {len(_REF_ONBELLEK)}")
    except Exception as e:
        print(f"    !! Referans çözümler alınamadı: {e}")
    return _REF_ONBELLEK


baslik("0. HAZIRLIK — DOSYALARIN BULUNMASI")
print(f"  Ana dizin : {ANA_DIZIN}")
print(f"  Çıktı     : {CIKTI_DIR}   (yalnızca buraya yazılır)")
if not os.path.isdir(ANA_DIZIN):
    print("\n  !! ANA_DIZIN bulunamadı. Drive bağlı mı? Yolu kontrol edin.")
    sys.exit(1)

YOL = {
    "kesif":     dosya_bul("kesif_kayitlar*.json"),
    "zincirler": dosya_bul("asama1_ogretmen_zincirleri*.jsonl"),
    "referans":  dosya_bul("asama1_ogretmen_verisi*.jsonl", zorunlu=False),
    "sade":      dosya_bul("asama2_sadelesmis*.jsonl"),
}
DEG_DIR   = klasor_bul("degerlendirme")
TOHUM_DIR = klasor_bul("degerlendirme_tohum")
print()
for k, v in YOL.items():
    print(f"  {k:14}: {v if v else '— yok —'}")
print(f"  {'degerlendirme':14}: {DEG_DIR if DEG_DIR else '— yok —'}")
print(f"  {'tohum':14}: {TOHUM_DIR if TOHUM_DIR else '— yok —'}")
SONUC["bulunan_dosyalar"] = {**YOL, "degerlendirme": DEG_DIR, "tohum": TOHUM_DIR}

if not any(YOL.values()):
    print("\n  !! Hiçbir girdi dosyası bulunamadı. Arama yapılan kökler:")
    for kok in KOKLER:
        print(f"       {kok}")
    print("\n  ANA_DIZIN içeriği:")
    try:
        for d in sorted(os.listdir(ANA_DIZIN))[:40]:
            tur = "[klasör]" if os.path.isdir(os.path.join(ANA_DIZIN, d)) else "        "
            print(f"       {tur} {d}")
    except Exception as e:
        print(f"       okunamadı: {e}")
    print("\n  Drive genelinde dosya adıyla arama yapılıyor (bir dakika sürebilir)...")
    for ad in ["kesif_kayitlar*.json", "asama1_ogretmen_zincirleri*.jsonl",
               "asama2_sadelesmis*.jsonl", "ozet_tablo*.json"]:
        vur = _ele(glob.glob(os.path.join(DRIVE_KOKU, "**", ad), recursive=True))
        print(f"    {ad}:")
        for v in vur[:5]:
            print(f"       {v}")
        if not vur:
            print("       — Drive'da hiç bulunamadı —")
    print("\n  Yukarıdaki yollardan birini ANA_DIZIN olarak ayarlayıp tekrar çalıştırın.")


# ------------------------------------------------------------
#  Zeyrek — sağlık testi (sessiz başarısızlığa karşı)
# ------------------------------------------------------------
import zeyrek
logging.getLogger("zeyrek").setLevel(logging.ERROR)
_analyzer = zeyrek.MorphAnalyzer()

_test = "Öğrencilerin kitaplarını okuyabilmeleri için kütüphaneye gittiler"
_kt = len(re.findall(r"\w+", _test, re.UNICODE)); _mt = 0
try:
    for _w in re.findall(r"\w+", _test, re.UNICODE):
        _s = _analyzer.analyze(_w)
        if _s and _s[0]:
            _mt += len(_s[0][0].morphemes)
except LookupError as _e:
    print("\n  !! NLTK veri paketi eksik. Şu iki satırı ayrı bir hücrede çalıştırıp")
    print("     bu betiği yeniden başlatın:\n")
    print("       import nltk")
    print("       nltk.download('punkt_tab'); nltk.download('punkt')\n")
    print(f"  Ayrıntı: {_e}")
    sys.exit(1)
assert _mt / _kt >= 1.5, (
    f"HATA: Zeyrek düzgün çalışmıyor! morfem/kelime={_mt/_kt:.2f} (>=1.5 olmalı). "
    "Sonuçlar geçersiz olur — devam etmeyin.")
print(f"\n  Zeyrek sağlık testi GEÇTİ (morfem/kelime = {_mt/_kt:.2f})")

_mcache = {}
def morfem_kelime(k):
    if k.isdigit():
        return 1
    if k not in _mcache:
        try:
            s = _analyzer.analyze(k)
            _mcache[k] = len(s[0][0].morphemes) if (s and s[0]) else 0
        except Exception:
            _mcache[k] = 0
    return _mcache[k]

def morfem_toplam(m):
    return sum(morfem_kelime(k) for k in re.findall(r"\w+", m, re.UNICODE))

def kelime_say(m):
    return len(re.findall(r"\w+", m, re.UNICODE))

def cumlelere_bol(m):
    return [p.strip() for p in re.split(r"(?<!\d)[.!?\n]+(?!\d)", m) if p.strip()]

def zincir_temizle(m):
    m = re.sub(r"</?think>", " ", m)
    m = re.sub(r"\\boxed\{([^}]*)\}", r"\1", m)
    m = re.sub(r"[*#`]+", " ", m)
    m = re.sub(r"[\u4e00-\u9fff]+", " ", m)
    return re.sub(r"\s+", " ", m).strip()

ISLEM = re.compile(r"\d\s*[+\-*/=×÷]\s*\d|\d+\s*(?:kez|katı|yüzde|%)")
def cekirdek_mi(c): return bool(ISLEM.search(c))

from difflib import SequenceMatcher
def benzer_mi(a, b, esik=BENZERLIK_ESIGI):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= esik

def birlestir(cs): return (". ".join(cs) + ".") if cs else ""


def ozet_yaz(ad, degerler):
    if not degerler:
        return {}
    d = {"n": len(degerler), "ortalama": round(ist.mean(degerler), 2),
         "medyan": round(ist.median(degerler), 2),
         "ss": round(ist.pstdev(degerler), 2) if len(degerler) > 1 else 0.0,
         "min": min(degerler), "maks": max(degerler)}
    print(f"    {ad:34} n={d['n']:4}  ort={d['ortalama']:8.1f}  "
          f"med={d['medyan']:7.1f}  ss={d['ss']:7.1f}  [{d['min']}-{d['maks']}]")
    return d


# ============================================================
#  BÖLÜM 1 — M1.1, M1.1b, M1.3
# ============================================================
def bolum1_kesif():
    baslik("1. KEŞİF ANALİZİ — M1.1, M1.1b, M1.3")
    if not YOL["kesif"]:
        print("  Atlandı: kesif_kayitlar.json yok."); return None

    from transformers import AutoTokenizer
    kayitlar = json_oku(YOL["kesif"])
    print(f"  Keşif kaydı: {len(kayitlar)} eşleştirilmiş problem")

    alt("M1.1 — Token sayımı (öğretmenin kendi tokenizer'ı, her iki dil için aynı)")
    tok = AutoTokenizer.from_pretrained(OGRETMEN_ADI)
    for k in kayitlar:
        k["en_token"] = len(tok(k["en_cozum"], add_special_tokens=False)["input_ids"])
        k["tr_token"] = len(tok(k["tr_cozum"], add_special_tokens=False)["input_ids"])

    s = {"en_token": ozet_yaz("EN token", [k["en_token"] for k in kayitlar]),
         "tr_token": ozet_yaz("TR token", [k["tr_token"] for k in kayitlar]),
         "en_kelime": ozet_yaz("EN kelime", [k["en_kelime"] for k in kayitlar]),
         "tr_kelime": ozet_yaz("TR kelime", [k["tr_kelime"] for k in kayitlar]),
         "tr_morfem": ozet_yaz("TR morfem", [k["tr_morfem"] for k in kayitlar])}
    o_tok = s["tr_token"]["ortalama"] / s["en_token"]["ortalama"]
    o_kel = s["tr_kelime"]["ortalama"] / s["en_kelime"]["ortalama"]
    o_esk = s["tr_morfem"]["ortalama"] / s["en_kelime"]["ortalama"]
    print(f"\n    TR token / EN token   = {o_tok:.2f}x   <-- hakemin istediği karşılaştırma")
    print(f"    TR kelime / EN kelime = {o_kel:.2f}x")
    print(f"    TR morfem / EN kelime = {o_esk:.2f}x   <-- makalede şu an bu var (farklı birim)")

    print("\n  Doğruluk kırılımı (Türkçe zincirler):")
    dg = [k for k in kayitlar if k["tr_dogru"]]; yn = [k for k in kayitlar if not k["tr_dogru"]]
    for ad, alan in [("token", "tr_token"), ("kelime", "tr_kelime"), ("morfem", "tr_morfem")]:
        sd = ozet_yaz(f"TR {ad} (doğru)", [k[alan] for k in dg])
        sy = ozet_yaz(f"TR {ad} (yanlış)", [k[alan] for k in yn])
        if sd and sy and sd["ortalama"]:
            print(f"    {'':34} yanlış/doğru = {sy['ortalama']/sd['ortalama']:.2f}x")
            s[f"oran_{ad}"] = round(sy["ortalama"] / sd["ortalama"], 3)
    s["oranlar"] = {"tr_token_bolu_en_token": round(o_tok, 3),
                    "tr_kelime_bolu_en_kelime": round(o_kel, 3),
                    "tr_morfem_bolu_en_kelime": round(o_esk, 3)}
    SONUC["M1.1"] = s

    # ---- M1.1b: İngilizce morfem (Morfessor) ----
    alt("M1.1b — EN morfem sayımı (Morfessor Baseline)")
    try:
        import morfessor
        sayac = Counter()
        for k in kayitlar:
            for w in re.findall(r"[A-Za-z]+", k["en_cozum"].lower()):
                sayac[w] += 1
        print(f"    Sözlük: {len(sayac)} farklı sözcük, {sum(sayac.values())} token")
        model = morfessor.BaselineModel()
        model.load_data([(c, w) for w, c in sayac.items()])   # morfessor 2.x: (sayı, sözcük)
        model.train_batch()
        ob = {}
        def en_morfessor(metin):
            t = 0
            for w in re.findall(r"\w+", metin, re.UNICODE):
                if w.isdigit(): t += 1; continue
                wl = w.lower()
                if wl not in ob:
                    try: ob[wl] = len(model.viterbi_segment(wl)[0])
                    except Exception: ob[wl] = 1
                t += ob[wl]
            return t

        # İkinci sayaç: kural tabanlı ek ayrıştırma.
        # Morfessor küçük derlemde AZ böler (EN morfem düşük -> oran yüksek);
        # kural tabanlı sayaç ÇOK böler (EN morfem yüksek -> oran düşük).
        # İkisi birlikte, İngilizce çözümleyici seçiminden doğan ARALIĞI verir.
        _ONEK = ("un", "re", "dis", "pre", "mis", "over", "under", "non", "inter")
        _COK  = ("ization", "ational", "ibility", "ability", "ousness", "fulness")
        _CEKIM = ("ing", "ed", "es", "s")
        _TUREV = ("tion", "sion", "ment", "ness", "ity", "ance", "ence",
                  "able", "ible", "ive", "ous", "ful", "less", "ally", "ly",
                  "ize", "ise", "ify", "al", "ic")
        def _kural(w):
            w = w.lower(); n = 1; kok = w
            for e in _COK:
                if kok.endswith(e) and len(kok) - len(e) >= 4:
                    return n + 2 + (1 if any(kok.startswith(p) for p in _ONEK) else 0)
            for e in _CEKIM:
                if kok.endswith(e) and len(kok) - len(e) >= 4:
                    kok = kok[:-len(e)]; n += 1; break
            for e in _TUREV:
                if kok.endswith(e) and len(kok) - len(e) >= 4:
                    kok = kok[:-len(e)]; n += 1; break
            if any(kok.startswith(p) for p in _ONEK) and len(kok) >= 6:
                n += 1
            return n
        def en_kural(metin):
            return sum(1 if w.isdigit() else _kural(w)
                       for w in re.findall(r"\w+", metin, re.UNICODE))

        for k in kayitlar:
            k["en_morfem"] = en_morfessor(k["en_cozum"])
            k["en_morfem_kural"] = en_kural(k["en_cozum"])
        sm = {"en_morfem_morfessor": ozet_yaz("EN morfem (Morfessor)",
                                              [k["en_morfem"] for k in kayitlar]),
              "en_morfem_kural": ozet_yaz("EN morfem (kural tabanlı)",
                                          [k["en_morfem_kural"] for k in kayitlar]),
              "tr_morfem": s["tr_morfem"]}
        if not sm["en_morfem_morfessor"].get("ortalama"):
            raise ValueError("EN morfem sayımı boş döndü")
        trm = s["tr_morfem"]["ortalama"]
        o_mf = trm / sm["en_morfem_morfessor"]["ortalama"]
        o_kr = trm / sm["en_morfem_kural"]["ortalama"]
        tr_y = ist.mean([k["tr_morfem"]/k["tr_kelime"] for k in kayitlar if k["tr_kelime"]])
        en_y = ist.mean([k["en_morfem"]/k["en_kelime"] for k in kayitlar if k["en_kelime"]])
        en_y2 = ist.mean([k["en_morfem_kural"]/k["en_kelime"] for k in kayitlar if k["en_kelime"]])
        print(f"\n    TR morfem / EN morfem (Morfessor)      = {o_mf:.2f}x   (üst sınır)")
        print(f"    TR morfem / EN morfem (kural tabanlı)  = {o_kr:.2f}x   (alt sınır)")
        print(f"    --> SAVUNULABİLİR ARALIK: {min(o_mf,o_kr):.2f}x - {max(o_mf,o_kr):.2f}x")
        print(f"\n    Sözcük başına morfem: TR {tr_y:.2f} | "
              f"EN {en_y:.2f} (Morfessor) / {en_y2:.2f} (kural)")
        print("\n    NOT: Morfessor bu küçük derlemde AZ böler (running, calculation"
              " bölünmüyor),")
        print("    kural tabanlı sayaç ise ÇOK böler. Gerçek değer aralığın içindedir.")
        print("    Makalede tek sayı yerine bu aralık verilmeli; en sağlam dayanak ise")
        print("    İngilizce çözümleyiciden bağımsız olan token karşılaştırmasıdır.")
        sm["oranlar"] = {"tr_bolu_en_morfessor": round(o_mf, 3),
                         "tr_bolu_en_kural": round(o_kr, 3),
                         "aralik": [round(min(o_mf, o_kr), 3), round(max(o_mf, o_kr), 3)],
                         "tr_yogunluk": round(tr_y, 3),
                         "en_yogunluk_morfessor": round(en_y, 3),
                         "en_yogunluk_kural": round(en_y2, 3)}
        SONUC["M1.1b"] = sm
    except Exception as e:
        print(f"    Morfessor adımı atlandı: {e}")

    # ---- M1.3: token sınırına dayanan zincirler ----
    alt(f"M1.3 — Keşif koşusunda token sınırı = {KESIF_TOKEN_SINIRI}")
    esik = KESIF_TOKEN_SINIRI - 8
    tr_tak = [k for k in kayitlar if k["tr_token"] >= esik]
    en_tak = [k for k in kayitlar if k["en_token"] >= esik]
    n = len(kayitlar)
    print(f"    TR sınıra dayanan: {len(tr_tak)}/{n} ({len(tr_tak)/n:.1%})")
    print(f"    EN sınıra dayanan: {len(en_tak)}/{n} ({len(en_tak)/n:.1%})")
    tam = [k for k in kayitlar if k["tr_token"] < esik]
    bilgi = {"token_siniri": KESIF_TOKEN_SINIRI, "tr_sinira_dayanan": len(tr_tak),
             "en_sinira_dayanan": len(en_tak), "toplam": n, "tamamlanan": len(tam)}
    if tam:
        d = [k for k in tam if k["tr_dogru"]]; y = [k for k in tam if not k["tr_dogru"]]
        print("\n    SADECE TAMAMLANAN zincirler (sınıra dayanmayanlar):")
        sd = ozet_yaz("TR kelime (doğru)", [k["tr_kelime"] for k in d])
        sy = ozet_yaz("TR kelime (yanlış)", [k["tr_kelime"] for k in y])
        if sd and sy and sd["ortalama"]:
            r = sy["ortalama"] / sd["ortalama"]
            print(f"    Tamamlananlarda yanlış/doğru uzunluk oranı = {r:.2f}x")
            print("    (Tüm zincirlerdeki oranla aynıysa, uzunluk farkı kesilmeden")
            print("     kaynaklanmıyor demektir — M1.3'ün can alıcı noktası.)")
            bilgi["tamamlanan_oran"] = round(r, 3)
    SONUC["M1.3"] = bilgi

    kaydet("kesif_zenginlestirilmis.json", kayitlar)
    return kayitlar


# ============================================================
#  BÖLÜM 2 — M1.2, R3-1
# ============================================================
def bolum2_uzunluk(kayitlar):
    baslik("2. UZUNLUK KONTROLLÜ MORFOLOJİ ANALİZİ — M1.2, R3-1")
    if not kayitlar:
        print("  Atlandı: keşif verisi yok."); return
    import pandas as pd, numpy as np
    import statsmodels.api as sm
    from scipy import stats

    df = pd.DataFrame(kayitlar)
    df = df[df["tr_kelime"] > 0].copy()
    df["yogunluk"] = df["tr_morfem"] / df["tr_kelime"]
    df["dogru"] = df["tr_dogru"].astype(int)
    print(f"  Kayıt: {len(df)} | doğru: {df['dogru'].sum()} | yanlış: {(1-df['dogru']).sum()}")

    alt("Morfem yükü sözcük sayısından ayrışıyor mu?")
    r, p = stats.pearsonr(df["tr_kelime"], df["tr_morfem"])
    print(f"    Pearson r (kelime, morfem) = {r:.4f}  (p={p:.2e})")
    print(f"    R² = {r**2:.4f}  ->  morfem yükünün %{100*r**2:.1f}'i sözcük sayısıyla açıklanıyor")
    d = df[df.dogru == 1]["yogunluk"]; y = df[df.dogru == 0]["yogunluk"]
    t, pt = stats.ttest_ind(d, y, equal_var=False)
    print(f"    Sözcük başına morfem: doğru {d.mean():.3f} | yanlış {y.mean():.3f} | "
          f"t={t:.3f}, p={pt:.4f}")
    SONUC["esdogrusallik"] = {"pearson_r": round(float(r), 4), "r_kare": round(float(r**2), 4),
                              "yogunluk_dogru": round(float(d.mean()), 4),
                              "yogunluk_yanlis": round(float(y.mean()), 4),
                              "p": round(float(pt), 5)}

    alt("M1.2 — Lojistik regresyon (standartlaştırılmış değişkenler)")
    ozet = {}
    for ad, dk in {"Sadece sözcük sayısı": ["tr_kelime"],
                   "Sadece morfem yükü": ["tr_morfem"],
                   "Sözcük + yoğunluk": ["tr_kelime", "yogunluk"]}.items():
        X = df[dk].copy(); X = (X - X.mean()) / X.std(); X = sm.add_constant(X)
        try:
            m = sm.Logit(df["dogru"], X).fit(disp=0)
        except Exception as e:
            print(f"    [{ad}] kurulamadı: {e}"); continue
        print(f"\n    [{ad}]  Pseudo R²={m.prsquared:.4f}  AIC={m.aic:.1f}")
        for v in dk:
            print(f"      {v:12} katsayı={m.params[v]:+.4f}  z={m.tvalues[v]:+.3f}  "
                  f"p={m.pvalues[v]:.4f}")
        ozet[ad] = {"pseudo_r2": round(float(m.prsquared), 4), "aic": round(float(m.aic), 2),
                    "katsayilar": {v: round(float(m.params[v]), 4) for v in dk},
                    "p": {v: round(float(m.pvalues[v]), 5) for v in dk}}
    print("\n    YORUM: 'Sözcük + yoğunluk' modelinde yoğunluk p'si anlamsızsa,")
    print("    morfolojik yoğunluk uzunluğun ötesinde bilgi taşımıyor demektir.")
    SONUC["M1.2_lojistik"] = ozet

    alt("M1.2 — Uzunluk eşleştirilmiş dilimler")
    try:
        df["grup"] = pd.qcut(df["tr_kelime"], 5, labels=False, duplicates="drop")
    except ValueError:
        print("    Dilimleme yapılamadı."); df["grup"] = 0
    satir = []
    print(f"    {'dilim':6} {'aralık':>14} {'n':>5} {'doğruluk':>9} "
          f"{'yoğ.doğru':>11} {'yoğ.yanlış':>11} {'p':>8}")
    print("    " + "-" * 70)
    for g, a in df.groupby("grup"):
        dd = a[a.dogru == 1]["yogunluk"]; yy = a[a.dogru == 0]["yogunluk"]
        pp = float("nan")
        if len(dd) >= 2 and len(yy) >= 2:
            pp = stats.ttest_ind(dd, yy, equal_var=False).pvalue
        ar = f"{a['tr_kelime'].min():.0f}-{a['tr_kelime'].max():.0f}"
        print(f"    {int(g):<6} {ar:>14} {len(a):>5} {a['dogru'].mean():>8.1%} "
              f"{(dd.mean() if len(dd) else float('nan')):>11.3f} "
              f"{(yy.mean() if len(yy) else float('nan')):>11.3f} {pp:>8.4f}")
        satir.append({"dilim": int(g), "aralik": ar, "n": len(a),
                      "dogruluk": round(float(a["dogru"].mean()), 4),
                      "p": round(float(pp), 5) if pp == pp else None})
    SONUC["M1.2_dilimler"] = satir

    alt("R3-1 — İngilizce karşılaştırmanın künyesi")
    kunye = {
        "veri_kaynagi": "GSM8K (EN) ve malhajar/gsm8k-tr (TR), test bölümü",
        "hizalama": "Nihai sayısal cevaba göre eşleştirme; yalnızca cevabı benzersiz "
                    "olan problemler alındı (indeks hizalaması güvenilir değil)",
        "eslesen_problem": int(len(df)),
        "icerik": "Paralel içerik — her iki dilde AYNI problem",
        "zincir_ureten_model": "DeepSeek-R1-Distill-Qwen-7B, iki dil için de aynı model",
        "cozumleme": f"do_sample=False (greedy), max_new_tokens={KESIF_TOKEN_SINIRI}",
        "tr_cozumleyici": "Zeyrek (Zemberek'in Python uyarlaması)",
        "en_cozumleyici": "Morfessor Baseline, bu derlem üzerinde denetimsiz eğitildi",
    }
    for k, v in kunye.items():
        print(f"    {k:22}: {v}")
    SONUC["R3-1_kunye"] = kunye


# ============================================================
#  BÖLÜM 3 — M6.1, M3.3, M7.3, M7.2
# ============================================================
def bolum3_veri():
    baslik("3. EĞİTİM VERİSİ VE SEÇİM YANLILIĞI — M6.1, M3.3, M7.2, M7.3")
    if not YOL["zincirler"]:
        print("  Atlandı: asama1_ogretmen_zincirleri.jsonl yok."); return None

    zincirler = jsonl_oku(YOL["zincirler"])
    n = len(zincirler)
    n_d = sum(1 for k in zincirler if k["dogru"])
    n_t = sum(1 for k in zincirler if k["turkce"])
    n_i = sum(1 for k in zincirler if k["dogru"] and k["turkce"])
    alt("M6.1 / M7.3 — Filtreleme hunisi")
    print(f"    Öğretmenin ürettiği zincir            : {n}")
    print(f"    Doğru cevaplı                         : {n_d} ({n_d/n:.1%})")
    print(f"    Türkçe filtresinden geçen             : {n_t} ({n_t/n:.1%})   <-- M7.3")
    print(f"    Her iki filtreden geçen (eğitim seti) : {n_i} ({n_i/n:.1%})")
    SONUC["M7.3"] = {"turkce_gecen": n_t, "toplam": n, "oran": round(n_t/n, 4)}

    bilgi = {"uretilen": n, "dogru": n_d, "turkce": n_t, "egitim_seti": n_i}

    if YOL["sade"]:
        alt("M6.1 / M3.3 — Sürüm başına token uzunluğu ve 1024'te kesilme")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(OGRENCI_ADI)
        sade = jsonl_oku(YOL["sade"])
        print(f"    Sadeleştirmeye giren zincir: {len(sade)}")
        surum = {}
        for alan, etiket in [("orijinal", "Orijinal"),
                             ("sade_morfem", "Morfem-sadeleştirilmiş"),
                             ("sade_kelime", "Sözcük-sadeleştirilmiş")]:
            uz = [len(tok(k[alan], add_special_tokens=False)["input_ids"]) for k in sade]
            kes = sum(1 for u in uz if u >= EGITIM_TOKEN_SINIRI)
            d = ozet_yaz(etiket, uz)
            print(f"    {'':34} 1024'te kesilen: {kes}/{len(uz)} ({kes/len(uz):.1%})")
            d["kesilen"] = kes
            surum[alan] = d
        bilgi["surumler"] = surum
        bilgi["sadelestirilen"] = len(sade)
        SONUC["M3.3"] = {a: surum[a]["kesilen"] for a in surum}
    SONUC["M6.1"] = bilgi

    # ---- M7.2 zorluk dağılımı ----
    alt("M7.2 — Elde tutulan ve elenen problemlerin zorluk dağılımı")
    referans = referans_sozlugu()
    if referans:
        def adim_say(c):
            a = len(re.findall(r"<<", c))
            return a if a else len([s for s in c.split("\n") if s.strip()])
        tut, ele = [], []
        for k in zincirler:
            c = referans.get(k["soru"])
            if not c: continue
            (tut if (k["dogru"] and k["turkce"]) else ele).append(adim_say(c))
        print(f"    Eşleşen problem: {len(tut)+len(ele)} / {len(zincirler)}")
        st = ozet_yaz("Elde tutulan (adım sayısı)", tut)
        se = ozet_yaz("Elenen (adım sayısı)", ele)
        dt, de = Counter(tut), Counter(ele)
        print("\n    Adım sayısı dağılımı:")
        for a in sorted(set(dt) | set(de)):
            print(f"      {a} adım: tutulan {dt[a]:3} ({dt[a]/max(1,len(tut)):5.1%})   "
                  f"elenen {de[a]:3} ({de[a]/max(1,len(ele)):5.1%})")
        z = {"tutulan": st, "elenen": se,
             "tutulan_dagilim": dict(dt), "elenen_dagilim": dict(de)}
        if tut and ele:
            try:
                from scipy.stats import mannwhitneyu
                u, p = mannwhitneyu(tut, ele, alternative="two-sided")
                print(f"\n    Mann-Whitney U={u:.0f}, p={p:.4f}")
                print("    (p<0.05 ise filtreleme eğitim setini kolay problemlere kaydırıyor.)")
                z["mannwhitney_u"] = float(u); z["mannwhitney_p"] = round(float(p), 5)
            except Exception:
                pass
        SONUC["M7.2"] = z
    return zincirler


# ============================================================
#  BÖLÜM 4 — M2.1, M2.3, M6.2, M6.3, M6.4, M6.6
# ============================================================
def sadelestir_izli(zincir, olcut="morfem", kural1=True, agirlik=True):
    temiz = zincir_temizle(zincir)
    cumleler = cumlelere_bol(temiz)
    if len(cumleler) <= 2:
        return temiz, cumleler, [{"cumle": c, "agirlik": None, "karar": "tut",
                                  "kural": "dokunulmadi"} for c in cumleler]
    ag = ([morfem_toplam(c) for c in cumleler] if olcut == "morfem"
          else [kelime_say(c) for c in cumleler])
    ort = sum(ag) / len(ag)
    tutulan, izler = [], []
    for c, a in zip(cumleler, ag):
        if kural1 and any(benzer_mi(c, t) for t in tutulan):
            izler.append({"cumle": c, "agirlik": a, "karar": "at", "kural": "K1_tekrar"}); continue
        if cekirdek_mi(c):
            tutulan.append(c)
            izler.append({"cumle": c, "agirlik": a, "karar": "tut", "kural": "K2_cekirdek"}); continue
        if agirlik and a > ort:
            izler.append({"cumle": c, "agirlik": a, "karar": "at", "kural": "K3_agir_dolgu"}); continue
        tutulan.append(c)
        izler.append({"cumle": c, "agirlik": a, "karar": "tut", "kural": "K4_hafif"})
    return temiz, tutulan, izler


def bolum4_ablasyon(zincirler):
    baslik("4. KURAL ABLASYONU VE ÖLÇÜT UYUMU — M2.1, M2.3, M6.2-M6.6")
    if not zincirler:
        print("  Atlandı."); return None
    from tqdm.auto import tqdm

    kaynak = [k for k in zincirler if k["dogru"] and k["turkce"]]
    print(f"  İşlenecek zincir: {len(kaynak)}")
    print("  ABLASYON TASARIMI: Kural 2 bir KORUMA kuralıdır; ağırlık filtresi kapalıyken")
    print("  etkisi yoktur, yani hakemin istediği 'ağırlık filtresiz Kural 1-2' ile")
    print("  'yalnız Kural 1' yapı gereği aynıdır. İki mekanizmayı gerçekten ayıran")
    print("  üç koşul koşuluyor: yalniz_kural1 / yalniz_agirlik / tam.\n")

    kosullar = {"tam": dict(kural1=True, agirlik=True),
                "yalniz_kural1": dict(kural1=True, agirlik=False),
                "yalniz_agirlik": dict(kural1=False, agirlik=True)}
    cikti, k_sayac, k_yuk = [], {}, {}
    kisa = 0; uyum_a = uyum_t = 0
    kapsama_tanimsiz = kapsama_toplam = 0

    for k in tqdm(kaynak, desc="  Ablasyon", file=sys.__stdout__):
        cevap = k["gercek_cevap"]
        temiz, _, izler = sadelestir_izli(k["ogretmen_zincir"], "morfem", **kosullar["tam"])
        kayit = {"soru": k["soru"], "gercek_cevap": cevap, "orijinal": temiz}
        if len(cumlelere_bol(temiz)) <= 2:
            kisa += 1
        for w in re.findall(r"\w+", temiz, re.UNICODE):
            kapsama_toplam += 1
            if not w.isdigit() and morfem_kelime(w) == 0:
                kapsama_tanimsiz += 1
        for iz in izler:
            if iz["karar"] == "at":
                k_sayac[iz["kural"]] = k_sayac.get(iz["kural"], 0) + 1
                k_yuk[iz["kural"]] = k_yuk.get(iz["kural"], 0) + morfem_toplam(iz["cumle"])
        for ad, ayar in kosullar.items():
            _, tutulan, _ = sadelestir_izli(k["ogretmen_zincir"], "morfem", **ayar)
            m = birlestir(tutulan)
            if cevap not in m: m = temiz
            kayit[f"sade_{ad}"] = m
            kayit[f"{ad}_morfem"] = morfem_toplam(m)
            kayit[f"{ad}_kelime"] = kelime_say(m)
        kayit["orij_morfem"] = morfem_toplam(temiz)
        kayit["orij_kelime"] = kelime_say(temiz)
        _, _, iz_k = sadelestir_izli(k["ogretmen_zincir"], "kelime", **kosullar["tam"])
        for a, b in zip(izler, iz_k):
            uyum_t += 1
            if a["karar"] == b["karar"]: uyum_a += 1
        cikti.append(kayit)

    alt("M2.1 — Kural düzeyinde ablasyon")
    t_say = sum(k_sayac.values()) or 1
    t_yuk = sum(k_yuk.values()) or 1
    print(f"    {'Kural':18} {'silinen cümle':>14} {'pay':>7} {'silinen morfem':>16} {'pay':>7}")
    print("    " + "-" * 66)
    for kural in sorted(k_sayac, key=lambda x: -k_yuk.get(x, 0)):
        c, y = k_sayac[kural], k_yuk.get(kural, 0)
        print(f"    {kural:18} {c:>14} {c/t_say:>7.1%} {y:>16} {y/t_yuk:>7.1%}")
    print(f"    {'TOPLAM':18} {t_say:>14} {'100.0%':>7} {t_yuk:>16} {'100.0%':>7}")

    def ort(a): return ist.mean([x[a] for x in cikti])
    oM = ort("orij_morfem")
    print(f"\n    Ortalama morfem yükü ve azalma:")
    print(f"      {'orijinal':18} {oM:8.1f}")
    azalma = {}
    for ad in kosullar:
        v = ort(f"{ad}_morfem"); azalma[ad] = 1 - v/oM
        print(f"      {ad:18} {v:8.1f}   azalma {100*azalma[ad]:5.1f}%")
    if azalma["tam"]:
        print(f"\n    Kural 1'in tek başına açıkladığı pay: "
              f"{100*azalma['yalniz_kural1']/azalma['tam']:.1f}% of toplam azalma")
        print("    (Bu oran yüksekse kazanç ağırlıklandırma ölçütünden bağımsızdır —")
        print("     hakemin M2.0'daki temel itirazı.)")

    alt("M2.3 — Ölçüt uyumu (cümle düzeyinde tut/at kararları)")
    print(f"    Aynı karar: {uyum_a}/{uyum_t} = {uyum_a/max(1,uyum_t):.1%}")

    alt("M6.2 / M6.3 / M6.4 / M6.6 — Yöntem ayrıntıları")
    kapsama = 1 - (kapsama_tanimsiz / max(1, kapsama_toplam))
    print(f"    M6.2 Kural 1 benzerlik  : difflib.SequenceMatcher, eşik {BENZERLIK_ESIGI}")
    print(f"    M6.3 Kural 2 hesaplama  : regex  {ISLEM.pattern}")
    print(f"    M6.4 Zeyrek seçim       : analyze(k)[0][0] — ilk çözümlemenin ilk ayrıştırması")
    print(f"         Sayılar            : rakamdan ibaret belirteçler 1 morfem")
    print(f"         Noktalama          : \\w+ dışında, sayılmaz")
    print(f"         Tanınmayan sözcük  : {kapsama_tanimsiz}/{kapsama_toplam} "
          f"({1-kapsama:.2%}) — 0 morfem")
    print(f"         Derlem kapsaması   : {kapsama:.2%}")
    print(f"    M6.6 Cümle bölücü       : regex  (?<!\\d)[.!?\\n]+(?!\\d)")
    print(f"         ≤2 cümleli zincir  : {kisa}/{len(cikti)} — dokunulmadan bırakıldı")

    SONUC["M2.1"] = {"kural_silinen_cumle": k_sayac, "kural_silinen_morfem": k_yuk,
                     "ortalama_morfem": {"orijinal": round(oM, 2),
                                         **{a: round(ort(f"{a}_morfem"), 2) for a in kosullar}},
                     "azalma_orani": {a: round(azalma[a], 4) for a in kosullar}}
    SONUC["M2.3"] = {"ayni_karar": uyum_a, "toplam": uyum_t,
                     "uyum_orani": round(uyum_a/max(1, uyum_t), 4)}
    SONUC["M6_ayrintilar"] = {"benzerlik_esigi": BENZERLIK_ESIGI,
                              "hesaplama_regex": ISLEM.pattern,
                              "zeyrek_secim": "analyze(k)[0][0]",
                              "taninmayan": kapsama_tanimsiz, "toplam_kelime": kapsama_toplam,
                              "kapsama": round(kapsama, 4),
                              "cumle_bolucu": r"(?<!\d)[.!?\n]+(?!\d)",
                              "kisa_zincir": kisa, "toplam_zincir": len(cikti)}

    yol = os.path.join(CIKTI_DIR, "ablasyon_surumleri.jsonl")
    with open(yol, "w", encoding="utf-8") as f:
        for k in cikti:
            f.write(json.dumps(k, ensure_ascii=False) + "\n")
    print(f"\n    Eğitim sürümleri yazıldı: {yol}")
    return cikti


# ============================================================
#  BÖLÜM 5 — M2.2, M2.4, R3-6
# ============================================================
def bolum5_temel(ablasyon):
    baslik("5. TEMEL YÖNTEM SÜRÜMLERİ — M2.2, M2.4, R3-6")
    if not ablasyon:
        print("  Atlandı."); return
    from tqdm.auto import tqdm

    referans = referans_sozlugu()

    def rastgele_sil(cumleler, hedef):
        if len(cumleler) <= hedef: return list(cumleler)
        cek = [i for i, c in enumerate(cumleler) if cekirdek_mi(c)]
        dig = [i for i, c in enumerate(cumleler) if not cekirdek_mi(c)]
        kal = set(cek); yer = max(0, hedef - len(cek))
        if yer and dig: kal |= set(random.sample(dig, min(yer, len(dig))))
        return [c for i, c in enumerate(cumleler) if i in kal]

    def duzgun_kes(cumleler, hedef):
        n = len(cumleler)
        if n <= hedef or hedef < 1: return list(cumleler)
        adim = n / hedef
        idx = sorted({min(n-1, int(i*adim)) for i in range(hedef)} | {n-1})
        return [cumleler[i] for i in idx]

    def ref_temizle(c):
        c = re.sub(r"<<[^>]*>>", "", c).split("####")[0]
        return re.sub(r"\s+", " ", c).strip()

    cikti = []; g_rast = g_kes = 0; r_eksik = r_cevapsiz = 0
    for k in tqdm(ablasyon, desc="  Temel yöntemler", file=sys.__stdout__):
        temiz = k["orijinal"]; cevap = k["gercek_cevap"]
        cumleler = cumlelere_bol(temiz)
        hedef = max(1, len(cumlelere_bol(k["sade_tam"])))
        kayit = {"soru": k["soru"], "gercek_cevap": cevap,
                 "orijinal": temiz, "sade_tam": k["sade_tam"]}
        m = birlestir(rastgele_sil(cumleler, hedef))
        if cevap not in m: m = temiz; g_rast += 1
        kayit["sade_rastgele"] = m
        m = birlestir(duzgun_kes(cumleler, hedef))
        if cevap not in m: m = temiz; g_kes += 1
        kayit["sade_kesme"] = m
        rk = referans.get(k["soru"])
        if not rk:
            r_eksik += 1; kayit["sade_referans"] = temiz
        else:
            rc = ref_temizle(rk)
            if cevap not in rc:
                rc += f" Cevap: {cevap}"; r_cevapsiz += 1
            kayit["sade_referans"] = rc
        for a in ["orijinal", "sade_tam", "sade_rastgele", "sade_kesme", "sade_referans"]:
            kayit[f"{a}_kelime"] = kelime_say(kayit[a])
            kayit[f"{a}_morfem"] = morfem_toplam(kayit[a])
        cikti.append(kayit)

    def ort(a, m): return ist.mean([k[f"{a}_{m}"] for k in cikti])
    oM = ort("orijinal", "morfem")
    alt("Sürüm karşılaştırması")
    print(f"    {'Sürüm':20} {'kelime':>8} {'morfem':>8} {'morfem azalma':>15}")
    print("    " + "-" * 56)
    kayit_ozet = {}
    for a, e in [("orijinal", "Orijinal"), ("sade_tam", "Önerilen (tam)"),
                 ("sade_rastgele", "Rastgele silme"), ("sade_kesme", "Düzgün kesme"),
                 ("sade_referans", "gsm8k-tr referans")]:
        print(f"    {e:20} {ort(a,'kelime'):>8.1f} {ort(a,'morfem'):>8.1f} "
              f"{100*(1-ort(a,'morfem')/oM):>14.1f}%")
        kayit_ozet[a] = {"kelime": round(ort(a, "kelime"), 2),
                         "morfem": round(ort(a, "morfem"), 2),
                         "azalma": round(1-ort(a, "morfem")/oM, 4)}
    print(f"\n    Güvenlik kontrolü geri alma: rastgele {g_rast}/{len(cikti)} "
          f"({g_rast/len(cikti):.1%}), kesme {g_kes}/{len(cikti)} ({g_kes/len(cikti):.1%})")
    print(f"    Referans bulunamayan: {r_eksik} | cevabı eklenen: {r_cevapsiz}")
    print("\n    NOT: Rastgele ve kesme sürümleri önerilen yöntemle AYNI cümle sayısına")
    print("    eşlenmiştir. Eğitim sonuçları yakınsa kazanç 'hangi cümlenin atıldığından'")
    print("    değil 'kaç cümlenin atıldığından' geliyor demektir (M2.2).")

    SONUC["temel_yontemler"] = {"tohum": TOHUM, "n": len(cikti), "ozet": kayit_ozet,
                                "guvenlik_geri_alma": {"rastgele": g_rast, "kesme": g_kes},
                                "referans_eksik": r_eksik, "referans_cevapsiz": r_cevapsiz}
    yol = os.path.join(CIKTI_DIR, "temel_surumleri.jsonl")
    with open(yol, "w", encoding="utf-8") as f:
        for k in cikti:
            f.write(json.dumps(k, ensure_ascii=False) + "\n")
    print(f"\n    Eğitim sürümleri yazıldı: {yol}")


# ============================================================
#  BÖLÜM 6 — M4.1-M4.4, M3.2, R3-2, R3-3
# ============================================================
def bolum6_istatistik():
    baslik("6. EŞLEŞTİRİLMİŞ İSTATİSTİKSEL TESTLER — M3.2, M4.1-M4.4, R3-2, R3-3")
    if not DEG_DIR:
        print("  Atlandı: degerlendirme/ klasörü bulunamadı."); return
    import numpy as np, pandas as pd
    from statsmodels.stats.contingency_tables import mcnemar
    rng = np.random.default_rng(TOHUM)

    def deg_bul(klasor, m, v, tohum=None):
        if not klasor:
            return None
        if tohum is None:
            adaylar = [y for y in glob.glob(os.path.join(klasor, f"{m}_{v}*.jsonl"))
                       if not re.search(r"_s\d+\.jsonl$", y)]
        else:
            adaylar = glob.glob(os.path.join(klasor, f"{m}_{v}*_s{tohum}.jsonl"))
        return sorted(adaylar, key=lambda y: -os.path.getsize(y))[0] if adaylar else None

    print(f"  degerlendirme klasöründeki dosyalar:")
    for d in sorted(os.listdir(DEG_DIR))[:20]:
        print(f"    {d}")

    veri = {}
    for m in MODELLER:
        for v in VERSIYONLAR:
            p = deg_bul(DEG_DIR, m, v)
            if p: veri[(m, v, 0)] = jsonl_oku(p)
            for s in (1, 2):
                p = deg_bul(TOHUM_DIR, m, v, s) or deg_bul(DEG_DIR, m, v, s)
                if p: veri[(m, v, s)] = jsonl_oku(p)
    print(f"  Yüklenen koşu: {len(veri)}")
    for a in sorted(veri):
        print(f"    {a[0]:8} {a[1]:13} tohum {a[2]}  n={len(veri[a])}")
    if not veri:
        print("  Dosya yok."); return

    def mcnemar_t(a, b, alan):
        n = min(len(a), len(b))
        x = [bool(a[i][alan]) for i in range(n)]; y = [bool(b[i][alan]) for i in range(n)]
        n01 = sum(1 for i in range(n) if x[i] and not y[i])
        n10 = sum(1 for i in range(n) if not x[i] and y[i])
        t = [[sum(1 for i in range(n) if x[i] and y[i]), n01],
             [n10, sum(1 for i in range(n) if not x[i] and not y[i])]]
        r = mcnemar(t, exact=True)
        return {"n": n, "sadece_A": n01, "sadece_B": n10, "p": float(r.pvalue)}

    def bootstrap(a, b, alan):
        n = min(len(a), len(b))
        x = np.array([float(bool(a[i][alan])) for i in range(n)])
        y = np.array([float(bool(b[i][alan])) for i in range(n)])
        idx = rng.integers(0, n, size=(BOOTSTRAP_N, n))
        f = x[idx].mean(axis=1) - y[idx].mean(axis=1)
        alt_, ust = np.percentile(f, [2.5, 97.5])
        return {"fark": round(float(x.mean()-y.mean()), 4),
                "ga_alt": round(float(alt_), 4), "ga_ust": round(float(ust), 4),
                "marj_icinde": bool(alt_ > -DENKLIK_MARJI and ust < DENKLIK_MARJI)}

    alt("M4.1 / M4.2 / M4.3 — Eşleştirilmiş testler (ana deney)")
    satirlar = []
    for m in MODELLER:
        mevcut = [v for v in VERSIYONLAR if (m, v, 0) in veri]
        if len(mevcut) < 2: continue
        print(f"\n    [{m}]")
        print(f"      {'karşılaştırma':30} {'ölçüt':9} {'fark':>8} {'%95 GA':>18} "
              f"{'McNemar p':>11} {'denk?':>7}")
        print("      " + "-" * 88)
        for v1, v2 in itertools.combinations(mevcut, 2):
            a, b = veri[(m, v1, 0)], veri[(m, v2, 0)]
            for alan, et in [("dogru", "doğruluk"), ("kesildi", "tükenme")]:
                mc = mcnemar_t(a, b, alan); bs = bootstrap(a, b, alan)
                dk = ("evet" if bs["marj_icinde"] else "hayır") if alan == "dogru" else ""
                print(f"      {v1+' - '+v2:30} {et:9} {bs['fark']:>+8.3f} "
                      f"[{bs['ga_alt']:>+6.3f},{bs['ga_ust']:>+6.3f}] {mc['p']:>11.4f} {dk:>7}")
                satirlar.append({"model": m, "v1": v1, "v2": v2, "olcut": alan, **mc, **bs})
    print(f"\n    Denklik marjı ±{DENKLIK_MARJI*100:.0f} puan. GA marjın içindeyse 'pratikte denk',")
    print("    değilse yalnızca 'fark saptanmadı' denebilir (M4.3).")
    SONUC["M4_eslesmis"] = satirlar

    alt("M3.2 — Tükenen ve tükenmeyen çıktılarda doğruluk")
    print(f"    {'model':8} {'versiyon':13} {'tükenen n':>10} {'tük.doğruluk':>13} "
          f"{'tükenmeyen n':>13} {'tükenmeyen doğruluk':>20}")
    print("    " + "-" * 82)
    m32 = []
    for m in MODELLER:
        for v in VERSIYONLAR:
            k = veri.get((m, v, 0))
            if not k: continue
            t = [x for x in k if x["kesildi"]]; nn = [x for x in k if not x["kesildi"]]
            dt = sum(x["dogru"] for x in t)/len(t) if t else float("nan")
            dn = sum(x["dogru"] for x in nn)/len(nn) if nn else float("nan")
            print(f"    {m:8} {v:13} {len(t):>10} {dt:>12.1%} {len(nn):>13} {dn:>19.1%}")
            m32.append({"model": m, "versiyon": v, "tukenen_n": len(t),
                        "tukenen_dogruluk": round(dt, 4) if t else None,
                        "tukenmeyen_n": len(nn),
                        "tukenmeyen_dogruluk": round(dn, 4) if nn else None})
    print("\n    Tükenen çıktılarda doğruluk sıfıra yakın değilse, o doğrular büyük")
    print("    olasılıkla 'son sayı' yedek kuralından geliyor. Tam sayım için yeniden")
    print("    değerlendirme gerekir (rev_F).")
    SONUC["M3.2"] = m32

    alt("M4.4 / R3-3 — Tek tek tohum değerleri ve varyans")
    kay = []
    for (m, v, s), k in veri.items():
        n = len(k)
        kay.append({"model": m, "versiyon": v, "tohum": s, "n": n,
                    "dogruluk": sum(x["dogru"] for x in k)/n,
                    "kesilme": sum(x["kesildi"] for x in k)/n,
                    "morfem": sum(x["morfem"] for x in k)/n})
    df = pd.DataFrame(kay)
    print(f"    {'model':8} {'versiyon':13} {'tohum':>6} {'n':>5} {'doğruluk':>9} "
          f"{'tükenme':>9} {'morfem':>9}")
    print("    " + "-" * 64)
    for _, r in df.sort_values(["model", "versiyon", "tohum"]).iterrows():
        print(f"    {r['model']:8} {r['versiyon']:13} {int(r['tohum']):>6} {int(r['n']):>5} "
              f"{r['dogruluk']:>8.1%} {r['kesilme']:>8.1%} {r['morfem']:>9.1f}")
    coklu = df.groupby(["model", "versiyon"]).filter(lambda g: len(g) > 1)
    if not coklu.empty:
        print("\n    Varyans ayrıştırması (çok tohumlu koşullar):")
        var = {}
        for o in ["dogruluk", "kesilme", "morfem"]:
            g = coklu.groupby(["model", "versiyon"])[o]
            ic = g.var(ddof=1).mean(); ar = g.mean().var(ddof=1)
            pay = ic/(ic+ar) if (ic+ar) else float("nan")
            print(f"      {o:10} tohum-içi={ic:.6f}  koşullar-arası={ar:.6f}  "
                  f"tohum payı={pay:.1%}")
            var[o] = {"tohum_ici": float(ic), "kosullar_arasi": float(ar),
                      "tohum_payi": float(pay)}
        SONUC["R3-3_varyans"] = var
    SONUC["M4.4_tohumlar"] = df.to_dict("records")

    alt("R3-2 — Tam ANOVA")
    try:
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
        for o in ["dogruluk", "kesilme", "morfem"]:
            print(f"\n    [{o}]  model × versiyon")
            try:
                mm = ols(f"{o} ~ C(model) * C(versiyon)", data=df).fit()
                t = sm.stats.anova_lm(mm, typ=2)
            except Exception as e:
                print(f"      kurulamadı: {e}"); continue
            kt_t = t["sum_sq"].sum(); kt_h = t.loc["Residual", "sum_sq"]
            print(f"      {'kaynak':24} {'KT':>10} {'sd':>5} {'F':>9} {'p':>9} "
                  f"{'eta²':>7} {'kısmi eta²':>11}")
            print("      " + "-" * 78)
            kaynaklar = {}
            for kk in t.index:
                kt = t.loc[kk, "sum_sq"]; sd = t.loc[kk, "df"]
                F = t.loc[kk, "F"]; p = t.loc[kk, "PR(>F)"]
                eta = kt/kt_t
                keta = kt/(kt+kt_h) if kk != "Residual" else float("nan")
                print(f"      {kk:24} {kt:>10.4f} {sd:>5.0f} "
                      f"{(F if F == F else 0):>9.3f} {(p if p == p else 1):>9.4f} "
                      f"{eta:>7.3f} {keta:>11.3f}")
                kaynaklar[str(kk)] = {"KT": round(float(kt), 6), "sd": int(sd),
                                      "F": round(float(F), 4) if F == F else None,
                                      "p": round(float(p), 6) if p == p else None,
                                      "eta2": round(float(eta), 4),
                                      "kismi_eta2": round(float(keta), 4) if keta == keta else None}
            print(f"      Model R² = {mm.rsquared:.4f}   gözlem = {len(df)}")
            SONUC[f"R3-2_anova_{o}"] = {"kaynaklar": kaynaklar,
                                        "r_kare": round(float(mm.rsquared), 4),
                                        "gozlem": int(len(df))}
    except ImportError:
        print("    statsmodels yok.")
    print("\n    NOT: Dört modelin tamamı tohum analizine girdiğinde (rev_E/rev_F sonrası)")
    print("    bu bölüm yeniden çalıştırılmalı — R3-4 ancak o zaman tam karşılanır.")


# ============================================================
#  ÇALIŞTIR
# ============================================================
def guvenli_calistir(ad, fn, *a):
    try:
        return fn(*a)
    except Exception:
        print(f"\n  !! {ad} sırasında hata — diğer bölümler devam ediyor:")
        print(traceback.format_exc())
        SONUC.setdefault("hatalar", {})[ad] = traceback.format_exc()
        return None


kesif = guvenli_calistir("Bölüm 1", bolum1_kesif)
guvenli_calistir("Bölüm 2", bolum2_uzunluk, kesif)
zincirler = guvenli_calistir("Bölüm 3", bolum3_veri)
ablasyon = guvenli_calistir("Bölüm 4", bolum4_ablasyon, zincirler)
guvenli_calistir("Bölüm 5", bolum5_temel, ablasyon)
guvenli_calistir("Bölüm 6", bolum6_istatistik)

baslik("TAMAMLANDI")
kaydet("tum_sonuclar.json", SONUC)
print(f"  Çıktı klasörü : {CIKTI_DIR}")
for d in sorted(os.listdir(CIKTI_DIR)):
    p = os.path.join(CIKTI_DIR, d)
    print(f"    {d:36} {os.path.getsize(p)/1024:8.1f} KB")
if "hatalar" in SONUC:
    print(f"\n  !! Hata alan bölümler: {', '.join(SONUC['hatalar'])}")
print("\n  Mevcut dosyaların hiçbiri değiştirilmedi.")

sys.stdout.flush()
sys.stdout = sys.__stdout__
try:
    arsiv = shutil.make_archive(os.path.join(ANA_DIZIN, CIKTI_KLASOR_ADI),
                                "zip", root_dir=CIKTI_DIR)
    print(f"\nArşiv hazır (bunu paylaşabilirsiniz): {arsiv}")
except Exception as e:
    print(f"\nArşiv oluşturulamadı: {e} — klasörü doğrudan paylaşabilirsiniz.")
print(f"Çalışma kaydı: {KAYIT_YOLU}")
