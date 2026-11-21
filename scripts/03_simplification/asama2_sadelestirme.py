# ============================================================
#  AŞAMA 2: MORFEM-FARKINDA SADELEŞTİRME (özgün katkı)
#  - Öğretmen zincirlerini sadeleştirir (morfem ölçütü = ANA YÖNTEM)
#  - Baseline karşılaştırması (kelime ölçütü = genel kısaltma)
#  - Güvenlik kontrolü: cevap korunmazsa orijinali tut
#  - Drive'a checkpoint'li kaydeder, progress bar, temiz çıktı
# ============================================================
# Colab: from google.colab import drive; drive.mount('/content/drive')
# ------------------------------------------------------------

import subprocess, sys, logging
for paket in ["zeyrek", "nltk", "tqdm"]:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", paket],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
import nltk
nltk.download("punkt_tab", quiet=True); nltk.download("punkt", quiet=True)

import os, re, json
import zeyrek
from difflib import SequenceMatcher
from tqdm.auto import tqdm
logging.getLogger("zeyrek").setLevel(logging.ERROR)

# ---------------- AYARLAR ----------------
DRIVE_DIR = os.environ.get(
    "PROJE_DIZINI",
    "/content/drive/MyDrive/turkish-cot-distillation/Project",
)
GIRDI  = os.path.join(DRIVE_DIR, "asama1_ogretmen_zincirleri.jsonl")
CIKTI  = os.path.join(DRIVE_DIR, "asama2_sadelesmis.jsonl")
# -----------------------------------------

_analyzer = zeyrek.MorphAnalyzer()

def zincir_temizle(m):
    m = re.sub(r"</?think>", " ", m)
    m = re.sub(r"\\boxed\{([^}]*)\}", r"\1", m)
    m = re.sub(r"[*#`]+", " ", m)
    m = re.sub(r"[\u4e00-\u9fff]+", " ", m)
    return re.sub(r"\s+", " ", m).strip()

def cumlelere_bol(m):
    return [p.strip() for p in re.split(r"(?<!\d)[.!?\n]+(?!\d)", m) if p.strip()]

def morfem_yuku(c):
    t = 0
    for k in re.findall(r"\w+", c, re.UNICODE):
        if k.isdigit(): t += 1; continue
        try:
            s = _analyzer.analyze(k)
            if s and s[0]: t += len(s[0][0].morphemes)
        except: pass
    return t

def kelime_say(m): return len(re.findall(r"\w+", m, re.UNICODE))
def morfem_toplam(m): return sum(morfem_yuku(c) for c in cumlelere_bol(m))

ISLEM = re.compile(r"\d\s*[+\-*/=×÷]\s*\d|\d+\s*(?:kez|katı|yüzde|%)")
def cekirdek_mi(c): return bool(ISLEM.search(c))
def benzer_mi(a, b, esik=0.75):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= esik

def sadelestir(zincir, olcut="morfem"):
    temiz = zincir_temizle(zincir)
    cumleler = cumlelere_bol(temiz)
    if len(cumleler) <= 2:
        return temiz, temiz
    if olcut == "morfem":
        agir = [morfem_yuku(c) for c in cumleler]
    else:
        agir = [kelime_say(c) for c in cumleler]
    ort = sum(agir)/len(agir)
    tutulan = []
    for c, a in zip(cumleler, agir):
        if any(benzer_mi(c, t) for t in tutulan): continue   # tekrar
        if cekirdek_mi(c): tutulan.append(c); continue        # hesap -> koru
        if a > ort: continue                                   # ağır dolgu -> at
        tutulan.append(c)                                      # hafif -> koru
    return temiz, ". ".join(tutulan) + "."

def kac_islenmis():
    if not os.path.exists(CIKTI): return 0
    with open(CIKTI, encoding="utf-8") as f:
        return sum(1 for _ in f)

def asama2_calistir():
    # Sadece doğru + Türkçe zincirleri al
    kaynak = []
    with open(GIRDI, encoding="utf-8") as f:
        for line in f:
            k = json.loads(line)
            if k["dogru"] and k["turkce"]:
                kaynak.append(k)
    print(f"Sadeleştirilecek (doğru+Türkçe) zincir: {len(kaynak)}")

    baslangic = kac_islenmis()
    if baslangic >= len(kaynak):
        print(f"Zaten {baslangic} işlenmiş. Yeni iş yok.")
    else:
        with open(CIKTI, "a", encoding="utf-8") as f:
            for idx in tqdm(range(baslangic, len(kaynak)), initial=baslangic,
                            total=len(kaynak), desc="Aşama 2 (sadeleştirme)"):
                k = kaynak[idx]
                temiz, sade_m = sadelestir(k["ogretmen_zincir"], "morfem")
                _, sade_k = sadelestir(k["ogretmen_zincir"], "kelime")
                cevap = k["gercek_cevap"]
                # GÜVENLİK: cevap sadeleşmişte yoksa orijinali koru
                guvenli_m = sade_m if cevap in sade_m else temiz
                guvenli_k = sade_k if cevap in sade_k else temiz
                kayit = {
                    "soru": k["soru"], "gercek_cevap": cevap,
                    "orijinal": temiz,
                    "sade_morfem": guvenli_m,
                    "sade_kelime": guvenli_k,
                    "orij_kelime": kelime_say(temiz), "orij_morfem": morfem_toplam(temiz),
                    "morfem_kelime": kelime_say(guvenli_m), "morfem_morfem": morfem_toplam(guvenli_m),
                    "kelime_kelime": kelime_say(guvenli_k), "kelime_morfem": morfem_toplam(guvenli_k),
                    "morfem_cevap_korundu": cevap in sade_m,
                    "kelime_cevap_korundu": cevap in sade_k,
                }
                f.write(json.dumps(kayit, ensure_ascii=False) + "\n")
                f.flush()

    # Toplu istatistik
    with open(CIKTI, encoding="utf-8") as f:
        kayitlar = [json.loads(l) for l in f]
    n = len(kayitlar)
    def ort(key): return sum(k[key] for k in kayitlar)/n
    print("\n" + "="*58)
    print(f"AŞAMA 2 ÖZET ({n} zincir)")
    print("="*58)
    print(f"Ortalama orijinal    : {ort('orij_kelime'):.0f} kelime, {ort('orij_morfem'):.0f} morfem")
    print(f"\n[MORFEM ölçütü = ANA YÖNTEM]")
    print(f"  Sadeleşmiş         : {ort('morfem_kelime'):.0f} kelime, {ort('morfem_morfem'):.0f} morfem")
    print(f"  Morfem azalması    : {100*(1-ort('morfem_morfem')/ort('orij_morfem')):.0f}%")
    print(f"  Cevap koruma       : {100*sum(k['morfem_cevap_korundu'] for k in kayitlar)/n:.0f}%")
    print(f"\n[KELİME ölçütü = BASELINE]")
    print(f"  Sadeleşmiş         : {ort('kelime_kelime'):.0f} kelime, {ort('kelime_morfem'):.0f} morfem")
    print(f"  Morfem azalması    : {100*(1-ort('kelime_morfem')/ort('orij_morfem')):.0f}%")
    print(f"  Cevap koruma       : {100*sum(k['kelime_cevap_korundu'] for k in kayitlar)/n:.0f}%")
    print(f"\nDosya: {CIKTI}")

asama2_calistir()
