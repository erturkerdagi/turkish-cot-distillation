# ============================================================
#  KEŞİF AŞAMASI — COLAB TEK HÜCRE (v4)
#  Yenilikler:
#   - Zeyrek WARNING logları susturuldu
#   - tqdm progress bar
#   - CHECKPOINT: yarıda kalırsa kaldığı yerden devam eder
# ============================================================
# Colab'da GPU: Runtime > Change runtime type > T4 GPU
# ------------------------------------------------------------

import subprocess, sys
for paket in ["zeyrek", "datasets", "nltk", "transformers", "accelerate", "tqdm"]:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", paket])

import logging
import nltk
nltk.download("punkt_tab", quiet=True); nltk.download("punkt", quiet=True)

import re, json, os, torch
import zeyrek
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

# Zeyrek'in gürültülü WARNING loglarını sustur
logging.getLogger("zeyrek").setLevel(logging.ERROR)
logging.getLogger("zeyrek.rulebasedanalyzer").setLevel(logging.ERROR)

# ---------------- AYARLAR ----------------
MODEL_ADI = "Qwen/Qwen2.5-1.5B-Instruct"
N_PROBLEM = 200
MAX_YENI_TOKEN = 1024
CHECKPOINT = "kesif_kayitlar.json"   # ilerleme buraya kaydedilir
# -----------------------------------------

print(f"Model yükleniyor: {MODEL_ADI} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ADI)
model = AutoModelForCausalLM.from_pretrained(MODEL_ADI, torch_dtype=torch.float16, device_map="auto")
print("Model yüklendi.\n")

def cozum_uret(soru, dil):
    if dil == "tr":
        icerik = f"Bu matematik problemini adım adım Türkçe çöz. Sonunda 'Cevap: X' yaz.\n\n{soru}"
    else:
        icerik = f"Solve this math problem step by step in English. End with 'Answer: X'.\n\n{soru}"
    mesajlar = [{"role": "user", "content": icerik}]
    metin = tokenizer.apply_chat_template(mesajlar, tokenize=False, add_generation_prompt=True)
    girdi = tokenizer(metin, return_tensors="pt").to(model.device)
    with torch.no_grad():
        cikti = model.generate(**girdi, max_new_tokens=MAX_YENI_TOKEN, do_sample=False)
    return tokenizer.decode(cikti[0][girdi["input_ids"].shape[1]:], skip_special_tokens=True)

def son_cevap(a): return a.split("####")[-1].strip().replace(",", "").replace(" ", "")

def veri_yukle(n):
    en = load_dataset("openai/gsm8k", "main", split="test")
    try:
        tr = load_dataset("malhajar/gsm8k-tr", "main", split="test")
    except Exception:
        tr = load_dataset("malhajar/gsm8k-tr", "main", split="validation")
    tr_index = {}
    for ex in tr:
        tr_index.setdefault(son_cevap(ex["answer"]), []).append(ex)
    ornekler = []
    for ex_en in en:
        c = son_cevap(ex_en["answer"])
        adaylar = tr_index.get(c, [])
        if len(adaylar) == 1:
            ornekler.append({"en_q": ex_en["question"], "tr_q": adaylar[0]["question"], "cevap": c})
        if len(ornekler) >= n:
            break
    return ornekler

def cevap_ayikla(metin):
    m = re.search(r"(?:cevap|answer)\s*[:=]?\s*\$?(-?\d[\d,\.]*)", metin, re.IGNORECASE)
    if m: return m.group(1).replace(",", "").rstrip(".")
    s = re.findall(r"-?\d[\d,\.]*", metin)
    return s[-1].replace(",", "").rstrip(".") if s else None

def dogru_mu(metin, gercek):
    t = cevap_ayikla(metin)
    return t is not None and t == gercek.strip()

_analyzer = zeyrek.MorphAnalyzer()
def kelime_say(m): return len(re.findall(r"\w+", m, re.UNICODE))
def morfem_yuku(metin):
    toplam = 0
    for k in re.findall(r"\w+", metin, re.UNICODE):
        if k.isdigit(): toplam += 1; continue
        try:
            s = _analyzer.analyze(k)
            if s and s[0]: toplam += len(s[0][0].morphemes)
        except: pass
    return toplam

def ort(l): return sum(l)/len(l) if l else 0.0
def grup_ozet(kayitlar, etiket):
    if not kayitlar:
        print(f"  [{etiket}] grup boş"); return
    en_kel=ort([k["en_kelime"] for k in kayitlar]); tr_kel=ort([k["tr_kelime"] for k in kayitlar]); tr_mor=ort([k["tr_morfem"] for k in kayitlar])
    print(f"  [{etiket}] n={len(kayitlar)}")
    print(f"     EN kelime={en_kel:.1f} | TR kelime={tr_kel:.1f} | TR morfem={tr_mor:.1f}")
    print(f"     TR morfem/EN kelime={tr_mor/max(en_kel,1):.2f}x | TR morfem/TR kelime={tr_mor/max(tr_kel,1):.2f}")

# ---------- CHECKPOINT yardımcıları ----------
def checkpoint_yukle():
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT, encoding="utf-8") as f:
            kayitlar = json.load(f)
        print(f"Checkpoint bulundu: {len(kayitlar)} problem zaten işlenmiş, devam ediliyor.")
        return kayitlar
    return []

def checkpoint_kaydet(kayitlar):
    # Önce geçici dosyaya yaz, sonra taşı (yarıda kesilme bozulmasını önler)
    gecici = CHECKPOINT + ".tmp"
    with open(gecici, "w", encoding="utf-8") as f:
        json.dump(kayitlar, f, ensure_ascii=False, indent=2)
    os.replace(gecici, CHECKPOINT)

def ozet_bas(kayitlar):
    n_o = len(kayitlar)
    if n_o == 0: return
    en_dog = sum(k["en_dogru"] for k in kayitlar)/n_o
    tr_dog = sum(k["tr_dogru"] for k in kayitlar)/n_o
    tr_dogrular  = [k for k in kayitlar if k["tr_dogru"]]
    tr_yanlislar = [k for k in kayitlar if not k["tr_dogru"]]
    print("\n" + "="*60)
    print(f"KEŞİF SONUÇLARI  (toplam {n_o} problem)")
    print("="*60)
    print(f"Doğruluk: EN {en_dog:.0%} | TR {tr_dog:.0%}\n")
    print("Morfem yükü — GRUPLARA GÖRE:")
    grup_ozet(kayitlar,     "TÜM zincirler")
    grup_ozet(tr_dogrular,  "TR DOĞRU zincirler  (ANA İDDİA)")
    grup_ozet(tr_yanlislar, "TR YANLIŞ zincirler")

def kesif_calistir(n=N_PROBLEM):
    veri = veri_yukle(n)
    print(f"Güvenli eşleşen çift: {len(veri)}")
    if not veri:
        print("HATA: eşleşme yok!"); return []

    kayitlar = checkpoint_yukle()
    baslangic = len(kayitlar)   # kaldığı yerden devam

    try:
        for idx in tqdm(range(baslangic, len(veri)), initial=baslangic, total=len(veri), desc="Keşif"):
            ornek = veri[idx]
            en_c = cozum_uret(ornek["en_q"], "en")
            tr_c = cozum_uret(ornek["tr_q"], "tr")
            kayitlar.append({
                "cevap": ornek["cevap"], "en_q": ornek["en_q"], "tr_q": ornek["tr_q"],
                "en_dogru": dogru_mu(en_c, ornek["cevap"]),
                "tr_dogru": dogru_mu(tr_c, ornek["cevap"]),
                "en_kelime": kelime_say(en_c), "tr_kelime": kelime_say(tr_c),
                "tr_morfem": morfem_yuku(tr_c),
                "en_cozum": en_c, "tr_cozum": tr_c,
            })
            checkpoint_kaydet(kayitlar)   # her problemden sonra kaydet
    except KeyboardInterrupt:
        print("\nDurduruldu — ilerleme kaydedildi, tekrar çalıştırınca kaldığı yerden devam eder.")

    ozet_bas(kayitlar)
    print("\nHam kayıtlar:", CHECKPOINT)
    if kayitlar:
        print("\n--- EŞLEŞME KONTROLÜ (ilk problem) ---")
        print("EN:", kayitlar[0]["en_q"][:120])
        print("TR:", kayitlar[0]["tr_q"][:120])
    return kayitlar

kayitlar = kesif_calistir()
