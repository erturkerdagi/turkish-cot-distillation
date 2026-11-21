# ============================================================
#  AŞAMA 1: ÖĞRETMEN ZİNCİRİ ÜRETME (Senaryo B)
#  Öğretmen: DeepSeek-R1-Distill-Qwen-7B (uzun Türkçe muhakeme)
#  - Drive'a kaydeder, checkpoint'li, progress bar, temiz çıktı
#  - Her problem için: uzun zincir + nihai cevap + doğru mu?
# ============================================================
# Colab'da önce:
#   from google.colab import drive
#   drive.mount('/content/drive')
# GPU: Colab Pro (L4/A100 önerilir)
# ------------------------------------------------------------

import subprocess, sys, logging
for paket in ["datasets", "transformers", "accelerate", "bitsandbytes", "tqdm"]:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", paket],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

import os, json, re, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# ---------------- AYARLAR ----------------
DRIVE_DIR = os.environ.get(
    "PROJE_DIZINI",
    "/content/drive/MyDrive/turkish-cot-distillation/Project",
)
OGRETMEN = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
N_PROBLEM = 300          # makul deneme; sonra artır -> kaldığı yerden devam
MAX_YENI_TOKEN = 2048    # R1 uzun düşünür, bol alan ver
CIKTI = os.path.join(DRIVE_DIR, "asama1_ogretmen_zincirleri.jsonl")
# -----------------------------------------

os.makedirs(DRIVE_DIR, exist_ok=True)

print(f"Öğretmen yükleniyor: {OGRETMEN} ...")
tokenizer = AutoTokenizer.from_pretrained(OGRETMEN)
model = AutoModelForCausalLM.from_pretrained(OGRETMEN, torch_dtype=torch.float16, device_map="auto")
print("Öğretmen yüklendi.\n")

def son_cevap(answer_metni):
    return answer_metni.split("####")[-1].strip().replace(",", "").replace(" ", "")

def ogretmen_coz(soru):
    """R1-distill'e Türkçe adım adım çözdürür. Uzun zincir bekleriz."""
    icerik = (f"Aşağıdaki matematik problemini Türkçe düşünerek adım adım çöz. "
              f"Tüm muhakemeni Türkçe yap. Sonunda 'Cevap: X' yaz.\n\n{soru}")
    mesajlar = [{"role": "user", "content": icerik}]
    metin = tokenizer.apply_chat_template(mesajlar, tokenize=False, add_generation_prompt=True)
    girdi = tokenizer(metin, return_tensors="pt").to(model.device)
    with torch.no_grad():
        cikti = model.generate(**girdi, max_new_tokens=MAX_YENI_TOKEN, do_sample=False,
                               pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(cikti[0][girdi["input_ids"].shape[1]:], skip_special_tokens=True)

def cevap_ayikla(metin):
    m = re.search(r"(?:cevap|answer)\s*[:=]?\s*\$?(-?\d[\d,\.]*)", metin, re.IGNORECASE)
    if m: return m.group(1).replace(",", "").rstrip(".")
    s = re.findall(r"-?\d[\d,\.]*", metin)
    return s[-1].replace(",", "").rstrip(".") if s else None

def turkce_mi(metin):
    """Kaba Türkçe kontrolü: Türkçe'ye özgü karakter/kelime oranı."""
    tr_isaret = len(re.findall(r"[çğıöşüÇĞİÖŞÜ]", metin))
    tr_kelime = len(re.findall(r"\b(bir|için|ve|ile|olarak|sonra|göre|kadar|eder|bu)\b", metin, re.IGNORECASE))
    return (tr_isaret + tr_kelime) >= 3   # eşik: en az 3 Türkçe işareti

def kac_islenmis():
    if not os.path.exists(CIKTI): return 0
    with open(CIKTI, encoding="utf-8") as f:
        return sum(1 for _ in f)

def asama1_calistir(n=N_PROBLEM):
    tr = load_dataset("malhajar/gsm8k-tr", "main", split="train")
    hedef = min(n, len(tr))
    baslangic = kac_islenmis()
    if baslangic >= hedef:
        print(f"Zaten {baslangic} işlenmiş, hedef {hedef}. Yeni iş yok."); return
    print(f"Mevcut: {len(tr)} | hedef: {hedef} | işlenmiş: {baslangic}\n")

    with open(CIKTI, "a", encoding="utf-8") as f:
        for idx in tqdm(range(baslangic, hedef), initial=baslangic, total=hedef, desc="Aşama 1 (öğretmen)"):
            ornek = tr[idx]
            gercek = son_cevap(ornek["answer"])
            zincir = ogretmen_coz(ornek["question"])
            tahmin = cevap_ayikla(zincir)
            kayit = {
                "soru": ornek["question"],
                "ogretmen_zincir": zincir,
                "gercek_cevap": gercek,
                "tahmin": tahmin,
                "dogru": (tahmin is not None and tahmin == gercek),
                "turkce": turkce_mi(zincir),
                "uzunluk_kelime": len(re.findall(r"\w+", zincir, re.UNICODE)),
            }
            f.write(json.dumps(kayit, ensure_ascii=False) + "\n")
            f.flush()

    # Özet
    with open(CIKTI, encoding="utf-8") as f:
        kayitlar = [json.loads(l) for l in f]
    n_o = len(kayitlar)
    dogru = sum(k["dogru"] for k in kayitlar)
    turkce = sum(k["turkce"] for k in kayitlar)
    ort_uz = sum(k["uzunluk_kelime"] for k in kayitlar)/n_o
    print(f"\n{'='*50}")
    print(f"AŞAMA 1 ÖZET ({n_o} problem)")
    print(f"{'='*50}")
    print(f"Doğru çözüm    : {dogru}/{n_o} ({dogru/n_o:.0%})")
    print(f"Türkçe zincir  : {turkce}/{n_o} ({turkce/n_o:.0%})")
    print(f"Ort. zincir uzunluğu: {ort_uz:.0f} kelime")
    print(f"Dosya: {CIKTI}")
    # Örnek
    ilk_dogru = next((k for k in kayitlar if k["dogru"]), kayitlar[0])
    print(f"\n--- ÖRNEK ZİNCİR (ilk {min(300,len(ilk_dogru['ogretmen_zincir']))} karakter) ---")
    print(ilk_dogru["ogretmen_zincir"][:300])

asama1_calistir()
