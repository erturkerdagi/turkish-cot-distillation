# ============================================================
#  AŞAMA 4: DEĞERLENDİRME — 9 model × test seti karşılaştırması
#  Her model için: doğruluk, muhakeme uzunluğu (kelime),
#  morfem yükü, thinking-exhaustion (kesilme) ölçülür.
#  - Checkpoint'li (model + problem bazında devam)
#  - Ana sonuç tablosunu üretir (3 model × 3 versiyon)
# ============================================================
# Colab: from google.colab import drive; drive.mount('/content/drive')
# ------------------------------------------------------------

import subprocess, sys, logging
for paket in ["peft","transformers","datasets","bitsandbytes","accelerate","zeyrek","nltk","tqdm"]:
    subprocess.run([sys.executable,"-m","pip","install","-q",paket],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
import nltk
nltk.download("punkt_tab", quiet=True); nltk.download("punkt", quiet=True)

import os, re, json, gc, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm.auto import tqdm
import zeyrek
logging.getLogger("zeyrek").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# ---------------- AYARLAR ----------------
DRIVE_DIR = os.environ.get(
    "PROJE_DIZINI",
    "/content/drive/MyDrive/turkish-cot-distillation/Project",
)
MODELLER_DIR = os.path.join(DRIVE_DIR, "modeller")
SONUC_DIR = os.path.join(DRIVE_DIR, "degerlendirme")
N_TEST = 300          # kaç test problemi (makul; artırılabilir)
MAX_TOKEN = 512

MODELLER = [("qwen05","Qwen/Qwen2.5-0.5B-Instruct"),
            ("qwen15","Qwen/Qwen2.5-1.5B-Instruct"),
            ("qwen3","Qwen/Qwen2.5-3B-Instruct"),
            ("llama3","meta-llama/Llama-3.2-3B-Instruct")]
VERSIYONLAR = ["orijinal","sade_morfem","sade_kelime"]
# -----------------------------------------
os.makedirs(SONUC_DIR, exist_ok=True)

_analyzer = zeyrek.MorphAnalyzer()
def son_cevap(a): return a.split("####")[-1].strip().replace(",","").replace(" ","")
def cevap_ayikla(m):
    x = re.search(r"(?:cevap|answer)\s*[:=]?\s*\$?(-?\d[\d,\.]*)", m, re.IGNORECASE)
    if x: return x.group(1).replace(",","").rstrip(".")
    s = re.findall(r"-?\d[\d,\.]*", m)
    return s[-1].replace(",","").rstrip(".") if s else None
def kelime_say(m): return len(re.findall(r"\w+", m, re.UNICODE))
def morfem_yuku(m):
    t=0
    for k in re.findall(r"\w+", m, re.UNICODE):
        if k.isdigit(): t+=1; continue
        try:
            s=_analyzer.analyze(k)
            if s and s[0]: t+=len(s[0][0].morphemes)
        except: pass
    return t

# Test problemlerini bir kez yükle
tr_test = load_dataset("malhajar/gsm8k-tr","main",split="test")
test_problemleri = [(tr_test[i]["question"], son_cevap(tr_test[i]["answer"])) for i in range(N_TEST)]

def model_degerlendir(model_etiket, temel_adi, versiyon):
    """Bir modeli test setinde değerlendirir, sonuç dosyasına checkpoint'li yazar."""
    ad = f"{model_etiket}_{versiyon}"
    sonuc_dosya = os.path.join(SONUC_DIR, f"{ad}.jsonl")
    adaptor = os.path.join(MODELLER_DIR, ad)

    # checkpoint: kaç problem işlenmiş?
    islenmis = 0
    if os.path.exists(sonuc_dosya):
        with open(sonuc_dosya, encoding="utf-8") as f:
            islenmis = sum(1 for _ in f)
    if islenmis >= N_TEST:
        return sonuc_dosya  # zaten bitti

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(temel_adi)
    temel = AutoModelForCausalLM.from_pretrained(temel_adi, quantization_config=bnb,
                                                 device_map="auto", torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(temel, adaptor); model.eval()

    with open(sonuc_dosya, "a", encoding="utf-8") as f:
        for i in tqdm(range(islenmis, N_TEST), initial=islenmis, total=N_TEST, desc=ad, leave=False):
            soru, gercek = test_problemleri[i]
            icerik = f"Bu matematik problemini adım adım Türkçe çöz. Sonunda 'Cevap: X' yaz.\n\n{soru}"
            msj = [{"role":"user","content":icerik}]
            metin = tok.apply_chat_template(msj, tokenize=False, add_generation_prompt=True)
            girdi = tok(metin, return_tensors="pt").to(model.device)
            with torch.no_grad():
                cikti = model.generate(**girdi, max_new_tokens=MAX_TOKEN, do_sample=False,
                                       pad_token_id=tok.eos_token_id)
            uretilen_ids = cikti[0][girdi["input_ids"].shape[1]:]
            cozum = tok.decode(uretilen_ids, skip_special_tokens=True)
            tahmin = cevap_ayikla(cozum)
            # thinking exhaustion: token limitine dayandıysa (cevap üretemeden kesildi)
            kesildi = len(uretilen_ids) >= MAX_TOKEN - 1
            f.write(json.dumps({
                "dogru": (tahmin==gercek), "kelime": kelime_say(cozum),
                "morfem": morfem_yuku(cozum), "kesildi": kesildi,
            }, ensure_ascii=False)+"\n")
            f.flush()

    del model, temel; gc.collect(); torch.cuda.empty_cache()
    return sonuc_dosya

def ozetle(sonuc_dosya):
    with open(sonuc_dosya, encoding="utf-8") as f:
        kayitlar=[json.loads(l) for l in f]
    n=len(kayitlar)
    return {
        "dogruluk": sum(k["dogru"] for k in kayitlar)/n,
        "kelime": sum(k["kelime"] for k in kayitlar)/n,
        "morfem": sum(k["morfem"] for k in kayitlar)/n,
        "kesilme": sum(k["kesildi"] for k in kayitlar)/n,
    }

def tum_degerlendir():
    ozet = {}
    for me, ma in MODELLER:
        for v in VERSIYONLAR:
            print(f"Değerlendiriliyor: {me} × {v}")
            dosya = model_degerlendir(me, ma, v)
            ozet[(me,v)] = ozetle(dosya)

    # SONUÇ TABLOSU
    print("\n"+"="*78)
    print("ANA SONUÇ TABLOSU (test seti, N={})".format(N_TEST))
    print("="*78)
    print(f"{'Model':8} {'Versiyon':13} {'Doğruluk':>9} {'Kelime':>8} {'Morfem':>8} {'Kesilme':>8}")
    print("-"*78)
    for me,_ in MODELLER:
        for v in VERSIYONLAR:
            o=ozet[(me,v)]
            print(f"{me:8} {v:13} {o['dogruluk']:>8.0%} {o['kelime']:>8.0f} {o['morfem']:>8.0f} {o['kesilme']:>8.0%}")
        print("-"*78)

    # Kaydet
    with open(os.path.join(SONUC_DIR,"ozet_tablo.json"),"w",encoding="utf-8") as f:
        json.dump({f"{k[0]}_{k[1]}":v for k,v in ozet.items()}, f, ensure_ascii=False, indent=2)
    print(f"\nÖzet kaydedildi: {os.path.join(SONUC_DIR,'ozet_tablo.json')}")

tum_degerlendir()
