# ============================================================
#  AŞAMA 5: ÇOKLU TOHUM DENEYİ (istatistiksel doğrulama)
#  2 model × 3 versiyon × 3 tohum = 18 eğitim + 18 değerlendirme
#  Mevcut tohum-0 sonuçları KORUNUR; sadece 2 ek tohum eklenir.
#  Test: 200 problem | Çıktı: ortalama ± standart sapma
# ============================================================
# Colab Pro A100: drive.mount + (Llama kullanılmıyor, HF login gerekmez)
# ------------------------------------------------------------

import subprocess, sys, logging
for p in ["peft","trl","bitsandbytes","transformers","datasets","accelerate","zeyrek","nltk","tqdm"]:
    subprocess.run([sys.executable,"-m","pip","install","-q",p],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
import nltk
nltk.download("punkt_tab", quiet=True); nltk.download("punkt", quiet=True)

import os, re, json, gc, random, statistics, torch, numpy as np
from datasets import Dataset, load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, set_seed)
from peft import LoraConfig, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer, SFTConfig
from tqdm.auto import tqdm
import zeyrek
logging.getLogger("zeyrek").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# ---------------- AYARLAR ----------------
DRIVE = os.environ.get(
    "PROJE_DIZINI",
    "/content/drive/MyDrive/turkish-cot-distillation/Project",
)
VERI   = os.path.join(DRIVE, "asama2_sadelesmis.jsonl")
MDIR   = os.path.join(DRIVE, "modeller_tohum")
SDIR   = os.path.join(DRIVE, "degerlendirme_tohum")
DURUM  = os.path.join(MDIR, "durum.json")

MODELLER = [("qwen15","Qwen/Qwen2.5-1.5B-Instruct"),
            ("qwen3", "Qwen/Qwen2.5-3B-Instruct")]
VERSIYONLAR = ["orijinal","sade_morfem","sade_kelime"]
TOHUMLAR = [1, 2]        # tohum 0 = mevcut sonuçlar (yeniden eğitilmez)
N_TEST = 200
MAX_TOKEN = 512
# -----------------------------------------
os.makedirs(MDIR, exist_ok=True); os.makedirs(SDIR, exist_ok=True)

# ---------- durum takibi ----------
def durum_yukle():
    if os.path.exists(DURUM):
        with open(DURUM, encoding="utf-8") as f:
            return set(tuple(x) for x in json.load(f))
    return set()
def durum_kaydet(b):
    tmp = DURUM + ".tmp"
    with open(tmp,"w",encoding="utf-8") as f: json.dump([list(x) for x in b], f)
    os.replace(tmp, DURUM)

# ---------- ölçüm yardımcıları ----------
_an = zeyrek.MorphAnalyzer()
def kelime_say(m): return len(re.findall(r"\w+", m, re.UNICODE))
def morfem_yuku(m):
    t=0
    for k in re.findall(r"\w+", m, re.UNICODE):
        if k.isdigit(): t+=1; continue
        try:
            s=_an.analyze(k)
            if s and s[0]: t+=len(s[0][0].morphemes)
        except: pass
    return t
def son_cevap(a): return a.split("####")[-1].strip().replace(",","").replace(" ","")
def cevap_ayikla(m):
    x=re.search(r"(?:cevap|answer)\s*[:=]?\s*\$?(-?\d[\d,\.]*)", m, re.IGNORECASE)
    if x: return x.group(1).replace(",","").rstrip(".")
    s=re.findall(r"-?\d[\d,\.]*", m)
    return s[-1].replace(",","").rstrip(".") if s else None

def veri_hazirla(versiyon):
    ex=[]
    with open(VERI, encoding="utf-8") as f:
        for line in f:
            k=json.loads(line)
            ex.append({"messages":[
                {"role":"user","content":f"Bu matematik problemini adım adım Türkçe çöz. Sonunda 'Cevap: X' yaz.\n\n{k['soru']}"},
                {"role":"assistant","content":k[versiyon]}]})
    return Dataset.from_list(ex)

# Test problemleri (bir kez)
_tr = load_dataset("malhajar/gsm8k-tr","main",split="test")
TEST = [(_tr[i]["question"], son_cevap(_tr[i]["answer"])) for i in range(N_TEST)]

# ---------- tek eğitim ----------
def egit(etiket, model_adi, versiyon, tohum):
    out = os.path.join(MDIR, f"{etiket}_{versiyon}_s{tohum}")
    os.makedirs(out, exist_ok=True)
    set_seed(tohum); random.seed(tohum); np.random.seed(tohum); torch.manual_seed(tohum)

    bnb=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    tok=AutoTokenizer.from_pretrained(model_adi)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(model_adi, quantization_config=bnb,
                                               device_map="auto", torch_dtype=torch.bfloat16)
    model=prepare_model_for_kbit_training(model)
    lora=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
    cfg=SFTConfig(output_dir=out, num_train_epochs=3, per_device_train_batch_size=2,
        gradient_accumulation_steps=4, learning_rate=2e-4, logging_steps=50,
        save_strategy="epoch", save_total_limit=1, bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant":False}, report_to="none",
        max_length=1024, seed=tohum, data_seed=tohum)
    tr=SFTTrainer(model=model, args=cfg, train_dataset=veri_hazirla(versiyon),
                  peft_config=lora, processing_class=tok)
    ck=None
    cks=[d for d in os.listdir(out) if d.startswith("checkpoint-")]
    if cks: ck=os.path.join(out, sorted(cks,key=lambda x:int(x.split("-")[1]))[-1])
    tr.train(resume_from_checkpoint=ck)
    tr.save_model(out); tok.save_pretrained(out)
    del model, tr; gc.collect(); torch.cuda.empty_cache()
    return out

# ---------- tek değerlendirme ----------
def degerlendir(etiket, model_adi, versiyon, tohum):
    ad=f"{etiket}_{versiyon}_s{tohum}"
    dosya=os.path.join(SDIR, f"{ad}.jsonl")
    islenmis=0
    if os.path.exists(dosya):
        with open(dosya, encoding="utf-8") as f: islenmis=sum(1 for _ in f)
    if islenmis>=N_TEST: return dosya

    bnb=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    tok=AutoTokenizer.from_pretrained(model_adi)
    base=AutoModelForCausalLM.from_pretrained(model_adi, quantization_config=bnb,
                                              device_map="auto", torch_dtype=torch.bfloat16)
    model=PeftModel.from_pretrained(base, os.path.join(MDIR, ad)); model.eval()

    with open(dosya,"a",encoding="utf-8") as f:
        for i in tqdm(range(islenmis, N_TEST), initial=islenmis, total=N_TEST, desc=ad, leave=False):
            soru,gercek=TEST[i]
            msj=[{"role":"user","content":f"Bu matematik problemini adım adım Türkçe çöz. Sonunda 'Cevap: X' yaz.\n\n{soru}"}]
            metin=tok.apply_chat_template(msj, tokenize=False, add_generation_prompt=True)
            g=tok(metin, return_tensors="pt").to(model.device)
            with torch.no_grad():
                o=model.generate(**g, max_new_tokens=MAX_TOKEN, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            ids=o[0][g["input_ids"].shape[1]:]
            c=tok.decode(ids, skip_special_tokens=True)
            f.write(json.dumps({"dogru": cevap_ayikla(c)==gercek, "kelime": kelime_say(c),
                                "morfem": morfem_yuku(c), "kesildi": len(ids)>=MAX_TOKEN-1},
                               ensure_ascii=False)+"\n"); f.flush()
    del model, base; gc.collect(); torch.cuda.empty_cache()
    return dosya

def ozet(dosya):
    with open(dosya, encoding="utf-8") as f: k=[json.loads(l) for l in f]
    n=len(k)
    return {"dogruluk":100*sum(x["dogru"] for x in k)/n,
            "kelime":sum(x["kelime"] for x in k)/n,
            "morfem":sum(x["morfem"] for x in k)/n,
            "kesilme":100*sum(x["kesildi"] for x in k)/n}

# ---------- ANA DÖNGÜ ----------
def calistir():
    bitenler=durum_yukle()
    isler=[(me,ma,v,s) for me,ma in MODELLER for v in VERSIYONLAR for s in TOHUMLAR]
    print(f"Toplam iş: {len(isler)} | biten: {len(bitenler)}\n")
    for i,(me,ma,v,s) in enumerate(isler,1):
        anahtar=(me,v,str(s))
        if anahtar in bitenler:
            print(f"[{i}/{len(isler)}] ATLA: {me} × {v} × tohum{s}"); continue
        print(f"[{i}/{len(isler)}] {me} × {v} × tohum{s} → eğitim...")
        egit(me,ma,v,s)
        print(f"                    → değerlendirme...")
        degerlendir(me,ma,v,s)
        bitenler.add(anahtar); durum_kaydet(bitenler)
        print(f"                    ✓ tamam\n")

    # ---------- İSTATİSTİKSEL ÖZET ----------
    print("\n"+"="*86)
    print(f"ÇOKLU TOHUM SONUÇLARI (n={len(TOHUMLAR)} ek tohum, {N_TEST} test problemi)")
    print("="*86)
    print(f"{'Model':8} {'Versiyon':13} {'Doğruluk':>16} {'Morfem':>16} {'Tükenme':>16}")
    print("-"*86)
    tablo={}
    for me,ma in MODELLER:
        for v in VERSIYONLAR:
            oz=[]
            for s in TOHUMLAR:
                d=os.path.join(SDIR, f"{me}_{v}_s{s}.jsonl")
                if os.path.exists(d): oz.append(ozet(d))
            if not oz: continue
            def ms(key):
                vals=[o[key] for o in oz]
                m=statistics.mean(vals)
                sd=statistics.stdev(vals) if len(vals)>1 else 0.0
                return m, sd
            dm,ds=ms("dogruluk"); mm,msd=ms("morfem"); km,ksd=ms("kesilme")
            tablo[(me,v)]={"dogruluk":(dm,ds),"morfem":(mm,msd),"kesilme":(km,ksd)}
            print(f"{me:8} {v:13} {dm:>10.1f} ±{ds:<4.1f} {mm:>10.0f} ±{msd:<4.0f} {km:>10.1f} ±{ksd:<4.1f}")
        print("-"*86)

    with open(os.path.join(SDIR,"tohum_ozet.json"),"w",encoding="utf-8") as f:
        json.dump({f"{k[0]}_{k[1]}":v for k,v in tablo.items()}, f, ensure_ascii=False, indent=2)
    print(f"\nÖzet: {os.path.join(SDIR,'tohum_ozet.json')}")
    print("NOT: Makalede tohum-0 (mevcut) sonuçlarıyla birlikte 3 tohumun ortalaması raporlanmalıdır.")

calistir()
