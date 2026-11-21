# ============================================================
#  AŞAMA 3 (TAM, v2): 4 MODEL × 3 VERSİYON = 12 QLoRA EĞİTİMİ
#  Modeller: Qwen 0.5B/1.5B/3B + Llama-3.2-3B (farklı aile)
#  - İki seviyeli checkpoint (durum + epoch)
#  - Her model kendi chat template'iyle (Qwen/Llama otomatik)
# ============================================================
# Colab Pro A100: from google.colab import drive; drive.mount('/content/drive')
# NOT: Llama modelleri için HF girişi gerekebilir:
#   from huggingface_hub import login; login(os.environ["HF_TOKEN"])
# ------------------------------------------------------------

import subprocess, sys, logging
for paket in ["peft","trl","bitsandbytes","transformers","datasets","accelerate"]:
    subprocess.run([sys.executable,"-m","pip","install","-q",paket],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

import os, json, gc, torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
logging.getLogger("transformers").setLevel(logging.ERROR)

# ---------------- AYARLAR ----------------
DRIVE_DIR = os.environ.get(
    "PROJE_DIZINI",
    "/content/drive/MyDrive/turkish-cot-distillation/Project",
)
VERI = os.path.join(DRIVE_DIR, "asama2_sadelesmis.jsonl")
MODELLER_DIR = os.path.join(DRIVE_DIR, "modeller")
DURUM_DOSYA = os.path.join(MODELLER_DIR, "egitim_durumu.json")

MODELLER = [
    ("qwen05",  "Qwen/Qwen2.5-0.5B-Instruct"),
    ("qwen15",  "Qwen/Qwen2.5-1.5B-Instruct"),
    ("qwen3",   "Qwen/Qwen2.5-3B-Instruct"),
    ("llama3",  "meta-llama/Llama-3.2-3B-Instruct"),   # FARKLI AİLE
]
VERSIYONLAR = ["orijinal","sade_morfem","sade_kelime"]
# -----------------------------------------
os.makedirs(MODELLER_DIR, exist_ok=True)

def durum_yukle():
    if os.path.exists(DURUM_DOSYA):
        with open(DURUM_DOSYA, encoding="utf-8") as f:
            return set(tuple(x) for x in json.load(f))
    return set()
def durum_kaydet(b):
    tmp=DURUM_DOSYA+".tmp"
    with open(tmp,"w",encoding="utf-8") as f: json.dump([list(x) for x in b],f,ensure_ascii=False)
    os.replace(tmp,DURUM_DOSYA)

def veri_hazirla(versiyon):
    ornekler=[]
    with open(VERI, encoding="utf-8") as f:
        for line in f:
            k=json.loads(line)
            ornekler.append({"messages":[
                {"role":"user","content":f"Bu matematik problemini adım adım Türkçe çöz. Sonunda 'Cevap: X' yaz.\n\n{k['soru']}"},
                {"role":"assistant","content":k[versiyon]},
            ]})
    return Dataset.from_list(ornekler)

def bir_egitim(model_etiket, model_adi, versiyon):
    cikti_dir=os.path.join(MODELLER_DIR, f"{model_etiket}_{versiyon}")
    os.makedirs(cikti_dir, exist_ok=True)
    dataset=veri_hazirla(versiyon)
    bnb=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    tok=AutoTokenizer.from_pretrained(model_adi)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(model_adi, quantization_config=bnb,
                                               device_map="auto", torch_dtype=torch.bfloat16)
    model=prepare_model_for_kbit_training(model)
    # LoRA hedef modülleri: hem Qwen hem Llama bu isimleri kullanır
    lora=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
    cfg=SFTConfig(output_dir=cikti_dir, num_train_epochs=3,
                  per_device_train_batch_size=2, gradient_accumulation_steps=4,
                  learning_rate=2e-4, logging_steps=20, save_strategy="epoch", save_total_limit=1,
                  bf16=True, gradient_checkpointing=True,
                  gradient_checkpointing_kwargs={"use_reentrant":False},
                  report_to="none", max_length=1024)
    trainer=SFTTrainer(model=model, args=cfg, train_dataset=dataset,
                       peft_config=lora, processing_class=tok)
    ckpt=None
    ckptler=[d for d in os.listdir(cikti_dir) if d.startswith("checkpoint-")]
    if ckptler:
        ckpt=os.path.join(cikti_dir, sorted(ckptler,key=lambda x:int(x.split("-")[1]))[-1])
    trainer.train(resume_from_checkpoint=ckpt)
    trainer.save_model(cikti_dir); tok.save_pretrained(cikti_dir)
    del model, trainer; gc.collect(); torch.cuda.empty_cache()

def tam_egitim():
    bitenler=durum_yukle()
    toplam=len(MODELLER)*len(VERSIYONLAR)
    print(f"Toplam eğitim: {toplam} | zaten biten: {len(bitenler)}\n")
    sayac=0
    for me,ma in MODELLER:
        for v in VERSIYONLAR:
            sayac+=1
            if (me,v) in bitenler:
                print(f"[{sayac}/{toplam}] ATLA (bitti): {me} × {v}"); continue
            print(f"[{sayac}/{toplam}] EĞİT: {me} × {v} ...")
            try:
                bir_egitim(me,ma,v)
                bitenler.add((me,v)); durum_kaydet(bitenler)
                print(f"           ✓ tamamlandı\n")
            except Exception as e:
                print(f"           ✗ HATA: {e}")
                raise
    print("="*50); print(f"TÜM EĞİTİMLER TAMAM: {len(bitenler)}/{toplam}")

tam_egitim()
