# ============================================================
#  REVİZYON E: YENİ SÜRÜM EĞİTİMLERİ (GPU)
#  Karşıladığı hakem maddeleri:
#    M2.1  - Kural ablasyonu sürümlerinin eğitimi
#    M2.2  - Rastgele silme temel yöntemi
#    M2.4  - gsm8k-tr referans çözümleriyle eğitim
#    R3-6  - Düzgün kesme temel yöntemi
#    M5.5 / R3-4 - Qwen0.5B ve Llama3B için tohum genişletmesi
#
#  Hiperparametreler asama3_tam_egitim_v2.py ile BİREBİR aynıdır.
#  Mevcut adaptörlere dokunulmaz; yenileri ayrı klasöre yazılır.
#  Kesinti olursa aynı hücreyi yeniden çalıştırın — kaldığı yerden sürer.
#
#  Tahmini süre (A100): 22 eğitim x ~8 dk = ~3 saat
# ============================================================

import subprocess, sys
for p in ["transformers", "peft", "trl", "bitsandbytes", "accelerate", "datasets"]:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", p],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

import os, gc, json, glob, random, logging, numpy as np, torch
from datasets import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, set_seed)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
logging.getLogger("transformers").setLevel(logging.ERROR)

# ---------------- AYARLAR ----------------
ANA_DIZIN = os.environ.get(
    "PROJE_DIZINI",
    "/content/drive/MyDrive/turkish-cot-distillation/Project",
)
REV_DIR      = os.path.join(ANA_DIZIN, "revizyon_ciktilari")
MODELLER_DIR = os.path.join(ANA_DIZIN, "modeller_revizyon")     # YENİ klasör
DURUM_DOSYA  = os.path.join(MODELLER_DIR, "egitim_durumu.json")

MODEL_ADI = {
    "qwen05": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen15": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen3":  "Qwen/Qwen2.5-3B-Instruct",
    "llama3": "meta-llama/Llama-3.2-3B-Instruct",
}

# --- A) Yeni sürümler: hangi dosyadan, hangi alan, hangi modeller ---
YENI_SURUMLER = [
    # (sürüm adı,        kaynak dosya,                      alan adı,             modeller)
    ("yalniz_kural1",   "ablasyon_surumleri.jsonl",         "sade_yalniz_kural1",  ["qwen15", "qwen3"]),
    ("yalniz_agirlik",  "ablasyon_surumleri.jsonl",         "sade_yalniz_agirlik", ["qwen15", "qwen3"]),
    ("rastgele",        "temel_surumleri_onarilmis.jsonl",  "sade_rastgele",       ["qwen15", "qwen3"]),
    ("kesme",           "temel_surumleri_onarilmis.jsonl",  "sade_kesme",          ["qwen15", "qwen3"]),
    ("referans",        "temel_surumleri_onarilmis.jsonl",  "sade_referans",       ["qwen15", "qwen3"]),
]

# --- B) Tohum genişletmesi: eksik modeller için 1 ve 2 numaralı tohumlar ---
TOHUM_MODELLER  = ["qwen05", "llama3"]
TOHUM_SURUMLER  = ["orijinal", "sade_morfem", "sade_kelime"]
TOHUMLAR        = [1, 2]
ANA_VERI_DESEN  = "asama2_sadelesmis*.jsonl"     # 1000'lik koşunun dosyası
# -----------------------------------------
os.makedirs(MODELLER_DIR, exist_ok=True)

ISTEM = "Bu matematik problemini adım adım Türkçe çöz. Sonunda 'Cevap: X' yaz.\n\n{soru}"


def dosya_bul(desen):
    for kok in [REV_DIR, ANA_DIZIN]:
        a = [y for y in glob.glob(os.path.join(kok, "**", desen), recursive=True)
             if "modeller_revizyon" not in y]
        if a:
            return sorted(a, key=lambda y: -os.path.getsize(y))[0]
    return None


def durum_yukle():
    if os.path.exists(DURUM_DOSYA):
        with open(DURUM_DOSYA, encoding="utf-8") as f:
            return set(tuple(x) for x in json.load(f))
    return set()

def durum_kaydet(b):
    t = DURUM_DOSYA + ".tmp"
    with open(t, "w", encoding="utf-8") as f:
        json.dump([list(x) for x in b], f, ensure_ascii=False)
    os.replace(t, DURUM_DOSYA)


def veri_hazirla(dosya, alan):
    ornekler = []
    with open(dosya, encoding="utf-8") as f:
        for satir in f:
            k = json.loads(satir)
            metin = k.get(alan)
            if not metin:
                continue
            ornekler.append({"messages": [
                {"role": "user", "content": ISTEM.format(soru=k["soru"])},
                {"role": "assistant", "content": metin},
            ]})
    return Dataset.from_list(ornekler)


def bir_egitim(etiket, model_adi, dataset, tohum=0):
    cikti_dir = os.path.join(MODELLER_DIR, etiket)
    os.makedirs(cikti_dir, exist_ok=True)
    set_seed(tohum); random.seed(tohum); np.random.seed(tohum); torch.manual_seed(tohum)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(model_adi)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_adi, quantization_config=bnb,
                                                 device_map="auto",
                                                 torch_dtype=torch.bfloat16)
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    cfg = SFTConfig(output_dir=cikti_dir, num_train_epochs=3,
                    per_device_train_batch_size=2, gradient_accumulation_steps=4,
                    learning_rate=2e-4, logging_steps=20,
                    save_strategy="epoch", save_total_limit=1,
                    bf16=True, gradient_checkpointing=True,
                    gradient_checkpointing_kwargs={"use_reentrant": False},
                    report_to="none", max_length=1024,
                    seed=tohum, data_seed=tohum)
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=dataset,
                         processing_class=tok, peft_config=lora)

    ckpt = [d for d in os.listdir(cikti_dir) if d.startswith("checkpoint-")]
    trainer.train(resume_from_checkpoint=bool(ckpt))
    trainer.save_model(cikti_dir)
    tok.save_pretrained(cikti_dir)
    del trainer, model; gc.collect(); torch.cuda.empty_cache()


def main():
    bitenler = durum_yukle()
    isler = []

    # --- A) yeni sürümler ---
    for surum, dosya_adi, alan, modeller in YENI_SURUMLER:
        yol = dosya_bul(dosya_adi)
        if not yol:
            print(f"  !! Atlandı ({surum}): {dosya_adi} bulunamadı")
            continue
        for m in modeller:
            isler.append((f"{m}_{surum}", MODEL_ADI[m], yol, alan, 0))

    # --- B) tohum genişletmesi ---
    ana_veri = dosya_bul(ANA_VERI_DESEN)
    if ana_veri:
        for m in TOHUM_MODELLER:
            for v in TOHUM_SURUMLER:
                for t in TOHUMLAR:
                    isler.append((f"{m}_{v}_s{t}", MODEL_ADI[m], ana_veri, v, t))
    else:
        print(f"  !! Tohum genişletmesi atlandı: {ANA_VERI_DESEN} bulunamadı")

    kalan = [i for i in isler if (i[0],) not in bitenler]
    print("=" * 70)
    print(f"Toplam eğitim: {len(isler)} | zaten biten: {len(isler)-len(kalan)} | "
          f"yapılacak: {len(kalan)}")
    print(f"Çıktı klasörü: {MODELLER_DIR}")
    print("=" * 70)

    for n, (etiket, model_adi, dosya, alan, tohum) in enumerate(kalan, 1):
        print(f"\n[{n}/{len(kalan)}] EĞİT: {etiket}  (alan={alan}, tohum={tohum})")
        try:
            ds = veri_hazirla(dosya, alan)
            print(f"           örnek sayısı: {len(ds)}")
            if len(ds) == 0:
                print("           !! veri boş, atlandı"); continue
            bir_egitim(etiket, model_adi, ds, tohum)
            bitenler.add((etiket,)); durum_kaydet(bitenler)
            print("           ✓ tamamlandı")
        except Exception as e:
            print(f"           ✗ HATA: {e}")
            gc.collect(); torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print(f"BİTTİ: {len(bitenler)}/{len(isler)}")
    print("Sıradaki adım: rev_F_degerlendirme.py")


main()
