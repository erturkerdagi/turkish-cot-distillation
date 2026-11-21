# ============================================================
#  AŞAMA 1: ÖĞRETMEN VERİSİ HAZIRLAMA (gsm8k-tr çözümleri)
#  - Drive'a kaydeder
#  - Checkpoint: sayıyı artırınca kaldığı yerden devam eder
#  - tqdm progress bar, temiz çıktı
# ============================================================
# Colab'da önce Drive'ı bağla:
#   from google.colab import drive
#   drive.mount('/content/drive')
# ------------------------------------------------------------

import subprocess, sys, logging
for paket in ["datasets", "tqdm"]:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", paket],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

import os, json, re
from datasets import load_dataset
from tqdm.auto import tqdm

# Gürültüyü kıs
logging.getLogger("datasets").setLevel(logging.ERROR)

# ---------------- AYARLAR ----------------
DRIVE_DIR = os.environ.get(
    "PROJE_DIZINI",
    "/content/drive/MyDrive/turkish-cot-distillation/Project",
)
N_PROBLEM = 500          # makul deneme; sonra artır -> kaldığı yerden devam eder
CIKTI = os.path.join(DRIVE_DIR, "asama1_ogretmen_verisi.jsonl")
# -----------------------------------------

os.makedirs(DRIVE_DIR, exist_ok=True)

def son_cevap(answer_metni):
    """GSM8K answer alanından #### sonrası nihai sayısal cevabı çıkarır."""
    return answer_metni.split("####")[-1].strip().replace(",", "").replace(" ", "")

def cozum_temizle(answer_metni):
    """#### işaretini ve sonrasını atıp sadece adım adım çözüm metnini bırakır."""
    return answer_metni.split("####")[0].strip()

def kac_islenmis():
    """Checkpoint: çıktı dosyasında kaç satır (problem) zaten var?"""
    if not os.path.exists(CIKTI):
        return 0
    with open(CIKTI, encoding="utf-8") as f:
        return sum(1 for _ in f)

def asama1_calistir(n=N_PROBLEM):
    print("gsm8k-tr yükleniyor...")
    tr = load_dataset("malhajar/gsm8k-tr", "main", split="train")
    toplam_mevcut = len(tr)
    hedef = min(n, toplam_mevcut)

    baslangic = kac_islenmis()
    if baslangic >= hedef:
        print(f"Zaten {baslangic} problem işlenmiş, hedef {hedef}. Yapılacak yeni iş yok.")
        return
    print(f"Toplam mevcut: {toplam_mevcut} | hedef: {hedef} | zaten işlenmiş: {baslangic}")

    # Append modunda aç (checkpoint'ten devam)
    with open(CIKTI, "a", encoding="utf-8") as f:
        for idx in tqdm(range(baslangic, hedef), initial=baslangic, total=hedef, desc="Aşama 1"):
            ornek = tr[idx]
            kayit = {
                "soru": ornek["question"],
                "ogretmen_cozum": cozum_temizle(ornek["answer"]),   # adım adım Türkçe çözüm
                "cevap": son_cevap(ornek["answer"]),                # nihai sayı
            }
            f.write(json.dumps(kayit, ensure_ascii=False) + "\n")
            f.flush()   # her satırı hemen diske yaz (yarıda kesilmeye dayanıklı)

    print(f"\nTamam. Toplam {kac_islenmis()} problem '{CIKTI}' dosyasına yazıldı.")

    # Küçük bir örnek göster (kontrol için)
    with open(CIKTI, encoding="utf-8") as f:
        ilk = json.loads(f.readline())
    print("\n--- ÖRNEK KAYIT ---")
    print("SORU:", ilk["soru"][:150])
    print("ÇÖZÜM:", ilk["ogretmen_cozum"][:200])
    print("CEVAP:", ilk["cevap"])

asama1_calistir()
