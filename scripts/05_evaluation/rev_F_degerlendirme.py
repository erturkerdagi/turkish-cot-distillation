# ============================================================
#  REVİZYON F: AYRINTILI DEĞERLENDİRME (GPU)
#  Karşıladığı hakem maddeleri:
#    M3.1  - Çoklu token bütçesinde değerlendirme (256/512/1024/2048)
#    M3.2  - Cevabın açık eşleşmeden mi 'son sayı' yedeğinden mi geldiği
#    M3.4  - İnce ayarsız taban modeller
#    M4.5  - Tohum koşularının ana deneyle AYNI 300 madde üzerinde
#    M2.1/M2.2/M2.4/R3-6 - Yeni sürümlerin değerlendirilmesi
#
#  ÖNCEKİ SÜRÜMDEN FARKLAR
#    1) TOPLU (batch) üretim — tek tek üretime göre 4-6 kat hızlı
#    2) Her problem için AYRINTILI kayıt: üretilen metin, token sayısı,
#       tükenme bayrağı, cevabın kaynağı. Böylece hakemin tüm istatistik
#       talepleri sonradan GPU'suz hesaplanabilir.
#    3) Bütçe ve problem sayısı dışarıdan ayarlanabilir
#
#  Mevcut degerlendirme1000/ klasörüne DOKUNULMAZ; yeni klasöre yazılır.
#  Kesinti olursa aynı hücreyi yeniden çalıştırın — kaldığı yerden sürer.
# ============================================================

import subprocess, sys
for p in ["transformers", "peft", "bitsandbytes", "accelerate", "datasets",
          "zeyrek", "nltk", "tqdm"]:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", p],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
import nltk
for _p in ["punkt_tab", "punkt"]:
    try: nltk.download(_p, quiet=True)
    except Exception: pass

import os, re, gc, json, glob, logging, torch, zeyrek
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
from tqdm.auto import tqdm
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("zeyrek").setLevel(logging.ERROR)

# ---------------- AYARLAR ----------------
ANA_DIZIN = os.environ.get(
    "PROJE_DIZINI",
    "/content/drive/MyDrive/turkish-cot-distillation/Project",
)
ESKI_MODELLER = os.path.join(ANA_DIZIN, "modeller1000")          # mevcut adaptörler
YENI_MODELLER = os.path.join(ANA_DIZIN, "modeller_revizyon")     # rev_E çıktısı
SONUC_DIR     = os.path.join(ANA_DIZIN, "degerlendirme_revizyon")

MODEL_ADI = {
    "qwen05": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen15": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen3":  "Qwen/Qwen2.5-3B-Instruct",
    "llama3": "meta-llama/Llama-3.2-3B-Instruct",
}

N_TEST      = 300        # ana deneyle aynı problemler (M4.5)
VARSAYILAN_BUTCE = 512   # makaledeki değerlendirme bütçesi
BATCH       = 16         # toplu üretim boyutu; bellek yetmezse 8'e düşürün
METIN_KAYDET = True      # üretilen metni de kaydet (nitel örnekler için)

# --- Hangi koşular yapılacak ---
# 0) ANA DENEY yeniden: mevcut adaptörler, ama ayrıntılı kayıtla.
#    Gerekli, çünkü M3.2 "kaç doğru cevap 'son sayı' yedeğinden geldi" diye
#    SAYI istiyor ve bu alan eski degerlendirme1000 dosyalarında yok.
#    Compute sıkıntısı varsa False yapıp en sona bırakılabilir (~1,5 saat).
ANA_YENIDEN = True
ANA_KOSULARI = [(m, v) for m in ["qwen05", "qwen15", "qwen3", "llama3"]
                for v in ["orijinal", "sade_morfem", "sade_kelime"]]

# 1) Yeni sürümler (rev_E çıktısı), varsayılan bütçede
YENI_SURUM_KOSULARI = [(m, v) for m in ["qwen15", "qwen3"]
                       for v in ["yalniz_kural1", "yalniz_agirlik",
                                 "rastgele", "kesme", "referans"]]
# 2) Tohum genişletmesi (rev_E çıktısı), 300 madde üzerinde
TOHUM_KOSULARI = [(m, f"{v}_s{t}") for m in ["qwen05", "llama3"]
                  for v in ["orijinal", "sade_morfem", "sade_kelime"]
                  for t in [1, 2]]
# 3) Mevcut tohum adaptörleri 200 yerine 300 madde ile yeniden (M4.5)
ESKI_TOHUM_KOSULARI = [(m, f"{v}_s{t}") for m in ["qwen15", "qwen3"]
                       for v in ["orijinal", "sade_morfem", "sade_kelime"]
                       for t in [1, 2]]
# 4) Çoklu bütçe taraması (M3.1) — daha küçük örneklem
BUTCE_TARAMASI = {"modeller": ["qwen15", "qwen3"],
                  "versiyonlar": ["orijinal", "sade_morfem", "sade_kelime"],
                  "butceler": [256, 1024, 2048],   # 512 zaten ana deneyde var
                  "n": 100}
# 5) Taban modeller (M3.4) — ince ayarsız
TABAN_MODELLER = ["qwen05", "qwen15", "qwen3", "llama3"]
# -----------------------------------------
os.makedirs(SONUC_DIR, exist_ok=True)

_analyzer = zeyrek.MorphAnalyzer()
_t = "Öğrencilerin kitaplarını okuyabilmeleri için kütüphaneye gittiler"
_k = len(re.findall(r"\w+", _t, re.UNICODE)); _m = 0
for _w in re.findall(r"\w+", _t, re.UNICODE):
    _s = _analyzer.analyze(_w)
    if _s and _s[0]: _m += len(_s[0][0].morphemes)
assert _m / _k >= 1.5, f"HATA: Zeyrek çalışmıyor (morfem/kelime={_m/_k:.2f})"
print(f"Zeyrek sağlık testi geçti: {_m/_k:.2f}")

ISTEM = "Bu matematik problemini adım adım Türkçe çöz. Sonunda 'Cevap: X' yaz.\n\n{soru}"

def son_cevap(a):
    return a.split("####")[-1].strip().replace(",", "").replace(" ", "")

def cevap_ayikla(m):
    """Cevabı ve KAYNAĞINI döndürür — M3.2 bu ayrımı istiyor."""
    x = re.search(r"(?:cevap|answer)\s*[:=]?\s*\$?(-?\d[\d,\.]*)", m, re.IGNORECASE)
    if x:
        return x.group(1).replace(",", "").rstrip("."), "acik"
    s = re.findall(r"-?\d[\d,\.]*", m)
    if s:
        return s[-1].replace(",", "").rstrip("."), "son_sayi"
    return None, "yok"

def kelime_say(m): return len(re.findall(r"\w+", m, re.UNICODE))

_mc = {}
def morfem_yuku(m):
    t = 0
    for k in re.findall(r"\w+", m, re.UNICODE):
        if k.isdigit(): t += 1; continue
        if k not in _mc:
            try:
                s = _analyzer.analyze(k)
                _mc[k] = len(s[0][0].morphemes) if (s and s[0]) else 0
            except Exception:
                _mc[k] = 0
        t += _mc[k]
    return t

print("Test problemleri yükleniyor...")
tr_test = load_dataset("malhajar/gsm8k-tr", "main", split="test")
TUM_PROBLEMLER = [(tr_test[i]["question"], son_cevap(tr_test[i]["answer"]))
                  for i in range(max(N_TEST, BUTCE_TARAMASI["n"]))]


def adaptor_bul(etiket):
    """Adaptör klasörünü yeni ve eski model klasörlerinde arar."""
    for kok in [YENI_MODELLER, ESKI_MODELLER]:
        y = os.path.join(kok, etiket)
        if os.path.isdir(y) and (os.path.exists(os.path.join(y, "adapter_config.json"))
                                 or glob.glob(os.path.join(y, "adapter_model*"))):
            return y
        for alt in glob.glob(os.path.join(kok, "**", etiket), recursive=True):
            if os.path.isdir(alt):
                return alt
    return None


def degerlendir(model_etiket, versiyon, butce=VARSAYILAN_BUTCE, n=N_TEST,
                taban=False):
    """Tek bir koşu. taban=True ise adaptör yüklenmez (M3.4)."""
    ad = (f"{model_etiket}_TABAN" if taban else f"{model_etiket}_{versiyon}")
    dosya_adi = f"{ad}_b{butce}_n{n}.jsonl"
    sonuc = os.path.join(SONUC_DIR, dosya_adi)

    islenmis = 0
    if os.path.exists(sonuc):
        with open(sonuc, encoding="utf-8") as f:
            islenmis = sum(1 for _ in f)
    if islenmis >= n:
        print(f"    {dosya_adi}: zaten tamam")
        return sonuc

    temel_adi = MODEL_ADI[model_etiket]
    adaptor = None
    if not taban:
        adaptor = adaptor_bul(ad)
        if adaptor is None:
            print(f"    !! {ad}: adaptör bulunamadı, atlandı")
            return None

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(temel_adi)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"                     # toplu üretim için şart
    model = AutoModelForCausalLM.from_pretrained(temel_adi, quantization_config=bnb,
                                                 device_map="auto",
                                                 torch_dtype=torch.bfloat16)
    if adaptor:
        model = PeftModel.from_pretrained(model, adaptor)
    model.eval()

    # Bitiş belirteçleri: Llama-3 <|eot_id|> ile biter, bu eos_token_id'den farklıdır.
    # Eksik bırakılırsa her Llama çıktısı "tükendi" sayılır — metrik bozulur.
    bitirici = set()
    for kaynak in (getattr(model.generation_config, "eos_token_id", None),
                   tok.eos_token_id):
        if isinstance(kaynak, (list, tuple)):
            bitirici.update(int(x) for x in kaynak)
        elif kaynak is not None:
            bitirici.add(int(kaynak))
    print(f"    bitiş belirteçleri: {sorted(bitirici)}")

    with open(sonuc, "a", encoding="utf-8") as f:
        ilerleme = tqdm(range(islenmis, n), initial=islenmis, total=n,
                        desc=f"  {dosya_adi}", leave=False)
        for bas in range(islenmis, n, BATCH):
            son = min(bas + BATCH, n)
            kume = TUM_PROBLEMLER[bas:son]
            metinler = [tok.apply_chat_template(
                            [{"role": "user", "content": ISTEM.format(soru=s)}],
                            tokenize=False, add_generation_prompt=True)
                        for s, _ in kume]
            # add_special_tokens asama4 ile aynı bırakıldı (varsayılan),
            # yoksa yeni koşular mevcut degerlendirme1000 sayılarıyla kıyaslanamaz
            girdi = tok(metinler, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                cikti = model.generate(**girdi, max_new_tokens=butce,
                                       do_sample=False, pad_token_id=tok.pad_token_id)
            girdi_uzunluk = girdi["input_ids"].shape[1]
            for j, (soru, gercek) in enumerate(kume):
                uretilen = cikti[j][girdi_uzunluk:]
                # gerçek üretim uzunluğu: ilk bitiş belirtecine kadar
                yer = None
                for bid in bitirici:
                    bul = (uretilen == bid).nonzero()
                    if len(bul):
                        ilk = int(bul[0].item())
                        yer = ilk if yer is None else min(yer, ilk)
                if yer is not None:
                    uzunluk, kesildi = yer, False
                else:
                    uzunluk = int(uretilen.shape[0])
                    kesildi = uzunluk >= butce - 1
                cozum = tok.decode(uretilen[:uzunluk], skip_special_tokens=True)
                tahmin, kaynak = cevap_ayikla(cozum)
                kayit = {
                    "indeks": bas + j,
                    "dogru": (tahmin == gercek),
                    "kelime": kelime_say(cozum),
                    "morfem": morfem_yuku(cozum),
                    "kesildi": kesildi,
                    "token": uzunluk,
                    "cevap_kaynagi": kaynak,
                    "tahmin": tahmin,
                    "gercek": gercek,
                }
                if METIN_KAYDET:
                    kayit["metin"] = cozum
                f.write(json.dumps(kayit, ensure_ascii=False) + "\n")
            f.flush()
            ilerleme.update(son - bas)
        ilerleme.close()

    del model; gc.collect(); torch.cuda.empty_cache()

    with open(sonuc, encoding="utf-8") as f:
        kayitlar = [json.loads(l) for l in f]
    nn = len(kayitlar)
    print(f"    {dosya_adi}: doğruluk {sum(k['dogru'] for k in kayitlar)/nn:.1%}  "
          f"tükenme {sum(k['kesildi'] for k in kayitlar)/nn:.1%}  "
          f"token {sum(k['token'] for k in kayitlar)/nn:.0f}  "
          f"son-sayı yedeği {sum(1 for k in kayitlar if k['cevap_kaynagi']=='son_sayi')/nn:.1%}")
    return sonuc


def main():
    if ANA_YENIDEN:
        print("\n" + "=" * 70)
        print("0) ANA DENEY — ayrıntılı kayıtla yeniden (M3.2)")
        print("=" * 70)
        for m, v in ANA_KOSULARI:
            degerlendir(m, v)

    print("\n" + "=" * 70)
    print("1) YENİ SÜRÜMLER (M2.1, M2.2, M2.4, R3-6)")
    print("=" * 70)
    for m, v in YENI_SURUM_KOSULARI:
        degerlendir(m, v)

    print("\n" + "=" * 70)
    print("2) TOHUM GENİŞLETMESİ — qwen05 ve llama3 (M5.5, R3-4)")
    print("=" * 70)
    for m, v in TOHUM_KOSULARI:
        degerlendir(m, v)

    print("\n" + "=" * 70)
    print("3) MEVCUT TOHUMLAR 300 MADDE ÜZERİNDE (M4.5)")
    print("=" * 70)
    for m, v in ESKI_TOHUM_KOSULARI:
        degerlendir(m, v)

    print("\n" + "=" * 70)
    print("4) TABAN MODELLER — ince ayarsız (M3.4)")
    print("=" * 70)
    for m in TABAN_MODELLER:
        degerlendir(m, "taban", taban=True)

    print("\n" + "=" * 70)
    print("5) ÇOKLU BÜTÇE TARAMASI (M3.1)")
    print("=" * 70)
    bt = BUTCE_TARAMASI
    for b in bt["butceler"]:
        for m in bt["modeller"]:
            for v in bt["versiyonlar"]:
                degerlendir(m, v, butce=b, n=bt["n"])
    # 512 bütçesi için de aynı örneklemde referans noktası
    for m in bt["modeller"]:
        for v in bt["versiyonlar"]:
            degerlendir(m, v, butce=512, n=bt["n"])

    print("\n" + "=" * 70)
    print("TAMAMLANDI")
    print("=" * 70)
    print(f"Sonuçlar: {SONUC_DIR}")
    dosyalar = sorted(os.listdir(SONUC_DIR))
    print(f"Üretilen dosya: {len(dosyalar)}")
    print("Sıradaki adım: rev_G istatistiklerini bu klasörle yeniden çalıştırın.")


main()
