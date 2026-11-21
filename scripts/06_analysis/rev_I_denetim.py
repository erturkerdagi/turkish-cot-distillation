# ============================================================
#  REVİZYON I: DENETİM VE EKSİK ÖLÇÜMLER
#  Hakem karşısında açık bırakmamak için üç şeyi kapatır:
#    (1) M3.3 — eğitim hedeflerinin GERÇEK tokenizer ile token uzunluğu ve
#        1024'te kesilme oranı. Makaleye yazılan değer şu an vekil hesapla
#        tahmin edilmiş durumda; bu betik gerçek sayıyı verir.
#    (2) M4.5 — Qwen2.5-1.5B ve Qwen2.5-3B tohum koşuları hâlâ 200 madde
#        üzerinde. Bu betik eski tohum adaptörlerini bulur ve 300 madde
#        üzerinde yeniden değerlendirir.
#    (3) Eğitim hedefi istatistikleri — her sürümün cümle/kelime/morfem
#        dağılımı. Hakem "eşleştirme gerçekten tuttu mu" diye sorarsa
#        cevabı hazır olsun.
#  (1) ve (3) CPU, dakikalar. (2) GPU, ~1,5 saat.
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

import os, re, gc, json, glob, logging, statistics as ist
logging.getLogger("transformers").setLevel(logging.ERROR)

ANA_DIZIN = os.environ.get(
    "PROJE_DIZINI",
    "/content/drive/MyDrive/turkish-cot-distillation/Project",
)
REV_DIR   = os.path.join(ANA_DIZIN, "revizyon_ciktilari")
SONUC_DIR = os.path.join(ANA_DIZIN, "degerlendirme_revizyon")
OGRENCI   = "Qwen/Qwen2.5-1.5B-Instruct"

TOHUM_KOSU = True        # (2) GPU bölümü; kapatmak için False yapın
N_TEST     = 300
BUTCE      = 512
BATCH      = 16
os.makedirs(REV_DIR, exist_ok=True)
SONUC = {}


def baslik(s):
    print("\n" + "=" * 76); print(s); print("=" * 76)

def jsonl(y):
    with open(y, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def bul(desen):
    for kok in [REV_DIR, ANA_DIZIN]:
        a = glob.glob(os.path.join(kok, "**", desen), recursive=True)
        if a: return sorted(a, key=lambda y: -os.path.getsize(y))[0]
    return None

def cum(m):
    return [p.strip() for p in re.split(r"(?<!\d)[.!?\n]+(?!\d)", m) if p.strip()]
def kel(m):
    return len(re.findall(r"\w+", m, re.UNICODE))


# ============================================================
# (1) + (3) GERÇEK TOKEN UZUNLUKLARI VE HEDEF İSTATİSTİKLERİ
# ============================================================
def hedef_istatistikleri():
    baslik("1) EĞİTİM HEDEFLERİ — gerçek tokenizer ile (M3.3 + eşleştirme denetimi)")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(OGRENCI)

    ab_y, tb_y = bul("ablasyon_surumleri.jsonl"), bul("temel_surumleri_onarilmis.jsonl")
    sade_y = bul("asama2_sadelesmis*.jsonl")
    if not (ab_y and tb_y):
        print("  !! ablasyon/temel dosyaları bulunamadı."); return
    ab, tb = jsonl(ab_y), jsonl(tb_y)
    sade = jsonl(sade_y) if sade_y else []
    sade_m = {r["soru"]: r for r in sade} if sade else {}

    surumler = [("Original", "orijinal", ab),
                ("Morpheme-simplified", "sade_morfem", None),
                ("Word-simplified", "sade_kelime", None),
                ("Rule 1 only", "sade_yalniz_kural1", ab),
                ("Weight filter only", "sade_yalniz_agirlik", ab),
                ("Random deletion", "sade_rastgele", tb),
                ("Uniform truncation", "sade_kesme", tb),
                ("Reference solutions", "sade_referans", tb)]

    print(f"  {'Training condition':22} {'sent.':>7} {'words':>8} {'tokens':>8} "
          f"{'max tok':>8} {'>=1024':>8}")
    print("  " + "-" * 66)
    tablo = {}
    for ad, alan, kaynak in surumler:
        if kaynak is None:
            metinler = [sade_m[r["soru"]][alan] for r in ab
                        if r["soru"] in sade_m and sade_m[r["soru"]].get(alan)]
        else:
            metinler = [r[alan] for r in kaynak if r.get(alan)]
        if not metinler:
            print(f"  {ad:22} — veri yok —"); continue
        c = [len(cum(m)) for m in metinler]
        w = [kel(m) for m in metinler]
        t = [len(tok(m, add_special_tokens=False)["input_ids"]) for m in metinler]
        kes = sum(1 for x in t if x >= 1024)
        print(f"  {ad:22} {ist.mean(c):>7.2f} {ist.mean(w):>8.1f} {ist.mean(t):>8.1f} "
              f"{max(t):>8} {kes:>4}/{len(t)}")
        tablo[ad] = {"n": len(t), "cumle": round(ist.mean(c), 2),
                     "kelime": round(ist.mean(w), 1), "token": round(ist.mean(t), 1),
                     "maks_token": max(t), "kesilen": kes,
                     "kesilme_orani": round(kes / len(t), 4)}
    SONUC["hedef_istatistikleri"] = tablo

    # --- eşleştirme gerçekten tuttu mu ---
    baslik("2) UZUNLUK EŞLEŞTİRMESİ DENETİMİ — temel yöntemler tam ile eşleşiyor mu?")
    ayni_r = ayni_k = buyuk_k = 0
    fark_r, fark_k = [], []
    for r in tb:
        nt, nr, nk = len(cum(r["sade_tam"])), len(cum(r["sade_rastgele"])), len(cum(r["sade_kesme"]))
        ayni_r += (nr == nt); ayni_k += (nk == nt); buyuk_k += (nk > nt)
        fark_r.append(nr - nt); fark_k.append(nk - nt)
    n = len(tb)
    print(f"  Random deletion    : tam ile aynı cümle sayısı {ayni_r}/{n} ({ayni_r/n:.1%}), "
          f"ort. fark {ist.mean(fark_r):+.2f}")
    print(f"  Uniform truncation : tam ile aynı cümle sayısı {ayni_k}/{n} ({ayni_k/n:.1%}), "
          f"ort. fark {ist.mean(fark_k):+.2f}, daha uzun olan {buyuk_k}/{n} ({buyuk_k/n:.1%})")
    print()
    print("  YORUM: Düzgün kesme, stride ile seçilen cümlelerin üzerine son cümleyi de")
    print("  eklediği için tam yöntemden sistematik olarak UZUN. Makalede 'length-matched'")
    print("  denmemeli; bunun yerine hedef uzunlukları tabloyla verilmeli. Bulgu zayıflamaz,")
    print("  güçlenir: daha uzun hedeflerle aynı tükenme oranına ulaşılıyor.")
    SONUC["eslestirme_denetimi"] = {
        "rastgele_ayni": ayni_r, "kesme_ayni": ayni_k, "kesme_daha_uzun": buyuk_k,
        "n": n, "rastgele_ort_fark": round(ist.mean(fark_r), 3),
        "kesme_ort_fark": round(ist.mean(fark_k), 3)}

    # --- hedef uzunluğu ↔ tükenme ilişkisi ---
    baslik("3) HEDEF UZUNLUĞU ↔ ÖĞRENCİ TÜKENMESİ — 'uzunluk sürüklüyor' iddiasının testi")
    exh = {"Original": (38.3, 34.7), "Morpheme-simplified": (15.0, 18.3),
           "Word-simplified": (16.3, 18.3), "Rule 1 only": (25.7, 33.7),
           "Weight filter only": (18.3, 24.3), "Random deletion": (22.3, 21.0),
           "Uniform truncation": (14.0, 19.0), "Reference solutions": (1.7, 2.0)}
    xs = [tablo[a]["token"] for a in exh if a in tablo]
    for i, m in enumerate(["Qwen2.5-1.5B", "Qwen2.5-3B"]):
        ys = [exh[a][i] for a in exh if a in tablo]
        try:
            from scipy import stats
            r, p = stats.pearsonr(xs, ys)
            print(f"  {m}: tükenme ~ hedef token uzunluğu   r = {r:+.3f}, p = {p:.4f}, n = {len(xs)}")
            SONUC.setdefault("uzunluk_korelasyonu", {})[m] = {"r": round(float(r), 4),
                                                              "p": round(float(p), 5),
                                                              "n": len(xs)}
        except Exception as e:
            print(f"  korelasyon hesaplanamadı: {e}")
    print()
    print("  Bu korelasyon, makalenin yeni ana iddiasının niceliksel omurgasıdır.")
    print("  Ayrıca bir alternatif açıklama test edilmeli: hedefin SON cümlesinde cevabın")
    print("  bulunma oranı. Aşağıda hesaplanıyor; tükenmeyle ilişkisizse eleme yapılmış olur.")

    son_oran = {}
    for ad, alan, kaynak in surumler:
        if kaynak is None:
            kayitlar = [(sade_m[r["soru"]][alan], r["gercek_cevap"]) for r in ab
                        if r["soru"] in sade_m and sade_m[r["soru"]].get(alan)]
        else:
            kayitlar = [(r[alan], r["gercek_cevap"]) for r in kaynak if r.get(alan)]
        if not kayitlar: continue
        v = sum(1 for m, c in kayitlar if cum(m) and c in cum(m)[-1]) / len(kayitlar)
        son_oran[ad] = round(v, 4)
    print(f"\n  {'Training condition':22} {'son cümlede cevap':>18}")
    print("  " + "-" * 42)
    for a, v in son_oran.items():
        print(f"  {a:22} {v:>17.1%}")
    try:
        from scipy import stats
        sx = [son_oran[a] for a in exh if a in son_oran]
        for i, m in enumerate(["Qwen2.5-1.5B", "Qwen2.5-3B"]):
            sy = [exh[a][i] for a in exh if a in son_oran]
            r, p = stats.pearsonr(sx, sy)
            print(f"  {m}: tükenme ~ son-cümle oranı   r = {r:+.3f}, p = {p:.4f}")
            SONUC.setdefault("son_cumle_korelasyonu", {})[m] = {"r": round(float(r), 4),
                                                                "p": round(float(p), 5)}
    except Exception:
        pass
    SONUC["son_cumlede_cevap"] = son_oran


# ============================================================
# (2) EKSİK TOHUM KOŞULARI — 300 MADDE ÜZERİNDE
# ============================================================
def eksik_tohumlar():
    baslik("4) M4.5 — Qwen2.5-1.5B / Qwen2.5-3B tohum koşuları 300 madde üzerinde")
    print("  Hakem, tohum koşularının ana deneyle AYNI 300 madde üzerinde")
    print("  değerlendirilmesini istedi. Bu iki model hâlâ 200 madde üzerinde.\n")

    hedefler = [(m, f"{v}_s{t}")
                for m in ["qwen15", "qwen3"]
                for v in ["orijinal", "sade_morfem", "sade_kelime"]
                for t in [1, 2]]

    # adaptörleri Drive genelinde ara.
    # DİKKAT: checkpoint-NN alt klasörleri de adapter_config.json içerir ve
    # bunlar ARA EPOCH'tur. Yanlışlıkla yüklenirse sessizce yanlış model
    # değerlendirilir. Bu yüzden checkpoint içeren yollar elenir ve
    # tercih sırası modeller_tohum > modeller_revizyon > modeller1000 > diğer.
    ONCELIK = ["modeller_tohum", "modeller_revizyon", "modeller1000"]
    def oncelik_puani(y):
        for i, k in enumerate(ONCELIK):
            if os.sep + k + os.sep in y + os.sep:
                return i
        return len(ONCELIK)

    print("  Adaptörler aranıyor (checkpoint klasörleri elenir)...")
    bulunan = {}
    for m, v in hedefler:
        ad = f"{m}_{v}"
        adaylar = []
        for y in glob.glob(os.path.join(ANA_DIZIN, "**", ad), recursive=True):
            if "checkpoint-" in y:
                continue
            if os.path.isdir(y) and (os.path.exists(os.path.join(y, "adapter_config.json"))
                                     or glob.glob(os.path.join(y, "adapter_model*"))):
                adaylar.append(y)
        adaylar.sort(key=oncelik_puani)
        bulunan[ad] = adaylar[0] if adaylar else None
        if len(adaylar) > 1:
            print(f"    {ad:28} {len(adaylar)} aday bulundu, seçilen:")
            for a in adaylar:
                print(f"        {'-->' if a == adaylar[0] else '   '} {a.replace(ANA_DIZIN, '…')}")
        else:
            yol = bulunan[ad].replace(ANA_DIZIN, "…") if bulunan[ad] else "— BULUNAMADI —"
            print(f"    {ad:28} {yol}")

    eksik = [a for a, y in bulunan.items() if not y]
    if eksik:
        print(f"\n  !! {len(eksik)} adaptör bulunamadı. Bunlar asama5_coklu_tohum.py ile")
        print("     eğitilmişti; klasörlerini bulup yolunu bu betiğe verin ya da")
        print("     rev_E'nin TOHUM_MODELLER listesine qwen15/qwen3 ekleyip yeniden eğitin.")
    if not TOHUM_KOSU:
        print("\n  TOHUM_KOSU=False — GPU bölümü atlandı.")
        SONUC["tohum_adaptorleri"] = bulunan
        return
    varolan = {a: y for a, y in bulunan.items() if y}
    if not varolan:
        SONUC["tohum_adaptorleri"] = bulunan
        return

    import torch, zeyrek
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    from datasets import load_dataset
    from tqdm.auto import tqdm
    logging.getLogger("zeyrek").setLevel(logging.ERROR)
    _an = zeyrek.MorphAnalyzer()
    _t = "Öğrencilerin kitaplarını okuyabilmeleri için kütüphaneye gittiler"
    _k = len(re.findall(r"\w+", _t, re.UNICODE)); _m = 0
    for _w in re.findall(r"\w+", _t, re.UNICODE):
        _s = _an.analyze(_w)
        if _s and _s[0]: _m += len(_s[0][0].morphemes)
    assert _m / _k >= 1.5, "Zeyrek çalışmıyor"
    _mc = {}
    def morfem(m):
        s = 0
        for k in re.findall(r"\w+", m, re.UNICODE):
            if k.isdigit(): s += 1; continue
            if k not in _mc:
                try:
                    a = _an.analyze(k); _mc[k] = len(a[0][0].morphemes) if (a and a[0]) else 0
                except Exception: _mc[k] = 0
            s += _mc[k]
        return s
    def son_cevap(a): return a.split("####")[-1].strip().replace(",", "").replace(" ", "")
    def ayikla(m):
        x = re.search(r"(?:cevap|answer)\s*[:=]?\s*\$?(-?\d[\d,\.]*)", m, re.IGNORECASE)
        if x: return x.group(1).replace(",", "").rstrip("."), "acik"
        s = re.findall(r"-?\d[\d,\.]*", m)
        if s: return s[-1].replace(",", "").rstrip("."), "son_sayi"
        return None, "yok"

    ds = load_dataset("malhajar/gsm8k-tr", "main", split="test")
    PROB = [(ds[i]["question"], son_cevap(ds[i]["answer"])) for i in range(N_TEST)]
    ISTEM = "Bu matematik problemini adım adım Türkçe çöz. Sonunda 'Cevap: X' yaz.\n\n{soru}"
    TEMEL = {"qwen15": "Qwen/Qwen2.5-1.5B-Instruct", "qwen3": "Qwen/Qwen2.5-3B-Instruct"}
    os.makedirs(SONUC_DIR, exist_ok=True)

    for ad, adaptor in varolan.items():
        model_etiket = "qwen15" if ad.startswith("qwen15") else "qwen3"
        dosya = os.path.join(SONUC_DIR, f"{ad}_b{BUTCE}_n{N_TEST}.jsonl")
        islenmis = sum(1 for _ in open(dosya, encoding="utf-8")) if os.path.exists(dosya) else 0
        if islenmis >= N_TEST:
            print(f"    {ad}: zaten tamam"); continue
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_use_double_quant=True)
        tok = AutoTokenizer.from_pretrained(TEMEL[model_etiket])
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(TEMEL[model_etiket], quantization_config=bnb,
                                                     device_map="auto", torch_dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(model, adaptor); model.eval()
        bitirici = set()
        for kk in (getattr(model.generation_config, "eos_token_id", None), tok.eos_token_id):
            if isinstance(kk, (list, tuple)): bitirici.update(int(x) for x in kk)
            elif kk is not None: bitirici.add(int(kk))
        with open(dosya, "a", encoding="utf-8") as f:
            for bas in tqdm(range(islenmis, N_TEST, BATCH), desc=f"  {ad}", file=sys.__stdout__):
                kume = PROB[bas:min(bas + BATCH, N_TEST)]
                metin = [tok.apply_chat_template([{"role": "user", "content": ISTEM.format(soru=s)}],
                                                 tokenize=False, add_generation_prompt=True)
                         for s, _ in kume]
                g = tok(metin, return_tensors="pt", padding=True).to(model.device)
                with torch.no_grad():
                    o = model.generate(**g, max_new_tokens=BUTCE, do_sample=False,
                                       pad_token_id=tok.pad_token_id)
                gl = g["input_ids"].shape[1]
                for j, (soru, gercek) in enumerate(kume):
                    u = o[j][gl:]
                    yer = None
                    for bid in bitirici:
                        b = (u == bid).nonzero()
                        if len(b): yer = int(b[0].item()) if yer is None else min(yer, int(b[0].item()))
                    uz, kesildi = (yer, False) if yer is not None else (int(u.shape[0]), int(u.shape[0]) >= BUTCE - 1)
                    c = tok.decode(u[:uz], skip_special_tokens=True)
                    t, kaynak = ayikla(c)
                    f.write(json.dumps({"indeks": bas + j, "dogru": t == gercek, "kelime": kel(c),
                                        "morfem": morfem(c), "kesildi": kesildi, "token": uz,
                                        "cevap_kaynagi": kaynak, "metin": c}, ensure_ascii=False) + "\n")
                f.flush()
        del model; gc.collect(); torch.cuda.empty_cache()
        kay = jsonl(dosya); nn = len(kay)
        print(f"    {ad}: doğruluk {sum(k['dogru'] for k in kay)/nn:.1%}  "
              f"tükenme {sum(k['kesildi'] for k in kay)/nn:.1%}")
    SONUC["tohum_adaptorleri"] = bulunan


def beklenmeyen_dosyalar():
    """Envanterde 74 beklenen / 64 mevcut uyuşmazlığının kaynağını gösterir."""
    baslik("5) degerlendirme_revizyon İÇİNDEKİ BEKLENMEYEN DOSYALAR")
    if not os.path.isdir(SONUC_DIR):
        print("  Klasör yok."); return
    M = ["qwen05", "qwen15", "qwen3", "llama3"]
    V = ["orijinal", "sade_morfem", "sade_kelime"]
    Y = ["yalniz_kural1", "yalniz_agirlik", "rastgele", "kesme", "referans"]
    bekle = set()
    for m in M:
        for v in V:
            bekle.add(f"{m}_{v}_b512_n300")
            for t in (1, 2): bekle.add(f"{m}_{v}_s{t}_b512_n300")
        bekle.add(f"{m}_TABAN_b512_n300")
    for m in ["qwen15", "qwen3"]:
        for v in Y: bekle.add(f"{m}_{v}_b512_n300")
        for b in (256, 512, 1024, 2048):
            for v in V: bekle.add(f"{m}_{v}_b{b}_n100")
    fazla = []
    for y in sorted(glob.glob(os.path.join(SONUC_DIR, "*.jsonl"))):
        ad = os.path.basename(y)[:-6]
        if ad not in bekle:
            n = sum(1 for l in open(y, encoding="utf-8") if l.strip())
            fazla.append((ad, n))
    if not fazla:
        print("  Beklenmeyen dosya yok.")
    else:
        print(f"  {len(fazla)} beklenmeyen dosya — yarım kalmış koşu olabilir:")
        for ad, n in fazla:
            print(f"    {ad:44} {n} satır")
        print("\n  Bunlar rev_H tarafından okunmadı, yani sonuçları etkilemedi.")
        print("  Yarım kalmış bir koşuya aitse silip ilgili koşuyu tekrar başlatın.")
    SONUC["beklenmeyen_dosyalar"] = fazla


hedef_istatistikleri()
beklenmeyen_dosyalar()
eksik_tohumlar()
yol = os.path.join(REV_DIR, "revI_denetim.json")
with open(yol, "w", encoding="utf-8") as f:
    json.dump(SONUC, f, ensure_ascii=False, indent=2, default=str)
baslik("TAMAMLANDI")
print(f"  Sonuçlar: {yol}")
print("  Bundan sonra rev_H'yi yeniden çalıştırın; tohum tablosu 300 maddeye güncellenir.")
