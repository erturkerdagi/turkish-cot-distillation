# ============================================================
#  REVİZYON — İKİ DÜZELTME
#
#  SORUN 1: Bölüm 6 YANLIŞ değerlendirme klasöründe çalıştı.
#           Bulunan klasörde 9 dosya, n=100 ve llama3 yok — bu ESKİ
#           100 problemlik koşu. Makalenin raporladığı 300 problemlik,
#           4 modelli koşu başka bir klasörde. Bu betik tüm aday
#           klasörleri listeler, doğrusunu seçer ve istatistikleri
#           yeniden hesaplar.
#
#  SORUN 2: temel_surumleri.jsonl içinde 268/525 kaydın sade_referans
#           alanı boş kaldı (yerel asama1_ogretmen_verisi.jsonl yalnızca
#           500 problem içeriyor, oysa 1000 problem işlendi). Eksik
#           referans çözümler gsm8k-tr veri kümesinden tamamlanır.
#
#  GPU GEREKMEZ.  Süre: 2-5 dakika.
#  Mevcut hiçbir dosya değiştirilmez; çıktılar revizyon_ciktilari/ içine.
# ============================================================

import subprocess, sys
for p in ["datasets", "statsmodels", "pandas", "numpy", "scipy"]:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", p],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

import os, re, json, glob, itertools, numpy as np, pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

ANA_DIZIN = os.environ.get(
    "PROJE_DIZINI",
    "/content/drive/MyDrive/turkish-cot-distillation/Project",
)
CIKTI_DIR = os.path.join(ANA_DIZIN, "revizyon_ciktilari")

# Hangi klasörün kullanılacağı. Boş bırakılırsa en çok model×sürüm içeren
# klasör otomatik seçilir; ad verilirse o klasör nerede olursa olsun bulunur.
ANA_KLASOR_ADI   = "degerlendirme1000"
TOHUM_KLASOR_ADI = "degerlendirme_tohum"

MODELLER    = ["qwen05", "qwen15", "qwen3", "llama3"]
VERSIYONLAR = ["orijinal", "sade_morfem", "sade_kelime"]
DENKLIK_MARJI, BOOTSTRAP_N, TOHUM = 0.03, 10000, 42
rng = np.random.default_rng(TOHUM)
os.makedirs(CIKTI_DIR, exist_ok=True)
SONUC = {}


def baslik(s):
    print("\n" + "=" * 78); print(s); print("=" * 78)

def jsonl_oku(y):
    with open(y, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ============================================================
#  A) DOĞRU DEĞERLENDİRME KLASÖRÜNÜ BUL
# ============================================================
def klasorleri_tara():
    baslik("A) DEĞERLENDİRME KLASÖRLERİ — hepsi taranıyor")
    adaylar = {}
    for yol in glob.glob(os.path.join(ANA_DIZIN, "**", "*_orijinal*.jsonl"), recursive=True):
        if "revizyon_ciktilari" in yol:
            continue
        adaylar.setdefault(os.path.dirname(yol), []).append(yol)

    rapor = []
    for klasor in sorted(adaylar):
        dosyalar = [d for d in os.listdir(klasor) if d.endswith(".jsonl")]
        ciftler, nler, modeller = set(), [], set()
        tohumlu = 0
        for d in dosyalar:
            if re.search(r"_s\d+\.jsonl$", d):
                tohumlu += 1
                continue
            for m in MODELLER:
                for v in VERSIYONLAR:
                    if d.startswith(f"{m}_{v}"):
                        ciftler.add((m, v)); modeller.add(m)
                        try:
                            with open(os.path.join(klasor, d), encoding="utf-8") as f:
                                nler.append(sum(1 for _ in f))
                        except Exception:
                            pass
        if ciftler or tohumlu:
            rapor.append({"klasor": klasor, "cift": len(ciftler),
                          "modeller": sorted(modeller),
                          "n_ort": int(np.mean(nler)) if nler else 0,
                          "n_maks": max(nler) if nler else 0,
                          "tohumlu_dosya": tohumlu})

    for r in rapor:
        print(f"\n  {r['klasor']}")
        print(f"     model×versiyon çifti : {r['cift']}   modeller: {', '.join(r['modeller'])}")
        print(f"     madde sayısı (ort/maks): {r['n_ort']} / {r['n_maks']}"
              f"   tohum dosyası: {r['tohumlu_dosya']}")

    if not rapor:
        print("  Hiç değerlendirme klasörü bulunamadı.")
        return None, None, rapor

    def ada_gore(ad):
        if not ad:
            return None
        esler = [r for r in rapor if os.path.basename(r["klasor"]) == ad]
        return esler[0]["klasor"] if esler else None

    ana_k = ada_gore(ANA_KLASOR_ADI)
    if ana_k:
        ana = next(r for r in rapor if r["klasor"] == ana_k)
        print(f"\n  >> ANA DENEY (adıyla belirtildi): {ana_k}")
    else:
        if ANA_KLASOR_ADI:
            print(f"\n  !! '{ANA_KLASOR_ADI}' adlı klasör bulunamadı; otomatik seçime geçiliyor.")
        ana = sorted(rapor, key=lambda r: (-r["cift"], -r["n_maks"]))[0]
        print(f"\n  >> ANA DENEY (otomatik seçildi) : {ana['klasor']}")
    print(f"     ({ana['cift']} çift, n≈{ana['n_ort']}, modeller: {', '.join(ana['modeller'])})")
    if ana["cift"] < 12 or ana["n_maks"] < 300:
        print("\n  !! UYARI: Makale 4 model × 3 sürüm × 300 madde raporluyor.")
        print("     Seçilen klasör bununla uyuşmuyor — yukarıdaki listeden doğrusunu")
        print("     seçip ANA_KLASOR_ADI değişkenine yazın.")

    tk = ada_gore(TOHUM_KLASOR_ADI)
    if not tk:
        tohum_k = sorted(rapor, key=lambda r: -r["tohumlu_dosya"])[0]
        tk = tohum_k["klasor"] if tohum_k["tohumlu_dosya"] else None
    print(f"  >> TOHUM klasörü            : {tk if tk else '— yok —'}")
    return ana["klasor"], tk, rapor


# ============================================================
#  B) İSTATİSTİKLERİ DOĞRU KLASÖRLE YENİDEN HESAPLA
# ============================================================
def deg_bul(klasor, m, v, tohum=None):
    if not klasor:
        return None
    if tohum is None:
        a = [y for y in glob.glob(os.path.join(klasor, f"{m}_{v}*.jsonl"))
             if not re.search(r"_s\d+\.jsonl$", y)]
    else:
        a = glob.glob(os.path.join(klasor, f"{m}_{v}*_s{tohum}.jsonl"))
    return sorted(a, key=lambda y: -os.path.getsize(y))[0] if a else None


def istatistikler(ANA_K, TOHUM_K):
    baslik("B) EŞLEŞTİRİLMİŞ İSTATİSTİKLER — M3.2, M4.1-M4.4, R3-2, R3-3")
    veri = {}
    for m in MODELLER:
        for v in VERSIYONLAR:
            p = deg_bul(ANA_K, m, v)
            if p: veri[(m, v, 0)] = jsonl_oku(p)
            for s in (1, 2, 3):
                p = deg_bul(TOHUM_K, m, v, s) or deg_bul(ANA_K, m, v, s)
                if p: veri[(m, v, s)] = jsonl_oku(p)
    print(f"  Yüklenen koşu: {len(veri)}")
    for a in sorted(veri):
        print(f"    {a[0]:8} {a[1]:13} tohum {a[2]}  n={len(veri[a])}")
    if not veri:
        return

    def mcn(a, b, alan):
        n = min(len(a), len(b))
        x = [bool(a[i][alan]) for i in range(n)]; y = [bool(b[i][alan]) for i in range(n)]
        n01 = sum(1 for i in range(n) if x[i] and not y[i])
        n10 = sum(1 for i in range(n) if not x[i] and y[i])
        t = [[sum(1 for i in range(n) if x[i] and y[i]), n01],
             [n10, sum(1 for i in range(n) if not x[i] and not y[i])]]
        return {"n": n, "sadece_A": n01, "sadece_B": n10,
                "p": float(mcnemar(t, exact=True).pvalue)}

    def boot(a, b, alan):
        n = min(len(a), len(b))
        x = np.array([float(bool(a[i][alan])) for i in range(n)])
        y = np.array([float(bool(b[i][alan])) for i in range(n)])
        idx = rng.integers(0, n, size=(BOOTSTRAP_N, n))
        f = x[idx].mean(axis=1) - y[idx].mean(axis=1)
        lo, hi = np.percentile(f, [2.5, 97.5])
        return {"fark": round(float(x.mean()-y.mean()), 4),
                "ga_alt": round(float(lo), 4), "ga_ust": round(float(hi), 4),
                "marj_icinde": bool(lo > -DENKLIK_MARJI and hi < DENKLIK_MARJI)}

    print("\n  >> M4.1 / M4.2 / M4.3 — Eşleştirilmiş testler")
    satir = []
    for m in MODELLER:
        mv = [v for v in VERSIYONLAR if (m, v, 0) in veri]
        if len(mv) < 2: continue
        print(f"\n    [{m}]")
        print(f"      {'karşılaştırma':30} {'ölçüt':9} {'fark':>8} {'%95 GA':>18} "
              f"{'McNemar p':>11} {'denk?':>7}")
        print("      " + "-" * 88)
        for v1, v2 in itertools.combinations(mv, 2):
            a, b = veri[(m, v1, 0)], veri[(m, v2, 0)]
            for alan, et in [("dogru", "doğruluk"), ("kesildi", "tükenme")]:
                mc, bs = mcn(a, b, alan), boot(a, b, alan)
                dk = ("evet" if bs["marj_icinde"] else "hayır") if alan == "dogru" else ""
                print(f"      {v1+' - '+v2:30} {et:9} {bs['fark']:>+8.3f} "
                      f"[{bs['ga_alt']:>+6.3f},{bs['ga_ust']:>+6.3f}] "
                      f"{mc['p']:>11.4f} {dk:>7}")
                satir.append({"model": m, "v1": v1, "v2": v2, "olcut": alan, **mc, **bs})
    SONUC["M4_eslesmis"] = satir

    print("\n  >> M3.2 — Tükenen / tükenmeyen çıktılarda doğruluk")
    print(f"    {'model':8} {'versiyon':13} {'tük.n':>7} {'tük.doğr':>9} "
          f"{'tükenmeyen n':>13} {'tükenmeyen doğr':>16}")
    print("    " + "-" * 70)
    m32 = []
    for m in MODELLER:
        for v in VERSIYONLAR:
            k = veri.get((m, v, 0))
            if not k: continue
            t = [x for x in k if x["kesildi"]]; nn = [x for x in k if not x["kesildi"]]
            dt = sum(x["dogru"] for x in t)/len(t) if t else float("nan")
            dn = sum(x["dogru"] for x in nn)/len(nn) if nn else float("nan")
            print(f"    {m:8} {v:13} {len(t):>7} {dt:>8.1%} {len(nn):>13} {dn:>15.1%}")
            m32.append({"model": m, "versiyon": v, "tukenen_n": len(t),
                        "tukenen_dogruluk": round(dt, 4) if t else None,
                        "tukenmeyen_n": len(nn),
                        "tukenmeyen_dogruluk": round(dn, 4) if nn else None})
    SONUC["M3.2"] = m32

    print("\n  >> M4.4 / R3-3 — Tohum değerleri ve varyans")
    kay = [{"model": m, "versiyon": v, "tohum": s, "n": len(k),
            "dogruluk": sum(x["dogru"] for x in k)/len(k),
            "kesilme": sum(x["kesildi"] for x in k)/len(k),
            "morfem": sum(x["morfem"] for x in k)/len(k)}
           for (m, v, s), k in veri.items()]
    df = pd.DataFrame(kay)
    print(f"    {'model':8} {'versiyon':13} {'tohum':>6} {'n':>5} {'doğruluk':>9} "
          f"{'tükenme':>9} {'morfem':>9}")
    print("    " + "-" * 64)
    for _, r in df.sort_values(["model", "versiyon", "tohum"]).iterrows():
        print(f"    {r['model']:8} {r['versiyon']:13} {int(r['tohum']):>6} {int(r['n']):>5} "
              f"{r['dogruluk']:>8.1%} {r['kesilme']:>8.1%} {r['morfem']:>9.1f}")
    coklu = df.groupby(["model", "versiyon"]).filter(lambda g: len(g) > 1)
    if not coklu.empty:
        print("\n    Varyans ayrıştırması:")
        var = {}
        for o in ["dogruluk", "kesilme", "morfem"]:
            g = coklu.groupby(["model", "versiyon"])[o]
            ic, ar = g.var(ddof=1).mean(), g.mean().var(ddof=1)
            pay = ic/(ic+ar) if (ic+ar) else float("nan")
            print(f"      {o:10} tohum-içi={ic:.6f}  koşullar-arası={ar:.6f}  "
                  f"tohum payı={pay:.1%}")
            var[o] = {"tohum_ici": float(ic), "kosullar_arasi": float(ar),
                      "tohum_payi": float(pay)}
        SONUC["R3-3_varyans"] = var
    SONUC["M4.4_tohumlar"] = df.to_dict("records")

    print("\n  >> R3-2 — Tam ANOVA")
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    for o in ["dogruluk", "kesilme", "morfem"]:
        print(f"\n    [{o}]")
        try:
            mm = ols(f"{o} ~ C(model) * C(versiyon)", data=df).fit()
            t = sm.stats.anova_lm(mm, typ=2)
        except Exception as e:
            print(f"      kurulamadı: {e}"); continue
        ktt, kth = t["sum_sq"].sum(), t.loc["Residual", "sum_sq"]
        print(f"      {'kaynak':24} {'KT':>12} {'sd':>4} {'F':>9} {'p':>9} {'kısmi eta²':>11}")
        print("      " + "-" * 73)
        kay2 = {}
        for kk in t.index:
            kt, sd = t.loc[kk, "sum_sq"], t.loc[kk, "df"]
            F, p = t.loc[kk, "F"], t.loc[kk, "PR(>F)"]
            ke = kt/(kt+kth) if kk != "Residual" else float("nan")
            print(f"      {kk:24} {kt:>12.4f} {sd:>4.0f} "
                  f"{(F if F == F else 0):>9.3f} {(p if p == p else 1):>9.4f} {ke:>11.3f}")
            kay2[str(kk)] = {"KT": round(float(kt), 6), "sd": int(sd),
                             "F": round(float(F), 4) if F == F else None,
                             "p": round(float(p), 6) if p == p else None,
                             "kismi_eta2": round(float(ke), 4) if ke == ke else None}
        print(f"      Model R² = {mm.rsquared:.4f}   gözlem = {len(df)}")
        SONUC[f"R3-2_anova_{o}"] = {"kaynaklar": kay2,
                                    "r_kare": round(float(mm.rsquared), 4),
                                    "gozlem": int(len(df))}


# ============================================================
#  C) sade_referans ALANINI TAMAMLA
# ============================================================
def referans_onar():
    baslik("C) sade_referans ONARIMI — eksik gsm8k-tr referans çözümleri")
    yol = os.path.join(CIKTI_DIR, "temel_surumleri.jsonl")
    if not os.path.exists(yol):
        print(f"  Bulunamadı: {yol}"); return
    kayitlar = jsonl_oku(yol)
    eksik = sum(1 for k in kayitlar if k["sade_referans"] == k["orijinal"])
    print(f"  Kayıt: {len(kayitlar)}   eksik referans: {eksik}")
    if not eksik:
        print("  Eksik yok."); return

    def nrm(s): return re.sub(r"\s+", " ", s).strip()
    ref = {}
    try:
        from datasets import load_dataset
        for bol in ["train", "test"]:
            try:
                ds = load_dataset("malhajar/gsm8k-tr", "main", split=bol)
            except Exception:
                continue
            for i in range(len(ds)):
                ref[nrm(ds[i]["question"])] = ds[i]["answer"]
        print(f"  gsm8k-tr veri kümesinden alınan referans: {len(ref)}")
    except Exception as e:
        print(f"  !! Veri kümesi indirilemedi: {e}"); return
    if not ref:
        print("  !! Referans alınamadı."); return

    def temizle(c):
        c = re.sub(r"<<[^>]*>>", "", c).split("####")[0]
        return re.sub(r"\s+", " ", c).strip()

    onarilan, hala_eksik, cevapsiz = 0, 0, 0
    for k in kayitlar:
        if k["sade_referans"] != k["orijinal"]:
            continue
        c = ref.get(nrm(k["soru"]))
        if not c:
            hala_eksik += 1; continue
        rc = temizle(c)
        if k["gercek_cevap"] not in rc:
            rc += f" Cevap: {k['gercek_cevap']}"; cevapsiz += 1
        k["sade_referans"] = rc
        k["sade_referans_kelime"] = len(re.findall(r"\w+", rc, re.UNICODE))
        k["sade_referans_morfem"] = None      # morfem yükü rev_TUMU'da yeniden hesaplanır
        onarilan += 1

    print(f"  Onarılan: {onarilan}   hâlâ eksik: {hala_eksik}   cevabı eklenen: {cevapsiz}")
    yeni = os.path.join(CIKTI_DIR, "temel_surumleri_onarilmis.jsonl")
    with open(yeni, "w", encoding="utf-8") as f:
        for k in kayitlar:
            f.write(json.dumps(k, ensure_ascii=False) + "\n")
    print(f"  Yazıldı: {yeni}")
    kel = [k["sade_referans_kelime"] for k in kayitlar if k.get("sade_referans_kelime")]
    if kel:
        print(f"  Referans sürümü ortalama uzunluk: {np.mean(kel):.1f} sözcük")
    SONUC["referans_onarim"] = {"onarilan": onarilan, "hala_eksik": hala_eksik,
                                "cevapsiz": cevapsiz, "toplam": len(kayitlar)}


ANA_K, TOHUM_K, rapor = klasorleri_tara()
SONUC["klasor_taramasi"] = rapor
if ANA_K:
    istatistikler(ANA_K, TOHUM_K)
referans_onar()

yol = os.path.join(CIKTI_DIR, "duzeltme_sonuclari.json")
with open(yol, "w", encoding="utf-8") as f:
    json.dump(SONUC, f, ensure_ascii=False, indent=2, default=str)
baslik("TAMAMLANDI")
print(f"  Sonuçlar: {yol}")
print("  Mevcut dosyaların hiçbiri değiştirilmedi.")
