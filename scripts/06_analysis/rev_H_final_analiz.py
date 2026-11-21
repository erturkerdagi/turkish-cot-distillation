# ============================================================
#  REVİZYON H: SON ANALİZ — rev_F çıktılarını hakem maddelerine çevirir
#  Karşıladığı hakem maddeleri:
#    M2.1 M2.2 M2.4 R3-6 - Ablasyon ve temel yöntemlerin eğitim sonuçları
#    M3.1 - Çoklu bütçe eğrileri
#    M3.2 - Cevabın kaynağı: kaç doğru 'son sayı' yedeğinden geldi
#    M3.4 - Taban modeller
#    M4.1-M4.5 - Eşleştirilmiş testler, 4 model x 3 sürüm x 3 tohum
#    M5.5 R3-3 R3-4 - Tohum analizi ve tam ANOVA
#    L18 L19 - Düzeltilmiş Şekil 7 ve 8
#  GPU GEREKMEZ. Süre: ~2 dakika.
#  ÖN KOŞUL: rev_F çalıştırılmış olmalı.
# ============================================================

import subprocess, sys
for p in ["statsmodels", "pandas", "numpy", "scipy", "matplotlib"]:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", p],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

import os, re, io, json, glob, itertools, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.stats.contingency_tables import mcnemar

ANA_DIZIN = os.environ.get(
    "PROJE_DIZINI",
    "/content/drive/MyDrive/turkish-cot-distillation/Project",
)
ESKI_DEG  = os.path.join(ANA_DIZIN, "degerlendirme1000")
ESKI_TOH  = os.path.join(ANA_DIZIN, "degerlendirme_tohum")
YENI_DEG  = os.path.join(ANA_DIZIN, "degerlendirme_revizyon")
CIKTI_DIR = os.path.join(ANA_DIZIN, "revizyon_ciktilari")
SEKIL_DIR = os.path.join(CIKTI_DIR, "sekiller")

MODELLER = ["qwen05", "qwen15", "qwen3", "llama3"]
ETIKET   = {"qwen05": "Qwen2.5-0.5B", "qwen15": "Qwen2.5-1.5B",
            "qwen3": "Qwen2.5-3B", "llama3": "Llama-3.2-3B"}
ANA_SURUM = ["orijinal", "sade_morfem", "sade_kelime"]
YENI_SURUM = ["yalniz_kural1", "yalniz_agirlik", "rastgele", "kesme", "referans"]
SURUM_EN = {"orijinal": "Original", "sade_morfem": "Morpheme-simplified",
            "sade_kelime": "Word-simplified", "yalniz_kural1": "Rule 1 only",
            "yalniz_agirlik": "Weight filter only", "rastgele": "Random deletion",
            "kesme": "Uniform truncation", "referans": "gsm8k-tr reference",
            "TABAN": "Base (no fine-tuning)"}
DENKLIK_MARJI, BOOTSTRAP_N = 0.03, 10000
rng = np.random.default_rng(42)
os.makedirs(SEKIL_DIR, exist_ok=True)
SONUC = {}

KAYIT = os.path.join(CIKTI_DIR, "revH_calisma_kaydi.txt")
class Tee(io.TextIOBase):
    def __init__(s, y):
        s.t = sys.__stdout__; s.f = open(y, "w", encoding="utf-8")
    def write(s, x):
        s.t.write(x); s.f.write(x); s.f.flush(); return len(x)
sys.stdout = Tee(KAYIT)

def baslik(s):
    print("\n" + "=" * 78); print(s); print("=" * 78)

def jsonl(y):
    with open(y, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ============================================================
#  TÜM KOŞULARI YÜKLE
# ============================================================
def kosulari_yukle():
    """(model, surum, tohum, butce, n) -> kayıtlar"""
    veri = {}
    # Eski ana deney: {m}_{v}.jsonl, n=300, b=512
    for y in glob.glob(os.path.join(ESKI_DEG, "*.jsonl")):
        ad = os.path.basename(y)[:-6]
        for m in MODELLER:
            if ad.startswith(m + "_"):
                v = ad[len(m) + 1:]
                k = jsonl(y)
                veri[(m, v, 0, 512, len(k))] = k
                break
    # Eski tohumlar: {m}_{v}_s{t}.jsonl, n=200 (yenisi varsa o kullanılacak)
    for y in glob.glob(os.path.join(ESKI_TOH, "*.jsonl")):
        ad = os.path.basename(y)[:-6]
        mt = re.match(r"(.+)_s(\d+)$", ad)
        if not mt: continue
        govde, t = mt.group(1), int(mt.group(2))
        for m in MODELLER:
            if govde.startswith(m + "_"):
                k = jsonl(y)
                veri[(m, govde[len(m)+1:], t, 512, len(k))] = k
                break
    # Yeni: {m}_{v}_b{b}_n{n}.jsonl  (v sonunda _s{t} olabilir)
    for y in glob.glob(os.path.join(YENI_DEG, "*.jsonl")):
        mt = re.match(r"(.+)_b(\d+)_n(\d+)$", os.path.basename(y)[:-6])
        if not mt: continue
        govde, b, n = mt.group(1), int(mt.group(2)), int(mt.group(3))
        for m in MODELLER:
            if govde.startswith(m + "_"):
                v = govde[len(m) + 1:]
                t = 0
                ms = re.match(r"(.+)_s(\d+)$", v)
                if ms: v, t = ms.group(1), int(ms.group(2))
                k = jsonl(y)
                veri[(m, v, t, b, len(k))] = k
                break
    return veri


def al(veri, m, v, t=0, b=512, n=None):
    """Belirtilen koşuyu getirir; n verilmezse en büyük örneklemi seçer."""
    adaylar = [(kk, vv) for kk, vv in veri.items()
               if kk[0] == m and kk[1] == v and kk[2] == t and kk[3] == b
               and (n is None or kk[4] == n)]
    if not adaylar: return None
    return sorted(adaylar, key=lambda x: -x[0][4])[0][1]


def oran(k, alan): return sum(bool(x[alan]) for x in k) / len(k)
def ort(k, alan):  return sum(x.get(alan, 0) for x in k) / len(k)

def mcn(a, b, alan):
    n = min(len(a), len(b))
    x = [bool(a[i][alan]) for i in range(n)]; y = [bool(b[i][alan]) for i in range(n)]
    n01 = sum(1 for i in range(n) if x[i] and not y[i])
    n10 = sum(1 for i in range(n) if not x[i] and y[i])
    t = [[sum(1 for i in range(n) if x[i] and y[i]), n01],
         [n10, sum(1 for i in range(n) if not x[i] and not y[i])]]
    return float(mcnemar(t, exact=True).pvalue)

def boot(a, b, alan):
    n = min(len(a), len(b))
    x = np.array([float(bool(a[i][alan])) for i in range(n)])
    y = np.array([float(bool(b[i][alan])) for i in range(n)])
    idx = rng.integers(0, n, size=(BOOTSTRAP_N, n))
    f = x[idx].mean(axis=1) - y[idx].mean(axis=1)
    lo, hi = np.percentile(f, [2.5, 97.5])
    return float(x.mean()-y.mean()), float(lo), float(hi)


# ============================================================
#  1) ABLASYON VE TEMEL YÖNTEMLER
# ============================================================
def bolum_ablasyon(veri):
    baslik("1) ABLASYON VE TEMEL YÖNTEMLER — M2.1, M2.2, M2.4, R3-6")
    satir = []
    for m in ["qwen15", "qwen3"]:
        orij = al(veri, m, "orijinal")
        tam = al(veri, m, "sade_morfem")
        if not orij: continue
        print(f"\n  [{ETIKET[m]}]")
        print(f"    {'sürüm':22} {'doğruluk':>9} {'tükenme':>9} {'morfem':>9} "
              f"{'p(doğr. vs orij)':>17} {'p(tük. vs orij)':>16}")
        print("    " + "-" * 86)
        for v in ANA_SURUM + YENI_SURUM:
            k = al(veri, m, v)
            if not k: continue
            pd_ = mcn(orij, k, "dogru") if v != "orijinal" else float("nan")
            pk_ = mcn(orij, k, "kesildi") if v != "orijinal" else float("nan")
            print(f"    {SURUM_EN.get(v,v):22} {oran(k,'dogru'):>8.1%} "
                  f"{oran(k,'kesildi'):>8.1%} {ort(k,'morfem'):>9.1f} "
                  f"{pd_:>17.4f} {pk_:>16.4f}")
            satir.append({"model": m, "surum": v, "n": len(k),
                          "dogruluk": round(oran(k, "dogru"), 4),
                          "kesilme": round(oran(k, "kesildi"), 4),
                          "morfem": round(ort(k, "morfem"), 2),
                          "p_dogruluk_vs_orijinal": None if pd_ != pd_ else round(pd_, 5),
                          "p_kesilme_vs_orijinal": None if pk_ != pk_ else round(pk_, 5)})
        if tam:
            print(f"\n    Önerilen yönteme (morfem) karşı:")
            for v in YENI_SURUM:
                k = al(veri, m, v)
                if not k: continue
                f_, lo, hi = boot(tam, k, "kesildi")
                print(f"      {SURUM_EN[v]:22} tükenme farkı {f_:+.3f} "
                      f"[{lo:+.3f},{hi:+.3f}]  McNemar p={mcn(tam,k,'kesildi'):.4f}")
    print("\n  YORUM: rastgele silme ve düzgün kesme, önerilen yöntemle aynı cümle")
    print("  sayısına eşlenmişti. Tükenme azalmaları belirgin biçimde daha küçükse,")
    print("  kazanç 'hangi cümlenin atıldığından' geliyor demektir (M2.2).")
    SONUC["ablasyon_egitim"] = satir


# ============================================================
#  2) ÇOKLU BÜTÇE TARAMASI — M3.1
# ============================================================
def bolum_butce(veri):
    baslik("2) ÇOKLU BÜTÇE TARAMASI — M3.1")
    butceler = sorted({kk[3] for kk in veri})
    print(f"  Bulunan bütçeler: {butceler}")
    kayit = []
    for m in ["qwen15", "qwen3"]:
        bas = False
        for v in ANA_SURUM:
            for b in butceler:
                k = al(veri, m, v, b=b, n=100)
                if not k: continue
                if not bas:
                    print(f"\n  [{ETIKET[m]}]  (n=100)")
                    print(f"    {'sürüm':22} {'bütçe':>7} {'doğruluk':>9} "
                          f"{'tükenme':>9} {'ort.token':>10}")
                    print("    " + "-" * 60); bas = True
                print(f"    {SURUM_EN[v]:22} {b:>7} {oran(k,'dogru'):>8.1%} "
                      f"{oran(k,'kesildi'):>8.1%} {ort(k,'token'):>10.0f}")
                kayit.append({"model": m, "surum": v, "butce": b, "n": len(k),
                              "dogruluk": round(oran(k, "dogru"), 4),
                              "kesilme": round(oran(k, "kesildi"), 4),
                              "token": round(ort(k, "token"), 1)})
    if not kayit:
        print("  Bütçe taraması verisi bulunamadı (rev_F 5. blok çalıştı mı?)")
        return
    print("\n  YORUM: Bütçe büyüdükçe orijinal sürümün doğruluğu sadeleştirilmiş")
    print("  sürümlere yaklaşıyorsa, kazanç dar bütçeye özgüdür. Fark korunuyorsa")
    print("  yöntem gerçekten verimlilik sağlıyor demektir (M3.0'daki itiraz).")
    SONUC["M3.1_butce"] = kayit
    sekil_butce(kayit)


def sekil_butce(kayit):
    df = pd.DataFrame(kayit)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    renk = {"orijinal": "#444444", "sade_morfem": "#1f77b4", "sade_kelime": "#ff7f0e"}
    isaret = {"orijinal": "o", "sade_morfem": "s", "sade_kelime": "^"}
    for i, m in enumerate(["qwen15", "qwen3"]):
        for j, (alan, etiket) in enumerate([("dogruluk", "Accuracy"),
                                            ("kesilme", "Exhaustion rate")]):
            ax = axes[i][j]
            for v in ANA_SURUM:
                alt = df[(df.model == m) & (df.surum == v)].sort_values("butce")
                if alt.empty: continue
                ax.plot(alt["butce"], alt[alan] * 100, marker=isaret[v],
                        color=renk[v], linewidth=1.8, markersize=6,
                        label=SURUM_EN[v])
            ax.set_xscale("log", base=2)
            ax.set_xticks(sorted(df["butce"].unique()))
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.set_xlabel("Token budget")
            ax.set_ylabel(f"{etiket} (%)")
            ax.grid(alpha=0.3, linestyle=":")
            ax.text(0.03, 0.95, ETIKET[m], transform=ax.transAxes,
                    va="top", fontsize=10, fontweight="bold")
            if i == 0 and j == 0:
                ax.legend(frameon=False, fontsize=9, loc="lower right")
    plt.tight_layout()
    y = os.path.join(SEKIL_DIR, "Fig9_budget_sweep_EN.png")
    plt.savefig(y, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Şekil yazıldı: {y}")


# ============================================================
#  3) CEVABIN KAYNAĞI — M3.2
# ============================================================
def bolum_kaynak(veri):
    baslik("3) CEVABIN KAYNAĞI VE TÜKENME KIRILIMI — M3.2")
    var = any("cevap_kaynagi" in (k[0] if k else {}) for k in veri.values())
    if not var:
        print("  cevap_kaynagi alanı yok — yalnızca rev_F çıktılarında bulunur.")
    print(f"  {'model':13} {'sürüm':22} {'açık':>7} {'son-sayı':>9} {'yok':>6} "
          f"{'son-sayıdan doğru':>18}")
    print("  " + "-" * 80)
    kayit = []
    for m in MODELLER:
        for v in ANA_SURUM:
            k = al(veri, m, v)
            if not k or "cevap_kaynagi" not in k[0]: continue
            n = len(k)
            a = sum(1 for x in k if x["cevap_kaynagi"] == "acik")
            s = sum(1 for x in k if x["cevap_kaynagi"] == "son_sayi")
            y = n - a - s
            sd = sum(1 for x in k if x["cevap_kaynagi"] == "son_sayi" and x["dogru"])
            print(f"  {ETIKET[m]:13} {SURUM_EN[v]:22} {a/n:>6.1%} {s/n:>8.1%} "
                  f"{y/n:>5.1%} {sd:>8} ({sd/max(1,sum(x['dogru'] for x in k)):.1%} "
                  f"of doğrular)")
            kayit.append({"model": m, "surum": v, "n": n, "acik": a, "son_sayi": s,
                          "yok": y, "son_sayidan_dogru": sd,
                          "toplam_dogru": sum(x["dogru"] for x in k)})
    if kayit:
        print("\n  YORUM: 'son-sayıdan doğru' oranı düşükse, yedek kural sahte doğru")
        print("  üretmiyor demektir — hakemin M3.0'daki endişesi karşılanır.")
        SONUC["M3.2_kaynak"] = kayit


# ============================================================
#  4) TABAN MODELLER — M3.4
# ============================================================
def bolum_taban(veri):
    baslik("4) TABAN MODELLER — M3.4")
    print(f"  {'model':13} {'sürüm':22} {'doğruluk':>9} {'tükenme':>9} {'morfem':>9}")
    print("  " + "-" * 66)
    kayit = []
    for m in MODELLER:
        tb = al(veri, m, "TABAN") or al(veri, m, "taban")
        if tb:
            print(f"  {ETIKET[m]:13} {'Base (no fine-tuning)':22} "
                  f"{oran(tb,'dogru'):>8.1%} {oran(tb,'kesildi'):>8.1%} "
                  f"{ort(tb,'morfem'):>9.1f}")
            kayit.append({"model": m, "surum": "TABAN", "n": len(tb),
                          "dogruluk": round(oran(tb, "dogru"), 4),
                          "kesilme": round(oran(tb, "kesildi"), 4)})
        for v in ANA_SURUM:
            k = al(veri, m, v)
            if not k: continue
            print(f"  {'':13} {SURUM_EN[v]:22} {oran(k,'dogru'):>8.1%} "
                  f"{oran(k,'kesildi'):>8.1%} {ort(k,'morfem'):>9.1f}")
        print()
    if kayit:
        print("  YORUM: Damıtma taban modele göre fayda sağlıyor mu? Hakem R3'ün")
        print("  'temel yöntem yok' itirazının doğrudan cevabı (M3.4).")
        SONUC["M3.4_taban"] = kayit


# ============================================================
#  5) EŞLEŞTİRİLMİŞ TESTLER VE TOHUM ANALİZİ
# ============================================================
def bolum_istatistik(veri):
    baslik("5) EŞLEŞTİRİLMİŞ TESTLER — M4.1, M4.2, M4.3")
    satir = []
    for m in MODELLER:
        mv = [v for v in ANA_SURUM if al(veri, m, v)]
        if len(mv) < 2: continue
        print(f"\n  [{ETIKET[m]}]")
        print(f"    {'karşılaştırma':34} {'ölçüt':9} {'fark':>8} {'%95 GA':>18} "
              f"{'McNemar p':>11} {'en kötü düşüş':>14}")
        print("    " + "-" * 96)
        for v1, v2 in itertools.combinations(mv, 2):
            a, b = al(veri, m, v1), al(veri, m, v2)
            for alan, et in [("dogru", "doğruluk"), ("kesildi", "tükenme")]:
                f_, lo, hi = boot(a, b, alan); p = mcn(a, b, alan)
                kotu = f"{max(0,hi)*100:.1f} puan" if alan == "dogru" else ""
                print(f"    {v1+' - '+v2:34} {et:9} {f_:>+8.3f} "
                      f"[{lo:>+6.3f},{hi:>+6.3f}] {p:>11.4f} {kotu:>14}")
                satir.append({"model": m, "v1": v1, "v2": v2, "olcut": alan,
                              "fark": round(f_, 4), "ga_alt": round(lo, 4),
                              "ga_ust": round(hi, 4), "p": round(p, 6),
                              "marj_icinde": bool(lo > -DENKLIK_MARJI and hi < DENKLIK_MARJI)})
    print(f"\n  'En kötü düşüş' = %95 GA'nın üst sınırı: sadeleştirmenin yol")
    print(f"  açabileceği en büyük doğruluk kaybı. M4.3'te yazılacak ifade budur.")
    SONUC["M4_eslesmis"] = satir

    baslik("6) TOHUM ANALİZİ VE ANOVA — M4.4, M4.5, M5.5, R3-2, R3-3, R3-4")
    kay = []
    for (m, v, t, b, n), k in veri.items():
        if v not in ANA_SURUM or b != 512: continue
        kay.append({"model": m, "versiyon": v, "tohum": t, "n": n,
                    "dogruluk": oran(k, "dogru"), "kesilme": oran(k, "kesildi"),
                    "morfem": ort(k, "morfem")})
    df = pd.DataFrame(kay)
    if df.empty:
        print("  Veri yok."); return
    # aynı (model,versiyon,tohum) için en büyük n'i tut
    df = df.sort_values("n", ascending=False).drop_duplicates(
        ["model", "versiyon", "tohum"]).sort_values(["model", "versiyon", "tohum"])
    print(f"  {'model':13} {'sürüm':22} {'tohum':>6} {'n':>5} {'doğruluk':>9} "
          f"{'tükenme':>9} {'morfem':>9}")
    print("  " + "-" * 76)
    for _, r in df.iterrows():
        print(f"  {ETIKET[r['model']]:13} {SURUM_EN[r['versiyon']]:22} "
              f"{int(r['tohum']):>6} {int(r['n']):>5} {r['dogruluk']:>8.1%} "
              f"{r['kesilme']:>8.1%} {r['morfem']:>9.1f}")
    coklu = df.groupby(["model", "versiyon"]).filter(lambda g: len(g) > 1)
    if not coklu.empty:
        print("\n  Varyans ayrıştırması:")
        var = {}
        for o in ["dogruluk", "kesilme", "morfem"]:
            g = coklu.groupby(["model", "versiyon"])[o]
            ic, ar = g.var(ddof=1).mean(), g.mean().var(ddof=1)
            pay = ic/(ic+ar) if (ic+ar) else float("nan")
            print(f"    {o:10} tohum-içi={ic:.6f}  koşullar-arası={ar:.6f}  "
                  f"tohum payı={pay:.1%}")
            var[o] = {"tohum_ici": float(ic), "kosullar_arasi": float(ar),
                      "tohum_payi": float(pay)}
        SONUC["R3-3_varyans"] = var
    SONUC["tohumlar"] = df.to_dict("records")

    print("\n  Tam ANOVA:")
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    for o in ["dogruluk", "kesilme", "morfem"]:
        try:
            mm = ols(f"{o} ~ C(model) * C(versiyon)", data=df).fit()
            t = sm.stats.anova_lm(mm, typ=2)
        except Exception as e:
            print(f"    [{o}] kurulamadı: {e}"); continue
        kth = t.loc["Residual", "sum_sq"]
        print(f"\n    [{o}]")
        print(f"      {'kaynak':24} {'KT':>12} {'sd':>4} {'F':>9} {'p':>9} {'kısmi η²':>10}")
        print("      " + "-" * 72)
        kk = {}
        for s_ in t.index:
            kt, sd = t.loc[s_, "sum_sq"], t.loc[s_, "df"]
            F, p = t.loc[s_, "F"], t.loc[s_, "PR(>F)"]
            ke = kt/(kt+kth) if s_ != "Residual" else float("nan")
            print(f"      {s_:24} {kt:>12.4f} {sd:>4.0f} "
                  f"{(F if F == F else 0):>9.3f} {(p if p == p else 1):>9.4f} {ke:>10.3f}")
            kk[str(s_)] = {"KT": round(float(kt), 6), "sd": int(sd),
                           "F": round(float(F), 4) if F == F else None,
                           "p": round(float(p), 6) if p == p else None,
                           "kismi_eta2": round(float(ke), 4) if ke == ke else None}
        print(f"      R² = {mm.rsquared:.4f}   gözlem = {len(df)}")
        SONUC[f"R3-2_anova_{o}"] = {"kaynaklar": kk,
                                    "r_kare": round(float(mm.rsquared), 4),
                                    "gozlem": int(len(df))}
    sekil_tohum(df)


def sekil_tohum(df):
    """L18: ±SS çubuklarının üzerine tek tek tohum değerleri."""
    coklu = df.groupby(["model", "versiyon"]).filter(lambda g: len(g) > 1)
    if coklu.empty:
        print("  (Şekil 8 atlandı: çok tohumlu koşul yok)"); return
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
    renk = {"orijinal": "#444444", "sade_morfem": "#1f77b4", "sade_kelime": "#ff7f0e"}
    for ax, (alan, et) in zip(axes, [("dogruluk", "Accuracy (%)"),
                                     ("kesilme", "Exhaustion rate (%)"),
                                     ("morfem", "Morpheme load")]):
        modeller = sorted(coklu["model"].unique(), key=lambda x: MODELLER.index(x))
        genislik = 0.26
        for i, v in enumerate(ANA_SURUM):
            xs, ys, es = [], [], []
            for j, m in enumerate(modeller):
                g = coklu[(coklu.model == m) & (coklu.versiyon == v)][alan]
                if g.empty: continue
                carp = 100 if alan != "morfem" else 1
                xs.append(j + (i - 1) * genislik)
                ys.append(g.mean() * carp); es.append(g.std(ddof=1) * carp)
                ax.scatter([xs[-1]] * len(g), g * carp, color="black", s=14,
                           zorder=3, alpha=0.85)
            ax.bar(xs, ys, genislik * 0.9, yerr=es, capsize=3,
                   color=renk[v], alpha=0.75,
                   label=SURUM_EN[v] if alan == "dogruluk" else None)
        ax.set_xticks(range(len(modeller)))
        # dört model dar panele sığmıyor; eğik yazılmazsa etiketler üst üste biniyor
        ax.set_xticklabels([ETIKET[m] for m in modeller], fontsize=8.5,
                           rotation=18, ha="right")
        ax.set_ylabel(et); ax.grid(axis="y", alpha=0.3, linestyle=":")
        if alan == "dogruluk":
            ax.legend(frameon=False, fontsize=8, loc="upper left")
    plt.tight_layout()
    y = os.path.join(SEKIL_DIR, "Fig8_variance_EN_duzeltilmis.png")
    plt.savefig(y, dpi=300, bbox_inches="tight"); plt.close()
    print(f"\n  Şekil yazıldı (L18 — tek tek tohum noktaları eklendi): {y}")


def main():
    veri = kosulari_yukle()
    baslik("0) YÜKLENEN KOŞULAR")
    print(f"  Toplam: {len(veri)}")
    for kk in sorted(veri):
        print(f"    {kk[0]:8} {kk[1]:18} tohum={kk[2]} bütçe={kk[3]:>5} n={kk[4]}")
    if not veri:
        print("  Hiç dosya bulunamadı — klasör yollarını kontrol edin."); return

    for ad, fn in [("ablasyon", bolum_ablasyon), ("bütçe", bolum_butce),
                   ("kaynak", bolum_kaynak), ("taban", bolum_taban),
                   ("istatistik", bolum_istatistik)]:
        try:
            fn(veri)
        except Exception as e:
            import traceback
            print(f"\n  !! {ad} bölümünde hata: {e}")
            print(traceback.format_exc())

    y = os.path.join(CIKTI_DIR, "revH_sonuclar.json")
    with open(y, "w", encoding="utf-8") as f:
        json.dump(SONUC, f, ensure_ascii=False, indent=2, default=str)
    baslik("TAMAMLANDI")
    print(f"  Sonuçlar : {y}")
    print(f"  Kayıt    : {KAYIT}")
    print(f"  Şekiller : {SEKIL_DIR}")
    sys.stdout.flush()


main()
