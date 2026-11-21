# ============================================================
#  REVİZYON ENVANTERİ — ne var, ne eksik, hangi dosya kullanıldı
#  GPU GEREKMEZ. Süre: 1-2 dakika.
#  Hiçbir dosyaya yazmaz, sadece sayar ve raporlar.
#
#  1000 / 525 / 300 NE DEMEK (karışıklık olmasın diye)
#    1000 : öğretmen modeline verilen EĞİTİM problemi (gsm8k-tr train)
#     525 : doğruluk + Türkçe filtrelerinden geçip eğitim kümesi olan zincir
#     300 : değerlendirmede kullanılan TEST problemi (gsm8k-tr test)
#     200 : eski tohum koşularının test problemi sayısı — hakem bunun 300
#           olmasını istiyor (M4.5), düzeltilmesi gereken tek yer burası
#     100 : bütçe taramasının örneklemi (hakem küçük örnekleme izin verdi)
# ============================================================

import os, re, json, glob
from collections import defaultdict

ANA_DIZIN = os.environ.get(
    "PROJE_DIZINI",
    "/content/drive/MyDrive/turkish-cot-distillation/Project",
)

MODELLER = ["qwen05", "qwen15", "qwen3", "llama3"]
ANA_SURUM = ["orijinal", "sade_morfem", "sade_kelime"]
YENI_SURUM = ["yalniz_kural1", "yalniz_agirlik", "rastgele", "kesme", "referans"]

def baslik(s):
    print("\n" + "=" * 78); print(s); print("=" * 78)

def say_satir(y):
    try:
        with open(y, encoding="utf-8") as f:
            return sum(1 for l in f if l.strip())
    except Exception:
        return -1

def adaptor_mu(k):
    return os.path.isdir(k) and (os.path.exists(os.path.join(k, "adapter_config.json"))
                                 or bool(glob.glob(os.path.join(k, "adapter_model*"))))

eksikler = []

# ============================================================
baslik("1) EĞİTİM VERİSİ DOSYALARI")
veri = {
    "Öğretmen zincirleri (beklenen 1000 satır)": "asama1_ogretmen_zincirleri*.jsonl",
    "Referans çözümler":                          "asama1_ogretmen_verisi*.jsonl",
    "Ana sadeleştirme (beklenen 525)":            "asama2_sadelesmis*.jsonl",
    "Keşif kayıtları (beklenen 209)":             "kesif_kayitlar*.json",
    "Ablasyon sürümleri (beklenen 525)":          "ablasyon_surumleri.jsonl",
    "Temel yöntemler, onarılmış (beklenen 525)":  "temel_surumleri_onarilmis.jsonl",
    "Temel yöntemler, onarılmamış":               "temel_surumleri.jsonl",
}
for ad, desen in veri.items():
    bulunan = [y for y in glob.glob(os.path.join(ANA_DIZIN, "**", desen), recursive=True)]
    if not bulunan:
        print(f"  {'YOK':>6}  {ad}")
        if "onarılmamış" not in ad:
            eksikler.append(f"Veri dosyası bulunamadı: {desen}")
        continue
    for y in sorted(bulunan, key=lambda x: -os.path.getsize(x)):
        n = say_satir(y) if y.endswith(".jsonl") else "-"
        if y.endswith(".json"):
            try:
                n = len(json.load(open(y, encoding="utf-8")))
            except Exception:
                n = "?"
        print(f"  {str(n):>6}  {ad}")
        print(f"          {y.replace(ANA_DIZIN, '…')}")
        break

# hangi alanlar var?
for desen, beklenen in [("ablasyon_surumleri.jsonl",
                         ["sade_tam", "sade_yalniz_kural1", "sade_yalniz_agirlik"]),
                        ("temel_surumleri_onarilmis.jsonl",
                         ["sade_rastgele", "sade_kesme", "sade_referans"])]:
    b = glob.glob(os.path.join(ANA_DIZIN, "**", desen), recursive=True)
    if not b: continue
    r = json.loads(open(b[0], encoding="utf-8").readline())
    yok = [a for a in beklenen if a not in r]
    print(f"\n  {desen} alanları: {'TAM' if not yok else 'EKSİK -> ' + ', '.join(yok)}")
    if yok: eksikler.append(f"{desen} içinde eksik alan: {yok}")
    if "sade_referans" in r:
        rows = [json.loads(l) for l in open(b[0], encoding="utf-8")]
        bos = sum(1 for x in rows if x["sade_referans"] == x["orijinal"])
        print(f"  sade_referans onarımı: {len(rows)-bos}/{len(rows)} dolu"
              f"{'  (ONARILMAMIŞ!)' if bos > 50 else '  (iyi)'}")
        if bos > 50: eksikler.append("temel_surumleri onarılmamış sürüm kullanılmış olabilir")

# ============================================================
baslik("2) ADAPTÖRLER (eğitilmiş modeller)")
tum_adaptorler = {}
for kok in glob.glob(os.path.join(ANA_DIZIN, "*")):
    if not os.path.isdir(kok): continue
    for alt in glob.glob(os.path.join(kok, "**"), recursive=True):
        if adaptor_mu(alt):
            tum_adaptorler[os.path.basename(alt)] = alt
print(f"  Toplam adaptör klasörü: {len(tum_adaptorler)}\n")
gruplar = defaultdict(list)
for ad in sorted(tum_adaptorler):
    kok = os.path.dirname(tum_adaptorler[ad]).replace(ANA_DIZIN, "…")
    gruplar[kok].append(ad)
for kok, adlar in gruplar.items():
    print(f"  {kok}  ({len(adlar)} adet)")
    for a in adlar: print(f"      {a}")

print("\n  --- Makalenin ihtiyaç duyduğu adaptörler ---")
gerekli = []
for m in MODELLER:
    for v in ANA_SURUM:
        gerekli.append((f"{m}_{v}", "ana deney"))
for m in ["qwen15", "qwen3"]:
    for v in YENI_SURUM:
        gerekli.append((f"{m}_{v}", "ablasyon/temel (M2, R3-6)"))
for m in MODELLER:
    for v in ANA_SURUM:
        for t in (1, 2):
            gerekli.append((f"{m}_{v}_s{t}", "tohum (M4.5, R3-4)"))
yok_liste = []
for ad, amac in gerekli:
    if ad not in tum_adaptorler:
        yok_liste.append((ad, amac))
if yok_liste:
    print(f"  BULUNAMAYAN: {len(yok_liste)}/{len(gerekli)}")
    for ad, amac in yok_liste:
        print(f"      {ad:32} ({amac})")
        eksikler.append(f"Adaptör yok: {ad} — {amac}")
else:
    print(f"  Hepsi var ({len(gerekli)}/{len(gerekli)})")

# ============================================================
baslik("3) DEĞERLENDİRME DOSYALARI")
def klasor_say(ad):
    k = None
    for y in glob.glob(os.path.join(ANA_DIZIN, "**", ad), recursive=True):
        if os.path.isdir(y): k = y; break
    if not k:
        print(f"  {ad}: KLASÖR YOK"); return None, {}
    dosyalar = {}
    for y in glob.glob(os.path.join(k, "*.jsonl")):
        dosyalar[os.path.basename(y)[:-6]] = say_satir(y)
    print(f"  {ad}: {len(dosyalar)} dosya")
    return k, dosyalar

_, eski = klasor_say("degerlendirme1000")
_, eski_toh = klasor_say("degerlendirme_tohum")
_, yeni = klasor_say("degerlendirme_revizyon")

print("\n  --- degerlendirme_revizyon beklenen içerik ---")
bekle = []
for m in MODELLER:
    for v in ANA_SURUM:
        bekle.append((f"{m}_{v}_b512_n300", "ana deney, ayrıntılı kayıt (M3.2)"))
for m in ["qwen15", "qwen3"]:
    for v in YENI_SURUM:
        bekle.append((f"{m}_{v}_b512_n300", "ablasyon/temel (M2, R3-6)"))
for m in MODELLER:
    for v in ANA_SURUM:
        for t in (1, 2):
            bekle.append((f"{m}_{v}_s{t}_b512_n300", "tohum 300 madde (M4.5)"))
for m in MODELLER:
    bekle.append((f"{m}_TABAN_b512_n300", "taban model (M3.4)"))
for b in (256, 512, 1024, 2048):
    for m in ["qwen15", "qwen3"]:
        for v in ANA_SURUM:
            bekle.append((f"{m}_{v}_b{b}_n100", f"bütçe taraması {b} (M3.1)"))

var = 0; eksik_deg = []
for ad, amac in bekle:
    if ad in yeni and yeni[ad] > 0:
        var += 1
    else:
        eksik_deg.append((ad, amac))
print(f"  Beklenen {len(bekle)} | var {var} | eksik {len(eksik_deg)}")
if eksik_deg:
    print("\n  EKSİK DEĞERLENDİRME DOSYALARI:")
    grup = defaultdict(list)
    for ad, amac in eksik_deg: grup[amac].append(ad)
    for amac, adlar in grup.items():
        print(f"    [{amac}]  {len(adlar)} dosya")
        for a in adlar[:6]: print(f"        {a}")
        if len(adlar) > 6: print(f"        … ve {len(adlar)-6} tane daha")
        eksikler.append(f"Değerlendirme eksik: {amac} ({len(adlar)} dosya)")

# satır sayısı kontrolü
print("\n  --- satır sayısı beklenenle uyuşmayanlar ---")
hata = 0
for ad, n in sorted(yeni.items()):
    m = re.search(r"_n(\d+)$", ad)
    if m and n != int(m.group(1)):
        print(f"    {ad}: {n} satır (beklenen {m.group(1)})"); hata += 1
        eksikler.append(f"Eksik satır: {ad} ({n}/{m.group(1)})")
if not hata: print("    yok — hepsi tam")

# alan kontrolü
print("\n  --- ayrıntılı kayıt alanları ---")
if yeni:
    ornek = sorted(yeni)[0]
    k = [y for y in glob.glob(os.path.join(ANA_DIZIN, "**", "degerlendirme_revizyon", "*.jsonl"),
                              recursive=True)][0]
    r = json.loads(open(k, encoding="utf-8").readline())
    gerek = ["dogru", "kelime", "morfem", "kesildi", "token", "cevap_kaynagi"]
    yok = [a for a in gerek if a not in r]
    print(f"    {'TAM' if not yok else 'EKSİK -> ' + str(yok)}   (alanlar: {list(r.keys())})")
    if yok: eksikler.append(f"Ayrıntılı kayıt alanı eksik: {yok}")

# ============================================================
baslik("4) ÖZET — YAPILMASI GEREKENLER")
if not eksikler:
    print("  Eksik bulunamadı. Tüm koşular tamam.")
else:
    for i, e in enumerate(eksikler, 1):
        print(f"  {i:2}. {e}")
print()
print("  NOT: 'tohum 300 madde' eksikse, iki seçenek var:")
print("    (a) Eski tohum adaptörleri bulunduysa rev_I_denetim.py bunları 300 madde")
print("        üzerinde yeniden değerlendirir (~1,5 saat GPU, eğitim gerekmez).")
print("    (b) Adaptörler bulunamadıysa rev_E_egitim.py içindeki TOHUM_MODELLER")
print("        satırını  [\"qwen05\", \"llama3\", \"qwen15\", \"qwen3\"]  yapıp yeniden")
print("        çalıştırın; tamamlananları atlar, yalnızca eksikleri eğitir.")
print()
print("  Bu raporu olduğu gibi paylaşabilirsiniz.")
