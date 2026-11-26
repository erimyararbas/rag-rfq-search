"""Seeded generator for the synthetic RFQ corpus and the eval query set.

All data is fictional. Customer names are built from neutral word lists.
`family_id` is a hidden label used only by the retrieval eval.
"""

import datetime as dt
import math
import random
import re

from app.data.families import FAMILIES, LANGS, Family
from app.data.query_vocab import GENERIC_MATERIAL, QUERY_VOCAB

# ---------------------------------------------------------------- customers
# (adjective-like words, nouns) per language; noun -> industry label
_ADJ = {
    "en": ["Bluefield", "Alderwick", "Corvane", "Silvermoor", "Brightmoor", "Ashgrove", "Redwater",
           "Whitlock", "Fernhollow", "Stonebridge", "Marlowe", "Kestrel Hill"],
    "de": ["Nordhalm", "Veltmar", "Hartwen", "Falkenrath", "Tannfeld", "Brandtal", "Eschwiler",
           "Kornfeld", "Wiesenau", "Steinhagen"],
    "tr": ["Karaçam", "Yeşilvadi", "Kuzeyova", "Demirpınar", "Akçayır", "Güneyköy", "Çamlıbel",
           "Kırmızıdere", "Boztepe", "Taşdelen"],
}
_NOUN = {
    "en": {"Robotics": "robotics", "Fluidtech": "hydraulics", "Motion": "drive technology",
           "Packaging": "packaging machinery", "Medtech": "medical devices", "Energy": "energy",
           "Mobility": "automotive", "Automation": "industrial automation", "Systems": "machine building"},
    "de": {"Pumpen": "pumps", "Maschinenbau": "machine building", "Antriebstechnik": "drive technology",
           "Agrar": "agricultural machinery", "Fördertechnik": "material handling",
           "Automation": "industrial automation", "Hydraulik": "hydraulics"},
    "tr": {"Makina": "machine building", "Otomasyon": "industrial automation", "Hidrolik": "hydraulics",
           "Enerji": "energy", "Ambalaj": "packaging machinery", "Tarım": "agricultural machinery"},
}
_COUNTRY = {"en": ["GB", "NL", "SE", "US", "IE", "DK", "CA", "AU"], "de": ["DE", "DE", "DE", "AT", "CH"],
            "tr": ["TR"]}
_SUFFIX = {"DE": ["GmbH", "GmbH & Co. KG", "AG"], "AT": ["GmbH"], "CH": ["AG"], "GB": ["Ltd"], "IE": ["Ltd"],
           "NL": ["B.V."], "SE": ["AB"], "DK": ["A/S"], "US": ["Inc."], "CA": ["Inc."], "AU": ["Pty Ltd"],
           "TR": ["A.Ş.", "Ltd. Şti.", "San. ve Tic. A.Ş."]}
_LANG_SHARE = (("en", 0.60), ("de", 0.25))  # tr takes the rest

# ---------------------------------------------------------------- sentence templates
# fields: qty name material process feats tol stds demand weeks
_AND = {"en": "and", "de": "und", "tr": "ve"}

DESC = {
    "en": {
        "intro": ["We would like to request a quotation for {qty} {name}.",
                  "Please quote {qty} {name} for a new project.",
                  "We are looking for a supplier for {qty} {name} and ask for your best offer."],
        "matproc": ["The material is {material} and the parts are to be made by {process}.",
                    "Material: {material}. Manufacturing process: {process}.",
                    "Please plan for {process} in {material}."],
        "feat": ["Key features include {feats}.", "The part has {feats}."],
        "tol": ["Required tolerances: {tol}.", "Please hold {tol}."],
        "std": ["The parts must comply with {stds}.", "Relevant standards: {stds}."],
        "demand": ["The estimated yearly demand is {demand} pieces and repeat orders are likely.",
                   "We expect follow-up orders of about {demand} pieces per year."],
        "deadline": ["Please send your quotation within {weeks} weeks.",
                     "We need your offer within {weeks} weeks."],
        "docs": ["Drawing and 3D model are attached.",
                 "The attachment contains the drawing as PDF and a STEP file."],
    },
    "de": {
        "intro": ["Wir bitten um ein Angebot für {qty} {name}.",
                  "Für ein neues Projekt benötigen wir {qty} {name}; bitte senden Sie uns Ihr bestes Angebot.",
                  "Bitte kalkulieren Sie {qty} {name}."],
        "matproc": ["Als Werkstoff ist {material} vorgesehen; die Fertigung erfolgt durch {process}.",
                    "Werkstoff: {material}. Fertigungsverfahren: {process}.",
                    "Bitte kalkulieren Sie mit dem Werkstoff {material} und dem Verfahren {process}."],
        "feat": ["Das Bauteil weist {feats} auf.", "Wesentliche Merkmale sind {feats}."],
        "tol": ["Geforderte Toleranzen: {tol}.", "Einzuhalten sind {tol}."],
        "std": ["Die Teile müssen den Normen {stds} entsprechen.", "Maßgebliche Normen: {stds}."],
        "demand": ["Der geschätzte Jahresbedarf beträgt {demand} Stück, Folgeaufträge sind wahrscheinlich.",
                   "Wir rechnen mit Folgeaufträgen von etwa {demand} Stück pro Jahr."],
        "deadline": ["Bitte senden Sie uns Ihr Angebot innerhalb von {weeks} Wochen.",
                     "Wir benötigen Ihr Angebot innerhalb von {weeks} Wochen."],
        "docs": ["Zeichnung und 3D-Modell sind beigefügt.",
                 "Im Anhang finden Sie die Zeichnung als PDF und eine STEP-Datei."],
    },
    "tr": {
        "intro": ["{qty} adet {name} için teklif talep ediyoruz.",
                  "Yeni bir proje için {qty} adet {name} temin etmek istiyoruz; en iyi teklifinizi bekliyoruz.",
                  "Lütfen {qty} adet {name} için fiyat veriniz."],
        "matproc": ["Malzeme {material} olup üretim yöntemi {process} olarak belirlenmiştir.",
                    "Malzeme: {material}. Üretim yöntemi: {process}.",
                    "Lütfen {material} malzeme ve {process} esas alınarak fiyatlandırınız."],
        "feat": ["Parça üzerinde {feats} bulunmaktadır.", "Başlıca özellikler: {feats}."],
        "tol": ["İstenen toleranslar: {tol}.", "Parçalarda {tol} sağlanmalıdır."],
        "std": ["Parçalar {stds} standartlarına uygun olmalıdır.", "İlgili standartlar: {stds}."],
        "demand": ["Tahmini yıllık ihtiyaç {demand} adettir ve tekrarlayan siparişler beklenmektedir.",
                   "Yıllık yaklaşık {demand} adetlik devam siparişleri öngörüyoruz."],
        "deadline": ["Teklifinizi {weeks} hafta içinde iletmenizi rica ederiz.",
                     "Teklifinize {weeks} hafta içinde ihtiyacımız var."],
        "docs": ["Çizim ve 3B model ekte yer almaktadır.",
                 "Ekte PDF çizim ve STEP dosyası bulunmaktadır."],
    },
}
_OPTIONAL = ["feat", "tol", "std", "demand", "deadline", "docs"]

TITLES = {
    "en": ["{Name} – {process}", "RFQ: {name}, material {material}", "Quote request {name} ({material})"],
    "de": ["{Name} – {process}", "Anfrage: {Name} aus {material}", "Angebotsanfrage {Name} ({material})"],
    "tr": ["{Name} – {process}", "Teklif talebi: {name} ({material})", "{Name} – {material}"],
}

NOTES = {
    "en": [
        "All edges must be deburred and broken to 0.2 to 0.5 mm unless stated otherwise on the drawing.",
        "Please supply a material certificate with every batch.",
        "A first article inspection report is required with the initial delivery.",
        "Parts must be packed individually to avoid transport damage and arrive clean and free of oil.",
        "Tooling, fixtures and gauges should be quoted separately, together with the expected lead time for the first samples.",
        "The drawing revision attached to this request is binding; any deviation needs written approval before production starts.",
        "Partial deliveries are acceptable if the full quantity arrives within {weeks} weeks of the order.",
        "Please state your available capacity for a yearly volume of {demand} pieces.",
        "Critical dimensions are flagged on the drawing and will be checked on receipt.",
        "Quote according to {s1}; alternatives are welcome if they reduce cost without affecting function.",
    ],
    "de": [
        "Alle Kanten sind zu entgraten und mit 0,2 bis 0,5 mm zu brechen, sofern die Zeichnung nichts anderes vorgibt.",
        "Bitte legen Sie jeder Charge ein Materialzeugnis bei.",
        "Mit der ersten Lieferung ist ein Erstmusterprüfbericht vorzulegen.",
        "Die Teile sind einzeln zu verpacken, damit keine Transportschäden entstehen, und sauber sowie ölfrei zu liefern.",
        "Werkzeuge, Vorrichtungen und Lehren sind separat anzubieten, einschließlich der voraussichtlichen Lieferzeit für die ersten Muster.",
        "Der Zeichnungsstand im Anhang ist verbindlich; Abweichungen bedürfen vor Fertigungsbeginn einer schriftlichen Freigabe.",
        "Teillieferungen sind zulässig, wenn die Gesamtmenge innerhalb von {weeks} Wochen nach Bestellung eintrifft.",
        "Bitte geben Sie Ihre verfügbare Kapazität für ein Jahresvolumen von {demand} Stück an.",
        "Kritische Maße sind in der Zeichnung gekennzeichnet und werden im Wareneingang geprüft.",
        "Das Angebot soll nach {s1} erfolgen; Alternativen sind willkommen, wenn sie die Kosten senken, ohne die Funktion zu beeinträchtigen.",
    ],
    "tr": [
        "Aksi çizimde belirtilmedikçe tüm kenarlar çapaksız olmalı ve 0,2 ile 0,5 mm arasında kırılmalıdır.",
        "Lütfen her partiyle birlikte malzeme sertifikası gönderiniz.",
        "İlk teslimatla birlikte ilk numune muayene raporu sunulmalıdır.",
        "Parçalar taşıma hasarını önlemek için tek tek paketlenmeli, temiz ve yağsız teslim edilmelidir.",
        "Kalıp, aparat ve mastarlar ayrı fiyatlandırılmalı; ilk numunelerin tahmini teslim süresi de belirtilmelidir.",
        "Ekteki çizim revizyonu bağlayıcıdır; üretime başlamadan önce herhangi bir sapma için yazılı onay alınmalıdır.",
        "Siparişten itibaren {weeks} hafta içinde toplam adet teslim edilirse kısmi sevkiyat kabul edilir.",
        "Lütfen yıllık {demand} adetlik hacim için mevcut kapasitenizi belirtiniz.",
        "Kritik ölçüler çizimde işaretlenmiştir ve mal kabulde kontrol edilecektir.",
        "Teklif {s1} esas alınarak hazırlanmalıdır; fonksiyonu etkilemeden maliyeti düşüren alternatifler memnuniyetle karşılanır.",
    ],
}

# Separate wording for eval queries, so they do not echo the corpus templates.
QUERY_TPL = {
    "en": ["Looking for {name} in {material}, made by {process}.",
           "Any past requests for {name}? Need {feats}.",
           "Which customers asked for {name} produced by {process} before?",
           "We need a quote for {name} ({material}) with tolerances of {tol}."],
    "de": ["Wir suchen {name} aus {material}, gefertigt durch {process}.",
           "Gab es schon Anfragen für {name}? Gefordert sind {feats}.",
           "Welche Kunden haben bereits {name} im Verfahren {process} angefragt?",
           "Wir benötigen ein Angebot für {name} ({material}) mit Toleranzen von {tol}."],
    "tr": ["{material} malzemeden {process} ile üretilecek {name} için teklif istiyoruz.",
           "Daha önce {name} için talep gelmiş mi? {Feats} isteniyor.",
           "{process} ile üretilen {name} için hangi müşteriler talepte bulundu?",
           "{name} için {tol} toleransla teklif almak istiyoruz."],
}
_MATERIAL_WORDS = {w.lower() for v in GENERIC_MATERIAL.values() for w in v} | {"alu", "inox", "zamak"}
_QUERY_LANG_WEIGHTS = (("en", 0.35), ("de", 0.35), ("tr", 0.30))


# ---------------------------------------------------------------- helpers
def _join(items: list[str], lang: str) -> str:
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f" {_AND[lang]} " + items[-1]


def _cap(text: str, lang: str) -> str:
    if not text:
        return text
    first = text[0]
    if lang == "tr" and first in "iı":
        return ("İ" if first == "i" else "I") + text[1:]
    return first.upper() + text[1:]


def _fmt_int(n: int, lang: str) -> str:
    s = f"{n:,}"
    return s if lang == "en" else s.replace(",", ".")


def _loc_num(text: str, lang: str) -> str:
    """Decimal comma for DE/TR in language-neutral tolerance strings."""
    return text if lang == "en" else text.replace(".", ",")


def _qty(rng: random.Random, fam: Family) -> int:
    lo, hi = fam.qty_range
    q = int(math.exp(rng.uniform(math.log(lo), math.log(hi))))
    if q >= 100:
        digits = len(str(q)) - 2
        q = round(q, -digits) if digits > 0 else q
    return max(lo, min(hi, q))


def _weighted(rng: random.Random, pairs) -> str:
    return rng.choices([p[0] for p in pairs], weights=[p[1] for p in pairs])[0]


def _make_customers(rng: random.Random, counts: dict[str, int]) -> dict[str, list[dict]]:
    """Per language: customers, each with 1-15 RFQ slots summing to counts[lang]."""
    out: dict[str, list[dict]] = {}
    for lang in LANGS:
        combos = [(a, n) for a in _ADJ[lang] for n in _NOUN[lang]]
        rng.shuffle(combos)
        remaining, customers = counts[lang], []
        while remaining > 0:
            slots = rng.randint(1, min(15, remaining))
            adj, noun = combos[len(customers)]
            country = rng.choice(_COUNTRY[lang])
            customers.append({
                "customer": f"{adj} {noun} {rng.choice(_SUFFIX[country])}",
                "customer_industry": _NOUN[lang][noun],
                "customer_country": country,
                "slots": slots,
            })
            remaining -= slots
        out[lang] = customers
    return out


def _describe(rng: random.Random, fam: Family, lang: str, ctx: dict) -> str:
    tpl = DESC[lang]
    kinds = ["intro", "matproc"] + rng.sample(_OPTIONAL, rng.randint(0, 4))
    return " ".join(rng.choice(tpl[k]).format(**ctx) for k in kinds)


def _notes(rng: random.Random, lang: str, ctx: dict) -> str:
    n_par = rng.choices([0, 1, 2, 3], weights=[25, 45, 20, 10])[0]
    if n_par == 0:
        return ""
    pool = list(NOTES[lang])
    rng.shuffle(pool)
    paragraphs = []
    for _ in range(n_par):
        k = rng.randint(2, 3)
        sents, pool = pool[:k], pool[k:]
        paragraphs.append(" ".join(s.format(**ctx) for s in sents))
    return "\n\n".join(paragraphs)


def _context(rng: random.Random, fam: Family, lang: str, qty: int) -> dict:
    proc_idx = rng.randrange(len(fam.processes["en"]))
    material = rng.choice(fam.materials)
    feat_idx = sorted(rng.sample(range(4), rng.randint(2, 3)))
    tols = rng.sample(fam.tolerances, rng.randint(1, 2))
    stds = rng.sample(fam.standards, 2)
    name = rng.choice(fam.names[lang])
    return {
        "proc_idx": proc_idx,
        "material": material,
        "qty": _fmt_int(qty, lang),
        "name": name,
        "Name": _cap(name, lang),
        "process": fam.processes[lang][proc_idx],
        "feats": _join([fam.features[lang][i] for i in feat_idx], lang),
        "tol": _join([_loc_num(t, lang) for t in tols], lang),
        "stds": _join(stds, lang),
        "s1": stds[0],
        "demand": _fmt_int(qty * rng.randint(2, 6), lang),
        "weeks": rng.randint(2, 4),
    }


# ---------------------------------------------------------------- public API
def generate_rfqs(seed: int = 42, n: int = 400) -> list[dict]:
    rng = random.Random(seed)

    counts = {"en": round(n * _LANG_SHARE[0][1]), "de": round(n * _LANG_SHARE[1][1])}
    counts["tr"] = n - counts["en"] - counts["de"]
    customers = _make_customers(rng, counts)
    slots = [(lang, c) for lang in LANGS for c in customers[lang] for _ in range(c["slots"])]
    rng.shuffle(slots)

    weights = [rng.uniform(0.5, 2.0) for _ in FAMILIES]
    base = [f for f in FAMILIES for _ in range(5)] if n >= 5 * len(FAMILIES) else []
    assigned = base + rng.choices(FAMILIES, weights=weights, k=n - len(base))
    rng.shuffle(assigned)

    start = dt.date(2023, 1, 1).toordinal()
    end = dt.date(2025, 12, 31).toordinal()
    rows = []
    for (lang, cust), fam in zip(slots, assigned):
        date = dt.date.fromordinal(rng.randint(start, end)).isoformat()
        qty = _qty(rng, fam)
        ctx = _context(rng, fam, lang, qty)
        title = rng.choice(TITLES[lang]).format(**ctx)
        rows.append({
            "date": date,
            "customer": cust["customer"],
            "customer_industry": cust["customer_industry"],
            "customer_country": cust["customer_country"],
            "language": lang,
            "title": title,
            "description": _describe(rng, fam, lang, ctx),
            "technical_notes": _notes(rng, lang, ctx),
            "material": ctx["material"],
            "process": fam.processes["en"][ctx["proc_idx"]],
            "quantity": qty,
            "status": rng.choices(["quoted", "won", "lost"], weights=[35, 30, 35])[0],
            "family_id": fam.family_id,
        })

    rows.sort(key=lambda r: r["date"])  # stable: ties keep generation order
    return [{"rfq_id": f"RFQ-{i:04d}", **r} for i, r in enumerate(rows, start=1)]


def _stems(text: str) -> set[str]:
    """Rough 5-char word stems (len >= 4), to detect shared vocabulary."""
    return {w[:5] for w in re.findall(r"\w+", text.lower()) if len(w) >= 4}


def generate_queries(seed: int = 7, n: int = 80) -> list[dict]:
    """Eval queries: separate wording from the corpus (see query_vocab.py).

    ~70% of queries use buyer-style paraphrases for part name and features,
    ~50% use a generic material word instead of the grade, and a process
    phrase is dropped when the part name already contains its words.
    """
    rng = random.Random(seed)
    order = [f for f in FAMILIES for _ in range(math.ceil(n / len(FAMILIES)))]
    rng.shuffle(order)
    queries = []
    for i, fam in enumerate(order[:n], start=1):
        lang = _weighted(rng, _QUERY_LANG_WEIGHTS)
        li = LANGS.index(lang)
        alt_names, alt_feats = QUERY_VOCAB[fam.family_id]
        name = rng.choice(alt_names[li] if rng.random() < 0.7 else fam.names[lang])
        feat_pool = alt_feats[li] if rng.random() < 0.7 else fam.features[lang]
        material = rng.choice(fam.materials)
        if rng.random() < 0.5:
            material = GENERIC_MATERIAL.get(material, (material,) * 3)[li]
        process = fam.processes[lang][rng.randrange(len(fam.processes["en"]))]
        tol = next((t for t in fam.tolerances if re.fullmatch(r"±[\d.]+ mm", t)), None)
        redundant = set()  # placeholders that would only echo the part name or have no value
        if _stems(process) & _stems(name):
            redundant.add("{process}")
        if material.lower()[:4] in name.lower() or any(w in name.lower() for w in _MATERIAL_WORDS):
            redundant.add("{material}")  # name already says (or contradicts) the material
        if tol is None:
            redundant.add("{tol}")
        tpls = [t for t in QUERY_TPL[lang] if not any(r in t for r in redundant)]
        feats = _join(rng.sample(feat_pool, 2), lang)
        text = rng.choice(tpls).format(
            name=name,
            material=material,
            process=process,
            feats=feats,
            Feats=_cap(feats, lang),
            tol=_loc_num(tol or "", lang),
        )
        queries.append({
            "query_id": f"Q-{i:03d}",
            "language": lang,
            "text": _cap(text, lang),
            "family_id": fam.family_id,
        })
    return queries
