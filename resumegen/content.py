"""Generate one realistic, internally-consistent resume (structured content).

Every value here is the *display string* that will be printed on the page,
plus normalised metadata (ISO dates, E.164 phone) used in the answer key.
"""
import random
import re
import unicodedata

from .data_regions import REGIONS, NANP_AREA, RELOCATION
from .data_degrees import DEG, FAMILY_UNI_TYPE, GRADES, EU_GRADES_BY_COUNTRY, INDIA_SCHOOL
from .data_industries import INDUSTRIES, GENERIC_LEAD, GENERIC_SOFT, GENERIC_AWARDS

NOW_Y, NOW_M = 2026, 10
REGION_BY_KEY = {r["key"]: r for r in REGIONS}
REGION_BY_CC = {}
for _r in REGIONS:
    REGION_BY_CC.setdefault(_r["cc"], []).append(_r)

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

GLOBAL_COMPANIES = [
    ("Google", {"software", "data", "design", "marketing", "sales"}), ("Microsoft", {"software", "data", "sales", "marketing", "design"}),
    ("Amazon", {"software", "data", "supply_chain", "hr", "finance"}), ("IBM", {"software", "data", "consulting", "sales"}),
    ("Oracle", {"software", "sales", "data"}), ("SAP", {"software", "consulting", "sales"}), ("Cisco", {"software", "electrical", "sales"}),
    ("Salesforce", {"software", "sales", "marketing"}), ("Accenture", {"software", "consulting", "data", "hr"}), ("Deloitte", {"consulting", "accounting", "data", "finance"}),
    ("PwC", {"accounting", "consulting", "finance"}), ("EY", {"accounting", "consulting", "finance"}), ("KPMG", {"accounting", "consulting"}),
    ("McKinsey & Company", {"consulting"}), ("Boston Consulting Group", {"consulting"}), ("Bain & Company", {"consulting"}), ("Capgemini", {"software", "consulting"}),
    ("JPMorgan Chase", {"finance", "data", "software"}), ("Citi", {"finance", "data", "accounting"}), ("HSBC", {"finance", "accounting", "data"}),
    ("Standard Chartered", {"finance", "data"}), ("Goldman Sachs", {"finance", "software"}), ("Unilever", {"marketing", "supply_chain", "hr", "finance", "sales"}),
    ("Nestlé", {"marketing", "supply_chain", "sales", "mechanical"}), ("Procter & Gamble", {"marketing", "supply_chain", "sales"}), ("PepsiCo", {"marketing", "supply_chain", "sales"}),
    ("The Coca-Cola Company", {"marketing", "sales"}), ("Siemens", {"electrical", "mechanical", "software"}), ("Bosch", {"mechanical", "electrical", "software"}),
    ("ABB", {"electrical", "mechanical"}), ("Schneider Electric", {"electrical", "software", "sales"}), ("Honeywell", {"electrical", "mechanical", "software"}),
    ("Shell", {"mechanical", "electrical", "supply_chain"}), ("Pfizer", {"pharma", "sales"}), ("Novartis", {"pharma", "data"}), ("AstraZeneca", {"pharma"}),
    ("GSK", {"pharma", "sales"}), ("Sanofi", {"pharma"}), ("Roche", {"pharma", "data"}), ("Abbott", {"pharma", "sales"}),
    ("Marriott International", {"hospitality"}), ("Hilton", {"hospitality"}), ("Accor", {"hospitality"}), ("IHG Hotels & Resorts", {"hospitality"}), ("Hyatt", {"hospitality"}),
    ("DHL", {"supply_chain"}), ("Maersk", {"supply_chain"}), ("FedEx", {"supply_chain"}), ("Ogilvy", {"marketing", "design", "media"}),
    ("Publicis Groupe", {"marketing", "design"}), ("WPP", {"marketing", "media"}), ("Dentsu", {"marketing", "media"}), ("AECOM", {"civil", "electrical"}),
    ("Jacobs", {"civil", "electrical", "mechanical"}), ("WSP", {"civil", "electrical"}), ("Arup", {"civil", "mechanical"}), ("L'Oréal", {"marketing", "supply_chain"}),
    ("Adobe", {"software", "design"}), ("Uber", {"software", "data", "supply_chain"}), ("Visa", {"software", "finance", "data"}), ("Mastercard", {"software", "finance", "data"}),
    ("Samsung", {"software", "electrical", "design"}), ("Reuters", {"media"}), ("Bloomberg", {"media", "finance", "software"}), ("Coursera", {"education"}),
    ("British Council", {"education"}), ("Teach For All", {"education"}),
]
GLOBAL_NAMES = None  # filled below

GLOBAL_LAW = ["Baker McKenzie", "Dentons", "Clifford Chance", "Linklaters", "DLA Piper", "Herbert Smith Freehills Kramer", "White & Case", "Allen Overy Shearman Sterling"]

GLOBAL_NAMES = {c for c, _ in GLOBAL_COMPANIES} | set(GLOBAL_LAW)

CURRENCY = {"INR": "₹", "USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "AUD": "A$", "CAD": "C$", "SGD": "S$", "NZD": "NZ$"}

INTERN_BULLETS = [
    "Supported the {area} team on {n} live projects, preparing analyses and documentation",
    "Built a tracker in {tool} that streamlined weekly reporting for a team of {team}",
    "Prepared research summaries and presentation decks for {n} stakeholder meetings",
    "Shadowed senior colleagues and proposed process improvements that the team adopted",
    "Collected and cleaned data from {n} sources to support a {area} initiative",
    "Presented final internship findings to department leadership",
    "Assisted with day-to-day {area} operations and maintained accurate records",
    "Drafted standard operating procedures for {n} recurring tasks",
]

ICONIC_ROLE = {  # medicine / nursing use hospitals, education uses schools, law uses firms
    "hospital": "hospitals", "school": "schools", "lawfirm": "lawfirms"}


def wchoice(rng, items):
    """items: list of (value, weight)"""
    tot = sum(w for _, w in items)
    x = rng.uniform(0, tot)
    for v, w in items:
        x -= w
        if x <= 0:
            return v
    return items[-1][0]


def ascii_fold(s):
    s = s.replace("ł", "l").replace("Ł", "L").replace("ß", "ss").replace("ı", "i").replace("İ", "I").replace("đ", "d").replace("Đ", "D").replace("ø", "o").replace("æ", "ae")
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def mabs(y, m):
    return y * 12 + (m - 1)


def ym(ab):
    return ab // 12, ab % 12 + 1


# ---------------------------------------------------------------------------
class DateFmt:
    """One consistent date style per resume."""

    def __init__(self, rng, india=False):
        self.kind = wchoice(rng, [("MMM YYYY", 34), ("MMMM YYYY", 18), ("MM/YYYY", 14), ("YYYY", 8), ("MMM. YYYY", 6), ("MM.YYYY", 6), ("YYYY-MM", 4), ("MMM 'YY", 5), ("MMM, YYYY", 5)])
        self.sep = wchoice(rng, [(" – ", 45), (" - ", 30), (" to ", 8), (" — ", 7), ("–", 6), (" ~ ", 2), (" | ", 2)])
        pres = [("Present", 70), ("Current", 12), ("Now", 6)]
        if india:
            pres += [("Till Date", 10), ("Present", 10)]
        self.present = wchoice(rng, pres)
        self.edu_style = wchoice(rng, [("range", 55), ("end", 30), ("range_year", 15)])

    def one(self, y, m):
        k = self.kind
        if k == "MMM YYYY":
            return f"{MONTHS[m-1][:3]} {y}"
        if k == "MMM. YYYY":
            mm = MONTHS[m-1]
            return f"{mm[:3]}. {y}" if len(mm) > 3 else f"{mm} {y}"
        if k == "MMM, YYYY":
            return f"{MONTHS[m-1][:3]}, {y}"
        if k == "MMMM YYYY":
            return f"{MONTHS[m-1]} {y}"
        if k == "MM/YYYY":
            return f"{m:02d}/{y}"
        if k == "MM.YYYY":
            return f"{m:02d}.{y}"
        if k == "YYYY-MM":
            return f"{y}-{m:02d}"
        if k == "MMM 'YY":
            return f"{MONTHS[m-1][:3]} '{y % 100:02d}"
        return f"{y}"

    def rng_(self, s, e):
        a = self.one(*s)
        b = self.present if e is None else self.one(*e)
        if a == b:
            return a
        return f"{a}{self.sep}{b}"

    def edu(self, s, e, expected=False):
        if expected:
            if self.edu_style == "end":
                return f"Expected {self.one(*e) if self.kind != 'YYYY' else e[0]}"
            return f"{s[0]}{self.sep}{e[0]} (Expected)" if self.edu_style == "range_year" else f"{self.one(*s)}{self.sep}{self.present}"
        if self.edu_style == "end":
            return self.one(*e)
        if self.edu_style == "range_year":
            return f"{s[0]}{self.sep}{e[0]}"
        return self.rng_(s, e)


# ---------------------------------------------------------------------------
class PersonGen:
    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.used_bullets = set()

    # ---- helpers -----------------------------------------------------------
    def money(self, cur, scale=None):
        r = self.rng
        scale = scale or r.choice(["k", "m", "m", "big"])
        if cur == "INR":
            if scale == "k":
                return f"₹{r.randint(5, 95)} lakh"
            v = round(r.uniform(1.2, 45), 1) if scale == "m" else r.randint(50, 900)
            return r.choice([f"₹{v} Cr", f"INR {v} crore", f"₹{v} crore"])
        sym = CURRENCY.get(cur)
        if scale == "k":
            v = f"{r.randint(80, 950)}K"
        elif scale == "m":
            v = f"{round(r.uniform(1.1, 25), 1)}M"
        else:
            v = f"{r.randint(30, 450)}M"
        if cur in ("JPY", "KRW", "IDR", "VND", "IRR", "COP"):
            v = f"{round(r.uniform(1.2, 90), 1)}B" if scale != "k" else f"{r.randint(20, 900)}M"
        if sym and r.random() < 0.8:
            return f"{sym}{v}"
        return f"{cur} {v}"

    def kfmt(self):
        r = self.rng
        return r.choice([f"{r.randint(10, 950)}K", f"{round(r.uniform(1.1, 9.5), 1)}M", f"{r.randint(2, 90)},000", f"{r.randint(10, 500)}K+"])

    def fill(self, text, tools):
        r = self.rng
        cur = self.work_region["currency"]

        def rep(m):
            k = m.group(1)
            if k == "tool":
                return tools[0] if tools else "Excel"
            if k == "tool2":
                return tools[1] if len(tools) > 1 else (tools[0] if tools else "Excel")
            if k == "pct":
                return str(r.randint(12, 60))
            if k == "pct_small":
                return str(r.randint(5, 25))
            if k == "n":
                return str(r.randint(3, 24))
            if k == "n_big":
                return str(r.choice([50, 60, 75, 80, 120, 150, 200, 250, 300, 400, 500, 120, 180]))
            if k == "team":
                return str(r.randint(3, 14))
            if k == "k":
                return self.kfmt()
            if k == "money":
                return self.money(cur)
            if k == "hours":
                return str(r.randint(4, 40))
            if k == "days":
                return str(r.randint(7, 30))
            if k == "months":
                return str(r.randint(3, 12))
            if k == "city":
                return self.city[0]
            return m.group(0)
        return re.sub(r"\{(\w+)\}", rep, text)

    # ---- identity -----------------------------------------------------------
    def make_name(self, reg, gender):
        r = self.rng
        first = r.choice(reg["first_m"] if gender == "m" else reg["first_f"])
        last = r.choice(reg["last"])
        pattern = wchoice(r, reg.get("names", [("gf", 1.0)]))
        given, family = first, last
        if pattern == "gf":
            full = f"{first} {last}"
        elif pattern == "gmf":
            mid = r.choice("ABCDEFGHJKLMNPRSTW")
            full = f"{first} {mid}. {last}"
        elif pattern == "FG":
            full = f"{last.upper()} {first}"
        elif pattern == "initial_given":
            full = r.choice([f"{last[0]}. {first}", f"{first} {last[0]}."]) if r.random() < 0.5 else f"{first} {last}"
        elif pattern == "vn":
            mid = r.choice(reg["middle_m"] if gender == "m" else reg["middle_f"])
            full = f"{last} {mid} {first}"
        elif pattern == "es":
            last2 = r.choice([x for x in reg["last"] if x != last])
            full = f"{first} {last} {last2}"
        elif pattern == "pt":
            last2 = r.choice([x for x in reg["last"] if x != last])
            full = f"{first} {last2} {last}"
        elif pattern == "ar":
            father = r.choice(reg["first_m"])
            full = f"{first} {father} {last}"
        elif pattern == "mono":
            full = first if r.random() < 0.4 else f"{first} {last}"
        else:
            full = f"{first} {last}"
        return full, given, family

    def make_email(self, given, family, reg):
        r = self.rng
        g = re.sub(r"[^a-z]", "", ascii_fold(given.split()[0]).lower())
        f = re.sub(r"[^a-z]", "", ascii_fold(family).lower()) or "mail"
        yy = r.randint(85, 99) if r.random() < 0.5 else r.randint(0, 6)
        local = wchoice(r, [(f"{g}.{f}", 30), (f"{g}{f}", 15), (f"{g[0]}{f}", 10), (f"{g}_{f}", 6), (f"{g}.{f}{yy:02d}", 10),
                            (f"{g}{f}{r.randint(1, 99)}", 8), (f"{f}.{g}", 6), (f"{g}{f[0]}{r.randint(10, 999)}", 6), (f"{g}.{f[:1]}", 3), (f"{g}{r.randint(1990, 2004)}", 6)])
        domain = wchoice(r, reg["email_domains"])
        return f"{local}@{domain}"

    def make_phone(self, reg, city_name):
        r = self.rng
        cc, prefixes, n, patterns = reg["phone"]
        if isinstance(prefixes, str) and prefixes.startswith("NANP"):
            areas = NANP_AREA[prefixes].get(city_name) or [a for v in NANP_AREA[prefixes].values() for a in v]
            digits = r.choice(areas) + str(r.randint(2, 9)) + "".join(str(r.randint(0, 9)) for _ in range(6))
            if digits[3:6] == "555":
                digits = digits[:3] + "556" + digits[6:]
        else:
            p = r.choice(prefixes)
            digits = p + "".join(str(r.randint(0, 9)) for _ in range(n - len(p)))
        pat = r.choice(patterns)
        it = iter(digits)
        disp = "".join(next(it) if ch == "#" else ch for ch in pat)
        return disp, f"+{cc}{digits}"

    def location(self, reg, city):
        name, state = city
        loc = reg["loc"]
        r = self.rng
        if loc == "us" and state:
            return r.choice([f"{name}, {state}", f"{name}, {state}", f"{name}, {state}, USA" if reg["cc"] == "US" else f"{name}, {state}, Canada"])
        if loc == "state" and state:
            return r.choice([f"{name}, {state}", f"{name}, {reg['country']}", f"{name}, {state}, {reg['country']}"])
        if loc == "uk":
            return r.choice([f"{name}, UK", f"{name}, United Kingdom", name])
        if name == reg["country"] or name == "Singapore":
            return name
        return f"{name}, {reg['country']}"

    # ---- education ------------------------------------------------------------
    def pick_uni(self, reg, family, exclude=()):
        types = FAMILY_UNI_TYPE.get(family, ["all"])
        unis = [u for u in reg["unis"] if u[0] not in exclude]
        for t in types:
            c = [u for u in unis if t in u[2]]
            if c:
                return self.rng.choice(c)
        c = [u for u in unis if "all" in u[2]]
        return self.rng.choice(c or unis)

    def degree_name(self, style, family, field, cc=None):
        if style == "us" and cc not in ("US", "CA") and family in ("law_b", "med_b", "law_m", "med_m"):
            style = "asia"
        table = DEG.get(style, DEG["us"])
        if family not in table:
            family = {"bca": "sci_b", "mca": "sci_m", "arch_b": "eng_b"}.get(family, family)
        pats = table.get(family) or DEG["us"].get(family) or ["{f}"]
        return self.rng.choice(pats).replace("{f}", field)

    def grade(self, style, cc):
        r = self.rng
        if r.random() < 0.4:
            return None
        if style == "eu" and cc in EU_GRADES_BY_COUNTRY:
            pats = EU_GRADES_BY_COUNTRY[cc]
        else:
            pats = GRADES.get(style, GRADES["us"])
        p = r.choice(pats)
        return p.format(cgpa=f"{r.uniform(6.8, 9.6):.2f}".rstrip("0").rstrip("."), pct=f"{r.uniform(62, 94):.1f}", gpa=f"{r.uniform(3.0, 3.97):.2f}",
                        ten=f"{r.uniform(7.0, 9.4):.1f}", de=f"{r.choice([1.0, 1.3, 1.5, 1.7, 1.8, 2.0, 2.1, 2.3, 2.5])}", it=r.randint(98, 110),
                        nl=f"{r.uniform(7.0, 8.8):.1f}", pt=r.randint(14, 18), pl=f"{r.choice([4.0, 4.5, 5.0, 4.3, 4.7])}")

    # ---- main ------------------------------------------------------------------
    def generate(self, rid, region_key=None, industry=None, level=None, richness=0):
        r = self.rng
        origin = REGION_BY_KEY[region_key] if region_key else wchoice(r, [(x, x["weight"]) for x in REGIONS])
        work = origin
        relocated = False
        if not origin.get("expat_hub") and origin["cc"] in RELOCATION and r.random() < 0.16:
            dest_key = wchoice(r, RELOCATION[origin["cc"]])
            work = REGION_BY_KEY[dest_key] if dest_key in REGION_BY_KEY else origin
            relocated = work is not origin
        self.origin, self.work_region = origin, work
        gender = r.choice("mf")
        name, given, family = self.make_name(origin, gender)

        ind_key = industry or wchoice(r, [(k, v["weight"]) for k, v in INDUSTRIES.items()])
        ind = INDUSTRIES[ind_key]
        level = level or wchoice(r, [("fresher", 18), ("junior", 25), ("mid", 25), ("senior", 20), ("exec", 12)])
        years = {"fresher": 0, "junior": r.randint(1, 3), "mid": r.randint(4, 7), "senior": r.randint(8, 13), "exec": r.randint(14, 24)}[level]

        self.city = r.choice(work["cities"])
        india_like = origin["style"] == "india"
        df = DateFmt(r, india=india_like and not relocated)
        now = mabs(NOW_Y, NOW_M)

        # ---------- education timeline ----------
        style_o = origin["style"]
        fam_b, field_b = r.choice(ind["degrees_b"])
        b_len = {"med_b": 5, "law_b": 5 if style_o == "india" else (3 if style_o == "uk" else 4)}.get(fam_b, 4 if style_o in ("india", "us", "asia") or fam_b == "eng_b" else 3)
        if style_o == "india" and fam_b in ("sci_b", "com_b", "arts_b", "bus_b", "bca"):
            b_len = 3
        student = level == "fresher" and r.random() < 0.5
        need_master = (ind_key in ("law", "medicine") and origin["cc"] in ("US", "CA")) or (ind_key == "medicine" and level not in ("fresher", "junior") and r.random() < 0.8)
        has_master = need_master or (level != "fresher" and r.random() < {"junior": 0.25, "mid": 0.4, "senior": 0.45, "exec": 0.5}[level])
        if level == "fresher" and r.random() < 0.15:
            has_master = True
        mba_mid = False
        fam_m = field_m = None
        if has_master:
            fam_m, field_m = r.choice(ind["degrees_m"])
            if need_master and ind_key == "law":
                fam_m, field_m = "law_m", "Law"
            if need_master and ind_key == "medicine":
                fam_m = "med_m"
            if fam_m == "mba" and years >= 6 and r.random() < 0.6:
                mba_mid = True
        m_len = 1 if (fam_m in ("edu_m",) or (fam_m and style_o in ("uk",))) else (3 if fam_m in ("law_m",) and style_o == "us" else (3 if fam_m == "med_m" else 2))
        if fam_m == "phd":
            m_len = 4

        career_start = now - years * 12 - r.randint(0, 5)
        grad_month = r.choice([5, 6, 7]) if style_o != "eu" else r.choice([6, 7, 9])
        edu = []
        ab = lambda y, m: mabs(y, m)
        if level == "fresher":
            if student:
                end_y = NOW_Y + 1
                expected_end = (end_y, grad_month)
            else:
                end_y = NOW_Y if r.random() < 0.7 else NOW_Y - 1
            last_end = (end_y, grad_month)
        else:
            cs_y, cs_m = ym(career_start)
            last_end = ym(career_start - r.randint(1, 3))
        # ordered list of degrees (oldest first): bachelor, [master]
        degrees = []
        if has_master and not mba_mid:
            m_end = last_end
            m_start = (m_end[0] - m_len, r.choice([7, 8, 9]))
            b_end = (m_start[0] if m_start[1] > 6 else m_start[0] - 1, grad_month)
            if b_end[0] >= m_start[0] and b_end[1] >= m_start[1]:
                b_end = (m_start[0], max(1, m_start[1] - 3))
            degrees = [("b", fam_b, field_b, (b_end[0] - b_len, r.choice([7, 8, 9])), b_end), ("m", fam_m, field_m, m_start, m_end)]
        else:
            b_end = last_end
            degrees = [("b", fam_b, field_b, (b_end[0] - b_len, r.choice([7, 8, 9])), b_end)]
        mba_gap = None
        if mba_mid:
            work_before = r.randint(2, min(6, years - 3))
            ms = career_start + work_before * 12 + r.randint(0, 6)
            me = ms + m_len * 12 - r.randint(2, 4)
            if m_len == 1:
                me = ms + 11
            mba_gap = (ms, me)
            degrees.append(("m", fam_m, field_m, ym(ms), ym(me)))

        # where each degree was studied
        abroad_master = relocated and r.random() < 0.55
        used = []
        for kind, fam, field, s, e in degrees:
            reg = work if (kind == "m" and abroad_master) else origin
            style = reg["style"]
            u = self.pick_uni(reg, fam, used)
            used.append(u[0])
            is_expected = level == "fresher" and student and e == degrees[-1][4]
            dates = df.edu(s, e, expected=is_expected)
            uloc = u[1] if (reg is work) else (u[1] if "," in u[1] else f"{u[1]}, {reg['country']}")
            if r.random() < 0.15 and "," not in uloc and reg is work:
                uloc = f"{u[1]}, {reg['country']}"
            edu.append(dict(kind=kind, degree=self.degree_name(style, fam, field, reg["cc"]), institution=u[0], location=uloc, dates=dates,
                            start=f"{s[0]}-{s[1]:02d}", end=("expected " if is_expected else "") + f"{e[0]}-{e[1]:02d}",
                            grade=self.grade(style, reg["cc"]), field=field,
                            year=(f"{e[0]} (Expected)" if is_expected else str(e[0]))))
        if origin.get("school_boards") and level in ("fresher", "junior") and r.random() < 0.6:
            b_start_y = int(edu[0]["start"][:4])
            sch = r.choice(origin["schools"])
            for i, (cls, boards) in enumerate(INDIA_SCHOOL[: r.choice([1, 2])]):
                y = b_start_y - i * 2
                edu.append(dict(kind="school", degree=f"{cls} ({r.choice(boards)})", institution=f"{sch}, {self.rng.choice(origin['cities'])[0]}" if r.random() < 0.3 else sch,
                                location=None, dates=str(y), start=None, end=f"{y}-05", grade=f"{r.uniform(78, 97):.1f}%", field=None, year=str(y)))
        edu_sorted = sorted([e for e in edu if e["kind"] != "school"], key=lambda e: e["end"].replace("expected ", ""), reverse=True) + [e for e in edu if e["kind"] == "school"]
        # Indian schools: Class XII before Class X when two
        self.edu = edu_sorted

        # ---------- skills ----------
        groups = ind["skill_groups"]
        skills, skill_groups = [], []
        n_groups = min(len(groups), r.choice([2, 3, 4]))
        for gname in r.sample(list(groups.keys()), n_groups) if r.random() < 0.3 else list(groups.keys())[:n_groups]:
            pool = groups[gname]
            k = min(len(pool), r.randint(3, 6 if level != "fresher" else 5))
            items = r.sample(pool, k)
            idx = []
            for it in items:
                if it not in skills:
                    idx.append(len(skills))
                    skills.append(it)
            if idx:
                skill_groups.append(dict(name=gname, idx=idx))
        if r.random() < 0.55:
            soft = r.sample(GENERIC_SOFT, r.randint(2, 4))
            idx = []
            for s in soft:
                idx.append(len(skills))
                skills.append(s)
            skill_groups.append(dict(name=r.choice(["Soft Skills", "Core Strengths", "Interpersonal", "Professional Skills"]), idx=idx))
        toolish = ("Tools", "Software", "Systems", "Languages", "Frameworks", "Cloud & DevOps", "Databases", "Data Engineering", "BI & Visualisation", "ML & AI")
        tool_skills = [skills[i] for g in skill_groups if g["name"] in toolish for i in g["idx"]]
        if not tool_skills:
            tool_skills = [x for g in groups for x in groups[g] if g in toolish][:6]
        if not tool_skills:
            tool_skills = [s for s in skills if s not in GENERIC_SOFT]
        self.tools = tool_skills or ["Excel"]
        skill_levels = [r.choice([3, 4, 4, 5, 5]) if i < 4 else r.choice([2, 3, 3, 4, 4, 5]) for i in range(len(skills))]

        # ---------- experience ----------
        experience = []
        if level == "fresher":
            n_int = wchoice(r, [(0, 25), (1, 50), (2, 25)])
            last_end_ab = mabs(*last_end) if not student else now
            t = mabs(*last_end) - (0 if not student else 12)
            for i in range(n_int):
                dur = r.randint(2, 6)
                e = min(t - r.randint(1, 8), now - 1)
                s = e - dur + 1
                t = s - 4
                experience.append(self.job(ind, ind_key, 0, s, e, df, n_bullets=r.randint(2, 3), intern=True))
        else:
            n_jobs = {"junior": wchoice(r, [(1, 55), (2, 45)]), "mid": wchoice(r, [(2, 50), (3, 40), (1, 10)]),
                      "senior": wchoice(r, [(3, 45), (4, 40), (2, 15)]), "exec": wchoice(r, [(4, 40), (5, 40), (6, 20)])}[level]
            end_last = now if r.random() < 0.86 else now - r.randint(1, 5)
            current = end_last == now
            segs = [(career_start, end_last)]
            if mba_gap:
                segs = [(career_start, mba_gap[0] - 1), (mba_gap[1] + 2, end_last)]
            total = sum(b - a + 1 for a, b in segs)
            # split months across n_jobs with min 8 months each
            n_jobs = max(n_jobs, -(-total // 66))
            n_jobs = max(1, min(n_jobs, total // 8, 6))
            cuts = sorted(r.sample(range(8, max(9, total - 7)), n_jobs - 1)) if n_jobs > 1 else []
            # ensure minimal spacing
            lens, prev = [], 0
            for c in cuts + [total]:
                lens.append(c - prev)
                prev = c
            # map lengths onto segments
            months = []
            for a, b in segs:
                months.extend(range(a, b + 1))
            ranges, pos = [], 0
            for L in lens:
                chunk = months[pos:pos + L]
                pos += L
                if not chunk:
                    continue
                ranges.append((chunk[0], chunk[-1]))
            # small gaps between jobs
            prev_company = None
            jobs = []
            for i, (s, e) in enumerate(ranges):
                yrs_at_start = (s - career_start) / 12
                lvl = 1 if yrs_at_start < 2 else 2 if yrs_at_start < 5 else 3 if yrs_at_start < 9 else 4 if yrs_at_start < 14 else 5
                if level == "junior":
                    lvl = min(lvl, 2)
                is_current = (i == len(ranges) - 1) and current
                same_co = prev_company is not None and prev_company != "same" and r.random() < 0.22
                in_origin = relocated and not abroad_master and i < len(ranges) - 1 and r.random() < 0.7
                if relocated and abroad_master and mba_gap is None:
                    in_origin = False
                if relocated and not abroad_master and i == 0:
                    in_origin = True
                jobs.append(dict(s=s, e=None if is_current else e, lvl=lvl, same=same_co, in_origin=in_origin))
                prev_company = "same" if same_co else True
            # build, newest first
            prev = None
            built = []
            for j in jobs:
                recent = j is jobs[-1] or j is jobs[-2] if len(jobs) > 1 else True
                nb = r.randint(3, 5) if recent else r.randint(2, 3)
                if level in ("senior", "exec"):
                    nb += r.randint(1, 2) if recent else 1
                nb += richness if recent else richness // 2
                job = self.job(ind, ind_key, j["lvl"], j["s"], j["e"], df, n_bullets=nb,
                               same_as=prev if j["same"] else None, region=origin if j["in_origin"] else work)
                built.append(job)
                prev = job
            experience = list(reversed(built))

        # ---------- projects ----------
        n_proj = {"fresher": r.randint(2, 4), "junior": wchoice(r, [(0, 50), (1, 30), (2, 20)])}.get(level, wchoice(r, [(0, 55), (1, 25), (2, 20)]))
        n_proj += richness // 2
        projects = []
        tech_ind = ind_key in ("software", "data", "design", "mechanical", "electrical", "civil")
        for name_, desc in r.sample(ind["projects"], min(n_proj, len(ind["projects"]))):
            p = dict(name=name_, description=self.fill(desc, self.tools))
            if tech_ind and r.random() < 0.75:
                p["tech"] = ", ".join(r.sample(self.tools, min(len(self.tools), r.randint(2, 4))))
            if r.random() < 0.35:
                y = r.randint(max(int(self.edu[0]["start"][:4]) if self.edu[0]["start"] else NOW_Y - 3, NOW_Y - years - 3), NOW_Y)
                p["dates"] = df.one(y, r.randint(1, 12)) if df.kind != "YYYY" else str(y)
            projects.append(p)

        # ---------- certifications / awards ----------
        start_y = int(self.edu[0]["start"][:4]) if self.edu and self.edu[0]["start"] else NOW_Y - 4
        n_cert = {"fresher": wchoice(r, [(0, 30), (1, 35), (2, 35)]), "junior": r.randint(0, 3)}.get(level, r.randint(0, 4))
        n_cert += richness // 2
        certs = []
        for nm, issuer in r.sample(ind["certs"], min(n_cert, len(ind["certs"]))):
            y = r.randint(min(start_y + 2, NOW_Y), NOW_Y)
            certs.append(dict(name=nm, issuer=issuer, year=str(y) if r.random() < 0.7 else df.one(y, r.randint(1, 12))))
        n_aw = wchoice(r, [(0, 45), (1, 30), (2, 18), (3, 7)])
        awards = []
        pool = [(a, "ind") for a in ind["awards"]] + list(GENERIC_AWARDS)
        for nm, src in r.sample(pool, n_aw):
            if src == "uni" or (src is None and not experience):
                issuer = self.edu[-1 if self.edu[-1]["kind"] != "school" else 0]["institution"] if self.edu else None
                y = int(self.edu[0]["end"][-7:-3]) - r.randint(0, 2) if self.edu else NOW_Y - 2
            elif src is None:
                ex = r.choice(experience)
                issuer = ex["company"]
                y = r.randint(int(ex["start"][:4]), NOW_Y if ex["end"] == "present" else int(ex["end"][:4]))
            else:
                issuer = None
                y = r.randint(max(start_y, NOW_Y - years - 2), NOW_Y)
            awards.append(dict(name=nm, issuer=issuer, year=str(min(y, NOW_Y))))

        # ---------- languages ----------
        prof_sets = [
            {"native": "Native", 5: "Fluent", 4: "Professional Working Proficiency", 3: "Intermediate", 2: "Basic"},
            {"native": "Native", 5: "C2", 4: "C1", 3: "B2", 2: "A2"},
            {"native": "Mother tongue", 5: "Full Professional Proficiency", 4: "Professional Working Proficiency", 3: "Limited Working Proficiency", 2: "Elementary Proficiency"},
            {"native": "Native", 5: "Fluent", 4: "Advanced", 3: "Conversational", 2: "Beginner"},
        ]
        ps = r.choice(prof_sets)
        langs = []
        native = origin["langs"][0]
        if origin["key"].startswith("IN-") and r.random() < 0.5:
            native = r.choice(origin["langs"][:3])
        langs.append(dict(language=native, proficiency=ps["native"], level=5))
        if "English" != native:
            langs.append(dict(language="English", proficiency=ps[r.choice([4, 5, 5, 4, 3])], level=4))
        extra = [l for l in origin["langs"] + work["langs"] if l not in [x["language"] for x in langs]]
        for l in r.sample(extra, min(len(extra), wchoice(r, [(0, 35), (1, 45), (2, 20)]))):
            lv = r.choice([2, 3, 3, 4, 5])
            langs.append(dict(language=l, proficiency=ps[lv], level=lv))

        # ---------- contact / links ----------
        email = self.make_email(given, family, origin)
        phone, e164 = self.make_phone(work, self.city[0])
        loc = self.location(work, self.city)
        handle = re.sub(r"[^a-z0-9]", "", ascii_fold(given.split()[0] + family).lower())
        links = []
        if r.random() < 0.78:
            sfx = r.choice(["", "", f"-{r.randint(10, 99)}", f"-{''.join(r.choice('0123456789abcdef') for _ in range(6))}", f"{r.randint(1, 9)}"])
            g2 = ascii_fold(given.split()[0]).lower()
            f2 = re.sub(r"[^a-z]", "", ascii_fold(family).lower())
            links.append(dict(type="LinkedIn", url=r.choice([f"linkedin.com/in/{g2}-{f2}{sfx}", f"linkedin.com/in/{g2}{f2}{sfx}", f"www.linkedin.com/in/{g2}-{f2}{sfx}"])))
        if ind_key in ("software", "data") and r.random() < 0.65:
            links.append(dict(type="GitHub", url=f"github.com/{r.choice([handle, handle[:10], ascii_fold(given.split()[0]).lower() + str(r.randint(1, 99)), g2[0] + f2 if links else handle])}"))
        if ind_key == "design" and r.random() < 0.85:
            links.append(dict(type="Portfolio", url=r.choice([f"behance.net/{handle}", f"dribbble.com/{handle}", f"{handle}.design", f"www.{handle}.com", f"{handle}.framer.website"])))
        if ind_key in ("media", "marketing") and r.random() < 0.3:
            links.append(dict(type="Blog", url=r.choice([f"medium.com/@{handle}", f"{handle}.substack.com", f"muckrack.com/{handle}"])))
        if ind_key in ("medicine", "pharma") and r.random() < 0.2:
            links.append(dict(type="ORCID", url=f"orcid.org/0000-000{r.randint(1, 3)}-{r.randint(1000, 9999)}-{r.randint(100, 999)}{r.choice('0123456789X')}"))
        if r.random() < 0.08:
            links.append(dict(type="Website", url=f"www.{handle}.com"))

        # ---------- headline ----------
        if experience and experience[0]["end"] == "present":
            headline = experience[0]["title"]
        elif level == "fresher":
            b = self.edu[0]["degree"].split(" in ")[0].split(",")[0]
            headline = r.choice([f"Final-year {b} Student" if student else f"{b} Graduate", f"Aspiring {r.choice(ind['titles'][1])}", ind["label"] + " Graduate"])
        else:
            headline = experience[0]["title"] if experience else ind["label"]
        if r.random() < 0.15 and experience:
            headline = f"{headline} | {years}+ Years in {ind['label']}" if years >= 3 else headline

        return dict(
            resume_id=rid, origin=origin["key"], origin_country=origin["country"], work_country=work["country"], work_cc=work["cc"],
            relocated=relocated, industry=ind_key, industry_label=ind["label"], level=level, years_experience=years, gender=gender,
            page_size=work["page"], india=india_like,
            name=name, headline=headline,
            contact=dict(email=email, phone=phone, phone_e164=e164, location=loc, city=self.city[0], country=work["country"]),
            links=links, education=self.edu, experience=experience, skills=skills, skill_groups=skill_groups, skill_levels=skill_levels,
            projects=projects, certifications=certs, awards=awards, languages=langs,
            date_format=df.kind, date_sep=df.sep, present_word=df.present,
        )

    def job(self, ind, ind_key, lvl, s, e, df, n_bullets=3, intern=False, same_as=None, region=None):
        r = self.rng
        region = region or self.work_region
        titles = ind["titles"][lvl]
        title = r.choice(titles)
        if same_as is not None:
            company = same_as["company"]
            city = same_as["_city"]
        else:
            company = self.pick_employer(ind, ind_key, region)
            city = self.city if region is self.work_region and r.random() < 0.75 else r.choice(region["cities"])
            if company in GLOBAL_NAMES and city not in region["cities"][:2]:
                city = r.choice(region["cities"][:2])  # multinationals sit in the main business hubs
        loc = city[0] if (region is self.work_region or city[0] == region["country"]) else f"{city[0]}, {region['country']}"
        if region["loc"] == "us" and city[1] and region is self.work_region and r.random() < 0.7:
            loc = f"{city[0]}, {city[1]}"
        if ind_key in ("software", "data", "design", "marketing", "media", "consulting", "sales", "hr") and r.random() < 0.07:
            loc = r.choice(["Remote", f"{city[0]} (Hybrid)", f"Remote – {region['country']}"])
        pool = [b for b in ind["bullets"] if b not in self.used_bullets]
        if len(pool) < 2:
            pool = [b for b in ind["bullets"] + GENERIC_LEAD if b not in self.used_bullets] or ind["bullets"][:]
        chosen = r.sample(pool, max(1, min(n_bullets, len(pool))))
        if lvl >= 3 and r.random() < 0.7:
            lead = [b for b in GENERIC_LEAD if b not in self.used_bullets]
            if lead:
                chosen[-1] = r.choice(lead)
        self.used_bullets.update(chosen)
        tools = r.sample(self.tools, min(2, len(self.tools)))
        if intern:
            ib = r.sample(INTERN_BULLETS, n_bullets)
            junior_ok = [b for b in ind["bullets"] if not re.search(r"\b(Led|led|Owned|Managed|managed|team of|Mentored|Trained|Supervised)\b", b)]
            if junior_ok and ind_key in ("software", "data", "design", "marketing", "finance", "consulting") and r.random() < 0.6:
                ib[0] = r.choice(junior_ok)
            chosen = [b.replace("{area}", ind["label"].split(" (")[0].lower()) for b in ib]
        bullets = [self.fill(b, tools) for b in chosen]
        sy, sm = ym(s)
        end = None if e is None else ym(e)
        return dict(title=title, company=company, location=loc, dates=df.rng_((sy, sm), end), start=f"{sy}-{sm:02d}",
                    start_text=df.one(sy, sm), end_text=df.present if end is None else df.one(*end),
                    end="present" if e is None else f"{end[0]}-{end[1]:02d}", bullets=bullets, _city=city, level=lvl)

    def pick_employer(self, ind, ind_key, region):
        r = self.rng
        emp = ind.get("employer")
        if emp == "hospital":
            return r.choice(region["hospitals"])
        if emp == "school":
            if r.random() < 0.65:
                return r.choice(region["schools"])
            return r.choice([u[0] for u in region["unis"]])
        if emp == "lawfirm":
            return r.choice(region["lawfirms"]) if r.random() < 0.7 else r.choice(GLOBAL_LAW)
        local = [c for c, inds in region["companies"] if ind_key in inds]
        glob = [c for c, inds in GLOBAL_COMPANIES if ind_key in inds]
        if ind_key == "media":
            local += [c for c, inds in region["companies"] if "media" in inds]
        used = getattr(self, "used_companies", set())
        local0, glob0 = local, glob
        local = [c for c in local if c not in used]
        glob = [c for c in glob if c not in used]
        if not local and not glob:
            local, glob = local0, glob0
        if local and (r.random() < 0.68 or not glob):
            c = r.choice(local)
        else:
            c = r.choice(glob or [c for c, _ in region["companies"]])
        self.used_companies = used | {c}
        return c
