"""Model A - NLP-only (layout-blind) resume parser (project spec section 8.1).

    OCR words (shared, from run_ocr.py)
      -> flattened into ONE text stream in raster order (left-to-right, top-to-bottom across the
         whole page width; spatial structure is discarded right here)
      -> regex (email, phone, links, dates) + spaCy NER (names, organisations, places)
         + section-heading rules
      -> JSON in the answer-key schema

    python pipeline\\model_nlp_only.py --data dataset --ocr ocr --out preds\\nlp_only
    python pipeline\\model_nlp_only.py ... --split test --tiers clean          (subset)
    python pipeline\\model_nlp_only.py ... --order tesseract                   (ablation: Tesseract's own block order)

Also writes preds\\nlp_only\\_flat\\<tier>\\RXXXX.txt = the flattened text the parser saw
(useful to show column interleaving in the report).
"""
import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from rapidfuzz import fuzz

# ------------------------------------------------------------------ lexicons --
HEADINGS = {
    "experience": ["experience", "work experience", "professional experience", "employment history", "employment", "work history",
                   "career history", "professional background", "internships", "internship experience", "industry exposure",
                   "relevant experience", "internship", "career summary"],
    "education": ["education", "academic background", "academic qualifications", "educational qualifications",
                  "education & qualifications", "education and qualifications", "qualifications", "academics", "academic profile"],
    "skills": ["skills", "technical skills", "core competencies", "key skills", "areas of expertise", "skills & tools",
               "competencies", "expertise", "skill set", "technical expertise"],
    "projects": ["projects", "key projects", "selected projects", "academic projects", "notable projects", "personal projects"],
    "certifications": ["certifications", "certificates", "licenses & certifications", "licences & certifications",
                       "professional certifications", "courses & certifications", "courses", "certification"],
    "awards": ["awards", "honours & awards", "honors & awards", "achievements", "awards & recognition", "honors", "honours",
               "accomplishments", "awards and achievements"],
    "languages": ["languages", "language proficiency", "languages known", "language skills"],
    "contact": ["contact", "contact details", "contact information", "get in touch", "personal details", "personal information"],
    "links": ["links", "online profiles", "profiles", "social", "social media"],
    "summary": ["summary", "profile", "objective", "career objective", "about me", "professional summary"],
    "declaration": ["declaration"],
}
HEAD_LOOKUP = sorted(((h, k) for k, hs in HEADINGS.items() for h in hs), key=lambda x: -len(x[0]))

TITLE_WORDS = r"""engineer|developer|manager|analyst|intern|trainee|director|nurse|teacher|consultant|associate|officer|designer|
lead|head|specialist|executive|coordinator|chef|representative|accountant|auditor|scientist|resident|physician|lecturer|
professor|principal|editor|reporter|writer|producer|partner|counsel|solicitor|attorney|advocate|architect|administrator|
supervisor|assistant|president|vp|chief|cto|ceo|cfo|coo|founder|researcher|pharmacist|chemist|surveyor|planner|buyer|
recruiter|generalist|strategist|copywriter|correspondent|marketer|sister|educator|tutor|instructor|technician|operator|
agent|controller|treasurer|programmer|scholar|fellow|registrar|consultant|staff|member|apprentice|commis|receptionist|
dean|counsellor|counselor|trader|banker|underwriter|actuary|economist|advisor|adviser|owner|clerk|cashier|sous"""
TITLE_RX = re.compile(r"\b(" + TITLE_WORDS.replace("\n", "") + r")s?\b", re.I)

DEGREE_RX = re.compile(r"""(?<![A-Za-z])(B\.?\s?Tech|M\.?\s?Tech|B\.E\.|M\.E\.|B\.?\s?Sc|M\.?\s?Sc|BSc|MSc|B\.S\.|M\.S\.|BS|MS|B\.A\.|M\.A\.|BA|MA|
BBA|BMS|BCA|MCA|B\.?\s?Com|M\.?\s?Com|B\.?\s?Des|M\.?\s?Des|B\.?\s?Arch|BEng|MEng|B\.Eng|M\.Eng|BFA|MFA|B\.F\.A\.|BN|BSN|MSN|DNP|
LLB|LL\.B\.|LLM|LL\.M\.|MBBS|MBChB|MPharm|M\.\s?Pharm|B\.\s?Pharm|BPharm|Pharm\.?\s?D|J\.D\.|JD|MD|M\.D\.|DO|DNB|MRCP|MRCS|
PGCE|PGDE|PGDM|B\.?\s?Ed|M\.?\s?Ed|Ph\.?\s?D|DPhil|MRes|MBA|M\.B\.A\.|BHM|BHMCT|GNM|Bachelor|Master|Doctor|Diploma|Doctorate|
Class\s+X{1,2}I*|Degree\s+in|Juris|Executive\s+MBA|Post\s+Graduate|Postgraduate|State\s+Examination|Specialist\s+Training|
Residency|Licenciatura|Teaching\s+Qualification|Medical\s+Degree|Legal\s+Practice|Bar\s+Professional)(?![A-Za-z])""".replace("\n", ""))
INST_RX = re.compile(r"(Universit|College|Institut|School|Academy|Polytechnic|École|Ecole|Politecnico|Hochschule|Faculty|Campus|"
                     r"\bIIT\b|\bNIT\b|\bBITS\b|\bIIM\b|INSEAD|ESADE|\bHEC\b|Insper|KAIST|POSTECH|Sciences Po|Conservatoire|Lyceum|Gymnasium|"
                     r"Kendriya|Vidyalaya|Public School|Medical College|Kasr|Sorbonne|NALSAR|XLRI|NMIMS|CEPT|SRCC|Unitec)", re.I)
LANGS = ["English", "Hindi", "Bengali", "Marathi", "Gujarati", "Tamil", "Telugu", "Kannada", "Malayalam", "Punjabi", "Urdu", "Odia",
         "Assamese", "Konkani", "Sindhi", "Pashto", "Nepali", "Newari", "Sinhala", "Mandarin Chinese", "Mandarin", "Cantonese", "Chinese",
         "Japanese", "Korean", "Vietnamese", "Thai", "Malay", "Indonesian", "Javanese", "Filipino", "Tagalog", "Cebuano", "Arabic",
         "Persian", "Farsi", "Turkish", "Azerbaijani", "Hebrew", "Kurdish", "Armenian", "French", "German", "Spanish", "Portuguese",
         "Italian", "Dutch", "Polish", "Ukrainian", "Russian", "Swedish", "Norwegian", "Danish", "Finnish", "Greek", "Czech",
         "Romanian", "Hungarian", "Catalan", "Basque", "Galician", "Irish", "Welsh", "Swahili", "Amharic", "Afaan Oromo", "Oromo",
         "Tigrinya", "Yoruba", "Igbo", "Hausa", "Twi", "Ga", "Ewe", "Kikuyu", "Luo", "Afrikaans", "isiZulu", "Zulu", "isiXhosa", "Xhosa",
         "Sesotho", "Te Reo Māori", "Maori", "Samoan", "Tatar"]
LANG_RX = re.compile(r"(?<![A-Za-z])(" + "|".join(re.escape(l) for l in sorted(LANGS, key=len, reverse=True)) + r")(?![A-Za-z])", re.I)
PROF_RX = re.compile(r"(native|mother tongue|fluent|bilingual|full professional proficiency|professional working proficiency|"
                     r"limited working proficiency|elementary proficiency|advanced|intermediate|conversational|beginner|basic|"
                     r"upper[- ]intermediate|working knowledge|proficient|[ABC][12])", re.I)
SKILL_LEVEL_WORDS = re.compile(r"\b(expert|advanced|intermediate|proficient|beginner|basic|elementary|good|excellent|\d{1,3}\s?%)\b", re.I)

MONTH = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
DATE = rf"(?:{MONTH}\.?,?\s*'?\d{{2,4}}|\d{{1,2}}[/.]\d{{4}}|\d{{4}}-\d{{2}}|(?:19|20)\d{{2}})"
END = rf"(?:{DATE}|present|current|now|till\s+date|today|ongoing|expected\s+\d{{4}})"
DATE_RANGE_RX = re.compile(rf"{DATE}\s*(?:–|-|—|to|~|\||until)\s*{END}|{DATE}\s+(?:present|current|now|till\s+date)|"
                           rf"(?:{MONTH}\.?,?\s*'?\d{{2,4}}|\d{{1,2}}[/.]\d{{4}}|\d{{4}}-\d{{2}})\s+{END}", re.I)
SINGLE_DATE_RX = re.compile(rf"{DATE}|Expected\s+{DATE}", re.I)
EMAIL_RX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RX = re.compile(r"(?:\(?\+\s?\d{1,3}\)?[\s.-]?)?\(?\d{2,5}\)?(?:[\s.-]?\d{2,5}){1,5}")
LINK_RX = re.compile(r"(?:https?://)?(?:www\.)?(?:linkedin\.com/in/|github\.com/|behance\.net/|dribbble\.com/|medium\.com/@|orcid\.org/|muckrack\.com/)[^\s|,;]+|"
                     r"(?:www\.)[A-Za-z0-9-]+\.[A-Za-z.]{2,}[^\s|,;]*|[A-Za-z0-9-]+\.(?:design|framer\.website|substack\.com|dev|io)\b[^\s|,;]*", re.I)
BULLET_RX = re.compile(r"^\s*(?:[•▪‣›◦·\-–*>»¢©®°]|[eo](?=\s[A-Z]))\s*")
LABEL_RX = re.compile(r"\b(Address|Location|Phone|Mobile|Telephone|Tel|Email|E-mail|LinkedIn|GitHub|Portfolio|Website|Blog|Contact)\b\s*:?", re.I)
TABLE_HDR_RX = re.compile(r"^\s*(s\.?\s?no\.?|organi[sz]ation|designation|company|role|duration|tenure|period|dates|location|"
                          r"key responsibilities|responsibilities|key contributions|qualification|institution|from\b)", re.I)
HEADWORDS = {"history", "experience", "background", "qualifications", "skills", "projects", "education", "profile", "summary"}


# --------------------------------------------------------------- flattening --
def flatten_raster(words):
    """Raster order: cluster words into horizontal lines across the FULL page width, then left->right.
    This is exactly the layout-blind step: two columns at the same height get merged into one line."""
    if not words:
        return []
    hs = sorted(w["bbox"][3] - w["bbox"][1] for w in words)
    mh = hs[len(hs) // 2] or 10
    ws = sorted(words, key=lambda w: (w["bbox"][1] + w["bbox"][3]) / 2)
    lines, cur, cy = [], [], None
    for w in ws:
        c = (w["bbox"][1] + w["bbox"][3]) / 2
        if cur and abs(c - cy) > 0.5 * mh:
            lines.append(cur)
            cur = []
        cur.append(w)
        cy = c if len(cur) == 1 else (cy * (len(cur) - 1) + c) / len(cur)
    if cur:
        lines.append(cur)
    return [" ".join(w["text"] for w in sorted(L, key=lambda w: w["bbox"][0])) for L in lines]


def flatten_tesseract(words):
    lines = defaultdict(list)
    for w in words:
        lines[w["line_id"]].append(w)
    return [" ".join(w["text"] for w in sorted(lines[k], key=lambda w: w["word"])) for k in sorted(lines)]


# ------------------------------------------------------------------- helpers --
def norm_head(s):
    return re.sub(r"[^a-z& ]", "", s.lower()).strip()


def heading_of(line):
    """Return (section_key, remainder) if the line is (or starts with) a section heading."""
    n = norm_head(line)
    if not n or len(n) > 60:
        return None
    if ":" in line and line.split(":", 1)[1].strip():
        return None  # "Languages: Python, SQL" = a skill category label, not a section heading
    for h, k in HEAD_LOOKUP:
        if n == h or fuzz.ratio(n, h) >= 90:
            return k, ""
    for h, k in HEAD_LOOKUP:
        if n.startswith(h + " ") and len(h) >= 5:
            rest_words = n[len(h):].split()
            if len(rest_words) <= 5:  # heading merged with text from another column
                return k, line[len(h):].strip(" :|-")
    return None


def split_sections(lines):
    secs = defaultdict(list)
    cur = "header"
    for ln in lines:
        h = heading_of(ln)
        if h:
            cur, rest = h
            if rest:
                secs[cur].append(rest)
            continue
        secs[cur].append(ln)
    return secs


def clean(s):
    return re.sub(r"\s+", " ", s).strip(" |,;:-–—•·")


def strip_dates(s):
    s = DATE_RANGE_RX.sub(" ", s)
    s = SINGLE_DATE_RX.sub(" ", s)
    return clean(s)


COUNTRIES = {"india", "usa", "us", "uk", "uae", "united states", "united kingdom", "united arab emirates", "canada", "australia",
             "new zealand", "ireland", "germany", "france", "spain", "italy", "netherlands", "poland", "sweden", "portugal", "greece",
             "ukraine", "russia", "china", "japan", "south korea", "korea", "vietnam", "philippines", "indonesia", "thailand",
             "singapore", "malaysia", "pakistan", "bangladesh", "sri lanka", "nepal", "saudi arabia", "egypt", "jordan", "lebanon",
             "turkey", "iran", "nigeria", "kenya", "south africa", "ghana", "ethiopia", "mexico", "brazil", "argentina", "colombia"}
LOC_RX = re.compile(r"([A-ZÀ-Ý][\w.'’-]+(?: [A-ZÀ-Ý][\w.'’-]+){0,2}),\s*([A-ZÀ-Ý][\w.'’-]+(?: [A-ZÀ-Ý][\w.'’-]+){0,2})(?:,\s*([A-ZÀ-Ý][\w.'’-]+(?: [A-ZÀ-Ý][\w.'’-]+){0,2}))?")


def placey(s, nlp):
    t = s.strip().lower()
    if t in COUNTRIES or re.fullmatch(r"[A-Z]{2,3}", s.strip()) or t.startswith("remote") or "(hybrid)" in t:
        return True
    return any(e.label_ in ("GPE", "LOC") and len(e.text) >= len(s.strip()) - 2 for e in nlp(s.strip()).ents)


# ---------------------------------------------------------------- extractors --
class Parser:
    def __init__(self, nlp):
        self.nlp = nlp

    def ents(self, text, label):
        return [e.text for e in self.nlp(text).ents if e.label_ == label]

    def email(self, text):
        m = EMAIL_RX.search(text)
        return m.group(0) if m else None

    def phone(self, lines):
        for ln in lines:
            for m in PHONE_RX.finditer(ln):
                s = m.group(0).strip()
                digits = re.sub(r"\D", "", s)
                if not 8 <= len(digits) <= 15:
                    continue
                if re.fullmatch(r"(19|20)\d{2}\s*[-–]\s*(19|20)\d{2}", s) or re.search(r"\d{1,2}/\d{4}", s):
                    continue
                if re.fullmatch(r"(19|20)\d{2}(\s+(19|20)\d{2})+", s):
                    continue
                return s
        return None

    def links(self, text):
        out = []
        for m in LINK_RX.finditer(text):
            u = m.group(0).rstrip(".")
            if "@" in u or u.lower() in (x.lower() for x in out):
                continue
            out.append(u)
        return out

    def name(self, header_lines):
        for ln in header_lines[:8]:
            t = clean(ln)
            if not t or heading_of(t) or EMAIL_RX.search(t) or sum(c.isdigit() for c in t) > 2:
                continue
            for e in self.nlp(t).ents:
                if e.label_ == "PERSON" and len(e.text.split()) >= 2:
                    return e.text
            toks = t.split()
            if 2 <= len(toks) <= 5 and all(re.match(r"^[A-ZÀ-Ý][\w'’.-]*$", x) or x.isupper() for x in toks) and not TITLE_RX.search(t):
                return t
        return None

    def strip_loc(self, s):
        """'University of Lagos, Lagos, Nigeria' -> 'University of Lagos';  'Roche San Francisco' -> 'Roche'"""
        if not s:
            return s
        parts = [p.strip() for p in s.split(",")]
        while len(parts) > 1 and placey(parts[-1], self.nlp):
            parts.pop()
        s = ", ".join(parts)
        ents = [e for e in self.nlp(s).ents if e.label_ in ("GPE", "LOC") and e.end_char == len(s)]
        if ents:
            head = s[:ents[-1].start_char].rstrip(" ,|-")
            if head and not re.search(r"\b(of|de|at|in|del|di|du)$", head, re.I) and len(head) >= 2:
                s = head
        return s

    def location(self, header_lines, name):
        header_lines = [LABEL_RX.sub(" | ", ln) for ln in header_lines]
        for ln in header_lines[:10]:
            for m in LOC_RX.finditer(ln):
                cand = m.group(0)
                if name and cand in name:
                    continue
                if any(placey(g, self.nlp) for g in m.groups() if g):
                    return cand
        for ln in header_lines[:10]:
            for part in re.split(r"\s[|•·◦]\s|\s{2,}", ln):
                p = clean(part)
                if not p or p == name or "@" in p or sum(c.isdigit() for c in p) > 2:
                    continue
                if re.fullmatch(r"[A-ZÀ-Ý][\w .'’-]+,\s*[A-ZÀ-Ý][\w .'’-]+(,\s*[A-Z][\w .'-]+)?", p) and len(p.split()) <= 6:
                    if self.ents(p, "GPE") or len(p.split(",")) >= 2:
                        return p
        for ln in header_lines[:10]:
            g = self.ents(ln, "GPE")
            if g:
                return g[0]
        return None

    def experience(self, lines):
        entries = []
        idx = [i for i, ln in enumerate(lines) if DATE_RANGE_RX.search(ln) and not BULLET_RX.match(ln)]
        for k, i in enumerate(idx):
            ln = lines[i]
            dm = DATE_RANGE_RX.search(ln)
            cands = []
            rest = strip_dates(ln)
            if rest and len(rest.split()) <= 12:
                cands.append(rest)
            for j in (i - 1, i + 1, i + 2):
                if 0 <= j < len(lines) and j not in idx and not BULLET_RX.match(lines[j]) and len(lines[j].split()) <= 12 \
                        and not TABLE_HDR_RX.match(lines[j]) and not heading_of(lines[j]):
                    cands.append(strip_dates(lines[j]))
            parts = []
            for c in cands:
                c = re.sub(r"^\d{1,2}\s+", "", c)  # table serial number
                for x in re.split(r"\s[|@·—–-]\s|,\s(?=[A-Z])|\s@\s", c):
                    x = clean(x)
                    if not x or x.lower() in HEADWORDS:
                        continue
                    # "Mercadona Supply Chain Intern" (flattened table row): leading organisation + title
                    if TITLE_RX.search(x) and len(x.split()) >= 3:
                        org = [e for e in self.nlp(x).ents if e.label_ == "ORG" and e.start_char == 0 and not TITLE_RX.search(e.text)]
                        if org and len(org[0].text) < len(x):
                            parts += [org[0].text, clean(x[len(org[0].text):])]
                            continue
                    parts.append(x)
            title = next((p for p in parts if TITLE_RX.search(p)), None)
            company = None
            for p in parts:
                if p == title or p.lower() in ("remote",) or (self.ents(p, "GPE") and not self.ents(p, "ORG")):
                    continue
                if TITLE_RX.search(p):
                    continue
                company = p
                break
            end = idx[k + 1] if k + 1 < len(idx) else len(lines)
            bullets = [clean(BULLET_RX.sub("", lines[j])) for j in range(i + 1, end) if BULLET_RX.match(lines[j])]
            company = self.strip_loc(company)
            if title or company:
                entries.append(dict(title=title, company=company, dates=clean(dm.group(0)), bullets=bullets))
        return entries

    def education(self, lines):
        entries, used = [], set()
        for i, ln in enumerate(lines):
            m = DEGREE_RX.search(ln)
            if not m or i in used:
                continue
            seg = ln[m.start():]
            seg = re.split(r"\s[|·—–]\s|\s{2,}", seg)[0]
            inst = None
            im = INST_RX.search(seg)
            if im and im.start() > 3:  # "B.Tech ... , XYZ University" on one line
                cut = seg.rfind(",", 0, im.start())
                if cut > 0:
                    inst = clean(seg[cut + 1:])
                    seg = seg[:cut]
            degree = strip_dates(re.sub(r"(CGPA|GPA|Grade|Percentage|Score)\b.*$", "", seg, flags=re.I))
            if not inst:
                for j in (i + 1, i - 1, i + 2):
                    if 0 <= j < len(lines) and INST_RX.search(lines[j]) and not DEGREE_RX.search(lines[j]):
                        inst = strip_dates(lines[j].split(" | ")[0])
                        used.add(j)
                        break
                if not inst:
                    rest = clean(ln[:m.start()]) or clean(ln[m.end():].split(",")[-1])
                    orgs = self.ents(ln, "ORG")
                    inst = orgs[0] if orgs else (rest if INST_RX.search(rest or "") else None)
            dm = DATE_RANGE_RX.search(ln) or SINGLE_DATE_RX.search(ln)
            entries.append(dict(degree=degree, institution=self.strip_loc(inst), dates=dm.group(0) if dm else None))
        return entries

    def skills(self, lines):
        out = []
        for ln in lines:
            ln = BULLET_RX.sub("", ln)
            if re.match(r"^\s*skill\s+proficiency", ln, re.I):
                continue
            if ":" in ln and len(ln.split(":")[0].split()) <= 3:
                ln = ln.split(":", 1)[1]
            ln = SKILL_LEVEL_WORDS.sub("|", ln)
            parts = re.split(r"\s*[,|•·;▪«»+]\s*|\s[-–—]\s|\s{2,}", ln)
            if len(parts) == 1 and len(ln.split()) > 5:
                continue
            for p in parts:
                p = clean(p)
                if 1 < len(p) <= 40 and len(p.split()) <= 5 and not heading_of(p) and not re.fullmatch(r"[\d\W]+", p):
                    if p.lower() not in (x.lower() for x in out):
                        out.append(p)
        return out

    def certifications(self, lines):
        out = []
        for ln in lines:
            ln = BULLET_RX.sub("", ln)
            if re.match(r"^\s*(s\.?\s?no\.?\s*)?certification\b.*\b(issu|year)", ln, re.I):
                continue
            ln2 = re.sub(r"^\d+\s+", "", ln)
            if len(ln2.split()) < 2:
                continue
            if out and len(strip_dates(ln2).split()) <= 4 and SINGLE_DATE_RX.search(ln2) and not out[-1].get("issuer"):
                out[-1]["issuer"] = strip_dates(ln2)
                continue
            parts = [clean(x) for x in re.split(r"\s[–—-]\s|,\s|\s\|\s|\s\(", ln2) if clean(x)]
            name = strip_dates(parts[0]) if parts else None
            if name and len(name) > 3:
                out.append(dict(name=name, issuer=strip_dates(parts[1]) if len(parts) > 1 else None))
        return out

    def languages(self, text):
        out, seen = [], set()
        for m in LANG_RX.finditer(text):
            lang = m.group(1)
            key = lang.lower()
            if key in seen:
                continue
            seen.add(key)
            pm = PROF_RX.search(text[m.end(): m.end() + 45])
            out.append(dict(language=lang, proficiency=pm.group(1) if pm else None))
        return out

    def projects(self, lines):
        out = []
        for i, ln in enumerate(lines):
            t = clean(BULLET_RX.sub("", re.sub(r"^\d+\s+", "", ln)))
            if not t or re.match(r"^(tech|technologies|stack|tools)", t, re.I) or t.endswith("."):
                continue
            w = t.split()
            caps = sum(1 for x in w if x[:1].isupper())
            if 2 <= len(w) <= 8 and caps / len(w) >= 0.6 and not DATE_RANGE_RX.search(t):
                if i + 1 < len(lines) and len(lines[i + 1].split()) >= 4:
                    out.append(dict(name=strip_dates(t)))
        return out

    # ------------------------------------------------------------ main --
    def parse(self, lines):
        text = "\n".join(lines)
        secs = split_sections(lines)
        header = secs.get("header", []) + secs.get("contact", [])
        res = {}
        nm = self.name(secs.get("header", []) or lines)
        if nm:
            res["name"] = nm
        contact = {}
        e = self.email(text)
        if e:
            contact["email"] = e
        ph = self.phone(header + lines)
        if ph:
            contact["phone"] = ph
        loc = self.location(header or lines, nm)
        if loc:
            contact["location"] = loc
        if contact:
            res["contact"] = contact
        lk = self.links(text)
        if lk:
            res["links"] = [{"url": u} for u in lk]
        if secs.get("education"):
            res["education"] = self.education(secs["education"])
        else:
            res["education"] = self.education(lines)
        if secs.get("experience"):
            res["experience"] = self.experience(secs["experience"])
        if secs.get("skills"):
            res["skills"] = self.skills(secs["skills"])
        if secs.get("projects"):
            res["projects"] = self.projects(secs["projects"])
        if secs.get("certifications"):
            res["certifications"] = self.certifications(secs["certifications"])
        lang_text = "\n".join(secs.get("languages", [])) or ""
        if lang_text:
            res["languages"] = self.languages(lang_text)
        return {k: v for k, v in res.items() if v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--ocr", default="ocr")
    ap.add_argument("--engine", default="tesseract")
    ap.add_argument("--out", default="preds/nlp_only")
    ap.add_argument("--tiers", default="clean,mild,heavy")
    ap.add_argument("--split", default="all")
    ap.add_argument("--order", default="raster", choices=["raster", "tesseract"])
    ap.add_argument("--spacy-model", default="en_core_web_sm", help="en_core_web_sm (fast) or en_core_web_trf (transformer)")
    a = ap.parse_args()
    import spacy
    try:
        nlp = spacy.load(a.spacy_model, disable=["lemmatizer"])
    except OSError:
        sys.exit(f"spaCy model missing. Run:  python -m spacy download {a.spacy_model}")
    parser = Parser(nlp)
    data, ocr, out = Path(a.data), Path(a.ocr) / a.engine, Path(a.out)
    tiers = a.tiers.split(",")
    pages = defaultdict(list)
    for r in csv.DictReader(open(data / "manifest.csv", encoding="utf-8")):
        if r["tier"] in tiers and (a.split == "all" or r["split"] == a.split):
            pages[(r["resume_id"], r["tier"])].append(r)
    t0, n = time.time(), 0
    for (rid, tier), rows in sorted(pages.items()):
        lines = []
        for r in sorted(rows, key=lambda r: int(r["page"])):
            jp = ocr / tier / f"{Path(r['image']).stem}.json"
            if not jp.exists():
                continue
            words = json.loads(jp.read_text(encoding="utf-8"))["words"]
            lines += flatten_raster(words) if a.order == "raster" else flatten_tesseract(words)
        t1 = time.time()
        pred = parser.parse(lines)
        pred = {"resume_id": rid, **pred, "_runtime_s": round(time.time() - t1, 3)}
        (out / tier).mkdir(parents=True, exist_ok=True)
        (out / tier / f"{rid}.json").write_text(json.dumps(pred, ensure_ascii=False, indent=1), encoding="utf-8")
        (out / "_flat" / tier).mkdir(parents=True, exist_ok=True)
        (out / "_flat" / tier / f"{rid}.txt").write_text("\n".join(lines), encoding="utf-8")
        n += 1
        if n % 50 == 0:
            print(f"{n}/{len(pages)} resumes parsed", flush=True)
    print(f"Done: {n} resumes x tiers in {time.time() - t0:.0f}s -> {out.resolve()}")


if __name__ == "__main__":
    main()
