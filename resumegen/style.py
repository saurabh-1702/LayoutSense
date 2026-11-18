"""Per-resume visual style: fonts, colours, spacing, heading synonyms, section order
and the variant of each content block. Randomised but deterministic per seed."""
import colorsys
import random

from .content import wchoice

LAYOUTS = {
    "single_column": ["s1_classic", "s2_modern", "s3_banded"],
    "multi_column": ["m1_sidebar_left", "m2_sidebar_right", "m3_two_equal"],
    "table_based": ["t1_grid", "t2_cv_table", "t3_form"],
    "infographic": ["i1_dark_sidebar", "i2_banner_cards", "i3_timeline_gauges"],
}
TEMPLATE_LAYOUT = {t: l for l, ts in LAYOUTS.items() for t in ts}

# Windows font first (the user renders on Windows), Linux metric-compatible fallback second.
SANS = ["Calibri, Carlito, sans-serif", "Arial, 'Liberation Sans', sans-serif", "'Segoe UI', Inter, sans-serif",
        "Helvetica, 'TeX Gyre Heros', Arial, sans-serif", "Verdana, 'DejaVu Sans', sans-serif", "Poppins, 'Segoe UI', sans-serif",
        "Tahoma, 'DejaVu Sans', sans-serif", "'Trebuchet MS', FreeSans, sans-serif", "Inter, 'Segoe UI', Arial, sans-serif"]
SERIF = ["Cambria, Caladea, serif", "'Times New Roman', 'Liberation Serif', serif", "Georgia, 'DejaVu Serif', serif",
         "Garamond, 'TeX Gyre Pagella', serif", "'Book Antiqua', Palatino, 'TeX Gyre Pagella', serif", "Lora, Georgia, serif",
         "Charter, 'Bitstream Charter', serif", "'Palatino Linotype', 'TeX Gyre Pagella', serif"]
WIDE_FONTS = ("Verdana", "Tahoma", "Poppins")

ACCENTS = ["#1F3A5F", "#0F766E", "#7A1F2B", "#2F5D3A", "#334155", "#1D4ED8", "#5B21B6", "#B45309", "#222222",
           "#2B6CB0", "#9F1239", "#047857", "#0E7490", "#4338CA", "#7C2D12", "#365314", "#1E293B", "#A21CAF"]

HEADINGS = {
    "experience": ["Experience", "Work Experience", "Professional Experience", "Employment History", "Career History", "Work History", "Professional Background"],
    "internships": ["Internships", "Internship Experience", "Experience", "Industry Exposure"],
    "education": ["Education", "Academic Background", "Education & Qualifications", "Academic Qualifications", "Educational Qualifications"],
    "skills": ["Skills", "Technical Skills", "Core Competencies", "Key Skills", "Areas of Expertise", "Skills & Tools", "Competencies"],
    "projects": ["Projects", "Key Projects", "Selected Projects", "Academic Projects", "Notable Projects"],
    "certifications": ["Certifications", "Licenses & Certifications", "Professional Certifications", "Certificates", "Courses & Certifications"],
    "awards": ["Awards", "Honours & Awards", "Achievements", "Awards & Recognition", "Honors"],
    "languages": ["Languages", "Language Proficiency", "Languages Known"],
    "links": ["Links", "Online Profiles", "Profiles", "Social"],
    "contact": ["Contact", "Contact Details", "Contact Information", "Get in Touch", "Personal Details"],
}


def tint(hex_color, amount):
    """Mix colour with white. amount 0..1 (1 = white)."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (int(c + (255 - c) * amount) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def make_style(template, person, seed, shrink=0):
    rng = random.Random(seed * 7919 + 13)
    layout = TEMPLATE_LAYOUT[template]
    serif = rng.random() < (0.45 if layout in ("single_column", "table_based") else 0.2)
    font = rng.choice(SERIF if serif else SANS)
    head_font = font if rng.random() < 0.7 else rng.choice(SANS if serif else SERIF)
    wide = any(w in font for w in WIDE_FONTS)
    base = rng.uniform(13.4, 15.3) - (1.0 if wide else 0)
    base -= 0.55 * shrink
    accent = rng.choice(ACCENTS)
    a4 = person["page_size"] == "A4"
    s = dict(
        template=template, layout=layout,
        pw=794 if a4 else 816, ph=1123 if a4 else 1056,
        font=font, head_font=head_font, fs=round(base, 2), lh=round(rng.uniform(1.22, 1.42), 2),
        ink=rng.choice(["#111111", "#1a1a1a", "#222222", "#262626", "#1f2933"]),
        muted=rng.choice(["#555555", "#5b6470", "#4b5563", "#666666"]),
        accent=accent, accent2=tint(accent, rng.uniform(0.82, 0.9)), track=tint(accent, 0.8),
        band_ink=accent if rng.random() < 0.6 else "#111111",
        name_color=accent if rng.random() < 0.4 else "#111111",
        bullet=rng.choice(["•", "•", "•", "–", "▪", "›", "◦", "‣"]),
        rule=rng.choice([1, 1, 2]), rule_color=rng.choice([accent, "#999999", "#333333"]),
        radius=rng.choice([3, 10, 12, 2]),
        mt=rng.randint(30, 50), mb=rng.randint(26, 40), mx=rng.randint(38, 62),
        name_fs=rng.randint(24, 32), name_ls=rng.choice([0, 0, 0.5, 1, 2]),
        name_upper=rng.random() < 0.3, show_headline=rng.random() < 0.8,
        sec_gap=rng.randint(7, 12), indent=rng.choice([0, 6, 10, 14]),
        sep=rng.choice([" | ", " • ", "  ·  ", " | ", "  |  ", " ◦ "]), sep2=rng.choice([" | ", ", ", " — ", " · ", " @ "]),
        skill_sep=rng.choice([", ", ", ", " • ", " | ", " · "]), lang_sep=rng.choice([", ", " • ", "; ", " | "]),
        show_loc=rng.random() < 0.85, icons=rng.random() < 0.5,
        tech_label=rng.choice(["Tech:", "Technologies:", "Stack:", "Tools:", "Tools used:"]),
        exp_v=rng.choice(["title_first", "title_first", "company_first", "inline", "stack"]),
        edu_v=rng.choice(["degree_first", "degree_first", "inst_first", "stack"]),
        skills_v=rng.choice(["grouped", "grouped", "inline", "tags"]),
        side_w=rng.randint(232, 268), shrink=shrink,
    )
    if person["experience"] and all(e.get("level", 1) == 0 for e in person["experience"]):
        heads = dict(HEADINGS, experience=HEADINGS["internships"])
    else:
        heads = HEADINGS
    upper = rng.random() < 0.5
    s["h"] = {k: (rng.choice(v).upper() if upper else rng.choice(v)) for k, v in heads.items()}
    s["bullet_color"] = accent if rng.random() < 0.5 else s["ink"]

    fresher = person["level"] == "fresher"
    order = ["education", "skills", "projects", "experience"] if fresher else ["experience", "education", "skills", "projects"]
    if not fresher and rng.random() < 0.3:
        order = ["skills", "experience", "projects", "education"]
    if fresher and rng.random() < 0.4:
        order = ["education", "experience", "projects", "skills"]
    tail = ["certifications", "awards", "languages"]
    rng.shuffle(tail)
    s["order"] = order + tail

    # two-column templates: which sections live in the side column
    side_pool = ["skills", "education", "certifications", "languages"]
    if rng.random() < 0.5:
        side_pool.append("awards")
    rng.shuffle(side_pool)
    s["order_side"] = ["contact"] + side_pool + ["links"]
    main = ["experience", "projects"] + [x for x in ["awards", "education", "certifications"] if x not in side_pool]
    if fresher:
        main = ["projects", "experience"] + [x for x in main if x not in ("projects", "experience")]
    s["order_main"] = main

    # infographic / table specific knobs
    s["col_dur"] = rng.choice(["Duration", "Period", "Tenure", "Dates"])
    s["col_score"] = rng.choice(["Score", "CGPA / %", "Grade", "Result"]) if person.get("india") else rng.choice(["Grade", "Result", "GPA / Honours"])
    s["col_board"] = rng.choice(["Institution / University", "Board / University", "College / University"]) if person.get("india") else "Institution"
    s["lcol"] = rng.randint(118, 150)
    s["pi_label"] = rng.choice(["Personal information", "", "Curriculum Vitae", ""])
    s["resp_label"] = rng.choice(["Key Responsibilities:", "Responsibilities & Achievements:", "Key Contributions:"])
    s["side_hi"] = rng.choice(["#ffffff", "#fbbf24", "#5eead4", "#93c5fd", "#fca5a5"])
    s["datecol"] = rng.randint(96, 118) if s["fs"] > 13.4 else rng.randint(90, 108)
    st = []
    if person["years_experience"] > 0:
        st.append((f"{person['years_experience']}+", "Years of Experience"))
    else:
        st.append((str(len(person["projects"])), "Projects"))
    st.append((str(len({e["company"] for e in person["experience"]})), "Companies" if person["years_experience"] else "Internships"))
    st.append((str(len(person["certifications"])) if person["certifications"] else str(len(person["skills"])), "Certifications" if person["certifications"] else "Skills"))
    st.append((str(len(person["languages"])), "Languages"))
    s["stats"] = st[:3] if rng.random() < 0.6 else [st[0], st[2], st[3]]
    s["side_bg"] = rng.choice([accent, "#1f2937", "#243447", "#2d2d2d", tint(accent, 0.12) if False else accent])
    s["side_skills"] = rng.choice(["bars", "dots", "list"])
    s["decl"] = person.get("india") and rng.random() < 0.7
    s["stripe"] = tint(accent, 0.93)
    s["th_bg"] = rng.choice([accent, "#e5e7eb", tint(accent, 0.75)])
    s["th_ink"] = "#ffffff" if s["th_bg"] == accent else "#111111"
    s["border"] = rng.choice(["#444444", "#666666", "#999999", accent])
    s["initials"] = "".join(w[0] for w in person["name"].replace(".", "").split()[:2]).upper()
    pct = []
    for i, lv in enumerate(person["skill_levels"]):
        pct.append(min(98, lv * 18 + (seed * 31 + i * 7) % 9))
    s["skill_pct"] = pct
    lvl_words = rng.choice([["Beginner", "Elementary", "Intermediate", "Advanced", "Expert"], ["Basic", "Basic", "Intermediate", "Proficient", "Expert"]])
    s["skill_word"] = [lvl_words[lv - 1] for lv in person["skill_levels"]]
    return s
