import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="MedCheck Advanced",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CSS
# =========================================================
CUSTOM_CSS = """
<style>
    .main {
        background:
            radial-gradient(circle at top left, rgba(37, 99, 235, 0.14), transparent 26%),
            radial-gradient(circle at top right, rgba(168, 85, 247, 0.12), transparent 22%),
            linear-gradient(180deg, #020617 0%, #081120 50%, #0b1220 100%);
    }

    .block-container {
        padding-top: 1.15rem;
        padding-bottom: 2rem;
        max-width: 1280px;
    }

    .hero {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.98) 0%, rgba(30, 58, 138, 0.98) 58%, rgba(59, 130, 246, 0.96) 100%);
        padding: 2rem;
        border-radius: 28px;
        color: white;
        box-shadow: 0 20px 45px rgba(2, 6, 23, 0.5);
        margin-bottom: 1.4rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .hero-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1rem;
        flex-wrap: wrap;
    }

    .hero-title-wrap {
        display: flex;
        align-items: center;
        gap: 0.95rem;
    }

    .hero-icon {
        width: 58px;
        height: 58px;
        border-radius: 18px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.7rem;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.18);
    }

    .hero h1 {
        margin: 0;
        font-size: 2.35rem;
        line-height: 1.05;
        color: #ffffff;
        letter-spacing: -0.02em;
    }

    .hero-subtitle {
        margin-top: 0.2rem;
        color: rgba(226,232,240,0.95);
        font-size: 0.98rem;
    }

    .hero p {
        margin-top: 1rem;
        margin-bottom: 0;
        color: #e2e8f0;
        font-size: 1rem;
        max-width: 970px;
    }

    .hero-badges {
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
    }

    .hero-badge {
        padding: 0.45rem 0.82rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.18);
        color: #f8fafc;
        font-size: 0.85rem;
        font-weight: 700;
    }

    .card {
        background: rgba(255,255,255,0.98);
        border-radius: 22px;
        padding: 1.2rem;
        box-shadow: 0 10px 30px rgba(2, 6, 23, 0.16);
        border: 1px solid rgba(148, 163, 184, 0.16);
        color: #0f172a;
    }

    .result-pill {
        display: inline-block;
        padding: 0.45rem 0.9rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 0.75rem;
    }

    .credible {
        background: #dcfce7;
        color: #166534;
    }

    .possibly {
        background: #fef3c7;
        color: #92400e;
    }

    .misleading {
        background: #fee2e2;
        color: #991b1b;
    }

    .nodecision {
        background: #e2e8f0;
        color: #334155;
    }

    .small-muted {
        color: #475569;
        font-size: 0.93rem;
    }

    .summary-box {
        background: #ffffff;
        border-left: 6px solid #2563eb;
        border-radius: 18px;
        padding: 1rem 1.1rem;
        color: #0f172a;
        box-shadow: 0 10px 30px rgba(2, 6, 23, 0.12);
        line-height: 1.65;
        margin-bottom: 0.75rem;
    }

    .explain-box {
        background: rgba(255,255,255,0.98);
        border: 1px solid rgba(148,163,184,0.18);
        border-radius: 18px;
        padding: 1rem;
        color: #0f172a;
        box-shadow: 0 10px 30px rgba(2, 6, 23, 0.12);
        line-height: 1.65;
    }

    .stTextArea textarea {
        background: #111827 !important;
        color: #f8fafc !important;
        border-radius: 16px !important;
        border: 1px solid rgba(148,163,184,0.24) !important;
    }

    .stButton > button {
        border-radius: 14px !important;
        font-weight: 700 !important;
    }

    section[data-testid="stSidebar"] * {
        color: #e5e7eb !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =========================================================
# EXPANDED HEALTH VOCABULARY
# =========================================================
HEALTH_KEYWORDS = {
    # general medical / healthcare
    "health", "healthy", "unhealthy", "healthcare", "medical", "medicine", "medicines",
    "medication", "medications", "drug", "drugs", "doctor", "doctors", "nurse",
    "hospital", "clinic", "patient", "patients", "diagnosis", "diagnose", "diagnosed",
    "disease", "diseases", "illness", "illnesses", "condition", "conditions",
    "symptom", "symptoms", "treatment", "treatments", "therapy", "therapies",
    "care", "prevention", "prevent", "prevents", "preventing", "risk", "risks",
    "side", "effect", "effects", "side-effect", "side-effects", "infection",
    "infections", "infectious", "bacteria", "bacterial", "virus", "viral",
    "fungus", "fungal", "parasite", "parasitic", "immune", "immunity",
    "immunization", "inflammation", "pain", "fever", "injury", "injuries",
    "wound", "wounds", "antibiotic", "antibiotics", "prescription", "dose",
    "dosage", "overdose", "toxicity", "toxic", "poison", "poisoning",

    # public health
    "covid", "covid19", "covid-19", "coronavirus", "sars", "pandemic",
    "epidemic", "outbreak", "quarantine", "isolation", "mask", "masks",
    "vaccine", "vaccines", "vaccination", "booster", "boosters", "immunized",
    "transmission", "contagious", "infected", "infect", "infects", "spread",
    "spreads", "variant", "variants",

    # diseases / conditions
    "cancer", "tumor", "tumour", "diabetes", "insulin", "glucose", "sugar",
    "hypertension", "blood", "pressure", "heart", "cardiac", "stroke",
    "kidney", "kidneys", "renal", "liver", "hepatic", "lung", "lungs",
    "respiratory", "asthma", "cholesterol", "obesity", "overweight",
    "anemia", "anaemia", "arthritis", "migraine", "seizure", "epilepsy",
    "allergy", "allergies", "allergic", "flu", "influenza", "cold", "cough",
    "pneumonia", "tuberculosis", "tb", "malaria", "dengue", "measles",
    "rabies", "hiv", "aids", "hepatitis", "diarrhea", "diarrhoea",
    "constipation", "ulcer", "acid", "reflux", "gerd",

    # mental health
    "mental", "depression", "depressed", "anxiety", "anxious", "stress",
    "stressed", "trauma", "ptsd", "therapy", "counseling", "suicide",
    "self-harm", "panic", "insomnia", "sleep", "sleeping", "rest",

    # reproductive / urinary / digestive
    "pregnancy", "pregnant", "fertility", "period", "menstruation",
    "menstrual", "contraceptive", "contraception", "urinary", "urine",
    "uti", "utis", "bladder", "urethra", "urination", "kidney", "stones",
    "digestive", "digestion", "stomach", "intestine", "gut", "bowel",
    "probiotic", "probiotics", "fiber", "fibre",

    # nutrition / diet / lifestyle
    "diet", "diets", "nutrition", "nutritional", "food", "foods", "meal",
    "meals", "eating", "eat", "eats", "drink", "drinks", "beverage",
    "beverages", "water", "hydration", "hydrate", "hydrates", "hydrated",
    "dehydration", "dehydrate", "dehydrated", "fluid", "fluids", "salt",
    "salty", "sodium", "protein", "proteins", "vitamin", "vitamins",
    "mineral", "minerals", "supplement", "supplements", "calcium",
    "iron", "zinc", "magnesium", "potassium", "carbohydrate",
    "carbohydrates", "carb", "carbs", "fat", "fats", "cholesterol",
    "calorie", "calories", "metabolism", "metabolic", "weight", "loss",
    "weightloss", "slimming", "obesity", "overweight", "belly", "waist",
    "appetite", "hunger", "hungry", "fasting", "intermittent", "skip",
    "skipping", "skipped", "breakfast", "lunch", "dinner", "snack",
    "snacks", "lemon", "lime", "citrus", "detox", "cleanse", "cleansing",
    "burn", "burns", "burning", "fat-burning", "fatburning",

    # lifestyle / physical activity
    "exercise", "exercises", "exercising", "physical", "activity",
    "workout", "workouts", "fitness", "sport", "sports", "running",
    "walking", "jogging", "gym", "training", "muscle", "muscles",
    "cramp", "cramps", "spasm", "spasms", "soreness", "sore",
    "shower", "showers", "showering", "bath", "baths", "bathing",

    # habits / substances
    "alcohol", "smoking", "smoke", "tobacco", "cigarette", "cigarettes",
    "vape", "vaping", "nicotine", "addiction", "drug", "drugs",

    # claim/effect language
    "cause", "causes", "caused", "causing", "linked", "link", "links",
    "associated", "association", "increase", "increases", "decrease",
    "decreases", "reduce", "reduces", "prevent", "prevents", "cure",
    "cures", "heals", "healing", "remedy", "deadly", "fatal", "mortality",
    "death", "deaths", "kills", "harm", "harmful", "benefit", "benefits",
    "improves", "improve", "boosts", "boost", "weakens", "strengthens",
}


# =========================================================
# EXPANDED NON-HEALTH VOCABULARY
# =========================================================
NON_HEALTH_KEYWORDS = {
    # technology
    "laptop", "laptops", "mobile", "phone", "phones", "smartphone", "smartphones",
    "iphone", "android", "computer", "computers", "tablet", "tablets",
    "keyboard", "mouse", "printer", "monitor", "gadget", "gadgets", "device",
    "devices", "app", "apps", "software", "hardware", "internet", "wifi",
    "router", "camera", "cameras", "charger", "battery", "batteries",
    "screen", "screens", "processor", "cpu", "gpu", "ram", "ssd", "hdd",

    # entertainment
    "movie", "movies", "film", "films", "music", "song", "songs", "album",
    "anime", "manga", "game", "games", "gaming", "netflix", "youtube",
    "tiktok", "celebrity", "actor", "actress", "concert", "band", "singer",

    # sports
    "basketball", "football", "soccer", "volleyball", "tennis", "baseball",
    "nba", "fifa", "team", "teams", "player", "players", "score", "scores",

    # finance / politics / legal
    "money", "bank", "banks", "stock", "stocks", "crypto", "bitcoin",
    "ethereum", "trading", "investment", "invest", "loan", "loans",
    "election", "elections", "politics", "president", "senator", "law",
    "laws", "court", "judge", "government", "tax", "taxes",

    # general / product comparisons
    "car", "cars", "motorcycle", "motorcycles", "travel", "restaurant",
    "restaurants", "fashion", "clothes", "shoes", "bag", "bags", "school",
    "homework", "assignment", "weather", "rain", "sunny", "cloudy",
    "better", "best", "worse", "worst", "comparison", "compare",
}


ANIMAL_TERMS = {
    "dog", "dogs", "cat", "cats", "pet", "pets", "puppy", "puppies",
    "kitten", "kittens", "animal", "animals", "veterinary", "vet",
    "horse", "horses", "cow", "cows", "goat", "goats", "bird", "birds",
    "rabbit", "rabbits", "hamster", "hamsters", "fish", "livestock",
}


HUMAN_TERMS = {
    "person", "people", "human", "humans", "patient", "patients", "man",
    "woman", "men", "women", "child", "children", "adult", "adults",
    "teen", "teens", "elderly", "body", "bodies", "muscle", "muscles",
    "athlete", "athletes", "student", "students", "someone", "individual",
    "individuals", "person's", "people's",
}


EXAGGERATED_TERMS = {
    "miracle", "instant", "instantly", "guaranteed", "always", "never",
    "100%", "permanent", "permanently", "works for everyone", "totally safe",
    "no side effects", "secret", "overnight", "completely", "proven cure",
    "doctors hate", "hidden truth", "one weird trick", "shocking", "magical",
    "amazing cure", "cure-all", "superfood", "detoxifies", "flushes toxins",
}


ABSOLUTE_TERMS = {
    "always", "never", "everyone", "all", "completely", "guaranteed",
    "totally", "permanent", "permanently", "100%", "zero risk", "no risk",
}


CURE_TERMS = {
    "cure", "cures", "cured", "treat", "treats", "treated", "treatment",
    "heals", "healing", "remedy", "remedies", "eliminates", "reverses",
    "fixes", "flushes", "detoxifies",
}


EVIDENCE_TERMS = {
    "study", "studies", "evidence", "research", "clinical", "trial",
    "trials", "journal", "scientific", "peer reviewed", "peer-reviewed",
    "review", "meta analysis", "meta-analysis", "guideline", "guidelines",
    "recommended", "medical evidence", "according to", "data", "researchers",
    "systematic", "randomized", "randomised", "experiment", "experiments",
    "published", "expert", "experts", "official", "authority", "authorities",
}


HIGH_CREDIBILITY_SOURCES = {
    "who", "world health organization", "doh", "department of health",
    "cdc", "centers for disease control", "nih", "national institutes of health",
    "pubmed", "ncbi", "mayo clinic", "thelancet", "lancet", "bmj", "nejm",
    "jamanetwork", "jama", "cleveland clinic", "johns hopkins", "fda",
    "harvard health", "nhs", "medlineplus", "healthnewsreview", "cochrane",
    "webmd", "healthline", "medicalnewstoday",
}


FACT_CHECKER_SOURCES = {
    "politifact", "snopes", "factcheck", "factcheck.org", "fullfact",
    "reuters", "apnews", "ap news", "health feedback", "science feedback",
    "usa today", "lead stories", "afp", "rappler", "boomlive",
}


LOW_CREDIBILITY_SOURCE_HINTS = {
    "unknown source", "chain message", "viral post", "forwarded message",
    "facebook post", "random blog", "unverified", "tiktok", "youtube comment",
    "whatsapp", "telegram", "anonymous", "rumor", "rumour", "hearsay",
    "no source", "not stated",
}


SUPPORTIVE_CONTENT_PHRASES = {
    "does cause", "can cause", "is associated with", "is linked to",
    "increases the risk", "raises the risk", "evidence shows",
    "studies show", "according to", "is true", "confirmed", "supported",
    "risk factor", "caused by", "leads to", "contributes to",
    "can be deadly", "causes death", "fatal", "mortality", "linked with",
    "associated with", "helps reduce", "reduce risk", "lower risk",
    "helps maintain", "important for hydration", "supports hydration",
    "helps hydration", "prevents dehydration", "can improve hydration",
}


REFUTING_CONTENT_PHRASES = {
    "does not cause", "do not cause", "no evidence", "not true", "false",
    "misleading", "unsupported", "debunked", "myth", "not supported",
    "lacks evidence", "no scientific evidence", "unproven", "incorrect",
    "exaggerated", "has not confirmed", "not confirmed", "did not confirm",
    "not proven", "does not show", "do not show", "not enough evidence",
    "widely disputed", "disputed", "misrepresents", "out of context",
    "not recommended", "not necessarily", "oversimplified",
}


NEGATION_WORDS = {
    "not", "no", "never", "without", "lacks", "lack", "unsupported",
    "unproven", "false", "cannot", "can't", "doesnt", "doesn't",
    "isnt", "isn't", "dont", "don't",
}


FALSE_LABELS = {
    "false", "mostly false", "pants on fire", "fake", "incorrect",
}


MISLEADING_LABELS = {
    "misleading", "partly false", "partially false", "half true", "mixed",
    "mixture", "unclear", "possibly misleading", "needs review",
}


CREDIBLE_LABELS = {
    "true", "mostly true", "correct", "credible", "supported",
}


# =========================================================
# SYNONYM / NORMALIZATION PATTERNS
# =========================================================
SYNONYM_PATTERNS = {
    r"\bcovid19\b": "covid",
    r"\bcovid 19\b": "covid",
    r"\bcovid-19\b": "covid",
    r"\bcoronavirus\b": "covid",

    r"\bsmoking\b": "tobacco smoking",
    r"\bsmoke\b": "tobacco smoking",
    r"\bcigarettes?\b": "tobacco smoking",
    r"\btobacco\b": "tobacco smoking",

    r"\bcauses?\b": "cause",
    r"\bcaused\b": "cause",
    r"\bcausing\b": "cause",
    r"\bprevents?\b": "prevent",
    r"\bprevented\b": "prevent",
    r"\bprotects?\b": "prevent",
    r"\bcures?\b": "cure",
    r"\bcured\b": "cure",

    r"\bheart attack\b": "heart disease",
    r"\bhigh blood pressure\b": "hypertension",
    r"\bblood sugar\b": "glucose",
    r"\bweight loss\b": "weight loss fat loss",
    r"\blosing weight\b": "weight loss fat loss",
    r"\blose weight\b": "weight loss fat loss",
    r"\bfat loss\b": "weight loss fat loss",

    r"\bphysical activity\b": "exercise",
    r"\bworkout\b": "exercise",
    r"\bworkouts\b": "exercise",
    r"\bworking out\b": "exercise",
    r"\bexercising\b": "exercise",

    r"\bspasms\b": "spasm",
    r"\bcramps\b": "cramp",
    r"\bshowering\b": "shower",
    r"\bbathing\b": "bath",
    r"\bbaths\b": "bath",

    r"\buti's\b": "urinary tract infection uti",
    r"\buti s\b": "urinary tract infection uti",
    r"\butis\b": "urinary tract infection uti",
    r"\buti\b": "urinary tract infection uti",
    r"\burinary tract infections?\b": "urinary tract infection uti",

    r"\bhydrated\b": "hydration",
    r"\bhydrate\b": "hydration",
    r"\bhydrates\b": "hydration",
    r"\bdehydrated\b": "dehydration",
    r"\bdehydrate\b": "dehydration",

    r"\bsalty\b": "salt sodium",
    r"\bsodium\b": "salt sodium",

    r"\bburns fat\b": "burn fat metabolism",
    r"\bburn fat\b": "burn fat metabolism",
    r"\bfat burning\b": "burn fat metabolism",
    r"\bfat-burning\b": "burn fat metabolism",
    r"\bfatburning\b": "burn fat metabolism",

    r"\bskipping meals\b": "skip meal fasting",
    r"\bskip meals\b": "skip meal fasting",
    r"\bskipped meals\b": "skip meal fasting",
    r"\bskipping breakfast\b": "skip meal fasting breakfast",
    r"\bintermittent fasting\b": "fasting intermittent fasting",
}


# =========================================================
# EXTRA EXPANDED LEXICAL RULES
# These update blocks keep the code readable while expanding coverage.
# =========================================================
HEALTH_KEYWORDS.update({
    # body systems / anatomy
    "brain", "nervous", "nerve", "nerves", "spine", "spinal", "bone", "bones", "joint", "joints",
    "skin", "dermatology", "rash", "eczema", "psoriasis", "acne", "hair", "scalp", "eye", "eyes",
    "vision", "ear", "ears", "hearing", "nose", "throat", "thyroid", "hormone", "hormones",
    "endocrine", "pancreas", "colon", "rectal", "prostate", "breast", "cervical", "ovarian",
    "immune system", "lymph", "lymph nodes", "artery", "arteries", "vein", "veins",

    # additional diseases / conditions
    "alzheimer", "dementia", "parkinson", "autism", "adhd", "bipolar", "schizophrenia",
    "osteoporosis", "copd", "bronchitis", "sinusitis", "appendicitis", "sepsis", "meningitis",
    "std", "sti", "chlamydia", "gonorrhea", "syphilis", "hpv", "herpes", "pcos", "endometriosis",
    "eczema", "psoriasis", "gout", "arthritis", "backpain", "backache", "headache", "nausea",
    "vomiting", "fatigue", "dizziness", "vertigo", "palpitation", "palpitations",

    # medications / health products
    "aspirin", "ibuprofen", "paracetamol", "acetaminophen", "insulin", "metformin", "antacid",
    "steroid", "steroids", "antihistamine", "antiviral", "antifungal", "deworming", "vaccine",
    "herbal", "herb", "tea", "essential oil", "ointment", "cream", "capsule", "tablet", "pill", "syrup",

    # nutrition / food claims
    "keto", "ketogenic", "vegan", "vegetarian", "paleo", "low-carb", "lowcarb", "low fat", "low-fat",
    "fiber", "fibre", "probiotic", "prebiotic", "omega", "omega-3", "fish oil", "green tea",
    "apple cider", "vinegar", "ginger", "garlic", "turmeric", "honey", "banana", "coffee", "caffeine",
    "energy drink", "softdrink", "soda", "juice", "milk", "dairy", "calorie deficit", "bmi",

    # symptoms / effects / risk language
    "relieves", "relief", "worsens", "worsen", "triggers", "trigger", "improves", "improve",
    "boost", "boosts", "lower", "lowers", "raise", "raises", "reduce", "reduces", "increase", "increases",
    "protect", "protects", "risk factor", "side effects", "complication", "complications", "safe", "unsafe",
})

NON_HEALTH_KEYWORDS.update({
    # more technology / gadgets
    "bluetooth", "headphones", "earbuds", "speaker", "television", "tv", "remote", "console", "playstation",
    "xbox", "nintendo", "website", "browser", "email", "gmail", "facebook", "instagram", "twitter", "x", "discord",
    "coding", "programming", "python", "java", "javascript", "database", "server", "cloud", "ai", "chatbot",

    # school/work not health-related by default
    "quiz", "exam", "grades", "teacher", "professor", "class", "course", "lesson", "module", "project", "report",
    "resume", "job", "salary", "office", "meeting", "deadline", "presentation", "document", "printer",

    # household / products / places
    "chair", "table", "bed", "room", "house", "apartment", "rent", "tenant", "landlord", "mall", "store",
    "shopping", "delivery", "package", "shipping", "brand", "price", "discount", "sale", "coupon",

    # arts / hobbies
    "painting", "drawing", "guitar", "piano", "dance", "photography", "camera", "editing", "vlog", "podcast",
})

EXAGGERATED_TERMS.update({
    "life-changing", "revolutionary", "breakthrough", "secret formula", "ancient secret", "natural cure",
    "doctor-approved miracle", "clinically proven miracle", "magic", "guaranteed results", "lose weight fast",
    "melt fat", "burn fat overnight", "flush fat", "detox your body", "toxins", "immune booster miracle",
})

ABSOLUTE_TERMS.update({
    "every time", "for all people", "for everyone", "no side effect", "no side effects", "fully safe",
    "completely safe", "completely cures", "permanent cure", "works instantly",
})

CURE_TERMS.update({
    "cure-all", "detox", "cleanse", "purify", "boost immunity", "immune boost", "melt", "burn", "burns",
    "remove toxins", "flush toxins", "natural remedy", "home remedy",
})

EVIDENCE_TERMS.update({
    "systematic review", "cohort", "case-control", "observational", "double blind", "placebo", "sample size",
    "statistically significant", "confidence interval", "risk ratio", "odds ratio", "medical guideline",
    "health authority", "peer reviewed journal", "clinical evidence", "expert consensus",
})

SUPPORTIVE_CONTENT_PHRASES.update({
    "may help", "can help", "is recommended", "has been shown", "research suggests", "clinical evidence suggests",
    "official guidance", "health authorities recommend", "associated with lower risk", "associated with higher risk",
    "supports normal function", "helps prevent", "can reduce", "may reduce", "effective for", "approved for",
})

REFUTING_CONTENT_PHRASES.update({
    "no reliable evidence", "little evidence", "not enough reliable evidence", "not medically proven",
    "not clinically proven", "not recommended by experts", "not a substitute", "does not directly cause",
    "does not directly cure", "does not burn fat", "no proof", "lack of proof", "claim is exaggerated",
    "claim is overstated", "causal link is not established", "correlation does not prove causation",
})

NEGATION_WORDS.update({
    "hardly", "rarely", "unlikely", "doubtful", "insufficient", "inconclusive", "limited", "unclear",
})

SYNONYM_PATTERNS.update({
    r"\bhigh bp\b": "hypertension blood pressure",
    r"\bbp\b": "blood pressure",
    r"\bheart attacks?\b": "heart disease",
    r"\bstroke risk\b": "stroke risk",
    r"\bblood glucose\b": "glucose",
    r"\bsugar level\b": "glucose",
    r"\bsugar levels\b": "glucose",
    r"\blose fat\b": "weight loss fat loss",
    r"\bmelts fat\b": "burn fat metabolism",
    r"\bmelt fat\b": "burn fat metabolism",
    r"\bflushes fat\b": "burn fat metabolism",
    r"\bdetox(es)?\b": "detox cleanse",
    r"\bapple cider vinegar\b": "apple cider vinegar diet weight loss",
    r"\bgreen tea\b": "green tea diet metabolism",
    r"\bginger tea\b": "ginger tea digestion",
    r"\bturmeric\b": "turmeric anti inflammatory",
    r"\bgarlic\b": "garlic health nutrition",
    r"\bvit c\b": "vitamin c",
    r"\bvitamin c\b": "vitamin c immunity",
    r"\bimmune booster\b": "immune boost immunity",
    r"\bboosts immunity\b": "immune boost immunity",
    r"\bboost immunity\b": "immune boost immunity",
    r"\bhangover cure\b": "alcohol hangover remedy",
    r"\bsleep deprivation\b": "sleep lack sleep",
    r"\bmental health\b": "mental health depression anxiety stress",
    r"\bdepression\b": "mental health depression",
    r"\banxiety\b": "mental health anxiety",
    r"\bstds\b": "sexually transmitted infection sti std",
    r"\bstis\b": "sexually transmitted infection sti std",
    r"\bhiv aids\b": "hiv aids infection",
    r"\bbirth control\b": "contraception contraceptive",
    r"\bpregnant\b": "pregnancy pregnant",
    r"\bkidney stone\b": "kidney stones renal",
    r"\bacid reflux\b": "acid reflux gerd",
})


# =========================================================
# BASIC UTILITIES
# =========================================================
def safe_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", safe_text(text)).strip()


def clean_text(text: str) -> str:
    text = safe_text(text).lower()
    text = text.replace("&", " and ")
    text = text.replace("’", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9%\s']", " ", text)
    text = normalize_whitespace(text)

    for pattern, replacement in SYNONYM_PATTERNS.items():
        text = re.sub(pattern, replacement, text)

    text = re.sub(r"[^a-z0-9%\s]", " ", text)
    return normalize_whitespace(text)


def words_set(text: str) -> set:
    return set(clean_text(text).split())


def contains_any(text: str, terms: set) -> bool:
    cleaned = clean_text(text)
    return any(term in cleaned for term in terms)


def find_hits(text: str, terms: set) -> List[str]:
    cleaned = clean_text(text)
    hits = []
    for term in terms:
        if term in cleaned:
            hits.append(term)
    return hits


def confidence_from_score(score: float) -> int:
    score = max(0.0, min(1.0, float(score)))
    return int(round(score * 100))


def match_strength_percent(score: float) -> str:
    return f"{confidence_from_score(score)}%"


def result_class_name(result: str) -> str:
    if result == "Credible":
        return "credible"
    if result == "Misleading":
        return "misleading"
    if result == "No Decision":
        return "nodecision"
    return "possibly"


def display_label(label: str) -> str:
    if label == "Possibly Misleading":
        return "Possibly Misleading (Uncertain)"
    return label


def display_label_dict(label_counts: Dict[str, int]) -> Dict[str, int]:
    return {display_label(k): v for k, v in label_counts.items()}


def classify_label(raw_label: str) -> str:
    label = safe_text(raw_label).lower().strip()

    if label in CREDIBLE_LABELS:
        return "Credible"
    if label in FALSE_LABELS:
        return "Misleading"
    if label in MISLEADING_LABELS:
        return "Possibly Misleading"

    if "true" in label and "false" not in label:
        return "Credible"
    if "false" in label:
        return "Misleading"
    if "misleading" in label or "mixed" in label or "unclear" in label:
        return "Possibly Misleading"

    return "Possibly Misleading"


def sentence_split(text: str) -> List[str]:
    text = normalize_whitespace(text)
    if not text:
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def phrase_is_negated(text: str, phrase: str, window: int = 5) -> bool:
    words = clean_text(text).split()
    phrase_words = clean_text(phrase).split()

    if not phrase_words:
        return False

    for i in range(len(words) - len(phrase_words) + 1):
        if words[i:i + len(phrase_words)] == phrase_words:
            start = max(0, i - window)
            context = words[start:i]
            if any(word in NEGATION_WORDS for word in context):
                return True

    return False


def non_negated_phrase_hits(text: str, phrases: set) -> List[str]:
    hits = []
    cleaned = clean_text(text)

    for phrase in phrases:
        if phrase in cleaned and not phrase_is_negated(text, phrase):
            hits.append(phrase)

    return hits


def negated_support_hits(text: str, support_phrases: set) -> List[str]:
    hits = []
    cleaned = clean_text(text)

    for phrase in support_phrases:
        if phrase in cleaned and phrase_is_negated(text, phrase):
            hits.append(phrase)

    return hits


def extract_urls(text: str) -> List[str]:
    return re.findall(r"https?://[^\s,]+", safe_text(text))


def extract_domains(source: str) -> List[str]:
    domains = []

    for url in extract_urls(source):
        try:
            netloc = urlparse(url).netloc.lower()
            if netloc.startswith("www."):
                netloc = netloc[4:]
            if netloc:
                domains.append(netloc)
        except Exception:
            pass

    raw = safe_text(source).lower()
    for token in re.split(r"[\s,]+", raw):
        token = token.strip()
        if "." in token and len(token) > 4:
            token = token.replace("https://", "").replace("http://", "").replace("www.", "")
            token = token.split("/")[0]
            domains.append(token)

    return list(dict.fromkeys(domains))


def parse_date_safely(date_text: str) -> Optional[datetime]:
    date_text = safe_text(date_text).strip()

    if not date_text or date_text.lower() in {"nan", "none", "unknown", "not available"}:
        return None

    common_formats = [
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%d-%b-%y",
        "%d-%b-%Y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%Y/%m/%d",
    ]

    for fmt in common_formats:
        try:
            return datetime.strptime(date_text[:30], fmt)
        except ValueError:
            pass

    try:
        parsed = pd.to_datetime(date_text, errors="coerce")
        if pd.isna(parsed):
            return None
        return parsed.to_pydatetime()
    except Exception:
        return None


def is_time_sensitive_claim(cleaned_claim: str) -> bool:
    time_sensitive_terms = {
        "covid", "vaccine", "variant", "outbreak", "treatment", "drug",
        "medicine", "guideline", "recommendation", "pandemic", "mask",
        "booster", "infection", "virus", "viral",
    }
    return contains_any(cleaned_claim, time_sensitive_terms)


# =========================================================
# HEALTH CONCEPTS
# =========================================================
HEALTH_CONCEPTS = {
    "hydration": {
        "hydration", "water", "fluid", "fluids", "dehydration", "hydrate",
        "dehydrated", "drink", "drinks",
    },
    "uti": {
        "urinary", "tract", "infection", "uti", "bladder", "urine",
        "urethra", "urination",
    },
    "salt_sodium": {
        "salt", "sodium", "salty",
    },
    "exercise": {
        "exercise", "physical", "activity", "workout", "fitness", "gym",
        "training", "running", "walking", "jogging",
    },
    "shower_bath": {
        "shower", "bath", "bathing", "cold", "hot",
    },
    "spasm_cramp": {
        "spasm", "cramp", "muscle", "soreness",
    },
    "smoking_cancer": {
        "tobacco", "smoking", "smoke", "cancer", "lung",
    },
    "covid": {
        "covid", "coronavirus", "pandemic", "vaccine", "virus",
        "booster", "mask",
    },
    "heart_blood": {
        "heart", "blood", "pressure", "hypertension", "stroke",
        "cholesterol", "cardiac",
    },
    "nutrition": {
        "diet", "nutrition", "food", "meal", "protein", "vitamin",
        "supplement", "calories", "calorie", "carbs", "carbohydrates",
    },
    "weight_loss": {
        "weight", "loss", "fat", "belly", "calorie", "calories",
        "metabolism", "appetite", "hunger", "slimming", "obesity",
        "overweight",
    },
    "meal_skipping": {
        "meal", "meals", "skip", "skipping", "fasting", "intermittent",
        "breakfast", "lunch", "dinner",
    },
    "fat_burning_claim": {
        "fat", "burn", "burns", "burning", "metabolism",
    },
    "lemon_detox": {
        "lemon", "citrus", "lime", "detox", "cleanse", "fat",
    },
    "mental_health": {
        "mental", "depression", "anxiety", "stress", "sleep", "insomnia",
        "therapy",
    },
    "medicine_treatment": {
        "medicine", "medication", "drug", "treatment", "therapy",
        "antibiotic", "prescription", "dose",
    },
    "reproductive_health": {
        "pregnancy", "pregnant", "fertility", "period", "menstruation",
        "contraceptive",
    },
    "digestive_health": {
        "stomach", "gut", "digestive", "digestion", "diarrhea",
        "constipation", "bowel", "probiotic", "fiber",
    },
}


# More health concept groups for context verification.
HEALTH_CONCEPTS.update({
    "blood_pressure": {"blood", "pressure", "hypertension", "bp", "stroke", "heart", "cardiac"},
    "diabetes_glucose": {"diabetes", "glucose", "sugar", "insulin", "metformin", "carbohydrate", "carbs"},
    "immune_vaccine": {"immune", "immunity", "vaccine", "vaccination", "booster", "antibody", "infection"},
    "respiratory_health": {"lung", "lungs", "respiratory", "asthma", "cough", "bronchitis", "pneumonia", "breathing"},
    "skin_health": {"skin", "rash", "eczema", "psoriasis", "acne", "dermatology"},
    "pain_inflammation": {"pain", "inflammation", "anti", "inflammatory", "arthritis", "soreness", "swelling"},
    "sleep_health": {"sleep", "sleeping", "insomnia", "rest", "fatigue", "tired"},
    "substance_use": {"alcohol", "smoking", "tobacco", "nicotine", "vape", "addiction"},
    "herbal_home_remedy": {"herbal", "herb", "tea", "ginger", "garlic", "turmeric", "honey", "lemon", "vinegar"},
    "detox_cleanse": {"detox", "cleanse", "toxins", "flush", "purify"},
    "women_reproductive": {"pregnancy", "pregnant", "fertility", "menstruation", "period", "contraception"},
    "sexual_health": {"hiv", "aids", "std", "sti", "hpv", "herpes", "chlamydia", "gonorrhea", "syphilis"},
    "digestive_symptom": {"stomach", "gut", "diarrhea", "constipation", "acid", "reflux", "gerd", "nausea", "vomiting"},
    "medical_treatment": {"medicine", "medication", "drug", "antibiotic", "antiviral", "treatment", "dose", "prescription"},
})

def extract_concepts(text: str) -> set:
    cleaned_words = words_set(text)
    cleaned_text = clean_text(text)
    concepts = set()

    for concept_name, concept_terms in HEALTH_CONCEPTS.items():
        if cleaned_words.intersection(concept_terms):
            concepts.add(concept_name)
        else:
            for term in concept_terms:
                if len(term.split()) > 1 and term in cleaned_text:
                    concepts.add(concept_name)

    return concepts


def health_domain_score(user_claim: str) -> Tuple[float, Dict[str, object]]:
    cleaned = clean_text(user_claim)
    ws = set(cleaned.split())

    health_hits = ws.intersection(HEALTH_KEYWORDS)
    non_health_hits = ws.intersection(NON_HEALTH_KEYWORDS)
    concepts = extract_concepts(user_claim)

    health_phrase_hits = []
    for term in HEALTH_KEYWORDS:
        if " " in term and term in cleaned:
            health_phrase_hits.append(term)

    score = 0.0
    score += min(0.50, len(health_hits) * 0.12)
    score += min(0.35, len(concepts) * 0.14)
    score += min(0.15, len(health_phrase_hits) * 0.08)

    if non_health_hits and not health_hits and not concepts:
        score -= 0.45

    if non_health_hits and (health_hits or concepts):
        score -= min(0.15, len(non_health_hits) * 0.03)

    score = max(0.0, min(1.0, score))

    report = {
        "health_hits": sorted(list(health_hits)),
        "non_health_hits": sorted(list(non_health_hits)),
        "concepts": sorted(list(concepts)),
        "health_phrase_hits": health_phrase_hits,
        "health_domain_score": score,
    }

    return score, report


def is_health_related_lexical(user_claim: str) -> Tuple[bool, str, Dict[str, object]]:
    score, report = health_domain_score(user_claim)

    if score >= 0.18:
        details = []

        if report["concepts"]:
            details.append(f"health concepts: {', '.join(report['concepts'][:8])}")

        if report["health_hits"]:
            details.append(f"health terms: {', '.join(report['health_hits'][:8])}")

        if report["health_phrase_hits"]:
            details.append(f"health phrases: {', '.join(report['health_phrase_hits'][:4])}")

        reason = "Accepted because the input appears health-related"
        if details:
            reason += " through " + "; ".join(details)
        reason += "."

        return True, reason, report

    if report["non_health_hits"]:
        return (
            False,
            "Rejected because the input appears unrelated to health. "
            f"Detected non-health terms: {', '.join(report['non_health_hits'][:8])}.",
            report,
        )

    return (
        False,
        "Rejected because the input did not contain enough health-related concepts. "
        "Please enter a claim about disease, treatment, symptoms, nutrition, medicine, public health, or health risk.",
        report,
    )


def detect_population_context(text: str) -> str:
    ws = words_set(text)
    animal_hits = ws.intersection(ANIMAL_TERMS)
    human_hits = ws.intersection(HUMAN_TERMS)

    if animal_hits and not human_hits:
        return "animal"
    if human_hits and not animal_hits:
        return "human"
    if animal_hits and human_hits:
        return "mixed"

    return "human"


def population_compatibility(user_context: str, case_context: str) -> float:
    if user_context == case_context:
        return 1.0
    if user_context == "mixed" or case_context == "mixed":
        return 0.60
    if user_context == "human" and case_context == "animal":
        return 0.10
    if user_context == "animal" and case_context == "human":
        return 0.25
    return 0.50


def detect_claim_type(text: str) -> str:
    cleaned = clean_text(text)

    has_number = bool(re.search(r"\b\d+(\.\d+)?\b", cleaned)) or "%" in cleaned
    if has_number and any(term in cleaned for term in ["rate", "percent", "mortality", "death", "risk", "cases"]):
        return "statistical"

    if any(term in cleaned for term in CURE_TERMS):
        return "treatment_or_cure"

    if any(term in cleaned for term in ["prevent", "protect", "vaccine", "immunity"]):
        return "prevention"

    if any(term in cleaned for term in ["cause", "link", "linked", "risk", "associated", "contributes"]):
        return "causal_or_risk"

    if any(term in cleaned for term in ["deadly", "fatal", "death", "deaths", "mortality", "kills"]):
        return "severity"

    if any(term in cleaned for term in ["symptom", "symptoms", "spasm", "cramp", "pain", "soreness"]):
        return "symptom_or_effect"

    if any(term in cleaned for term in ["hydration", "water", "fluid", "fluids"]):
        return "health_benefit"

    if any(term in cleaned for term in ["weight", "fat", "metabolism", "calorie", "calories", "fasting"]):
        return "nutrition_weight"

    return "general"


def type_compatibility(user_type: str, case_type: str) -> float:
    if user_type == case_type:
        return 1.0

    compatible_pairs = {
        ("causal_or_risk", "severity"),
        ("severity", "causal_or_risk"),
        ("prevention", "causal_or_risk"),
        ("causal_or_risk", "prevention"),
        ("causal_or_risk", "symptom_or_effect"),
        ("symptom_or_effect", "causal_or_risk"),
        ("health_benefit", "prevention"),
        ("prevention", "health_benefit"),
        ("nutrition_weight", "health_benefit"),
        ("health_benefit", "nutrition_weight"),
        ("nutrition_weight", "causal_or_risk"),
        ("causal_or_risk", "nutrition_weight"),
    }

    if (user_type, case_type) in compatible_pairs:
        return 0.70

    if user_type == "general" or case_type == "general":
        return 0.50

    if user_type == "statistical" and case_type != "statistical":
        return 0.35

    return 0.45


def extract_claim_keywords(cleaned_claim: str) -> List[str]:
    stopwords = {
        "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
        "to", "of", "in", "on", "for", "with", "by", "from", "that",
        "this", "it", "as", "at", "be", "can", "may", "will", "does",
        "do", "most", "some", "many", "much", "claim", "says", "than",
        "better", "after", "before", "during", "into", "about", "helps",
        "help", "good", "bad",
    }

    words = [w for w in cleaned_claim.split() if len(w) > 2 and w not in stopwords]
    return list(dict.fromkeys(words))


def entity_overlap_score(user_claim: str, candidate_text: str) -> float:
    claim_keywords = extract_claim_keywords(clean_text(user_claim))
    if not claim_keywords:
        return 0.0

    candidate_clean = clean_text(candidate_text)
    hits = [kw for kw in claim_keywords if kw in candidate_clean]

    return len(hits) / max(1, len(claim_keywords))


# =========================================================
# KNOWN EXPERT RULES / MYTH SAFEGUARDS
# =========================================================
KNOWN_CONTEXT_RULES = [
    {
        "name": "showering_after_exercise_spasm_myth",
        "groups": [
            {"shower", "bath"},
            {"exercise", "activity", "workout"},
            {"spasm", "cramp"},
        ],
        "decision": "Misleading",
        "confidence": 32,
        "reason": (
            "Known-myth safeguard activated: the claim suggests that showering after physical activity causes spasms or cramps. "
            "This is treated as unsupported or myth-like unless strong medical evidence directly supports it."
        ),
    },
    {
        "name": "salty_foods_cause_uti_misleading",
        "groups": [
            {"urinary", "uti", "infection", "bladder"},
            {"salt", "sodium"},
            {"cause"},
        ],
        "decision": "Misleading",
        "confidence": 35,
        "reason": (
            "Known-rule safeguard activated: UTIs are primarily associated with bacterial infection, not simply salty food. "
            "Salty foods may affect hydration or urinary irritation, but saying salty foods directly cause UTIs is unsupported."
        ),
    },
    {
        "name": "water_improves_hydration_credible",
        "groups": [
            {"water", "fluid", "fluids"},
            {"hydration", "dehydration", "hydrate"},
        ],
        "decision": "Credible",
        "confidence": 82,
        "reason": (
            "Known-health fact activated: water and fluids support hydration. "
            "This part of the claim is generally credible."
        ),
    },
    {
        "name": "lemon_burns_fat_myth",
        "groups": [
            {"lemon", "citrus", "lime"},
            {"fat", "burn", "burns", "burning", "metabolism"},
        ],
        "decision": "Misleading",
        "confidence": 30,
        "reason": (
            "Known-myth safeguard activated: the claim suggests that lemon burns body fat. "
            "This is treated as misleading because lemon does not directly burn body fat."
        ),
    },
    {
        "name": "skipping_meals_weight_loss_oversimplified",
        "groups": [
            {"skip", "skipping", "meal", "meals", "fasting"},
            {"weight", "loss", "fat"},
        ],
        "decision": "Possibly Misleading",
        "confidence": 48,
        "reason": (
            "Known-rule safeguard activated: the claim suggests that skipping meals helps weight loss. "
            "This is treated cautiously because skipping meals may reduce calories in some cases, but it is an oversimplified and potentially unhealthy weight-loss claim."
        ),
    },
    {
        "name": "smoking_causes_cancer_credible",
        "groups": [
            {"tobacco", "smoking"},
            {"cancer", "lung"},
        ],
        "decision": "Credible",
        "confidence": 86,
        "reason": (
            "Known-health fact activated: smoking is a recognized risk factor for cancer, especially lung cancer."
        ),
    },
    {
        "name": "antibiotics_cure_virus_misleading",
        "groups": [
            {"antibiotic", "antibiotics"},
            {"virus", "viral"},
            {"cure", "treat"},
        ],
        "decision": "Misleading",
        "confidence": 28,
        "reason": (
            "Known-rule safeguard activated: antibiotics treat bacterial infections and do not cure viral infections."
        ),
    },
    {
        "name": "miracle_cure_warning",
        "groups": [
            {"miracle", "instant", "guaranteed"},
            {"cure", "treatment", "healing"},
        ],
        "decision": "Misleading",
        "confidence": 25,
        "reason": (
            "Known-misinformation pattern activated: miracle, instant, or guaranteed cure wording is a strong warning sign."
        ),
    },
]


def detect_known_context_rules(user_claim: str) -> List[Dict[str, object]]:
    ws = words_set(user_claim)
    matched_rules = []

    for rule in KNOWN_CONTEXT_RULES:
        matched_all_groups = True

        for group in rule["groups"]:
            if not ws.intersection(group):
                matched_all_groups = False
                break

        if matched_all_groups:
            matched_rules.append(rule)

    return matched_rules


def resolve_known_rule_effect(user_claim: str) -> Tuple[Optional[str], Optional[int], List[str]]:
    matched_rules = detect_known_context_rules(user_claim)

    if not matched_rules:
        return None, None, []

    decisions = [r["decision"] for r in matched_rules]
    reasons = [r["reason"] for r in matched_rules]

    has_credible = "Credible" in decisions
    has_misleading = "Misleading" in decisions
    has_possibly = "Possibly Misleading" in decisions

    if has_credible and has_misleading:
        return "Possibly Misleading", 55, reasons + [
            "Mixed-claim safeguard activated: the input contains both a credible component and a misleading or unsupported component."
        ]

    if has_misleading:
        confidence = min(int(r["confidence"]) for r in matched_rules if r["decision"] == "Misleading")
        return "Misleading", confidence, reasons

    if has_possibly:
        confidence = min(int(r["confidence"]) for r in matched_rules if r["decision"] == "Possibly Misleading")
        return "Possibly Misleading", confidence, reasons

    if has_credible:
        confidence = max(int(r["confidence"]) for r in matched_rules if r["decision"] == "Credible")
        return "Credible", confidence, reasons

    return None, None, reasons


# =========================================================
# DATASET LOADING
# =========================================================
def find_best_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    normalized_map = {c.lower().strip(): c for c in df.columns}

    for candidate in candidates:
        if candidate in normalized_map:
            return normalized_map[candidate]

    for column in df.columns:
        col_lower = column.lower().strip()
        for candidate in candidates:
            if candidate in col_lower:
                return column

    return None


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    df = pd.read_csv(path, sep=sep)

    if df.empty:
        raise ValueError("Dataset is empty.")

    claim_id_col = find_best_column(df, ["claim_id", "id"])
    claim_col = find_best_column(df, ["claim", "statement", "title", "text"])
    date_col = find_best_column(df, ["date_published", "published", "date", "publication_date"])
    explanation_col = find_best_column(df, ["explanation", "justification", "reason", "rationale"])
    fact_checker_col = find_best_column(df, ["fact_checkers", "fact checker", "factchecker", "reviewer", "author"])
    main_text_col = find_best_column(df, ["main_text", "main text", "article", "body", "content", "full_text"])
    source_col = find_best_column(df, ["sources", "source", "publisher", "origin", "url", "link"])
    label_col = find_best_column(df, ["label", "verdict", "rating", "truth_rating"])
    subjects_col = find_best_column(df, ["subjects", "subject", "topic", "tags", "category"])

    if claim_col is None or label_col is None:
        raise ValueError("Could not detect required columns. Your file must contain at least claim and label columns.")

    working = pd.DataFrame({
        "claim_id": df[claim_id_col].map(safe_text) if claim_id_col else "",
        "claim": df[claim_col].map(safe_text),
        "date_published": df[date_col].map(safe_text) if date_col else "",
        "explanation": df[explanation_col].map(safe_text) if explanation_col else "",
        "fact_checkers": df[fact_checker_col].map(safe_text) if fact_checker_col else "",
        "main_text": df[main_text_col].map(safe_text) if main_text_col else "",
        "sources": df[source_col].map(safe_text) if source_col else "",
        "label": df[label_col].map(safe_text),
        "subjects": df[subjects_col].map(safe_text) if subjects_col else "",
    })

    working = working.dropna(subset=["claim"]).copy()
    working["claim"] = working["claim"].map(normalize_whitespace)
    working = working[working["claim"] != ""].reset_index(drop=True)

    working["label_class"] = working["label"].apply(classify_label)

    working["clean_claim"] = working["claim"].map(clean_text)
    working["clean_explanation"] = working["explanation"].map(clean_text)
    working["clean_main_text"] = working["main_text"].map(clean_text)
    working["clean_sources"] = working["sources"].map(clean_text)
    working["clean_subjects"] = working["subjects"].map(clean_text)
    working["clean_fact_checkers"] = working["fact_checkers"].map(clean_text)
    working["clean_date"] = working["date_published"].map(clean_text)
    working["clean_claim_id"] = working["claim_id"].map(clean_text)

    working["combined_text"] = (
        working["clean_claim_id"] + " " +
        working["clean_claim"] + " " +
        working["clean_date"] + " " +
        working["clean_explanation"] + " " +
        working["clean_fact_checkers"] + " " +
        working["clean_main_text"] + " " +
        working["clean_sources"] + " " +
        working["label_class"].map(clean_text) + " " +
        working["clean_subjects"]
    ).map(normalize_whitespace)

    return working


@st.cache_resource(show_spinner=False)
def prepare_case_base(dataset_path: str):
    df = load_dataset(Path(dataset_path))

    claim_vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 3),
        min_df=1,
        sublinear_tf=True,
    )
    claim_matrix = claim_vectorizer.fit_transform(df["clean_claim"])

    content_vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 3),
        min_df=1,
        sublinear_tf=True,
        max_features=100000,
    )
    content_matrix = content_vectorizer.fit_transform(df["combined_text"])

    classifier = None
    classifier_classes = []

    try:
        labels = df["label_class"]
        if labels.nunique() >= 2 and len(df) >= 10:
            classifier = LogisticRegression(
                max_iter=1500,
                class_weight="balanced",
                solver="liblinear",
            )
            classifier.fit(content_matrix, labels)
            classifier_classes = list(classifier.classes_)
    except Exception:
        classifier = None
        classifier_classes = []

    return df, claim_vectorizer, claim_matrix, content_vectorizer, content_matrix, classifier, classifier_classes


# =========================================================
# SIMILARITY + CONTEXT VERIFICATION
# =========================================================
def build_case_blob(row: pd.Series) -> str:
    return " ".join([
        safe_text(row.get("claim_id", "")),
        safe_text(row.get("claim", "")),
        safe_text(row.get("date_published", "")),
        safe_text(row.get("explanation", "")),
        safe_text(row.get("fact_checkers", "")),
        safe_text(row.get("main_text", "")),
        safe_text(row.get("sources", "")),
        safe_text(row.get("label", "")),
        safe_text(row.get("subjects", "")),
    ])


def compute_advanced_similarity(
    user_claim: str,
    df: pd.DataFrame,
    claim_similarities: np.ndarray,
    content_similarities: np.ndarray,
    candidate_indices: np.ndarray,
) -> np.ndarray:
    user_type = detect_claim_type(user_claim)
    user_context = detect_population_context(user_claim)
    user_clean = clean_text(user_claim)
    user_concepts = extract_concepts(user_claim)

    advanced_scores = []

    for idx in candidate_indices:
        row = df.iloc[idx]

        case_claim = safe_text(row["claim"])
        case_blob = build_case_blob(row)

        case_type = detect_claim_type(case_claim)
        case_context = detect_population_context(case_blob)

        type_score = type_compatibility(user_type, case_type)
        population_score = population_compatibility(user_context, case_context)
        overlap_score = entity_overlap_score(user_claim, case_blob)

        case_concepts = extract_concepts(case_blob)

        if user_concepts:
            concept_overlap = len(user_concepts.intersection(case_concepts)) / max(1, len(user_concepts))
        else:
            concept_overlap = 0.0

        exact_bonus = 0.0
        if clean_text(case_claim) == user_clean:
            exact_bonus = 0.18

        score = (
            0.28 * float(claim_similarities[idx])
            + 0.20 * float(content_similarities[idx])
            + 0.16 * overlap_score
            + 0.18 * concept_overlap
            + 0.10 * type_score
            + 0.08 * population_score
            + exact_bonus
        )

        if population_score < 0.30:
            score *= 0.50

        if concept_overlap == 0 and overlap_score < 0.25:
            score *= 0.50

        advanced_scores.append(max(0.0, min(1.0, score)))

    return np.array(advanced_scores)


def contextual_relevance_score(user_claim: str, case_row: pd.Series) -> Tuple[float, str]:
    case_blob = build_case_blob(case_row)

    user_clean = clean_text(user_claim)
    case_clean = clean_text(case_blob)

    user_keywords = extract_claim_keywords(user_clean)

    if not user_keywords:
        return 0.0, "No important claim terms were extracted from the user input."

    keyword_hits = [kw for kw in user_keywords if kw in case_clean]
    keyword_coverage = len(keyword_hits) / max(1, len(user_keywords))

    user_concepts = extract_concepts(user_claim)
    case_concepts = extract_concepts(case_blob)

    if user_concepts:
        concept_overlap = len(user_concepts.intersection(case_concepts)) / max(1, len(user_concepts))
    else:
        concept_overlap = 0.0

    user_type = detect_claim_type(user_claim)
    case_type = detect_claim_type(safe_text(case_row.get("claim", "")))
    type_score = type_compatibility(user_type, case_type)

    user_population = detect_population_context(user_claim)
    case_population = detect_population_context(case_blob)
    population_score = population_compatibility(user_population, case_population)

    similarity_score = float(case_row.get("advanced_similarity", 0.0))

    subject_clean = clean_text(safe_text(case_row.get("subjects", "")))
    subject_overlap = 0.0

    if subject_clean:
        subject_words = set(subject_clean.split())
        claim_words = set(user_clean.split())
        overlap = subject_words.intersection(claim_words)
        subject_overlap = min(1.0, len(overlap) / max(1, len(claim_words)))

    relevance = (
        0.22 * similarity_score
        + 0.22 * keyword_coverage
        + 0.30 * concept_overlap
        + 0.10 * type_score
        + 0.10 * population_score
        + 0.06 * subject_overlap
    )

    if population_score < 0.30:
        relevance *= 0.45

    if concept_overlap == 0 and keyword_coverage < 0.35:
        relevance *= 0.45

    explanation = (
        f"Context relevance checked keyword coverage ({keyword_coverage:.2f}), "
        f"concept overlap ({concept_overlap:.2f}), claim-type compatibility ({type_score:.2f}), "
        f"population compatibility ({population_score:.2f}), subject overlap ({subject_overlap:.2f}), "
        f"and similarity ({similarity_score:.2f}). "
        f"Matched important terms: {', '.join(keyword_hits[:8]) if keyword_hits else 'none'}. "
        f"Matched concepts: {', '.join(sorted(user_concepts.intersection(case_concepts))) if user_concepts.intersection(case_concepts) else 'none'}."
    )

    return max(0.0, min(1.0, relevance)), explanation


def should_generate_decision(user_claim: str, top_cases: pd.DataFrame) -> Tuple[bool, str]:
    known_decision, known_confidence, known_reasons = resolve_known_rule_effect(user_claim)

    if known_decision is not None:
        return True, "Decision allowed because a known expert-rule pattern matched the user claim."

    if top_cases.empty:
        return False, "This input may not match any stored cases in the system closely enough. Please enter another clear health-related claim."

    best_context_relevance = float(top_cases.iloc[0].get("context_relevance", 0.0))
    best_similarity = float(top_cases.iloc[0].get("advanced_similarity", 0.0))
    relevant_case_count = int((top_cases["context_relevance"] >= 0.45).sum())

    if best_similarity < 0.18:
        return False, (
            "This input may not match any stored cases in the system closely enough. "
            "Please enter another clear health-related claim."
        )

    if best_context_relevance < 0.35 and relevant_case_count < 2:
        return False, (
            "This input may not match any stored cases in the system closely enough. "
            "Please enter another clear health-related claim."
        )

    return True, "Decision allowed because retrieved cases passed the minimum context-relevance check."


# =========================================================
# PARAMETER SCORING
# =========================================================
def score_claim_id_quality(claim_id: str) -> Tuple[float, str, str]:
    text = safe_text(claim_id).strip()

    if text and text.lower() not in {"nan", "none", "unknown"}:
        return 1.0, "Present", "Claim ID was found, which improves traceability of the retrieved case."

    return 0.35, "Missing", "Claim ID was missing or unclear, so traceability is weaker."


def score_source_quality(source: str) -> Tuple[float, str, str]:
    source_text = safe_text(source)
    source_clean = clean_text(source_text)
    domains = extract_domains(source_text)
    domain_text = " ".join(domains)

    combined = source_clean + " " + domain_text

    strong_hits = find_hits(combined, HIGH_CREDIBILITY_SOURCES)
    factcheck_hits = find_hits(combined, FACT_CHECKER_SOURCES)
    weak_hits = find_hits(combined, LOW_CREDIBILITY_SOURCE_HINTS)

    if strong_hits:
        quality = "Strong"
        reason = f"The source was rated Strong because it contains recognized health authority cues such as {', '.join(strong_hits[:3])}."
        return 1.0, quality, reason

    if factcheck_hits:
        quality = "Fact-check source"
        reason = f"The source was rated Fact-check source because it includes fact-checking or reputable media cues such as {', '.join(factcheck_hits[:3])}."
        return 0.85, quality, reason

    if weak_hits or source_clean in {"", "nan", "unknown source", "none"}:
        quality = "Weak"
        reason = "The source was rated Weak because the source is missing, unclear, or contains weak-source indicators."
        return 0.15, quality, reason

    if domains:
        quality = "Moderate"
        reason = f"The source was rated Moderate because source links were found, but no major health authority was detected. Domains: {', '.join(domains[:3])}."
        return 0.55, quality, reason

    quality = "Limited"
    reason = "The source was rated Limited because only minimal source information was available."
    return 0.35, quality, reason


def score_fact_checkers(fact_checkers: str) -> Tuple[float, str, str]:
    text = clean_text(fact_checkers)

    if text in {"", "nan", "none", "unknown"}:
        quality = "Missing"
        reason = "The fact-checker status was rated Missing because no reviewer, author, or fact-checker was detected."
        return 0.20, quality, reason

    if len(text.split()) >= 2:
        quality = "Present"
        reason = f"The fact-checker status was rated Present because reviewer/fact-checker information was found: {fact_checkers}."
        return 1.0, quality, reason

    quality = "Limited"
    reason = "The fact-checker status was rated Limited because some reviewer information was detected but it was incomplete."
    return 0.60, quality, reason


def score_date_quality(date_published: str, user_claim: str) -> Tuple[float, str, str]:
    cleaned_claim = clean_text(user_claim)
    parsed_date = parse_date_safely(date_published)

    if parsed_date is None:
        quality = "Unknown"
        reason = "The date quality was rated Unknown because the publication date was missing or unreadable."
        return 0.45, quality, reason

    current_year = datetime.now().year
    age = current_year - parsed_date.year

    if not is_time_sensitive_claim(cleaned_claim):
        quality = "Acceptable"
        reason = f"The date quality was rated Acceptable because the case has a publication date ({parsed_date.date()}) and the claim is not highly time-sensitive."
        return 0.70, quality, reason

    if age <= 3:
        quality = "Recent"
        reason = f"The date quality was rated Recent because the case was published on {parsed_date.date()}."
        return 1.0, quality, reason

    if age <= 7:
        quality = "Moderate age"
        reason = f"The date quality was rated Moderate age because the case was published on {parsed_date.date()}."
        return 0.60, quality, reason

    quality = "Old"
    reason = f"The date quality was rated Old because the case was published on {parsed_date.date()}, which may be outdated for time-sensitive health topics."
    return 0.30, quality, reason


def score_subject_alignment(subjects: str, user_claim: str) -> Tuple[float, str, str]:
    subj_clean = clean_text(subjects)
    claim_clean = clean_text(user_claim)

    if subj_clean in {"", "nan", "none"}:
        quality = "Missing"
        reason = "Subject alignment was rated Missing because no subject/category information was available."
        return 0.40, quality, reason

    subject_words = set(subj_clean.split())
    claim_words = set(claim_clean.split())
    overlap = subject_words.intersection(claim_words)

    user_concepts = extract_concepts(user_claim)
    subject_concepts = extract_concepts(subjects)
    concept_overlap = user_concepts.intersection(subject_concepts)

    if concept_overlap:
        quality = "Aligned"
        reason = f"Subject alignment was rated Aligned because the subject field matches claim concepts: {', '.join(sorted(concept_overlap))}."
        return 1.0, quality, reason

    if overlap:
        quality = "Partially aligned"
        reason = f"Subject alignment was rated Partially aligned because subject terms overlap with the claim through: {', '.join(sorted(overlap)[:5])}."
        return 0.75, quality, reason

    if any(term in subj_clean for term in HEALTH_KEYWORDS):
        quality = "Health-related"
        reason = "Subject alignment was rated Health-related because the subject field is within the health domain, but it does not strongly match the exact claim."
        return 0.60, quality, reason

    quality = "Weak"
    reason = "Subject alignment was rated Weak because the subject/category field did not clearly match the user claim."
    return 0.35, quality, reason


def score_explanation_strength(explanation: str) -> Tuple[float, str, str]:
    clean = clean_text(explanation)
    word_count = len(clean.split())

    if clean in {"", "nan", "none", "no explanation available"} or word_count == 0:
        quality = "Missing"
        reason = "The explanation quality was rated Missing because no useful explanation was available."
        return 0.10, quality, reason

    support_hits = non_negated_phrase_hits(clean, SUPPORTIVE_CONTENT_PHRASES)
    negated_support = negated_support_hits(clean, SUPPORTIVE_CONTENT_PHRASES)
    refute_hits = find_hits(clean, REFUTING_CONTENT_PHRASES)
    evidence_hits = find_hits(clean, EVIDENCE_TERMS)

    score = 0.35
    details = []

    if word_count >= 80:
        score += 0.30
        details.append("it is detailed")
    elif word_count >= 30:
        score += 0.15
        details.append("it has moderate detail")
    else:
        score -= 0.10
        details.append("it is short")

    if evidence_hits:
        score += 0.20
        details.append(f"it includes evidence-related terms such as {', '.join(evidence_hits[:3])}")

    if support_hits:
        score += 0.10
        details.append(f"it includes supportive wording such as {', '.join(support_hits[:3])}")

    if negated_support:
        score -= 0.20
        details.append(f"it includes negated support phrases such as {', '.join(negated_support[:3])}")

    if refute_hits:
        score -= 0.25
        details.append(f"it includes refuting/cautionary wording such as {', '.join(refute_hits[:3])}")

    score = max(0.05, min(1.0, score))

    if score >= 0.80:
        quality = "Strong"
    elif score >= 0.60:
        quality = "Moderate"
    elif score >= 0.35:
        quality = "Limited"
    else:
        quality = "Weak"

    reason = f"The explanation quality was rated {quality} because " + "; ".join(details) + "."
    return score, quality, reason


def evidence_sentences_for_claim(user_claim: str, text: str, max_sentences: int = 3) -> List[str]:
    claim_keywords = extract_claim_keywords(clean_text(user_claim))
    if not claim_keywords:
        return []

    sentences = sentence_split(text)
    scored = []

    for sentence in sentences:
        clean_sentence = clean_text(sentence)
        hits = [kw for kw in claim_keywords if kw in clean_sentence]
        support_hits = non_negated_phrase_hits(clean_sentence, SUPPORTIVE_CONTENT_PHRASES)
        refute_hits = find_hits(clean_sentence, REFUTING_CONTENT_PHRASES)

        score = len(hits) + 1.5 * len(support_hits) + 1.5 * len(refute_hits)
        if score > 0:
            scored.append((score, sentence))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored[:max_sentences]]


def score_main_text_support(user_claim: str, main_text: str, explanation: str) -> Tuple[float, str, str, List[str]]:
    claim_clean = clean_text(user_claim)
    combined_raw = safe_text(main_text) + " " + safe_text(explanation)
    combined = clean_text(combined_raw)

    if not combined:
        quality = "Missing"
        reason = "Main text support was rated Missing because no main text or explanation content was available."
        return 0.10, quality, reason, []

    keywords = extract_claim_keywords(claim_clean)
    keyword_hits = [kw for kw in keywords if kw in combined]
    coverage = len(keyword_hits) / max(1, len(keywords))

    user_concepts = extract_concepts(user_claim)
    text_concepts = extract_concepts(combined_raw)

    if user_concepts:
        concept_overlap = len(user_concepts.intersection(text_concepts)) / max(1, len(user_concepts))
    else:
        concept_overlap = 0.0

    support_hits = non_negated_phrase_hits(combined, SUPPORTIVE_CONTENT_PHRASES)
    negated_support = negated_support_hits(combined, SUPPORTIVE_CONTENT_PHRASES)
    refute_hits = find_hits(combined, REFUTING_CONTENT_PHRASES)
    evidence_sentences = evidence_sentences_for_claim(user_claim, combined_raw)

    score = 0.30
    details = []

    if coverage >= 0.75:
        score += 0.22
        details.append(f"it covers most important claim terms: {', '.join(keyword_hits[:5])}")
    elif coverage >= 0.45:
        score += 0.10
        details.append(f"it covers some claim terms: {', '.join(keyword_hits[:5])}")
    else:
        score -= 0.15
        details.append("it does not strongly cover the important terms from the user claim")

    if concept_overlap >= 0.70:
        score += 0.20
        details.append(f"it matches the claim concepts: {', '.join(sorted(user_concepts.intersection(text_concepts)))}")
    elif concept_overlap > 0:
        score += 0.08
        details.append(f"it partially matches the claim concepts: {', '.join(sorted(user_concepts.intersection(text_concepts)))}")
    else:
        score -= 0.12
        details.append("it does not strongly match the claim concepts")

    if support_hits:
        score += 0.12
        details.append(f"it contains supportive wording such as {', '.join(support_hits[:3])}")

    if negated_support:
        score -= 0.22
        details.append(f"it contains negated support phrases such as {', '.join(negated_support[:3])}")

    if refute_hits:
        score -= 0.30
        details.append(f"it contains refuting/cautionary wording such as {', '.join(refute_hits[:3])}")

    user_context = detect_population_context(user_claim)
    case_context = detect_population_context(combined_raw)

    if population_compatibility(user_context, case_context) < 0.30:
        score -= 0.35
        details.append(f"there is a population/context mismatch: user claim appears {user_context}, retrieved content appears {case_context}")

    score = max(0.05, min(1.0, score))

    if score >= 0.80:
        quality = "Strongly supports"
    elif score >= 0.62:
        quality = "Supports"
    elif score >= 0.45:
        quality = "Neutral"
    elif score >= 0.28:
        quality = "Limited support"
    else:
        quality = "Weak or conflicting"

    reason = f"Main text support was rated {quality} because " + "; ".join(details) + "."
    return score, quality, reason, evidence_sentences


def linguistic_warning_adjustment(user_claim: str) -> Tuple[float, List[str], Dict[str, List[str]]]:
    clean = clean_text(user_claim)

    exaggerated_hits = find_hits(clean, EXAGGERATED_TERMS)
    absolute_hits = find_hits(clean, ABSOLUTE_TERMS)
    cure_hits = find_hits(clean, CURE_TERMS)
    evidence_hits = find_hits(clean, EVIDENCE_TERMS)

    adjustment = 0.0
    reasons = []

    if len(exaggerated_hits) >= 2:
        adjustment -= 0.18
        reasons.append(f"Multiple exaggerated words were detected: {', '.join(exaggerated_hits[:5])}.")
    elif len(exaggerated_hits) == 1:
        adjustment -= 0.08
        reasons.append(f"One exaggerated word was detected: {exaggerated_hits[0]}.")

    if cure_hits and not evidence_hits:
        adjustment -= 0.10
        reasons.append("The claim talks about cure/treatment but does not include evidence-related wording.")

    if absolute_hits:
        adjustment -= 0.05
        reasons.append(f"Absolute wording was detected: {', '.join(absolute_hits[:5])}.")

    if evidence_hits:
        adjustment += 0.03
        reasons.append(f"Evidence-related wording was detected: {', '.join(evidence_hits[:4])}.")

    features = {
        "exaggerated_hits": exaggerated_hits,
        "absolute_hits": absolute_hits,
        "cure_hits": cure_hits,
        "evidence_hits": evidence_hits,
    }

    return adjustment, reasons, features


def classifier_probability(
    cleaned_claim: str,
    content_vectorizer,
    classifier,
    classifier_classes: List[str],
) -> Dict[str, float]:
    probabilities = {
        "Credible": 0.33,
        "Possibly Misleading": 0.34,
        "Misleading": 0.33,
    }

    if classifier is None:
        return probabilities

    try:
        vector = content_vectorizer.transform([cleaned_claim])
        probs = classifier.predict_proba(vector)[0]

        probabilities = {
            "Credible": 0.0,
            "Possibly Misleading": 0.0,
            "Misleading": 0.0,
        }

        for cls, prob in zip(classifier_classes, probs):
            normalized = classify_label(cls)
            probabilities[normalized] = max(probabilities.get(normalized, 0.0), float(prob))

        total = sum(probabilities.values())
        if total > 0:
            probabilities = {k: v / total for k, v in probabilities.items()}

        return probabilities
    except Exception:
        return probabilities


def weighted_label_credibility(top_cases: pd.DataFrame) -> float:
    label_values = {
        "Credible": 1.0,
        "Possibly Misleading": 0.50,
        "Misleading": 0.0,
    }

    total_weight = 0.0
    total_score = 0.0

    for _, row in top_cases.iterrows():
        label = classify_label(row["label"])
        similarity = float(row.get("advanced_similarity", 0.0))
        context_relevance = float(row.get("context_relevance", 1.0))
        population_score = float(row.get("population_compatibility", 1.0))

        weight = similarity * context_relevance * (0.70 + 0.30 * population_score)

        if context_relevance < 0.35:
            weight *= 0.20

        if similarity < 0.25:
            weight *= 0.20

        total_weight += weight
        total_score += weight * label_values[label]

    if total_weight <= 0:
        return 0.50

    return total_score / total_weight


def final_result_from_confidence(confidence: int) -> str:
    if confidence >= 70:
        return "Credible"
    if confidence >= 40:
        return "Possibly Misleading"
    return "Misleading"


def calculate_evidence_score(
    user_claim: str,
    top_cases: pd.DataFrame,
    parameter_report: Dict[str, object],
    classifier_probs: Dict[str, float],
) -> Tuple[int, Dict[str, object], List[str]]:
    safeguards = []

    best_similarity = float(top_cases.iloc[0]["advanced_similarity"])
    avg_similarity = float(top_cases["advanced_similarity"].mean())
    avg_context = float(top_cases["context_relevance"].mean())
    label_credibility = weighted_label_credibility(top_cases)

    source_score = float(parameter_report.get("source_score", 0.35))
    factchecker_score = float(parameter_report.get("factchecker_score", 0.20))
    explanation_score = float(parameter_report.get("explanation_score", 0.10))
    main_text_score = float(parameter_report.get("main_text_score", 0.10))
    date_score = float(parameter_report.get("date_score", 0.45))
    subject_score = float(parameter_report.get("subject_score", 0.40))
    claim_id_score = float(parameter_report.get("claim_id_score", 0.35))

    classifier_credible = float(classifier_probs.get("Credible", 0.33))

    warning_adjustment, warning_reasons, linguistic_features = linguistic_warning_adjustment(user_claim)
    known_decision, known_confidence, known_reasons = resolve_known_rule_effect(user_claim)

    base_score = (
        0.14 * best_similarity
        + 0.08 * avg_similarity
        + 0.20 * avg_context
        + 0.16 * label_credibility
        + 0.17 * main_text_score
        + 0.08 * source_score
        + 0.06 * explanation_score
        + 0.04 * factchecker_score
        + 0.03 * subject_score
        + 0.02 * date_score
        + 0.01 * claim_id_score
        + 0.01 * classifier_credible
    )

    adjusted_score = base_score + warning_adjustment

    user_context = detect_population_context(user_claim)
    top_case_text = " ".join([
        safe_text(top_cases.iloc[0].get("claim", "")),
        safe_text(top_cases.iloc[0].get("main_text", "")),
        safe_text(top_cases.iloc[0].get("explanation", "")),
    ])
    top_context = detect_population_context(top_case_text)

    if population_compatibility(user_context, top_context) < 0.30:
        adjusted_score -= 0.25
        safeguards.append(
            "Context-mismatch safeguard activated: the retrieved case appears to discuss animals/pets, while the user claim appears to discuss human health."
        )

    if best_similarity < 0.30:
        adjusted_score -= 0.12
        safeguards.append(
            "Low-match safeguard activated: the closest retrieved case was weak, so the system reduced confidence."
        )

    if avg_context < 0.40:
        adjusted_score -= 0.15
        safeguards.append(
            "Context-relevance safeguard activated: the retrieved cases were not strongly aligned with the user’s intended health context."
        )

    if (
        parameter_report.get("source_quality") == "Weak"
        and parameter_report.get("explanation_quality") in {"Weak", "Missing"}
        and parameter_report.get("main_text_support") in {"Neutral", "Weak or conflicting", "Missing", "Limited support"}
    ):
        adjusted_score -= 0.18
        safeguards.append(
            "Weak-evidence safeguard activated: weak source, weak explanation, and neutral/weak main-text support cannot justify a strong credible result."
        )

    if parameter_report.get("main_text_support") in {"Weak or conflicting", "Missing"}:
        adjusted_score -= 0.10
        safeguards.append(
            "Main-text safeguard activated: the retrieved content did not sufficiently support the exact user claim."
        )

    if linguistic_features["exaggerated_hits"] or (
        linguistic_features["cure_hits"] and not linguistic_features["evidence_hits"]
    ):
        safeguards.append(
            "Warning-language safeguard activated: exaggerated or cure-style wording made the system more cautious."
        )

    if known_decision is not None and known_confidence is not None:
        if known_decision == "Credible":
            adjusted_score = max(adjusted_score, known_confidence / 100)
        elif known_decision == "Misleading":
            adjusted_score = min(adjusted_score, known_confidence / 100)
        else:
            adjusted_score = min(max(adjusted_score, 0.40), known_confidence / 100)

        safeguards.extend(known_reasons)

    adjusted_score = max(0.0, min(1.0, adjusted_score))
    evidence_confidence = int(round(adjusted_score * 100))

    breakdown = {
        "best_similarity": best_similarity,
        "average_similarity": avg_similarity,
        "average_context_relevance": avg_context,
        "label_credibility": label_credibility,
        "main_text_score": main_text_score,
        "source_score": source_score,
        "explanation_score": explanation_score,
        "factchecker_score": factchecker_score,
        "subject_score": subject_score,
        "date_score": date_score,
        "claim_id_score": claim_id_score,
        "classifier_credible": classifier_credible,
        "warning_adjustment": warning_adjustment,
        "base_score": base_score,
        "final_raw_score": adjusted_score,
        "evidence_confidence": evidence_confidence,
        "known_decision": known_decision,
        "known_confidence": known_confidence,
        "known_reasons": known_reasons,
        "warning_reasons": warning_reasons,
        "linguistic_features": linguistic_features,
    }

    return evidence_confidence, breakdown, safeguards


# =========================================================
# MAIN ANALYSIS
# =========================================================
def analyze_claim(
    user_claim: str,
    df: pd.DataFrame,
    claim_vectorizer: TfidfVectorizer,
    claim_matrix,
    content_vectorizer: TfidfVectorizer,
    content_matrix,
    classifier,
    classifier_classes: List[str],
    top_k: int = 7,
) -> Dict[str, object]:
    user_claim = normalize_whitespace(user_claim)

    if not user_claim:
        return {
            "status": "error",
            "message": "Please enter a health claim before analyzing."
        }

    lexical_health, health_reason, health_report = is_health_related_lexical(user_claim)

    if not lexical_health:
        return {
            "status": "error",
            "message": (
                "This input was rejected because it may not be a health-related claim or a claim at all, "
                "or it may not match any stored cases in the system. Please enter another clear health-related claim. "
                "A valid claim usually states a relationship or effect, such as ‘X causes Y’ or ‘X helps prevent Y.’"
            ),
            "health_report": health_report,
        }

    cleaned_claim = clean_text(user_claim)

    claim_vector = claim_vectorizer.transform([cleaned_claim])
    content_vector = content_vectorizer.transform([cleaned_claim])

    claim_similarities = cosine_similarity(claim_vector, claim_matrix).flatten()
    content_similarities = cosine_similarity(content_vector, content_matrix).flatten()

    base_similarities = (0.52 * claim_similarities) + (0.48 * content_similarities)

    pool_size = min(400, len(df))
    candidate_indices = np.argsort(base_similarities)[::-1][:pool_size]

    advanced_scores = compute_advanced_similarity(
        user_claim=user_claim,
        df=df,
        claim_similarities=claim_similarities,
        content_similarities=content_similarities,
        candidate_indices=candidate_indices,
    )

    ranked_pool_order = np.argsort(advanced_scores)[::-1]
    selected_positions = ranked_pool_order[:max(1, min(top_k, len(ranked_pool_order)))]
    top_indices = candidate_indices[selected_positions]
    top_advanced_scores = advanced_scores[selected_positions]

    top_cases = df.iloc[top_indices].copy().reset_index(drop=True)

    user_type = detect_claim_type(user_claim)
    user_context = detect_population_context(user_claim)

    top_cases["claim_similarity"] = claim_similarities[top_indices]
    top_cases["content_similarity"] = content_similarities[top_indices]
    top_cases["advanced_similarity"] = top_advanced_scores
    top_cases["case_result"] = top_cases["label"].apply(classify_label)
    top_cases["claim_type"] = top_cases["claim"].apply(detect_claim_type)
    top_cases["user_claim_type"] = user_type
    top_cases["user_population_context"] = user_context

    top_cases["case_population_context"] = top_cases.apply(
        lambda row: detect_population_context(
            safe_text(row["claim"]) + " " +
            safe_text(row["main_text"]) + " " +
            safe_text(row["explanation"])
        ),
        axis=1,
    )

    top_cases["type_compatibility"] = top_cases["claim_type"].apply(
        lambda t: type_compatibility(user_type, t)
    )

    top_cases["population_compatibility"] = top_cases["case_population_context"].apply(
        lambda c: population_compatibility(user_context, c)
    )

    context_results = top_cases.apply(
        lambda row: contextual_relevance_score(user_claim, row),
        axis=1,
    )

    top_cases["context_relevance"] = [result[0] for result in context_results]
    top_cases["context_explanation"] = [result[1] for result in context_results]

    decision_allowed, decision_reason = should_generate_decision(user_claim, top_cases)

    if not decision_allowed:
        return {
            "status": "no_decision",
            "message": decision_reason,
            "top_cases": top_cases,
            "health_report": health_report,
        }

    top = top_cases.iloc[0]

    claim_id_score, claim_id_quality, claim_id_reason = score_claim_id_quality(top["claim_id"])
    source_score, source_quality, source_reason = score_source_quality(top["sources"])
    factchecker_score, factchecker_quality, factchecker_reason = score_fact_checkers(top["fact_checkers"])
    explanation_score, explanation_quality, explanation_reason = score_explanation_strength(top["explanation"])
    main_score, main_support, main_reason, evidence_sentences = score_main_text_support(
        user_claim,
        top["main_text"],
        top["explanation"],
    )
    date_score, date_quality, date_reason = score_date_quality(top["date_published"], user_claim)
    subject_score, subject_quality, subject_reason = score_subject_alignment(top["subjects"], user_claim)

    labels = top_cases["label"].astype(str).apply(classify_label)
    credible_ratio = float((labels == "Credible").mean())
    possibly_ratio = float((labels == "Possibly Misleading").mean())
    misleading_ratio = float((labels == "Misleading").mean())

    classifier_probs = classifier_probability(
        cleaned_claim=cleaned_claim,
        content_vectorizer=content_vectorizer,
        classifier=classifier,
        classifier_classes=classifier_classes,
    )

    parameter_report = {
        "claim_id_score": claim_id_score,
        "claim_id_quality": claim_id_quality,
        "claim_id_reason": claim_id_reason,

        "source_score": source_score,
        "source_quality": source_quality,
        "source_reason": source_reason,

        "factchecker_score": factchecker_score,
        "factchecker_quality": factchecker_quality,
        "factchecker_reason": factchecker_reason,

        "explanation_score": explanation_score,
        "explanation_quality": explanation_quality,
        "explanation_reason": explanation_reason,

        "main_text_score": main_score,
        "main_text_support": main_support,
        "main_text_reason": main_reason,
        "evidence_sentences": evidence_sentences,

        "date_score": date_score,
        "date_quality": date_quality,
        "date_reason": date_reason,

        "subject_score": subject_score,
        "subject_quality": subject_quality,
        "subject_reason": subject_reason,

        "credible_ratio": credible_ratio,
        "possibly_ratio": possibly_ratio,
        "misleading_ratio": misleading_ratio,

        "classifier_probabilities": classifier_probs,
        "health_report": health_report,
    }

    evidence_confidence, score_breakdown, safeguards = calculate_evidence_score(
        user_claim=user_claim,
        top_cases=top_cases,
        parameter_report=parameter_report,
        classifier_probs=classifier_probs,
    )

    final_result = final_result_from_confidence(evidence_confidence)

    expert_messages = []
    expert_messages.extend(score_breakdown.get("known_reasons", []))
    expert_messages.extend(score_breakdown.get("warning_reasons", []))
    expert_messages.append(health_reason)
    expert_messages.append(claim_id_reason)
    expert_messages.append(source_reason)
    expert_messages.append(factchecker_reason)
    expert_messages.append(explanation_reason)
    expert_messages.append(main_reason)
    expert_messages.append(date_reason)
    expert_messages.append(subject_reason)

    explanation_summary, explanation_detail = synthesize_explanations(
        final_result=final_result,
        confidence=evidence_confidence,
        top_cases=top_cases,
        expert_messages=expert_messages,
        safeguards=safeguards,
        parameter_report=parameter_report,
        score_breakdown=score_breakdown,
    )

    confidence_rows = [
        ["Strongest advanced match", f"{score_breakdown['best_similarity']:.4f}", "14% weight"],
        ["Average top-case match", f"{score_breakdown['average_similarity']:.4f}", "8% weight"],
        ["Average context relevance", f"{score_breakdown['average_context_relevance']:.4f}", "20% weight"],
        ["Retrieved-label credibility", f"{score_breakdown['label_credibility']:.4f}", "16% weight"],
        ["Main-text support", f"{score_breakdown['main_text_score']:.4f}", "17% weight"],
        ["Source quality", f"{score_breakdown['source_score']:.4f}", "8% weight"],
        ["Explanation quality", f"{score_breakdown['explanation_score']:.4f}", "6% weight"],
        ["Fact-checker quality", f"{score_breakdown['factchecker_score']:.4f}", "4% weight"],
        ["Subject alignment", f"{score_breakdown['subject_score']:.4f}", "3% weight"],
        ["Date quality", f"{score_breakdown['date_score']:.4f}", "2% weight"],
        ["Claim ID traceability", f"{score_breakdown['claim_id_score']:.4f}", "1% weight"],
        ["Classifier credible probability", f"{score_breakdown['classifier_credible']:.4f}", "1% weight"],
        ["Warning-language adjustment", f"{score_breakdown['warning_adjustment']:.4f}", "Penalty/bonus"],
        ["Base evidence score", f"{score_breakdown['base_score']:.4f}", "Before safeguards"],
        ["Final raw evidence score", f"{score_breakdown['final_raw_score']:.4f}", "After safeguards"],
        ["Displayed confidence", f"{evidence_confidence}%", "Final threshold anchor"],
    ]

    cbr_phases = {
        "Retrieve": (
            "The system converts the user claim into TF-IDF features and retrieves similar cases using cosine similarity. "
            "It also re-ranks cases using claim type, concept overlap, keyword coverage, and population/context compatibility."
        ),
        "Reuse": (
            "The system reuses the full case record: claim_id, claim, date_published, explanation, "
            "fact_checkers, main_text, sources, label, and subjects."
        ),
        "Revise": (
            "The system revises the evidence using expert rules, context verification, source quality, "
            "explanation quality, main-text support, fact-checker status, and safeguards."
        ),
        "Retain": (
            "The system does not automatically store user inputs. New cases should only be retained after expert validation."
        ),
    }

    return {
        "status": "ok",
        "input_claim": user_claim,
        "health_check_reason": health_reason,
        "health_report": health_report,
        "final_result": final_result,
        "confidence": evidence_confidence,
        "strongest_similarity": float(top_cases.iloc[0]["advanced_similarity"]),
        "top_cases": top_cases,
        "explanation_summary": explanation_summary,
        "explanation_detail": explanation_detail,
        "expert_messages": expert_messages,
        "safeguards": safeguards,
        "parameter_report": parameter_report,
        "score_breakdown": score_breakdown,
        "confidence_rows": confidence_rows,
        "cbr_phases": cbr_phases,
    }


# =========================================================
# EXPLANATION SYNTHESIS
# =========================================================
def synthesize_explanations(
    final_result: str,
    confidence: int,
    top_cases: pd.DataFrame,
    expert_messages: List[str],
    safeguards: List[str],
    parameter_report: Dict[str, object],
    score_breakdown: Dict[str, object],
) -> Tuple[str, str]:
    top = top_cases.iloc[0]

    best_claim_id = safe_text(top.get("claim_id", "")) or "No claim ID listed"
    best_case = safe_text(top.get("claim", ""))
    best_source = safe_text(top.get("sources", "")) or "No source listed"
    best_factchecker = safe_text(top.get("fact_checkers", "")) or "No fact-checker listed"
    best_date = safe_text(top.get("date_published", "")) or "No date listed"
    best_subjects = safe_text(top.get("subjects", "")) or "No subjects listed"
    best_main = safe_text(top.get("main_text", ""))
    best_expl = safe_text(top.get("explanation", ""))

    labels = top_cases["label"].astype(str).apply(classify_label).value_counts().to_dict()
    display_labels = display_label_dict(labels)
    display_final_result = display_label(final_result)

    evidence_sentences = parameter_report.get("evidence_sentences", [])
    if evidence_sentences:
        evidence_text = " ".join([
            f"“{s[:260]}...”" if len(s) > 260 else f"“{s}”"
            for s in evidence_sentences[:3]
        ])
    else:
        evidence_text = "No direct evidence sentence was extracted."

    main_preview = normalize_whitespace(best_main[:450]) or "No main text was available."
    expl_preview = normalize_whitespace(best_expl[:450]) or "No explanation was available."

    key_reasons = " ".join((safeguards + expert_messages)[:5])
    if not key_reasons:
        key_reasons = "The system mainly relied on retrieved case similarity and evidence score."

    top_context_relevance = float(top.get("context_relevance", 0.0))
    main_support = safe_text(parameter_report.get("main_text_support", "Unknown"))
    source_quality = safe_text(parameter_report.get("source_quality", "Unknown"))
    explanation_quality = safe_text(parameter_report.get("explanation_quality", "Unknown"))
    subject_quality = safe_text(parameter_report.get("subject_quality", "Unknown"))
    known_reasons = score_breakdown.get("known_reasons", []) or []
    warning_reasons = score_breakdown.get("warning_reasons", []) or []

    if known_reasons:
        simple_reason = known_reasons[0]
    elif safeguards:
        simple_reason = safeguards[0]
    elif main_support in {"Strongly supports", "Supports"}:
        simple_reason = "The closest case and its main text gave enough support for the user claim."
    elif main_support in {"Weak or conflicting", "Missing", "Limited support"}:
        simple_reason = "The retrieved evidence did not strongly support the exact wording of the user claim."
    elif warning_reasons:
        simple_reason = warning_reasons[0]
    else:
        simple_reason = "The system combined similarity, context, source quality, explanation quality, and retrieved labels to reach the result."

    summary = (
        f"MedCheck classified the claim as **{display_final_result}** with **{confidence}% evidence-based confidence**. "
        f"In simple terms, the system first checked if the input was a clear health claim, then compared it with similar stored cases using TF-IDF and cosine similarity. "
        f"After that, it verified whether the retrieved cases actually matched the same context using health concepts, claim type, population context, subject alignment, and main-text support. "
        f"The closest retrieved case was **'{best_case}'** with **{top_context_relevance:.0%} context relevance**, and the retrieved labels leaned as **{display_labels}**. "
        f"The system also checked the source (**{source_quality}**), explanation (**{explanation_quality}**), main text (**{main_support}**), and subject alignment (**{subject_quality}**). "
        f"Main reason: {simple_reason}"
    )

    detail = (
        f"### Detailed Reasoning\n\n"
        f"**Final Result:** {display_final_result}\n\n"
        f"**Evidence-Based Confidence:** {confidence}%\n\n"
        f"**Threshold Used:** Credible ≥ 70%, Possibly Misleading (Uncertain) = 40%–69%, Misleading < 40%.\n\n"
        f"**Closest Retrieved Case:** {best_case}\n\n"
        f"**Retrieved Label Pattern:** {display_labels}\n\n"
        f"---\n\n"
        f"### Full Dataset Parameters Checked\n\n"
        f"**Claim ID:** {best_claim_id}\n\n"
        f"**Claim:** {best_case}\n\n"
        f"**Date Published:** {best_date}\n\n"
        f"**Explanation:** {expl_preview}\n\n"
        f"**Fact-Checkers:** {best_factchecker}\n\n"
        f"**Main Text:** {main_preview}\n\n"
        f"**Sources:** {best_source}\n\n"
        f"**Label:** {safe_text(top.get('label', ''))}\n\n"
        f"**Subjects:** {best_subjects}\n\n"
        f"---\n\n"
        f"### Parameter Assessments\n\n"
        f"**Claim ID Traceability: {parameter_report.get('claim_id_quality')}**  \n"
        f"{parameter_report.get('claim_id_reason')}\n\n"
        f"**Source Quality: {parameter_report.get('source_quality')}**  \n"
        f"{parameter_report.get('source_reason')}\n\n"
        f"**Fact-Checker Status: {parameter_report.get('factchecker_quality')}**  \n"
        f"{parameter_report.get('factchecker_reason')}\n\n"
        f"**Explanation Quality: {parameter_report.get('explanation_quality')}**  \n"
        f"{parameter_report.get('explanation_reason')}\n\n"
        f"**Main Text Support: {parameter_report.get('main_text_support')}**  \n"
        f"{parameter_report.get('main_text_reason')}\n\n"
        f"**Date Quality: {parameter_report.get('date_quality')}**  \n"
        f"{parameter_report.get('date_reason')}\n\n"
        f"**Subject Alignment: {parameter_report.get('subject_quality')}**  \n"
        f"{parameter_report.get('subject_reason')}\n\n"
        f"---\n\n"
        f"### Evidence Sentences\n\n"
        f"{evidence_text}\n\n"
        f"---\n\n"
        f"### Expert Rule Reasoning\n\n"
        f"{' '.join(expert_messages[:14]) if expert_messages else 'No major expert-rule messages were triggered.'}\n\n"
        f"### Safeguards Applied\n\n"
        f"{' '.join(safeguards) if safeguards else 'No safeguard override was needed.'}\n\n"
        f"---\n\n"
        f"### Evidence Score Computation\n\n"
        f"The final confidence came from the raw evidence score. The system combined strongest match, average match, "
        f"context relevance, retrieved-label credibility, main-text support, source quality, explanation quality, "
        f"fact-checker status, subject alignment, date quality, claim ID traceability, classifier support, warning-language checks, "
        f"and known expert-rule safeguards. The final raw evidence score was **{score_breakdown.get('final_raw_score'):.4f}**, "
        f"which became **{confidence}%**."
    )

    return summary, detail


# =========================================================
# DASHBOARD + UI HELPERS
# =========================================================
def build_dashboard_label_counts(df: pd.DataFrame) -> pd.Series:
    counts = df["label_class"].value_counts()
    return counts.reindex(["Credible", "Possibly Misleading", "Misleading"], fill_value=0)


def show_case_distribution_card(df: pd.DataFrame) -> None:
    counts = build_dashboard_label_counts(df)
    total_cases = int(counts.sum())

    pie = go.Figure(
        data=[
            go.Pie(
                labels=[display_label(label) for label in counts.index.tolist()],
                values=counts.values.tolist(),
                hole=0.58,
                textinfo="percent",
                textfont=dict(size=14, color="#0f172a"),
                marker=dict(colors=["#22c55e", "#f59e0b", "#ef4444"]),
                sort=False,
            )
        ]
    )

    pie.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        annotations=[
            dict(
                text=f"<b>{total_cases:,}</b><br>Total Cases",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=20, color="#0f172a"),
            )
        ],
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            x=0.5,
            xanchor="center",
            font=dict(color="#0f172a"),
        ),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#0f172a"),
    )

    st.markdown("### Stored Cases")
    st.caption("Distribution of normalized credibility labels in the knowledge base")
    st.plotly_chart(pie, use_container_width=True, config={"displayModeBar": False})


def show_unique_labels_card(df: pd.DataFrame) -> None:
    counts = build_dashboard_label_counts(df)

    bar = go.Figure(
        data=[
            go.Bar(
                x=[display_label(label) for label in counts.index.tolist()],
                y=counts.values.tolist(),
                text=counts.values.tolist(),
                textposition="outside",
                marker=dict(color=["#22c55e", "#f59e0b", "#ef4444"]),
                hovertemplate="%{x}: %{y} cases<extra></extra>",
            )
        ]
    )

    bar.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title="Cases",
        xaxis_title="Label",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(color="#0f172a"),
        xaxis=dict(tickfont=dict(color="#0f172a"), title_font=dict(color="#0f172a")),
        yaxis=dict(
            tickfont=dict(color="#0f172a"),
            title_font=dict(color="#0f172a"),
            gridcolor="rgba(148,163,184,0.28)",
        ),
    )

    st.markdown(f"### Unique Labels ({int(df['label'].nunique())})")
    st.caption("Case count per normalized decision category")
    st.plotly_chart(bar, use_container_width=True, config={"displayModeBar": False})


def show_gauge(result: str, confidence: int) -> None:
    if result == "Credible":
        bar_color = "#16a34a"
    elif result == "Misleading":
        bar_color = "#dc2626"
    else:
        bar_color = "#f59e0b"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confidence,
            number={"suffix": "%"},
            title={"text": "Evidence-Based Confidence"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": bar_color},
                "steps": [
                    {"range": [0, 40], "color": "#fee2e2"},
                    {"range": [40, 70], "color": "#fef3c7"},
                    {"range": [70, 100], "color": "#dcfce7"},
                ],
            },
        )
    )

    fig.update_layout(height=320, margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(fig, use_container_width=True)


def show_parameter_assessment_guide() -> None:
    with st.expander("📘 Parameter Assessment Guide", expanded=False):
        st.caption("Open this guide to understand how MedCheck evaluates each parameter before giving a credibility result.")

        st.markdown("#### 🧾 Claim ID Traceability")
        st.write("Checks whether the retrieved case has a claim ID. A present claim ID helps track the case inside the dataset, but it does not directly prove that the claim is true or false.")
        st.markdown("**Present** — The case can be traced using a claim ID.")
        st.markdown("**Missing** — The case has weak traceability because no claim ID was found.")

        st.divider()

        st.markdown("#### 🔎 Source Quality")
        st.write("Checks whether the retrieved case comes from a reliable and traceable source.")
        st.markdown("**Strong** — Recognized health authority or medical source such as WHO, CDC, NIH, DOH, PubMed, Mayo Clinic, NHS, Cochrane, FDA, or similar.")
        st.markdown("**Fact-check source** — Known fact-checking or reputable media source such as PolitiFact, Snopes, Reuters, AP News, Health Feedback, Science Feedback, AFP, or Rappler.")
        st.markdown("**Moderate** — Source link exists, but it is not clearly a major health authority.")
        st.markdown("**Limited** — Minimal source information is available.")
        st.markdown("**Weak** — Missing, unclear, anonymous, viral, or weak-source information.")

        st.divider()

        st.markdown("#### 🧾 Fact-Checker Status")
        st.write("Checks whether the case has identifiable reviewer, author, or fact-checker information.")
        st.markdown("**Present** — Reviewer, author, or fact-checker is listed.")
        st.markdown("**Limited** — Some reviewer information exists but is incomplete.")
        st.markdown("**Missing** — No reviewer or fact-checker was detected.")

        st.divider()

        st.markdown("#### 📘 Explanation Quality")
        st.write("Checks whether the matched case explains why the claim is credible, uncertain, or misleading.")
        st.markdown("**Strong** — Detailed and evidence-aware explanation.")
        st.markdown("**Moderate** — Useful explanation but not very complete.")
        st.markdown("**Limited** — Short or only partly helpful.")
        st.markdown("**Weak / Missing** — Very weak, vague, missing, or contains cautionary/refuting wording.")

        st.divider()

        st.markdown("#### 🧠 Main Text Support")
        st.write("Checks whether the article/main content supports the exact user claim, not just similar words.")
        st.markdown("**Strongly supports** — Directly supports the exact claim.")
        st.markdown("**Supports** — Generally supports the claim.")
        st.markdown("**Neutral** — Related but does not clearly prove or disprove the exact claim.")
        st.markdown("**Limited support** — Related topic, but narrow, uncertain, or incomplete.")
        st.markdown("**Weak / Conflicting / Missing** — Does not support, contradicts, or lacks evidence.")

        st.divider()

        st.markdown("#### 📅 Date Quality")
        st.write("Checks whether the publication date is available and whether it is recent enough for time-sensitive health topics.")
        st.markdown("**Recent** — Stronger for time-sensitive topics like COVID, vaccines, or treatment guidelines.")
        st.markdown("**Moderate age** — Usable but may need caution for fast-changing health topics.")
        st.markdown("**Acceptable** — Date exists and the topic is not highly time-sensitive.")
        st.markdown("**Old / Unknown** — May reduce trust, especially for updated medical topics.")

        st.divider()

        st.markdown("#### 🏷️ Subject Alignment")
        st.write("Checks whether the dataset subject/category matches the user claim topic.")
        st.markdown("**Aligned** — Subject matches the detected health concept.")
        st.markdown("**Partially aligned** — Some subject terms overlap with the claim.")
        st.markdown("**Health-related** — Same general health domain but not an exact topic match.")
        st.markdown("**Weak / Missing** — Subject does not clearly match or is unavailable.")

        st.divider()

        st.markdown("#### 🧩 Context Relevance")
        st.write("Checks if retrieved cases match the meaning of the input, not only one keyword.")
        st.markdown("It considers keyword coverage, health concept overlap, claim type, human/animal population context, subject overlap, and similarity score.")

        st.divider()

        st.markdown("#### ⚖️ Retrieved-Label Credibility")
        st.write("Checks whether the retrieved cases lean toward Credible, Possibly Misleading (Uncertain), or Misleading. Labels from weak or unrelated matches are given less weight.")

        st.divider()

        st.markdown("#### 🧪 Linguistic Warning Rules")
        st.write("Checks risky wording such as miracle, instant, guaranteed, 100%, always, never, cure-all, detox, flush toxins, and similar exaggerated or absolute terms.")
        st.write("If this language appears without strong evidence, MedCheck becomes more cautious.")

        st.divider()

        st.markdown("#### 📊 Confidence Threshold")
        st.markdown("**Credible:** 70% and above")
        st.markdown("**Possibly Misleading (Uncertain):** 40% to 69%")
        st.markdown("**Misleading:** below 40%")
        st.write("The confidence score is the final anchor of the classification after context verification and safeguards.")


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.markdown("## ⚙️ MedCheck Settings")

dataset_path = st.sidebar.text_input(
    "Dataset path",
    value="reduceddataset4k.csv",
    help="Point this to your CSV or TSV file.",
)

top_k = st.sidebar.slider(
    "Top-K similar cases",
    min_value=3,
    max_value=15,
    value=7,
    step=1,
)

show_match = st.sidebar.toggle("Show retrieved cases", value=True)
show_parameters = st.sidebar.toggle("Show parameter analysis", value=True)
show_cbr = st.sidebar.toggle("Show CBR reasoning trace", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "by Group 4"
)


# =========================================================
# HEADER
# =========================================================
st.markdown(
    """
    <div class='hero'>
        <div class='hero-top'>
            <div class='hero-title-wrap'>
                <div class='hero-icon'>🩺</div>
                <div>
                    <h1>MedCheck</h1>
                    <div class='hero-subtitle'>A Case-Based Expert System for Health Claim Credibility and Misinformation Detection</div>
                </div>
            </div>
            <div class='hero-badges'>
                <div class='hero-badge'>Hirondelle D. Simbulan</div>
                <div class='hero-badge'>Kendrick A. Roslin</div>
                <div class='hero-badge'>Mark Kendrick P. Roxas</div>
            </div>
        </div>
        <p>
Disclaimer: MedCheck is an academic prototype designed to support health claim credibility checking. It does not replace medical experts, licensed healthcare professionals, or professional medical advice. The results generated by the system should be used only as a reference and should not be treated as a final diagnosis, treatment recommendation, or medical decision. For any health concern, users should consult a qualified healthcare professional.        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOAD DATA
# =========================================================
data_loaded = False

try:
    (
        case_df,
        claim_vectorizer,
        claim_matrix,
        content_vectorizer,
        content_matrix,
        classifier,
        classifier_classes,
    ) = prepare_case_base(dataset_path)
    data_loaded = True
except Exception as exc:
    data_loaded = False
    st.error(f"Dataset loading error: {exc}")
    st.info("Place your dataset file in the same folder as this app, or update the path in the sidebar.")


# =========================================================
# DASHBOARD
# =========================================================
if data_loaded:
    d1, d2 = st.columns(2)

    with d1:
        with st.container(border=True):
            show_case_distribution_card(case_df)

    with d2:
        with st.container(border=True):
            show_unique_labels_card(case_df)


# =========================================================
# MAIN APP
# =========================================================
left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Enter a health claim")

    user_claim = st.text_area(
        "Health claim",
        placeholder="Example: Skipping meals helps weight loss. / Lemon burns fat. / Water improves hydration and UTIs are caused by salty foods.",
        height=180,
        label_visibility="collapsed",
    )

    analyze_clicked = st.button(
        "Analyze Claim",
        use_container_width=True,
        type="primary",
        disabled=not data_loaded,
    )

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("How MedCheck works")
    st.markdown(
        """
        1. Strictly rejects non-health inputs.  
        2. Accepts broader health/nutrition claims using expanded vocabulary.  
        3. Retrieves similar cases using **TF-IDF + cosine similarity**.  
        4. Uses **all dataset parameters** for case representation.  
        5. Verifies context using concepts, claim type, population, subjects, and main text.  
        6. Blocks weak keyword-only matches from becoming decisions.  
        7. Applies known health/myth safeguards.  
        8. Uses the **raw evidence score** as the final threshold anchor.  
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# ANALYZE BUTTON LOGIC
# =========================================================
if analyze_clicked and data_loaded:
    result = analyze_claim(
        user_claim=user_claim,
        df=case_df,
        claim_vectorizer=claim_vectorizer,
        claim_matrix=claim_matrix,
        content_vectorizer=content_vectorizer,
        content_matrix=content_matrix,
        classifier=classifier,
        classifier_classes=classifier_classes,
        top_k=top_k,
    )

    st.markdown("### Analysis Result")

    st.markdown(
        """
        <div style='background:#ffffff; padding:12px; border-radius:12px; border:1px solid #e5e7eb; margin-bottom:10px; color:#0f172a;'>
        <b>Decision Threshold Reference:</b><br>
        • <span style='color:#16a34a; font-weight:600;'>Credible:</span> ≥ 70% raw evidence confidence<br>
        • <span style='color:#f59e0b; font-weight:600;'>Possibly Misleading (Uncertain):</span> 40% – 69% raw evidence confidence<br>
        • <span style='color:#dc2626; font-weight:600;'>Misleading:</span> &lt; 40% raw evidence confidence<br>
        • <span style='color:#334155; font-weight:600;'>No Decision:</span> non-health input or weak context match
        </div>
        """,
        unsafe_allow_html=True,
    )

    show_parameter_assessment_guide()

    if result["status"] == "error":
        st.warning(result["message"])

        if "health_report" in result:
            with st.expander("View health-domain check details"):
                st.json(result["health_report"])

    elif result["status"] == "no_decision":
        st.info(result["message"])

        if "health_report" in result:
            with st.expander("View health-domain check details"):
                st.json(result["health_report"])

        if show_match and "top_cases" in result:
            st.markdown("### Weak Retrieved Cases Checked")
            tc = result["top_cases"].copy()
            tc["Match Strength"] = tc["advanced_similarity"].map(match_strength_percent)
            tc["Context Relevance"] = tc["context_relevance"].map(match_strength_percent)
            if "case_result" in tc.columns:
                tc["case_result"] = tc["case_result"].map(display_label)
            if "label" in tc.columns:
                tc["label"] = tc["label"].map(lambda value: display_label(safe_text(value)) if safe_text(value) == "Possibly Misleading" else safe_text(value))

            display_cols = [
                "claim_id",
                "claim",
                "label",
                "subjects",
                "Match Strength",
                "Context Relevance",
                "context_explanation",
            ]

            display_cols = [c for c in display_cols if c in tc.columns]

            st.dataframe(
                tc[display_cols].rename(columns={
                    "claim_id": "Claim ID",
                    "claim": "Retrieved Claim",
                    "label": "Original Label",
                    "subjects": "Subjects",
                    "context_explanation": "Context Explanation",
                }),
                use_container_width=True,
                hide_index=True,
            )

    else:
        show_gauge(result["final_result"], result["confidence"])

        result_label = display_label(result["final_result"])
        class_name = result_class_name(result["final_result"])

        st.markdown(
            f"""
            <div class='card'>
                <div class='result-pill {class_name}'>{result_label}</div>
                <h2 style='margin-top:0;'>Evidence-Based Confidence: {result['confidence']}%</h2>
                <p class='small-muted'>
                    The final classification follows the raw evidence score after context verification and expert safeguards.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        m1, m2, m3 = st.columns(3)

        with m1:
            st.metric("Final Result", result_label)

        with m2:
            st.metric("Confidence", f"{result['confidence']}%")

        with m3:
            st.metric("Closest Match", match_strength_percent(result["strongest_similarity"]))

        st.markdown("### Expert Explanation")
        st.markdown(f"<div class='summary-box'>{result['explanation_summary']}</div>", unsafe_allow_html=True)

        with st.expander("View detailed explanation"):
            st.markdown(f"<div class='explain-box'>{result['explanation_detail']}</div>", unsafe_allow_html=True)

        if show_match:
            st.markdown(f"### Top-{top_k} Similar Cases")
            tc = result["top_cases"].copy()

            tc["Match Strength"] = tc["advanced_similarity"].map(match_strength_percent)
            tc["Claim Match"] = tc["claim_similarity"].map(match_strength_percent)
            tc["Content Match"] = tc["content_similarity"].map(match_strength_percent)
            tc["Type Compatibility"] = tc["type_compatibility"].map(lambda x: f"{x:.2f}")
            tc["Population Compatibility"] = tc["population_compatibility"].map(lambda x: f"{x:.2f}")
            tc["Context Relevance"] = tc["context_relevance"].map(match_strength_percent)
            if "case_result" in tc.columns:
                tc["case_result"] = tc["case_result"].map(display_label)
            if "label" in tc.columns:
                tc["label"] = tc["label"].map(lambda value: display_label(safe_text(value)) if safe_text(value) == "Possibly Misleading" else safe_text(value))

            display_cols = [
                "claim_id",
                "claim",
                "label",
                "case_result",
                "date_published",
                "fact_checkers",
                "sources",
                "subjects",
                "claim_type",
                "case_population_context",
                "Match Strength",
                "Claim Match",
                "Content Match",
                "Type Compatibility",
                "Population Compatibility",
                "Context Relevance",
                "context_explanation",
            ]

            display_cols = [c for c in display_cols if c in tc.columns]

            tc = tc[display_cols].rename(columns={
                "claim_id": "Claim ID",
                "claim": "Retrieved Claim",
                "label": "Original Label",
                "case_result": "Normalized Result",
                "date_published": "Date Published",
                "fact_checkers": "Fact Checkers",
                "sources": "Sources",
                "subjects": "Subjects",
                "claim_type": "Matched Claim Type",
                "case_population_context": "Case Context",
                "context_explanation": "Context Explanation",
            })

            st.dataframe(tc, use_container_width=True, hide_index=True)

        if show_parameters:
            st.markdown("### Current Parameter Analysis")
            pr = result["parameter_report"]

            param_df = pd.DataFrame(
                [
                    ["Claim ID Traceability", pr.get("claim_id_quality", "Unknown"), pr.get("claim_id_reason", "")],
                    ["Source Quality", pr.get("source_quality", "Unknown"), pr.get("source_reason", "")],
                    ["Fact-Checker Status", pr.get("factchecker_quality", "Unknown"), pr.get("factchecker_reason", "")],
                    ["Explanation Quality", pr.get("explanation_quality", "Unknown"), pr.get("explanation_reason", "")],
                    ["Main Text Support", pr.get("main_text_support", "Unknown"), pr.get("main_text_reason", "")],
                    ["Date Quality", pr.get("date_quality", "Unknown"), pr.get("date_reason", "")],
                    ["Subject Alignment", pr.get("subject_quality", "Unknown"), pr.get("subject_reason", "")],
                    ["Credible Ratio", f"{pr.get('credible_ratio', 0):.2f}", "Share of retrieved cases normalized as Credible."],
                    ["Possibly Misleading (Uncertain) Ratio", f"{pr.get('possibly_ratio', 0):.2f}", "Share of retrieved cases normalized as Possibly Misleading (Uncertain)."],
                    ["Misleading Ratio", f"{pr.get('misleading_ratio', 0):.2f}", "Share of retrieved cases normalized as Misleading."],
                ],
                columns=["Parameter", "Assessment", "Why"],
            )

            st.dataframe(param_df, use_container_width=True, hide_index=True)

            with st.expander("Health-domain vocabulary check"):
                st.json(result.get("health_report", {}))

            st.markdown("### Safeguards Applied")
            if result["safeguards"]:
                for i, msg in enumerate(result["safeguards"], start=1):
                    st.write(f"**Safeguard {i}:** {msg}")
            else:
                st.write("No safeguard override was needed.")

            st.markdown("### Expert Rule Reasons")
            for i, msg in enumerate(result["expert_messages"][:14], start=1):
                st.write(f"**Reason {i}:** {msg}")

            st.markdown("### Confidence Computation Breakdown")
            conf_df = pd.DataFrame(
                result["confidence_rows"],
                columns=["Component", "Value", "Role"],
            )
            st.dataframe(conf_df, use_container_width=True, hide_index=True)

        if show_cbr:
            st.markdown("### Case-Based Reasoning Trace")
            st.markdown("<div class='card'>", unsafe_allow_html=True)

            for phase, text in result["cbr_phases"].items():
                st.write(f"**{phase}:** {text}")

            st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption(
    "Built with Python, Streamlit, TF-IDF, cosine similarity, case-based reasoning, "
    "expanded vocabulary, full-parameter evidence scoring, context verification, and expert safeguards. "
    "For academic prototype use only; this tool does not replace professional medical advice."
)
