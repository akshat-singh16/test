import re
import pandas as pd
import numpy as np
import pdfplumber
from dateutil import parser as dateparser


# ----------- Helpers -----------

def normalize_amount(raw: str) -> float:
    """
    Convert raw amount strings into clean float values.
    """
    if raw is None:
        return np.nan

    s = str(raw)
    s = re.sub(r"[₹$,]", "", s)
    s = s.replace("(", "-").replace(")", "")
    s = s.strip()

    try:
        return float(s)
    except:
        match = re.search(r"-?\d+(\.\d+)?", s)
        return float(match.group(0)) if match else np.nan


def parse_date_guess(raw: str):
    if raw is None:
        return pd.NaT
    try:
        cleaned = re.sub(r"(\d)(st|nd|rd|th)", r"\1", str(raw))
        dt = dateparser.parse(cleaned, dayfirst=True, fuzzy=True)
        return pd.Timestamp(dt.date())
    except:
        return pd.NaT


# ----------- PDF Extraction -----------

def extract_text_from_pdf(path: str) -> str:
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += (page.extract_text() or "") + "\n"
    return text


def parse_lines_to_transactions(text: str) -> pd.DataFrame:
    rows = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # detect date + amount
        date_match = re.search(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", line)
        amount_match = re.search(r"[₹$]?\s*-?\(?\d[\d,]*(\.\d+)?\)?", line)

        if date_match and amount_match:
            date_str = date_match.group(0)
            amt_str = amount_match.group(0)

            txn_date = parse_date_guess(date_str)
            amount = normalize_amount(amt_str)

            merchant = line.replace(date_str, "").replace(amt_str, "").strip()

            rows.append({
                "txn_date": txn_date,
                "amount": amount,
                "merchant": merchant[:100],
                "description": line
            })

    return pd.DataFrame(rows)


# ----------- CSV Extraction -----------

def load_csv_transactions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Identify columns
    date_col = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
    amt_col = next((c for c in df.columns if "amount" in c.lower()), df.columns[1])
    desc_col = next((c for c in df.columns if "desc" in c.lower()), df.columns[-1])

    out = pd.DataFrame()
    out["txn_date"] = df[date_col].apply(parse_date_guess)
    out["amount"] = df[amt_col].apply(normalize_amount)
    out["merchant"] = df[desc_col].astype(str).str[:100]
    out["description"] = df[desc_col].astype(str)

    return out
