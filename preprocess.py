"""preprocess.py
Utilities to extract transactions from PDF or CSV and produce a tidy DataFrame.
"""
import re
import io
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
import pdfplumber
from dateutil import parser as dateparser




CURRENCY_RE = re.compile(r'[\u20B9₹$€£]')




def normalize_amount(raw: str) -> float:
if raw is None:
return np.nan
s = str(raw)
s = CURRENCY_RE.sub('', s)
s = s.replace(',', '').strip()
neg = False
if '(' in s and ')' in s:
neg = True
s = s.replace('(', '').replace(')', '')
m = re.search(r'-?\d+(?:\.\d+)?', s)
if not m:
return np.nan
val = float(m.group(0))
return -abs(val) if neg or str(val).startswith('-') else float(val)




def parse_date_guess(raw: str):
if raw is None:
return pd.NaT
try:
cleaned = re.sub(r'(\d)(st|nd|rd|th)', r"\1", str(raw))
dt = dateparser.parse(cleaned, dayfirst=True, fuzzy=True)
return pd.Timestamp(dt.date())
except Exception:
return pd.NaT




def extract_text_from_pdf(path: str) -> str:
text = []
with pdfplumber.open(path) as pdf:
for p in pdf.pages:
t = p.extract_text() or ''
text.append(t)
return '\n'.join(text)




def parse_lines_to_transactions(text: str) -> pd.DataFrame:
rows = []
for line in text.splitlines():
print('Provide --pdf or --csv')
