import io
import csv
import threading
import re
from datetime import datetime
from pathlib import Path
from typing import List

from flask import Flask, request, jsonify, send_from_directory
import numpy as np
from PIL import Image
import utils.LLMUtil as LLMUtil
import pytesseract

test_image_path = "data/images/2/PXL_20260106_133002098.jpg"
test_data_path = "data/data/2.json"
known_partners = [
    "LANA K.",
    "UNIFITNES, D.O.O.",
    "MDDSZ-DRZAVNE STIPENDIJE - ISCSD 2",
    "ASPIRIA d.o.o.",
    "PayPal Europe S.a.r.l. et Cie S.C.A",
    "TELEKOM SLOVENIJE D.D.",
    "PE DRAVSKE TERASE"
]


def run_test(image):
    transactions = extract_transactions_from_image(image=image, locations=known_partners)

    return transactions

def extract_transactions_from_image(image: Image.Image, locations: List[str]):
    print("Zacenjam ekstrakcijo transakcij iz slike...")

    text = pytesseract.image_to_string(image, lang='eng')
    print("Ekstrakcija koncana.")
    print(text)
    #print(compose_llm_prompt(text, locations))
    print("Posiljam besedilo LLM-ju za nadaljnjo obdelavo...")
    try:
        response = LLMUtil.ask_MrGPT(
            compose_llm_prompt(text, locations),
            "You are a bank statement parser.",
            "llama-3.3-70b-versatile"
        )
    except Exception as e:
        print("LLM request failed:", e)
        response = None
    print("Prejeto od LLM: ", response)
    return response if response is not None else text


def compose_llm_prompt(text: str, known_partners: List[str]) -> str:

    prompt = f"""
Iz naslednjega bančnega izpiska izlušči transakcije in jih pretvori v JSON.

**LASTNIK RAČUNA:** KUDER LUKA (IBAN: SI56 6100 0002 2720 075)

**KRITIČNA PRAVILA ZA POLJE `change`:**
- Če je transakcija ODLIV (stanje pada): `change` mora biti **POZITIVNO število**
- Če je transakcija PRILIV (stanje narašča): `change` mora biti **POZITIVNO število**
- **NIKOLI ne uporabljaj negativnih številk** v `change`
- Primer: če stanje pade iz 4800.17 na 4563.12, je `change: 237.05` (ne -237.05)
- Ne vstavljaj pike za tisočice (npr. 1.000.00 ni dovoljeno, uporabi 1000.00)

**PRAVILA ZA POLJE `partner`:**
- Normaliziraj imena (odstrani odvečne presledke, popravi tipkarske napake)
  - Lahko se zgodi, da bo ime, priimek združeno (npr. "LANAK.", "DAVIDG.")
    - v takih primerih ko je zadnji znak ".", sta zadnja dva znaka priimek
    - Razdeli ime in priimek pravilno (npr. "LANAK." -> "LANA K.", "DAVIDG." -> "DAVID G.")
  - Pazi na partnerje, ki imajo poln priimek in so brez presledka (npr. "KUDERLUKA", "KRAMARENEJ")
    - Razdeli jih pravilno tako, da poiščeš kje je ime (npr. "KUDERLUKA" -> "KUDER LUKA", "KRAMARENEJ" -> "KRAMAR ENEJ")
- Razen v posebnih primerih, kjer je jasno navedeno drugače, vzami dve besedi kot ime in priimek

**PRAVILA ZA POLJE `description`:**
- Kopiraj celoten opis transakcije kot je v izpisu
- Ohrani originalne podrobnosti (reference na PayPal, opombe, itd.)

**PRAVILA ZA POLJE `known_partner`:**
- `true` **samo** če se partner **točno** ujema s seznamom znanih partnerjev spodaj
- Primerjaj celotno ime (npr. "PayPal Europe S.a.r.l. et Cie S.C.A" ≠ "PayPal")
- `false` za vse ostale, vključno z "KUDER LUKA"
- Normalizacija imena izvajaj enako kot za polje `partner`

**ZNANI PARTNERJI (case-sensitive):**
{chr(10).join(f"- {p}" for p in known_partners)}

**FORMATO:**
- Datumi: `DD.MM.YYYY`
- Števila: decimalna pika (npr. `237.05`)
- Boolean: `true`/`false` (lowercase)
- Vrni **samo** veljaven JSON, brez dodatnega besedila ali markdown

**PRIMER:**
{{
  "transactions": [
    {{
      "date": "17.02.2025",
      "reference": "729098128",
      "partner": "KUDER LUKA",
      "description": "ZAVERSKI",
      "change": 237.05,
      "balance": 4563.12,
      "outgoing": true,
      "known_partner": false
    }},
    {{
      "date": "14.02.2025",
      "reference": "728503613",
      "partner": "LANA K.",
      "description": "lidl",
      "change": 20.00,
      "balance": 4864.69,
      "outgoing": false,
      "known_partner": true
    }}
  ]
}}

**BESEDILO ZA OBDELAVO:**
{text}
"""
    return prompt
