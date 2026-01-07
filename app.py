import io
import csv
import threading
import re
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
import numpy as np
from PIL import Image

import pytesseract

from ocr.main import extract_transactions_from_image, run_test

app = Flask(__name__, static_folder='static/')

# Pot do CSV datoteke
CSV_PATH = Path('rezultati.csv')
csv_lock = threading.Lock()

test_mode = True

# Ustvari CSV datoteko če ne obstaja
if not CSV_PATH.exists():
    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['cas', 'povprecna_vrednost', 'velikost_slike'])


def zapisi_v_csv(cas: str, povprecna_vrednost: float, velikost: tuple) -> None:
    """Zapiše rezultat v CSV datoteko (thread-safe)."""
    with csv_lock:
        with open(CSV_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([cas, povprecna_vrednost, f'{velikost[0]}x{velikost[1]}'])


@app.route('/')
def index():
    """Servira glavno HTML stran."""
    return send_from_directory('static/', 'index.html')


@app.route('/analiziraj', methods=['POST'])
def analiziraj():
    """
    Prejme sliko in časovno značko, izračuna povprečno vrednost.
    
    Pričakovani format zahtevka:
    - multipart/form-data
    - polje 'slika': slika (JPEG, PNG, ...)
    - polje 'cas': časovna značka (ISO format ali timestamp)
    """
    # Preveri ali je slika prisotna
    if 'slika' not in request.files:
        return jsonify({
            'napaka': 'Manjka polje "slika"'
        }), 400
    
    slika_file = request.files['slika']
    
    if slika_file.filename == '':
        return jsonify({
            'napaka': 'Prazno ime datoteke'
        }), 400
    
    # Pridobi časovno značko
    cas = request.form.get('cas')
    if cas is None:
        # Če časovna značka ni podana, uporabi trenutni čas
        cas = datetime.now().isoformat()
    
    try:
        # Naloži sliko
        slika_bytes = slika_file.read()
        slika_pil = Image.open(io.BytesIO(slika_bytes))
        
        # Pretvori v numpy tabelo
        slika_np = np.array(slika_pil)
        
        # Analiziraj
        povprecna_vrednost = float(np.mean(slika_np))
        
        # Zapiši rezultat v bazo
        velikost = slika_np.shape[:2]
        zapisi_v_csv(cas, povprecna_vrednost, velikost)

        transactions = extract_transactions_from_image(slika_pil)

        return jsonify({
            'cas': cas,
            'povprecna_vrednost': povprecna_vrednost,
            'velikost_slike': {
                'visina': velikost[0],
                'sirina': velikost[1]
            },
            'transactions': transactions,
            'status': 'uspeh'
        })
    
    except Exception as e:
        return jsonify({
            'napaka': f'Napaka pri obdelavi slike: {str(e)}',
            'status': 'napaka'
        }), 500


@app.route('/zgodovina', methods=['GET'])
def zgodovina():
    """Vrne zgodovino vseh analiz."""
    try:
        with csv_lock:
            with open(CSV_PATH, 'r') as f:
                reader = csv.DictReader(f)
                rezultati = list(reader)
        
        return jsonify({
            'stevilo_rezultatov': len(rezultati),
            'rezultati': rezultati,
            'status': 'uspeh'
        })
    
    except Exception as e:
        return jsonify({
            'napaka': f'Napaka pri branju zgodovine: {str(e)}',
            'status': 'napaka'
        }), 500
@app.route('/test', methods=['GET'])
def test():
    try:
        transactions = run_test()

        print(transactions)

        return jsonify({
            'transactions': transactions,
            'status': 'uspeh'
        })
    except Exception as e:
        print(f'Napaka pri izvajanju testa: {str(e)}')
        return jsonify({
            'napaka': f'Napaka pri izvajanju testa: {str(e)}',
            'status': 'napaka'
        }), 500

if __name__ == '__main__':
    print('Zaganjam Flask storitev na http://localhost:5000')
    print('Endpoints:')
    print('  GET  /              - Spletna stran z kamero')
    print('  POST /analiziraj    - Analiziraj sliko')
    print('  GET  /zgodovina     - Prikaži zgodovino')
    print('\nOdpri brskalnik na: http://localhost:5000')
    app.run(host='localhost', port=5000, debug=True)
