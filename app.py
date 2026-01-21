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
from utils.testingUtil import run_comparison_test, get_statistics_summary

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
    if 'slika' not in request.files:
        return jsonify({
            'napaka': 'Manjka polje "slika"'
        }), 400
    
    slika_file = request.files['slika']
    
    if slika_file.filename == '':
        return jsonify({
            'napaka': 'Prazno ime datoteke'
        }), 400

    cas = request.form.get('cas')
    if cas is None:
        cas = datetime.now().isoformat()
    
    try:
        slika_bytes = slika_file.read()
        slika_pil = Image.open(io.BytesIO(slika_bytes))

        data = extract_transactions_from_image(slika_pil)

        return jsonify({
            'cas': cas,
            'data': data,
        })
    
    except Exception as e:
        return jsonify({
            'napaka': f'Napaka pri obdelavi slike: {str(e)}',
            'status': 'napaka'
        }), 500


@app.route('/test', methods=['POST'])
def test():
    print("Prejeto zahtevo za testiranje.")
    if 'slika' not in request.files:
        return jsonify({
            'napaka': 'Manjka polje "slika"'
        }), 400

    slika_file = request.files['slika']

    if slika_file.filename == '':
        return jsonify({
            'napaka': 'Prazno ime datoteke'
        }), 400

    time = request.form.get('cas')
    if time is None:
        time = datetime.now().isoformat()

    statement_number = request.form.get('primerjalna_stevilka')
    if statement_number is None:
        statement_number = '1'

    try:
        slika_bytes = slika_file.read()
        slika_pil = Image.open(io.BytesIO(slika_bytes))

        #test_image_path = 'data/images/5/PXL_20260106_132658419.jpg'
        #slika_pil = Image.open(test_image_path)

        data = run_test(slika_pil)
        
        # Izvedi primerjalni test
        test_result = run_comparison_test(statement_number, data)
        statistics = get_statistics_summary(test_result)

        return jsonify({
            'type': 'test',
            'time': time,
            'statistics': statistics,
            'data': data,
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
    print('\nOdpri brskalnik na: http://localhost:5000')
    app.run(host='localhost', port=5001, debug=True)
