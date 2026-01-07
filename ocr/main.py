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

test_image_path = "data/images/2/PXL_20260106_133002098.jpg"
test_data_path = "data/data/2.json"

def run_test():
    image = Image.open(test_image_path)
#    transactions = extract_transactions_from_image(image)
    transactions = pytesseract.image_to_string(image)

    return transactions


def extract_transactions_from_image(image: Image.Image):
    print("Začen ekstrakcija transakcij iz slike...")

    text = pytesseract.image_to_string(image, lang='eng')

    # TODO: Iz besedila tukaj dobi transakicje in druge podatke

    print("Ekstrakcija končana.")

    return transactions