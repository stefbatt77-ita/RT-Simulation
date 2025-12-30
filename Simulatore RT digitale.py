import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter
import random
import pandas as pd

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Aero-NDT Ultimate v10.6", layout="wide", page_icon="✈️")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #e0e0e0; }
    h1, h2, h3 { color: #ffffff; }
    div.stButton > button { font-weight: bold; border-radius: 6px; height: 3em; }
    div.stButton > button:first-child { background-color: #d32f2f; color: white; border: none; }
    .stTable { background-color: #1e1e1e; }
    </style>
    """, unsafe_allow_html=True)

# --- INIZIALIZZAZIONE STATO (Previene AttributeError) ---
keys_to_init = [
    'exam_progress', 'exam_results', 'exam_cases', 'current_case',
    's_img', 's_bboxes', 's_defs', 's_eval', 'mode',
    'e_img', 'e_defs'
]
for key in keys_to_init:
    if key not in st.session_state:
        if key == 'exam_results' or key == 'exam_cases': st.session_state[key] = []
        elif key == 'exam_progress': st.session_state[key] = 0
        elif key == 'mode': st.session_state[key] = "STUDIO (Training)"
        elif key == 's_eval': st.session_state[key] = False
        else: st.session_state[key] = None

# --- MOTORE FISICO E GENERAZIONE ---
def get_ideal_params(material, thickness):
    # Calcolo parametri ideali per grigio medio e buon contrasto
    props = {
        "Al-2024 (Avional)": {"b": 45, "k": 3.2},
        "Ti-6Al-4V":         {"b": 65, "k": 4.8},
        "Inconel 718":       {"b": 95, "k": 7.8},
        "Steel 17-4 PH":     {"b": 85, "k": 6.5}
    }
    p = props[material]
    id_kv = int(p["b"] + (thickness * p["k"]))
    id_kv = max(40, min(250, id_kv)) # Limitiamo ai limiti macchina
    
    # mAs esponenziali rispetto allo spessore
    id_mas = round(12 * np.exp(thickness * 0.06), 1)
    return id_kv, id_mas

def generate_scan_final(kv, ma, time, material, thickness, selected_defect=None):
    size = 600
    # Fisica attenuazione
    mu_map = {"Al-2024 (Avional)": 0.02, "Ti-6Al-4V": 0.045, "Inconel 718": 0.09, "Steel 17-4 PH": 0.075}
    safe_kv = max(10, kv)
    mu = mu_map[material] * (120/safe_kv)**1.5
    
    m_sp = np.full((size, size), float(thickness), dtype=float)
    
    # 1. IQI A FILI (ISO 19232-1) - Alto
    for i in range(7):
        m_sp[40:140, 80 + i*40 : 80 + i*40 + 2] += (0.4 - i*0.05)
    
    # 2. DUPLEX (ISO 19232-5) - Basso
    for i in range(13):
        m_sp[500:540, 150 + i*25 : 150 + i*25 + 2] += (0.7 / (i+1))

    # Etichette (Simulate togliendo spessore)
    m_sp[145:155, 80:120] -= 0.5 # Label "ISO"
    m_sp[545:555, 150:200] -= 0.5 # Label "DUP"

    # 3. GENERAZIONE DIFETTI
    defects_list = []
    bboxes = []
    
    # Logica di selezione
    to_generate = []
    if selected_defect == "Casuale (Multiplo)":
        num = random.randint(1, 3)
        possibles = ["Cricca", "Porosità Singola", "Cluster Porosità", "Inclusione Tungsteno", "Mancata Fusione"]
        to_generate = random.sample(possibles, num)
    elif selected_defect == "Nessun Difetto":
        to_generate = []
    else:
        to_generate = [selected_defect]

    # Rendering Difetti
    for d_type in to_generate:
        # Coordinate random (evitando IQI)
        rx, ry = random.randint(100, 500), random.randint(180, 450)
        
        if d_type == "Cricca":
            # Random Walk per forma irregolare
            points_x, points_y = [], []
            curr_x, curr_y = rx, ry
            angle = random.uniform(0, 360)
            length = random.randint(50, 100)
            for _ in range(length):
                rad = np.radians(angle)
                curr_x += np.cos(rad) + random.uniform(-0.5, 0.5)
                curr_y += np.sin(rad) + random.uniform(-0.5, 0.5)
                if 0 <= int(curr_x) < size and 0 <= int(curr_y) < size:
                    m_sp[int(curr_y), int(curr_x)] -= 0.8
                    points_x.append(curr_x); points_y.append(curr_y)
            
            if points_x:
                bboxes.append({"x": min(points_x)-10, "y": min(points_y)-10, 
                               "w": max(points_x)-min(points_x)+20, "h": max(points_y)-min(points_y)+20, "t": "Cricca"})

        elif d_type == "Porosità Singola":
            y, x = np.ogrid[:size, :size]
            m_sp[(x-rx)**2 + (y-ry)**2