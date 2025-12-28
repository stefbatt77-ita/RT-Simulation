import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter, laplace, rotate
import random
import pandas as pd

# --- CONFIGURAZIONE ---
st.set_page_config(page_title="Aero-NDT Advanced v10.3", layout="wide", page_icon="☢️")

# --- MOTORE DI GENERAZIONE AVANZATO ---
def generate_scan_v10_3(kv, ma, time, material, thickness, exam_mode=False):
    size = 600
    mu_map = {"Al-2024 (Avional)": 0.02, "Ti-6Al-4V": 0.045, "Inconel 718": 0.09, "Steel 17-4 PH": 0.075}
    mu = mu_map[material] * (120/max(10, kv))**1.5
    m_sp = np.full((size, size), float(thickness), dtype=float)
    
    # 1. IQI & DUPLEX CON ETICHETTE
    # IQI Fili (Alto)
    for i in range(7):
        m_sp[40:140, 80 + i*40 : 80 + i*40 + 2] += (0.4 - i*0.05)
    # Duplex (Basso)
    for i in range(13):
        m_sp[500:540, 150 + i*25 : 150 + i*25 + 2] += (0.7 / (i+1))
    
    # Rappresentazione scritte (simulate come sottrazioni di spessore)
    m_sp[145:160, 80:150] -= 0.5 # Etichetta ISO
    m_sp[545:560, 150:250] -= 0.5 # Etichetta DUPLEX

    # 2. GENERAZIONE DIFETTI MULTIPLI
    num_defects = random.randint(1, 3) if exam_mode else 1
    detected_defects = []
    all_bboxes = []

    for _ in range(num_defects):
        def_type = random.choice(["Cricca", "Porosità Singola", "Cluster Porosità", "Inclusione Tungsteno", "Mancata Fusione"])
        rx = random.randint(100, 500)
        ry = random.randint(180, 450) # Evita sovrapposizione con IQI
        
        if def_type == "Cricca":
            angle = random.randint(0, 360)
            length = random.randint(50, 150)
            # Creazione cricca orientata
            crack_mask = np.zeros((size, size))
            curr_x, curr_y = rx, ry
            for i in range(length):
                rad = np.radians(angle)
                curr_x += np.cos(rad) + random.uniform(-0.5, 0.5)
                curr_y += np.sin(rad) + random.uniform(-0.5, 0.5)
                if 0 <= int(curr_y) < size and 0 <= int(curr_x) < size:
                    m_sp[int(curr_y), int(curr_x)] -= 0.8
            all_bboxes.append({"x": rx-20, "y": ry-20, "w": 60, "h": 60, "t": "Cricca"})

        elif def_type == "Porosità Singola":
            y, x = np.ogrid[:size, :size]
            m_sp[(x-rx)**2 + (y-ry)**2 <= 5**2] -= 2.5
            all_bboxes.append({"x": rx-10, "y": ry-10, "w": 20, "h": 20, "t": "Porosità"})

        elif def_type == "Cluster Porosità":
            for _ in range(10):
                cx, cy = rx + random.randint(-25, 25), ry + random.randint(-25, 25)
                y, x = np.ogrid[:size, :size]
                m_sp[(x-cx)**2 + (y-cy)**2 <= 3**2] -= 1.8
            all_bboxes.append({"x": rx-35, "y": ry-35, "w": 70, "h": 70, "t": "Cluster"})

        elif def_type == "Inclusione Tungsteno":
            y, x = np.ogrid[:size, :size]
            m_sp[(x-rx)**2 + (y-ry)**2 <= 4**2] += 12.0
            all_bboxes.append({"x": rx-8, "y": ry-8, "w": 16, "h": 16, "t": "Tungsteno"})

        elif def_type == "Mancata Fusione":
            h, w = random.randint(100, 300), 3
            m_sp[ry:ry+h, rx:rx+w] -= 1.5
            all_bboxes.append({"x": rx-5, "y": ry, "w": 15, "h": h, "t": "Mancata Fusione"})
        
        detected_defects.append(def_type)

    # Rendering Finale
    dose = (kv**2) * ma * time * 0.05
    signal = dose * np.exp(-mu * m_sp)
    signal = gaussian_filter(signal, sigma=1.1)
    noise = np.random.normal(0, np.sqrt(signal + 1) * 2.2, (size, size))
    raw = np.clip(signal + noise, 0, 65535).astype(np.uint16)
    
    return raw, detected_defects, all_bboxes

# --- LOGICA DI VALUTAZIONE ---
def grade_parameter(user_val, ideal_val):
    perc = (abs(user_val - ideal_val) / ideal_val) * 100
    if perc <= 10: return "OTTIMO"
    elif perc <= 30: return "BUONO"
    elif perc <= 50: return "SUFFICIENTE"
    else: return "INSUFFICIENTE"

def get_ideal_parameters(material, thickness):
    # Logica semplificata per calcolo ideali
    base_kv = {"Al-2024 (Avional)": 45, "Ti-6Al-4V": 65, "Inconel 718": 95, "Steel 17-4 PH": 85}
    k_kv = {"Al-2024 (Avional)": 3.2, "Ti-6Al-4V": 4.8, "Inconel 718": 7.8, "Steel 17-4 PH": 6.5}
    ideal_kv = int(base_kv[material] + (thickness * k_kv[material]))
    ideal_mas = round(12 * np.exp(thickness * 0.06), 1)
    return ideal_kv, ideal_mas

# --- INTERFACCIA STREAMLIT ---
if 'exam_results' not in st.session_state: st.session_state.exam_results = []
if 'exam_progress' not in st.session_state: st.session_state.exam_progress = 0

mode = st.sidebar.radio("Modalità", ["STUDIO", "ESAME"])

# === MODALITÀ STUDIO ===
if mode == "STUDIO":
    st.title("📘 Studio Avanzato: Difetti Multipli & Orientamento")
    c1, c2 = st.columns([1, 2])
    with c1:
        mat = st.selectbox("Materiale", ["Al-2024 (Avional)", "Ti-6Al-4V", "Inconel 718", "Steel 17-4 PH"])
        thick = st.number_input("Spessore (mm)", 5, 30, 10)
        kv = st.slider("kV", 40, 250, 90)
        ma = st.slider("mA", 1.0, 15.0, 5.0)
        ti = st.slider("Tempo (s)", 1, 120, 30)
        if st.button("GENERA RADIOGRAFIA"):
            img, defs, bboxes = generate_scan_v10_3(kv, ma, ti, mat, thick, exam_mode=True)
            st.session_state.study_img = img
            st.session_state.study_bboxes = bboxes
            st.session_state.study_defs = defs
            st.session_state.study_eval = False

    with c2:
        if 'study_img' in st.session_state:
            l = st.slider("Level", 0, 65535, 32768)
            w = st.slider("Width", 100, 65535, 40000)
            fig, ax = plt.subplots(figsize=(8,8), facecolor='black')
            ax.imshow(st.session_state.study_img, cmap='gray_r', vmin=l-w//2, vmax=l+w//2)
            # Etichette simulate nell'immagine
            ax.text(80, 155, "ISO W10", color='white', alpha=0.5, fontsize=8)
            ax.text(150, 555, "DUPLEX EN", color='white', alpha=0.5, fontsize=8)
            
            if st.session_state.get('study_eval'):
                for b in st.session_state.study_bboxes:
                    rect = patches.Rectangle((b['x'], b['y']), b['w'], b['h'], linewidth=1, edgecolor='red', facecolor='none')
                    ax.add_patch(rect)
                    ax.text(b['x'], b['y']-5, b['t'], color='red', fontsize=8)
            ax.axis('off')
            st.pyplot(fig)
            if st.button("MOSTRA DIFETTI E IDEALI"):
                st.session_state.study_eval = True
                id_kv, id_mas = get_ideal_parameters(mat, thick)
                st.write(f"**Difetti Presenti:** {', '.join(st.session_state.study_defs)}")
                st.write(f"**Parametri Ideali:** {id_kv} kV, {id_mas} mAs")
                st.rerun()

# === MODALITÀ ESAME ===
elif mode == "ESAME":
    st.title("🎓 Esame di Certificazione Livello II")
    if st.session_state.exam_progress == 0:
        if st.button("INIZIA TEST (5 CASI COMPLESSI)"):
            st.session_state.exam_cases = [{"mat": random.choice(["Ti-6Al-4V", "Steel 17-4 PH", "Inconel 718"]), "thick": random.randint(8, 25)} for _ in range(5)]
            st.session_state.exam_progress = 1
            st.rerun()
    
    elif st.session_state.exam_progress > 5:
        st.header("🏁 Risultato Finale")
        st.table(pd.DataFrame(st.session_state.exam_results))
        if st.button("RESET"):
            st.session_state.exam_progress = 0
            st.session_state.exam_results = []
            st.rerun()
            
    else:
        case = st.session_state.exam_cases[st.session_state.exam_progress-1]
        st.subheader(f"Caso {st.session_state.exam_progress} di 5")
        st.info(f"Ispezionare: **{case['mat']}**, spessore **{case['thick']} mm**")
        
        c1, c2 = st.columns([1, 2])
        with c1:
            kv_e = st.slider("kV", 40, 250, 100, key=f"kve{st.session_state.exam_progress}")
            ma_e = st.slider("mA", 1.0, 15.0, 5.0, key=f"mae{st.session_state.exam_progress}")
            ti_e = st.slider("Tempo (s)", 1, 120, 20, key=f"tie{st.session_state.exam_progress}")
            if st.button("SCATTA"):
                img, defs, bboxes = generate_scan_v10_3(kv_e, ma_e, ti_e, case['mat'], case['thick'], exam_mode=True)
                st.session_state.e_img = img
                st.session_state.e_defs = defs

        with c2:
            if 'e_img' in st.session_state:
                l_e = st.slider("L", 0, 65535, 32768)
                w_e = st.slider("W", 100, 65535, 40000)
                fig, ax = plt.subplots(facecolor='black')
                ax.imshow(st.session_state.e_img, cmap='gray_r', vmin=l_e-w_e//2, vmax=l_e+w_e//2)
                ax.axis('off')
                st.pyplot(fig)
                
                selected = st.multiselect("Quali difetti hai trovato?", ["Cricca", "Porosità Singola", "Cluster Porosità", "Inclusione Tungsteno", "Mancata Fusione"])
                if st.button("CONFERMA"):
                    id_kv, id_mas = get_ideal_parameters(case['mat'], case['thick'])
                    # Verifica se il set di difetti coincide
                    correct = set(selected) == set(st.session_state.e_defs)
                    st.session_state.exam_results.append({
                        "Caso": st.session_state.exam_progress,
                        "Voto kV": grade_parameter(kv_e, id_kv),
                        "Voto mAs": grade_parameter(ma_e*ti_e, id_mas),
                        "Diagnosi": "CORRETTO" if correct else "ERRATO",
                        "Realtà": ", ".join(st.session_state.e_defs)
                    })
                    st.session_state.exam_progress += 1
                    del st.session_state.e_img
                    st.rerun()