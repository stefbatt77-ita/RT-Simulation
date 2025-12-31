import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter
import random
import pandas as pd

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Aero-NDT Ultimate v10.9", layout="wide", page_icon="✈️")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #e0e0e0; }
    h1, h2, h3 { color: #ffffff; }
    div.stButton > button { font-weight: bold; border-radius: 6px; height: 3em; }
    div.stButton > button:first-child { background-color: #d32f2f; color: white; border: none; }
    .stTable { background-color: #1e1e1e; }
    </style>
    """, unsafe_allow_html=True)

# --- INIZIALIZZAZIONE STATO ---
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

# --- MOTORE FISICO ---
def get_ideal_params(material, thickness):
    props = {
        "Al-2024 (Avional)": {"b": 45, "k": 3.2},
        "Ti-6Al-4V":         {"b": 65, "k": 4.8},
        "Inconel 718":       {"b": 95, "k": 7.8},
        "Steel 17-4 PH":     {"b": 85, "k": 6.5}
    }
    p = props[material]
    id_kv = int(p["b"] + (thickness * p["k"]))
    id_kv = max(40, min(250, id_kv))
    id_mas = round(12 * np.exp(thickness * 0.06), 1)
    return id_kv, id_mas

def generate_scan_final(kv, ma, time, material, thickness, selected_defect="Casuale (Multiplo)"):
    size = 600
    mu_map = {"Al-2024 (Avional)": 0.02, "Ti-6Al-4V": 0.045, "Inconel 718": 0.09, "Steel 17-4 PH": 0.075}
    safe_kv = max(10, kv)
    mu = mu_map[material] * (120/safe_kv)**1.5
    
    m_sp = np.full((size, size), float(thickness), dtype=float)
    
    # IQI
    for i in range(7):
        m_sp[40:140, 80 + i*40 : 80 + i*40 + 2] += (0.4 - i*0.05)
    for i in range(13):
        m_sp[500:540, 150 + i*25 : 150 + i*25 + 2] += (0.7 / (i+1))

    # Etichette
    m_sp[145:155, 80:120] -= 0.5
    m_sp[545:555, 150:200] -= 0.5

    # DIFETTI
    defects_list = []
    bboxes = []
    
    to_generate = []
    # Generazione sempre randomica se "Casuale (Multiplo)"
    if selected_defect == "Casuale (Multiplo)":
        num = random.randint(1, 3)
        possibles = ["Cricca", "Porosità Singola", "Cluster Porosità", "Inclusione Tungsteno", "Mancata Fusione"]
        to_generate = random.sample(possibles, num)
    elif selected_defect == "Nessun Difetto":
        to_generate = []
    else:
        to_generate = [selected_defect]

    for d_type in to_generate:
        rx, ry = random.randint(100, 500), random.randint(180, 450)
        
        if d_type == "Cricca":
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
            mask = (x - rx)**2 + (y - ry)**2 <= 6**2
            m_sp[mask] -= 2.5
            bboxes.append({"x": rx-10, "y": ry-10, "w": 20, "h": 20, "t": "Porosità"})

        elif d_type == "Cluster Porosità":
            min_x, max_x, min_y, max_y = 600, 0, 600, 0
            for _ in range(8):
                cx, cy = rx + random.randint(-25, 25), ry + random.randint(-25, 25)
                y, x = np.ogrid[:size, :size]
                mask = (x - cx)**2 + (y - cy)**2 <= 3**2
                m_sp[mask] -= 1.8
                min_x = min(min_x, cx); max_x = max(max_x, cx)
                min_y = min(min_y, cy); max_y = max(max_y, cy)
            bboxes.append({"x": min_x-10, "y": min_y-10, "w": max_x-min_x+20, "h": max_y-min_y+20, "t": "Cluster"})

        elif d_type == "Inclusione Tungsteno":
            y, x = np.ogrid[:size, :size]
            mask = (x - rx)**2 + (y - ry)**2 <= 4**2
            m_sp[mask] += 15.0
            bboxes.append({"x": rx-8, "y": ry-8, "w": 16, "h": 16, "t": "Tungsteno"})

        elif d_type == "Mancata Fusione":
            h = random.randint(60, 150)
            m_sp[ry:ry+h, rx:rx+3] -= 1.6
            bboxes.append({"x": rx-5, "y": ry, "w": 15, "h": h, "t": "LoF"})
            
        defects_list.append(d_type)

    dose = (kv**2) * ma * time * 0.05
    signal = dose * np.exp(-mu * m_sp)
    signal = gaussian_filter(signal, sigma=1.1)
    noise = np.random.normal(0, np.sqrt(signal + 1) * 2.2, (size, size))
    return np.clip(signal + noise, 0, 65535).astype(np.uint16), defects_list, bboxes

def grade_param(u, i):
    if i == 0: return "N/A"
    diff = abs(u - i)
    perc = (diff / i) * 100
    if perc <= 10: return "OTTIMO"
    if perc <= 30: return "BUONO"
    if perc <= 50: return "SUFFICIENTE"
    return "INSUFFICIENTE"

# --- SIDEBAR ---
with st.sidebar:
    st.title("🎛️ Pannello NDT")
    new_mode = st.radio("Seleziona Modalità", ["STUDIO (Training)", "ESAME (Certificazione)"])
    if new_mode != st.session_state['mode']:
        st.session_state['mode'] = new_mode
        st.session_state['exam_progress'] = 0
        st.session_state['exam_results'] = []
        st.session_state['s_img'] = None
        st.session_state['e_img'] = None
        st.rerun()

# === MODALITÀ STUDIO ===
if st.session_state['mode'] == "STUDIO (Training)":
    st.title("📘 Ambiente di Studio Avanzato")
    st.info("I difetti sono generati casualmente. Regola i parametri per trovarli, poi clicca Verifica per vedere la soluzione.")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Parametri Scansione")
        mat = st.selectbox("Materiale", ["Al-2024 (Avional)", "Ti-6Al-4V", "Inconel 718", "Steel 17-4 PH"])
        thick = st.number_input("Spessore (mm)", 5, 30, 10)
        # NOTA: Rimossa la scelta manuale del difetto per forzare il random
        
        st.divider()
        kv = st.slider("kV", 40, 250, 90)
        ma = st.slider("mA", 1.0, 15.0, 5.0)
        ti = st.slider("Tempo (s)", 1, 120, 25)
        
        if st.button("ACQUISICI SCANSIONE (DIFETTI RANDOM)"):
            # Forza la generazione casuale multipla
            img, defs, bboxes = generate_scan_final(kv, ma, ti, mat, thick, "Casuale (Multiplo)")
            st.session_state.s_img = img
            st.session_state.s_bboxes = bboxes
            st.session_state.s_defs = defs
            st.session_state.s_eval = False

    with c2:
        if st.session_state.s_img is not None:
            c_l, c_w = st.columns(2)
            lev = c_l.slider("Livello (L)", 0, 65535, 32768)
            wid = c_w.slider("Finestra (W)", 100, 65535, 40000)
            
            fig, ax = plt.subplots(figsize=(8,8), facecolor='black')
            vmin, vmax = lev - wid//2, lev + wid//2
            ax.imshow(st.session_state.s_img, cmap='gray_r', vmin=vmin, vmax=vmax)
            ax.text(85, 135, "ISO W10", color='white', alpha=0.5, fontsize=8)
            ax.text(155, 535, "DUPLEX IQI", color='white', alpha=0.5, fontsize=8)

            if st.session_state.s_eval:
                for b in st.session_state.s_bboxes:
                    rect = patches.Rectangle((b['x'], b['y']), b['w'], b['h'], linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
                    ax.add_patch(rect)
                    ax.text(b['x'], b['y']-5, b['t'], color='red', fontsize=10, fontweight='bold')
            ax.axis('off')
            st.pyplot(fig)
            
            st.divider()
            # Input per autovalutazione prima di vedere la soluzione
            st.multiselect("Fai la tua diagnosi (Opzionale):", ["Cricca", "Porosità Singola", "Cluster Porosità", "Inclusione Tungsteno", "Mancata Fusione"], key="study_guess")
            
            if st.button("VERIFICA E MOSTRA SOLUZIONE"):
                st.session_state.s_eval = True
                st.rerun()

            if st.session_state.s_eval:
                id_k, id_m = get_ideal_params(mat, thick)
                st.divider()
                st.subheader("📊 Analisi Tecnica e Soluzione")
                
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.write("#### Parametri Tecnici")
                    res_data = {
                        "Parametro": ["Tensione (kV)", "Esposizione (mAs)"],
                        "Tuo Valore": [f"{kv}", f"{ma*ti:.1f}"],
                        "Ideale (Calc)": [f"{id_k}", f"{id_m}"],
                        "Valutazione": [grade_param(kv, id_k), grade_param(ma*ti, id_m)]
                    }
                    st.table(pd.DataFrame(res_data))
                
                with col_res2:
                    st.write("#### Difetti Rilevati")
                    if st.session_state.s_defs:
                        for d in st.session_state.s_defs:
                            st.error(f"⚠️ {d}")
                    else:
                        st.success("✅ Nessun Difetto (Pezzo Sano)")

# === MODALITÀ ESAME ===
elif st.session_state['mode'] == "ESAME (Certificazione)":
    st.title("🎓 Esame di Certificazione EN4179")
    
    if st.session_state.exam_progress == 0:
        if st.button("INIZIA ESAME (5 Casi)"):
            st.session_state.exam_cases = [{"mat": random.choice(["Ti-6Al-4V", "Steel 17-4 PH", "Inconel 718"]), "thick": random.randint(8, 25), "defect": "Casuale (Multiplo)"} for _ in range(5)]
            st.session_state.exam_progress = 1
            st.rerun()

    elif st.session_state.exam_progress > 5:
        st.header("🏁 Risultati Finali")
        if st.session_state.exam_results:
            df = pd.DataFrame(st.session_state.exam_results)
            st.table(df)
            corr = df[df["Diagnosi"] == "CORRETTO"].shape[0]
            st.metric("Punteggio Diagnostico", f"{corr} / 5")
        if st.button("NUOVA SESSIONE"):
            st.session_state.exam_progress = 0
            st.session_state.exam_results = []
            st.rerun()

    else:
        idx = st.session_state.exam_progress
        case = st.session_state.exam_cases[idx-1]
        st.subheader(f"Caso {idx} di 5")
        st.info(f"Specimen: **{case['mat']}**, Spessore: **{case['thick']} mm**")
        
        c1, c2 = st.columns([1, 2])
        with c1:
            kve = st.slider("kV", 40, 250, 100, key=f"ke{idx}")
            mae = st.slider("mA", 1.0, 15.0, 5.0, key=f"me{idx}")
            tie = st.slider("Tempo (s)", 1, 120, 20, key=f"te{idx}")
            if st.button("SCATTA"):
                img, defs, bboxes = generate_scan_final(kve, mae, tie, case['mat'], case['thick'], case['defect'])
                st.session_state.e_img = img
                st.session_state.e_defs = defs
        
        with c2:
            if st.session_state.get('e_img') is not None:
                le, we = st.slider("L", 0, 65535, 32768, key=f"le{idx}"), st.slider("W", 100, 65535, 40000, key=f"we{idx}")
                fig, ax = plt.subplots(facecolor='black')
                ax.imshow(st.session_state.e_img, cmap='gray_r', vmin=le-we//2, vmax=le+we//2)
                ax.axis('off')
                st.pyplot(fig)
                sel_defs = st.multiselect("Difetti Rilevati:", ["Cricca", "Porosità Singola", "Cluster Porosità", "Inclusione Tungsteno", "Mancata Fusione"], key=f"sel{idx}")
                if st.button("CONFERMA E PROCEDI"):
                    id_k, id_m = get_ideal_params(case['mat'], case['thick'])
                    correct = set(sel_defs) == set(st.session_state.e_defs)
                    st.session_state.exam_results.append({
                        "Caso": idx,
                        "Materiale": case['mat'],
                        "kV Voto": grade_param(kve, id_k),
                        "mAs Voto": grade_param(mae*tie, id_m),
                        "Diagnosi": "CORRETTO" if correct else "ERRATO",
                        "Realtà": ", ".join(st.session_state.e_defs) if st.session_state.e_defs else "Sano"
                    })
                    st.session_state.exam_progress += 1
                    st.session_state.e_img = None
                    st.rerun()