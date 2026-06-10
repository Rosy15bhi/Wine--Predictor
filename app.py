import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import base64
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# ===================== CONFIGURAZIONE =====================
st.set_page_config(
    page_title="Wine Label Neuro Predictor",
    page_icon="🍷",
    layout="wide"
)

# ===================== CARICAMENTO E TRAINING MODELLO =====================
@st.cache_resource
def load_and_train_model():
    df = pd.read_excel('catalogo_completo.xlsm', engine='openpyxl')
    features = ['eleganza', 'completezza', 'visibilità', 'coerenza', 'design', 'attrattività per giovani']
    target = 'indice neuro'
    X = df[features]
    y = df[target]
    model = Ridge(alpha=1.0)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    cv_mae = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
    model.fit(X, y)
    metrics = {
        'r2_mean': cv_scores.mean(),
        'r2_std': cv_scores.std(),
        'mae_mean': (-cv_mae).mean(),
        'mae_std': cv_mae.std(),
        'n_samples': len(df),
        'features': features,
        'coef': model.coef_.tolist()
    }
    return model, metrics, df

# ===================== ANALISI IMMAGINE CON CLAUDE =====================
def analyze_image_with_claude(image_bytes, image_type, api_key):
    """Manda l'immagine a Claude e ottiene i 6 punteggi + descrizioni."""
    
    image_b64 = base64.standard_b64encode(image_bytes).decode('utf-8')
    
    prompt = """Sei un esperto di neuromarketing applicato al settore vitivinicolo, specializzato nell'analisi estetica delle etichette di vino secondo i criteri di neuroengagement IULM.

Analizza questa etichetta di vino e assegna un punteggio da 1 a 10 per ciascuno dei seguenti criteri, poi genera una descrizione professionale in italiano per ognuno.

Criteri da valutare:
1. ELEGANZA: raffinatezza visiva, uso dei colori, tipografia, qualità estetica complessiva
2. COMPLETEZZA: presenza e chiarezza delle informazioni tecniche (denominazione, annata, gradazione, produttore)
3. VISIBILITA': capacità di emergere sullo scaffale, contrasto, leggibilità a distanza
4. COERENZA: coerenza cromatica e stilistica con la tipologia del vino
5. DESIGN: qualità grafica complessiva, layout, originalità degli elementi visivi
6. ATTRATTIVITA' PER GIOVANI: appeal verso un pubblico giovane e contemporaneo

Rispondi SOLO con un JSON valido in questo formato esatto, senza markdown o altro testo:
{
  "brand": "nome del brand se leggibile, altrimenti stringa vuota",
  "nome": "nome del vino se leggibile, altrimenti stringa vuota",
  "tipo": "tipologia del vino se leggibile (es. Rosso, Bianco, Rosato, Spumante), altrimenti stringa vuota",
  "punteggi": {
    "eleganza": 8.5,
    "completezza": 7.0,
    "visibilità": 9.0,
    "coerenza": 8.0,
    "design": 8.5,
    "attrattività per giovani": 7.5
  },
  "descrizioni": {
    "generale": "descrizione generale dell'etichetta in 2-3 frasi",
    "eleganza": "descrizione del criterio eleganza in 1-2 frasi",
    "completezza": "descrizione del criterio completezza in 1-2 frasi",
    "visibilità": "descrizione del criterio visibilità in 1-2 frasi",
    "coerenza": "descrizione del criterio coerenza in 1-2 frasi",
    "design": "descrizione del criterio design in 1-2 frasi",
    "attrattività": "descrizione del criterio attrattività per giovani in 1-2 frasi"
  }
}"""

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json"},
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1500,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": image_type,
                                    "data": image_b64
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            },
            timeout=60
        )
        
        if response.status_code == 200:
            content = response.json()['content'][0]['text']
            content = content.strip()
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
            return json.loads(content.strip())
        else:
            st.error(f"Errore API: {response.status_code} — {response.text}")
            return None
            
    except Exception as e:
        st.error(f"Errore durante l'analisi: {e}")
        return None

# ===================== INTERFACCIA =====================
st.title("🍷 Wine Label Neuro Predictor")
st.markdown("**Carica la foto di un'etichetta — il sistema la analizza e predice l'indice neuro automaticamente**")

# Carica modello
with st.spinner("Caricamento modello predittivo..."):
    try:
        model, metrics, df = load_and_train_model()
        model_loaded = True
    except Exception as e:
        st.error(f"Errore nel caricamento del dataset: {e}")
        st.info("Assicurati che il file 'catalogo_completo.xlsm' sia nella stessa cartella dell'app.")
        model_loaded = False

if model_loaded:
    
    # Sidebar con info modello
    with st.sidebar:
        st.header("📊 Modello Predittivo")
        st.metric("Algoritmo", "Ridge Regression")
        st.metric("R² (5-fold CV)", f"{metrics['r2_mean']:.4f} ± {metrics['r2_std']:.4f}")
        st.metric("MAE (5-fold CV)", f"{metrics['mae_mean']:.4f} ± {metrics['mae_std']:.4f}")
        st.metric("Etichette in training", metrics['n_samples'])
        
        st.markdown("---")
        st.markdown("**Peso dei criteri nel modello:**")
        features = metrics['features']
        coefs = metrics['coef']
        for feat, coef in zip(features, coefs):
            st.markdown(f"- {feat}: `{coef:.3f}`")
        
        st.markdown("---")
        st.markdown("**Come funziona:**")
        st.markdown("1. Carica la foto dell'etichetta")
        st.markdown("2. Claude analizza l'immagine")
        st.markdown("3. Assegna i 6 punteggi automaticamente")
        st.markdown("4. Il modello predice l'indice neuro")
        st.markdown("5. Vengono generate le descrizioni")
    
    # Check API key
    api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        st.warning("⚠️ Chiave API non configurata. Vai su Settings → Secrets e aggiungi ANTHROPIC_API_KEY.")
        st.stop()
    
    # Upload immagine
    st.markdown("---")
    uploaded_file = st.file_uploader(
        "📸 Carica la foto dell'etichetta",
        type=["jpg", "jpeg", "png", "webp"],
        help="Carica una foto chiara dell'etichetta fronte del vino"
    )
    
    if uploaded_file:
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            st.markdown("#### Immagine caricata")
            st.image(uploaded_file, use_container_width=True)
            
            analizza = st.button("🔍 Analizza Etichetta", type="primary", use_container_width=True)
        
        with col2:
            if analizza:
                # Leggi bytes e tipo immagine
                image_bytes = uploaded_file.read()
                ext = uploaded_file.name.split('.')[-1].lower()
                type_map = {'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'png': 'image/png', 'webp': 'image/webp'}
                image_type = type_map.get(ext, 'image/jpeg')
                
                with st.spinner("🤖 Claude sta analizzando l'etichetta..."):
                    risultato = analyze_image_with_claude(image_bytes, image_type, api_key)
                
                if risultato:
                    punteggi = risultato.get('punteggi', {})
                    descrizioni = risultato.get('descrizioni', {})
                    brand = risultato.get('brand', '')
                    nome = risultato.get('nome', '')
                    tipo = risultato.get('tipo', '')
                    
                    # Predizione indice neuro
                    features_list = ['eleganza', 'completezza', 'visibilità', 'coerenza', 'design', 'attrattività per giovani']
                    X_input = pd.DataFrame([[punteggi.get(f, 8.0) for f in features_list]], columns=features_list)
                    indice_predetto = model.predict(X_input)[0]
                    indice_predetto = np.clip(indice_predetto, 1.0, 10.0)
                    
                    # Info vino rilevate
                    if brand or nome or tipo:
                        st.markdown("#### 🍾 Vino identificato")
                        if brand:
                            st.markdown(f"**Brand:** {brand}")
                        if nome:
                            st.markdown(f"**Nome:** {nome}")
                        if tipo:
                            st.markdown(f"**Tipologia:** {tipo}")
                    
                    st.markdown("---")
                    
                    # Risultato principale
                    if indice_predetto >= 9.0:
                        emoji = "🟢"
                        valutazione = "Eccellente"
                    elif indice_predetto >= 8.0:
                        emoji = "🟡"
                        valutazione = "Molto buono"
                    elif indice_predetto >= 7.0:
                        emoji = "🟠"
                        valutazione = "Buono"
                    else:
                        emoji = "🔴"
                        valutazione = "Da migliorare"
                    
                    st.metric(
                        label=f"{emoji} Indice Neuro Predetto",
                        value=f"{indice_predetto:.2f} / 10",
                        delta=valutazione
                    )
                    
                    # Punteggi per criterio
                    st.markdown("#### 📊 Punteggi per criterio")
                    punteggi_df = pd.DataFrame({
                        'Criterio': list(punteggi.keys()),
                        'Punteggio': list(punteggi.values())
                    })
                    st.bar_chart(punteggi_df.set_index('Criterio'))
                    
                    # Descrizione generale
                    st.markdown("---")
                    st.markdown("#### 📝 Analisi dell'etichetta")
                    st.info(descrizioni.get('generale', ''))
                    
                    # Descrizioni per criterio
                    with st.expander("📋 Dettaglio per criterio"):
                        for criterio, label in [
                            ('eleganza', 'Eleganza'),
                            ('completezza', 'Completezza'),
                            ('visibilità', 'Visibilità'),
                            ('coerenza', 'Coerenza'),
                            ('design', 'Design'),
                            ('attrattività', 'Attrattività per giovani')
                        ]:
                            punteggio_key = criterio if criterio != 'attrattività' else 'attrattività per giovani'
                            valore = punteggi.get(punteggio_key, 'N/D')
                            desc = descrizioni.get(criterio, 'N/D')
                            st.markdown(f"**{label} ({valore}/10):** {desc}")
                    
                    # Vini simili
                    st.markdown("---")
                    st.markdown("#### 🔍 Vini simili nel dataset")
                    df_temp = df.copy()
                    df_temp['distanza'] = np.sqrt(sum([
                        (df_temp[feat] - punteggi.get(feat, 8.0))**2
                        for feat in features_list
                    ]))
                    simili = df_temp.nsmallest(3, 'distanza')[['brand', 'nome', 'tipo', 'indice neuro']]
                    st.dataframe(simili, use_container_width=True, hide_index=True)
            
            else:
                st.info("👈 Clicca **Analizza Etichetta** per avviare l'analisi automatica.")
    
    else:
        st.markdown("---")
        st.markdown("### Come funziona")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("**📸 1. Carica**")
            st.markdown("Carica la foto dell'etichetta del vino")
        with col2:
            st.markdown("**🤖 2. Analisi**")
            st.markdown("Claude analizza visivamente l'etichetta")
        with col3:
            st.markdown("**🎯 3. Punteggi**")
            st.markdown("Assegna automaticamente i 6 criteri IULM")
        with col4:
            st.markdown("**📈 4. Predizione**")
            st.markdown("Il modello calcola l'indice neuro")
        
        st.markdown("---")
        st.markdown("#### 📊 Statistiche del dataset di riferimento")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Etichette analizzate", metrics['n_samples'])
        col2.metric("Indice minimo", "5.67")
        col3.metric("Indice massimo", "9.80")
        col4.metric("Media", f"{df['indice neuro'].mean():.2f}")
