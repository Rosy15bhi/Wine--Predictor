import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import requests
import json
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ===================== CONFIGURAZIONE =====================
st.set_page_config(
    page_title="Wine Label Predictor",
    page_icon="🍷",
    layout="wide"
)

# ===================== CARICAMENTO E TRAINING MODELLO =====================
@st.cache_resource
def load_and_train_model():
    """Carica il dataset e addestra il modello Ridge Regression."""
    df = pd.read_excel('catalogo_completo.xlsm', engine='openpyxl')
    
    features = ['eleganza', 'completezza', 'visibilità', 'coerenza', 'design', 'attrattività per giovani']
    target = 'indice neuro'
    
    X = df[features]
    y = df[target]
    
    # Ridge Regression - miglior bilanciamento tra performance e interpretabilità
    model = Ridge(alpha=1.0)
    
    # Cross-validation per valutazione
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    cv_mae = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
    
    # Training finale su tutto il dataset
    model.fit(X, y)
    
    metrics = {
        'r2_mean': cv_scores.mean(),
        'r2_std': cv_scores.std(),
        'mae_mean': (-cv_mae).mean(),
        'mae_std': cv_mae.std(),
        'n_samples': len(df),
        'features': features
    }
    
    return model, metrics, df

# ===================== GENERAZIONE DESCRIZIONI VIA API =====================
def generate_descriptions(brand, nome, tipo, punteggi, indice_predetto):
    """Genera descrizioni testuali usando l'API di Claude."""
    
    api_key = st.secrets.get("ANTHROPIC_API_KEY", os.environ.get("ANTHROPIC_API_KEY", ""))
    
    if not api_key:
        return None
    
    prompt = f"""Sei un esperto di neuromarketing applicato al settore vitivinicolo, specializzato nell'analisi estetica delle etichette di vino secondo i criteri di neuroengagement IULM.

Devi generare descrizioni professionali per un'etichetta di vino con i seguenti dati:

Vino: {nome} - {brand}
Tipologia: {tipo}
Indice Neuro predetto: {indice_predetto:.2f}/10

Punteggi per criterio:
- Eleganza: {punteggi['eleganza']}/10
- Completezza informativa: {punteggi['completezza']}/10
- Visibilità: {punteggi['visibilità']}/10
- Coerenza: {punteggi['coerenza']}/10
- Design: {punteggi['design']}/10
- Attrattività per giovani: {punteggi['attrattività per giovani']}/10

Genera 7 descrizioni distinte, una per ciascun criterio più una generale. Ogni descrizione deve essere in italiano, professionale, di 1-2 frasi, coerente con il punteggio assegnato (punteggi alti = descrizioni positive, punteggi bassi = descrizioni critiche ma costruttive).

Rispondi SOLO con un JSON valido in questo formato esatto, senza markdown o altro testo:
{{
  "generale": "...",
  "eleganza": "...",
  "completezza": "...",
  "visibilità": "...",
  "coerenza": "...",
  "design": "...",
  "attrattività": "..."
}}"""

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json"},
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1000,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=30
        )
        
        if response.status_code == 200:
            content = response.json()['content'][0]['text']
            # Pulizia del JSON
            content = content.strip()
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
            return json.loads(content)
    except Exception as e:
        st.error(f"Errore generazione descrizioni: {e}")
    
    return None

# ===================== INTERFACCIA UTENTE =====================
st.title("🍷 Wine Label Neuro Predictor")
st.markdown("**Sistema predittivo per l'indice neuro di etichette vinicole** — basato su criteri di neuroengagement IULM")

# Carica modello
with st.spinner("Caricamento modello..."):
    try:
        model, metrics, df = load_and_train_model()
        model_loaded = True
    except Exception as e:
        st.error(f"Errore nel caricamento del dataset: {e}")
        st.info("Assicurati che il file 'catalogo_completo.xlsm' sia nella stessa cartella dell'app.")
        model_loaded = False

if model_loaded:
    # Metriche modello nella sidebar
    with st.sidebar:
        st.header("📊 Informazioni Modello")
        st.metric("Algoritmo", "Ridge Regression")
        st.metric("R² (5-fold CV)", f"{metrics['r2_mean']:.4f} ± {metrics['r2_std']:.4f}")
        st.metric("MAE (5-fold CV)", f"{metrics['mae_mean']:.4f} ± {metrics['mae_std']:.4f}")
        st.metric("Campioni training", metrics['n_samples'])
        
        st.markdown("---")
        st.markdown("**Interpretazione R²:**")
        st.markdown(f"Il modello spiega il **{metrics['r2_mean']*100:.1f}%** della varianza dell'indice neuro.")
        
        st.markdown("---")
        st.markdown("**Coefficienti del modello:**")
        for feat, coef in zip(metrics['features'], model.coef_):
            st.markdown(f"- {feat}: `{coef:.4f}`")
        
        st.markdown("---")
        st.markdown("**Dataset:**")
        st.markdown(f"- {metrics['n_samples']} etichette analizzate")
        st.markdown(f"- Range indice neuro: 5.67 — 9.80")
        st.markdown(f"- Media: 8.75")

    # Layout principale
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📝 Inserisci i dati dell'etichetta")
        
        brand = st.text_input("Brand / Cantina", placeholder="es. Cantele")
        nome = st.text_input("Nome del vino", placeholder="es. Teresa Manara")
        tipo = st.selectbox("Tipologia", [
            "Rosso", "Bianco", "Rosato", "Spumante", 
            "Spumante Metodo Classico DOC", "Prosecco",
            "Passito", "Lambrusco", "Altro"
        ])
        
        st.markdown("---")
        st.markdown("### 🎯 Punteggi per criterio (1-10)")
        
        eleganza = st.slider("Eleganza", 1.0, 10.0, 8.0, 0.1,
            help="Raffinatezza visiva, uso dei colori, tipografia")
        completezza = st.slider("Completezza informativa", 1.0, 10.0, 8.0, 0.1,
            help="Presenza e chiarezza delle informazioni tecniche")
        visibilita = st.slider("Visibilità", 1.0, 10.0, 8.0, 0.1,
            help="Capacità di emergere sullo scaffale, contrasto")
        coerenza = st.slider("Coerenza", 1.0, 10.0, 8.0, 0.1,
            help="Coerenza cromatica e stilistica con la tipologia del vino")
        design = st.slider("Design", 1.0, 10.0, 8.0, 0.1,
            help="Qualità grafica complessiva, layout, elementi visivi")
        attrattivita = st.slider("Attrattività per giovani", 1.0, 10.0, 8.0, 0.1,
            help="Appeal verso un pubblico giovane e contemporaneo")
        
        predici = st.button("🔮 Predici Indice Neuro", type="primary", use_container_width=True)

    with col2:
        st.header("📈 Risultato della Predizione")
        
        if predici:
            punteggi = {
                'eleganza': eleganza,
                'completezza': completezza,
                'visibilità': visibilita,
                'coerenza': coerenza,
                'design': design,
                'attrattività per giovani': attrattivita
            }
            
            # Predizione
            X_input = pd.DataFrame([punteggi])
            indice_predetto = model.predict(X_input)[0]
            indice_predetto = np.clip(indice_predetto, 1.0, 10.0)
            
            # Visualizzazione risultato
            if indice_predetto >= 9.0:
                color = "🟢"
                valutazione = "Eccellente"
            elif indice_predetto >= 8.0:
                color = "🟡"
                valutazione = "Molto buono"
            elif indice_predetto >= 7.0:
                color = "🟠"
                valutazione = "Buono"
            else:
                color = "🔴"
                valutazione = "Da migliorare"
            
            st.metric(
                label=f"{color} Indice Neuro Predetto",
                value=f"{indice_predetto:.2f} / 10",
                delta=f"{valutazione}"
            )
            
            # Grafico radar dei punteggi
            st.markdown("#### Profilo dell'etichetta")
            
            # Barre per ogni criterio
            criteri_df = pd.DataFrame({
                'Criterio': list(punteggi.keys()),
                'Punteggio': list(punteggi.values())
            })
            st.bar_chart(criteri_df.set_index('Criterio'))
            
            # Media dei punteggi
            media = np.mean(list(punteggi.values()))
            st.markdown(f"**Media punteggi:** {media:.2f}/10")
            st.markdown(f"**Indice Neuro predetto:** {indice_predetto:.2f}/10")
            
            # Vini simili nel dataset
            st.markdown("---")
            st.markdown("#### 🔍 Vini simili nel dataset")
            
            df_temp = df.copy()
            df_temp['distanza'] = np.sqrt(sum([
                (df_temp[feat] - val)**2 
                for feat, val in punteggi.items()
            ]))
            simili = df_temp.nsmallest(3, 'distanza')[['brand', 'nome', 'tipo', 'indice neuro']]
            st.dataframe(simili, use_container_width=True, hide_index=True)
            
            # Generazione descrizioni
            st.markdown("---")
            st.markdown("#### 📝 Descrizioni generate automaticamente")
            
            if brand and nome:
                with st.spinner("Generazione descrizioni in corso..."):
                    descrizioni = generate_descriptions(brand, nome, tipo, punteggi, indice_predetto)
                
                if descrizioni:
                    st.success("✅ Descrizioni generate con successo!")
                    
                    st.markdown(f"**Descrizione generale:** {descrizioni.get('generale', 'N/D')}")
                    
                    with st.expander("📋 Descrizioni per criterio"):
                        st.markdown(f"**Eleganza:** {descrizioni.get('eleganza', 'N/D')}")
                        st.markdown(f"**Completezza:** {descrizioni.get('completezza', 'N/D')}")
                        st.markdown(f"**Visibilità:** {descrizioni.get('visibilità', 'N/D')}")
                        st.markdown(f"**Coerenza:** {descrizioni.get('coerenza', 'N/D')}")
                        st.markdown(f"**Design:** {descrizioni.get('design', 'N/D')}")
                        st.markdown(f"**Attrattività giovani:** {descrizioni.get('attrattività', 'N/D')}")
                else:
                    st.info("💡 Per generare descrizioni automatiche, configura la chiave API di Anthropic nel file `.streamlit/secrets.toml`")
            else:
                st.info("💡 Inserisci brand e nome del vino per generare le descrizioni automatiche.")
        
        else:
            st.info("👈 Inserisci i punteggi e clicca **Predici Indice Neuro** per ottenere il risultato.")
            
            # Statistiche dataset
            st.markdown("---")
            st.markdown("#### 📊 Distribuzione nel dataset")
            st.markdown(f"- **Vini analizzati:** {len(df)}")
            st.markdown(f"- **Indice minimo:** {df['indice neuro'].min():.2f}")
            st.markdown(f"- **Indice massimo:** {df['indice neuro'].max():.2f}")
            st.markdown(f"- **Media:** {df['indice neuro'].mean():.2f}")
            st.markdown(f"- **Deviazione standard:** {df['indice neuro'].std():.2f}")
