import streamlit as st
import streamlit.components.v1 as components
import openai
import duckdb
import pandas as pd
import os
import numpy as np
import json
import re
import unicodedata

# --- 1. CONFIGURATION & STYLE ---
st.set_page_config(
    page_title="Terribot | Assistant Territorial",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stChatInput {padding-bottom: 20px;}
    .stDataFrame {border: 1px solid #f0f2f6; border-radius: 5px;}
    
    /* Style pour les étapes de raisonnement */
    .reasoning-step {
        font-size: 0.85em;
        color: #555;
        border-left: 3px solid #FF4B4B;
        padding-left: 10px;
        margin-bottom: 10px;
    }
    
    /* Bouton reset custom */
    div.stButton > button:first-child {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. ANIMATION ---
def inject_placeholder_animation():
    components.html("""
    <script>
        const questions = [
            "Quelle est la structure par âge à Saint-Malo ?",
            "Le chômage baisse-t-il à Bordeaux ?",
            "Quelle est la part des cadres à Lyon ?",
            "Toutes les communes de la Seine-Saint-Denis",
            "Niveau de vie médian à Biarritz ?",
            "Tous les EPCI de la région Bretagne",
            "Densité de population à Paris ?",
            "Combien de résidences secondaires à La Baule ?",
            "Les jeunes partent-ils de Charleville-Mézières ?",
            "Quel est le taux de pauvreté à Roubaix ?"
        ];
        let idx = 0;
        function cyclePlaceholder() {
            const textArea = window.parent.document.querySelector('textarea[data-testid="stChatInputTextArea"]');
            if (textArea) {
                if (!window.parent.document.getElementById('placeholder-anim')) {
                    const style = window.parent.document.createElement('style');
                    style.id = 'placeholder-anim';
                    style.innerHTML = `
                        textarea[data-testid="stChatInputTextArea"]::placeholder {
                            transition: opacity 0.5s ease-in-out;
                            opacity: 1;
                        }
                        textarea[data-testid="stChatInputTextArea"].fade-out::placeholder {
                            opacity: 0;
                        }
                    `;
                    window.parent.document.head.appendChild(style);
                }
                textArea.classList.add('fade-out');
                setTimeout(() => {
                    textArea.setAttribute('placeholder', questions[idx]);
                    idx = (idx + 1) % questions.length;
                    textArea.classList.remove('fade-out');
                }, 500);
            }
        }
        setInterval(cyclePlaceholder, 4000);
        setTimeout(cyclePlaceholder, 100);
    </script>
    """, height=0)

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("🤖 Terribot")
    st.caption("v0.13 - 12 janvier 2026")
    st.divider()
    
    # Bouton Reset
    if st.button("🗑️ Nouvelle conversation", type="secondary", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_geo_context = None
        st.session_state.pending_prompt = None
        st.session_state.ambiguity_candidates = None
        st.rerun()
        st.session_state.messages = [{"role": "assistant", "content": "Bonjour ! Quel territoire souhaitez-vous analyser ?"}]

    st.divider()
    
    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
        st.success("🔒 API Connectée")
    else:
        api_key = st.text_input("Clé API OpenAI", type="password", placeholder="sk-...")
        if not api_key:
            st.warning("Requis pour démarrer.")
            st.stop()

    st.divider()
    with st.expander("📚 Sources de données"):
        st.markdown("""
        - **INSEE** : Recensement (Pop, Logement, Emploi)
        - **RPLS** : Logement social
        - **Philosofi** : Revenus & Pauvreté
        - **Sirene** : Entreprises
        """)
        
    st.info("💡 **Astuce :** L'IA choisit elle-même la variable du graphique selon votre question.")

client = openai.OpenAI(api_key=api_key)
MODEL_NAME = "gpt-5.2-2025-12-11"  # Mis à jour vers un modèle standard valide, ajustez si nécessaire
EMBEDDING_MODEL = "text-embedding-3-small"

# --- 4. FONCTIONS INTELLIGENTES (FORMATAGE & SÉLECTION) ---
def get_chart_configuration(df: pd.DataFrame, question: str, glossaire_context: str, client, model: str):
    """
    Fusionne la sélection des variables (1 ou plusieurs) et la détection des formats.
    """
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    # Nettoyage des colonnes techniques
    numeric_cols = [c for c in numeric_cols if c.upper() not in ["AN", "ANNEE", "YEAR", "ID", "CODGEO"]]
    
    if not numeric_cols: return {"selected_columns": [], "formats": {}}

    # Stats légères pour aider l'IA
    stats = {}
    for c in numeric_cols[:10]: # On limite pour ne pas exploser le contexte
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(s) > 0:
            stats[c] = {"min": float(s.min()), "max": float(s.max())}

    payload = {
        "question": question,
        "available_columns": numeric_cols,
        "data_stats": stats,
        "glossaire_sample": (glossaire_context or "")[-2000:],
    }

    system_prompt = """
    Tu es un expert Dataviz. Configure le graphique pour répondre à la question.
    
    TA MISSION :
    1. Choisis les colonnes à afficher ('selected_columns'). 
       - Si la question implique une comparaison de plusieurs indicateurs (ex: "Compare le chômage et la pauvreté"), choisis plusieurs colonnes.
       - Si c'est une simple évolution, une seule suffit.
       - Choisis des variables comparables entre les territoires (ex: taux, part, pour X habitants).
    2. Définis le format ('formats') pour chaque colonne choisie.
    
    JSON ATTENDU :
    {
      "selected_columns": ["col1", "col2"],
      "formats": {
        "col1": { "kind": "percent|currency|number", "decimals": 1, "title": "Titre Axe Y" }
      }
    }
    """

    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
        data = json.loads(resp.choices[0].message.content)
        
        # Fallback si l'IA ne renvoie rien de cohérent
        if not data.get("selected_columns"):
            data["selected_columns"] = [numeric_cols[0]]
            
        # Filtrer pour être sûr que les colonnes existent
        data["selected_columns"] = [c for c in data["selected_columns"] if c in df.columns]
        
        with st.expander("🎨 IA : Configuration Graphique (Fusionnée)", expanded=False):
             st.json(data)
             
        return data
    except Exception as e:
        return {"selected_columns": [numeric_cols[0]], "formats": {}}

def style_df(df: pd.DataFrame, specs: dict):
    """Applique le formatage (Pandas Styler) pour l'affichage tableau."""
    def fr_num(x, decimals=0, suffix=""):
        if pd.isna(x): return ""
        try:
            s = f"{x:,.{decimals}f}".replace(",", " ").replace(".", ",")
            return (s + (f" {suffix}" if suffix else "")).strip()
        except: return str(x)

    sty = df.style
    for col, s in (specs or {}).items():
        if col not in df.columns: continue
        if not pd.api.types.is_numeric_dtype(df[col]): continue

        kind = (s.get("kind") or "number").lower()
        dec = int(s.get("decimals", 0))
        pb = s.get("percent_base")
        unit = s.get("unit", "")

        if kind == "currency":
            sty = sty.format({col: lambda v, d=dec: fr_num(v, d, "€")})
        elif kind == "percent":
            if pb == "0-1":
                sty = sty.format({col: lambda v, d=dec: fr_num(v * 100.0, d, "%")})
            else:
                sty = sty.format({col: lambda v, d=dec: fr_num(v, d, "%")})
        else:
            sty = sty.format({col: lambda v, d=dec, u=unit: fr_num(v, d, u)})
    return sty

# --- 5. FONCTIONS VECTORIELLES ---
@st.cache_resource
def get_glossary_embeddings(df_glossaire):
    if df_glossaire.empty: return None, []
    cache_dir = "data"
    if not os.path.exists(cache_dir): os.makedirs(cache_dir)
    cache_path = os.path.join(cache_dir, "embeddings_cache.npy")
    
    df_glossaire['combined_text'] = (
        "Src:" + df_glossaire.iloc[:, 0].fillna("").astype(str) + 
        "|Tab:" + df_glossaire.iloc[:, 1].fillna("").astype(str) + 
        "|An:" + df_glossaire.iloc[:, 3].fillna("").astype(str) +
        "|Var:" + df_glossaire.iloc[:, 4].fillna("").astype(str) + 
        "|Def:" + df_glossaire.iloc[:, 5].fillna("").astype(str)
    )
    
    clean_texts = [str(t).strip()[:1000] for t in df_glossaire['combined_text'].tolist() if len(str(t).strip()) > 2]
    if not clean_texts: return None, []
    valid_indices = [i for i, t in enumerate(df_glossaire['combined_text']) if len(str(t).strip()) > 2]

    if os.path.exists(cache_path):
        try:
            embeddings = np.load(cache_path)
            if len(embeddings) == len(clean_texts): return embeddings, valid_indices
        except: pass 

    all_embeddings = []
    BATCH_SIZE = 100 
    try:
        progress_bar = st.sidebar.progress(0, text="Chargement IA...")
        for i in range(0, len(clean_texts), BATCH_SIZE):
            batch = clean_texts[i : i + BATCH_SIZE]
            response = client.embeddings.create(input=batch, model=EMBEDDING_MODEL)
            all_embeddings.extend([d.embedding for d in response.data])
            progress_bar.progress(min((i + BATCH_SIZE) / len(clean_texts), 1.0))
        progress_bar.empty()
        final_embeddings = np.array(all_embeddings)
        np.save(cache_path, final_embeddings)
        return final_embeddings, valid_indices
    except Exception as e:
        st.sidebar.error(f"Erreur IA: {e}")
        if os.path.exists(cache_path): os.remove(cache_path)
        return None, []

def semantic_search(query, df_glossaire, glossary_embeddings, valid_indices, top_k=60):
    if glossary_embeddings is None or df_glossaire.empty: return pd.DataFrame()
    try:
        excluded_tabs = ['Indices', 'INDICES', 'QPV', 'IRIS', 'Indice', 'indices', 'qpv', 'iris'] 
        query_resp = client.embeddings.create(input=[query], model=EMBEDDING_MODEL)
        query_vec = np.array(query_resp.data[0].embedding)
        similarities = np.dot(glossary_embeddings, query_vec)
        
        df_results = df_glossaire.iloc[valid_indices].copy()
        if len(df_results) != len(similarities):
             min_len = min(len(df_results), len(similarities))
             df_results = df_results.iloc[:min_len]
             similarities = similarities[:min_len]
             
        df_results['similarity'] = similarities
        mask = ~df_results.iloc[:, 1].astype(str).isin(excluded_tabs)
        mask_content = ~df_results.iloc[:, 4].astype(str).str.contains(r'IRIS|QPV', case=False, regex=True)
        return df_results[mask & mask_content].sort_values('similarity', ascending=False).head(top_k)
    except: return pd.DataFrame()

# --- 6. MOTEUR DE DONNÉES ---
@st.cache_resource
def init_db():
    con = duckdb.connect(database=':memory:')
    data_folder = "data"
    if not os.path.exists(data_folder): os.makedirs(data_folder)

    files = [f for f in os.listdir(data_folder) if f.endswith('.parquet')]
    schema_map = {} 
    
    for f in files:
        table_name = f.replace('.parquet', '').replace('-', '_').replace(' ', '_').upper()
        file_path = os.path.join(data_folder, f)
        
        try:
            temp_view = f"temp_{table_name}"
            con.execute(f"CREATE OR REPLACE VIEW {temp_view} AS SELECT * FROM read_parquet('{file_path}') LIMIT 1")
            cols_info = con.execute(f"DESCRIBE {temp_view}").fetchall()
            col_defs = []
            for col_name, col_type, _, _, _, _ in cols_info:
                if col_name.upper() in ['ID', 'CODGEO', 'CODE_GEO']:
                    col_defs.append(f"CAST(\"{col_name}\" AS VARCHAR) AS \"{col_name}\"")
                else:
                    col_defs.append(f"\"{col_name}\"")
            select_stmt = ", ".join(col_defs)
            con.execute(f"CREATE OR REPLACE VIEW \"{table_name}\" AS SELECT {select_stmt} FROM read_parquet('{file_path}')")
            con.execute(f"DROP VIEW {temp_view}")
            for col in cols_info:
                real_col = col[0]
                clean_key = real_col.lower().replace("-", "").replace("_", "")
                schema_map[clean_key] = (table_name, real_col)
                schema_map[real_col.lower()] = (table_name, real_col)
        except Exception as e:
            try: con.execute(f'CREATE OR REPLACE VIEW "{table_name}" AS SELECT * FROM read_parquet(\'{file_path}\')')
            except: pass

    df_glossaire = pd.DataFrame()
    try: df_glossaire = pd.read_csv(os.path.join(data_folder, "Glossaire.txt"), encoding='utf-8', sep=None, engine='python')
    except: 
        try: df_glossaire = pd.read_csv(os.path.join(data_folder, "Glossaire.txt"), encoding='latin-1', sep=None, engine='python')
        except: pass

    glossary_embeddings, valid_indices = None, []
    if not df_glossaire.empty:
        glossary_embeddings, valid_indices = get_glossary_embeddings(df_glossaire)

    territoires_path = os.path.join(data_folder, "territoires.txt")
    if os.path.exists(territoires_path):
        try:
            con.execute(f"CREATE OR REPLACE VIEW territoires AS SELECT * FROM read_csv_auto('{territoires_path}', all_varchar=True)")
            if con.execute("SELECT count(*) FROM territoires").fetchone()[0] == 0:
                con.execute(f"CREATE OR REPLACE VIEW territoires AS SELECT * FROM read_csv('{territoires_path}', delim=';', all_varchar=True)")
        except: st.sidebar.error("Erreur critique : Fichier Territoires illisible.")

    return con, schema_map, df_glossaire, glossary_embeddings, valid_indices

con, schema_map, df_glossaire, glossary_embeddings, valid_indices = init_db()

# --- 7. INTELLIGENCE GÉOGRAPHIQUE ---
def clean_search_term(text):
    """Nettoie le terme de recherche pour ne garder que le nom géographique."""
    if not isinstance(text, str): return ""
    
    # 1. Normalisation unicode
    text = text.lower()
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")
    
    # 2. Remplacements standards
    text = text.replace('-', ' ').replace("'", " ").replace("’", " ")
    
    # 3. Suppression des mots-clés administratifs
    keywords = [
        "communaute de communes", "communaute d agglomeration", "communaute urbaine",
        "metropole", "syndicat", "agglomeration", "territoire", "grand", 
        "cc ", "ca ", "cu ", " de ", " du ", " d ", " le ", " la ", " les ", " et "
    ]
    
    for kw in keywords:
        text = text.replace(kw, " ")
        
    return text.strip()

def search_territory_smart(con, input_str):
    """
    Retourne :
    - Soit un tuple unique (ID, NOM, ...) si certitude absolue (Code ou 1 seul résultat Token)
    - Soit une liste de tuples [...] si ambiguïté (Plusieurs résultats proches)
    - Soit None
    """
    clean_input = clean_search_term(input_str)
    
    # 1. Code Exact -> Certitude (On renvoie un tuple unique)
    if input_str.strip().isdigit():
        try:
            res = con.execute(f"SELECT ID, NOM_COUV, COMP1, COMP2, COMP3 FROM territoires WHERE ID = '{input_str.strip()}' LIMIT 1").fetchone()
            if res: return res 
        except: pass

    # 2. Token Search (Mots clés)
    words = [w for w in clean_input.split() if len(w) > 2]
    if words:
        conditions = [f"strip_accents(lower(NOM_COUV)) LIKE '%{w}%'" for w in words]
        sql_keywords = f"""
        SELECT ID, NOM_COUV, COMP1, COMP2, COMP3
        FROM territoires WHERE {" AND ".join(conditions)}
        ORDER BY length(NOM_COUV) ASC LIMIT 5
        """
        try:
            results = con.execute(sql_keywords).fetchall()
            if len(results) == 1: return results[0] # Un seul candidat -> Certitude
            if len(results) > 1: return results # Plusieurs candidats -> Ambiguïté
        except: pass

    # 3. Fuzzy Search (Jaro-Winkler)
    sql_fuzzy = f"""
    WITH clean_data AS (
        SELECT ID, NOM_COUV, COMP1, COMP2, COMP3,
        lower(replace(replace(replace(NOM_COUV, '-', ' '), '''', ' '), '’', ' ')) as nom_simple
        FROM territoires
    )
    SELECT ID, NOM_COUV, COMP1, COMP2, COMP3,
    jaro_winkler_similarity(nom_simple, '{clean_input}') as score
    FROM clean_data
    WHERE score > 0.65 
    ORDER BY score DESC LIMIT 5
    """
    try:
        results = con.execute(sql_fuzzy).fetchall()
        if not results: return None
        
        # Logique de tri des scores
        top_score = results[0][5] # Le score est en 6ème position
        # On garde ceux qui sont proches du meilleur score (différence < 0.1)
        candidates = [r for r in results if (top_score - r[5]) < 0.1]
        
        if len(candidates) == 1: return candidates[0][:5] # Un seul -> Certitude
        
        # On retourne la liste (sans le score) -> Ambiguïté
        return [c[:5] for c in candidates]
        
    except: pass
    
    return None

def analyze_territorial_scope(con, rewritten_prompt):
    if not con: return None
    try:
        extraction = client.chat.completions.create(
            model=  MODEL_NAME,
            messages=[
                {"role": "system", "content": "Extrais les entités géographiques. SI un département est précisé (ex: 'Montreuil 93'), combine-le (ex: 'Montreuil (93)'). JSON: {\"lieux\": [\"Cergy\", \"Montreuil (93)\"]}. Si aucun, {\"lieux\": []}."},
                {"role": "user", "content": rewritten_prompt}
            ],
            response_format={"type": "json_object"}
        )
        data = json.loads(extraction.choices[0].message.content)
        lieux_cites = data.get("lieux", [])
        
        with st.expander("🌍 Trace : Extraction Territoires (IA)", expanded=False):
            st.json(data)
            
    except Exception as e: 
        st.error(f"Erreur Extraction: {e}")
        return None

    if not lieux_cites: return None

    found_ids = []
    found_names = []
    primary_res = None
    
    debug_search = []
    # Gestion de l'ambiguïté sur le premier lieu trouvé (le plus important)
    # On suppose que l'utilisateur cherche principalement un lieu
    first_lieu = lieux_cites[0]
    search_res = search_territory_smart(con, first_lieu)
    
    # Cas Ambiguïté : search_res est une LISTE
    if isinstance(search_res, list):
        return {
            "ambiguity": True,
            # MODIFICATION ICI : On ajoute "comps" pour stocker les comparateurs
            "candidates": [{
                "id": r[0], 
                "nom": r[1], 
                "comps": [str(c) for c in [r[2], r[3], r[4]] if c and str(c) not in ['None', 'nan', '']]
            } for r in search_res],
            "input_text": first_lieu,
            "display_context": "Ambiguïté détectée"
        }
    
    # Cas Normal : search_res est un TUPLE (ou None)
    if search_res:
        primary_res = search_res
        found_ids.append(str(primary_res[0]))
        found_names.append(primary_res[1])
        debug_search.append({"Recherche": first_lieu, "Trouvé": primary_res[1], "ID": primary_res[0]})
    else:
        debug_search.append({"Recherche": first_lieu, "Trouvé": "NON"})

    if not found_ids: return None

    final_ids_list = list(found_ids)
    if primary_res:
        _, _, c1, c2, c3 = primary_res
        comps = [str(c) for c in [c1, c2, c3] if c and str(c).strip() not in ['', 'None', 'nan', 'null']]
        final_ids_list.extend(comps)
        
    final_ids_list.append('FR')
    unique_ids = list(dict.fromkeys(final_ids_list))
    
    parent_clause = ""
    display_suffix = ""
    prompt_lower = rewritten_prompt.lower()
    is_hierarchical = any(k in prompt_lower for k in ["toutes les", "tous les", "liste des", "l'ensemble des", "communes de"])
    
    if is_hierarchical and primary_res:
        type_filter = ""
        if any(k in prompt_lower for k in ["commune", "ville"]):
            type_filter = "length(t.ID) = 5" 
            display_suffix = " (Communes)"
        elif any(k in prompt_lower for k in ["epci", "interco", "agglomération", "ept"]):
            type_filter = "length(t.ID) = 9"
            display_suffix = " (EPCI)"
            
        if type_filter:
            target_id = str(primary_res[0])
            parent_clause = f"OR ((t.COMP1 = '{target_id}' OR t.COMP2 = '{target_id}' OR t.COMP3 = '{target_id}') AND {type_filter})"
    
    display_name = " & ".join(found_names[:3])
    
    return {
        "target_name": display_name,
        "target_id": str(found_ids[0]),
        "all_ids": unique_ids,
        "parent_clause": parent_clause,
        "display_context": f"{display_name}{display_suffix}",
        "debug_search": debug_search,
        "lieux_cites": lieux_cites
    }

# --- 8. VISUALISATION AUTO (SMART FORMAT & SELECTION IA) ---
def auto_plot_data(df, sorted_ids, config=None):
    """
    Trace le graphique avec respect strict de l'ordre des couleurs :
    Rouge (Cible), Orange (Comp1), Vert (Comp2), Bleu (Comp3), Violet (FR/Autre).
    """
    if config is None: config = {}
    selected_metrics = config.get("selected_columns", [])
    format_specs = config.get("formats", {})
    
    # --- COULEURS IMPOSÉES ---
    color_range = ["#EB2C30", "#F38331", "#97D422", "#1DB5C5", "#5C368D"]
    
    cols = df.columns.tolist()
    label_col = next((c for c in cols if c.upper() in ["NOM_COUV", "TERRITOIRE", "LIBELLE", "VILLE"]), None)
    date_col = next((c for c in cols if c.upper() in ["AN", "ANNEE", "YEAR", "DATE"]), None)
    id_col = next((c for c in cols if c.upper() == "ID"), None)

    if not selected_metrics or not label_col: return

    # --- TRI CRITIQUE ---
    # On force le tri du DF selon l'ordre exact de 'sorted_ids' pour que les couleurs s'alignent.
    df_plot = df.copy()
    if id_col:
        id_order_map = {str(uid): i for i, uid in enumerate(sorted_ids)}
        df_plot['sort_order'] = df_plot[id_col].astype(str).map(id_order_map)
        df_plot = df_plot.sort_values('sort_order').drop(columns=['sort_order'])
    
    # On extrait les labels dans l'ordre trié -> C'est notre Domain pour Vega
    sorted_labels = df_plot[label_col].unique().tolist()

    # Si pas de date et trop de territoires (et 1 seule métrique), on coupe à 5 pour ne pas surcharger
    if not date_col and len(df_plot) > 5 and len(selected_metrics) == 1:
        df_plot = df_plot.head(5)
        sorted_labels = df_plot[label_col].unique().tolist() # Re-update labels

    # --- MELT ---
    id_vars = [label_col]
    if date_col: id_vars.append(date_col)
    df_melted = df_plot.melt(id_vars=id_vars, value_vars=selected_metrics, var_name="Indicateur", value_name="Valeur")

    # --- FORMATS ---
    first_metric = selected_metrics[0]
    spec = format_specs.get(first_metric, {})
    title_y = spec.get("title", "Valeur")
    y_format = ",.1f"
    if spec.get("kind") == "percent":
        y_format = ".1%"
        if spec.get("percent_base") == "0-100": df_melted["Valeur"] = df_melted["Valeur"] / 100.0
    elif spec.get("kind") == "currency": y_format = ",.0f"

    # --- CONFIG COMMUNE ---
    vega_config = {
        "locale": {"number": {"decimal": ",", "thousands": "\u00a0", "grouping": [3]}},
        "axis": {"labelFontSize": 11, "titleFontSize": 12},
        "legend": {"labelFontSize": 11, "titleFontSize": 12, "orient": "bottom"}
    }
    
    # --- DÉFINITION COULEUR STRICTE ---
    # La couleur dépend TOUJOURS du Territoire, avec le domaine forcé
    color_def = {
        "field": label_col, 
        "type": "nominal", 
        "scale": {"domain": sorted_labels, "range": color_range}, 
        "title": "Territoire",
        "legend": {"orient": "bottom"}
    }
    
    chart = None
    is_multi_metric = len(selected_metrics) > 1

    # CAS 1 : GRAPHIQUE TEMPOREL (LIGNE)
    if date_col:
        chart_encoding = {
            "x": {"field": date_col, "type": "ordinal", "title": "Année"},
            "y": {"field": "Valeur", "type": "quantitative", "title": title_y, "axis": {"format": y_format}},
            "color": color_def, # Rouge = Cible
            "tooltip": [{"field": label_col}, {"field": "Indicateur"}, {"field": date_col}, {"field": "Valeur", "format": y_format}]
        }
        
        # Si multi-metric, on distingue les variables par le style de trait
        if is_multi_metric:
            chart_encoding["strokeDash"] = {"field": "Indicateur", "title": "Variable"}

        chart = {
            "config": vega_config,
            "mark": {"type": "line", "point": True, "tooltip": True},
            "encoding": chart_encoding
        }

    # CAS 2 : GRAPHIQUE BARRES (COMPARAISON)
    else:
        # Configuration Multi-Variables "Groupée par Variable"
        # X = Indicateur (pour grouper les KPIs)
        # xOffset = Territoire (pour comparer Rouge vs Orange côte à côte)
        # Color = Territoire (pour garder l'identité visuelle)
        
        if is_multi_metric:
             chart_encoding = {
                "x": {"field": "Indicateur", "type": "nominal", "axis": {"labelAngle": 0, "title": None}},
                "y": {"field": "Valeur", "type": "quantitative", "title": title_y, "axis": {"format": y_format}},
                "color": color_def,
                "xOffset": {"field": label_col}, # C'est ici que se fait le groupement
                "tooltip": [{"field": label_col}, {"field": "Indicateur"}, {"field": "Valeur", "format": y_format}]
            }
        else:
            # Config classique
            chart_encoding = {
                "x": {"field": label_col, "type": "nominal", "sort": sorted_labels, "axis": {"labelAngle": -45}, "title": None},
                "y": {"field": "Valeur", "type": "quantitative", "title": title_y, "axis": {"format": y_format}},
                "color": color_def,
                "tooltip": [{"field": label_col}, {"field": "Valeur", "format": y_format}]
            }

        chart = {
            "config": vega_config,
            "mark": {"type": "bar", "cornerRadiusEnd": 3, "tooltip": True},
            "encoding": chart_encoding
        }
    
    st.vega_lite_chart(df_melted, chart, use_container_width=True)


# --- 9. UI PRINCIPALE ---
st.title("🗺️ Terribot")
st.markdown("#### L'expert des données territoriales")

# Initialisation des variables de session pour l'ambiguïté
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Bonjour ! Quel territoire souhaitez-vous analyser ?"}]
if "current_geo_context" not in st.session_state:
    st.session_state.current_geo_context = None
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None
if "ambiguity_candidates" not in st.session_state:
    st.session_state.ambiguity_candidates = None

for msg in st.session_state.messages:
    avatar = "🤖" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        # 1. DEBUG (Replié)
        if "debug_info" in msg:
            with st.expander("🔧 Détails techniques (Historique)", expanded=False):
                # Reformulation
                if "reformulation" in msg["debug_info"]:
                    st.markdown("**Reformulation :**")
                    st.info(msg["debug_info"]["reformulation"])
                
                # Extraction & Résolution
                if "geo_extraction" in msg["debug_info"]:
                    st.markdown("**Extraction Territoires :**")
                    st.json(msg["debug_info"]["geo_extraction"])
                if "geo_resolution" in msg["debug_info"]:
                    st.markdown("**Résolution Territoires :**")
                    st.table(pd.DataFrame(msg["debug_info"]["geo_resolution"]))
                if "final_ids" in msg["debug_info"]:
                    st.markdown("**IDs retenus :**")
                    st.code(str(msg["debug_info"]["final_ids"]))
                
                # RAG & SQL
                if "rag_context" in msg["debug_info"]:
                    st.markdown("**Variables trouvées (RAG) :**")
                    st.text(msg["debug_info"]["rag_context"])
                if "sql_query" in msg["debug_info"]:
                    st.markdown("**Requête SQL :**")
                    st.code(msg["debug_info"]["sql_query"], language="sql")

        # 2. TABLEAU DONNÉES (Replié)
        if "data" in msg:
            with st.expander("📊 Voir les données brutes", expanded=False):
                df_display = msg["data"]
                specs = msg.get("format_specs", {})
                try:
                    st.dataframe(style_df(df_display, specs), use_container_width=True)
                except:
                    st.dataframe(df_display, use_container_width=True)
                
                # Bouton Download CSV
                try:
                    csv = df_display.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Télécharger les données (CSV)",
                        data=csv,
                        file_name="terribot_export.csv",
                        mime="text/csv",
                    )
                except: pass

        # 3. GRAPHIQUE (Déplié / Visible directement)
        if "data" in msg and "ID" in msg["data"].columns:
            specs = msg.get("format_specs", {})
            sorted_ids = msg["data"]["ID"].tolist() 
            if "debug_info" in msg and "final_ids" in msg["debug_info"]:
                 sorted_ids = msg["debug_info"]["final_ids"]
            
            # Récupérer la métrique choisie par l'IA si dispo
            # Ici on récupère plutôt la config entière si elle a été sauvée
            # Pour la rétrocompatibilité ou si selected_metric est utilisé seul :
            forced_cols = [msg.get("selected_metric")] if msg.get("selected_metric") else []
            saved_config = {"selected_columns": forced_cols, "formats": specs}
            # (idéalement on sauverait 'chart_config' en entier dans msg)
            
            try: auto_plot_data(msg["data"], sorted_ids, config=saved_config)
            except: pass

        # 4. ANALYSE (Visible)
        st.markdown(msg["content"])


# --- 10. TRAITEMENT ET GESTION AMBIGUÏTÉ ---
inject_placeholder_animation()

# Initialisation de la variable de déclenchement si elle n'existe pas
if "trigger_run_prompt" not in st.session_state:
    st.session_state.trigger_run_prompt = None

# -- A. RÉSOLUTION D'AMBIGUÏTÉ (Affichage des boutons si nécessaire) --
if st.session_state.ambiguity_candidates:
    st.warning(f"🤔 Plusieurs territoires trouvés pour '{st.session_state.ambiguity_candidates[0].get('nom', 'ce lieu')}'. Veuillez préciser :")
    
    cols = st.columns(min(len(st.session_state.ambiguity_candidates), 4))
    
    for i, cand in enumerate(st.session_state.ambiguity_candidates[:4]):
        # On affiche le bouton
        if cols[i].button(f"{cand['nom']} ({cand['id']})", key=f"amb_btn_{cand['id']}"):
            
            # 1. Construction de la liste
            ordered_ids = [str(cand['id'])]
            if "comps" in cand and isinstance(cand["comps"], list):
                 # On nettoie bien les comparateurs
                 valid_comps = [str(c) for c in cand["comps"] if c and str(c).lower() not in ['none', 'nan', 'null', '']]
                 ordered_ids.extend(valid_comps)
            ordered_ids.append('FR')
            
            # Dédoublonnage
            final_ids_ordered = list(dict.fromkeys(ordered_ids))
            
            # 2. Mise à jour du contexte
            st.session_state.current_geo_context = {
                "target_name": cand['nom'],
                "target_id": str(cand['id']),
                "all_ids": final_ids_ordered, # C'est CRUCIAL que cette liste soit pleine ici
                "parent_clause": "",
                "display_context": cand['nom'],
                "debug_search": [{"Trouvé": cand['nom'], "Source": "Choix Utilisateur"}],
                "lieux_cites": [cand['nom']]
            }
            
            st.session_state.trigger_run_prompt = st.session_state.pending_prompt
            
            # 3. LE VERROU (Important !)
            st.session_state.force_geo_context = True 
            
            st.session_state.ambiguity_candidates = None
            st.session_state.pending_prompt = None
            st.rerun()

# -- B. INPUT PRINCIPAL --
user_input = st.chat_input("Posez votre question...")

# -- C. LOGIQUE DE DÉCISION (Quel prompt traiter ?) --
prompt_to_process = None

# Priorité 1 : On vient de cliquer sur un bouton (variable stockée en session)
if st.session_state.trigger_run_prompt:
    prompt_to_process = st.session_state.trigger_run_prompt
    st.session_state.trigger_run_prompt = None # On consomme le trigger pour ne pas boucler

# Priorité 2 : L'utilisateur vient de taper une nouvelle question
elif user_input:
    prompt_to_process = user_input

# --- D. EXÉCUTION DU TRAITEMENT ---
if prompt_to_process:
    # Si c'est un nouvel input utilisateur, on l'ajoute à l'historique
    # (On vérifie pour éviter les doublons lors de la reprise après ambiguïté)
    last_msg = st.session_state.messages[-1] if st.session_state.messages else {}
    if last_msg.get("content") != prompt_to_process or last_msg.get("role") != "user":
        st.session_state.messages.append({"role": "user", "content": prompt_to_process})
        with st.chat_message("user", avatar="👤"): st.markdown(prompt_to_process)

    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        debug_container = {} 
        
        try:
            # 1. REFORMULATION
            history_text = "\n".join([f"{m['role']}: {m.get('content','')}" for m in st.session_state.messages[-4:]])
            current_geo_name = st.session_state.current_geo_context['target_name'] if st.session_state.current_geo_context else ""
            
            reformulation = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": f"""
                    Tu es un expert en reformulation. CONTEXTE GEO ACTUEL : '{current_geo_name}'.
                    OBJECTIFS :
                    1. Rendre la question autonome.
                    2. SI "ramène à la population" ou "et pour X ?", REPRENDS le SUJET PRÉCÉDENT.
                    3. Si aucun lieu explicite dans la question, réinjecte '{current_geo_name}'.
                    """},
                    {"role": "user", "content": f"Historique:\n{history_text}\n\nDernière question: {prompt_to_process}"}
                ]
            )
            rewritten_prompt = reformulation.choices[0].message.content
            debug_container["reformulation"] = f"Original: {prompt_to_process}\nReformulé: {rewritten_prompt}"
            
            with st.expander("🤔 Trace : Reformulation (IA)", expanded=False):
                st.write(f"**Question originale :** {prompt_to_process}")
                st.write(f"**Reformulée :** {rewritten_prompt}")

            # 2. GEO SCOPE
            new_context = None
            
            # --- MODIFICATION ICI : Gestion du Verrou ---
            if st.session_state.get("force_geo_context"):
                st.session_state.force_geo_context = False # On consomme le verrou
                # On ne lance PAS analyze_territorial_scope, on garde l'existant
                if st.session_state.current_geo_context:
                    geo_context = st.session_state.current_geo_context
                    message_placeholder.info(f"📍 **Périmètre validé :** {geo_context['display_context']}")
                    # On force new_context à None pour sauter les blocs suivants
                    new_context = None 
            else:
                # Analyse normale
                new_context = analyze_territorial_scope(con, rewritten_prompt)
                
            # --- GESTION DE L'AMBIGUÏTÉ DÉTECTÉE ---
            # Si une ambiguïté est détectée ET que ce n'est pas le contexte qu'on vient juste de forcer
            if new_context and new_context.get("ambiguity"):
                # Petite sécurité : si le lieu ambigu est le même que celui qu'on a déjà validé, on ignore l'ambiguïté
                if st.session_state.current_geo_context and new_context['input_text'] in st.session_state.current_geo_context['target_name']:
                     pass # On garde le contexte actuel
                else:
                    # On stocke l'état et on arrête l'exécution pour afficher les boutons au prochain tour
                    st.session_state.ambiguity_candidates = new_context['candidates']
                    st.session_state.pending_prompt = prompt_to_process
                    st.session_state.messages.append({"role": "assistant", "content": f"🤔 J'ai un doute sur le lieu **{new_context['input_text']}**. Veuillez choisir ci-dessus."})
                    st.rerun()

            # Mise à jour du contexte si un nouveau lieu valide est trouvé
            if new_context and not new_context.get("ambiguity"):
                st.session_state.current_geo_context = new_context
                message_placeholder.info(f"📍 **Périmètre :** {new_context['display_context']}")
                
                debug_container["geo_extraction"] = new_context["lieux_cites"]
                debug_container["geo_resolution"] = new_context["debug_search"]
                debug_container["final_ids"] = new_context["all_ids"]
            
            # Si on n'a rien trouvé de nouveau, on utilise le contexte existant (celui du bouton par exemple)
            elif st.session_state.current_geo_context:
                 geo_context = st.session_state.current_geo_context
                 # On ne réaffiche pas l'info si elle n'a pas changé, ou on peut la laisser pour confirmation
            
            elif not st.session_state.current_geo_context:
                message_placeholder.warning("⚠️ Je ne détecte pas de territoire. Précisez une ville.")
                st.stop()
            
            geo_context = st.session_state.current_geo_context

            # 3. RAG (Recherche Variables)
            glossaire_context = ""
            glossaire_context += "✅ TAB:\"EVO\" | VAR:\"POP_22\" | DEF:\"Population en 2022\" | SRC:INSEE | AN:2022\n"
            
            if not df_glossaire.empty and glossary_embeddings is not None:
                with st.spinner("Recherche des données..."):
                    results_semantic = semantic_search(rewritten_prompt, df_glossaire, glossary_embeddings, valid_indices, top_k=50)
                    mask_pop = df_glossaire.iloc[:, 5].astype(str).str.contains(r'population|habitant|recensement', case=False, regex=True)
                    results_pop = df_glossaire[mask_pop].head(15)
                    final_results = pd.concat([results_semantic, results_pop]).drop_duplicates().head(80)
                    
                    rows = []
                    for _, row in final_results.iterrows():
                        try:
                            s, y, v, d = str(row.iloc[0]), str(row.iloc[3]), str(row.iloc[4]), str(row.iloc[5])
                            clean = v.strip().lower().replace("-", "").replace("_", "")
                            if clean in schema_map:
                                t, c = schema_map[clean]
                                rows.append(f"✅ TAB:\"{t}\" | VAR:\"{c}\" | DEF:\"{d}\" | SRC:{s} | AN:{y}")
                            elif v.lower() in schema_map:
                                t, c = schema_map[v.lower()]
                                rows.append(f"✅ TAB:\"{t}\" | VAR:\"{c}\" | DEF:\"{d}\" | SRC:{s} | AN:{y}")
                        except: continue
                    glossaire_context += "\n".join(rows)
                    debug_container["rag_context"] = glossaire_context[:2000] + "..."
                    
                    with st.expander("📚 Trace : Variables trouvées (RAG)", expanded=False):
                        st.text(glossaire_context)

            # 4. SQL GENERATION
            ids_sql = ", ".join([f"'{str(i)}'" for i in geo_context['all_ids']])
            parent_clause = geo_context.get('parent_clause', '')
            
            system_prompt = f"""
            Tu es Terribot.
            CONTEXTE DONNÉES (Glossaire) :
            {glossaire_context}
            
            SCHEMA TABLE "TERRITOIRES" (alias t) :
            - "ID" (VARCHAR)
            - "NOM_COUV" (VARCHAR)
            
            MISSION : Répondre à "{rewritten_prompt}" via UNE SEULE requête SQL.
            
            🚨 RÈGLES CRITIQUES :
            
            1. PÉRIMÈTRE GÉOGRAPHIQUE :
               - Copie STRICTEMENT cette clause WHERE :
               - `WHERE (t."ID" IN ({ids_sql}) {parent_clause})`
               - ⛔ INTERDIT : N'ajoute JAMAIS de condition `AND t."NOM_COUV" = ...`.
               - ⛔ INTERDIT : N'ajoute JAMAIS de condition `AND t."NOM_COUV" LIKE ...`.
               - La liste des IDs contient déjà la cible + les comparateurs + la France. Si tu filtres sur le nom, tu perds les comparaisons.
            
            2. STRUCTURE DES DONNÉES :
               - Tables format LARGE (une colonne par variable).
               - Pas de colonne "VAR" ou "VALUE". Utilise le nom de la variable directement (ex: `e."POP_22"`).
               - Utilise `TRY_CAST(table."colonne" AS DOUBLE)`.
            
            3. JOINTURES :
               - `LEFT JOIN` sur la colonne "ID".
            
            4. INTELLIGENCE :
               - Calcule toujours des ratios pour comparer (ex: / POP_22).
               - Gère les divisions par zéro : `NULLIF(..., 0)`.
               - Réfléchis bien à la variable à utiliser, et si ça répond à la question.
               - INTERDIT : faire des approximations au doigt mouillé.
               - Les noms de variables que tu crées doivent être descriptifs, compréhensibles par un inconnu, sans contexte supplémentaire.
            
            Réponds uniquement le SQL.
            """

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": rewritten_prompt}],
                temperature=0
            )
            sql_query_raw = response.choices[0].message.content.replace("```sql", "").replace("```", "").strip()
            sql_query = sql_query_raw.split(";")[0]
            debug_container["sql_query"] = sql_query
            
            with st.expander("💻 Trace : Génération SQL (IA)", expanded=False):
                st.code(sql_query, language="sql")

            # 5. EXECUTION & VISUALISATION
            if con:
                try:
                    df = con.execute(sql_query).df()
                    
                    if not df.empty:
                        with st.status("Génération de l'analyse et du graphique...", expanded=True) as status:
                            
                            st.write("🎨 Configuration intelligente du graphique...")
                            # Appel unique fusionné pour choisir colonnes + formats
                            chart_config = get_chart_configuration(df, rewritten_prompt, glossaire_context, client, MODEL_NAME)
                            
                            st.write("📊 Création du graphique...")
                            # On passe la config entière à la fonction de plot
                            auto_plot_data(df, geo_context['all_ids'], config=chart_config)
                            
                            st.write("📝 Rédaction de la synthèse...")
                            analysis = client.chat.completions.create(
                                model=MODEL_NAME,  
                                messages=[
                                    {"role": "system", "content": f"""
                                    Tu es Terribot. Analyse les résultats pour : {rewritten_prompt}.
                                    CONSIGNE : Sois très SYNTHÉTIQUE (max 15 lignes). Va droit au but et utilise des puces.
                                    Respecte les unités fournies : {json.dumps(chart_config.get('formats', {}), ensure_ascii=False)}
                                    """},
                                    {"role": "user", "content": df.to_string()}
                                ]
                            )
                            status.update(label="Terminé", state="complete", expanded=False)
                        
                        final_resp = analysis.choices[0].message.content
                        
                        # Affichage des données (Expander)
                        with st.expander("📊 Voir les données brutes", expanded=False):
                            try: st.dataframe(style_df(df, chart_config.get('formats', {})), use_container_width=True)
                            except: st.dataframe(df, use_container_width=True)
                            try:
                                csv = df.to_csv(index=False).encode('utf-8')
                                st.download_button("📥 Télécharger (CSV)", data=csv, file_name="terribot_export.csv", mime="text/csv")
                            except: pass

                        # Affichage du graphique (Visible)
                        auto_plot_data(df, geo_context['all_ids'], config=chart_config)

                        message_placeholder.markdown(final_resp)
                        
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": final_resp, 
                            "data": df,
                            "format_specs": chart_config.get('formats', {}),
                            # On stocke selected_metric pour compatibilité, mais c'est bien une liste dans chart_config
                            "selected_metric": chart_config.get('selected_columns', [])[0] if chart_config.get('selected_columns') else None,
                            "debug_info": debug_container
                        })
                    else:
                        message_placeholder.warning("Aucune donnée trouvée.")
                except Exception as e:
                    message_placeholder.error("Erreur technique.")
                    with st.expander("Debug Erreur"): st.write(e)

        except Exception as e:
            st.error(f"Erreur : {e}")