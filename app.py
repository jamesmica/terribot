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
    st.caption("Intelligence Territoriale v5.1 (UI Order)")
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
    st.info("💡 **Dataviz :** L'IA choisit elle-même la variable du graphique selon votre question.")

client = openai.OpenAI(api_key=api_key)
MODEL_NAME = "gpt-5.2-2025-12-11" # Si ce modèle n'existe pas, OpenAI fallbackera sur un gpt-4o ou similaire selon votre plan, mais je garde le string tel quel.
EMBEDDING_MODEL = "text-embedding-3-small"

# --- 4. FONCTIONS INTELLIGENTES (FORMATAGE & SÉLECTION) ---
def infer_format_specs_with_ai(df: pd.DataFrame, question: str, glossaire_context: str, client, model: str):
    """Demande à l'IA de deviner le format (%, €, nombre) pour chaque colonne numérique."""
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_cols: return {}

    stats = {}
    for c in numeric_cols:
        s = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(s) == 0: continue
        stats[c] = {"min": float(s.min()), "max": float(s.max()), "mean": float(s.mean())}

    payload = {
        "question": question,
        "columns": df.columns.tolist(),
        "numeric_columns": numeric_cols,
        "stats": stats,
        "glossaire_context": (glossaire_context or "")[-4000:],
    }

    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": """
Tu es un expert en visualisation de données. Ton but est de déterminer le format d'affichage optimal pour chaque colonne numérique.

Retour attendu (JSON strict) :
{
  "columns": {
    "<nom_colonne>": {
      "kind": "percent|currency|count|number",
      "percent_base": "0-1|0-100|null",
      "decimals": 0-2,
      "unit": "%|€|hab|",
      "title": "Titre court et propre pour l'axe"
    }
  }
}

Règles de décision :
1. "percent" : Si c'est un taux, une part, une évolution.
   - Si les valeurs sont entre 0 et 1 (ex: 0.15), percent_base="0-1".
   - Si les valeurs sont entre 0 et 100 (ex: 15.0), percent_base="0-100".
2. "currency" : Si c'est un revenu, un prix, un montant (unit="€").
3. "count" : Si c'est une population, un nombre de logements, un volume brut.
4. "number" : Par défaut (ex: densité, ratio sans unité).

Sois logique avec les stats fournies (min/max).
"""},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
        data = json.loads(resp.choices[0].message.content)
        
        # --- APPARENCE : Affichage du raisonnement IA ---
        with st.expander("🎨 IA : Détection des Formats", expanded=False):
             st.caption("L'IA analyse les stats (min/max) et le nom des colonnes pour deviner les unités.")
             st.json(data)
        # ------------------------------------------------
        
        return data.get("columns", {}) or {}
    except:
        return {}

def select_best_metric_for_chart(df: pd.DataFrame, question: str, client, model: str):
    """L'IA choisit la colonne la plus pertinente pour le graphique en fonction de la question."""
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    # On retire les colonnes années/id si présentes par erreur dans les numériques
    numeric_cols = [c for c in numeric_cols if c.upper() not in ["AN", "ANNEE", "YEAR", "ID"]]
    
    if not numeric_cols: return None
    if len(numeric_cols) == 1: return numeric_cols[0] # Pas le choix

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Tu es un expert Dataviz. Ta mission : Choisir la colonne numérique la plus pertinente pour faire un graphique qui répond à la question de l'utilisateur. Renvoie un JSON : {\"column\": \"nom_exact_colonne\"}."},
                {"role": "user", "content": f"Question reformulée : {question}\nColonnes disponibles : {numeric_cols}"}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        selected_json = json.loads(response.choices[0].message.content)
        selected = selected_json.get("column")
        
        # --- APPARENCE : Affichage du choix IA ---
        with st.expander("📊 IA : Choix de la métrique Graphique", expanded=False):
             st.caption(f"Colonnes dispos : {numeric_cols}")
             st.write(f"Choix de l'IA : **{selected}**")
        # -----------------------------------------

        if selected in df.columns:
            return selected
        return numeric_cols[0] # Fallback
    except:
        return numeric_cols[0]

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
def normalize_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")
    text = text.replace('-', ' ').replace("'", " ").replace("’", " ")
    text = text.replace("st ", "saint ").replace("ste ", "sainte ")
    return text.strip()

def search_territory_smart(con, input_str):
    """ Recherche ID ou Nom (robuste aux variantes) """
    clean_raw = input_str.strip()
    
    dept_hint = ""
    if "(" in clean_raw and ")" in clean_raw:
        parts = clean_raw.split("(")
        clean_raw = parts[0].strip()
        dept_hint = parts[1].replace(")", "").strip()

    if clean_raw.isdigit() and len(clean_raw) <= 3: 
        query = f"SELECT ID, NOM_COUV, COMP1, COMP2, COMP3 FROM territoires WHERE ID = 'D{clean_raw}' LIMIT 1"
        try:
            res = con.execute(query).fetchone()
            if res: return res
        except: pass

    norm_input = normalize_text(clean_raw)
    sql_clean_col = "lower(replace(replace(replace(replace(NOM_COUV, '-', ' '), '''', ' '), '’', ' '), 'St ', 'Saint '))"
    
    sql_dept_filter = ""
    if dept_hint and dept_hint.isdigit():
        sql_dept_filter = f"AND (ID LIKE '{dept_hint}%' OR COMP2 = 'D{dept_hint}')"
    
    query = f"""
    SELECT ID, NOM_COUV, COMP1, COMP2, COMP3 
    FROM territoires 
    WHERE ("ID" = '{clean_raw}' OR {sql_clean_col} LIKE '%{norm_input}%')
    {sql_dept_filter}
    ORDER BY length("NOM_COUV") ASC LIMIT 1
    """
    try:
        res = con.execute(query).fetchone()
        if res: return res
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
    for lieu in lieux_cites:
        res = search_territory_smart(con, lieu)
        if res:
            found_ids.append(str(res[0]))
            found_names.append(res[1])
            debug_search.append({"Recherche": lieu, "Trouvé": res[1], "ID": res[0]})
            if not primary_res: primary_res = res
        else:
            debug_search.append({"Recherche": lieu, "Trouvé": "NON"})

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
def auto_plot_data(df, sorted_ids, format_specs=None, forced_metric=None):
    """
    Trace un graphique optimisé.
    - Max 5 territoires (sauf si explicitement demandé, ici on coupe par défaut).
    - Couleurs imposées.
    - Format d'unités via format_specs (IA).
    - Sélection de variable via forced_metric (IA) ou heuristic.
    """
    color_range = ["#EB2C30", "#F38331", "#97D422", "#1DB5C5", "#5C368D"]
    format_specs = format_specs or {}

    cols = df.columns.tolist()
    label_col = next((c for c in cols if c.upper() in ["NOM_COUV", "TERRITOIRE", "LIBELLE", "VILLE"]), None)
    date_col = next((c for c in cols if c.upper() in ["AN", "ANNEE", "YEAR", "DATE"]), None)
    id_col = next((c for c in cols if c.upper() == "ID"), None)
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if date_col in numeric_cols: numeric_cols.remove(date_col)
    
    if not numeric_cols or not label_col: return

    # --- TRI & FILTRE 5 ---
    df_plot = df.copy()
    if id_col:
        id_order_map = {str(uid): i for i, uid in enumerate(sorted_ids)}
        df_plot['sort_order'] = df_plot[id_col].astype(str).map(id_order_map)
        df_plot = df_plot.sort_values('sort_order').drop(columns=['sort_order'])
    
    # Règle : Max 5 territoires si pas de date
    if not date_col and len(df_plot) > 5:
        df_plot = df_plot.head(5)

    sorted_labels = df_plot[label_col].unique().tolist()

    # --- CHOIX MÉTRIQUE (IA OU HEURISTIQUE) ---
    if forced_metric and forced_metric in numeric_cols:
        target_metric = forced_metric
    else:
        # Fallback heuristic
        valid_metrics = [c for c in numeric_cols if not any(k in c.lower() for k in ['pop', 'habitant', 'nombre_personne'])]
        if not valid_metrics: valid_metrics = numeric_cols
        
        target_metric = valid_metrics[0]
        ratio_keywords = ['%', 'taux', 'part', 'ratio', 'evol', 'densite', 'revenu', 'ind', 'score', '/']
        for col in valid_metrics:
            if any(k in col.lower() for k in ratio_keywords):
                target_metric = col
                break

    # --- FORMAT VIA IA ---
    # On récupère les infos de formatage pour l'axe Y
    spec = format_specs.get(target_metric, {})
    kind = spec.get("kind", "number")
    pb = spec.get("percent_base")
    dec = int(spec.get("decimals", 1))
    title = spec.get("title", target_metric)
    
    y_format = f",.{dec}f"
    y_field = target_metric
    
    # Adaptation Vega selon format
    if kind == "percent":
        title = f"{title} (%)" if "%" not in title else title
        y_format = f".{dec}%"
        if pb == "0-100":
            # Vega attend 0-1 pour le format %, donc on divise
            df_plot[target_metric + "_frac"] = df_plot[target_metric] / 100.0
            y_field = target_metric + "_frac"
            
    elif kind == "currency":
        title = f"{title} (€)" if "€" not in title else title
        y_format = f"$.{dec}f" # Vega utilise d3-format, $ marche souvent pour currency locale selon config, sinon ","
        # Note: Vega-Lite localize support est parfois limité en simple json, on reste sur du standard
        y_format = f",.{dec}f" # Séparateur milliers standard

    # --- PLOT ---
    if date_col:
        st.vega_lite_chart(df_plot, {
            "mark": {"type": "line", "point": True, "tooltip": True},
            "encoding": {
                "x": {"field": date_col, "type": "ordinal", "title": "Année"},
                "y": {"field": y_field, "type": "quantitative", "title": title, "axis": {"format": y_format}},
                "color": {"field": label_col, "type": "nominal", "scale": {"domain": sorted_labels, "range": color_range}, "title": "Territoire"},
                "tooltip": [{"field": label_col}, {"field": date_col}, {"field": y_field, "format": y_format, "title": title}]
            }
        }, use_container_width=True)
    else:
        st.vega_lite_chart(df_plot, {
            "mark": {"type": "bar", "cornerRadiusEnd": 4, "tooltip": True},
            "encoding": {
                "x": {"field": label_col, "type": "nominal", "sort": sorted_labels, "axis": {"labelAngle": -45}, "title": None},
                "y": {"field": y_field, "type": "quantitative", "title": title, "axis": {"format": y_format}},
                "color": {"field": label_col, "type": "nominal", "scale": {"domain": sorted_labels, "range": color_range}, "legend": None},
                "tooltip": [{"field": label_col}, {"field": y_field, "format": y_format, "title": title}]
            }
        }, use_container_width=True)

# --- 9. UI PRINCIPALE ---
st.title("🗺️ Terribot")
st.markdown("#### L'expert des données territoriales")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Bonjour ! Quel territoire souhaitez-vous analyser ?"}]
if "current_geo_context" not in st.session_state:
    st.session_state.current_geo_context = None

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

        # 3. GRAPHIQUE (Déplié / Visible directement)
        if "data" in msg and "ID" in msg["data"].columns:
            specs = msg.get("format_specs", {})
            sorted_ids = msg["data"]["ID"].tolist() 
            if "debug_info" in msg and "final_ids" in msg["debug_info"]:
                 sorted_ids = msg["debug_info"]["final_ids"]
            
            # Récupérer la métrique choisie par l'IA si dispo
            forced = msg.get("selected_metric", None)
            try: auto_plot_data(msg["data"], sorted_ids, format_specs=specs, forced_metric=forced)
            except: pass

        # 4. ANALYSE (Visible)
        st.markdown(msg["content"])


# --- 10. TRAITEMENT ---
inject_placeholder_animation()

if prompt := st.chat_input("Posez votre question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"): st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        debug_container = {} 
        
        try:
            # A. REFORMULATION
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
                    3. Si aucun lieu, réinjecte '{current_geo_name}'.
                    """},
                    {"role": "user", "content": f"Historique:\n{history_text}\n\nDernière question: {prompt}"}
                ]
            )
            rewritten_prompt = reformulation.choices[0].message.content
            debug_container["reformulation"] = f"Original: {prompt}\nReformulé: {rewritten_prompt}"
            
            # --- APPARENCE : Trace Reformulation ---
            with st.expander("🤔 Trace : Reformulation (IA)", expanded=False):
                st.write(f"**Question originale :** {prompt}")
                st.write(f"**Reformulée :** {rewritten_prompt}")
            # --------------------------------------

            # B. GEO SCOPE
            new_context = analyze_territorial_scope(con, rewritten_prompt)
            
            if new_context:
                st.session_state.current_geo_context = new_context
                message_placeholder.info(f"📍 **Périmètre :** {new_context['display_context']}")
                
                debug_container["geo_extraction"] = new_context["lieux_cites"]
                debug_container["geo_resolution"] = new_context["debug_search"]
                debug_container["final_ids"] = new_context["all_ids"]
                        
            elif not st.session_state.current_geo_context:
                message_placeholder.warning("⚠️ Je ne détecte pas de territoire. Précisez une ville.")
                st.stop()
            
            geo_context = st.session_state.current_geo_context

            # C. RAG
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
                    
                    # --- APPARENCE : Trace RAG ---
                    with st.expander("📚 Trace : Variables trouvées (RAG)", expanded=False):
                        st.text(glossaire_context)
                    # -----------------------------

            # D. SQL
            ids_sql = ", ".join([f"'{str(i)}'" for i in geo_context['all_ids']])
            parent_clause = geo_context.get('parent_clause', '')
            
            system_prompt = f"""
            Tu es Terribot.
            CONTEXTE DONNÉES :
            {glossaire_context}
            
            SCHEMA TABLE "TERRITOIRES" (alias t) :
            - "ID" (VARCHAR)
            - "NOM_COUV" (VARCHAR)
            
            MISSION : Répondre à "{rewritten_prompt}" via UNE SEULE requête SQL.
            
            RÈGLES CRITIQUES :
            1. PÉRIMÈTRE :
               - WHERE (t."ID" IN ({ids_sql}) {parent_clause})
               - 🚨 INTERDIT : Ne fais PAS de UNION ALL complexe.
            
            2. INTELLIGENCE DE DONNÉES :
               - 🚨 OBLIGATOIRE : Si tu sélectionnes des volumes, tu DOIS AUSSI sélectionner la Population Totale (table EVO, var POP_22) et calculer un ratio (ex: Taux pour 1000 hab).
               - Exemple : `TRY_CAST(d.crimes AS DOUBLE) / NULLIF(TRY_CAST(e.POP_22 AS DOUBLE), 0) * 1000 AS crimes_pour_1000_hab`.
            
            3. CASTING :
               - DONNÉES -> TRY_CAST("col" AS DOUBLE).

            4. GESTION DES SÉRIES TEMPORELLES (Années en colonnes) :
            Si tu dois transformer des colonnes (ex: pop_2010, pop_2011...) en lignes (axe temporel) :
            - UTILISE la clause `UNPIVOT` de DuckDB.
            - Syntaxe stricte : `UNPIVOT (val_col FOR name_col IN (col1, col2, ...))`
            - 🚨 CRITIQUE : Dans le `SELECT` qui contient le UNPIVOT, N'UTILISE PAS les alias des tables jointes (ex: ne fais pas `t.ID` mais juste `ID`). L'UNPIVOT absorbe les tables précédentes.
            - Ensuite, nettoie la colonne 'annee' (ex: REPLACE(annee, 'pop_', '')) et CAST la en INT.
            
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
            
            # --- APPARENCE : Trace SQL ---
            with st.expander("💻 Trace : Génération SQL (IA)", expanded=False):
                st.code(sql_query, language="sql")
            # -----------------------------

            # E. EXECUTION
            if con:
                try:
                    df = con.execute(sql_query).df()
                    
                    if not df.empty:
                        # STATUS POUR VIZ ET ANALYSE
                        with st.status("Génération de l'analyse et du graphique...", expanded=True) as status:
                            
                            st.write("🔍 Détection des formats (%, €, hab)...")
                            format_specs = infer_format_specs_with_ai(
                                df=df, 
                                question=rewritten_prompt, 
                                glossaire_context=glossaire_context,
                                client=client,
                                model=MODEL_NAME
                            )
                            
                            st.write("🧠 Choix de la variable graphique...")
                            selected_metric = select_best_metric_for_chart(
                                df=df, 
                                question=rewritten_prompt, 
                                client=client, 
                                model=MODEL_NAME
                            )
                            
                            st.write("📊 Création du graphique optimisé...")
                            auto_plot_data(
                                df, 
                                geo_context['all_ids'], 
                                format_specs=format_specs, 
                                forced_metric=selected_metric
                            )
                            
                            st.write("📝 Rédaction de la synthèse...")
                            analysis = client.chat.completions.create(
                                model=MODEL_NAME,
                                messages=[
                                    {"role": "system", "content": f"""
                                    Tu es Terribot. Analyse les résultats pour : {rewritten_prompt}.
                                    CONSIGNE : Sois très SYNTHÉTIQUE (max 10 lignes). Va droit au but (Oui/Non). Utilise des puces.
                                    Respecte les unités fournies dans FORMAT_SPECS : {json.dumps(format_specs, ensure_ascii=False)}
                                    """},
                                    {"role": "user", "content": df.to_string()}
                                ]
                            )
                            status.update(label="Terminé", state="complete", expanded=False)
                        
                        final_resp = analysis.choices[0].message.content
                        
                        # AFFICHAGE FINAL (Ordre demandé : Debug -> Tableau -> Graph -> Analyse)
                        # 1. Debug (déjà affiché progressivement, mais on garde le bloc historique final)
                        # Le bloc progressif est déjà visible au dessus. On n'affiche plus rien ici pour éviter les doublons immédiats.
                        
                        # 2. Tableau
                        with st.expander("📊 Voir les données brutes", expanded=False):
                            try:
                                st.dataframe(style_df(df, format_specs), use_container_width=True)
                            except:
                                st.dataframe(df, use_container_width=True)

                        # 3. Graphique (déjà affiché dans le status mais on le refait propre ici pour persistance)
                        auto_plot_data(
                            df, 
                            geo_context['all_ids'], 
                            format_specs=format_specs, 
                            forced_metric=selected_metric
                        )

                        # 4. Analyse
                        message_placeholder.markdown(final_resp)
                        
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": final_resp, 
                            "data": df,
                            "format_specs": format_specs,
                            "selected_metric": selected_metric,
                            "debug_info": debug_container
                        })
                    else:
                        message_placeholder.warning("Aucune donnée trouvée.")
                except Exception as e:
                    message_placeholder.error("Erreur technique.")
                    with st.expander("Debug Erreur"): st.write(e)

        except Exception as e:
            st.error(f"Erreur : {e}")