import streamlit as st
import streamlit.components.v1 as components
import openai
import duckdb
import pandas as pd  # <--- C'était l'import manquant
import os
import numpy as np
import json
import re
import unicodedata

print("[TERRIBOT] ✅ Script importé / démarrage du fichier")

# --- 0. SYSTÈME DE LOGS (A METTRE TOUT EN HAUT APRES LES IMPORTS) ---
import sys
import datetime
import os
import difflib

# Création du dossier de logs si inexistant
if not os.path.exists("logs"):
    os.makedirs("logs")

# Classe qui dédouble la sortie (Terminal + Fichier)
class DualLogger(object):
    def __init__(self):
        self.terminal = sys.stdout
        # Nom de fichier unique basé sur l'heure de lancement
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log = open(f"logs/session_{timestamp}.txt", "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Force l'écriture immédiate

    def flush(self):
        # Nécessaire pour la compatibilité système
        self.terminal.flush()
        self.log.flush()

# On redirige tout print() vers notre Logger
sys.stdout = DualLogger()

print(f"[TERRIBOT] 📝 Démarrage de l'enregistrement des logs")

def log_code_changes():
    """
    Compare le code actuel avec la dernière version exécutée.
    Log les différences (Ajouts/Suppressions) et met à jour le snapshot.
    """
    snapshot_path = "logs/.app_last_run.py.bak" # Fichier caché pour stocker l'état précédent
    current_file = __file__ # Le fichier app.py actuel
    
    # 1. Lire le code actuel
    try:
        with open(current_file, "r", encoding="utf-8") as f:
            current_code = f.readlines()
    except Exception:
        return # Si on ne peut pas lire le fichier, on abandonne

    # 2. Lire l'ancienne version (si elle existe)
    if os.path.exists(snapshot_path):
        with open(snapshot_path, "r", encoding="utf-8") as f:
            old_code = f.readlines()
        
        # 3. Calculer les différences
        diff = list(difflib.unified_diff(old_code, current_code, n=0))
        
        added = []
        removed = []
        
        for line in diff:
            # On ignore les en-têtes de diff (---, +++, @@)
            if line.startswith('---') or line.startswith('+++') or line.startswith('@@'):
                continue
            if line.startswith('+'):
                added.append(line[1:].strip()) # On enlève le "+"
            elif line.startswith('-'):
                removed.append(line[1:].strip()) # On enlève le "-"

        # 4. Écrire dans les logs SI changement
        if added or removed:
            print("\n" + "="*40)
            print("🛠️ CODE MODIFIÉ DPUIS LA DERNIÈRE EXÉCUTION")
            
            if removed:
                print("🔴 CODE SUPPRIMÉ :")
                for line in removed: print(f"   - {line}")
            
            if added:
                print("🟢 CODE AJOUTÉ :")
                for line in added: print(f"   + {line}")
            
            print("="*40 + "\n")
    else:
        # Première exécution : on ne log rien de spécial, ou on peut logger "Version Initiale"
        pass

    # 5. Mettre à jour le snapshot pour la prochaine fois
    try:
        with open(snapshot_path, "w", encoding="utf-8") as f:
            f.writelines(current_code)
    except Exception as e:
        print(f"⚠️ Impossible de sauvegarder le snapshot du code : {e}")

# --- 0. SYSTÈME DE LOGS/CODE ---
log_code_changes() 
# -----------------

def _dbg(label, **kw):
    try:
        payload = " ".join([f"{k}={repr(v)[:200]}" for k, v in kw.items()])
    except Exception:
        payload = "(payload error)"
    print(f"[TERRIBOT][DBG] {label} :: {payload}")

# --- 1. CONFIGURATION & STYLE (DOIT ÊTRE EN PREMIER) ---
st.set_page_config(
    page_title="Terribot | Assistant Territorial",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def standardize_name(name):
    """Nettoie un nom (fichier ou source) pour en faire un identifiant SQL valide et unique."""
    if not isinstance(name, str): return "UNKNOWN"
    # 1. On garde que les lettres et chiffres
    # 2. On remplace tout le reste par des underscores
    # 3. On passe en majuscule
    clean = re.sub(r'[^a-zA-Z0-9]', '_', name.upper())
    # 4. On évite les doubles underscores (ex: ACT__10 -> ACT_10)
    clean = re.sub(r'_+', '_', clean).strip('_')
    return clean

# --- 2. FONCTIONS DE DONNÉES ---
@st.cache_resource
def get_db_connection():
    # Connexion en mémoire
    print("[TERRIBOT][DB] ▶️ get_db_connection() ENTER")
    con = duckdb.connect(database=":memory:")
    
    # A. CHARGEMENT DYNAMIQUE DES DONNÉES (VUES)
    data_dir = "data" 
    if not os.path.exists(data_dir): os.makedirs(data_dir)
    _dbg("db.data_dir", data_dir=data_dir, exists=os.path.exists(data_dir))


    # On liste tous les parquets
    try:
        parquet_files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
        print(f"[TERRIBOT][DB] 📦 Parquets détectés: {len(parquet_files)} -> {parquet_files[:10]}")

        
        valid_tables = [] # Liste pour stocker les noms officiels

        schemas = {}

        for f in parquet_files:
            try:
                # 1. On nettoie le nom (ex: "act-10.parquet" -> "ACT_10")
                raw_name = f.replace('.parquet', '').upper()
                # On remplace tout ce qui n'est pas lettre/chiffre par _
                table_name = re.sub(r'[^A-Z0-9]', '_', raw_name)
                
                file_path = os.path.join(data_dir, f).replace("\\", "/")
                con.execute(f'CREATE OR REPLACE VIEW "{table_name}" AS SELECT * FROM \'{file_path}\'')
                
                # 2. On ajoute à la liste officielle
                valid_tables.append(table_name)
                # --- NOUVEAU : On récupère les colonnes réelles tout de suite ---
                cols_info = con.execute(f"DESCRIBE \"{table_name}\"").fetchall()
                # On stocke la liste des noms de colonnes pour cette table
                schemas[table_name] = [c[0] for c in cols_info]
                
            except Exception as e_file:
                print(f"❌ Erreur sur le fichier {f} : {e_file}")

        # 3. SAUVEGARDE GLOBALE (C'est la clé !)
        st.session_state.valid_tables_list = valid_tables
        st.session_state.db_schemas = schemas 
        print(f"[TERRIBOT][DB] 📋 Tables valides enregistrées : {len(valid_tables)}")

        print(f"[TERRIBOT][DB] 📦 {len(parquet_files)} vues créées.") # Résumé en une ligne

    except Exception as e:
        print(f"⚠️ Erreur listing dossier : {e}")

    # B. CHARGEMENT DES MÉTA-DONNÉES (Glossaire & Territoires)
    try:
        glossaire_path = os.path.join(data_dir, "Glossaire.txt").replace("\\", "/")
        territoires_path = os.path.join(data_dir, "territoires.txt").replace("\\", "/")
        _dbg("db.meta_paths", glossaire_path=glossaire_path, territoires_path=territoires_path,
        glossaire_exists=os.path.exists(glossaire_path), territoires_exists=os.path.exists(territoires_path))

        con.execute(f"""
            CREATE OR REPLACE TABLE glossaire AS 
            SELECT * FROM read_csv('{glossaire_path}', auto_detect=TRUE, ignore_errors=TRUE)
        """)
        
        con.execute(f"""
            CREATE OR REPLACE TABLE territoires AS 
            SELECT * FROM read_csv('{territoires_path}', auto_detect=TRUE, all_varchar=TRUE)
        """)
    except Exception as e:
        print(f"⚠️ Erreur chargement meta-fichiers : {e}")

    # C. INDEX FTS
    try:
        print("[TERRIBOT][DB] 🔎 FTS init...")
        con.execute("INSTALL fts; LOAD fts;")
        con.execute("PRAGMA create_fts_index('glossaire', 'Nom au sein de la base de données', 'Intitulé détaillé')")
        print("[TERRIBOT][DB] ✅ FTS index created on glossaire")
    except Exception as e:
        print(f"[TERRIBOT][DB] ⚠️ FTS init failed: {e}")
    print("[TERRIBOT][DB] ✅ get_db_connection() EXIT")
    return con

# On utilise la connexion définie tout en haut (Point A)
con = get_db_connection()

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
            "Compare le revenu médian à Bordeaux et à Toulouse",
            "Quel est le taux de chômage des jeunes à Marseille ?",
            "Quelle est la part des cadres à Neuilly-sur-Seine ?",
            "Compare la pauvreté à Roubaix avec la moyenne nationale",
            "Y a-t-il plus de propriétaires à Vannes ou à Lorient ?",
            "Quelle est la part des 15-24 ans à Rennes ?",
            "Compare le niveau de vie à Vincennes et Saint-Mandé",
            "Combien de résidences secondaires à La Rochelle ?",
            "Quel est le taux de bacheliers à Strasbourg ?",
            "Y a t il beaucoup de jeunes à Saint-Michel dans l'Aisne ?",
            "Compare la densité de population à Lyon et Villeurbanne",
            "La part des familles monoparentales à Saint-Denis",
            "Compare le chômage à Lens avec le département du Pas-de-Calais",
            "Quelle est la part de logements sociaux à Sarcelles ?",
            "Les revenus sont-ils plus élevés à Nantes ou à Angers ?",
            "Compare la part des seniors (65+) à Nice et Menton",
            "Quel est le taux d'activité des femmes à Lille ?",
            "Compare les non-diplômés à Maubeuge et Valenciennes",
            "Quelle est la taille moyenne des ménages à Paris ?",
            "Compare le revenu des habitants de Fontenay-sous-Bois aux villes voisines",
            "Quelle est la part des maisons à Brest ?"
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
    st.caption("v0.15 - 19 janvier 2026")
    st.divider()
    
    # Bouton Reset
    if st.button("🗑️ Nouvelle conversation", type="secondary", width='stretch'):
        st.session_state.messages = []
        st.session_state.messages = [{"role": "assistant", "content": "Bonjour ! Quel territoire souhaitez-vous analyser ?"}]
        st.session_state.current_geo_context = None
        st.session_state.pending_prompt = None
        st.session_state.ambiguity_candidates = None
        st.rerun()

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
        - **Filosofi** : Revenus & Pauvreté
        - **Sirene** : Entreprises
        """)
        
    st.info("💡 **Astuce :** L'IA choisit elle-même la variable du graphique selon votre question.")

client = openai.OpenAI(api_key=api_key)
MODEL_NAME = "gpt-5.2-2025-12-11"  # Mis à jour vers un modèle standard valide, ajustez si nécessaire
EMBEDDING_MODEL = "text-embedding-3-small"

# --- 4. FONCTIONS INTELLIGENTES (FORMATAGE & SÉLECTION) ---
def get_chart_configuration(df: pd.DataFrame, question: str, glossaire_context: str, client, model: str):
    """
    Fusionne la sélection des variables et la détection des formats et labels courts.
    """
    _dbg("chart.config.enter", question=question[:160], df_rows=len(df), df_cols=len(df.columns))

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c.upper() not in ["AN", "ANNEE", "YEAR", "ID", "CODGEO"]]
    _dbg("chart.numeric_cols", count=len(numeric_cols), sample=numeric_cols[:15])

    if not numeric_cols: return {"selected_columns": [], "formats": {}}

    stats = {}
    for c in numeric_cols[:10]:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(s) > 0: stats[c] = {"min": float(s.min()), "max": float(s.max())}

    payload = {
        "question": question,
        "available_columns": numeric_cols,
        "data_stats": stats,
        "glossaire_sample": (glossaire_context or "")[-2000:],
    }

    system_prompt = """
    Tu es un expert Dataviz. Configure le graphique.
    
    TA MISSION :
    1. Choisis la ou les colonnes ('selected_columns') pour répondre à la question.
       - Choisis toujours une seule variable pour répondre à la question. Priorité à la qualité du graph par rapport à la question.
       Le seul cas où tu peux choisir plusieurs variables : si tu veux faire une courbe, un histogramme groupé de plusieurs variables, ou un histogramme empilé avec plusieurs variables qui ont le même dénominateur et dont le total fait 100%.
       - Les valeurs absolues ne sont pas comparables entre deux territoires de tailles différentes (il faut des taux, des parts, des moyennes, des médianes).
    2. Définis le format ET un label court ('formats') pour chaque colonne.
       - 'label': Un nom très court pour l'axe du graphique (ex: "15-24 ans" au lieu de "part_pop_15_24").
       - 'title': Le titre complet pour l'infobulle (ex: "Part des 15-24 ans au sein de la population").
    3. Définis un TITRE GLOBAL pour le graphique ('chart_title').
       - Exemple : "Répartition des logements selon le DPE en 2025" ou "Évolution du chômage".
    
    JSON ATTENDU :
    {
      "selected_columns": ["col1", "col2"],
      "formats": {
        "col1": { "kind": "percent|currency|number", "decimals": 1, "label": "Titre Court Axe", "title": "Titre Long Tooltip" }
      }
    }
    """

    try:
        _dbg("chart.config.payload", payload_keys=list(payload.keys()), glossaire_tail=(payload["glossaire_sample"][-200:] if payload.get("glossaire_sample") else ""))

        resp = client.chat.completions.create(
            model=model, temperature=0, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
        )
        data = json.loads(resp.choices[0].message.content)
        _dbg("chart.config.raw", selected=data.get("selected_columns"), formats_keys=list((data.get("formats") or {}).keys())[:10])

        if not data.get("selected_columns"): data["selected_columns"] = [numeric_cols[0]]
        data["selected_columns"] = [c for c in data["selected_columns"] if c in df.columns]
        return data
    except: return {"selected_columns": [numeric_cols[0]], "formats": {}}

def style_df(df: pd.DataFrame, specs: dict):
    """Applique le formatage pour l'affichage (Styler)."""
    # On travaille sur une copie pour ne pas casser le DF original
    df_display = df.copy()
    
    # On force la conversion en numérique pour être sûr
    for col in df_display.columns:
        df_display[col] = pd.to_numeric(df_display[col], errors='ignore')

    def fr_num(x, decimals=0, suffix="", factor=1):
        if pd.isna(x): return "-" # Tiret pour les nulls
        if not isinstance(x, (int, float)): return str(x)
        try:
            val = x * factor
            # Format français (espace millier, virgule décimale)
            fmt = f"{{:,.{decimals}f}}"
            s = fmt.format(val).replace(",", " ").replace(".", ",")
            return (s + (f" {suffix}" if suffix else "")).strip()
        except: return str(x)

    # On prépare le dictionnaire de formatage
    format_dict = {}
    
    # On itère sur TOUTES les colonnes du tableau (et pas juste celles du graph)
    for col in df_display.columns:
        # On ignore les colonnes non numériques (Textes, IDs...)
        if not pd.api.types.is_numeric_dtype(df_display[col]): continue
        
        # On récupère la config IA si elle existe, sinon des valeurs par défaut
        s = specs.get(col, {})
        kind = (s.get("kind") or "number").lower()
        dec = int(s.get("decimals", 1)) # Par défaut 1 décimale
        
        # --- RÈGLE INTELLIGENTE : 0 décimale si tout est > 100 ---
        try:
            # On regarde les valeurs non nulles
            valid_vals = df_display[col].dropna().abs()
            if not valid_vals.empty:
                # Si la plus petite valeur est supérieure à 100 (ex: Pop, Revenus, Années)
                if valid_vals.min() >= 100:
                    dec = 0
                # Cas spécial pour les entiers parfaits (ex: nb d'écoles = 3.0 -> 3)
                elif (valid_vals % 1 == 0).all():
                    dec = 0
        except: pass
        # ---------------------------------------------------------

        if kind == "currency":
            format_dict[col] = lambda x, d=dec: fr_num(x, d, "€")
        elif kind == "percent":
            # Heuristique : Si c'est < 5 (ex: 0.15), on multiplie par 100.
            format_dict[col] = lambda x, d=dec: fr_num(x, d, "%", factor=100 if abs(x)<5 else 1) 
        else:
            format_dict[col] = lambda x, d=dec: fr_num(x, d, "")

    return df_display.style.format(format_dict)


    # --- NOUVEAU : FONCTION DE RÉPARATION SQL ---
def generate_and_fix_sql(client, model, system_prompt, user_prompt, con, max_retries=3):
    """Génère le SQL et tente de le corriger en injectant le schéma réel en cas d'erreur."""
    _dbg("sql.fix.enter", max_retries=max_retries)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    for attempt in range(max_retries + 1):
        print(f"[TERRIBOT][SQL] ▶️ attempt {attempt}/{max_retries}")
        
        try:
            # 1. Génération
            response = client.chat.completions.create(
                model=model, messages=messages, temperature=0
            )
            sql_query_raw = response.choices[0].message.content.replace("```sql", "").replace("```", "").strip()
            sql_query = sql_query_raw.split(";")[0]
            
            # 2. Vérification (Dry Run)
            con.execute(f"EXPLAIN {sql_query}")
            print("[TERRIBOT][SQL] ✅ EXPLAIN OK")
            return sql_query 
            
        except Exception as e:
            error_msg = str(e).split("\n")[0]
            print(f"[TERRIBOT][SQL] ❌ DuckDB error: {error_msg}")

            if attempt < max_retries:
                schema_hint = ""
                # On cherche si l'erreur mentionne une table ou un alias (ex: Values list "d"...)
                match_table = re.search(r'Values list "([^"]+)"', error_msg)
                
                if match_table:
                    faulty_ref = match_table.group(1)
                    real_table = faulty_ref
                    
                    # --- AMÉLIORATION : Résolution d'alias (ex: 'd' -> 'DIPL') ---
                    # On cherche dans le SQL un motif du type : JOIN "MA_TABLE" d  ou  JOIN MA_TABLE AS d
                    # Le regex cherche un mot (la table) juste avant l'alias fautif
                    alias_pattern = r'(?:FROM|JOIN)\s+(?:["\']?)([a-zA-Z0-9_\.\-]+)(?:["\']?)\s+(?:AS\s+)?\b' + re.escape(faulty_ref) + r'\b'
                    alias_match = re.search(alias_pattern, sql_query, re.IGNORECASE)
                    
                    if alias_match:
                        real_table = alias_match.group(1)
                        print(f"[TERRIBOT][SQL] 🕵️ Alias résolu : '{faulty_ref}' -> '{real_table}'")

                    try:
                        # On récupère les colonnes réelles de la VRAIE table
                        cols = con.execute(f"DESCRIBE \"{real_table}\"").fetchall()
                        col_names = [c[0] for c in cols]
                        # On donne un indice précis à l'IA
                        schema_hint = f"\n\n🚨 INFO CRITIQUE : L'alias '{faulty_ref}' correspond à la table '{real_table}'. Voici ses colonnes RÉELLES : {', '.join(col_names[:50])}..."
                    except Exception as desc_err:
                        print(f"[TERRIBOT][SQL] ⚠️ Impossible de décrire '{real_table}': {desc_err}")
                
                print("[TERRIBOT][SQL] 🛠️ Asking model to fix SQL with Schema Hint")
                messages.append({"role": "assistant", "content": sql_query})
                fix_prompt = f"Erreur DuckDB : {error_msg}. {schema_hint}\nCorrige la requête SQL. Utilise les guillemets doubles pour les noms de colonnes avec tirets. Ne réponds que le SQL."
                messages.append({"role": "user", "content": fix_prompt})
            else:
                raise e

# --- 5. FONCTIONS VECTORIELLES ---
@st.cache_resource
def get_glossary_embeddings(df_glossaire):
    print("[TERRIBOT][EMB] ▶️ get_glossary_embeddings ENTER")
    _dbg("emb.df_glossaire", empty=df_glossaire.empty, rows=len(df_glossaire), cols=list(df_glossaire.columns)[:8])

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
    _dbg("emb.cleaned", clean_texts_len=len(clean_texts), valid_indices_len=len(valid_indices))

    _dbg("emb.cache", cache_path=cache_path, cache_exists=os.path.exists(cache_path))
    if os.path.exists(cache_path):
        try:
            embeddings = np.load(cache_path)
            _dbg("emb.cache_loaded", embeddings_shape=getattr(embeddings, "shape", None))
            if len(embeddings) == len(clean_texts): return embeddings, valid_indices
        except: pass 

    all_embeddings = []
    BATCH_SIZE = 100 
    try:
        progress_bar = st.sidebar.progress(0, text="Chargement IA...")
        for i in range(0, len(clean_texts), BATCH_SIZE):
            batch = clean_texts[i : i + BATCH_SIZE]
            _dbg("emb.batch", i=i, batch_size=len(batch), total=len(clean_texts))
            response = client.embeddings.create(input=batch, model=EMBEDDING_MODEL)
            all_embeddings.extend([d.embedding for d in response.data])
            progress_bar.progress(min((i + BATCH_SIZE) / len(clean_texts), 1.0))
        progress_bar.empty()

        final_embeddings = np.array(all_embeddings)
        _dbg("emb.done", final_shape=getattr(final_embeddings, "shape", None))
        print("[TERRIBOT][EMB] ✅ embeddings ready")
        np.save(cache_path, final_embeddings)
        return final_embeddings, valid_indices
    except Exception as e:
        st.sidebar.error(f"Erreur IA: {e}")
        if os.path.exists(cache_path): os.remove(cache_path)
        return None, []

def semantic_search(query, df_glossaire, glossary_embeddings, valid_indices, top_k=80, threshold=0.38):
    if glossary_embeddings is None or df_glossaire.empty: return pd.DataFrame()
    _dbg("rag.semantic.enter", query=query[:120], top_k=top_k, threshold=threshold,
     emb_shape=getattr(glossary_embeddings, "shape", None), valid_indices_len=len(valid_indices))

    try:
        query_resp = client.embeddings.create(input=[query], model=EMBEDDING_MODEL)
        query_vec = np.array(query_resp.data[0].embedding)
        similarities = np.dot(glossary_embeddings, query_vec)
        
        df_results = df_glossaire.iloc[valid_indices].copy()
        min_len = min(len(df_results), len(similarities))
        df_results = df_results.iloc[:min_len]
        df_results['similarity'] = similarities[:min_len]
        _dbg("rag.semantic.sim", similarities_len=len(similarities), sim_max=float(np.max(similarities)) if len(similarities) else None)

        # 1. FILTRE PAR SEUIL (RAG Threshold)
        df_results = df_results[df_results['similarity'] > threshold]
        _dbg("rag.semantic.after_threshold", kept_rows=len(df_results))

        # 2. Filtres techniques (PAS DE BOOSTING ICI)
        mask_content = ~df_results.iloc[:, 4].astype(str).str.contains(r'IRIS|QPV', case=False, regex=True)
        _dbg("rag.semantic.return", final_rows=min(top_k, len(df_results)))

        return df_results[mask_content].sort_values('similarity', ascending=False).head(top_k)
    except: return pd.DataFrame()

def hybrid_variable_search(query, con, df_glossaire, glossary_embeddings, valid_indices, top_k=80):
    candidates = {} 
    _dbg("rag.hybrid.enter", query=query[:160], top_k=top_k)

    # 1. RECHERCHE VECTORIELLE
    df_sem = semantic_search(query, df_glossaire, glossary_embeddings, valid_indices, top_k=top_k, threshold=0.35)
    _dbg("rag.hybrid.semantic", sem_rows=len(df_sem))

    for _, row in df_sem.iterrows():
        var = row['Nom au sein de la base de données']
        candidates[var] = (row.get('similarity', 0.5), row)

    # 2. RECHERCHE FTS (DuckDB)
    clean_query = re.sub(r'[^a-zA-Z0-9àâäéèêëîïôöùûüç ]', '', query)
    try:
        keywords = [w for w in clean_query.split() if len(w) > 3]
        if keywords:
            search_phrase = " OR ".join([f"'{kw}'" for kw in keywords])
            sql_fts = f"""
                SELECT *
                FROM glossaire 
                WHERE match_bm25("Nom au sein de la base de données", {search_phrase}) IS NOT NULL 
                   OR match_bm25("Intitulé détaillé", {search_phrase}) IS NOT NULL
                LIMIT {top_k}
            """
            df_fts = con.execute(sql_fts).df()
            _dbg("rag.hybrid.fts", fts_rows=len(df_fts), keywords=keywords)

            for _, row in df_fts.iterrows():
                var = row['Nom au sein de la base de données']
                candidates[var] = (0.9, row) 
    except: pass

    # 3. CONSTRUCTION DU CONTEXTE (CORRIGÉ)
    sorted_vars = sorted(candidates.items(), key=lambda x: x[1][0], reverse=True)[:top_k]
    from difflib import get_close_matches

    valid_tables = st.session_state.get("valid_tables_list", [])
    db_schemas = st.session_state.get("db_schemas", {}) # <--- Récupération des schémas

    result_context = ""
    for var, (score, row) in sorted_vars:
        desc = row['Intitulé détaillé']
        raw_source = str(row.get('Onglet', row.iloc[1])).upper()
        
        # 1. Résolution de la TABLE (Code précédent)
        candidate_name = re.sub(r'[^A-Z0-9]', '_', raw_source)
        final_table_name = "UNKNOWN"
        
        if candidate_name in valid_tables:
            final_table_name = candidate_name
        else:
            matches = get_close_matches(candidate_name, valid_tables, n=1, cutoff=0.4)
            if matches: final_table_name = matches[0]
            else:
                for t in valid_tables:
                    if t in candidate_name or candidate_name in t:
                        final_table_name = t
                        break
        
        # 2. Résolution de la COLONNE (NOUVEAU & CRITIQUE)
        # Le glossaire dit "3-5_AUTREG", mais la base a peut-être "3_5_AUTREG"
        physical_column = var # Par défaut, on espère que c'est bon
        
        if final_table_name in db_schemas:
            real_cols = db_schemas[final_table_name]
            
            if var in real_cols:
                # C'est parfait, la colonne existe telle quelle
                physical_column = var
            else:
                # Aïe, le glossaire ment. On cherche la colonne réelle la plus proche.
                # On cherche d'abord une correspondance exacte insensible à la casse/tirets
                normalized_var = var.replace("-", "_").replace(".", "_").upper()
                
                found = False
                for rc in real_cols:
                    if rc.replace("-", "_").replace(".", "_").upper() == normalized_var:
                        physical_column = rc
                        found = True
                        break
                
                # Si toujours pas trouvé, on y va au Fuzzy Match dans la liste des colonnes
                if not found:
                    col_matches = get_close_matches(var, real_cols, n=1, cutoff=0.6)
                    if col_matches:
                        physical_column = col_matches[0]
                        # On loggue la correction pour info
                        # print(f"[RAG] 🔧 Correction colonne: {var} -> {physical_column}")

        # 3. Injection du nom PHYSIQUE dans le prompt
        # L'IA reçoit directement le nom qui marche. Plus besoin de deviner.
        result_context += f"✅ TABLE: \"{final_table_name}\" | VAR: \"{physical_column}\" | DESC: \"{desc}\"\n"

    _dbg("rag.hybrid.context", context_len=len(result_context))
    return result_context

# --- 6. MOTEUR DE DONNÉES (Unifié) ---


# On récupère le DataFrame du glossaire depuis DuckDB pour l'IA vectorielle
# (C'est le lien entre le monde SQL et le monde Vectoriel)
try:
    df_glossaire = con.execute("SELECT * FROM glossaire").df()
except Exception as e:
    st.error(f"Erreur chargement glossaire: {e}")
    df_glossaire = pd.DataFrame()

# Initialisation des embeddings (Ton code existant, adapté)
glossary_embeddings, valid_indices = None, []
if not df_glossaire.empty:
    glossary_embeddings, valid_indices = get_glossary_embeddings(df_glossaire)

# Création de la map pour le mapping rapide (code existant simplifié)
schema_map = {}
if not df_glossaire.empty:
    # On suppose que les colonnes sont : 'Nom au sein de la base de données' et 'Intitulé détaillé'
    # Adapte les indices si ton fichier change
    for _, row in df_glossaire.iterrows():
        var_name = str(row['Nom au sein de la base de données'])
        desc = str(row['Intitulé détaillé'])
        clean_key = var_name.lower().replace("-", "").replace("_", "")
        # On mappe vers (Table, Colonne) - ici on suppose que tout est dans 'data_act'
        schema_map[clean_key] = ("data_act", var_name)

# --- 7. INTELLIGENCE GÉOGRAPHIQUE ---
def clean_search_term(text):
    """Nettoie le terme de recherche pour ne garder que le nom géographique."""
    if not isinstance(text, str): return ""
    
    # 1. Normalisation unicode
    text = text.lower()
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")
    
    # 2. Remplacements standards
    text = text.replace('-', ' ').replace("'", " ").replace("’", " ")
        
    return text.strip()

def search_territory_smart(con, input_str):
    """
    Recherche intelligente avec priorité au Code Département si détecté.
    """
    _dbg("geo.search_smart.enter", input_str=input_str)

    clean_input = clean_search_term(input_str)
    if len(clean_input) < 2: return None

    # 1. Détection de Code Département (ex: "Fort-de-France 972")
    # On cherche s'il y a un nombre de 2 ou 3 chiffres à la fin ou au début
    dept_code = None
    match = re.search(r'\b(97\d|\d{2})\b', input_str)
    if match:
        dept_code = match.group(1)
        # On enlève le code du nom pour la recherche texte
        clean_input = clean_input.replace(dept_code, "").strip()
    _dbg("geo.search_smart.dept", dept_code=dept_code, clean_input=clean_input)

    # 2. Match Exact sur le Code INSEE (Priorité Absolue)
    if input_str.strip().isdigit():
        try:
            _dbg("geo.search_smart.sql", sql=("ID_exact" if input_str.strip().isdigit() else "strict_or_fuzzy"))
            res = con.execute(f"SELECT ID, NOM_COUV, COMP1, COMP2, COMP3 FROM territoires WHERE ID = '{input_str.strip()}' LIMIT 1").fetchone()
            if res: return res 
        except: pass

    # 3. Token Search (Mots clés) avec Filtre Département optionnel
    words = [w for w in clean_input.split() if len(w) > 1]
    if words:
        # Construction de la clause WHERE
        conditions = [f"strip_accents(lower(NOM_COUV)) LIKE '%{w}%'" for w in words]
        where_clause = " AND ".join(conditions)
        
        # AJOUT DU FILTRE DEPT SI DÉTECTÉ
        if dept_code:
            where_clause += f" AND ID LIKE '{dept_code}%'"

        sql_strict = f"""
        SELECT ID, NOM_COUV, COMP1, COMP2, COMP3
        FROM territoires WHERE {where_clause}
        ORDER BY length(NOM_COUV) ASC LIMIT 5
        """
        try:
            _dbg("geo.search_smart.sql", sql=("ID_exact" if input_str.strip().isdigit() else "strict_or_fuzzy"))
            results = con.execute(sql_strict).fetchall()
            print(f"[TERRIBOT][GEO] ✅ search_territory_smart results: {len(results)}")

            if len(results) == 1: return results[0] 
            if len(results) > 1: return results
        except: pass

    # 4. Fuzzy Search (Jaro-Winkler) - Seulement si pas de dept_code (trop risqué sinon)
    if not dept_code:
        sql_fuzzy = f"""
        WITH clean_data AS (
            SELECT ID, NOM_COUV, COMP1, COMP2, COMP3,
            lower(replace(replace(replace(NOM_COUV, '-', ' '), '''', ' '), '’', ' ')) as nom_simple
            FROM territoires
        )
        SELECT ID, NOM_COUV, COMP1, COMP2, COMP3,
        jaro_winkler_similarity(nom_simple, '{clean_input}') as score
        FROM clean_data
        WHERE score > 0.88 
        ORDER BY score DESC LIMIT 5
        """
        try:
            _dbg("geo.search_smart.sql", sql=("ID_exact" if input_str.strip().isdigit() else "strict_or_fuzzy"))
            results = con.execute(sql_fuzzy).fetchall()
            if not results: return None
            top_score = results[0][5]
            candidates = [r for r in results if (top_score - r[5]) < 0.05]
            if len(candidates) == 1: return candidates[0][:5]
            print(f"[TERRIBOT][GEO] ✅ search_territory_smart results: {len(results)}")
            return [c[:5] for c in candidates]
        except: pass
    
    return None

def get_broad_candidates(con, input_str, limit=15):
    """
    Récupère une liste large de candidats potentiels via DuckDB (FTS + Fuzzy).
    """
    _dbg("geo.broad_candidates.enter", input_str=input_str, limit=limit)
    clean_input = clean_search_term(input_str)
    
    # SQL : On cherche large (Fuzzy + Contient)
    # CORRECTION CRITIQUE : Séparation des types dans le CASE pour éviter 'Conversion Error'
    sql = f"""
    WITH candidates AS (
        SELECT 
            ID, 
            NOM_COUV, 
            COMP1, COMP2, COMP3,
            CASE 
                WHEN length(ID) IN (4,5) THEN 'Commune'
                WHEN length(ID) = 9 THEN 'EPCI/Interco'
                WHEN ID = 'FR' THEN 'Pays'
                -- CORRECTION ICI : On vérifie d'abord la longueur, PUIS les IDs spécifiques textuels
                WHEN length(ID) IN (2,3) OR ID IN ('D971','D972','D973','D974','D976') THEN 'Département'
                -- AJOUT : Gestion explicite des Régions (souvent Rxx ou longueur 2/3 selon la source)
                WHEN ID LIKE 'R%' THEN 'Région' 
                ELSE 'Autre'
            END as TYPE_TERRITOIRE,
            
            -- BOOST DE SCORE :
            -- 1. Base Jaro-Winkler
            jaro_winkler_similarity(lower(NOM_COUV), '{clean_input}') 
            -- 2. Bonus si c'est une Région ou un Département (IDs courts ou R..)
            + (CASE WHEN length(ID) <= 3 OR ID LIKE 'R%' THEN 0.15 ELSE 0 END) 
            -- 3. Bonus Match Exact
            + (CASE WHEN lower(NOM_COUV) = '{clean_input}' THEN 0.3 ELSE 0 END)
            as score
        FROM territoires
        WHERE strip_accents(lower(NOM_COUV)) LIKE '%{clean_input}%' 
           OR jaro_winkler_similarity(lower(NOM_COUV), '{clean_input}') > 0.85
    )
    SELECT * FROM candidates 
    ORDER BY score DESC 
    LIMIT {limit}
    """
    
    try:
        _dbg("geo.broad_candidates.sql", sql_preview=sql[:300])
        df_candidates = con.execute(sql).df()
        _dbg("geo.broad_candidates.result", rows=len(df_candidates), sample=df_candidates.head(3).to_dict(orient="records"))

        return df_candidates.to_dict(orient='records')
    except Exception as e:
        print(f"❌ Erreur SQL Candidates: {e}")
        return []

def ai_validate_territory(client, model, user_query, candidates, full_sentence_context=""):
    """
    Demande à l'IA de choisir le meilleur code INSEE parmi les candidats.
    """
    _dbg("geo.ai_validate.enter", user_query=user_query, candidates_len=len(candidates))

    if not candidates: return None

    system_prompt = """
    Tu es un expert géographe rattaché au code officiel géographique (INSEE).
    
    TA MISSION :
    Identifier le territoire unique qui correspond à la recherche de l'utilisateur parmi une liste de candidats.
    
    RÈGLES DE DÉCISION :
    1. Si l'utilisateur tape juste le nom d'une ville (ex: "Dunkerque"), c'est TOUJOURS la "Commune" (ID 4 ou 5 chiffres). Pas l'EPCI.
    2. Si l'utilisateur précise "Agglo", "Metropole", "Communauté", "CU", "Grand...", c'est l'"EPCI/Interco" (ID 9 chiffres).
    3. Si l'utilisateur tape un numéro (ex: "59"), c'est le Département.
    4. En cas de doute total (ex: homonymes parfaits dans deux départements sans contexte), renvoie "AMBIGUITE".
    
    FORMAT DE RÉPONSE JSON ATTENDU :
    {
        "selected_id": "code_insee_choisi" OU null,
        "reason": "explication courte",
        "is_ambiguous": true/false
    }
    """

    user_message = f"""
    CONTEXTE GLOBAL (Phrase utilisateur) : "{full_sentence_context}"
    
    TERME RECHERCHÉ ACTUELLEMENT : "{user_query}"
    
    Candidats trouvés en base pour "{user_query}" :
    {json.dumps(candidates, ensure_ascii=False, indent=2)}
    """

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        _dbg("geo.ai_validate.exit", raw=response.choices[0].message.content[:400])

        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return None

def analyze_territorial_scope(con, rewritten_prompt):
    # 1. Extraction (inchangé)
    try:
        extraction = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Extrais les lieux géographiques exacts. JSON: {\"lieux\": [\"Lieu 1\", \"Lieu 2\"]}"},
                {"role": "user", "content": rewritten_prompt}
            ],
            response_format={"type": "json_object"}
        )
        lieux_cites = json.loads(extraction.choices[0].message.content).get("lieux", [])
    except: return None

    if not lieux_cites: return None
    
    # --- CORRECTION : BOUCLE SUR TOUS LES LIEUX ---
    found_ids = []
    target_name = None
    target_id = None
    debug_info = []

    # On considère le premier cité comme le "Sujet Principal" pour le titre
    first_pass = True

    for lieu in lieux_cites:
        # Recherche large pour CE lieu
        candidates = get_broad_candidates(con, lieu)
        
        # Validation IA pour CE lieu
        ai_decision = ai_validate_territory(client, MODEL_NAME, lieu, candidates, full_sentence_context=rewritten_prompt)
        
        if ai_decision and ai_decision.get("selected_id"):
             sel_id = ai_decision["selected_id"]
             # On retrouve le nom officiel
             winner = next((c for c in candidates if str(c['ID']).lstrip('0') == str(sel_id).lstrip('0')), None)
             
             if winner:
                 found_ids.append(str(winner['ID']))
                 debug_info.append({"Recherche": lieu, "Trouvé": winner['NOM_COUV'], "ID": winner['ID']})
                 
                 # Si c'est le premier (Grande-Synthe), on garde ses infos comme "Target"
                 if first_pass:
                     target_id = str(winner['ID'])
                     target_name = winner['NOM_COUV']
                     # On ajoute aussi ses parents (EPCI, Dept) pour le contexte global
                     comps = [str(winner[c]) for c in ['COMP1', 'COMP2', 'COMP3'] if winner[c]]
                     found_ids.extend(comps)
                     first_pass = False

    # Si on n'a rien trouvé du tout
    if not found_ids: return None

    # On ajoute toujours France pour référence
    found_ids.append('FR')
    
    # Dédoublonnage propre
    unique_ids = list(dict.fromkeys([x for x in found_ids if x]))

    return {
        "target_name": target_name or lieux_cites[0],
        "target_id": target_id or unique_ids[0],
        "all_ids": unique_ids, # <--- Contient maintenant Grande-Synthe, Dunkerque ET Coudekerque
        "parent_clause": "", 
        "display_context": ", ".join(lieux_cites),
        "debug_search": debug_info,
        "lieux_cites": lieux_cites
    }

# --- 8. VISUALISATION AUTO (HEURISTIQUE %) ---
def auto_plot_data(df, sorted_ids, config=None, con=None):
    if config is None: config = {}
    _dbg("plot.enter", df_rows=len(df), df_cols=list(df.columns)[:12], sorted_ids=sorted_ids[:10], selected_metrics=(config or {}).get("selected_columns"))

    selected_metrics = config.get("selected_columns", [])
    format_specs = config.get("formats", {})
    
    color_range = ["#EB2C30", "#F38331", "#97D422", "#1DB5C5", "#5C368D"]
    
    cols = df.columns.tolist()
    label_col = next((c for c in cols if c.upper() in ["NOM_COUV", "TERRITOIRE", "LIBELLE", "VILLE"]), None)
    date_col = next((c for c in cols if c.upper() in ["AN", "ANNEE", "YEAR", "DATE"]), None)
    id_col = next((c for c in cols if c.upper() == "ID"), None)
    _dbg("plot.detect_cols", label_col=label_col, date_col=date_col, id_col=id_col)

    if not selected_metrics or not label_col: return

    # 0. PREP
    df_plot = df.copy()
    if id_col: df_plot[id_col] = df_plot[id_col].astype(str)
    
    # 1. LOGIQUE TOP 5 (Pertinence)
    available_ids_in_data = df_plot[id_col].unique().tolist()
    top_5_ids = [str(x) for x in sorted_ids if str(x) in available_ids_in_data][:5]
    if not top_5_ids: top_5_ids = available_ids_in_data[:5]

    # 2. TRI VISUEL (TAILLE)
    final_display_order = []
    
# 2. TRI VISUEL (Cible en premier, puis les autres par taille)
    final_display_order = []
    
    # On récupère les IDs disponibles dans la data
    available_ids = df_plot[id_col].unique().tolist()
    
    # On filtre la liste d'entrée pour ne garder que ceux qui ont des données
    candidates = [str(x) for x in sorted_ids if str(x) in available_ids]
    if not candidates: candidates = available_ids # Fallback

    if candidates:
        # A. La Cible est TOUJOURS le premier élément de la liste 'candidates'
        # (car analyze_territorial_scope met toujours le target_id en premier)
        target_id = candidates[0]
        
        # B. Les Comparateurs sont le reste de la liste
        comparators = candidates[1:] 
        
        # C. On ne trie QUE les comparateurs
        if con and comparators:
            try:
                valid_tables = st.session_state.get("valid_tables_list", [])
                evo_table = next((t for t in valid_tables if "EVO" in t), None)
                
                if evo_table:
                    # Recherche colonne POP (inchangée)
                    evo_cols = [c[0].upper() for c in con.execute(f"DESCRIBE \"{evo_table}\"").fetchall()]
                    pop_candidates = [c for c in evo_cols if "POP" in c and any(char.isdigit() for char in c)]
                    pop_col = sorted(pop_candidates)[-1] if pop_candidates else None
                    if not pop_col: pop_col = next((c for c in evo_cols if c in ["POP", "PMUN", "PTOT", "POPULATION"]), None)

                    if pop_col:
                        ids_sql = ", ".join([f"'{i}'" for i in comparators])
                        # Tri Ascendant des comparateurs uniquement
                        q_sort = f"""
                            SELECT t.ID 
                            FROM territoires t
                            LEFT JOIN "{evo_table}" e ON t.ID = e.ID
                            WHERE t.ID IN ({ids_sql}) 
                            ORDER BY TRY_CAST(e."{pop_col}" AS DOUBLE) ASC
                        """
                        # On récupère la liste triée
                        comparators = [str(x[0]) for x in con.execute(q_sort).fetchall()]
                        _dbg("plot.sort", status="success", col=pop_col)
            except Exception as e:
                _dbg("plot.sort", status="error", msg=str(e))

        # D. Assemblage Final : Cible + Comparateurs triés
        final_display_order = [target_id] + comparators
        
        # E. On coupe à 15 pour la lisibilité
        final_display_order = final_display_order[:15]
        
        # F. Application du filtre
        df_plot = df_plot[df_plot[id_col].isin(final_display_order)]

    # 3. RENOMMAGE
    rename_map = {}
    new_selected_metrics = []
    for m in selected_metrics:
        short_label = format_specs.get(m, {}).get("label", m)
        rename_map[m] = short_label
        new_selected_metrics.append(short_label)
    if rename_map: df_plot = df_plot.rename(columns=rename_map)

    # 4. TRI DF
    if id_col:
        id_order_map = {str(uid): i for i, uid in enumerate(final_display_order)}
        df_plot['sort_order'] = df_plot[id_col].map(id_order_map)
        df_plot = df_plot.sort_values('sort_order').drop(columns=['sort_order'])
    sorted_labels = df_plot[label_col].unique().tolist() 

    # 5. FORMATS & CONFIG
    original_metric = selected_metrics[0]
    spec = format_specs.get(original_metric, {})
    title_y = spec.get("title", spec.get("label", "Valeur"))
    
    y_format = ",.1f"
    is_percent = spec.get("kind") == "percent"
    if is_percent: y_format = ".1%"
    elif spec.get("kind") == "currency": y_format = ",.0f"

    # 6. MELT
    id_vars = [label_col]
    if date_col: id_vars.append(date_col)
    df_melted = df_plot.melt(id_vars=id_vars, value_vars=new_selected_metrics, var_name="Indicateur", value_name="Valeur")
    
    # 7. HEURISTIQUE DE CORRECTION DU % (1600% -> 16%)
    if is_percent:
        # Si c'est censé être du % mais que la moyenne des valeurs est > 1.5, 
        # c'est que les données sont en base 100 (ex: 15.5) et pas en base 1 (0.155)
        # Vega attend du base 1 pour afficher %. On divise donc par 100.
        val_mean = df_melted["Valeur"].mean()
        if val_mean > 1.5:
             df_melted["Valeur"] = df_melted["Valeur"] / 100.0

    # 8. VEGA
    vega_config = {
        "locale": {"number": {"decimal": ",", "thousands": "\u00a0", "grouping": [3]}},
        "axis": {"labelFontSize": 11, "titleFontSize": 12},
        "legend": {"labelFontSize": 11, "titleFontSize": 12, "orient": "bottom", "layout": {"bottom": {"anchor": "middle"}}}
    }
    color_def = {
        "field": label_col, 
        "type": "nominal", 
        "scale": {"domain": sorted_labels, "range": color_range}, 
        "title": "",
        "legend": {"orient": "bottom"}
    }
    chart = None
    is_multi_metric = len(new_selected_metrics) > 1

    if date_col:
        chart_encoding = {
            "x": {"field": date_col, "type": "ordinal", "title": "Année"},
            "y": {"field": "Valeur", "type": "quantitative", "title": "", "axis": {"format": y_format}},
            "color": color_def,
            "tooltip": [{"field": label_col}, {"field": "Indicateur", "title": "Variable"}, {"field": date_col}, {"field": "Valeur", "format": y_format}]
        }
        if is_multi_metric: chart_encoding["strokeDash"] = {"field": "Indicateur", "title": "Variable"}
        chart = {"config": vega_config, "mark": {"type": "line", "point": True, "tooltip": True}, "encoding": chart_encoding}
    else:
        if is_multi_metric:
             chart_encoding = {
                "x": {"field": "Indicateur", "type": "nominal", "axis": {"labelAngle": 0, "title": None}},
                "y": {"field": "Valeur", "type": "quantitative", "title": "", "axis": {"format": y_format}},
                "color": color_def,
                "xOffset": {"field": label_col},
                "tooltip": [{"field": label_col}, {"field": "Indicateur", "title": "Variable"}, {"field": "Valeur", "format": y_format}]
            }
        else:
            chart_encoding = {
                "x": {"field": label_col, "type": "nominal", "sort": sorted_labels, "axis": {"labelAngle": 0}, "title": None, "labelLimit": 1000},  # <--- CORRECTION 1 : Affiche le nom complet (jusqu'à 500px)
                "y": {"field": "Valeur", "type": "quantitative", "title": "", "axis": {"format": y_format}},
                "color": color_def,
                "tooltip": [{"field": label_col}, {"field": "Valeur", "format": y_format}]
            }
        chart = {"config": vega_config, "mark": {"type": "bar", "cornerRadiusEnd": 3, "tooltip": True}, "encoding": chart_encoding}

    _dbg("plot.vega", melted_rows=len(df_melted), is_percent=is_percent, is_multi_metric=is_multi_metric, y_format=y_format)
    chart["title"] = {
        "text": f"{title_y}",
        "anchor": "middle", 
        "fontSize": 16,
        "offset": 10
    }
    st.vega_lite_chart(df_melted, chart, width='stretch')


# --- 9. UI PRINCIPALE ---
st.title("🗺️ Terribot")
st.markdown("#### L'expert des données territoriales")

# Initialisation des variables de session pour l'ambiguïté
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Bonjour ! Quel territoire souhaitez-vous analyser ?"}]
if "current_geo_context" not in st.session_state:
    st.session_state.current_geo_context = None
if "force_geo_context" not in st.session_state:
    st.session_state.force_geo_context = False
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None
if "pending_geo_text" not in st.session_state:
    st.session_state.pending_geo_text = None
if "ambiguity_candidates" not in st.session_state:
    st.session_state.ambiguity_candidates = None

for msg in st.session_state.messages:
    avatar = "🤖" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        
        # 1. TEXTE
        st.markdown(msg["content"])
        
        # 2. DEBUG COMPLET (Reconstitué)
        # On cherche la liste d'étapes "steps" qu'on a sauvegardée
        debug_steps = msg.get("debug_info", {}).get("steps", [])
        
        if debug_steps:
            with st.expander("🧠 Trace de raisonnement (Terminé)", expanded=False):
                for step in debug_steps:
                    col_icon, col_txt = st.columns([1, 15])
                    with col_icon: st.write(step['icon'])
                    with col_txt:
                        st.markdown(f"**{step['label']}**")
                        if step['type'] == 'text':
                            st.caption(step['content'])
                        elif step['type'] == 'code':
                            st.code(step['content'], language="sql")
                        elif step['type'] == 'json':
                            st.json(step['content'])
                        elif step['type'] == 'table':
                            st.dataframe(pd.DataFrame(step['content']), hide_index=True)
                    st.divider()
        
        # Fallback pour les anciens messages (compatibilité)
        elif "debug_info" in msg and msg["debug_info"]:
             with st.expander("🔧 Détails techniques (Ancien)", expanded=False):
                 st.write(msg["debug_info"])

        # 3. GRAPHIQUE & DATA (Reste identique)
        if "data" in msg and not msg["data"].empty:
            try:
                # --- CORRECTION ---
                # On essaie de récupérer la config complète sauvegardée
                saved_config = msg.get("chart_config")
                
                # Fallback (rétro-compatibilité pour vos anciens messages de la session en cours)
                if not saved_config:
                    specs = msg.get("format_specs", {})
                    # C'est cette ligne qui cassait vos graphs en ne prenant que [0]
                    col = msg.get("selected_metric")
                    saved_config = {"selected_columns": [col] if col else [], "formats": specs}
                
                final_ids = msg.get("debug_info", {}).get("final_ids", [])
                
                # Affichage Graphique avec la BONNE config
                auto_plot_data(msg["data"], final_ids, config=saved_config, con=con)
                
                # Affichage Data (Expander)
                with st.expander("📊 Données brutes"):
                    # On utilise les formats stockés dans la config
                    formats = saved_config.get("formats", {})
                    st.dataframe(style_df(msg["data"], formats), width='stretch')
            except Exception as e: 
                pass
            
# --- 10. TRAITEMENT ET GESTION AMBIGUÏTÉ ---
inject_placeholder_animation()

# Initialisation de la variable de déclenchement si elle n'existe pas
if "trigger_run_prompt" not in st.session_state:
    st.session_state.trigger_run_prompt = None

# -- A. RÉSOLUTION D'AMBIGUÏTÉ (Affichage des boutons si nécessaire) --
if st.session_state.ambiguity_candidates:
    _dbg("ui.ambiguity.render", candidates=st.session_state.ambiguity_candidates)
    
    st.warning(f"🤔 Plusieurs territoires trouvés pour '{st.session_state.get('pending_geo_text','ce lieu')}'. Veuillez préciser :")
    cols = st.columns(min(len(st.session_state.ambiguity_candidates), 4))
    
    for i, cand in enumerate(st.session_state.ambiguity_candidates[:4]):
        # On affiche le bouton
        if cols[i].button(f"{cand['nom']} ({cand['id']})", key=f"amb_btn_{cand['id']}"):
            print("[TERRIBOT][UI] ✅ User selected ambiguity candidate")
            _dbg("ui.ambiguity.choice", cand=cand)

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
            _dbg("ui.ambiguity.context_set", current_geo_context=st.session_state.current_geo_context)
            print("[TERRIBOT][UI] 🔁 rerun after ambiguity resolution")

            st.rerun()

# -- B. INPUT PRINCIPAL --
user_input = st.chat_input("Posez votre question...")

# -- C. LOGIQUE DE DÉCISION (Quel prompt traiter ?) --
prompt_to_process = None

was_trigger = bool(st.session_state.trigger_run_prompt)

# Priorité 1 : On vient de cliquer sur un bouton (variable stockée en session)
if st.session_state.trigger_run_prompt:
    prompt_to_process = st.session_state.trigger_run_prompt
    st.session_state.trigger_run_prompt = None # On consomme le trigger pour ne pas boucler

# Priorité 2 : L'utilisateur vient de taper une nouvelle question
elif user_input:
    prompt_to_process = user_input

# --- D. EXÉCUTION DU TRAITEMENT ---
if prompt_to_process:
    print("[TERRIBOT] ===============================")
    _dbg("pipeline.start", prompt_to_process=prompt_to_process, from_trigger=was_trigger)

    _dbg("session.state", has_geo=bool(st.session_state.current_geo_context),
        ambiguity=bool(st.session_state.ambiguity_candidates),
        messages=len(st.session_state.messages))

    # Si c'est un nouvel input utilisateur, on l'ajoute à l'historique
    # (On vérifie pour éviter les doublons lors de la reprise après ambiguïté)
    last_msg = st.session_state.messages[-1] if st.session_state.messages else {}
    if last_msg.get("content") != prompt_to_process or last_msg.get("role") != "user":
        st.session_state.messages.append({"role": "user", "content": prompt_to_process})
        with st.chat_message("user", avatar="👤"): st.markdown(prompt_to_process)

# Réponse Assistant
    with st.chat_message("assistant", avatar="🤖"):
            # Placeholders pour l'affichage progressif
            # 1. DÉFINITION DE L'ORDRE D'AFFICHAGE (Haut -> Bas)
            chart_placeholder = st.empty()   # Le graphique en haut
            data_placeholder = st.empty()    # Les données au milieu (NOUVEAU)
            message_placeholder = st.empty() # Le texte en bas


            # --- 🛑 MODIFICATION ICI : LE CONTENEUR JETABLE ---    
            loader_placeholder = st.empty()  # Un placeholder dédié pour le chargement
            
            # On crée le statut À L'INTÉRIEUR de ce placeholder
            with loader_placeholder:
                status_container = st.status("J'analyse votre demande...", expanded=False)
            
            # Le reste des initialisations reste inchangé...
            debug_container = {}
            debug_steps = []
            debug_container["steps"] = debug_steps

            full_response_text = ""
            df = pd.DataFrame()
            chart_config = {}
        
            try:
                with status_container:
                    # 1. REFORMULATION
                    status_container.update(label="Je reformule pour bien comprendre...")
                    history_text = "\n".join([f"{m['role']}: {m.get('content','')}" for m in st.session_state.messages[-4:]])
                    current_geo_name = st.session_state.current_geo_context['target_name'] if st.session_state.current_geo_context else ""

                    _dbg("pipeline.rewrite.call", history_tail=history_text[-400:], current_geo_name=current_geo_name)

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
                    _dbg("pipeline.rewrite.done", rewritten_prompt=rewritten_prompt)

                    debug_container["reformulation"] = f"Original: {prompt_to_process}\nReformulé: {rewritten_prompt}"
                    
                    with st.expander("🤔 Trace : Reformulation (IA)", expanded=False):
                        st.write("🔄 Compréhension...")
                        st.write(f"**Question originale :** {prompt_to_process}")
                        st.write(f"**Reformulée :** {rewritten_prompt}")

                    # 2. GEO SCOPE
                    new_context = None
                    status_container.update(label="🌍 Je recherche les territoires mentionnés...")
                    _dbg("pipeline.geo.before", force_geo_context=bool(st.session_state.get("force_geo_context")),
                        current_geo=st.session_state.current_geo_context.get("target_name") if st.session_state.current_geo_context else None)

                    # --- MODIFICATION ICI : Gestion du Verrou ---
                    if st.session_state.get("force_geo_context"):
                        st.session_state.force_geo_context = False # On consomme le verrou
                        print("[TERRIBOT][PIPE] 🔒 force_geo_context consumed -> keep existing context")
                        _dbg("pipeline.geo.locked_context", geo=st.session_state.current_geo_context)

                        # On ne lance PAS analyze_territorial_scope, on garde l'existant
                        if st.session_state.current_geo_context:
                            geo_context = st.session_state.current_geo_context
                            message_placeholder.info(f"📍 **Périmètre validé :** {geo_context['display_context']}")
                            # On force new_context à None pour sauter les blocs suivants
                            new_context = None 
                    else:
                        # Analyse normale
                        print("[TERRIBOT][PIPE] 🌍 analyze_territorial_scope() running")

                        new_context = analyze_territorial_scope(con, rewritten_prompt)
                        _dbg("pipeline.geo.after", new_context=new_context)

                        
                    # --- GESTION DE L'AMBIGUÏTÉ DÉTECTÉE ---
                    # Si une ambiguïté est détectée ET que ce n'est pas le contexte qu'on vient juste de forcer
                    if new_context and new_context.get("ambiguity"):
                        # Petite sécurité : si le lieu ambigu est le même que celui qu'on a déjà validé, on ignore l'ambiguïté
                        if st.session_state.current_geo_context and new_context['input_text'] in st.session_state.current_geo_context['target_name']:
                            pass # On garde le contexte actuel
                        else:
                            # On stocke l'état et on arrête l'exécution pour afficher les boutons au prochain tour
                            print("[TERRIBOT][PIPE] ⚠️ Ambiguity flow triggered -> storing candidates + rerun")
                            _dbg("pipeline.ambiguity", input_text=new_context.get("input_text"), candidates=new_context.get("candidates"))

                            st.session_state.ambiguity_candidates = new_context['candidates']
                            st.session_state.pending_geo_text = new_context.get("input_text")

                            
                            st.session_state.pending_prompt = prompt_to_process
                            print("[TERRIBOT][BUG?] debug_steps referenced here — is it defined in this scope?")

                            debug_container["steps"] = debug_steps # <--- SAUVEGARDE
                            debug_container["final_ids"] = (st.session_state.current_geo_context or {}).get("all_ids", [])

                            st.session_state.messages.append({"role": "assistant", "content": f"🤔 J'ai un doute sur le lieu **{new_context['input_text']}**. Veuillez choisir ci-dessus."})
                            st.rerun()

                    # Mise à jour du contexte si un nouveau lieu valide est trouvé
                    if new_context and not new_context.get("ambiguity"):
                        st.session_state.current_geo_context = new_context
                        _dbg("pipeline.geo.context_set", geo=st.session_state.current_geo_context)

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
                    # --- CORRECTION ICI : ON FORCE LA SAUVEGARDE DES IDS ---
                    # Cela garantit que le graphique pourra être reconstruit depuis l'historique
                    if new_context:
                     # <--- APPEND STEP
                        debug_steps.append({"icon": "🔎", "label": "Résolution Géo", "type": "table", "content": new_context["debug_search"]})
                    if geo_context:
                        debug_container["final_ids"] = geo_context['all_ids']
                    # -------------------------------------------------------
                    # 3. RAG (Recherche Variables - Méthode Hybride)
                    status_container.update(label="📚 Je cherche les indicateurs pertinents dans le glossaire...")
                    # On appelle notre nouvelle fonction combinée
                    print("[TERRIBOT][PIPE] 📚 RAG hybrid_variable_search() start")
                    _dbg("pipeline.rag.inputs", rewritten_prompt=rewritten_prompt[:200], df_glossaire_rows=len(df_glossaire))

                    glossaire_context = hybrid_variable_search(
                        rewritten_prompt, 
                        con, 
                        df_glossaire, 
                        glossary_embeddings, 
                        valid_indices
                    )
                    _dbg("pipeline.rag.done", glossaire_context_len=len(glossaire_context), preview=glossaire_context[:400])

                    # Debugging visuel
                    debug_container["rag_context"] = glossaire_context
                    with st.expander("📚 Trace : Variables identifiées", expanded=False):
                        st.text(glossaire_context)
                        
                    if not glossaire_context:
                        # Fallback si rien n'est trouvé
                        glossaire_context = "Aucune variable spécifique trouvée. Essaie d'utiliser des connaissances générales ou signale l'absence de données."

                    # 4. SQL GENERATION
                    ids_sql = ", ".join([f"'{str(i)}'" for i in geo_context['all_ids']])
                    parent_clause = geo_context.get('parent_clause', '')
                    status_container.update(label="🔢 Je récupère les données chiffrées...")
                    system_prompt = f"""
                    Tu es Terribot.
                    
                    CONTEXTE DONNÉES (Glossaire) :
                    {glossaire_context}
                    
                    SCHEMA TABLE "TERRITOIRES" (alias t) :
                    - "ID" (VARCHAR) : Code INSEE
                    - "NOM_COUV" (VARCHAR) : Nom de la commune
                    
                    MISSION : Répondre à "{rewritten_prompt}" via UNE SEULE requête SQL.
                    
                    🚨 RÈGLES CRITIQUES (A RESPECTER ABSOLUMENT) :
                    
                    1. VARIABLES ET TABLES (ANTI-HALLUCINATION) :
                    - Utilise **UNIQUEMENT** les variables listées dans le CONTEXTE ci-dessus.
                    - Si une variable 2022 (ex: P22_...) n'est pas dans la liste, NE L'INVENTE PAS. Utilise l'année disponible la plus proche (ex: P20_... ou P19_...).
                    - Le contexte t'indique la table source (ex: ✅ TABLE: "ACT_10"). Utilise ce nom exact dans ton JOIN.
                    - Jointure : `FROM territoires t LEFT JOIN "NOM_TABLE" d ON t."ID" = d."ID"`
                    - Choisis toujours la variable la PLUS RÉCENTE disponible.
                    
                    2. PÉRIMÈTRE GÉOGRAPHIQUE :
                    - Copie STRICTEMENT cette clause WHERE :
                    - `WHERE (t."ID" IN ({ids_sql}) {parent_clause})`
                    - ⛔ INTERDIT : N'ajoute JAMAIS de condition sur "NOM_COUV".
                    
                    3. CALCULS ET TYPES :
                    - Tables format LARGE. Pas de colonne "VAR".
                    - Utilise `TRY_CAST(table."colonne" AS DOUBLE)` pour tout calcul.
                    - Calcule toujours des ratios (ex: Part du chômage = CHOM / ACT) pour rendre les territoires comparables.
                    - N'utilise pas des variables trop complexes ou peu lisibles (ex: des rangs, des indices composites)
                    - Calcul des parts et des taux simples, évite les ratios, les rangs, les différences
                    - Gère la division par zéro : `NULLIF(..., 0)`.

                    4. SYNTAXE NOMS DE COLONNES (Tirets et Spéciaux) :
                    - ⚠️ CRITIQUE : Les noms de colonnes contiennent souvent des tirets (-) ou des points (.).
                    - NE LES MODIFIE PAS. Utilise EXACTEMENT le nom fourni dans le CONTEXTE glossaire.
                    - Utilise TOUJOURS des guillemets doubles pour entourer les noms de colonnes.
                    - Exemple : Si le contexte indique "3-5_AUTREG", écris SELECT t."3-5_AUTREG" ... (et NON 3_5_AUTREG).

                    5. FORMAT DE SORTIE STRICT :
                    - ⛔ INTERDICTION d'utiliser des alias (AS) sur les colonnes d'identifiant.
                    - La colonne identifiant DOIT s'appeler "ID".
                    - La colonne nom DOIT s'appeler "NOM_COUV".
                    - Exemple CORRECT : SELECT t."ID", t."NOM_COUV", ...
                    - Exemple INTERDIT : SELECT t."ID" as code_insee, ..
                    
                    Réponds uniquement le SQL.
                    """

                    _dbg("pipeline.sql.gen.call", ids_count=len(geo_context.get("all_ids", [])), parent_clause=parent_clause, sys_prompt_len=len(system_prompt))

                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": rewritten_prompt}],
                        temperature=0
                    )
                    sql_query_raw = response.choices[0].message.content.replace("```sql", "").replace("```", "").strip()
                    sql_query = sql_query_raw.split(";")[0]
                    _dbg("pipeline.sql.gen.raw", sql_query=sql_query[:500])

                    debug_container["sql_query"] = sql_query
                    
                    with st.expander("💻 Trace : Génération SQL (IA)", expanded=False):
                        st.code(sql_query, language="sql")

                    sql_query = generate_and_fix_sql(client, MODEL_NAME, system_prompt, rewritten_prompt, con)
                    _dbg("sql.exec.about_to_run", sql=sql_query[:500], ids=geo_context.get("all_ids", [])[:10], ids_count=len(geo_context.get("all_ids", [])))

                    debug_container["sql_query"] = sql_query

                    if con:
                        df = con.execute(sql_query).df()
                        _dbg("sql.exec.result", empty=df.empty, rows=len(df), cols=list(df.columns))
                        
                        if not df.empty:
                            _dbg("sql.exec.head", head=df.head(3).to_dict(orient="records"))
                            
                            status_container.update(label="🎨 Je prépare la visualisation...")
                            
                            # On configure le graph PENDANT que le loader est encore là
                            print("[TERRIBOT][PIPE] 📈 get_chart_configuration() start")
                            chart_config = get_chart_configuration(df, rewritten_prompt, glossaire_context, client, MODEL_NAME)
                            _dbg("pipeline.chart_config.done", selected=chart_config.get("selected_columns"), formats=chart_config.get("formats"))
                            status_container.update(label="Terminé", state="complete")
                        else:
                            status_container.update(label="Aucune donnée trouvée", state="error")
                            message_placeholder.warning("Aucune donnée trouvée.")
                            st.stop()

                # --- SORTIE DU CONTEXTE 'with status_container:' ---
                loader_placeholder.empty()
                # A. Affichage du Graphique (une seule fois ici via le placeholder)
                if not df.empty:
                    with chart_placeholder:
                # On récupère les IDs finaux depuis le debug_container
                        current_ids = debug_container.get("final_ids", [])
                        
                        auto_plot_data(df, current_ids, config=chart_config, con=con)
                
                with data_placeholder:
                        with st.expander("📊 Voir les données brutes", expanded=False):
                            st.dataframe(style_df(df, chart_config.get('formats', {})), width='stretch')
                
                # B. Streaming du Texte
                if not df.empty:
                    print("[TERRIBOT][PIPE] 📝 Streaming response start")
                    _dbg("pipeline.stream.inputs", df_rows=len(df), df_cols=list(df.columns), formats=chart_config.get("formats"))

                    stream = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": f"""
                            Tu es Terribot, un expert en analyse territoriale s'adressant à des élus et agents  des collectivités locales en France.
                            
                            TON RÔLE :
                            Traduire les données brutes ci-jointes en une réponse naturelle, fluide et professionnelle.
                            Proposer une piste de réflexion pour aller plus loin, sous forme d'une question pour proposer un autre graphique.
                            
                            RÈGLES D'OR (À RESPECTER STRICTEMENT) :
                            1. ⛔ NE JAMAIS mentionner "le tableau", "vos données", "la colonne", "l'extrait" ou "la ligne". Fais comme si tu connaissais ces chiffres par cœur.
                            2. ⛔ NE JAMAIS citer les noms techniques des variables (ex: "taux_chomage_15_64" ou "indicateur_voisins"). Utilise le langage courant ("Taux de chômage").
                            3. ⛔ SI une colonne contient des 0 et des 1 (booléens), NE LES CITE PAS. Interprète-les (ex: "C'est supérieur à la moyenne").
                            4. CONTEXTUALISE : Si des villes demandées sont absentes des données, dis simplement "Je dispose des données pour X et Y" sans dire "dans le fichier fourni".
                            5. STRUCTURE : Va à l'essentiel.
                            
                            Unités des données : {json.dumps(chart_config.get('formats', {}))}
                            """},
                            {"role": "user", "content": df.to_string()}
                        ],
                        stream=True
                    )
                    full_response_text = message_placeholder.write_stream(stream)
                    _dbg("pipeline.stream.done", response_len=len(full_response_text) if full_response_text else 0)
                    print("[TERRIBOT][PIPE] ✅ Pipeline done")

                    # D. Sauvegarde Historique
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response_text,
                        "data": df,
                        "chart_config": chart_config, 
                        "debug_info": debug_container
                    })

            except Exception as e:
                st.error(f"Erreur : {e}")
                print("[TERRIBOT][FATAL] Exception:", repr(e))
                import traceback
                print(traceback.format_exc())
