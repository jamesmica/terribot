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
import folium
from streamlit_folium import st_folium
import branca.colormap as cm

print("[TERRIBOT] ✅ Script importé / démarrage du fichier")

# --- 0. OUTILS UTILITAIRES (A METTRE TOUT EN HAUT APRES LES IMPORTS) ---
import datetime
import subprocess
import urllib.request
import urllib.error

# --- DATA SETUP: Download parquet files if needed (bypasses Git LFS issues) ---
def check_and_download_data():
    """Check if data files exist, download if needed (bypasses Git LFS quota issues)."""
    data_dir = "data"

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print("[TERRIBOT][SETUP] ⚠️  Data directory created")

    # Check if we have actual parquet files (not just LFS pointers)
    parquet_files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]

    if not parquet_files:
        print("[TERRIBOT][SETUP] ⚠️  No parquet files found, checking for LFS pointers...")

    # Check if files are LFS pointers (ASCII text, ~130 bytes, starts with "version https://git-lfs")
    lfs_pointers = 0
    actual_files = 0

    for f in parquet_files:
        filepath = os.path.join(data_dir, f)
        file_size = os.path.getsize(filepath)

        if file_size < 200:  # Likely an LFS pointer
            try:
                with open(filepath, 'r') as fp:
                    content = fp.read(100)
                    if 'git-lfs' in content:
                        lfs_pointers += 1
                    else:
                        actual_files += 1
            except:
                actual_files += 1
        else:
            actual_files += 1

    # If we have LFS pointers or no files, try to download
    if lfs_pointers > 0 or len(parquet_files) < 70:
        print(f"[TERRIBOT][SETUP] ⚠️  Found {lfs_pointers} LFS pointers, {actual_files} actual files")
        print("[TERRIBOT][SETUP] 📥 Attempting to download data files...")
        print("[TERRIBOT][SETUP] ℹ️  See DATA_SETUP.md for manual setup instructions")

        try:
            result = subprocess.run(
                ["python", "download_data.py"],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode == 0:
                print("[TERRIBOT][SETUP] ✅ Data files downloaded successfully")
            else:
                print(f"[TERRIBOT][SETUP] ⚠️  Data download failed: {result.stderr}")
                print("[TERRIBOT][SETUP] ℹ️  The app will try to continue, but may fail if data is missing")
        except subprocess.TimeoutExpired:
            print("[TERRIBOT][SETUP] ⚠️  Data download timed out")
        except Exception as e:
            print(f"[TERRIBOT][SETUP] ⚠️  Could not run download script: {e}")
            print("[TERRIBOT][SETUP] ℹ️  Please run 'python download_data.py' manually")
    else:
        print(f"[TERRIBOT][SETUP] ✅ Found {actual_files} data files")

# Run data check on startup
try:
    check_and_download_data()
except Exception as e:
    print(f"[TERRIBOT][SETUP] ⚠️  Data setup check failed: {e}")

def _dbg(label, **kw):
    try:
        payload = " ".join([f"{k}={repr(v)[:200]}" for k, v in kw.items()])
    except Exception:
        payload = "(payload error)"

    # ✅ Ajouter le timestamp dans le message principal
    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]  # Heure avec millisecondes
    message = f"[TERRIBOT][DBG][{timestamp}] {label} :: {payload}"
    print(message)

    try:
        if "debug_logs" not in st.session_state:
            st.session_state.debug_logs = []
        st.session_state.debug_logs.append(message)
        st.session_state.debug_logs = st.session_state.debug_logs[-200:]
    except Exception:
        pass

# --- HELPER FUNCTIONS FOR INSEE CODES ---
def is_valid_insee_code(code_str):
    """
    Vérifie si une chaîne est un code INSEE valide (commune, EPCI, etc.).
    Gère les cas spéciaux de la Corse (2A, 2B).

    Returns:
        bool: True si le code est valide
    """
    if not code_str:
        return False

    code_str = str(code_str).strip()

    # Codes purement numériques (communes, EPCI, etc.)
    if code_str.isdigit():
        return True

    # Codes Corse : commence par 2A ou 2B suivi de chiffres
    # Ex: "2A004", "2B033"
    if len(code_str) >= 3:
        dept_prefix = code_str[:2].upper()
        if dept_prefix in ['2A', '2B']:
            # Vérifier que le reste est numérique
            rest = code_str[2:]
            return rest.isdigit() if rest else True

    return False

def is_commune_or_epci_code(code_str):
    """
    Vérifie si un code est un code de commune (4-5 caractères) ou EPCI (9 caractères).
    Gère les codes Corse (2A, 2B).

    Returns:
        bool: True si le code est un code de commune ou EPCI
    """
    if not code_str:
        return False

    code_str = str(code_str).strip()

    # Vérifier que c'est un code INSEE valide
    if not is_valid_insee_code(code_str):
        return False

    # Vérifier la longueur
    return len(code_str) in (4, 5, 9)

# --- 1. CONFIGURATION & STYLE (DOIT ÊTRE EN PREMIER) ---
st.set_page_config(
    page_title="Terribot | Assistant Territorial",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state=440
)

# CSS custom pour corriger la largeur des graphiques Vega-Lite
st.markdown("""
<style>
    /* Forcer la largeur correcte pour les graphiques Vega-Lite */
    .stVegaLiteChart > div {
        width: auto !important;
        max-width: 800px !important;
    }

    /* Corriger l'alignement du conteneur parent */
    .element-container:has(.stVegaLiteChart) {
        width: auto !important;
        max-width: 800px !important;
    }

    /* Éviter que les graphiques débordent sur mobile */
    @media (max-width: 800px) {
        .stVegaLiteChart > div,
        .element-container:has(.stVegaLiteChart) {
            width: 100% !important;
        }
    }

    /* Boutons d'ambiguïté : hauteur uniforme et texte centré */
    div[data-testid="stButton"] > button {
        min-height: 120px;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        white-space: normal;
    }

    /* Ne pas affecter les boutons de la sidebar */
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
        min-height: unset;
    }
</style>
""", unsafe_allow_html=True)

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
        _dbg("db.tables_saved",
             count=len(valid_tables),
             sample=valid_tables[:10] if valid_tables else [],
             schemas_count=len(schemas))

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
            SELECT * FROM read_csv('{glossaire_path}', delim=';', header=TRUE, ignore_errors=TRUE)
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

    # D. CRÉATION DE LA FONCTION strip_accents
    try:
        print("[TERRIBOT][DB] 🔧 Creating strip_accents function...")
        import unicodedata

        def python_strip_accents(text):
            """Supprime les accents d'une chaîne de caractères."""
            if text is None:
                return None
            if not isinstance(text, str):
                text = str(text)
            # Normalisation NFD + suppression des accents
            normalized = unicodedata.normalize('NFD', text)
            without_accents = ''.join(char for char in normalized if unicodedata.category(char) != 'Mn')
            return without_accents

        # Enregistrer la fonction Python dans DuckDB
        con.create_function("strip_accents", python_strip_accents)
        print("[TERRIBOT][DB] ✅ strip_accents function registered")
    except Exception as e:
        print(f"[TERRIBOT][DB] ⚠️ strip_accents creation failed: {e}")

    print("[TERRIBOT][DB] ✅ get_db_connection() EXIT")
    return con

# On utilise la connexion définie tout en haut (Point A)
con = get_db_connection()

st.markdown("""
<style>
    /* Cache le menu et footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Max width for Vega-Lite charts */
    div[data-testid="stVegaLiteChart"] {
        max-width: 800px;
        margin: 0 auto;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. ANIMATION ---
def inject_placeholder_animation():
    components.html("""
    <script>
const questions = [
            "Fais un état des lieux de la précarité à Manosque",
            "Toulouse est-elle une ville vieillissante ?",
            "Comment se portent les familles monoparentales à Roubaix ?",
            "Les jeunes quittent-ils le territoire de Saint-Étienne ?",
            "Quel est le profil socio-économique des habitants de Sarcelles ?",
            "L'accès à l'emploi des femmes progresse-t-il à Lens ?",
            "Compare la situation sociale de Mulhouse avec Strasbourg",
            "Qui sont les habitants de Limoges Métropole ?",
            "La précarité des seniors est-elle un enjeu à Perpignan ?",
            "Comment évolue la structure familiale à Nantes ?",
            "Les conditions de logement se dégradent-elles à Marseille ?",
            "Dresse un portrait social de la communauté d'agglomération de Béthune",
            "L'isolement des personnes âgées est-il marqué à Nice ?",
            "Quels sont les publics vulnérables à Calais ?",
            "Compare le tissu social de Maubeuge et Valenciennes",
            "La population de Brest rajeunit-elle ou vieillit-elle ?",
            "Quel est l'état de l'insertion professionnelle des jeunes à Amiens ?",
            "Les inégalités se creusent-elles entre Grigny et le reste de l'Essonne ?",
            "Fais un diagnostic de la monoparentalité dans les Bouches-du-Rhône",
            "La population active diminue-t-elle à Saint-Nazaire ?",
            "Quels enjeux sociaux identifie-t-on sur le territoire de Dunkerque ?"
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
    st.markdown("""
    <style>
    /* Aligner le titre tout en haut avec la flèche de réduction */
    section[data-testid="stSidebar"] > div {
      padding-top: 1rem !important;
    }

    section[data-testid="stSidebar"] h1 {
      margin-top: 0 !important;
      padding-top: 0 !important;
    }

    /* Limiter la taille des graphiques dans la sidebar */
    section[data-testid="stSidebar"] .stVegaLiteChart {
      max-width: 400px !important;
      width: 100% !important;
    }

    section[data-testid="stSidebar"] .stVegaLiteChart > div {
      max-width: 400px !important;
      width: 100% !important;
      height: 247px !important;
    }

    section[data-testid="stSidebar"] .stVegaLiteChart canvas {
      max-width: 400px !important;
      height: 247px !important;
    }
    /* Remonter uniquement le titre principal (le premier h1) */ 
    div[data-testid="stSidebarUserContent"] { 
    margin-top: -26px !important; 
    }

    div[data-testid="stSidebarHeader"] {
    z-index: 1;
    position: absolute;
    width: 40px; 
    right: 0;
    margin-top:-14px;
    }

    /* Style pour le bouton nouvelle conversation */
    section[data-testid="stSidebar"] button[kind="primary"] {
    height: 38px !important;
    padding: 0.25rem 0.75rem !important;
    font-size: 0.875rem !important;
    margin-top: 0.5rem !important;
    width: auto;
    margin-right: 80px;
    }
    /* H1 Terribot sans espace en dessous */
    section[data-testid="stSidebar"] h1 {
    margin-top: 0 !important;
    padding-top: 0 !important;
    padding-bottom: 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Titre et bouton nouvelle conversation sur la même ligne
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🤖 Terribot")
        st.caption("v0.19.1 - 26 janvier 2026")
    with col2:
        # Bouton nouvelle conversation
        if st.button("🔄", help="Nouvelle conversation", type="primary", use_container_width=True):
            # Réinitialiser l'état de la session
            st.session_state.messages = [{"role": "assistant", "content": "Bonjour ! Quel territoire souhaitez-vous analyser ?"}]
            st.session_state.current_geo_context = None
            st.session_state.force_geo_context = False
            st.session_state.pending_prompt = None
            st.session_state.ambiguity_candidates = None
            st.session_state.pending_geo_text = None
            st.rerun()

    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

    if api_key:
        api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    else:
        api_key = st.text_input("Clé API OpenAI", type="password", placeholder="sk-...")
        if not api_key:
            st.warning("Requis pour démarrer.")
            st.stop()

    # 🔧 Panneau de visualisation interactif dans la sidebar
    # Créer un placeholder qui sera rempli plus tard (après la définition des fonctions)
    sidebar_viz_placeholder = st.empty()

    # Stocker le placeholder dans session_state pour y accéder plus tard
    st.session_state.sidebar_viz_placeholder = sidebar_viz_placeholder

    # 🐛 Panneau de debug dans la sidebar
    sidebar_debug_placeholder = st.empty()
    st.session_state.sidebar_debug_placeholder = sidebar_debug_placeholder

client = openai.OpenAI(api_key=api_key)
MODEL_NAME = "gpt-5.2"  # Mis à jour vers un modèle standard valide, ajustez si nécessaire
EMBEDDING_MODEL = "text-embedding-3-small"

# --- 3.1 HELPERS OPENAI RESPONSES ---
def build_messages(system_prompt: str, user_prompt: str):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def extract_response_text(response) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return text
    try:
        parts = []
        for output in response.output:
            for content in output.content:
                if hasattr(content, "text"):
                    parts.append(content.text)
        return "".join(parts)
    except Exception:
        return ""


def stream_response_text(response_stream):
    for event in response_stream:
        if getattr(event, "type", "") == "response.output_text.delta":
            yield event.delta

@st.cache_data(show_spinner=False)
def load_territoires_text():
    data_dir = "data"
    territoires_path = os.path.join(data_dir, "territoires.txt")
    if not os.path.exists(territoires_path):
        return ""
    with open(territoires_path, "r", encoding="utf-8") as f:
        return f.read()

def format_conversation_context(messages):
    lines = []
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)

def build_geo_context_from_id(con, territory_id, source_label, search_query=None):
    row = con.execute(
        "SELECT ID, NOM_COUV, COMP1, COMP2, COMP3 FROM territoires WHERE ID = ? LIMIT 1",
        [str(territory_id)],
    ).fetchone()
    if not row:
        return None

    comps = [row[2], row[3], row[4]]
    ordered_ids = [str(row[0])]
    ordered_ids.extend(
        [
            str(comp)
            for comp in comps
            if comp and str(comp).lower() not in ["none", "nan", "null", ""]
        ]
    )
    ordered_ids.append("FR")
    final_ids_ordered = list(dict.fromkeys(ordered_ids))

    debug_entry = {
        "Trouvé": row[1],
        "ID": str(row[0]),
        "Source": source_label,
    }
    if search_query:
        debug_entry["Recherche"] = search_query

    return {
        "target_name": row[1],
        "target_id": str(row[0]),
        "all_ids": final_ids_ordered,
        "parent_clause": "",
        "display_context": row[1],
        "debug_search": [debug_entry],
        "lieux_cites": [row[1]],
    }

def chunk_text_by_bytes(text, max_bytes):
    encoded = text.encode("utf-8")
    chunks = []
    start = 0
    while start < len(encoded):
        end = min(start + max_bytes, len(encoded))
        chunk = encoded[start:end].decode("utf-8", errors="ignore")
        chunks.append(chunk)
        start = end
    return chunks

def ai_select_territory_from_full_context(
    client,
    model,
    territoires_text,
    conversation_text,
    pending_geo_text=None,
    max_chunk_bytes=200000,
):
    system_prompt = """
    Tu es un expert géographe français. Ta mission est d'identifier le territoire exact
    en te basant sur la discussion et les extraits successifs du fichier territoires.txt.

    RÈGLES :
    - Choisis UNIQUEMENT un ID qui existe dans la colonne ID du chunk fourni.
    - Si plusieurs correspondances, prends la plus pertinente selon le contexte.
    - Si aucun territoire ne convient dans ce chunk, retourne null.

    FORMAT DE RÉPONSE JSON STRICT :
    {
        "selected_id": "ID_exact_ou_null",
        "confidence": 0.0,
        "reason": "explication courte"
    }

    Réponds uniquement avec le JSON, sans texte additionnel.
    """

    chunks = chunk_text_by_bytes(territoires_text, max_chunk_bytes)
    best_result = None
    best_confidence = -1.0

    for idx, chunk in enumerate(chunks, start=1):
        user_prompt = f"""
        TERRITOIRE CIBLE (si mention explicite) : {pending_geo_text or "Non spécifié"}

        DISCUSSION COMPLÈTE :
        {conversation_text}

        EXTRAIT TERRITOIRES.TXT (CHUNK {idx}/{len(chunks)}) :
        {chunk}
        """

        try:
            response = client.responses.create(
                model=model,
                input=build_messages(system_prompt, user_prompt),
                temperature=0,
            )
            raw_response = extract_response_text(response)
            _dbg("geo.other_territory.response", chunk=idx, raw=raw_response[:400])

            chunk_result = json.loads(raw_response)
            selected_id = chunk_result.get("selected_id") if chunk_result else None
            confidence = chunk_result.get("confidence", 0.0) if chunk_result else 0.0
        except Exception as error:
            _dbg("geo.other_territory.error", chunk=idx, error=str(error))
            continue

        if selected_id and str(selected_id).lower() not in ["null", "none", ""]:
            try:
                confidence_value = float(confidence)
            except (TypeError, ValueError):
                confidence_value = 0.0
            if confidence_value > best_confidence:
                best_confidence = confidence_value
                best_result = chunk_result

    return best_result

# --- 4. FONCTIONS INTELLIGENTES (FORMATAGE & SÉLECTION) ---
def get_chart_configuration(df: pd.DataFrame, question: str, glossaire_context: str, client, model: str):
    """
    Fusionne la sélection des variables et la détection des formats et labels courts.
    """
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c.upper() not in ["AN", "ANNEE", "YEAR", "ID", "CODGEO"]]

    if not numeric_cols: return {"selected_columns": [], "formats": {}}

    stats = {}
    for c in numeric_cols[:20]:
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
        resp = client.responses.create(
            model=model,
            temperature=0,
            input=build_messages(system_prompt, json.dumps(payload, ensure_ascii=False)),
        )
        data = json.loads(extract_response_text(resp))

        if not data.get("selected_columns"): data["selected_columns"] = [numeric_cols[0]]
        data["selected_columns"] = [c for c in data["selected_columns"] if c in df.columns]
        return data
    except: return {"selected_columns": [numeric_cols[0]], "formats": {}}

def ai_enhance_formats(df: pd.DataFrame, initial_specs: dict, client, model):
    """
    Améliore les specs de formatage en utilisant l'IA pour analyser le contexte et les données.
    Retourne un dictionnaire de specs amélioré.
    """
    try:
        # Analyser les colonnes du DataFrame
        columns_info = []
        for col in df.columns:
            if col.upper() in ["ID", "CODGEO"]:
                continue

            col_data = {
                "column_name": col,
                "initial_spec": initial_specs.get(col, {}),
            }

            # Analyser les valeurs si c'est numérique
            if pd.api.types.is_numeric_dtype(df[col]):
                values = pd.to_numeric(df[col], errors='coerce').dropna()
                if not values.empty:
                    col_data["stats"] = {
                        "min": float(values.min()),
                        "max": float(values.max()),
                        "mean": float(values.mean()),
                        "has_decimals": not (values % 1 == 0).all(),
                        "sample_values": values.head(5).tolist()
                    }

            columns_info.append(col_data)

        if not columns_info:
            return initial_specs

        # Demander à l'IA d'améliorer le formatage
        system_prompt = """Tu es un expert en visualisation de données statistiques françaises et formatage de nombres.

TA MISSION : Analyser chaque colonne et déterminer le meilleur formatage en fonction du NOM de la colonne ET des VALEURS.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 ARBRE DE DÉCISION POUR LE TYPE (kind)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1️⃣ DÉTECTER SI C'EST UN POURCENTAGE :

   OPTION A : Ratio en base 1 (0-1) à convertir en pourcentage
   ✓ Conditions : max ≤ 1.5 ET min ≥ 0
   ✓ Exemple : 0.157, 0.432, 0.891 → Ce sont des ratios
   → kind="percent", percent_factor=100, decimals=1

   OPTION B : Pourcentage déjà en base 100
   ✓ Conditions : (nom contient "taux", "part", "pct", "pourcent", "%" OU "ratio")
                   ET max ≤ 100 ET max > 1.5
   ✓ Exemple : 15.7, 23.4, 45.2 dans colonne "Taux de chômage"
   → kind="percent", percent_factor=1, decimals=1

   ⚠️ EXCEPTION CRITIQUE - Taux de pauvreté :
   ✓ Si nom contient "TP60" → TOUJOURS percent_factor=1 (déjà en base 100)

2️⃣ DÉTECTER SI C'EST UNE DEVISE :

   ✓ Nom contient : "€", "euro", "revenu", "salaire", "montant", "prix", "budget", "coût"
   → kind="currency"

3️⃣ DÉTECTER LES RATIOS "PAR HABITANT" (PAS des pourcentages !) :

   ✓ Nom contient : "par habitant", "pour 100 hab", "pour 1 000 hab", "pour 10 000 hab", "/ hab"
   ✓ IMPORTANT : Ce sont des NOMBRES, PAS des pourcentages
   → kind="number" (et NON percent !)

4️⃣ SINON :
   → kind="number"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔢 RÈGLES POUR LES DÉCIMALES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Tester dans cet ordre :

1. Si max > 100 → decimals=0 (séparateur de milliers automatique)

2. Si kind="percent" → decimals=1 (toujours 1 décimale pour les %)

3. Si kind="currency" :
   - Si max > 100 → decimals=0
   - Sinon → decimals=1

4. Si "par habitant" dans nom ET max < 100 → decimals=1

5. Si has_decimals=true ET max < 100 → decimals=1

6. Sinon → decimals=0

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📝 FORMAT DE RÉPONSE JSON (OBLIGATOIRE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{
    "column_name": {
        "kind": "percent|currency|number",
        "decimals": 0-2,
        "percent_factor": 1 ou 100,  // OBLIGATOIRE si kind="percent", sinon omettre
        "title": "Titre lisible sans code technique",
        "label": "Nom court"
    },
    ...
}

EXEMPLES CONCRETS :

{
    "TP6019": {
        "kind": "percent",
        "decimals": 1,
        "percent_factor": 1,
        "title": "Taux de pauvreté",
        "label": "Taux pauvreté"
    },
    "NBMENFISC19": {
        "kind": "number",
        "decimals": 0,
        "title": "Nombre de ménages fiscaux",
        "label": "Ménages"
    },
    "MED19": {
        "kind": "currency",
        "decimals": 0,
        "title": "Revenu médian",
        "label": "Revenu médian"
    },
    "density_per_km2": {
        "kind": "number",
        "decimals": 1,
        "title": "Densité de population par km²",
        "label": "Densité/km²"
    }
}

⚠️ IMPORTANT :
- percent_factor doit être 1 ou 100 (jamais autre chose)
- Ne jamais confondre "par habitant" avec des pourcentages
- Les valeurs entre 0-1 sont TOUJOURS des ratios à convertir (sauf TP60)
- Répondre UNIQUEMENT avec le JSON, sans texte avant ou après
"""

        user_message = f"""Voici les colonnes à formater :

{json.dumps(columns_info, ensure_ascii=False, indent=2)}

Analyse chaque colonne et retourne le formatage optimal au format JSON."""

        response = client.responses.create(
            model=model,
            input=build_messages(system_prompt, user_message),
            temperature=0,
        )
        ai_response = extract_response_text(response)

        # Parser la réponse
        enhanced_specs = json.loads(ai_response)

        # Fusionner avec les specs initiales (l'IA a priorité)
        final_specs = initial_specs.copy()
        for col_name, spec in enhanced_specs.items():
            if col_name in df.columns:
                final_specs[col_name] = spec

        return final_specs

    except Exception as e:
        _dbg("format.ai_enhance.error", error=str(e))
        return initial_specs  # Fallback sur les specs initiales en cas d'erreur


def extract_sql_variables_from_context(glossaire_context: str) -> dict:
    """
    Extrait les noms de variables SQL techniques depuis le glossaire_context.
    Retourne un dict {nom_sql_technique: description}.

    Format attendu dans le contexte :
    ✅ TABLE: "TABLE_NAME" | VAR: "physical_column" | DESC: "description"
    """
    import re
    variables = {}

    if not glossaire_context:
        return variables

    # Pattern pour extraire VAR et DESC
    pattern = r'VAR:\s*"([^"]+)"\s*\|\s*DESC:\s*"([^"]+)"'
    matches = re.findall(pattern, glossaire_context)

    for var_name, description in matches:
        variables[var_name] = description

    print(f"[TERRIBOT][METADATA] 📋 Variables SQL extraites du contexte : {list(variables.keys())}")
    return variables


def split_sql_select_fields(select_clause: str) -> list:
    fields = []
    buffer = []
    depth = 0
    in_single_quote = False
    in_double_quote = False

    for char in select_clause:
        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
        elif char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
        elif char == "(" and not in_single_quote and not in_double_quote:
            depth += 1
        elif char == ")" and not in_single_quote and not in_double_quote and depth > 0:
            depth -= 1
        elif char == "," and not in_single_quote and not in_double_quote and depth == 0:
            field = "".join(buffer).strip()
            if field:
                fields.append(field)
            buffer = []
            continue
        buffer.append(char)

    trailing = "".join(buffer).strip()
    if trailing:
        fields.append(trailing)
    return fields


def parse_sql_select_expressions(sql_query: str) -> dict:
    if not sql_query:
        return {}

    match = re.search(r"select\s+(.*?)\s+from\s", sql_query, re.IGNORECASE | re.DOTALL)
    if not match:
        return {}

    select_clause = match.group(1)
    fields = split_sql_select_fields(select_clause)
    expressions = {}

    for field in fields:
        alias_match = re.search(r"\s+AS\s+([\"\w\-]+)\s*$", field, re.IGNORECASE)
        if alias_match:
            alias = alias_match.group(1).strip('"')
            expression = re.sub(r"\s+AS\s+[\"\w\-]+\s*$", "", field, flags=re.IGNORECASE).strip()
        else:
            tokens = field.strip().split()
            if len(tokens) > 1:
                alias = tokens[-1].strip('"')
                expression = " ".join(tokens[:-1]).strip()
            else:
                alias = field.split(".")[-1].strip().strip('"')
                expression = field.strip()

        if alias:
            expressions[alias] = expression

    return expressions


def simplify_calculation_expression(expression: str) -> str:
    if not expression:
        return ""
    rendered = expression
    rendered = re.sub(r"/\*.*?\*/", "", rendered, flags=re.DOTALL)
    rendered = re.sub(r"\bTRY_CAST\s*\(\s*([^)]+?)\s+AS\s+\w+\s*\)", r"\1", rendered, flags=re.IGNORECASE)
    rendered = re.sub(r"\bCAST\s*\(\s*([^)]+?)\s+AS\s+\w+\s*\)", r"\1", rendered, flags=re.IGNORECASE)
    rendered = re.sub(r"\bNULLIF\s*\(\s*([^,]+?)\s*,\s*[^)]+\)", r"\1", rendered, flags=re.IGNORECASE)
    rendered = re.sub(r"\bCOALESCE\s*\(\s*([^)]+?)\s*\)", r"\1", rendered, flags=re.IGNORECASE)
    rendered = re.sub(r"\bROUND\s*\(\s*([^)]+?)\s*\)", r"\1", rendered, flags=re.IGNORECASE)
    for _ in range(3):
        rendered = re.sub(r"\b[A-Z_]+\s*\(\s*([^)]+?)\s*\)", r"\1", rendered, flags=re.IGNORECASE)
    rendered = re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\.", "", rendered)
    rendered = re.sub(r"\s+", " ", rendered).strip()
    return rendered


def build_calculation_display(expression: str, var_definitions: dict) -> str:
    if not expression:
        return ""

    rendered = expression
    for var_name, definition in var_definitions.items():
        if not definition:
            continue
        quoted_pattern = rf"\"{re.escape(var_name)}\""
        dotted_pattern = rf"\b\w+\.{re.escape(var_name)}\b"
        rendered = re.sub(quoted_pattern, definition, rendered)
        rendered = re.sub(dotted_pattern, definition, rendered)

    rendered = simplify_calculation_expression(rendered)
    return rendered


def normalize_glossary_key(value: str) -> str:
    if value is None:
        return ""
    text = str(value).strip().upper()
    if text.startswith("X") and len(text) > 1 and text[1].isdigit():
        text = text[1:]
    text = re.sub(r"[_\-\s]", "", text)
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


def get_column_metadata(
    df: pd.DataFrame,
    specs: dict,
    con,
    glossaire_context: str = "",
    sql_query: str = "",
):
    """
    Extrait les métadonnées des colonnes depuis le glossaire (source, année, calcul).
    Les sources et intitulés détaillés listés correspondent aux colonnes réellement
    affichées dans le tableau df, en explicitant les variables du glossaire qui
    composent chaque colonne calculée quand c'est possible.

    Retourne un dict {col_name: {source, year, definition, calculation}}.
    """
    metadata = {}

    try:
        # Récupérer toutes les lignes du glossaire
        glossaire_df = con.execute("SELECT * FROM glossaire").df()
        print(f"[TERRIBOT][METADATA] 📚 Glossaire chargé : {len(glossaire_df)} entrées")
        print(f"[TERRIBOT][METADATA] 📊 Colonnes du DataFrame à traiter : {list(df.columns)}")

        # Créer un index uppercase une seule fois pour optimiser les recherches
        if 'Nom au sein de la base de données' not in glossaire_df.columns:
            print("[TERRIBOT][METADATA] ⚠️ Colonne 'Nom au sein de la base de données' non trouvée")
            return metadata

        # Nettoyer et créer une colonne uppercase pour la recherche
        name_series = glossaire_df['Nom au sein de la base de données'].fillna('').astype(str)
        glossaire_df['_name_upper'] = name_series.str.upper()
        glossaire_df['_name_norm'] = name_series.apply(normalize_glossary_key)

        sql_expressions = parse_sql_select_expressions(sql_query)

        print(f"[TERRIBOT][METADATA] 🔄 Matching des colonnes du dataframe")
        for col in df.columns:
            # Ignorer les colonnes système
            if col.upper() in ["ID", "AN", "ANNEE", "YEAR", "CODGEO", "NOM_COUV", "NOM"]:
                continue

            print(f"[TERRIBOT][METADATA] 🔍 Recherche métadonnées pour colonne : '{col}'")

            components = []
            component_sources = set()
            component_years = set()

            if col in sql_expressions:
                expression = sql_expressions[col]
                var_candidates = set()
                var_candidates.update(re.findall(r'"([^"]+)"', expression))
                var_candidates.update(re.findall(r"\b\w+\.([A-Za-z0-9_\-]+)\b", expression))

                var_definitions = {}
                for var_name in var_candidates:
                    candidate_names = {var_name}
                    candidate_names.add(var_name.replace("-", "_"))
                    candidate_names.add(var_name.replace("_", "-"))
                    if var_name.lower().startswith("x") and len(var_name) > 1 and var_name[1].isdigit():
                        candidate_names.add(var_name[1:])

                    matches = pd.DataFrame()
                    for candidate in candidate_names:
                        candidate_upper = candidate.upper()
                        matches = glossaire_df[glossaire_df['_name_upper'] == candidate_upper]
                        if not matches.empty:
                            break
                        candidate_norm = normalize_glossary_key(candidate)
                        if candidate_norm:
                            matches = glossaire_df[glossaire_df['_name_norm'] == candidate_norm]
                            if not matches.empty:
                                break
                    if matches.empty:
                        continue
                    row = matches.iloc[0]
                    definition = str(row.get('Intitulé détaillé', '')).strip()
                    source = str(row.get('Source', '')).strip()
                    year = str(row.get('Année de référence', '')).strip()
                    table = str(row.get('Onglet', '')).strip()
                    components.append({
                        'name': var_name,
                        'definition': definition,
                    })
                    var_definitions[var_name] = definition
                    if source and source.upper() not in ['', 'NAN', 'NONE']:
                        component_sources.add(source)
                    elif table:
                        component_sources.add(table)
                    if year and year.upper() not in ['', 'NAN', 'NONE']:
                        component_years.add(year)

                if components:
                    calculation = build_calculation_display(expression, var_definitions)
                    metadata[col] = {
                        'source': ", ".join(sorted(component_sources)),
                        'year': ", ".join(sorted(component_years)),
                        'definition': "",
                        'calculation': calculation,
                        'components': components,
                    }
                    print(f"[TERRIBOT][METADATA]   ✅ Calcul détecté pour {col}")
                    continue

            # Chercher dans le glossaire avec le nom de colonne
            col_normalized = col.replace("_", "-").upper()
            matches = glossaire_df[glossaire_df['_name_upper'] == col_normalized]

            if matches.empty:
                col_normalized = col.replace("-", "_").upper()
                matches = glossaire_df[glossaire_df['_name_upper'] == col_normalized]

            if matches.empty:
                col_normalized = col.upper()
                matches = glossaire_df[glossaire_df['_name_upper'] == col_normalized]

            if matches.empty:
                col_norm_key = normalize_glossary_key(col)
                if col_norm_key:
                    matches = glossaire_df[glossaire_df['_name_norm'] == col_norm_key]

            if not matches.empty:
                row = matches.iloc[0]
                source = str(row.get('Source', '')).strip()
                year = str(row.get('Année de référence', '')).strip()
                definition = str(row.get('Intitulé détaillé', '')).strip()
                table = str(row.get('Onglet', '')).strip()

                metadata[col] = {
                    'source': source if source and source.upper() not in ['', 'NAN', 'NONE'] else table,
                    'year': year if year and year.upper() not in ['', 'NAN', 'NONE'] else '',
                    'definition': definition,
                    'calculation': '',
                    'components': []
                }
                print(f"[TERRIBOT][METADATA]   ✅ Trouvé: {definition[:50]}...")
            else:
                print(f"[TERRIBOT][METADATA]   ❌ Non trouvé dans le glossaire")

        print(f"[TERRIBOT][METADATA] 📊 Total : {len(metadata)} variables avec métadonnées")

    except Exception as e:
        print(f"[TERRIBOT][METADATA] ⚠️ Erreur : {str(e)}")
        import traceback
        traceback.print_exc()
        _dbg("metadata.extract.error", error=str(e))

    return metadata


def build_metadata_tooltip(meta: dict) -> str:
    if not meta:
        return ""
    description = meta.get("calculation") or meta.get("definition") or ""
    parts = []
    if description:
        parts.append(description)
    source = meta.get("source") or ""
    if source:
        parts.append(f"Source : {source}")
    return "\n\n".join(parts).strip()


def style_df(df: pd.DataFrame, specs: dict, metadata=None):
    """Applique le formatage pour l'affichage (DataFrame avec formatage français)."""
    # On travaille sur une copie pour ne pas casser le DF original
    df_display = df.copy()
    metadata = metadata or {}

    # Renommer NOM_COUV en Nom si la colonne existe
    if "NOM_COUV" in df_display.columns:
        df_display = df_display.rename(columns={"NOM_COUV": "Nom"})

    # On force la conversion en numérique pour être sûr
    for col in df_display.columns:
        try:
            df_display[col] = pd.to_numeric(df_display[col])
        except (ValueError, TypeError):
            # Si la conversion échoue, on garde la colonne telle quelle
            pass

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

    # On prépare le dictionnaire de formatage pour column_config
    column_config = {}

    # On itère sur TOUTES les colonnes du tableau (et pas juste celles du graph)
    for col in df_display.columns:
        # Ignorer ID et colonnes d'années
        if col.upper() in ["ID", "AN", "ANNEE", "YEAR", "CODGEO"]:
            continue

        # Figer la colonne "Nom" à gauche
        if col == "Nom":
            column_config[col] = st.column_config.TextColumn(
                col,
                width="medium",
                pinned="left",
                help=build_metadata_tooltip(metadata.get(col))
            )
            continue

        # On ignore les colonnes non numériques
        if not pd.api.types.is_numeric_dtype(df_display[col]):
            if col in metadata:
                column_config[col] = st.column_config.Column(
                    col,
                    help=build_metadata_tooltip(metadata.get(col))
                )
            continue

        # 🎨 Récupérer les specs fournies par l'IA (ou valeurs par défaut)
        s = specs.get(col, specs.get("NOM_COUV", {})) if col == "Nom" else specs.get(col, {})
        kind = (s.get("kind") or "number").lower()
        dec = int(s.get("decimals", 1))  # Par défaut 1 décimale
        percent_factor = int(s.get("percent_factor", 1))  # 1 ou 100 selon l'IA

        # ✅ Formater selon le type déterminé par l'IA
        if kind in ["currency", "euro"]:
            df_display[col] = df_display[col].apply(lambda x: fr_num(x, dec, "€") if pd.notna(x) else "-")
        elif kind == "percent":
            # ✅ Utiliser le percent_factor fourni par l'IA (plus d'heuristique !)
            df_display[col] = df_display[col].apply(lambda x: fr_num(x, dec, "%", factor=percent_factor) if pd.notna(x) else "-")
        else:
            df_display[col] = df_display[col].apply(lambda x: fr_num(x, dec, "") if pd.notna(x) else "-")

        if col in metadata:
            column_config[col] = st.column_config.Column(
                col,
                help=build_metadata_tooltip(metadata.get(col))
            )

    return df_display, column_config


def render_metadata_details(metadata: dict):
    if not metadata:
        return

    details = []
    for col, meta in metadata.items():
        calculation = meta.get('calculation', '').strip()
        definition = meta.get('definition', '').strip()
        components = meta.get('components') or []
        if calculation:
            details.append(("calc", col, calculation, components))
        elif definition:
            details.append(("def", col, definition, components))

    if not details:
        return

    with st.expander("ℹ️ Détails des indicateurs", expanded=False):
        for mode, col, desc, components in details:
            if mode == "calc":
                st.caption(f"**{col}** = {desc}")
            else:
                st.caption(f"**{col}** : {desc}")
            if components:
                lines = [
                    f"- `{comp['name']}` : {comp['definition']}"
                    for comp in components
                    if comp.get('definition')
                ]
                if lines:
                    st.markdown("\n".join(lines))


    # --- FONCTION DE RÉPARATION SQL ---
def generate_and_fix_sql(client, model, system_prompt, user_prompt, con, max_retries=3):
    """
    Génère le SQL et tente de le corriger en injectant le schéma réel en cas d'erreur.
    Retourne la requête SQL valide ou lève une exception après max_retries.

    AMÉLIORATIONS :
    - Détection améliorée des erreurs de colonnes manquantes
    - Injection automatique des schémas complets en cas d'erreur
    - Messages d'erreur plus explicites
    """
    _dbg("sql.fix.enter", max_retries=max_retries)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    sql_query = None
    db_schemas = st.session_state.get("db_schemas", {})

    for attempt in range(max_retries + 1):
        print(f"[TERRIBOT][SQL] ▶️ Tentative {attempt + 1}/{max_retries + 1}")

        try:
            # 1. Génération
            response = client.responses.create(
                model=model,
                input=messages,
                temperature=0,
                timeout=60,
            )
            sql_query_raw = extract_response_text(response).replace("```sql", "").replace("```", "").strip()
            sql_query = sql_query_raw.split(";")[0].strip()

            # Vérification basique que c'est du SQL
            if not sql_query.upper().startswith("SELECT"):
                _dbg("sql.fix.invalid_sql", sql_preview=sql_query[:100])
                raise ValueError("La réponse n'est pas une requête SELECT valide")

            # 2. Vérification (Dry Run)
            con.execute(f"EXPLAIN {sql_query}")
            print("[TERRIBOT][SQL] ✅ SQL validé avec succès")
            return sql_query

        except Exception as e:
            error_msg = str(e)
            error_preview = error_msg.split("\n")[0][:300]
            print(f"[TERRIBOT][SQL] ❌ Erreur DuckDB : {error_preview}")

            if attempt < max_retries:
                schema_hint = ""

                # === DÉTECTION AMÉLIORÉE DES ERREURS ===

                # 1. Colonne manquante dans une table
                match_col = re.search(r'Table (?:with name )?(?:")?([^"]+)(?:")? does not have a column named "([^"]+)"', error_msg)
                if match_col:
                    table_name = match_col.group(1)
                    missing_col = match_col.group(2)
                    print(f"[TERRIBOT][SQL] 🔍 Colonne manquante détectée : '{missing_col}' dans '{table_name}'")

                    # Résolution d'alias vers vrai nom de table
                    if sql_query:
                        alias_pattern = r'(?:FROM|JOIN)\s+(?:["\']?)([a-zA-Z0-9_\.\-]+)(?:["\']?)\s+(?:AS\s+)?\b' + re.escape(table_name) + r'\b'
                        alias_match = re.search(alias_pattern, sql_query, re.IGNORECASE)
                        if alias_match:
                            table_name = alias_match.group(1).strip('"')
                            print(f"[TERRIBOT][SQL] 🕵️ Alias résolu : '{match_col.group(1)}' -> '{table_name}'")

                    # Récupération du schéma complet
                    if table_name in db_schemas:
                        col_names = db_schemas[table_name]
                    else:
                        try:
                            cols = con.execute(f'DESCRIBE "{table_name}"').fetchall()
                            col_names = [c[0] for c in cols]
                        except:
                            col_names = []

                    if col_names:
                        # Recherche de colonnes similaires
                        from difflib import get_close_matches
                        suggestions = get_close_matches(missing_col, col_names, n=5, cutoff=0.4)

                        # Formatage des colonnes avec guillemets
                        cols_formatted = ', '.join([f'"{c}"' for c in col_names[:100]])
                        suggestions_formatted = ', '.join([f'"{s}"' for s in suggestions])

                        schema_hint = f"\n\n🚨 ERREUR : La colonne \"{missing_col}\" n'existe pas dans la table \"{table_name}\".\n"
                        schema_hint += f"📋 Colonnes RÉELLES disponibles dans \"{table_name}\" :\n"
                        schema_hint += f"   {cols_formatted}\n"
                        if suggestions:
                            schema_hint += f"\n💡 Colonnes similaires suggérées : {suggestions_formatted}\n"
                        schema_hint += "\n⚠️ UTILISE EXACTEMENT les noms de colonnes ci-dessus (avec guillemets doubles)."

                # 2. Table référencée qui n'existe pas
                match_table_not_found = re.search(r'Table with name ([^ ]+) does not exist', error_msg)
                if match_table_not_found and not schema_hint:
                    missing_table = match_table_not_found.group(1).strip('"')
                    valid_tables = st.session_state.get("valid_tables_list", [])
                    from difflib import get_close_matches
                    suggestions = get_close_matches(missing_table.upper(), valid_tables, n=3, cutoff=0.4)
                    schema_hint = f"\n\n🚨 ERREUR : La table \"{missing_table}\" n'existe pas.\n"
                    schema_hint += f"📋 Tables disponibles : {', '.join(valid_tables)}\n"
                    if suggestions:
                        schema_hint += f"💡 Tables similaires suggérées : {', '.join(suggestions)}"

                # 3. Erreur générique - injection de tous les schémas des tables utilisées
                if not schema_hint and sql_query:
                    # Extraction des tables utilisées dans le SQL
                    tables_in_query = re.findall(r'(?:FROM|JOIN)\s+["\']?([a-zA-Z0-9_\.\-]+)["\']?', sql_query, re.IGNORECASE)
                    tables_in_query = [t.strip('"') for t in tables_in_query if t.lower() != 'territoires']

                    if tables_in_query:
                        schema_hint = "\n\n🚨 ERREUR SQL détectée. Voici les schémas COMPLETS des tables que tu utilises :\n"
                        for table in tables_in_query:
                            if table in db_schemas:
                                cols = db_schemas[table]
                                cols_formatted = ', '.join([f'"{c}"' for c in cols[:100]])
                                schema_hint += f'\n📋 TABLE "{table}" - Colonnes : {cols_formatted}\n'

                print("[TERRIBOT][SQL] 🛠️ Demande de correction avec schéma complet")
                if sql_query:
                    messages.append({"role": "assistant", "content": sql_query})

                fix_prompt = f"❌ Erreur DuckDB :\n{error_preview}\n{schema_hint}\n\n🔧 CORRIGE la requête SQL :\n- Vérifie que TOUTES les colonnes utilisées existent dans les schémas fournis\n- Utilise TOUJOURS des guillemets doubles pour les noms de colonnes\n- Ne modifie PAS les noms de colonnes, utilise-les EXACTEMENT comme dans le schéma\n\nNe réponds que le SQL corrigé."
                messages.append({"role": "user", "content": fix_prompt})
            else:
                _dbg("sql.fix.max_retries_reached", error=error_preview)
                print(f"[TERRIBOT][SQL] ⛔ Nombre maximum de tentatives atteint ({max_retries + 1})")
                raise Exception(f"Impossible de générer un SQL valide après {max_retries + 1} tentatives. Dernière erreur : {error_preview}")

    return sql_query

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
    """
    Recherche sémantique dans le glossaire via embeddings.
    """
    if glossary_embeddings is None or df_glossaire.empty:
        _dbg("rag.semantic.skip", reason="no_embeddings_or_glossaire")
        return pd.DataFrame()

    try:
        # Création de l'embedding de la requête
        query_resp = client.embeddings.create(input=[query[:1000]], model=EMBEDDING_MODEL, timeout=30)
        query_vec = np.array(query_resp.data[0].embedding)

        # Calcul des similarités
        similarities = np.dot(glossary_embeddings, query_vec)

        # Construction du DataFrame de résultats
        df_results = df_glossaire.iloc[valid_indices].copy()
        min_len = min(len(df_results), len(similarities))
        df_results = df_results.iloc[:min_len]
        df_results['similarity'] = similarities[:min_len]

        # 1. FILTRE PAR SEUIL (RAG Threshold)
        df_results = df_results[df_results['similarity'] > threshold]

        if df_results.empty:
            # Si aucun résultat au-dessus du seuil, on prend les top résultats quand même
            _dbg("rag.semantic.threshold_fallback", threshold=threshold)
            df_results = df_glossaire.iloc[valid_indices].copy().iloc[:min_len]
            df_results['similarity'] = similarities[:min_len]
            df_results = df_results.nlargest(top_k, 'similarity')

        # 2. Filtres techniques (exclusion IRIS/QPV)
        try:
            var_col = df_results.columns[4] if len(df_results.columns) > 4 else df_results.columns[0]
            mask_content = ~df_results[var_col].astype(str).str.contains(r'IRIS|QPV', case=False, regex=True, na=False)
            df_results = df_results[mask_content]
        except Exception as e_filter:
            _dbg("rag.semantic.filter_error", error=str(e_filter))
        return df_results.sort_values('similarity', ascending=False).head(top_k)

    except Exception as e:
        _dbg("rag.semantic.error", error=str(e))
        return pd.DataFrame()

def hybrid_variable_search(query, con, df_glossaire, glossary_embeddings, valid_indices, top_k=80):
    _dbg("rag.hybrid.start",
         query=query[:200],
         df_glossaire_rows=len(df_glossaire),
         has_embeddings=glossary_embeddings is not None,
         valid_indices_count=len(valid_indices) if valid_indices is not None else 0,
         top_k=top_k)

    print(f"\n[TERRIBOT][RAG] 🔍 Recherche de variables pour : '{query}'")
    print(f"[TERRIBOT][RAG] 📚 Glossaire : {len(df_glossaire)} entrées")

    candidates = {}

    # 1. RECHERCHE VECTORIELLE
    _dbg("rag.hybrid.semantic_search_start", query=query[:100])
    df_sem = semantic_search(query, df_glossaire, glossary_embeddings, valid_indices, top_k=top_k, threshold=0.35)
    _dbg("rag.hybrid.semantic_search_done", results_count=len(df_sem))

    print(f"[TERRIBOT][RAG] 🔎 Recherche sémantique : {len(df_sem)} résultats")
    if len(df_sem) > 0:
        for i, (_, row) in enumerate(df_sem.head(5).iterrows()):
            var = row['Nom au sein de la base de données']
            sim = row.get('similarity', 0)
            print(f"[TERRIBOT][RAG]   {i+1}. {var} (similarité: {sim:.3f})")

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

            print(f"[TERRIBOT][RAG] 🔎 Recherche FTS (mots-clés: {keywords}) : {len(df_fts)} résultats")
            if len(df_fts) > 0:
                for i, (_, row) in enumerate(df_fts.head(5).iterrows()):
                    var = row['Nom au sein de la base de données']
                    print(f"[TERRIBOT][RAG]   {i+1}. {var}")

            for _, row in df_fts.iterrows():
                var = row['Nom au sein de la base de données']
                candidates[var] = (0.9, row)
    except Exception as e:
        print(f"[TERRIBOT][RAG] ⚠️ Erreur FTS : {str(e)[:100]}")

    # 3. CONSTRUCTION DU CONTEXTE (CORRIGÉ)
    sorted_vars = sorted(candidates.items(), key=lambda x: x[1][0], reverse=True)[:top_k]
    from difflib import get_close_matches

    valid_tables = st.session_state.get("valid_tables_list", [])
    db_schemas = st.session_state.get("db_schemas", {})

    _dbg("rag.hybrid.session_state_check",
         has_valid_tables=len(valid_tables) > 0,
         valid_tables_count=len(valid_tables),
         has_db_schemas=len(db_schemas) > 0)

    if not valid_tables:
        try:
            # FALLBACK: Rebuild table list from parquet files in data/ directory
            # This matches the logic in get_db_connection()
            import os
            data_dir = "data"
            if os.path.exists(data_dir):
                parquet_files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
                valid_tables = []
                for f in parquet_files:
                    # Apply same normalization as in get_db_connection()
                    raw_name = f.replace('.parquet', '').upper()
                    table_name = re.sub(r'[^A-Z0-9]', '_', raw_name)
                    valid_tables.append(table_name)

                _dbg("rag.hybrid.tables_fallback",
                     count=len(valid_tables),
                     sample=valid_tables[:5] if valid_tables else [],
                     method="parquet_scan")

                # Update session_state so next time we don't need fallback
                st.session_state.valid_tables_list = valid_tables

                # Also build schemas for these tables
                if not db_schemas:
                    db_schemas = {}
                    for table_name in valid_tables:
                        try:
                            cols_info = con.execute(f"DESCRIBE \"{table_name}\"").fetchall()
                            db_schemas[table_name] = [c[0] for c in cols_info]
                        except:
                            pass
                    st.session_state.db_schemas = db_schemas
                    _dbg("rag.hybrid.schemas_fallback", count=len(db_schemas))
            else:
                _dbg("rag.hybrid.tables_fallback_error", error="data directory not found")
                valid_tables = []
        except Exception as e_tables:
            _dbg("rag.hybrid.tables_fallback_error", error=str(e_tables))
            valid_tables = []

    normalized_table_map = {standardize_name(t): t for t in valid_tables}

    result_context = ""
    for var, (score, row) in sorted_vars:
        desc = row['Intitulé détaillé']
        raw_source = str(row.get('Onglet', row.iloc[1])).upper()
        
        # 1. Résolution de la TABLE (Code précédent)
        if raw_source in ("", "NONE", "NAN"):
            _dbg("rag.hybrid.table_unknown", var=var, raw_source=raw_source)
            continue

        candidate_name = re.sub(r'[^A-Z0-9]', '_', raw_source)
        candidate_key = standardize_name(candidate_name)
        final_table_name = "UNKNOWN"

        if candidate_key in normalized_table_map:
            final_table_name = normalized_table_map[candidate_key]
        else:
            matches = get_close_matches(candidate_name, valid_tables, n=1, cutoff=0.4)
            if matches: final_table_name = matches[0]
            else:
                for t in valid_tables:
                    if t in candidate_name or candidate_name in t:
                        final_table_name = t
                        break

        if final_table_name == "UNKNOWN":
            _dbg("rag.hybrid.table_unknown",
                 var=var,
                 raw_source=raw_source,
                 candidate_name=candidate_name,
                 candidate_key=candidate_key,
                 valid_tables_count=len(valid_tables),
                 normalized_map_keys=list(normalized_table_map.keys())[:10],
                 CMF_10_in_valid=("CMF_10" in valid_tables),
                 CMF_10_in_map=(candidate_key in normalized_table_map))
            continue

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

    _dbg("rag.hybrid.end",
         candidates_count=len(sorted_vars),
         result_length=len(result_context),
         result_preview=result_context[:400])

    # Compter le nombre de variables effectivement trouvées
    num_variables_found = result_context.count("✅ TABLE:")
    print(f"[TERRIBOT][RAG] ✅ {num_variables_found} variables trouvées et validées")
    if num_variables_found > 0:
        # Afficher les 3 premières variables pour debug
        lines = [line for line in result_context.split("\n") if line.startswith("✅")]
        print(f"[TERRIBOT][RAG] 📊 Premières variables : ")
        for line in lines[:5]:
            print(f"[TERRIBOT][RAG]    {line}")
    else:
        print(f"[TERRIBOT][RAG] ⚠️ Aucune variable trouvée pour cette recherche !")

    return result_context

def extract_table_schemas_from_context(glossaire_context, con):
    """
    Extrait les noms de tables mentionnées dans le glossaire_context
    et retourne leurs schémas complets (toutes les colonnes).
    Cela permet de donner à l'IA TOUS les noms de colonnes disponibles,
    pour éviter les hallucinations.
    """
    import re

    # Extraction des noms de tables depuis le contexte
    # Format: ✅ TABLE: "NOM_TABLE" | VAR: "colonne" | DESC: "..."
    table_pattern = r'TABLE:\s*"([^"]+)"'
    table_names = set(re.findall(table_pattern, glossaire_context))

    schemas_dict = {}
    db_schemas = st.session_state.get("db_schemas", {})

    for table_name in table_names:
        if table_name in db_schemas:
            schemas_dict[table_name] = db_schemas[table_name]
        else:
            # Fallback: récupération directe depuis DuckDB
            try:
                cols = con.execute(f'DESCRIBE "{table_name}"').fetchall()
                schemas_dict[table_name] = [c[0] for c in cols]
            except Exception as e:
                print(f"[SCHEMA] ⚠️ Impossible de récupérer le schéma de {table_name}: {e}")
                schemas_dict[table_name] = []

    # Construction du texte de schéma
    schema_text = "\n\n📊 SCHÉMAS COMPLETS DES TABLES (Colonnes réelles disponibles) :\n"
    for table_name, columns in schemas_dict.items():
        if columns:
            # Limiter à 100 colonnes pour ne pas surcharger le prompt
            cols_display = columns[:100]
            remaining = len(columns) - 100
            cols_formatted = ', '.join([f'"{c}"' for c in cols_display])
            schema_text += f'\n🗂️ TABLE: "{table_name}"\n'
            schema_text += f'   Colonnes: {cols_formatted}'
            if remaining > 0:
                schema_text += f' ... et {remaining} autres colonnes'
            schema_text += '\n'

    return schema_text

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

    # 2. Expansion des synonymes EPCI (AVANT tokenisation)
    # Remplacer les abréviations par les termes complets pour améliorer la recherche
    epci_replacements = {
        r'\bcc\b': 'communaute de communes',
        r'\bc\.c\b': 'communaute de communes',
        r'\bca\b': 'communaute d agglomeration',
        r'\bc\.a\b': 'communaute d agglomeration',
        r'\bcu\b': 'communaute urbaine',
        r'\bc\.u\b': 'communaute urbaine',
    }

    for pattern, replacement in epci_replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # 3. Remplacements standards
    text = re.sub(r"[\-‐‑‒–—―]", " ", text)
    text = text.replace("'", " ").replace("'", " ")

    return text.strip()

def search_territory_smart(con, input_str):
    """
    Recherche intelligente avec priorité au Code Département si détecté.
    """
    _dbg("geo.search_smart.enter", input_str=input_str)

    clean_input = clean_search_term(input_str)
    if len(clean_input) < 2: return None

    # 1. Détection de Code Département (ex: "Fort-de-France 972", "Ajaccio 2A")
    # On cherche s'il y a un nombre de 2 ou 3 chiffres, ou 2A/2B (Corse) à la fin ou au début
    dept_code = None
    match = re.search(r'\b(97\d|\d{2}|2[AB])\b', input_str.upper())
    if match:
        dept_code = match.group(1)
        # On enlève le code du nom pour la recherche texte
        clean_input = clean_input.replace(dept_code.lower(), "").replace(dept_code, "").strip()
    _dbg("geo.search_smart.dept", dept_code=dept_code, clean_input=clean_input)

    # 2. Match Exact sur le Code INSEE (Priorité Absolue)
    if is_valid_insee_code(input_str.strip()):
        try:
            _dbg("geo.search_smart.sql", sql=("ID_exact" if is_valid_insee_code(input_str.strip()) else "strict_or_fuzzy"))
            res = con.execute("SELECT ID, NOM_COUV, COMP1, COMP2, COMP3 FROM territoires WHERE ID = ? LIMIT 1", [input_str.strip()]).fetchone()
            if res: return res
        except: pass

    # 2b. Recherche EXACTE sur le nom normalisé (nouvelle étape pour améliorer la précision)
    try:
        sql_exact_name = """
        SELECT ID, NOM_COUV, COMP1, COMP2, COMP3
        FROM territoires
        WHERE strip_accents(lower(replace(replace(NOM_COUV, '-', ' '), '''', ' '))) = ?
        LIMIT 5
        """
        results = con.execute(sql_exact_name, [clean_input.replace('-', ' ')]).fetchall()
        if len(results) == 1:
            return results[0]
        if len(results) > 1:
            return results
    except Exception as e:
        _dbg("geo.search_smart.exact_name_error", error=str(e))

    # 3. Token Search (Mots clés) avec Filtre Département optionnel
    words = [w.replace("'", "''") for w in clean_input.split() if len(w) > 1]  # Escape single quotes
    if words:
        # Construction de la clause WHERE avec paramètres
        conditions = []
        params = []

        for w in words:
            conditions.append("strip_accents(lower(NOM_COUV)) LIKE ?")
            params.append(f'%{w}%')

        where_clause = " AND ".join(conditions)

        # AJOUT DU FILTRE DEPT SI DÉTECTÉ
        if dept_code:
            # Escape et valider dept_code (doit être 2-3 chiffres)
            if dept_code.replace('A', '').replace('B', '').isdigit() and len(dept_code) <= 3:
                where_clause += " AND ID LIKE ?"
                params.append(f'{dept_code}%')

        sql_strict = f"""
        SELECT ID, NOM_COUV, COMP1, COMP2, COMP3
        FROM territoires WHERE {where_clause}
        ORDER BY length(NOM_COUV) ASC LIMIT 5
        """
        try:
            _dbg("geo.search_smart.sql", sql=("ID_exact" if input_str.strip().isdigit() else "strict_or_fuzzy"))
            results = con.execute(sql_strict, params).fetchall()
            print(f"[TERRIBOT][GEO] ✅ search_territory_smart results: {len(results)}")

            if len(results) == 1: return results[0]
            if len(results) > 1: return results
        except: pass

    # 4. Fuzzy Search (Jaro-Winkler) - Seulement si pas de dept_code (trop risqué sinon)
    if not dept_code:
        # Escape single quotes in clean_input
        clean_input_escaped = clean_input.replace("'", "''")

        sql_fuzzy = """
        WITH clean_data AS (
            SELECT ID, NOM_COUV, COMP1, COMP2, COMP3,
            lower(replace(replace(replace(NOM_COUV, '-', ' '), '''', ' '), ''', ' ')) as nom_simple
            FROM territoires
        )
        SELECT ID, NOM_COUV, COMP1, COMP2, COMP3,
        jaro_winkler_similarity(nom_simple, ?) as score
        FROM clean_data
        WHERE score > 0.88
        ORDER BY score DESC LIMIT 5
        """
        try:
            _dbg("geo.search_smart.sql", sql=("ID_exact" if input_str.strip().isdigit() else "strict_or_fuzzy"))
            results = con.execute(sql_fuzzy, [clean_input]).fetchall()
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
    Inclut une recherche spécifique pour les régions.
    """
    _dbg("geo.broad_candidates.enter", input_str=input_str, limit=limit)
    clean_input = clean_search_term(input_str)

    # NOUVEAU : Liste des régions françaises connues pour fallback
    REGIONS_MAPPING = {
        "ile de france": "R11", "ile-de-france": "R11", "idf": "R11",
        "centre val de loire": "R24", "bourgogne franche comte": "R27",
        "normandie": "R28", "hauts de france": "R32", "grand est": "R44",
        "pays de la loire": "R52", "bretagne": "R53", "nouvelle aquitaine": "R75",
        "occitanie": "R76", "auvergne rhone alpes": "R84", "paca": "R93",
        "provence alpes cote d azur": "R93", "corse": "R94"
    }

    # Vérification directe si c'est une région connue
    region_id = REGIONS_MAPPING.get(clean_input.replace('-', ' ').replace("'", " "))
    if region_id:
        _dbg("geo.broad_candidates.region_direct", clean_input=clean_input, region_id=region_id)
        try:
            sql_region = "SELECT ID, NOM_COUV, COMP1, COMP2, COMP3 FROM territoires WHERE ID = ?"
            df_region = con.execute(sql_region, [region_id]).df()
            if not df_region.empty:
                df_region['TYPE_TERRITOIRE'] = 'Région'
                df_region['score'] = 1.5  # Score élevé pour match direct
                return df_region.to_dict(orient='records')
        except Exception as e:
            _dbg("geo.broad_candidates.region_error", error=str(e))

    # SQL : On cherche large (Fuzzy + Contient)
    # Normalisation du terme de recherche pour la comparaison
    clean_for_sql = clean_input.replace("'", " ").replace("-", " ")

    # Déterminer les boosts pour les EPCI
    epci_boost = 0.25 if any(keyword in clean_for_sql.lower() for keyword in ['cc ', 'ca ', 'cu ', 'metropole']) else 0

    sql = """
    WITH candidates AS (
        SELECT
            ID,
            NOM_COUV,
            COMP1, COMP2, COMP3,
            CASE
                WHEN length(ID) IN (4,5) THEN 'Commune'
                WHEN length(ID) = 9 THEN 'EPCI/Interco'
                WHEN ID = 'FR' THEN 'Pays'
                WHEN ID LIKE 'D%' THEN 'Département'
                WHEN ID LIKE 'R%' THEN 'Région'
                ELSE 'Autre'
            END as TYPE_TERRITOIRE,

            -- BOOST DE SCORE :
            jaro_winkler_similarity(
                lower(replace(regexp_replace(NOM_COUV, '[\\-‐‑‒–—―]', ' ', 'g'), '''', ' ')),
                ?
            )
            + (CASE WHEN ID LIKE 'R%' THEN 0.2 ELSE 0 END)
            + (CASE WHEN ID LIKE 'D%' THEN 0.15 ELSE 0 END)
            + (CASE WHEN length(ID) = 9 THEN ? ELSE 0 END)
            + (CASE WHEN lower(replace(regexp_replace(NOM_COUV, '[\\-‐‑‒–—―]', ' ', 'g'), '''', ' ')) = ? THEN 0.3 ELSE 0 END)
            as score
        FROM territoires
        WHERE strip_accents(lower(replace(regexp_replace(NOM_COUV, '[\\-‐‑‒–—―]', ' ', 'g'), '''', ' '))) LIKE ?
           OR jaro_winkler_similarity(
                lower(replace(regexp_replace(NOM_COUV, '[\\-‐‑‒–—―]', ' ', 'g'), '''', ' ')),
                ?
              ) > 0.75
    )
    SELECT * FROM candidates
    ORDER BY score DESC
    LIMIT ?
    """

    try:
        params = [clean_for_sql, epci_boost, clean_for_sql, f'%{clean_for_sql}%', clean_for_sql, limit]
        df_candidates = con.execute(sql, params).df()

        # NOUVEAU : Recherche spécifique EPCI si peu de résultats et qu'on détecte un préfixe EPCI
        epci_keywords_lower = clean_input.lower()
        has_epci_prefix = any(prefix in epci_keywords_lower for prefix in ['cc ', 'ca ', 'cu ', 'metropole', 'communaute'])

        if has_epci_prefix and len(df_candidates) < 5:
            _dbg("geo.broad_candidates.epci_specific_search")

            # Extraire les mots-clés (sans les préfixes CC/CA/CU)
            keywords = clean_for_sql
            for prefix in ['cc ', 'ca ', 'cu ', 'communaute de communes', 'communaute d agglomeration', 'communaute urbaine', 'metropole']:
                keywords = keywords.replace(prefix, '')
            keywords = keywords.strip()

            # Recherche permissive dans les EPCI uniquement
            sql_epci = """
            SELECT ID, NOM_COUV, COMP1, COMP2, COMP3, 'EPCI/Interco' as TYPE_TERRITOIRE,
                   jaro_winkler_similarity(
                       strip_accents(lower(replace(replace(NOM_COUV, '-', ' '), '''', ' '))),
                       ?
                   ) as score
            FROM territoires
            WHERE length(ID) = 9
              AND strip_accents(lower(NOM_COUV)) LIKE ?
            ORDER BY score DESC
            LIMIT 10
            """
            try:
                df_epci = con.execute(sql_epci, [keywords, f'%{keywords}%']).df()
                if not df_epci.empty:
                    _dbg("geo.broad_candidates.epci_found", rows=len(df_epci))
                    # Fusionner avec les candidats existants
                    df_candidates = pd.concat([df_epci, df_candidates]).drop_duplicates(subset=['ID']).head(limit)
            except Exception as epci_error:
                _dbg("geo.broad_candidates.epci_error", error=str(epci_error))

        # Recherche étendue sur les régions si toujours vide
        if df_candidates.empty:
            # Recherche étendue sur les régions
            sql_regions = """
            SELECT ID, NOM_COUV, COMP1, COMP2, COMP3, 'Région' as TYPE_TERRITOIRE,
                   jaro_winkler_similarity(lower(NOM_COUV), ?) as score
            FROM territoires
            WHERE ID LIKE 'R%'
              AND jaro_winkler_similarity(lower(NOM_COUV), ?) > 0.6
            ORDER BY score DESC
            LIMIT 5
            """
            try:
                df_regions = con.execute(sql_regions, [clean_input, clean_input]).df()
                if not df_regions.empty:
                    _dbg("geo.broad_candidates.regions_fallback", rows=len(df_regions))
                    return df_regions.to_dict(orient='records')
            except:
                pass

        return df_candidates.to_dict(orient='records')
    except Exception as e:
        print(f"❌ Erreur SQL Candidates: {e}")
        _dbg("geo.broad_candidates.error", error=str(e))
        return []

def normalize_geo_id(raw_id, candidates):
    """
    Normalise un ID retourné par l'IA pour matcher avec la base de données.
    Gère les cas : "04112" -> "4112", "04" -> "D4", "11" -> "R11", "2A" -> "D2A", "2B" -> "D2B"
    """
    if not raw_id:
        return None

    raw_id = str(raw_id).strip().upper()  # Uppercase pour gérer 2a -> 2A
    candidate_ids = [str(c.get('ID', '')) for c in candidates]

    # 1. Match exact
    if raw_id in candidate_ids:
        return raw_id

    # 2. Match sans zéro initial (communes: "04112" -> "4112")
    # Mais attention à ne pas stripper si c'est un code Corse (2A, 2B)
    if not any(char.isalpha() for char in raw_id):  # Si pas de lettre
        stripped = raw_id.lstrip('0')
        if stripped and stripped in candidate_ids:
            return stripped

    # 3. Gestion spéciale Corse (2A, 2B)
    if raw_id in ['2A', '2B']:
        d_corse = f"D{raw_id}"
        if d_corse in candidate_ids:
            return d_corse
        # Essayer aussi sans D
        if raw_id in candidate_ids:
            return raw_id

    # 4. Match avec préfixe D (départements: "04" ou "94" ou "2A" -> "D4" ou "D94" ou "D2A")
    if (raw_id.isdigit() and len(raw_id) <= 3) or raw_id in ['2A', '2B']:
        # Essayer avec D + code
        if raw_id.isdigit():
            d_code = f"D{raw_id.lstrip('0')}"
        else:
            d_code = f"D{raw_id}"

        if d_code in candidate_ids:
            return d_code

        # Essayer D + code complet (pour DOM: 971 -> D971)
        d_full = f"D{raw_id}"
        if d_full in candidate_ids:
            return d_full

    # 5. Match avec préfixe R (régions: "11" -> "R11")
    if raw_id.isdigit() and len(raw_id) <= 2:
        r_code = f"R{raw_id}"
        if r_code in candidate_ids:
            return r_code

    # 6. Gestion des départements avec préfixe déjà présent
    # Ex: "D04" -> "D4" ou "D2A" -> "D2A"
    if raw_id.startswith('D'):
        dept_num = raw_id[1:]
        # Si c'est un nombre, essayer sans les zéros
        if dept_num.isdigit():
            d_normalized = f"D{dept_num.lstrip('0')}"
            if d_normalized in candidate_ids:
                return d_normalized
        # Si c'est déjà 2A ou 2B, c'est OK
        if raw_id in candidate_ids:
            return raw_id

    # 7. Gestion des régions avec préfixe déjà présent
    # Ex: "R04" -> "R4"
    if raw_id.startswith('R'):
        region_num = raw_id[1:]
        if region_num.isdigit():
            r_normalized = f"R{region_num.lstrip('0')}"
            if r_normalized in candidate_ids:
                return r_normalized

    # 8. Fuzzy match : chercher si un candidat contient l'ID (ex: "4112" dans ["4112", "28232"])
    for cid in candidate_ids:
        # Normaliser les deux codes pour comparaison
        cid_clean = str(cid).lstrip('0').replace('D', '').replace('R', '').upper()
        raw_clean = raw_id.lstrip('0').replace('D', '').replace('R', '').upper()

        if cid_clean == raw_clean:
            return cid

    # 9. Fallback : retourner le premier candidat avec le meilleur score
    _dbg("geo.normalize.fallback", raw_id=raw_id, candidates_sample=candidate_ids[:5])
    return None

def ai_select_territory(client, model, system_prompt, user_message, debug_label):
    try:
        response = client.responses.create(
            model=model,
            input=build_messages(system_prompt, user_message),
            temperature=0,
        )
        raw_response = extract_response_text(response)
        _dbg(debug_label, raw=raw_response[:400])
        return json.loads(raw_response)
    except Exception as error:
        _dbg(f"{debug_label}.error", error=str(error))
        return None


def ai_validate_territory(client, model, user_query, candidates, full_sentence_context=""):
    """
    Demande à l'IA de choisir le meilleur code INSEE parmi les candidats.
    """
    _dbg("geo.ai_validate.enter", user_query=user_query, candidates_len=len(candidates))

    if not candidates: return None

    system_prompt = """
    Tu es un expert géographe rattaché au code officiel géographique (INSEE) et au SIREN des EPCI.

    TA MISSION :
    Identifier le territoire unique qui correspond à la recherche de l'utilisateur parmi une liste de candidats.

    RÈGLES DE DÉCISION STRICTES :
    1. Si l'utilisateur tape juste le nom d'une ville (ex: "Dunkerque", "Manosque"), c'est TOUJOURS la "Commune" (ID 4 ou 5 chiffres). Pas l'EPCI, pas le Département.
       ⚠️ MÊME si le contexte mentionne un département (ex: "Manosque, Alpes-de-Haute-Provence"), choisis LA COMMUNE.

    1bis. ARRONDISSEMENTS : Si l'utilisateur mentionne un arrondissement (ex: "15e arrondissement de Paris", "3ème arrondissement de Lyon", "Marseille 8e"):
       - Cherche UNIQUEMENT le candidat de type "Commune" correspondant à cet arrondissement spécifique
       - Format dans la base : "Paris 15e Arrondissement" (code 75115), "Lyon 3e Arrondissement" (69383), "Marseille 8e Arrondissement" (13208)
       - NE choisis JAMAIS la commune principale (Paris, Lyon, Marseille) si un numéro d'arrondissement est mentionné
       - Match le numéro exact de l'arrondissement

    2. Si l'utilisateur précise explicitement un EPCI avec son préfixe (ex: "CC Durance Luberon", "CA Durance Lubéron Verdon", "Métropole de Lyon", "CU d'Arras"):
       - Cherche UNIQUEMENT le candidat de type "EPCI/Interco" (ID 9 chiffres)
       - NE CHOISIS JAMAIS le département, même s'il est mentionné dans le contexte
       - Match le nom de l'EPCI en ignorant les variantes d'orthographe (Luberon/Lubéron, tirets, espaces)
       - Privilégie les candidats contenant tous les mots-clés du nom recherché

    3. Si l'utilisateur tape SEULEMENT un nom de département (ex: "Alpes-de-Haute-Provence") OU un numéro (ex: "04"), c'est le Département.

    4. AMBIGUÏTÉ - Retourne is_ambiguous: true et confidence: "low" dans ces cas :
       - Homonymes parfaits dans différents départements sans contexte clair (ex: "Saint-Denis" existe partout)
       - Plusieurs EPCI avec des noms similaires
       - Doute entre commune et EPCI (sauf si préfixe CC/CA/CU présent)
       - Confiance < 80% sur le choix

    5. CONFIANCE :
       - "high" : Correspondance exacte, aucun doute (>90%)
       - "medium" : Bonne correspondance mais avec contexte incomplet (70-90%)
       - "low" : Correspondance incertaine, plusieurs choix possibles (<70%)

    PRÉFIXES D'EPCI À RECONNAÎTRE :
    - CC = Communauté de Communes
    - CA = Communauté d'Agglomération
    - CU = Communauté Urbaine
    - Métropole = Métropole
    - Grand(e) = souvent un EPCI (ex: Grand Paris, Grand Lyon)

    ⚠️ IMPORTANT - UTILISE EXACTEMENT L'ID DU CANDIDAT :
    - Si le candidat a l'ID "4112", réponds "4112" (PAS "04112")
    - Si le candidat a l'ID "D4", réponds "D4" (PAS "04" ou "4")
    - Si le candidat a l'ID "R11", réponds "R11" (PAS "11")
    - Si le candidat a l'ID "200068468", réponds "200068468" (code SIREN EPCI)

    FORMAT DE RÉPONSE JSON ATTENDU :
    {
        "selected_id": "code_insee_exact_du_candidat" OU null,
        "reason": "explication courte",
        "is_ambiguous": true/false,
        "confidence": "high|medium|low",
        "candidates_for_user": [
            {"id": "code1", "name": "Nom complet 1", "type": "Commune|EPCI|Département", "context": "Info supplémentaire"},
            {"id": "code2", "name": "Nom complet 2", "type": "Commune|EPCI|Département", "context": "Info supplémentaire"}
        ]
    }

    ⚠️ Si is_ambiguous: true, remplis OBLIGATOIREMENT candidates_for_user avec 2 à 5 candidats pertinents.
    Si is_ambiguous: false, tu peux laisser candidates_for_user vide [].
    """

    user_message = f"""
    CONTEXTE GLOBAL (Phrase utilisateur) : "{full_sentence_context}"

    TERME RECHERCHÉ ACTUELLEMENT : "{user_query}"

    Candidats trouvés en base pour "{user_query}" :
    {json.dumps(candidates, ensure_ascii=False, indent=2)}
    """

    result = ai_select_territory(
        client,
        model,
        system_prompt,
        user_message,
        "geo.ai_validate.exit",
    )

    # NORMALISATION CRITIQUE : L'IA peut retourner "04112" mais la base a "4112"
    if result and result.get("selected_id"):
        original_id = result["selected_id"]
        normalized_id = normalize_geo_id(original_id, candidates)

        if normalized_id and normalized_id != original_id:
            _dbg("geo.ai_validate.normalized", original=original_id, normalized=normalized_id)
            result["selected_id"] = normalized_id
        elif not normalized_id and candidates:
            # Fallback : prendre le premier candidat si l'ID IA ne matche rien
            fallback_id = str(candidates[0].get('ID', ''))
            _dbg("geo.ai_validate.fallback", original=original_id, fallback=fallback_id)
            result["selected_id"] = fallback_id

    return result

def ai_fallback_territory_search(con, user_prompt):
    """
    Fallback IA : cherche le territoire dans territoires.txt quand la recherche classique échoue.
    OPTIMISÉ: Recherche locale d'abord, puis web search seulement si nécessaire.
    Étape 1 : Recherche sémantique dans un échantillon stratégique (RAPIDE)
    Étape 2 : Si échec, recherche web pour trouver le code INSEE ou SIREN (LENT)
    Étape 3 : Recherche dans territoires.txt avec ce code
    """
    _dbg("geo.fallback.enter", user_prompt=user_prompt)

    try:
        # =================================================================
        # ÉTAPE 0 : RECHERCHE LOCALE PAR SIMILARITÉ TEXTUELLE EN PREMIER (TRÈS RAPIDE)
        # =================================================================
        _dbg("geo.fallback.fast_local_search")

        # Extraire les mots-clés du prompt (au moins 3 caractères)
        keywords = [word.strip() for word in user_prompt.split() if len(word.strip()) >= 3]

        if keywords:
            # Construire une requête avec LIKE pour chaque mot-clé (limiter à 3)
            keywords_limited = keywords[:3]
            like_clauses = " OR ".join(["strip_accents(LOWER(NOM_COUV)) LIKE strip_accents(LOWER(?))" for _ in keywords_limited])
            params = [f'%{kw}%' for kw in keywords_limited]

            fast_sql = f"""
            SELECT ID, NOM_COUV,
                CASE
                    WHEN length(ID) IN (4,5) THEN 'Commune'
                    WHEN length(ID) = 9 THEN 'EPCI/Interco'
                    WHEN ID = 'FR' THEN 'Pays'
                    WHEN ID LIKE 'D%' THEN 'Département'
                    WHEN ID LIKE 'R%' THEN 'Région'
                    ELSE 'Autre'
                END as TYPE_TERRITOIRE
            FROM territoires
            WHERE {like_clauses}
            ORDER BY
                CASE
                    WHEN strip_accents(LOWER(NOM_COUV)) = strip_accents(LOWER(?)) THEN 1
                    ELSE 2
                END,
                TYPE_TERRITOIRE, NOM_COUV
            LIMIT 100
            """

            try:
                # Ajouter le prompt complet pour le tri par correspondance exacte
                fast_params = params + [user_prompt]
                fast_df = con.execute(fast_sql, fast_params).df()

                if not fast_df.empty:
                    _dbg("geo.fallback.fast_local_results", count=len(fast_df))

                    # Si on a une correspondance exacte en premier résultat, la retourner immédiatement
                    if len(fast_df) > 0:
                        first_result = fast_df.iloc[0]
                        first_name_clean = unicodedata.normalize('NFD', str(first_result['NOM_COUV']).lower()).encode('ascii', 'ignore').decode("utf-8")
                        prompt_clean = unicodedata.normalize('NFD', user_prompt.lower()).encode('ascii', 'ignore').decode("utf-8")

                        if first_name_clean == prompt_clean:
                            # Correspondance exacte trouvée !
                            _dbg("geo.fallback.exact_match", id=first_result['ID'], name=first_result['NOM_COUV'])
                            return {
                                'target_id': first_result['ID'],
                                'target_name': first_result['NOM_COUV'],
                                'all_ids': [first_result['ID']],
                                'display_context': f"{first_result['NOM_COUV']} ({first_result['ID']})",
                                'lieux_cites': [user_prompt],
                                'debug_search': [{"Recherche": user_prompt, "Trouvé": first_result['NOM_COUV'], "ID": first_result['ID'], "Source": "Recherche locale exacte"}],
                                'parent_clause': ''
                            }

                    # Sinon, envoyer les résultats à l'IA pour sélection
                    fast_territories = fast_df.head(50).to_dict(orient='records')

                    fast_system_prompt = """
                    Tu es un expert géographe français. Voici les résultats d'une recherche locale par similarité.

                    TA MISSION :
                    Trouver le territoire le PLUS PERTINENT basé sur la requête utilisateur.

                    RÈGLES :
                    1. Privilégie les correspondances exactes ou très proches du nom
                    2. Comprends les abréviations : CC = Communauté de Communes, CA = Communauté d'Agglomération, CU = Communauté Urbaine
                    3. Sois tolérant aux fautes de frappe et variations orthographiques
                    4. Si plusieurs résultats similaires, privilégie selon le contexte
                    5. Si aucun résultat ne semble vraiment correspondre, retourne null

                    FORMAT DE RÉPONSE JSON ATTENDU :
                    {
                        "selected_id": "code_exact_du_territoire" OU null,
                        "reason": "explication courte de ton choix",
                        "confidence": "high/medium/low"
                    }
                    """

                    fast_user_message = f"""
                    REQUÊTE UTILISATEUR : "{user_prompt}"

                    RÉSULTATS DE RECHERCHE LOCALE :
                    {json.dumps(fast_territories, ensure_ascii=False, indent=2)}

                    Trouve le territoire le plus pertinent.
                    """

                    fast_result = ai_select_territory(
                        client,
                        MODEL_NAME,
                        fast_system_prompt,
                        fast_user_message,
                        "geo.fallback.fast_response",
                    )

                    if fast_result and fast_result.get("selected_id") and fast_result.get("confidence") in ["high", "medium"]:
                        selected_id = fast_result["selected_id"]

                        # Vérifier que l'ID existe
                        verify_sql = "SELECT ID, NOM_COUV FROM territoires WHERE ID = ?"
                        verify_df = con.execute(verify_sql, [selected_id]).df()

                        if not verify_df.empty:
                            territory_info = verify_df.iloc[0]
                            _dbg("geo.fallback.fast_success", id=selected_id, name=territory_info['NOM_COUV'])

                            return {
                                'target_id': selected_id,
                                'target_name': territory_info['NOM_COUV'],
                                'all_ids': [selected_id],
                                'display_context': f"{territory_info['NOM_COUV']} ({selected_id})",
                                'lieux_cites': [user_prompt],
                                'debug_search': [{"Recherche": user_prompt, "Trouvé": territory_info['NOM_COUV'], "ID": selected_id, "Source": "Recherche locale rapide"}],
                                'parent_clause': ''
                            }
            except Exception as fast_error:
                _dbg("geo.fallback.fast_error", error=str(fast_error))

        # =================================================================
        # ÉTAPE 1 : RECHERCHE WEB SI RECHERCHE LOCALE A ÉCHOUÉ (LENT)
        # =================================================================
        _dbg("geo.fallback.web_search", query=user_prompt)

        web_search_result = client.responses.create(
            model=MODEL_NAME,
            input=build_messages(
                """Tu es un expert en géographie française et en codes officiels (INSEE, SIREN).

                TA MISSION :
                À partir de la requête utilisateur, utilise l'outil de recherche web pour trouver le code officiel exact du territoire :
                - Code INSEE pour les communes (5 chiffres)
                - Code SIREN pour les EPCI/intercommunalités (9 chiffres)
                - Code département (ex: 61, 75, 2A)
                - Code région

                STRATÉGIE DE RECHERCHE :
                1. Pour une commune : cherche "code INSEE [nom de la commune]"
                2. Pour un EPCI : cherche "code SIREN [nom de l'EPCI]" ou "SIREN [nom exact]"
                3. Pour un département : cherche "code département [nom]"
                4. Privilégie les sources officielles : insee.fr, data.gouv.fr, collectivites-locales.gouv.fr

                FORMAT DE RÉPONSE JSON :
                {
                    "codes_found": ["code1", "code2", ...],
                    "territory_type": "commune|epci|departement|region",
                    "source": "url de la source",
                    "confidence": "high|medium|low"
                }

                Si aucun code n'est trouvé, retourne : {"codes_found": [], "confidence": "none"}
                """,
                f'Trouve le code officiel pour : "{user_prompt}"'
            ),
            temperature=0,
            tools=[{"type": "web_search", "enabled": True}]
        )
        web_response = extract_response_text(web_search_result)
        _dbg("geo.fallback.web_response", response=web_response[:500])

        try:
            web_data = json.loads(web_response)
            codes_found = web_data.get("codes_found", [])

            if codes_found and web_data.get("confidence") in ["high", "medium"]:
                # ÉTAPE 2 : Vérifier si ces codes existent dans territoires.txt
                for code in codes_found:
                    # Nettoyer le code
                    code_clean = str(code).strip()

                    # Chercher dans la base
                    verify_sql = "SELECT ID, NOM_COUV FROM territoires WHERE ID = ?"
                    verify_df = con.execute(verify_sql, [code_clean]).df()

                    if not verify_df.empty:
                        territory_info = verify_df.iloc[0]
                        _dbg("geo.fallback.web_success", code=code_clean, name=territory_info['NOM_COUV'])

                        return {
                            'target_id': code_clean,
                            'target_name': territory_info['NOM_COUV'],
                            'all_ids': [code_clean],
                            'display_context': f"{territory_info['NOM_COUV']} ({code_clean})",
                            'lieux_cites': [user_prompt],
                            'debug_search': [{"Recherche": user_prompt, "Trouvé": territory_info['NOM_COUV'], "ID": code_clean, "Source": "Recherche Web + territoires.txt"}],
                            'parent_clause': ''
                        }

                _dbg("geo.fallback.web_codes_not_in_db", codes=codes_found)
        except json.JSONDecodeError:
            _dbg("geo.fallback.web_json_error")

        # ÉTAPE 3 : Si la recherche web a échoué, faire une recherche sémantique dans un échantillon
        _dbg("geo.fallback.semantic_search")

        # Charger un échantillon stratégique : tous les départements, régions et grandes villes
        sql_sample = """
        SELECT ID, NOM_COUV,
            CASE
                WHEN length(ID) IN (4,5) THEN 'Commune'
                WHEN length(ID) = 9 THEN 'EPCI/Interco'
                WHEN ID = 'FR' THEN 'Pays'
                WHEN ID LIKE 'D%' THEN 'Département'
                WHEN ID LIKE 'R%' THEN 'Région'
                ELSE 'Autre'
            END as TYPE_TERRITOIRE
        FROM territoires
        WHERE ID LIKE 'D%' OR ID LIKE 'R%' OR ID = 'FR'
           OR (length(ID) IN (4,5) AND NOM_COUV IN (
               'Paris', 'Marseille', 'Lyon', 'Toulouse', 'Nice', 'Nantes', 'Strasbourg',
               'Montpellier', 'Bordeaux', 'Lille', 'Rennes', 'Reims', 'Le Havre',
               'Saint-Étienne', 'Toulon', 'Grenoble', 'Dijon', 'Angers', 'Nîmes',
               'Villeurbanne', 'Le Mans', 'Aix-en-Provence', 'Clermont-Ferrand', 'Brest',
               'Tours', 'Amiens', 'Limoges', 'Annecy', 'Perpignan', 'Boulogne-Billancourt',
               'Metz', 'Besançon', 'Orléans', 'Saint-Denis', 'Argenteuil', 'Rouen',
               'Mulhouse', 'Montreuil', 'Caen', 'Nancy'
           ))
        ORDER BY TYPE_TERRITOIRE, NOM_COUV
        LIMIT 500
        """
        df_sample = con.execute(sql_sample).df()

        if df_sample.empty:
            _dbg("geo.fallback.no_sample")
            return None

        # Convertir en format JSON pour l'IA
        territories_list = df_sample.to_dict(orient='records')

        system_prompt = """
        Tu es un expert géographe français. L'utilisateur cherche un territoire mais la recherche classique a échoué.

        TA MISSION :
        Trouver le territoire le plus pertinent dans la liste ci-dessous basé sur la requête utilisateur.

        RÈGLES :
        1. Si l'utilisateur mentionne une ville, cherche d'abord dans les communes
        2. Si l'utilisateur mentionne un département (nom ou numéro), cherche dans les départements (ID commence par 'D')
        3. Si l'utilisateur mentionne une région, cherche dans les régions (ID commence par 'R')
        4. Sois tolérant aux fautes de frappe et variations orthographiques
        5. Si vraiment rien ne correspond, retourne null

        FORMAT DE RÉPONSE JSON ATTENDU :
        {
            "selected_id": "code_exact_du_territoire" OU null,
            "reason": "explication courte de ton choix",
            "confidence": "high/medium/low"
        }
        """

        user_message = f"""
        REQUÊTE UTILISATEUR : "{user_prompt}"

        TERRITOIRES DISPONIBLES :
        {json.dumps(territories_list[:200], ensure_ascii=False, indent=2)}

        Trouve le territoire le plus pertinent.
        """

        result = ai_select_territory(
            client,
            MODEL_NAME,
            system_prompt,
            user_message,
            "geo.fallback.response",
        )

        if result and result.get("selected_id") and result.get("confidence") in ["high", "medium"]:
            selected_id = result["selected_id"]

            # Vérifier que l'ID existe dans la base complète
            verify_sql = f"SELECT ID, NOM_COUV FROM territoires WHERE ID = '{selected_id}'"
            verify_df = con.execute(verify_sql).df()

            if not verify_df.empty:
                territory_info = verify_df.iloc[0]
                _dbg("geo.fallback.success", id=selected_id, name=territory_info['NOM_COUV'])

                # Retourner un contexte géographique similaire à analyze_territorial_scope
                return {
                    'target_id': selected_id,
                    'target_name': territory_info['NOM_COUV'],
                    'all_ids': [selected_id],
                    'display_context': f"{territory_info['NOM_COUV']} ({selected_id})",
                    'lieux_cites': [user_prompt],
                    'debug_search': [{"Recherche": user_prompt, "Trouvé": territory_info['NOM_COUV'], "ID": selected_id, "Source": "Fallback IA"}],
                    'parent_clause': ''
                }

        # ÉTAPE 3.5 : Recherche par similarité textuelle dans toute la base
        _dbg("geo.fallback.fuzzy_search")

        # Extraire les mots-clés du prompt (au moins 3 caractères)
        keywords = [word.strip() for word in user_prompt.split() if len(word.strip()) >= 3]

        if keywords:
            # Construire une requête avec LIKE pour chaque mot-clé (limiter à 3)
            keywords_limited = keywords[:3]
            like_clauses = " OR ".join(["strip_accents(LOWER(NOM_COUV)) LIKE strip_accents(LOWER(?))" for _ in keywords_limited])
            params = [f'%{kw}%' for kw in keywords_limited]

            fuzzy_sql = f"""
            SELECT ID, NOM_COUV,
                CASE
                    WHEN length(ID) IN (4,5) THEN 'Commune'
                    WHEN length(ID) = 9 THEN 'EPCI/Interco'
                    WHEN ID = 'FR' THEN 'Pays'
                    WHEN ID LIKE 'D%' THEN 'Département'
                    WHEN ID LIKE 'R%' THEN 'Région'
                    ELSE 'Autre'
                END as TYPE_TERRITOIRE
            FROM territoires
            WHERE {like_clauses}
            ORDER BY TYPE_TERRITOIRE, NOM_COUV
            LIMIT 100
            """

            try:
                fuzzy_df = con.execute(fuzzy_sql, params).df()

                if not fuzzy_df.empty:
                    _dbg("geo.fallback.fuzzy_results", count=len(fuzzy_df))

                    # Envoyer les résultats à l'IA pour sélection
                    fuzzy_territories = fuzzy_df.to_dict(orient='records')

                    fuzzy_system_prompt = """
                    Tu es un expert géographe français. L'utilisateur cherche un territoire et voici les résultats d'une recherche par similarité.

                    TA MISSION :
                    Trouver le territoire le PLUS PERTINENT basé sur la requête utilisateur.

                    RÈGLES :
                    1. Privilégie les correspondances exactes ou très proches du nom
                    2. Si plusieurs résultats similaires, privilégie les communes aux départements
                    3. Sois tolérant aux fautes de frappe et variations orthographiques
                    4. Si aucun résultat ne semble vraiment correspondre, retourne null

                    FORMAT DE RÉPONSE JSON ATTENDU :
                    {
                        "selected_id": "code_exact_du_territoire" OU null,
                        "reason": "explication courte de ton choix",
                        "confidence": "high/medium/low"
                    }
                    """

                    fuzzy_user_message = f"""
                    REQUÊTE UTILISATEUR : "{user_prompt}"

                    RÉSULTATS DE RECHERCHE :
                    {json.dumps(fuzzy_territories, ensure_ascii=False, indent=2)}

                    Trouve le territoire le plus pertinent.
                    """

                    fuzzy_result = ai_select_territory(
                        client,
                        MODEL_NAME,
                        fuzzy_system_prompt,
                        fuzzy_user_message,
                        "geo.fallback.fuzzy_response",
                    )

                    if fuzzy_result and fuzzy_result.get("selected_id") and fuzzy_result.get("confidence") in ["high", "medium"]:
                        selected_id = fuzzy_result["selected_id"]

                        # Vérifier que l'ID existe
                        verify_sql = "SELECT ID, NOM_COUV FROM territoires WHERE ID = ?"
                        verify_df = con.execute(verify_sql, [selected_id]).df()

                        if not verify_df.empty:
                            territory_info = verify_df.iloc[0]
                            _dbg("geo.fallback.fuzzy_success", id=selected_id, name=territory_info['NOM_COUV'])

                            return {
                                'target_id': selected_id,
                                'target_name': territory_info['NOM_COUV'],
                                'all_ids': [selected_id],
                                'display_context': f"{territory_info['NOM_COUV']} ({selected_id})",
                                'lieux_cites': [user_prompt],
                                'debug_search': [{"Recherche": user_prompt, "Trouvé": territory_info['NOM_COUV'], "ID": selected_id, "Source": "Recherche par similarité"}],
                                'parent_clause': ''
                            }
            except Exception as fuzzy_error:
                _dbg("geo.fallback.fuzzy_error", error=str(fuzzy_error))

        # Si aucune méthode n'a fonctionné, appeler le fallback ultime
        _dbg("geo.fallback.calling_ultimate")
        return ultimate_ai_fallback(con, user_prompt)

    except Exception as e:
        _dbg("geo.fallback.error", error=str(e))
        return None

def ultimate_ai_fallback(con, user_prompt):
    """
    Fallback IA ULTIME qui n'échoue JAMAIS.
    Utilise toute la base territoires.txt avec stratégie intelligente par niveaux.

    Stratégie :
    1. Recherche dans départements + régions + grandes villes (rapide)
    2. Si échec : recherche par chunks de communes par département
    3. Si échec : recherche dans tous les EPCI
    4. En dernier recours : retourne "FR" (France entière)
    """
    _dbg("geo.ultimate_fallback.enter", user_prompt=user_prompt)

    try:
        # NIVEAU 1 : Départements, Régions, France + 100 plus grandes villes
        _dbg("geo.ultimate_fallback.level1")

        sql_level1 = """
        SELECT ID, NOM_COUV,
            CASE
                WHEN length(ID) IN (4,5) THEN 'Commune'
                WHEN length(ID) = 9 THEN 'EPCI'
                WHEN ID = 'FR' THEN 'France'
                WHEN ID LIKE 'D%' THEN 'Département'
                WHEN ID LIKE 'R%' THEN 'Région'
                ELSE 'Autre'
            END as TYPE_TERRITOIRE
        FROM territoires
        WHERE ID LIKE 'D%' OR ID LIKE 'R%' OR ID = 'FR'
        ORDER BY TYPE_TERRITOIRE, NOM_COUV
        """

        df_level1 = con.execute(sql_level1).df()

        if not df_level1.empty:
            territories_level1 = df_level1.to_dict(orient='records')

            system_prompt_level1 = """
            Tu es un expert géographe français spécialisé dans les codes officiels (INSEE, SIREN).

            TA MISSION :
            À partir de la requête utilisateur, trouve le territoire le PLUS PERTINENT parmi la liste fournie.

            RÈGLES DE DÉCISION :
            1. Si l'utilisateur mentionne un département (nom ou numéro) : choisis le département (ID commence par 'D')
            2. Si l'utilisateur mentionne une région : choisis la région (ID commence par 'R')
            3. Si l'utilisateur mentionne "France" ou un territoire national : choisis 'FR'
            4. Si l'utilisateur mentionne une commune : retourne null (on cherchera plus précisément après)
            5. Sois tolérant aux fautes d'orthographe et variations (ex: "Ile de France" = "Île-de-France")
            6. Pour les départements d'outre-mer : 971=Guadeloupe, 972=Martinique, 973=Guyane, 974=Réunion, 976=Mayotte
            7. Pour la Corse : 2A=Corse-du-Sud, 2B=Haute-Corse

            FORMAT DE RÉPONSE JSON :
            {
                "selected_id": "code_exact" OU null,
                "reason": "explication courte",
                "confidence": "high|medium|low|none"
            }

            Si aucun territoire ne correspond, retourne : {"selected_id": null, "confidence": "none"}
            """

            user_message_level1 = f"""
            REQUÊTE UTILISATEUR : "{user_prompt}"

            TERRITOIRES DISPONIBLES (Départements, Régions, France) :
            {json.dumps(territories_level1, ensure_ascii=False, indent=2)}

            Quel est le territoire le plus pertinent ?
            """

            result_level1 = ai_select_territory(
                client,
                MODEL_NAME,
                system_prompt_level1,
                user_message_level1,
                "geo.ultimate_fallback.level1_response",
            )

            if result_level1 and result_level1.get("selected_id") and result_level1.get("confidence") in ["high", "medium"]:
                selected_id = result_level1["selected_id"]

                # Vérification dans la base
                verify_sql = "SELECT ID, NOM_COUV FROM territoires WHERE ID = ?"
                verify_df = con.execute(verify_sql, [selected_id]).df()

                if not verify_df.empty:
                    territory_info = verify_df.iloc[0]
                    _dbg("geo.ultimate_fallback.level1_success", id=selected_id, name=territory_info['NOM_COUV'])

                    return {
                        'target_id': selected_id,
                        'target_name': territory_info['NOM_COUV'],
                        'all_ids': [selected_id],
                        'display_context': f"{territory_info['NOM_COUV']} ({selected_id})",
                        'lieux_cites': [user_prompt],
                        'debug_search': [{"Recherche": user_prompt, "Trouvé": territory_info['NOM_COUV'], "ID": selected_id, "Source": "Ultimate AI Fallback - Niveau 1"}],
                        'parent_clause': ''
                    }

        # NIVEAU 2 : Recherche sémantique dans TOUTES les communes
        # On charge par chunks pour ne pas dépasser les limites
        _dbg("geo.ultimate_fallback.level2")

        sql_communes = """
        SELECT ID, NOM_COUV, COMP2,
            'Commune' as TYPE_TERRITOIRE
        FROM territoires
        WHERE length(ID) IN (4,5)
        ORDER BY NOM_COUV
        """

        df_communes = con.execute(sql_communes).df()

        if not df_communes.empty:
            # Stratégie : envoyer un échantillon représentatif (toutes les communes de A à Z)
            # On prend 1 commune sur 10 pour avoir environ 3600 communes
            sample_communes = df_communes.iloc[::10].to_dict(orient='records')

            system_prompt_level2 = """
            Tu es un expert géographe français. L'utilisateur cherche probablement une commune.

            TA MISSION :
            Trouve la commune la PLUS PERTINENTE dans cette liste représentative.

            RÈGLES :
            1. Cherche une correspondance exacte ou très proche du nom
            2. Ignore les accents et tirets dans la comparaison
            3. Sois tolérant aux fautes de frappe
            4. Si plusieurs communes ont un nom similaire, choisis celle qui correspond le mieux au contexte
            5. Si vraiment aucune ne correspond, retourne null

            FORMAT DE RÉPONSE JSON :
            {
                "selected_id": "code_insee_exact" OU null,
                "reason": "explication courte",
                "confidence": "high|medium|low|none"
            }
            """

            user_message_level2 = f"""
            REQUÊTE UTILISATEUR : "{user_prompt}"

            COMMUNES DISPONIBLES (échantillon représentatif - 1 sur 10) :
            {json.dumps(sample_communes[:500], ensure_ascii=False, indent=2)}

            Quelle est la commune la plus pertinente ?
            """

            result_level2 = ai_select_territory(
                client,
                MODEL_NAME,
                system_prompt_level2,
                user_message_level2,
                "geo.ultimate_fallback.level2_response",
            )

            if result_level2 and result_level2.get("selected_id") and result_level2.get("confidence") in ["high", "medium"]:
                selected_id = result_level2["selected_id"]

                verify_sql = "SELECT ID, NOM_COUV FROM territoires WHERE ID = ?"
                verify_df = con.execute(verify_sql, [selected_id]).df()

                if not verify_df.empty:
                    territory_info = verify_df.iloc[0]
                    _dbg("geo.ultimate_fallback.level2_success", id=selected_id, name=territory_info['NOM_COUV'])

                    return {
                        'target_id': selected_id,
                        'target_name': territory_info['NOM_COUV'],
                        'all_ids': [selected_id],
                        'display_context': f"{territory_info['NOM_COUV']} ({selected_id})",
                        'lieux_cites': [user_prompt],
                        'debug_search': [{"Recherche": user_prompt, "Trouvé": territory_info['NOM_COUV'], "ID": selected_id, "Source": "Ultimate AI Fallback - Niveau 2 (Communes)"}],
                        'parent_clause': ''
                    }

            # Si on a une confiance "low", on peut rechercher plus précisément
            if result_level2 and result_level2.get("confidence") == "low" and result_level2.get("selected_id"):
                # Extraire le nom de la commune suggérée
                suggested_id = result_level2["selected_id"]
                suggested_row = df_communes[df_communes['ID'] == suggested_id]

                if not suggested_row.empty:
                    dept_code = suggested_row.iloc[0].get('COMP2')

                    if dept_code and str(dept_code) not in ['', 'None', 'NaN']:
                        # Rechercher toutes les communes de ce département
                        dept_communes = df_communes[df_communes['COMP2'] == dept_code].to_dict(orient='records')

                        system_prompt_level2b = """
                        Tu es un expert géographe français. Voici toutes les communes d'un département spécifique.

                        TA MISSION :
                        Trouve LA commune exacte qui correspond à la requête utilisateur.

                        FORMAT DE RÉPONSE JSON :
                        {
                            "selected_id": "code_insee_exact" OU null,
                            "reason": "explication courte",
                            "confidence": "high|medium|low|none"
                        }
                        """

                        user_message_level2b = f"""
                        REQUÊTE UTILISATEUR : "{user_prompt}"

                        COMMUNES DU DÉPARTEMENT {dept_code} :
                        {json.dumps(dept_communes, ensure_ascii=False, indent=2)}

                        Quelle est LA bonne commune ?
                        """

                        result_level2b = ai_select_territory(
                            client,
                            MODEL_NAME,
                            system_prompt_level2b,
                            user_message_level2b,
                            "geo.ultimate_fallback.level2b_response",
                        )

                        if result_level2b and result_level2b.get("selected_id"):
                            selected_id = result_level2b["selected_id"]

                            verify_sql = "SELECT ID, NOM_COUV FROM territoires WHERE ID = ?"
                            verify_df = con.execute(verify_sql, [selected_id]).df()

                            if not verify_df.empty:
                                territory_info = verify_df.iloc[0]
                                _dbg("geo.ultimate_fallback.level2b_success", id=selected_id, name=territory_info['NOM_COUV'])

                                return {
                                    'target_id': selected_id,
                                    'target_name': territory_info['NOM_COUV'],
                                    'all_ids': [selected_id],
                                    'display_context': f"{territory_info['NOM_COUV']} ({selected_id})",
                                    'lieux_cites': [user_prompt],
                                    'debug_search': [{"Recherche": user_prompt, "Trouvé": territory_info['NOM_COUV'], "ID": selected_id, "Source": "Ultimate AI Fallback - Niveau 2B (Département précis)"}],
                                    'parent_clause': ''
                                }

        # NIVEAU 3 : Recherche dans les EPCI
        _dbg("geo.ultimate_fallback.level3")

        sql_epci = """
        SELECT ID, NOM_COUV, 'EPCI' as TYPE_TERRITOIRE
        FROM territoires
        WHERE length(ID) = 9
        ORDER BY NOM_COUV
        LIMIT 1000
        """

        df_epci = con.execute(sql_epci).df()

        if not df_epci.empty:
            epci_list = df_epci.to_dict(orient='records')

            system_prompt_level3 = """
            Tu es un expert en intercommunalités françaises (EPCI).

            TA MISSION :
            Trouve l'EPCI le PLUS PERTINENT si l'utilisateur cherche une intercommunalité.

            PRÉFIXES D'EPCI :
            - CC = Communauté de Communes
            - CA = Communauté d'Agglomération
            - CU = Communauté Urbaine
            - Métropole = Métropole
            - Grand/Grande = souvent un EPCI

            FORMAT DE RÉPONSE JSON :
            {
                "selected_id": "code_siren_exact" OU null,
                "reason": "explication courte",
                "confidence": "high|medium|low|none"
            }

            Si ce n'est PAS un EPCI, retourne : {"selected_id": null, "confidence": "none"}
            """

            user_message_level3 = f"""
            REQUÊTE UTILISATEUR : "{user_prompt}"

            EPCI DISPONIBLES (1000 premiers) :
            {json.dumps(epci_list[:200], ensure_ascii=False, indent=2)}

            Est-ce un EPCI ? Si oui, lequel ?
            """

            result_level3 = ai_select_territory(
                client,
                MODEL_NAME,
                system_prompt_level3,
                user_message_level3,
                "geo.ultimate_fallback.level3_response",
            )

            if result_level3 and result_level3.get("selected_id") and result_level3.get("confidence") in ["high", "medium"]:
                selected_id = result_level3["selected_id"]

                verify_sql = "SELECT ID, NOM_COUV FROM territoires WHERE ID = ?"
                verify_df = con.execute(verify_sql, [selected_id]).df()

                if not verify_df.empty:
                    territory_info = verify_df.iloc[0]
                    _dbg("geo.ultimate_fallback.level3_success", id=selected_id, name=territory_info['NOM_COUV'])

                    return {
                        'target_id': selected_id,
                        'target_name': territory_info['NOM_COUV'],
                        'all_ids': [selected_id],
                        'display_context': f"{territory_info['NOM_COUV']} ({selected_id})",
                        'lieux_cites': [user_prompt],
                        'debug_search': [{"Recherche": user_prompt, "Trouvé": territory_info['NOM_COUV'], "ID": selected_id, "Source": "Ultimate AI Fallback - Niveau 3 (EPCI)"}],
                        'parent_clause': ''
                    }

        # NIVEAU 4 : DERNIER RECOURS - Retourne France entière
        _dbg("geo.ultimate_fallback.level4_default_france")

        return {
            'target_id': 'FR',
            'target_name': 'France',
            'all_ids': ['FR'],
            'display_context': 'France (territoire non identifié précisément)',
            'lieux_cites': [user_prompt],
            'debug_search': [{"Recherche": user_prompt, "Trouvé": "France (par défaut)", "ID": "FR", "Source": "Ultimate AI Fallback - Niveau 4 (Défaut)"}],
            'parent_clause': ''
        }

    except Exception as e:
        _dbg("geo.ultimate_fallback.error", error=str(e))
        # Même en cas d'erreur, on retourne France
        return {
            'target_id': 'FR',
            'target_name': 'France',
            'all_ids': ['FR'],
            'display_context': 'France (erreur de résolution)',
            'lieux_cites': [user_prompt],
            'debug_search': [{"Recherche": user_prompt, "Erreur": str(e), "ID": "FR", "Source": "Ultimate AI Fallback - Défaut (Erreur)"}],
            'parent_clause': ''
        }

def analyze_territorial_scope(con, rewritten_prompt):
    """
    Analyse le prompt pour extraire et résoudre les territoires mentionnés.
    Retourne un contexte géographique complet avec IDs et noms.
    """
    _dbg("geo.analyze.start", rewritten_prompt=rewritten_prompt[:200])

    # 1. Extraction des lieux ET du contexte département via IA
    try:
        extraction = client.responses.create(
            model=MODEL_NAME,
            input=build_messages(
                """Extrais TOUS les lieux géographiques et territoires mentionnés dans le texte.

                IMPORTANT - Types de territoires à détecter :
                - Communes (ex: "Paris", "L'Aigle", "Saint-Denis", "Fontenay", "Lyon", "Marseille")
                - EPCI/Intercommunalités (ex: "CC des Pays de L'Aigle", "Métropole de Lyon", "Grand Paris", "CU d'Arras", "CA Durance Luberon")
                - Départements (ex: "Orne", "61", "Hauts-de-Seine", "dans le 94", "département 04")
                - Régions (ex: "Normandie", "Île-de-France", "PACA")
                - Pays (ex: "France métropolitaine") - SEULEMENT si explicitement mentionné

                RÈGLES CRITIQUES :
                - Extrait UNIQUEMENT les lieux EXPLICITEMENT mentionnés dans le texte
                - NE DEVINE PAS de territoire plus large (ex: si "Saint-Denis" est mentionné, NE DIS PAS "France")
                - NE GÉNÉRALISE PAS (ex: "Paris" ≠ "France", "Lyon" ≠ "France")
                - Conserve EXACTEMENT le nom tel qu'écrit (avec "CC", "CU", "CA", "Métropole", "Grand", etc.)
                - Ne raccourcis PAS les noms (garde "CC des Pays de L'Aigle", pas juste "L'Aigle")
                - Si plusieurs territoires sont mentionnés, extrais-les tous
                - Si un contexte département est précisé (ex: "dans le 94", "département 61"), extrais-le séparément
                - Ne prends que les territoires explicitement cité, pas les précisions de lieux. (ex: Fontenay-sous-bois, D94, France, ne pas extraire France)

                ATTENTION AUX PIÈGES :
                - "Saint-Denis" → commune Saint-Denis (PAS "France")
                - "Paris" → commune Paris (PAS "France")
                - "Lyon" → commune Lyon (PAS "France")
                - "taux de pauvreté en France" → pays France (OUI)

                FORMAT DE RÉPONSE JSON :
                {
                    "lieux": ["Territoire 1", "Territoire 2"],
                    "departement_context": "94" OU null
                }

                EXEMPLES :
                - "Quel est le taux de pauvreté à Saint-Denis ?" → {"lieux": ["Saint-Denis"], "departement_context": null}
                - "Fontenay dans le 94" → {"lieux": ["Fontenay"], "departement_context": "94"}
                - "CC Durance Luberon" → {"lieux": ["CC Durance Luberon"], "departement_context": null}
                - "Manosque, Alpes-de-Haute-Provence" → {"lieux": ["Manosque"], "departement_context": "04"}
                - "Quel est le taux de chômage en France ?" → {"lieux": ["France"], "departement_context": null}
                - "Population de Lyon" → {"lieux": ["Lyon"], "departement_context": null}

                RÉPONDS UNIQUEMENT AVEC LE JSON, SANS EXPLICATION.
                """,
                rewritten_prompt,
            ),
            timeout=30,
        )
        extraction_result = json.loads(extract_response_text(extraction))
        lieux_cites = extraction_result.get("lieux", [])
        departement_context = extraction_result.get("departement_context")

        _dbg("geo.analyze.extraction", lieux=lieux_cites, dept_context=departement_context)
    except Exception as e:
        _dbg("geo.analyze.extraction_error", error=str(e))
        return None

    if not lieux_cites:
        return None

    # 2. Résolution de chaque lieu
    found_ids = []
    target_name = None
    target_id = None
    debug_info = []
    first_pass = True

    for lieu in lieux_cites:
        try:
            # Recherche large pour CE lieu
            candidates = get_broad_candidates(con, lieu)

            # FILTRAGE PAR DÉPARTEMENT si contexte présent
            if departement_context and candidates:
                _dbg("geo.analyze.dept_filter", dept=departement_context, before=len(candidates))

                # Normaliser le code département (ex: "94" → "D94", "04" → "D4")
                dept_code_variants = [
                    departement_context,
                    f"D{departement_context}",
                    f"D{departement_context.lstrip('0')}"
                ]

                filtered_candidates = []
                for c in candidates:
                    comp2 = str(c.get('COMP2', ''))
                    # Vérifier si le département du candidat correspond
                    if comp2 in dept_code_variants:
                        filtered_candidates.append(c)

                if filtered_candidates:
                    candidates = filtered_candidates
                    _dbg("geo.analyze.dept_filter_success", after=len(candidates))
                else:
                    _dbg("geo.analyze.dept_filter_no_match", tried=dept_code_variants)

            # NOUVEAU : Si c'est un EPCI (détection de préfixe) et qu'on n'a pas de candidats ou des candidats de mauvaise qualité, utiliser le fallback web
            lieu_lower = lieu.lower()
            is_epci_query = any(prefix in lieu_lower for prefix in ['cc ', 'ca ', 'cu ', 'metropole', 'communaute'])

            if is_epci_query and (not candidates or all(c.get('TYPE_TERRITOIRE') != 'EPCI/Interco' for c in candidates)):
                _dbg("geo.analyze.epci_web_fallback", lieu=lieu, candidates_count=len(candidates) if candidates else 0)

                # Appeler directement le fallback web pour trouver le code SIREN
                web_result = ai_fallback_territory_search(con, lieu)
                if web_result and not web_result.get("needs_user_clarification"):
                    # On a trouvé via web search !
                    return web_result

            if not candidates:
                _dbg("geo.analyze.no_candidates", lieu=lieu)
                debug_info.append({"Recherche": lieu, "Trouvé": "Aucun candidat", "ID": None})
                continue

            # Validation IA pour CE lieu
            ai_decision = ai_validate_territory(client, MODEL_NAME, lieu, candidates, full_sentence_context=rewritten_prompt)

            # VÉRIFICATION D'AMBIGUÏTÉ OU CONFIANCE FAIBLE
            if ai_decision:
                is_ambiguous = ai_decision.get("is_ambiguous", False)
                confidence = ai_decision.get("confidence", "high")
                candidates_for_user = ai_decision.get("candidates_for_user", [])

                # Si ambiguïté OU confiance faible/moyenne, on demande à l'utilisateur
                if is_ambiguous or confidence in ["low", "medium"]:
                    _dbg("geo.analyze.ambiguity_detected", lieu=lieu, is_ambiguous=is_ambiguous, confidence=confidence)

                    # Préparer les candidats pour l'utilisateur
                    if not candidates_for_user:
                        # Si l'IA n'a pas fourni de candidats, on utilise les candidats de la recherche
                        candidates_for_user = []
                        for c in candidates[:5]:  # Limiter à 5
                            type_territoire = c.get('TYPE_TERRITOIRE', 'Autre')
                            comp2 = c.get('COMP2', '')
                            dept_name = ''
                            if comp2 and str(comp2) not in ['', 'None', 'NaN']:
                                # Chercher le nom du département
                                try:
                                    dept_info = con.execute("SELECT NOM_COUV FROM territoires WHERE ID = ?", [str(comp2)]).fetchone()
                                    if dept_info:
                                        dept_name = f" ({dept_info[0]})"
                                except:
                                    pass

                            candidates_for_user.append({
                                "id": str(c['ID']),
                                "name": c['NOM_COUV'],
                                "type": type_territoire,
                                "context": f"{type_territoire}{dept_name}"
                            })

                    # Retourner un résultat spécial qui indique qu'il faut demander à l'utilisateur
                    return {
                        "needs_user_clarification": True,
                        "query": lieu,
                        "candidates": candidates_for_user,
                        "confidence": confidence,
                        "reason": ai_decision.get("reason", "Plusieurs territoires possibles")
                    }

            if ai_decision and ai_decision.get("selected_id"):
                sel_id = str(ai_decision["selected_id"])

                # Recherche du candidat correspondant (avec plusieurs stratégies de matching)
                winner = None

                # Stratégie 1: Match exact
                winner = next((c for c in candidates if str(c['ID']) == sel_id), None)

                # Stratégie 2: Match sans zéro initial
                if not winner:
                    winner = next((c for c in candidates if str(c['ID']).lstrip('0') == sel_id.lstrip('0')), None)

                # Stratégie 3: Match en ignorant préfixe D/R
                if not winner:
                    sel_id_clean = sel_id.replace('D', '').replace('R', '').lstrip('0')
                    for c in candidates:
                        cid_clean = str(c['ID']).replace('D', '').replace('R', '').lstrip('0')
                        if cid_clean == sel_id_clean:
                            winner = c
                            break

                # Stratégie 4: Fallback sur le premier candidat si aucun match
                if not winner and candidates:
                    winner = candidates[0]
                    _dbg("geo.analyze.fallback_first", lieu=lieu, fallback_id=winner['ID'])

                if winner:
                    winner_id = str(winner['ID'])
                    found_ids.append(winner_id)
                    debug_info.append({"Recherche": lieu, "Trouvé": winner['NOM_COUV'], "ID": winner_id, "Confiance": ai_decision.get("confidence", "unknown")})

                    # Premier lieu = cible principale
                    if first_pass:
                        target_id = winner_id
                        target_name = winner['NOM_COUV']
                        # Ajouter les parents (EPCI, Dept, Région) pour comparaison
                        for comp_key in ['COMP1', 'COMP2', 'COMP3']:
                            comp_val = winner.get(comp_key)
                            # Vérifier que la valeur n'est pas None, vide ou NaN
                            if comp_val is not None and comp_val != '' and not pd.isna(comp_val):
                                comp_val_str = str(comp_val).strip()
                                if comp_val_str and comp_val_str.lower() not in ['none', 'nan', 'null']:
                                    found_ids.append(comp_val_str)
                                    _dbg("geo.analyze.parent_added", comp_key=comp_key, comp_val=comp_val_str)
                        first_pass = False
            else:
                # Pas de décision IA ou ambiguïté
                _dbg("geo.analyze.no_decision", lieu=lieu, ai_response=ai_decision)
                debug_info.append({"Recherche": lieu, "Trouvé": "Non résolu", "ID": None})

        except Exception as e_lieu:
            _dbg("geo.analyze.lieu_error", lieu=lieu, error=str(e_lieu))
            debug_info.append({"Recherche": lieu, "Trouvé": f"Erreur: {e_lieu}", "ID": None})
            continue

    # 3. Finalisation
    if not found_ids:
        _dbg("geo.analyze.no_results")
        return None

    # Ajouter France pour référence nationale
    found_ids.append('FR')

    # Dédoublonnage en préservant l'ordre
    unique_ids = list(dict.fromkeys([x for x in found_ids if x and str(x).lower() not in ['none', 'nan', 'null', '']]))

    result = {
        "target_name": target_name or lieux_cites[0],
        "target_id": target_id or unique_ids[0],
        "all_ids": unique_ids,
        "parent_clause": "",
        "display_context": ", ".join(lieux_cites),
        "debug_search": debug_info,
        "lieux_cites": lieux_cites
    }

    _dbg("geo.analyze.result", target=result["target_name"], ids_count=len(unique_ids), all_ids=unique_ids)
    return result

# --- 8.1. PALETTES & CARTES EPCI (VEGA LITE) ---
BASE_PALETTE = ["#EB2C30", "#F38331", "#97D422", "#1DB5C5", "#5C368D"]
EXTRA_PALETTE = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
    "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC", "#2F4B7C",
    "#A05195", "#D45087", "#F95D6A", "#FFA600"
]

@st.cache_data(ttl=3600)  # Cache pendant 1 heure pour améliorer les performances
def fetch_geojson(url):
    """Télécharge et met en cache le GeoJSON pour éviter les téléchargements répétés."""
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            if response.status != 200:
                return None
            return json.loads(response.read().decode("utf-8"))
    except Exception as e:
        _dbg("map.fetch.error", url=url, error=str(e))
        return None


def render_epci_choropleth(
    con,
    df,
    commune_id,
    commune_name,
    metric_col,
    metric_spec,
    diagnostic=True,
    sql_query=None,
    in_sidebar=False
):
    _dbg(
        "map.render.start",
        commune_id=commune_id,
        commune_name=commune_name,
        metric_col=metric_col,
        df_cols=list(df.columns),
        df_rows=len(df),
        df_ids=df["ID"].astype(str).unique().tolist()[:10] if "ID" in df.columns else []
    )

    # Détecter si l'ID passé est une commune ou un EPCI
    commune_id_str = str(commune_id)
    is_epci = is_valid_insee_code(commune_id_str) and len(commune_id_str) == 9

    if is_epci:
        # L'ID est déjà un EPCI, on l'utilise directement
        epci_id = commune_id_str
        epci_name_row = con.execute(
            "SELECT NOM_COUV FROM territoires WHERE ID = ? LIMIT 1",
            [epci_id]
        ).fetchone()
        epci_name = epci_name_row[0] if epci_name_row else commune_name
        _dbg("map.epci.direct", epci_id=epci_id, epci_name=epci_name)
    else:
        # L'ID est une commune, on récupère son EPCI
        try:
            epci_id = con.execute(
                "SELECT COMP1 FROM territoires WHERE ID = ? LIMIT 1",
                [commune_id_str]
            ).fetchone()
        except Exception as e:
            _dbg("map.epci.query_error", commune_id=commune_id, error=str(e))
            st.warning("Impossible de récupérer l'EPCI pour cette commune.")
            return

        if not epci_id or not epci_id[0]:
            _dbg("map.epci.missing", commune_id=commune_id)
            st.info("Aucun EPCI disponible pour cette commune.")
            return

        epci_id = str(epci_id[0])
        _dbg("map.epci.found", commune_id=commune_id, epci_id=epci_id)
        epci_name_row = con.execute(
            "SELECT NOM_COUV FROM territoires WHERE ID = ? LIMIT 1",
            [epci_id]
        ).fetchone()
        epci_name = epci_name_row[0] if epci_name_row else epci_id

    if metric_col not in df.columns:
        _dbg("map.metric.missing", metric_col=metric_col, df_cols=list(df.columns))
        st.info("La carte choroplèthe n'est pas disponible pour cet indicateur.")
        if diagnostic:
            st.caption(
                f"Diagnostic : colonne '{metric_col}' absente des données retournées."
            )
        return

    # Essayer d'abord de récupérer le GeoJSON EPCI
    geojson = fetch_geojson(
        f"https://geo.api.gouv.fr/epcis/{epci_id}/communes?format=geojson&geometry=contour&fields=code,nom"
    )

    # Variable pour tracker si on est en mode fallback département
    is_dept_fallback = False
    dept_code = None
    territory_name = epci_name

    if not geojson:
        _dbg("map.geojson.unavailable", epci_id=epci_id)
        # FALLBACK : Essayer avec le département (pour les EPT d'Ile-de-France)
        try:
            dept_row = con.execute(
                "SELECT COMP2, NOM_COUV FROM territoires WHERE ID = ? LIMIT 1",
                [epci_id]
            ).fetchone()
            if dept_row and dept_row[0] and dept_row[0].startswith("D"):
                dept_code = dept_row[0][1:]  # Enlever le "D"
                _dbg("map.geojson.fallback_dept", epci_id=epci_id, dept_code=dept_code)
                geojson = fetch_geojson(
                    f"https://geo.api.gouv.fr/departements/{dept_code}/communes?format=geojson&geometry=contour&fields=code,nom"
                )
                if geojson:
                    is_dept_fallback = True
                    # Récupérer le nom du département
                    dept_name_row = con.execute(
                        "SELECT NOM_COUV FROM territoires WHERE ID = ? LIMIT 1",
                        [f"D{dept_code}"]
                    ).fetchone()
                    if dept_name_row:
                        territory_name = dept_name_row[0]
                    _dbg("map.geojson.fallback_success", epci_id=epci_id, dept_code=dept_code, features=len(geojson.get("features", [])))
        except Exception as e:
            _dbg("map.geojson.fallback_error", epci_id=epci_id, error=str(e))

    if not geojson:
        st.warning("Le fond de carte des communes EPCI est indisponible pour le moment.")
        if diagnostic:
            st.caption(
                "Diagnostic : GeoJSON indisponible via geo.api.gouv.fr (réseau ou service temporairement bloqué)."
            )
        return
    _dbg(
        "map.geojson.loaded",
        epci_id=epci_id,
        features=len(geojson.get("features", [])),
        is_dept_fallback=is_dept_fallback
    )

    # Récupérer les IDs des communes selon le mode (EPCI ou département)
    if is_dept_fallback:
        # Mode département : récupérer toutes les communes du département
        commune_ids = [
            row[0]
            for row in con.execute(
                "SELECT ID FROM territoires WHERE COMP2 = ? AND length(ID) IN (4, 5)",
                [f"D{dept_code}"]
            ).fetchall()
        ]
        _dbg("map.dept.commune_ids", dept_code=dept_code, count=len(commune_ids))
    else:
        # Mode EPCI : récupérer les communes de l'EPCI
        commune_ids = [
            row[0]
            for row in con.execute(
                "SELECT ID FROM territoires WHERE COMP1 = ? AND length(ID) IN (4, 5)",
                [epci_id]
            ).fetchall()
        ]
        _dbg("map.epci.commune_ids", epci_id=epci_id, count=len(commune_ids))

    # 🔧 TOUJOURS requêter les données pour toutes les communes
    df_epci_source = None

    # Méthode 1 : Si sql_query est fourni, l'adapter pour inclure toutes les communes
    if sql_query:
        ids_sql = ", ".join([f"'{str(cid)}'" for cid in commune_ids])
        epci_sql = re.sub(
            r'(WHERE\s*\(t\."ID"\s+IN\s*)\([^\)]*\)',
            rf'\1({ids_sql})',
            sql_query,
            flags=re.IGNORECASE
        )
        if epci_sql != sql_query:
            try:
                df_epci_source = con.execute(epci_sql).df()
                _dbg(
                    "map.data.sql_refetch",
                    territory_id=epci_id,
                    is_dept=is_dept_fallback,
                    rows=len(df_epci_source),
                    sql_preview=epci_sql[:300]
                )
            except Exception as e:
                _dbg("map.data.sql_refetch_error", territory_id=epci_id, error=str(e))
        else:
            _dbg("map.data.sql_refetch_skip", territory_id=epci_id, reason="no_where_match")

    # Méthode 2 : Si sql_query n'a pas fonctionné, construire une requête automatique
    if df_epci_source is None or df_epci_source.empty:
        _dbg("map.data.auto_query", territory_id=epci_id, metric_col=metric_col, is_dept=is_dept_fallback)

        # Trouver la table qui contient metric_col
        valid_tables = st.session_state.get("valid_tables_list", [])
        db_schemas = st.session_state.get("db_schemas", {})

        target_table = None
        for table_name in valid_tables:
            if table_name in db_schemas:
                cols = db_schemas[table_name]
            else:
                try:
                    cols = [c[0] for c in con.execute(f"DESCRIBE \"{table_name}\"").fetchall()]
                except:
                    continue

            if metric_col in cols:
                target_table = table_name
                _dbg("map.data.table_found", table=table_name, metric=metric_col)
                break

        if target_table:
            # Construire une requête simple pour récupérer les données
            ids_sql = ", ".join([f"'{str(cid)}'" for cid in commune_ids])
            auto_sql = f"""
                SELECT t.ID, t.NOM_COUV, d."{metric_col}"
                FROM territoires t
                LEFT JOIN "{target_table}" d ON t.ID = d.ID
                WHERE t.ID IN ({ids_sql})
            """

            try:
                df_epci_source = con.execute(auto_sql).df()
                _dbg(
                    "map.data.auto_query_success",
                    territory_id=epci_id,
                    is_dept=is_dept_fallback,
                    table=target_table,
                    rows=len(df_epci_source),
                    sql_preview=auto_sql[:300]
                )
            except Exception as e:
                _dbg("map.data.auto_query_error", territory_id=epci_id, error=str(e))
        else:
            _dbg("map.data.table_not_found", metric_col=metric_col)

    # Fallback : utiliser le DataFrame passé en paramètre si tout a échoué
    if df_epci_source is None:
        df_epci_source = df
        _dbg("map.data.fallback_to_input_df")

    df_epci = df_epci_source[
        df_epci_source["ID"].astype(str).isin([str(cid) for cid in commune_ids])
    ].copy()
    _dbg(
        "map.data.filtered",
        territory_id=epci_id,
        is_dept=is_dept_fallback,
        df_epci_rows=len(df_epci),
        df_epci_ids=df_epci["ID"].astype(str).unique().tolist()[:10] if "ID" in df_epci.columns else []
    )
    _dbg(
        "map.data.id_types",
        df_id_dtype=str(df["ID"].dtype) if "ID" in df.columns else None,
        df_epci_id_dtype=str(df_epci["ID"].dtype) if "ID" in df_epci.columns else None,
        commune_id_type=str(type(commune_id)),
        territory_id_type=str(type(epci_id))
    )
    if metric_col in df_epci.columns:
        df_epci["valeur"] = pd.to_numeric(df_epci[metric_col], errors="coerce")
    else:
        df_epci["valeur"] = pd.Series(dtype="float64")

    if df_epci.empty:
        _dbg("map.data.empty", territory_id=epci_id, metric_col=metric_col, is_dept=is_dept_fallback)
        territory_type = "département" if is_dept_fallback else "EPCI"
        st.info(f"Aucune donnée disponible pour les communes de ce {territory_type}.")
        if diagnostic:
            st.caption(
                f"Diagnostic : aucune commune trouvée dans les données pour '{metric_col}'."
            )
        return
    if df_epci["valeur"].notna().sum() == 0:
        _dbg("map.data.no_values", territory_id=epci_id, metric_col=metric_col, is_dept=is_dept_fallback)
        territory_type = "département" if is_dept_fallback else "EPCI"
        st.info(f"Aucune valeur exploitable pour les communes de ce {territory_type}.")
        if diagnostic:
            st.caption(
                f"Diagnostic : toutes les valeurs de '{metric_col}' sont nulles ou non numériques."
            )
        return

    _dbg(
        "map.data.numeric",
        territory_id=epci_id,
        metric_col=metric_col,
        is_dept=is_dept_fallback,
        non_null=df_epci["valeur"].notna().sum(),
        non_numeric=(df_epci["valeur"].isna().sum()),
        sample_values=df_epci["valeur"].dropna().head(5).tolist()
    )
    value_map = {}
    for _, row in df_epci.iterrows():
        if pd.notna(row["valeur"]):
            code = str(row["ID"])
            # Ajouter un zéro initial si le code a 4 caractères (ex: 4112 → 04112)
            if len(code) == 4:
                code = "0" + code
            value_map[code] = row["valeur"]
    _dbg(
        "map.data.stats",
        epci_id=epci_id,
        rows=len(df_epci),
        values=len(value_map),
        min_value=df_epci["valeur"].min(),
        max_value=df_epci["valeur"].max()
    )
    missing_values = []
    for feature in geojson.get("features", []):
        code = str(feature.get("properties", {}).get("code", ""))
        # Normaliser le code pour la comparaison
        normalized_code = ("0" + code) if len(code) == 4 else code
        if normalized_code not in value_map:
            missing_values.append(code)
    _dbg(
        "map.data.coverage",
        epci_id=epci_id,
        features=len(geojson.get("features", [])),
        values=len(value_map),
        missing=len(missing_values),
        missing_sample=missing_values[:10]
    )
    if diagnostic and missing_values:
        territory_type = "le département" if is_dept_fallback else "l'EPCI"
        st.caption(
            f"Diagnostic : données disponibles pour {len(value_map)} commune(s) sur "
            f"{len(geojson.get('features', []))} dans {territory_type}."
        )
    for feature in geojson.get("features", []):
        code = str(feature.get("properties", {}).get("code", ""))
        # Normaliser le code pour la recherche dans value_map
        normalized_code = ("0" + code) if len(code) == 4 else code
        feature.setdefault("properties", {})["value"] = value_map.get(normalized_code)

    metric_label = metric_spec.get("label", metric_col)
    metric_title = metric_spec.get("title", metric_label)
    kind = (metric_spec.get("kind") or "").lower()
    if kind == "percent":
        metric_format = ".1%"
    else:
        metric_format = ",.0f"

    _dbg(
        "map.render.ready",
        epci_id=epci_id,
        epci_name=epci_name,
        metric_col=metric_col,
        metric_format=metric_format
    )

    # Calcul du centre de la carte
    coords = []
    for feature in geojson.get("features", []):
        geometry = feature.get("geometry", {})
        if geometry.get("type") == "Polygon":
            for ring in geometry.get("coordinates", []):
                coords.extend(ring)
        elif geometry.get("type") == "MultiPolygon":
            for polygon in geometry.get("coordinates", []):
                for ring in polygon:
                    coords.extend(ring)

    if coords:
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        center_lat = (min(lats) + max(lats)) / 2
        center_lon = (min(lons) + max(lons)) / 2
    else:
        center_lat, center_lon = 49.9, 2.3  # Fallback

    _dbg("map.render.folium", epci_id=epci_id, metric_col=metric_col, center_lat=center_lat, center_lon=center_lon)

    # Création de la carte Folium avec fond neutre
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles="CartoDB positron"  # Fond plus neutre et moins coloré
    )

    from branca.element import Element

    css = """
    <style>
      .leaflet-container { font-size: 12px !important; }
      .leaflet-tooltip { font-size: 12px !important; }
      .leaflet-popup-content { font-size: 12px !important; }
      .leaflet-control-attribution,
      .leaflet-control-attribution a { font-size: 10px !important; }
      .leaflet-bottom { top:0 !important; }
    </style>
    """
    m.get_root().header.add_child(Element(css))

    # 🔧 Ajouter le contour de l'EPCI (sans remplissage) - seulement si pas en mode département
    if not is_dept_fallback:
        try:
            epci_geojson = fetch_geojson(
                f"https://geo.api.gouv.fr/epcis/{epci_id}?format=geojson&geometry=contour"
            )
            if epci_geojson:
                folium.GeoJson(
                    epci_geojson,
                    style_function=lambda x: {
                        'fillColor': 'none',
                        'fillOpacity': 0,
                        'color': '#808080',  # Gris
                        'weight': 3,
                        'opacity': 0.8
                    },
                    name="Contour EPCI"
                ).add_to(m)
        except:
            pass  # Si le contour n'est pas disponible, on continue sans

    # Préparation des données pour Choropleth
    # Folium Choropleth attend un dict {code: value}
    choropleth_data = {}
    for feature in geojson.get("features", []):
        code = feature.get("properties", {}).get("code")
        value = feature.get("properties", {}).get("value")
        if code and value is not None:
            choropleth_data[code] = value

    # Créer la palette verte avec catégories distinctes (comme dans l'image)
    colors = ["#cfe8cf", "#a9d5a9", "#7fc77f", "#56a35a"]
    values_list = list(choropleth_data.values())

    if values_list:
        vmin = min(values_list)
        vmax = max(values_list)

        # Créer 4 catégories égales (quartiles)
        quartiles = [vmin + (vmax - vmin) * i / 4 for i in range(5)]

        def get_color(value):
            """Retourne la couleur selon la catégorie"""
            if value is None or pd.isna(value):
                return "#e0e0e0"  # Gris pour données manquantes
            elif value < quartiles[1]:
                return colors[0]  # Vert très clair
            elif value < quartiles[2]:
                return colors[1]  # Vert clair
            elif value < quartiles[3]:
                return colors[2]  # Vert moyen
            else:
                return colors[3]  # Vert foncé

        # Fonction de style pour chaque feature
        def style_function(feature):
            code = feature['properties'].get('code')
            value = choropleth_data.get(code)
            return {
                'fillColor': get_color(value),
                'fillOpacity': 0.75,
                'color': 'white',
                'weight': 1.5,
                'opacity': 1
            }

        # Ajouter la couche GeoJson stylisée
        folium.GeoJson(
            geojson,
            style_function=style_function,
            name="choropleth"
        ).add_to(m)
    else:
        # Pas de données, afficher en gris
        def style_function(feature):
            return {
                'fillColor': '#e0e0e0',
                'fillOpacity': 0.5,
                'color': 'white',
                'weight': 1.5
            }

        folium.GeoJson(
            geojson,
            style_function=style_function,
            name="choropleth"
        ).add_to(m)

    # Fonction de formatage français
    def fr_num(x, decimals=1, suffix="", factor=1):
        if pd.isna(x): return "-"
        if not isinstance(x, (int, float)): return str(x)
        try:
            val = x * factor
            fmt = f"{{:,.{decimals}f}}"
            s = fmt.format(val).replace(",", " ").replace(".", ",")
            return (s + (f" {suffix}" if suffix else "")).strip()
        except: return str(x)

    # ---------- FORMATAGE / FACTEUR % (FOURNI PAR L'IA) ----------
    # On calcule valid_values dans tous les cas pour éviter NameError
    all_values = [f.get("properties", {}).get("value") for f in geojson.get("features", [])]
    valid_values = [v for v in all_values if v is not None and not pd.isna(v)]

    # ✅ Utiliser le percent_factor fourni par l'IA (plus d'heuristique !)
    percent_factor = int(metric_spec.get("percent_factor", 1))  # 1 ou 100 selon l'IA
    # ✅ Utiliser les décimales fournies par l'IA (comme pour les graphiques)
    decimals = int(metric_spec.get("decimals", 1))
    _dbg("map.percent.factor", factor=percent_factor, decimals=decimals, sample_values=valid_values[:5] if valid_values else [])

    # ---------- TOOLTIP ----------
    for feature in geojson.get("features", []):
        nom = feature.get("properties", {}).get("nom", "")
        value = feature.get("properties", {}).get("value")

        if value is not None:
            if kind == "percent":
                value_str = fr_num(value, decimals=decimals, suffix="%", factor=percent_factor)
            else:
                value_str = fr_num(value, decimals=decimals)

            tooltip_text = f"<b>{nom}</b><br>{metric_title}: {value_str}"

            folium.GeoJson(
                feature,
                style_function=lambda x: {"fillOpacity": 0, "weight": 0},
                tooltip=folium.Tooltip(tooltip_text, style="width:128px; white-space:normal;")
            ).add_to(m)

    # ---------- LÉGENDE ----------
    if valid_values and values_list:
        min_val = min(valid_values)
        max_val = max(valid_values)

        # Calculer les quartiles pour la légende
        quartiles = [min_val + (max_val - min_val) * i / 4 for i in range(5)]

        # Formater les valeurs pour chaque catégorie
        def format_legend_value(val):
            if kind == "percent":
                return fr_num(val, decimals=decimals, suffix="%", factor=percent_factor)
            else:
                # 🔧 FIX: Pas de suffix pour les nombres (suffix n'était pas défini)
                return fr_num(val, decimals=decimals, suffix="")

        cat1_label = f"Moins de {format_legend_value(quartiles[1])}"
        cat2_label = f"De {format_legend_value(quartiles[1])} à {format_legend_value(quartiles[2])}"
        cat3_label = f"De {format_legend_value(quartiles[2])} à {format_legend_value(quartiles[3])}"
        cat4_label = f"Plus de {format_legend_value(quartiles[3])}"

        # 🔧 Légende avec titre, en bas, 120px de hauteur
        # Support du multiligne pour les titres longs
        if len(metric_title) > 50:
            words = metric_title.split()
            lines = []
            current_line = ""

            for word in words:
                if len(current_line) + len(word) + 1 <= 50:
                    current_line += (" " if current_line else "") + word
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word

            if current_line:
                lines.append(current_line)

            metric_title_html = "<br>".join(lines)
        else:
            metric_title_html = metric_title

        legend_html = f'''
        <div style="position: absolute;
                    bottom: 0; left: 0; right: 0; width: 100%;
                    height: 100px;
                    background-color: white; z-index:9999;
                    padding: 2px;
                    box-sizing: border-box;">
            <div style="font-weight: bold; font-size: 12px; margin-bottom: 8px; color: #333;">
                {metric_title_html}
            </div>
            <div style="display: grid; gap: 8px; grid-template-columns: 30% 30% 30%; grid-template-rows: 16px 16px;">
                <div style="display: flex; align-items: center; gap: 4px;">
                    <div style="width: 18px; height: 18px; background-color: {colors[0]}; border: 1px solid #ccc;"></div>
                    <span style="font-size: 10px; color: #555;">{cat1_label}</span>
                </div>
                <div style="display: flex; align-items: center; gap: 4px;">
                    <div style="width: 18px; height: 18px; background-color: {colors[2]}; border: 1px solid #ccc;"></div>
                    <span style="font-size: 10px; color: #555;">{cat3_label}</span>
                </div>
                <div style="display: flex; align-items: center; gap: 4px;">
                    <div style="width: 18px; height: 18px; background-color: #e0e0e0; border: 1px solid #ccc;"></div>
                    <span style="font-size: 10px; color: #555;">Données non disponibles</span>
                </div>
                <div style="display: flex; align-items: center; gap: 4px;">
                    <div style="width: 18px; height: 18px; background-color: {colors[1]}; border: 1px solid #ccc;"></div>
                    <span style="font-size: 10px; color: #555;">{cat2_label}</span>
                </div>
                <div style="display: flex; align-items: center; gap: 4px;">
                    <div style="width: 18px; height: 18px; background-color: {colors[3]}; border: 1px solid #ccc;"></div>
                    <span style="font-size: 10px; color: #555;">{cat4_label}</span>
                </div>
            </div>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))

    # ---------- AFFICHAGE STREAMLIT (IMPORTANT) ----------
    # 🔧 Adapter la taille selon le contexte (sidebar ou main)
    map_height = 300
    make_map_responsive= """
     <style>
     .stCustomComponentV1 {
         width: 100%; 
         height: 400px;
         } 
     </style>
    """
    st.markdown(make_map_responsive, unsafe_allow_html=True)
    st_folium(m, height=map_height, returned_objects=[], width='stretch')


# --- 8. VISUALISATION AUTO (HEURISTIQUE %) ---
def auto_plot_data(df, sorted_ids, config=None, con=None, in_sidebar=False):
    if config is None: config = {}
    # Log supprimé pour réduire verbosité

    selected_metrics = config.get("selected_columns", [])
    format_specs = config.get("formats", {})
    
    base_palette = BASE_PALETTE
    extra_palette = EXTRA_PALETTE
    
    cols = df.columns.tolist()
    label_col = next((c for c in cols if c.upper() in ["NOM_COUV", "TERRITOIRE", "LIBELLE", "VILLE"]), None)
    date_col = next((c for c in cols if c.upper() in ["AN", "ANNEE", "YEAR", "DATE"]), None)
    id_col = next((c for c in cols if c.upper() == "ID"), None)

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
                db_schemas = st.session_state.get("db_schemas", {})

                # Recherche de la table contenant la population (EVO prioritaire, puis SUP)
                pop_table = None
                pop_col = None

                # Liste des tables candidates pour la population
                candidate_tables = [t for t in valid_tables if any(x in t for x in ["EVO", "SUP", "POP"])]

                for table_name in candidate_tables:
                    if table_name in db_schemas:
                        cols = db_schemas[table_name]
                    else:
                        try:
                            cols = [c[0] for c in con.execute(f"DESCRIBE \"{table_name}\"").fetchall()]
                        except:
                            continue

                    # Chercher une colonne population récente
                    cols_upper = [c.upper() for c in cols]
                    pop_candidates = [c for c, cu in zip(cols, cols_upper)
                                      if ("POP" in cu or "PMUN" in cu or "PTOT" in cu)
                                      and any(char.isdigit() for char in c)]

                    if pop_candidates:
                        # Trier par année décroissante (P22 > P20 > P16)
                        pop_candidates_sorted = sorted(pop_candidates, key=lambda x: ''.join(filter(str.isdigit, x)), reverse=True)
                        pop_col = pop_candidates_sorted[0]
                        pop_table = table_name
                        break

                    # Fallback sur colonne générique
                    if not pop_col:
                        for c, cu in zip(cols, cols_upper):
                            if cu in ["POP", "PMUN", "PTOT", "POPULATION", "POP_MUNI", "POP_MOCO_40"]:
                                pop_col = c
                                pop_table = table_name
                                break
                        if pop_col:
                            break

                if pop_table and pop_col:
                    ids_sql = ", ".join([f"'{i}'" for i in comparators])
                    q_sort = f"""
                        SELECT t.ID
                        FROM territoires t
                        LEFT JOIN "{pop_table}" e ON t.ID = e.ID
                        WHERE t.ID IN ({ids_sql})
                        ORDER BY TRY_CAST(e."{pop_col}" AS DOUBLE) ASC
                    """
                    try:
                        sorted_result = con.execute(q_sort).fetchall()
                        if sorted_result:
                            comparators = [str(x[0]) for x in sorted_result]
                            _dbg("plot.sort", status="success", table=pop_table, col=pop_col)
                    except Exception as e_sort:
                        _dbg("plot.sort", status="query_failed", error=str(e_sort))
                else:
                    _dbg("plot.sort", status="failed", reason="Colonne population introuvable")
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

    # ✅ Utiliser les specs fournies par l'IA
    is_percent = spec.get("kind") == "percent"
    is_currency = spec.get("kind") in ["currency", "euro"]
    decimals = int(spec.get("decimals", 1))
    percent_factor = int(spec.get("percent_factor", 1))  # 1 ou 100 selon l'IA

    # Utiliser chart_title si disponible et multi-métrique, sinon le titre de la première métrique
    has_multiple_metrics = len(selected_metrics) > 1
    if has_multiple_metrics and config.get("chart_title"):
        title_y = config.get("chart_title")
    else:
        title_y = spec.get("title", spec.get("label", "Valeur"))

    # Ajouter "(en €)" au titre si c'est une donnée monétaire
    if is_currency and "(en €)" not in title_y and "€" not in title_y:
        title_y = f"{title_y} (en €)"

    # ✅ Format basé sur les specs IA (plus d'heuristique sur les valeurs !)
    y_suffix = ""

    if is_percent:
        # Pour les pourcentages : selon le percent_factor
        if percent_factor == 100:
            # Base 1 (0-1) -> format Vega qui multiplie automatiquement
            y_format = f".{decimals}%"
            y_suffix = ""  # Le % est géré par Vega
        else:
            # Base 100 (déjà en %) -> afficher tel quel + %
            y_format = f",.{decimals}f"
            y_suffix = " %"
    elif is_currency:
        y_format = f",.{decimals}f"
        y_suffix = " €"
    else:
        # Nombres normaux
        y_format = f",.{decimals}f"
        y_suffix = ""

    # 6. MELT
    id_vars = [label_col]
    if date_col: id_vars.append(date_col)
    df_melted = df_plot.melt(id_vars=id_vars, value_vars=new_selected_metrics, var_name="Indicateur", value_name="Valeur")

    # Conversion explicite en numérique (crucial pour Vega-Lite)
    df_melted["Valeur"] = pd.to_numeric(df_melted["Valeur"], errors='coerce')

    # Pour les graphiques temporels, ne garder que le territoire cible + France
    if date_col and id_col:
        target_id = candidates[0] if candidates else None
        # Trouver l'ID de la France (FR ou France métropolitaine)
        france_ids = [uid for uid in available_ids if str(uid) in ['FR', 'FRMETRO', 'FXX']]
        keep_ids = [target_id] if target_id else []
        if france_ids:
            keep_ids.extend(france_ids[:1])  # Ajouter seulement la première France trouvée
        df_melted = df_melted[df_melted[label_col].isin(
            df_plot[df_plot[id_col].isin(keep_ids)][label_col].unique()
        )]

        # Normaliser la courbe France pour comparaison avec le territoire cible
        territories_in_data = df_melted[label_col].unique()
        if len(territories_in_data) == 2:
            # Identifier le territoire cible et la France
            france_label = [lbl for lbl in territories_in_data if "France" in lbl or "FR" in lbl]
            target_label = [lbl for lbl in territories_in_data if lbl not in france_label]

            if france_label and target_label:
                france_label = france_label[0]
                target_label = target_label[0]

                # Calculer le ratio moyen pour normaliser
                target_mean = df_melted[df_melted[label_col] == target_label]["Valeur"].mean()
                france_mean = df_melted[df_melted[label_col] == france_label]["Valeur"].mean()

                if france_mean > 0:
                    ratio = target_mean / france_mean
                    # Appliquer le ratio aux valeurs de France pour mise à l'échelle
                    df_melted.loc[df_melted[label_col] == france_label, "Valeur"] *= ratio

                # Renommer "France" en "Tendance France" dans la légende
                df_melted.loc[df_melted[label_col] == france_label, label_col] = f"Tendance {france_label}"

    # 8. VEGA
    is_multi_metric = len(new_selected_metrics) > 1
    # Toujours utiliser des graphiques groupés, jamais empilés
    is_stacked = False
    # Désactivation de l'échelle logarithmique
    y_scale = None

    vega_config = {
        "locale": {"number": {"decimal": ",", "thousands": "\u00a0", "grouping": [3]}},
        "axis": {
            "labelFontSize": 11,
            "titleFontSize": 12,
            "labelColor": "#2c3e50",
            "titleColor": "#2c3e50",
            "gridColor": "#e8ecf0",
            "gridOpacity": 0.5,
            "domainColor": "#cbd5e0"
        },
        "legend": {
            "labelFontSize": 11,
            "titleFontSize": 12,
            "labelColor": "#2c3e50",
            "titleColor": "#2c3e50",
            "orient": "bottom",
            "layout": {"bottom": {"anchor": "middle"}}
        },
        "title": {
            "color": "#2c3e50",
            "fontSize": 14
        },
        "background": "white"
    }
    color_domain = sorted_labels
    if is_multi_metric and is_stacked:
        color_domain = new_selected_metrics
    palette = base_palette + extra_palette
    if len(color_domain) > len(palette):
        palette = palette * ((len(color_domain) // len(palette)) + 1)
    color_def = {
        "field": label_col,
        "type": "nominal",
        "scale": {"domain": color_domain, "range": palette[:len(color_domain)]},
        "title": None,
        "legend": {"orient": "bottom"}
    }
    chart = None

    if date_col:
        # Calculer le domaine dynamique pour l'axe Y
        values = df_melted["Valeur"].dropna()
        if not values.empty:
            y_min = values.min()
            y_max = values.max()
            # Ajouter une marge de 20% en haut et en bas
            margin = (y_max - y_min) * 0.2
            y_domain = [y_min - margin, y_max + margin]
        else:
            y_domain = None

        y_axis_def = {"field": "Valeur", "type": "quantitative", "title": None, "axis": {"format": y_format}}
        # 🔧 Ajouter le suffixe (%, €) si nécessaire
        if y_suffix:
            y_axis_def["axis"]["labelExpr"] = f"format(datum.value, '{y_format}') + '{y_suffix}'"
        if y_domain:
            y_axis_def["scale"] = {"domain": y_domain}
        elif y_scale:
            y_axis_def["scale"] = y_scale

        # Couleurs spécifiques pour les courbes : bleu turquoise (cible) et orange (France)
        labels_in_data = df_melted[label_col].unique().tolist()
        color_map_line = []
        for lbl in labels_in_data:
            if "Tendance" in lbl or "France" in lbl or "FR" in lbl:
                color_map_line.append("#F38331")  # Orange pour France
            else:
                color_map_line.append("#1DB5C5")  # Bleu turquoise pour cible

        # Créer deux layers: un pour cible (avec points) et un pour France (sans points)
        base_encoding = {
            "x": {"field": date_col, "type": "ordinal", "title": "Année"},
            "y": y_axis_def,
            "color": {
                "field": label_col,
                "type": "nominal",
                "scale": {"domain": labels_in_data, "range": color_map_line},
                "title": None,
                "legend": {"orient": "bottom", "layout": {"bottom": {"anchor": "middle"}}}
            }
        }
        if is_multi_metric: base_encoding["strokeDash"] = {"field": "Indicateur", "title": "Variable"}

        # Layer pour le territoire cible (avec points et tooltip)
        target_label = [lbl for lbl in labels_in_data if "Tendance" not in lbl and "France" not in lbl and "FR" not in lbl]
        france_label = [lbl for lbl in labels_in_data if "Tendance" in lbl or "France" in lbl or "FR" in lbl]

        layers = []
        if target_label:
            target_encoding = base_encoding.copy()
            target_encoding["tooltip"] = [{"field": label_col, "title": "Nom"}, {"field": "Indicateur", "title": "Variable"}, {"field": date_col}, {"field": "Valeur", "format": y_format}]
            layers.append({
                "transform": [{"filter": f"datum['{label_col}'] == '{target_label[0]}'"}],
                "mark": {"type": "line", "point": True, "tooltip": True},
                "encoding": target_encoding
            })

        if france_label:
            # Pour Tendance France : pas de tooltip
            france_encoding = base_encoding.copy()
            layers.append({
                "transform": [{"filter": f"datum['{label_col}'] == '{france_label[0]}'"}],
                "mark": {"type": "line", "point": False, "tooltip": False},
                "encoding": france_encoding
            })

        chart = {"config": vega_config, "layer": layers}
    else:
        if is_multi_metric and is_stacked:
            y_stack = "normalize" if is_percent else True
            y_axis_def = {"field": "Valeur", "type": "quantitative", "title": None, "axis": {"format": y_format}, "stack": y_stack}
            # 🔧 Ajouter le suffixe (%, €) si nécessaire
            if y_suffix:
                y_axis_def["axis"]["labelExpr"] = f"format(datum.value, '{y_format}') + '{y_suffix}'"
            if y_scale: y_axis_def["scale"] = y_scale
            chart_encoding = {
                "x": {"field": label_col, "type": "nominal", "sort": sorted_labels, "axis": {"labelAngle": 0, "labelLimit": 80, "labelLineHeight": 12, "labelAlign": "center"}, "title": None},
                "y": y_axis_def,
                "color": {"field": "Indicateur", "type": "nominal", "title": None, "scale": {"domain": new_selected_metrics, "range": palette[:len(new_selected_metrics)]}, "legend": {"orient": "bottom", "layout": {"bottom": {"anchor": "middle"}}}},
                "tooltip": [{"field": label_col, "title": "Nom"}, {"field": "Indicateur", "title": "Variable"}, {"field": "Valeur", "format": y_format}]
            }
        elif is_multi_metric:
            y_axis_def = {"field": "Valeur", "type": "quantitative", "title": None, "axis": {"format": y_format}}
            # 🔧 Ajouter le suffixe (%, €) si nécessaire
            if y_suffix:
                y_axis_def["axis"]["labelExpr"] = f"format(datum.value, '{y_format}') + '{y_suffix}'"
            if y_scale: y_axis_def["scale"] = y_scale
            # Ajouter layout à color_def pour ce cas
            color_def_multi = color_def.copy()
            color_def_multi["legend"] = {"orient": "bottom", "layout": {"bottom": {"anchor": "middle"}}}
            chart_encoding = {
                "x": {"field": "Indicateur", "type": "nominal", "axis": {"labelAngle": 0, "labelLimit": 80, "labelLineHeight": 12, "labelAlign": "center", "title": None}},
                "y": y_axis_def,
                "color": color_def_multi,
                "xOffset": {"field": label_col},
                "tooltip": [{"field": label_col, "title": "Nom"}, {"field": "Indicateur", "title": "Variable"}, {"field": "Valeur", "format": y_format}]
            }
        else:
            y_axis_def = {"field": "Valeur", "type": "quantitative", "title": None, "axis": {"format": y_format}}
            # 🔧 Ajouter le suffixe (%, €) si nécessaire
            if y_suffix:
                y_axis_def["axis"]["labelExpr"] = f"format(datum.value, '{y_format}') + '{y_suffix}'"
            if y_scale: y_axis_def["scale"] = y_scale
            bar_colors = palette[:len(sorted_labels)] if sorted_labels else palette[:1]
            chart_encoding = {
                "x": {"field": label_col, "type": "nominal", "sort": sorted_labels, "axis": {"labelAngle": 0, "labelLimit": 80, "labelLineHeight": 12, "labelAlign": "center"}, "title": None},
                "y": y_axis_def,
                "color": {
                    "field": label_col,
                    "type": "nominal",
                    "scale": {"domain": sorted_labels, "range": bar_colors},
                    "legend": None
                },
                "tooltip": [{"field": label_col, "title": "Nom"}, {"field": "Valeur", "format": y_format}]
            }
        chart = {"config": vega_config, "mark": {"type": "bar", "cornerRadiusEnd": 3, "tooltip": True}, "encoding": chart_encoding}

    # Ajouter le titre en haut du graphique (centré) avec support du multiligne
    # Si le titre est trop long (> 50 caractères), on le découpe sur plusieurs lignes
    if len(title_y) > 50:
        # Découper intelligemment en gardant les mots entiers
        words = title_y.split()
        lines = []
        current_line = ""

        for word in words:
            if len(current_line) + len(word) + 1 <= 50:
                current_line += (" " if current_line else "") + word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        chart["title"] = {
            "text": lines,  # Vega-Lite accepte un tableau pour le multiligne
            "anchor": "middle",
            "fontSize": 14
        }
    else:
        chart["title"] = {
            "text": title_y,
            "anchor": "middle",
            "fontSize": 14
        }

    # Adapter la hauteur selon le contexte (sidebar ou main)
    chart["height"] = 247 if in_sidebar else 400

    # Limiter la largeur dans la sidebar
    if in_sidebar:
        chart["width"] = 400

    st.vega_lite_chart(df_melted, chart, width='content' if in_sidebar else 'stretch')


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

for i_msg, msg in enumerate(st.session_state.messages):
    avatar = "🤖" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        
        # 1. TEXTE
        st.markdown(msg["content"])
        
        # 2. DEBUG COMPLET (Reconstitué)
        # On cherche la liste d'étapes "steps" qu'on a sauvegardée
        debug_steps = msg.get("debug_info", {}).get("steps", [])

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
                with st.expander("📊 Voir le graphique", expanded=False):
                    auto_plot_data(msg["data"], final_ids, config=saved_config, con=con)

                # Affichage Data (Expander) - Uniformisé avec le nouveau message
                with st.expander("📝 Voir les données brutes", expanded=False):
                    # On utilise les formats stockés dans la config
                    formats = saved_config.get("formats", {})

                    # Récupérer le glossaire_context depuis debug_info si disponible
                    rag_context = msg.get("debug_info", {}).get("rag_context", "")
                    sql_query = msg.get("debug_info", {}).get("sql_query", "")

                    # Extraire les métadonnées des colonnes
                    metadata = get_column_metadata(msg["data"], formats, con, rag_context, sql_query)

                    styled_df, col_config = style_df(msg["data"], formats, metadata)
                    st.dataframe(styled_df, hide_index=True, column_config=col_config, width='stretch')

            except Exception as e: 
                pass
            
# --- 10. TRAITEMENT ET GESTION AMBIGUÏTÉ ---
inject_placeholder_animation()

# Initialisation de la variable de déclenchement si elle n'existe pas
if "trigger_run_prompt" not in st.session_state:
    st.session_state.trigger_run_prompt = None
    
# --- Mode test Codex : injecte un prompt sans UI ---
test_prompt = os.getenv("TERRIBOT_TEST_PROMPT")
if test_prompt and not st.session_state.get("_test_prompt_ran"):
    st.session_state.trigger_run_prompt = test_prompt
    st.session_state["_test_prompt_ran"] = True


# -- A. RÉSOLUTION D'AMBIGUÏTÉ (Affichage des boutons si nécessaire) --
if st.session_state.ambiguity_candidates:
    _dbg("ui.ambiguity.render", candidates=st.session_state.ambiguity_candidates)

    st.warning(f"🤔 Plusieurs territoires trouvés pour **{st.session_state.get('pending_geo_text','ce lieu')}**. Veuillez préciser :")

    # Affichage amélioré avec informations contextuelles
    candidates = st.session_state.ambiguity_candidates[:6]
    display_items = candidates + [{"id": "other", "nom": "Autre territoire", "is_other": True}]
    num_items = len(display_items)
    num_cols = min(3, num_items)
    cols = st.columns(num_cols)

    for i, cand in enumerate(display_items):
        col_index = i % num_cols

        if cand.get("is_other"):
            if cols[col_index].button(cand["nom"], key="amb_other_territory", width='stretch'):
                territoires_text = load_territoires_text()
                if not territoires_text:
                    st.error("Impossible de charger territoires.txt pour la recherche avancée.")
                else:
                    conversation_text = format_conversation_context(st.session_state.messages)
                    pending_geo_text = st.session_state.get("pending_geo_text")
                    pending_prompt = st.session_state.get("pending_prompt")

                    with st.spinner("Recherche avancée du territoire..."):
                        ai_result = ai_select_territory_from_full_context(
                            client,
                            MODEL_NAME,
                            territoires_text,
                            conversation_text,
                            pending_geo_text=pending_geo_text,
                        )

                    selected_id = ai_result.get("selected_id") if ai_result else None
                    if selected_id and str(selected_id).lower() not in ["null", "none", ""]:
                        new_context = build_geo_context_from_id(
                            con,
                            selected_id,
                            "Choix IA (Autre territoire)",
                            search_query=pending_geo_text or pending_prompt,
                        )
                        if new_context:
                            st.session_state.current_geo_context = new_context
                            st.session_state.trigger_run_prompt = pending_prompt
                            st.session_state.force_geo_context = True
                            st.session_state.ambiguity_candidates = None
                            st.session_state.pending_prompt = None
                            st.session_state.pending_geo_text = None
                            _dbg("ui.ambiguity.other_territory", current_geo_context=new_context)
                            st.rerun()
                        else:
                            st.error("Le territoire proposé n'a pas été trouvé dans la base.")
                    else:
                        st.error("Je n'ai pas pu identifier un territoire avec la recherche avancée.")
            continue

        # Construire le label du bouton avec contexte
        button_label = f"**{cand['nom']}**"
        if 'type' in cand and cand['type']:
            button_label += f"\n\n_{cand['type']}_"
        if 'context' in cand and cand['context']:
            button_label += f"\n\n{cand['context']}"
        button_label += f"\n\nCode: `{cand['id']}`"

        # On affiche le bouton
        if cols[col_index].button(button_label, key=f"amb_btn_{cand['id']}_{i}", width='stretch'):
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
    # Effacer les anciens boutons de confirmation quand l'utilisateur tape un nouveau prompt
    st.session_state.ambiguity_candidates = None
    st.session_state.pending_prompt = None
    st.session_state.pending_geo_text = None

# --- Helper pour messages d'attente personnalisés ---
def get_waiting_message(step, territory_name=None, prompt=None):
    """
    Génère un message d'attente personnalisé basé sur le contexte.

    Args:
        step: Type d'étape ('reformulation', 'geo_search', 'rag', 'sql', 'viz', etc.)
        territory_name: Nom du territoire détecté (optionnel)
        prompt: Le prompt pour extraire des indices sur les indicateurs (optionnel)

    Returns:
        str: Message personnalisé
    """
    import hashlib

    # Utiliser un hash du prompt pour avoir des variations déterministes
    variant = 0
    if prompt:
        hash_val = int(hashlib.md5(prompt.encode()).hexdigest()[:4], 16)
        variant = hash_val % 3  # 3 variations possibles

    messages = {
        'reformulation': [
            "J'analyse votre demande...",
            "Je reformule pour bien comprendre...",
            "Je clarifie la question..."
        ],
        'geo_search': [
            f"🌍 Je recherche {territory_name}..." if territory_name else "🌍 Je recherche les territoires mentionnés...",
            f"🌍 Je localise {territory_name}..." if territory_name else "🌍 J'identifie les territoires...",
            f"🌍 Détection de {territory_name}..." if territory_name else "🌍 Détection géographique en cours..."
        ],
        'rag': [
            f"📚 Je cherche les indicateurs pour {territory_name}..." if territory_name else "📚 Je cherche les indicateurs pertinents dans le glossaire...",
            f"📚 Je parcours le glossaire pour {territory_name}..." if territory_name else "📚 Je parcours le glossaire des variables...",
            f"📚 Recherche des variables disponibles pour {territory_name}..." if territory_name else "📚 Recherche des variables disponibles..."
        ],
        'sql': [
            f"🔢 Je récupère les données de {territory_name}..." if territory_name else "🔢 Je récupère les données chiffrées...",
            f"🔢 Extraction des chiffres pour {territory_name}..." if territory_name else "🔢 Extraction des données statistiques...",
            f"🔢 Je collecte les statistiques de {territory_name}..." if territory_name else "🔢 Je collecte les données..."
        ],
        'viz': [
            f"🎨 Je prépare la visualisation pour {territory_name}..." if territory_name else "🎨 Je prépare la visualisation...",
            "🎨 Je crée le graphique...",
            "🎨 Mise en forme des résultats..."
        ],
        'fallback': [
            "🔍 Recherche élargie du territoire avec l'IA...",
            "🔍 Je cherche dans d'autres sources...",
            "🔍 Recherche approfondie en cours..."
        ],
        'complete': [
            "✅ Terminé",
            "✅ Analyse terminée",
            "✅ C'est prêt !"
        ],
        'not_found': [
            "Aucune donnée trouvée",
            "Aucun résultat",
            "Pas de données disponibles"
        ],
        'error': [
            "⚠️ Territoire non identifié",
            "⚠️ Territoire introuvable",
            "⚠️ Lieu non reconnu"
        ]
    }

    return messages.get(step, ["En cours..."])[variant]


def check_territory_mentioned(prompt: str, client, model):
    """
    Vérifie si le prompt mentionne un territoire (ville, commune, département, région, etc.).
    Retourne True si un territoire est mentionné, False sinon.
    """
    system_prompt = """Tu es un assistant qui détecte si un utilisateur mentionne un lieu géographique (territoire) dans sa question.

Ta tâche est de déterminer si l'utilisateur mentionne un territoire (ville, commune, département, région, pays, intercommunalité, EPCI, etc.).

Exemples qui MENTIONNENT un territoire (réponds "OUI") :
- "Analyse la ville de Lyon"
- "Quel est le taux de chômage à Marseille ?"
- "Compare Paris et Lyon"
- "Montre-moi les données pour l'Île-de-France"
- "Qu'en est-il à Champigny-sur-Marne ?"
- "Compare avec le département du Val-de-Marne"
- "Analyse l'intercommunalité Grand Paris Sud Est Avenir"

Exemples qui NE mentionnent PAS de territoire (réponds "NON") :
- "Oui"
- "Non"
- "D'accord"
- "Montre-moi le taux de chômage" (sans lieu mentionné)
- "Et la population ?" (sans lieu mentionné)
- "Quelle est l'évolution ?"
- "Montre-moi un graphique"
- "Peux-tu détailler ?"
- "Compare les communes" (sans nommer de communes spécifiques)

Réponds uniquement "OUI" ou "NON"."""

    user_prompt = f"Question de l'utilisateur : {prompt}"

    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = client.responses.create(
            model=model,
            input=messages,
            temperature=0,
            timeout=10,
        )

        answer = extract_response_text(response).strip().upper()
        _dbg("territory_mentioned_check", prompt=prompt, answer=answer)

        return "OUI" in answer

    except Exception as e:
        _dbg("territory_mentioned_check.error", error=str(e))
        return True  # En cas d'erreur, on suppose qu'un territoire est mentionné (comportement par défaut)


# --- D. EXÉCUTION DU TRAITEMENT ---
if prompt_to_process:
    print("[TERRIBOT] ===============================")
    _dbg("pipeline.start", prompt_to_process=prompt_to_process, from_trigger=was_trigger)

    _dbg("session.state", has_geo=bool(st.session_state.current_geo_context),
        ambiguity=bool(st.session_state.ambiguity_candidates),
        messages=len(st.session_state.messages))

    # ⚠️ VÉRIFICATION DE MENTION DE TERRITOIRE
    # Si on a déjà un contexte géographique, vérifier si l'utilisateur mentionne un territoire
    # Si NON → utiliser le contexte actuel sans recherche géographique
    # Si OUI → continuer le flux normal avec recherche géographique
    skip_geo_search = False

    if st.session_state.current_geo_context and not was_trigger and not st.session_state.force_geo_context:
        # Vérifier si un territoire est mentionné dans le prompt
        territory_mentioned = check_territory_mentioned(
            prompt_to_process,
            client,
            MODEL_NAME
        )

        if not territory_mentioned:
            _dbg("no_territory_mentioned", prompt=prompt_to_process)
            # Pas de territoire mentionné → on va utiliser le contexte actuel sans recherche
            skip_geo_search = True
            # Forcer l'utilisation du contexte actuel
            st.session_state.force_geo_context = True
        else:
            _dbg("territory_mentioned", prompt=prompt_to_process)
            # Un territoire est mentionné → flux normal

    # Stocker skip_geo_search dans session_state pour l'utiliser plus tard
    st.session_state.skip_geo_search = skip_geo_search

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
            territory_selector_placeholder = st.empty()  # 🔧 Sélecteur de territoire (en haut, avant tout)
            chart_placeholder = st.empty()   # Le graphique en haut
            data_placeholder = st.empty()    # Les données au milieu (NOUVEAU)
            message_placeholder = st.empty() # Le texte en bas


            # --- 🛑 MODIFICATION ICI : LE CONTENEUR JETABLE ---
            loader_placeholder = st.empty()  # Un placeholder dédié pour le chargement
            
            # On crée le statut À L'INTÉRIEUR de ce placeholder
            with loader_placeholder:
                status_container = st.status(get_waiting_message('reformulation', prompt=prompt_to_process), expanded=False)

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
                    status_container.update(label=get_waiting_message('reformulation', prompt=prompt_to_process))
                    history_text = "\n".join([f"{m['role']}: {m.get('content','')}" for m in st.session_state.messages[-4:]])
                    current_geo_name = st.session_state.current_geo_context['target_name'] if st.session_state.current_geo_context else ""

                    _dbg("pipeline.rewrite.inputs",
                         prompt_to_process=prompt_to_process,
                         history_tail=history_text[-400:],
                         current_geo_name=current_geo_name,
                         has_current_geo=bool(st.session_state.current_geo_context))

                    reformulation = client.responses.create(
                        model=MODEL_NAME,
                        input=build_messages(
                            f"""
                            Tu es un expert en reformulation de questions territoriales. CONTEXTE GEO ACTUEL : '{current_geo_name}'.

                            OBJECTIFS :
                            1. Rendre la question autonome et claire
                            2. SI "ramène à la population" ou "et pour X ?", REPRENDS le SUJET PRÉCÉDENT
                            3. Si aucun lieu explicite dans la question, réinjecte '{current_geo_name}'

                            RÈGLES CRITIQUES :
                            - Conserve les noms de lieux mentionnés (ex: "Saint-Denis" reste "Saint-Denis", PAS "France")
                            - NE généralise PAS un lieu spécifique vers un territoire plus large
                            - NE remplace PAS un nom de commune/ville par un pays/région
                            - Si l'utilisateur mentionne "Saint-Denis", "Paris", "Lyon", etc., GARDE le nom exact

                            EXEMPLES :
                            - "Quel est le taux de pauvreté à Saint-Denis ?" → "Quel est le taux de pauvreté à Saint-Denis ?"
                            - "et pour Fontenay ?" (après Paris) → "Quel est le taux de pauvreté à Fontenay ?"
                            - "quelle est la population ?" (contexte: Lyon) → "Quelle est la population de Lyon ?"

                            RÉPONDS UNIQUEMENT AVEC LA QUESTION REFORMULÉE, SANS EXPLICATION.
                            """,
                            f"Historique:\n{history_text}\n\nDernière question: {prompt_to_process}",
                        )
                    )
                    rewritten_prompt = extract_response_text(reformulation)
                    _dbg("pipeline.rewrite.done",
                         original=prompt_to_process,
                         rewritten=rewritten_prompt,
                         changed=prompt_to_process != rewritten_prompt)

                    debug_container["reformulation"] = f"Original: {prompt_to_process}\nReformulé: {rewritten_prompt}"
                    
                    with st.expander("🤔 Trace : Reformulation (IA)", expanded=False):
                        st.write("🔄 Compréhension...")
                        st.write(f"**Question originale :** {prompt_to_process}")
                        st.write(f"**Reformulée :** {rewritten_prompt}")

                    # 2. GEO SCOPE
                    new_context = None
                    current_territory = st.session_state.current_geo_context.get("target_name") if st.session_state.current_geo_context else None
                    # Ne pas afficher l'ancien nom de territoire pendant la recherche du nouveau
                    status_container.update(label=get_waiting_message('geo_search', territory_name=None, prompt=rewritten_prompt))
                    _dbg("pipeline.geo.before", force_geo_context=bool(st.session_state.get("force_geo_context")),
                        current_geo=current_territory)

                    # --- MODIFICATION ICI : Gestion du Verrou et Skip Geo Search ---
                    if st.session_state.get("force_geo_context") or st.session_state.get("skip_geo_search"):
                        # Consommer les flags
                        if st.session_state.get("force_geo_context"):
                            st.session_state.force_geo_context = False
                            print("[TERRIBOT][PIPE] 🔒 force_geo_context consumed -> keep existing context")
                        if st.session_state.get("skip_geo_search"):
                            st.session_state.skip_geo_search = False
                            print("[TERRIBOT][PIPE] ⏭️ skip_geo_search consumed -> keep existing context (no territory in prompt)")

                        _dbg("pipeline.geo.locked_context", geo=st.session_state.current_geo_context)

                        # On ne lance PAS analyze_territorial_scope, on garde l'existant
                        if st.session_state.current_geo_context:
                            geo_context = st.session_state.current_geo_context
                            # On force new_context à None pour sauter les blocs suivants
                            new_context = None
                    else:
                        # Pas de skip_geo_search, donc on fait la recherche normale
                        print("[TERRIBOT][PIPE] 🌍 analyze_territorial_scope() running")
                        _dbg("pipeline.geo.before_analysis", rewritten_prompt=rewritten_prompt[:200])
                        new_context = analyze_territorial_scope(con, rewritten_prompt)
                        _dbg("pipeline.geo.after",
                             success=new_context is not None,
                             target_id=new_context.get('target_id') if new_context else None,
                             target_name=new_context.get('target_name') if new_context else None,
                             all_ids_count=len(new_context.get('all_ids', [])) if new_context else 0,
                             display_context=new_context.get('display_context') if new_context else None)

                        
                    # --- GESTION DE L'AMBIGUÏTÉ DÉTECTÉE ---
                    # Si une ambiguïté est détectée OU une clarification utilisateur est nécessaire
                    fallback_context = None  # 🔧 FIX: Initialiser pour éviter NameError

                    # NOUVEAU: Gestion du besoin de clarification utilisateur
                    if new_context and new_context.get("needs_user_clarification"):
                        print("[TERRIBOT][PIPE] 🤔 User clarification needed -> storing candidates + rerun")
                        _dbg("pipeline.clarification", query=new_context.get("query"), candidates=new_context.get("candidates"), confidence=new_context.get("confidence"))

                        # Formater les candidats pour l'UI
                        formatted_candidates = []
                        for cand in new_context.get("candidates", []):
                            # Récupérer les infos complètes du candidat depuis la base
                            try:
                                cand_full_info = con.execute("SELECT ID, NOM_COUV, COMP1, COMP2, COMP3 FROM territoires WHERE ID = ?", [cand['id']]).fetchone()
                                if cand_full_info:
                                    formatted_candidates.append({
                                        "id": str(cand_full_info[0]),
                                        "nom": cand_full_info[1],
                                        "comps": [cand_full_info[2], cand_full_info[3], cand_full_info[4]],
                                        "type": cand.get("type", ""),
                                        "context": cand.get("context", "")
                                    })
                            except Exception as e:
                                _dbg("pipeline.clarification.candidate_error", error=str(e))
                                # Si erreur, utiliser les infos minimales
                                formatted_candidates.append({
                                    "id": cand['id'],
                                    "nom": cand['name'],
                                    "comps": [],
                                    "type": cand.get("type", ""),
                                    "context": cand.get("context", "")
                                })

                        st.session_state.ambiguity_candidates = formatted_candidates
                        st.session_state.pending_geo_text = new_context.get("query")
                        st.session_state.pending_prompt = prompt_to_process

                        debug_container["steps"] = debug_steps
                        debug_container["final_ids"] = (st.session_state.current_geo_context or {}).get("all_ids", [])

                        confidence_msg = {
                            "low": "faible confiance",
                            "medium": "confiance moyenne",
                            "high": "haute confiance"
                        }.get(new_context.get("confidence", "low"), "incertain")

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"🤔 Plusieurs territoires possibles pour **{new_context['query']}** ({confidence_msg}). {new_context.get('reason', '')} Veuillez choisir ci-dessus."
                        })
                        st.rerun()

                    # ANCIEN: Gestion de l'ambiguïté classique
                    elif new_context and new_context.get("ambiguity"):
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

                        debug_container["geo_extraction"] = new_context["lieux_cites"]
                        debug_container["geo_resolution"] = new_context["debug_search"]
                        debug_container["final_ids"] = new_context["all_ids"]

                    # Si on n'a rien trouvé de nouveau, on utilise le contexte existant (celui du bouton par exemple)
                    elif st.session_state.current_geo_context:
                        geo_context = st.session_state.current_geo_context
                        # On ne réaffiche pas l'info si elle n'a pas changé, ou on peut la laisser pour confirmation

                    elif not st.session_state.current_geo_context:
                        # 🆕 FALLBACK IA : Tenter une recherche sémantique dans territoires.txt
                        status_container.update(label=get_waiting_message('fallback', prompt=prompt_to_process))
                        fallback_context = ai_fallback_territory_search(con, prompt_to_process)

                        if fallback_context:
                            st.session_state.current_geo_context = fallback_context
                            _dbg("pipeline.fallback.success", context=fallback_context)

                            debug_container["geo_extraction"] = fallback_context["lieux_cites"]
                            debug_container["geo_resolution"] = fallback_context["debug_search"]
                            debug_container["final_ids"] = fallback_context["all_ids"]
                            debug_steps.append({"icon": "🔎", "label": "Résolution Géo (Fallback IA)", "type": "table", "content": fallback_context["debug_search"]})
                        else:
                            # 🆕 DEMANDER DES PRÉCISIONS : Envoyer un message conversationnel
                            status_container.update(label=get_waiting_message('error', prompt=prompt_to_process), state="error")

                            # Message demandant des précisions
                            clarification_message = """🤔 Je n'arrive pas à identifier le territoire dont vous parlez.

Pouvez-vous préciser votre recherche en indiquant :
- Le nom complet de la commune (ex: "Alençon", "Paris 15e arrondissement")
- Le nom du département avec son numéro (ex: "Orne", "Orne 61", "département 61")
- Le nom de la région (ex: "Normandie", "Île-de-France")
- Le nom complet de l'intercommunalité avec son type (ex: "CC des Pays de L'Aigle", "Métropole de Lyon")

Vous pouvez aussi préciser le contexte géographique (ex: "Alençon dans l'Orne" si plusieurs communes portent le même nom)."""

                            message_placeholder.warning(clarification_message)

                            # Sauvegarder le message dans l'historique
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": clarification_message,
                                "debug_info": debug_container
                            })

                            st.stop()
                    
                    geo_context = st.session_state.current_geo_context
                    # --- CORRECTION ICI : ON FORCE LA SAUVEGARDE DES IDS ---
                    # Cela garantit que le graphique pourra être reconstruit depuis l'historique
                    if new_context:
                     # <--- APPEND STEP
                        debug_steps.append({"icon": "🔎", "label": "Résolution Géo", "type": "table", "content": new_context["debug_search"]})
                    if geo_context:
                        debug_container["final_ids"] = geo_context['all_ids']
                    # 3. RAG (Recherche Variables - Méthode Hybride)
                    territory_for_rag = geo_context.get('target_name') if geo_context else None
                    status_container.update(label=get_waiting_message('rag', territory_name=territory_for_rag, prompt=rewritten_prompt))
                    # On appelle notre nouvelle fonction combinée
                    print("[TERRIBOT][PIPE] 📚 RAG hybrid_variable_search() start")
                    _dbg("pipeline.rag.inputs",
                         rewritten_prompt=rewritten_prompt[:200],
                         df_glossaire_rows=len(df_glossaire),
                         has_embeddings=glossary_embeddings is not None,
                         valid_indices_count=len(valid_indices) if valid_indices is not None else 0)

                    glossaire_context = hybrid_variable_search(
                        rewritten_prompt,
                        con,
                        df_glossaire,
                        glossary_embeddings,
                        valid_indices
                    )
                    _dbg("pipeline.rag.done",
                         glossaire_context_len=len(glossaire_context),
                         preview=glossaire_context[:400],
                         is_empty=len(glossaire_context.strip()) == 0)

                    # Debugging visuel
                    debug_container["rag_context"] = glossaire_context
                    with st.expander("📚 Trace : Variables identifiées", expanded=False):
                        st.text(glossaire_context)
                        
                    if not glossaire_context:
                        # Fallback si rien n'est trouvé
                        glossaire_context = "Aucune variable spécifique trouvée. Essaie d'utiliser des connaissances générales ou signale l'absence de données."

                    # Stocker dans session_state pour le debug
                    if "debug_data" not in st.session_state:
                        st.session_state.debug_data = {}
                    st.session_state.debug_data["search_query"] = rewritten_prompt
                    st.session_state.debug_data["rag_results"] = glossaire_context

                    # 4. SQL GENERATION
                    ids_sql = ", ".join([f"'{str(i)}'" for i in geo_context['all_ids']])
                    parent_clause = geo_context.get('parent_clause', '')
                    territory_for_sql = geo_context.get('target_name') if geo_context else None
                    status_container.update(label=get_waiting_message('sql', territory_name=territory_for_sql, prompt=rewritten_prompt))

                    _dbg("pipeline.sql.inputs",
                         all_ids=geo_context.get('all_ids', [])[:10],
                         all_ids_count=len(geo_context.get('all_ids', [])),
                         parent_clause=parent_clause,
                         territory_name=territory_for_sql,
                         glossaire_context_preview=glossaire_context[:300])

                    # Extraction des schémas complets des tables utilisées
                    try:
                        table_schemas = extract_table_schemas_from_context(glossaire_context, con)
                        _dbg("pipeline.sql.schemas_extracted", schemas_length=len(table_schemas))
                    except Exception as e:
                        print(f"[TERRIBOT][SCHEMA] ⚠️ Erreur extraction schémas: {e}")
                        _dbg("pipeline.sql.schemas_error", error=str(e))
                        table_schemas = ""  # Fallback: continuer sans les schémas complets

                    system_prompt = f"""
                    Tu es Terribot.

                    CONTEXTE DONNÉES (Glossaire) :
                    {glossaire_context}
                    {table_schemas}

                    SCHEMA TABLE "TERRITOIRES" (alias t) :
                    - "ID" (VARCHAR) : Code INSEE
                    - "NOM_COUV" (VARCHAR) : Nom de la commune

                    MISSION : Répondre à "{rewritten_prompt}" via UNE SEULE requête SQL.
                    
                    🚨 RÈGLES CRITIQUES (A RESPECTER ABSOLUMENT) :

                    1. VARIABLES ET TABLES (ANTI-HALLUCINATION) :
                    - 🔴 IMPÉRATIF : Utilise **UNIQUEMENT** les colonnes listées dans le CONTEXTE DONNÉES et les SCHÉMAS COMPLETS ci-dessus.
                    - 🔴 VÉRIFIE que chaque colonne que tu utilises existe dans le schéma de sa table.
                    - Si une variable 2022 (ex: P22_...) n'est pas dans la liste, NE L'INVENTE PAS. Utilise l'année disponible la plus proche (ex: P20_... ou P19_...).
                    - Le contexte t'indique la table source (ex: ✅ TABLE: "ACT_10"). Utilise ce nom exact dans ton JOIN.
                    - Avant d'utiliser une colonne, VÉRIFIE qu'elle existe dans le schéma de cette table fourni ci-dessus.
                    - Jointure : `FROM territoires t LEFT JOIN "NOM_TABLE" d ON t."ID" = d."ID"`
                    - Choisis toujours la variable la PLUS RÉCENTE disponible.
                    - ⛔ N'INVENTE JAMAIS de noms de colonnes qui n'existent pas dans les schémas fournis.
                    
                    2. PÉRIMÈTRE GÉOGRAPHIQUE :
                    - Copie STRICTEMENT cette clause WHERE :
                    - `WHERE (t."ID" IN ({ids_sql}) {parent_clause})`
                    - ⛔ INTERDIT : N'ajoute JAMAIS de condition sur "NOM_COUV".
                    
                    3. CALCULS ET TYPES :
                    - Tables format LARGE. Pas de colonne "VAR".
                    - Utilise `TRY_CAST(table."colonne" AS DOUBLE)` pour tout calcul.
                    - Calcule toujours des ratios (ex: Part du chômage = CHOM / ACT) pour rendre les territoires comparables. Sauf si on te demande une question simple et directe comme "Quelle est la population ?" "Quel est le nombre d'habitants"
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

                    # Génération SQL avec retry automatique et injection de schéma en cas d'erreur
                    sql_query = generate_and_fix_sql(client, MODEL_NAME, system_prompt, rewritten_prompt, con)
                    _dbg("pipeline.sql.gen.raw", sql_query=sql_query[:500])

                    debug_container["sql_query"] = sql_query

                    # Stocker le SQL dans session_state pour le debug
                    if "debug_data" not in st.session_state:
                        st.session_state.debug_data = {}
                    st.session_state.debug_data["sql_query"] = sql_query

                    with st.expander("💻 Trace : Génération SQL (IA)", expanded=False):
                        st.code(sql_query, language="sql")
                    _dbg("sql.exec.about_to_run", sql=sql_query[:500], ids=geo_context.get("all_ids", [])[:10], ids_count=len(geo_context.get("all_ids", [])))

                    debug_container["sql_query"] = sql_query

                    if con:
                        try:
                            _dbg("sql.exec.before_run", sql_preview=sql_query[:500], con_type=type(con).__name__)
                            df = con.execute(sql_query).df()
                            _dbg("sql.exec.result", empty=df.empty, rows=len(df), cols=list(df.columns), shape=df.shape)
                        except Exception as e:
                            _dbg("sql.exec.error", error=str(e), error_type=type(e).__name__)
                            raise e

                        if not df.empty:
                            _dbg("sql.exec.head", head=df.head(3).to_dict(orient="records"))
                            _dbg("sql.exec.dtypes", dtypes=str(df.dtypes.to_dict()))
                            # Compter les valeurs nulles par colonne
                            null_counts = df.isnull().sum().to_dict()
                            _dbg("sql.exec.null_counts", nulls=null_counts)

                            territory_for_viz = geo_context.get('target_name') if geo_context else None
                            status_container.update(label=get_waiting_message('viz', territory_name=territory_for_viz, prompt=rewritten_prompt))

                            # On configure le graph PENDANT que le loader est encore là
                            print("[TERRIBOT][PIPE] 📈 get_chart_configuration() start")
                            chart_config = get_chart_configuration(df, rewritten_prompt, glossaire_context, client, MODEL_NAME)
                            _dbg("pipeline.chart_config.done", selected=chart_config.get("selected_columns"), formats=chart_config.get("formats"))

                            # 🆕 Améliorer les formats avec l'IA de manière intelligente et contextuelle
                            print("[TERRIBOT][PIPE] 🎨 ai_enhance_formats() start")
                            initial_formats = chart_config.get("formats", {})
                            enhanced_formats = ai_enhance_formats(df, initial_formats, client, MODEL_NAME)
                            chart_config["formats"] = enhanced_formats
                            _dbg("pipeline.chart_config.enhanced", formats=enhanced_formats)

                            status_container.update(label=get_waiting_message('complete', prompt=rewritten_prompt), state="complete")
                        else:
                            _dbg("sql.exec.empty_dataframe", warning="DataFrame is empty - no data returned from SQL query")
                            status_container.update(label=get_waiting_message('not_found', prompt=rewritten_prompt), state="error")
                            message_placeholder.warning("Aucune donnée trouvée.")
                            st.stop()

                # --- SORTIE DU CONTEXTE 'with status_container:' ---
                loader_placeholder.empty()

                # A. Vérification de l'éligibilité de la carte et affichage
                map_eligible = False  # Initialisation par défaut
                if not df.empty:
                    target_id = str(geo_context.get("target_id", ""))
                    selected_cols = chart_config.get("selected_columns", [])
                    formats = chart_config.get("formats", {})
                    metric_col = selected_cols[0] if selected_cols else None
                    metric_spec = formats.get(metric_col, {}) if metric_col else {}
                    metric_kind = (metric_spec.get("kind") or "").lower()
                    metric_label = (metric_spec.get("label") or metric_col or "").lower()

                    # Vérifier si la carte est éligible
                    map_allowed = metric_kind == "percent" or any(
                        kw in metric_label for kw in ["taux", "part", "ratio", "moyen", "moyenne", "médiane"]
                    )
                    is_commune_or_epci = target_id and is_commune_or_epci_code(target_id)
                    map_eligible = map_allowed and is_commune_or_epci and metric_col

                    _dbg(
                        "map.eligibility.check",
                        metric_col=metric_col,
                        metric_kind=metric_kind,
                        metric_label=metric_label,
                        map_allowed=map_allowed,
                        target_id=target_id,
                        map_eligible=map_eligible
                    )

                    # B. Affichage avec expanders mutuellement exclusifs
                    with chart_placeholder:
                        current_ids = debug_container.get("final_ids", [])

                        if map_eligible:
                            # Si la carte est éligible, créer un choix entre graphique et carte
                            viz_choice = st.radio(
                                "Visualisation",
                                ["📊 Graphique", "🗺️ Carte"],
                                horizontal=True,
                                key=f"viz_choice_{len(st.session_state.messages)}"
                            )

                            if viz_choice == "📊 Graphique":
                                with st.expander("📊 Voir le graphique", expanded=False):
                                    auto_plot_data(df, current_ids, config=chart_config, con=con)
                            else:  # Carte
                                render_epci_choropleth(
                                    con,
                                    df,
                                    target_id,
                                    geo_context.get("target_name", target_id),
                                    metric_col,
                                    metric_spec,
                                    sql_query=debug_container.get("sql_query")
                                )
                        else:
                            # Pas de carte éligible, afficher uniquement le graphique
                            with st.expander("📊 Voir le graphique", expanded=False):
                                auto_plot_data(df, current_ids, config=chart_config, con=con)

                    # Sauvegarder les informations pour les actions rapides
                    debug_container["map_eligible"] = map_eligible
                    if map_eligible:
                        debug_container["map_target_id"] = target_id
                        debug_container["map_metric_col"] = metric_col
                        debug_container["map_metric_spec"] = metric_spec

                    # Affichage des données brutes (seulement si df n'est pas vide)
                    with data_placeholder:
                        with st.expander("📝 Voir les données brutes", expanded=False):
                            formats = chart_config.get('formats', {})

                            # Extraire les métadonnées des colonnes
                            metadata = get_column_metadata(df, formats, con, glossaire_context, debug_container.get("sql_query", ""))

                            styled_df, col_config = style_df(df, formats, metadata)
                            st.dataframe(styled_df, hide_index=True, column_config=col_config, width='stretch')

                    # 🔧 Stocker les données pour affichage dans la sidebar
                    st.session_state.current_viz_data = {
                        "df": df.copy(),
                        "chart_config": chart_config.copy(),
                        "final_ids": debug_container.get("final_ids", geo_context.get("all_ids", [])),
                        "geo_context": geo_context.copy() if geo_context else {},
                        "sql_query": debug_container.get("sql_query"),
                        "glossaire_context": debug_container.get("rag_context", "")
                    }

                # C. Streaming du Texte
                if not df.empty:
                    print("[TERRIBOT][PIPE] 📝 Streaming response start")
                    _dbg("pipeline.stream.inputs", df_rows=len(df), df_cols=list(df.columns), formats=chart_config.get("formats"))

                    stream = client.responses.create(
                        model=MODEL_NAME,
                        input=build_messages(
                            f"""
                            Tu es Terribot, un expert en analyse territoriale s'adressant à des élus et agents  des collectivités locales en France.

                            TON RÔLE :
                            Traduire les données brutes ci-jointes en une réponse naturelle, fluide et professionnelle.
                            Proposer une piste de réflexion pour aller plus loin, sous forme d'une question pour proposer un autre graphique.

                            RÈGLES D'OR (À RESPECTER STRICTEMENT) :
                            1. ⛔ NE JAMAIS mentionner "le tableau", "vos données", "la colonne", "l'extrait" ou "la ligne". Fais comme si tu connaissais ces chiffres par cœur.
                            2. ⛔ NE JAMAIS citer les noms techniques des variables (ex: "taux_chomage_15_64" ou "indicateur_voisins"). Utilise le langage courant ("Taux de chômage").
                            3. ⛔ SI une colonne contient des 0 et des 1 (booléens), NE LES CITE PAS. Interprète-les (ex: "C'est supérieur à la moyenne").
                            4. CONTEXTUALISE : Si des villes demandées ou indicateurs sont absents des données, dis simplement "Je dispose seulement des données A et B pour X et Y" sans dire "dans le fichier fourni".
                            5. STRUCTURE : Va à l'essentiel. Parle toujours des données des échelons territoriaux les plus locaux en principal sujet, et compare les aux autres valeurs.

                            Unités des données : {json.dumps(chart_config.get('formats', {}))}
                            """,
                            df.to_string(),
                        ),
                        stream=True,
                    )
                    full_response_text = message_placeholder.write_stream(stream_response_text(stream))
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
                import traceback
                error_trace = traceback.format_exc()
                error_msg = str(e)

                # Log détaillé pour debug
                print("[TERRIBOT][FATAL] Exception:", repr(e))
                print(error_trace)
                _dbg("pipeline.error", error_type=type(e).__name__, error_msg=error_msg[:200])

                # Message utilisateur adapté selon le type d'erreur
                if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                    st.error("⏱️ La requête a pris trop de temps. Réessayez avec une question plus simple.")
                elif "rate limit" in error_msg.lower() or "429" in error_msg:
                    st.error("🚦 Trop de requêtes. Attendez quelques secondes et réessayez.")
                elif "api" in error_msg.lower() or "openai" in error_msg.lower():
                    st.error("🔌 Erreur de connexion à l'IA. Vérifiez votre clé API.")
                elif "sql" in error_msg.lower() or "duckdb" in error_msg.lower():
                    st.error("📊 Erreur lors de la récupération des données. La variable demandée n'existe peut-être pas.")
                else:
                    st.error(f"❌ Une erreur s'est produite : {error_msg[:150]}")

                # Sauvegarde de l'erreur dans l'historique pour debug
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "⚠️ Je n'ai pas pu traiter votre demande. Essayez de reformuler votre question.",
                    "debug_info": {"error": error_msg, "trace": error_trace[-500:]}
                })


# --- RENDU DE LA SIDEBAR VISUALISATIONS (À LA FIN) ---
# Cette section est exécutée à la fin pour que current_viz_data soit déjà défini
if "sidebar_viz_placeholder" in st.session_state:
    sidebar_viz_placeholder = st.session_state.sidebar_viz_placeholder

    with sidebar_viz_placeholder.container():
        # Afficher les visualisations du dernier message s'il y en a
        if "current_viz_data" in st.session_state and st.session_state.current_viz_data:
            # 🔧 Afficher le titre seulement quand il y a des données de visualisation
            st.markdown("### Tracer un graphique ou une carte")
            viz_data = st.session_state.current_viz_data
            df = viz_data.get("df")
            chart_config = viz_data.get("chart_config", {})
            final_ids = viz_data.get("final_ids", [])
            geo_context_viz = viz_data.get("geo_context", {})
            sql_query_viz = viz_data.get("sql_query")
            glossaire_context_viz = viz_data.get("glossaire_context", "")

            if df is not None and not df.empty:
                try:
                    formats = chart_config.get("formats", {})
                    numeric_candidates = []
                    for col in df.columns:
                        if col.upper() in ["AN", "ANNEE", "YEAR", "ID", "CODGEO"]:
                            continue
                        s = pd.to_numeric(df[col], errors="coerce")
                        if s.notna().any():
                            numeric_candidates.append(col)

                    if numeric_candidates:
                        def format_metric_label(col):
                            spec = formats.get(col, {})
                            return spec.get("title") or spec.get("label") or col
                            
                        # Radio buttons pour choisir le type de visualisation (pas de rerun)
                        viz_type = st.radio(
                            "Type",
                            ["📊 Graphique", "🗺️ Carte"],
                            horizontal=True,
                            key="sidebar_viz_type_radio",
                            label_visibility="collapsed"
                        )
                        
                        # Sélecteur de variable
                        selected_metric = st.selectbox(
                            "Variable",
                            numeric_candidates,
                            index=0,
                            format_func=format_metric_label,
                            key="sidebar_metric_selector",
                            label_visibility="collapsed"
                        )

                        # Créer la config pour la visualisation
                        manual_spec = formats.get(
                            selected_metric,
                            {"kind": "number", "label": selected_metric, "title": selected_metric}
                        )
                        manual_config = {"selected_columns": [selected_metric], "formats": formats}
                        target_id = str(geo_context_viz.get("target_id", ""))

                        # Afficher la visualisation choisie
                        if viz_type == "📊 Graphique":
                            auto_plot_data(df, final_ids, config=manual_config, con=con, in_sidebar=True)
                        elif viz_type == "🗺️ Carte":
                            is_commune = is_valid_insee_code(target_id) and len(target_id) in (4, 5)
                            is_epci = is_valid_insee_code(target_id) and len(target_id) == 9
                            if is_commune or is_epci:
                                render_epci_choropleth(
                                    con,
                                    df,
                                    target_id,
                                    geo_context_viz.get("target_name", target_id),
                                    selected_metric,
                                    manual_spec,
                                    sql_query=sql_query_viz,
                                    in_sidebar=True
                                )
                            else:
                                st.info("Carte seulement disponible pour commune ou EPCI.")

                        # Afficher la source et l'intitulé détaillé/calcul pour la variable affichée
                        metadata = get_column_metadata(df, formats, con, glossaire_context_viz, sql_query_viz or "")
                        selected_meta = metadata.get(selected_metric) if metadata else None
                        if selected_meta:
                            detail_value = selected_meta.get("calculation") or selected_meta.get("definition") or ""
                            if detail_value:
                                st.caption(detail_value)

                            if selected_meta.get("source"):
                                st.caption(f"**Source** : {selected_meta['source']}")
                    else:
                        st.caption("Aucune variable numérique disponible.")
                except Exception as e:
                    st.error(f"Erreur d'affichage : {str(e)[:100]}")
                    print(f"[SIDEBAR] Erreur: {e}")
            else:
                st.caption("Aucune donnée disponible pour visualisation.")
        else:
            st.caption("Les visualisations apparaîtront ici après une question.")

# --- PANNEAU DE DEBUG (SIDEBAR) ---
if "sidebar_debug_placeholder" in st.session_state:
    sidebar_debug_placeholder = st.session_state.sidebar_debug_placeholder

    with sidebar_debug_placeholder.container():
        if "debug_data" in st.session_state and st.session_state.debug_data:
            with st.expander("🐛 Debug", expanded=False):
                debug_data = st.session_state.debug_data

                # SQL Query
                if "sql_query" in debug_data and debug_data["sql_query"]:
                    st.markdown("**SQL Produit par GPT :**")
                    st.code(debug_data["sql_query"], language="sql")

                # RAG Results
                if "rag_results" in debug_data and debug_data["rag_results"]:
                    st.markdown("**Variables trouvées (RAG) :**")
                    rag_text = debug_data["rag_results"]
                    # Limiter l'affichage si trop long
                    if len(rag_text) > 2000:
                        st.text_area("RAG Context", rag_text[:2000] + "...", height=150)
                    else:
                        st.text_area("RAG Context", rag_text, height=150)

                # Search Query
                if "search_query" in debug_data and debug_data["search_query"]:
                    st.markdown("**Question reformulée pour RAG :**")
                    st.caption(debug_data["search_query"])
