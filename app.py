import streamlit as st
import openai
import duckdb
import pandas as pd
import os

# --- CONFIGURATION ---
st.set_page_config(page_title="Ithea Data Assistant", layout="centered")

# Récupération de la clé API
# Vérifie si la clé est dans les secrets, sinon demande dans la sidebar
if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
else:
    api_key = st.sidebar.text_input("Clé API OpenAI", type="password")

if not api_key:
    st.warning("Veuillez entrer une clé API pour continuer.")
    st.stop()

client = openai.OpenAI(api_key=api_key)

# --- MOTEUR DE DONNÉES (DUCKDB) ---
@st.cache_resource
def init_db():
    """Initialise DuckDB et charge virtuellement les fichiers Parquet"""
    con = duckdb.connect(database=':memory:')
    
    # 1. Chargement automatique des PARQUETS du dossier 'data'
    data_folder = "data"
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        st.error(f"Le dossier '{data_folder}' n'existe pas. Créez-le et mettez vos fichiers parquet dedans.")
        return None, ""

    files = [f for f in os.listdir(data_folder) if f.endswith('.parquet')]
    schema_info = []

    for f in files:
        # Nettoyage du nom de la table
        table_name = f.replace('.parquet', '').replace('-', '_').replace(' ', '_').lower()
        file_path = os.path.join(data_folder, f)
        
        # Création d'une VUE (ne charge pas la RAM, lit directement le fichier)
        # ⚠️ IMPORTANT : On ajoute des guillemets "{table_name}" pour gérer les fichiers comme "all.parquet"
        con.execute(f'CREATE OR REPLACE VIEW "{table_name}" AS SELECT * FROM \'{file_path}\'')
        schema_info.append(table_name)

    # 2. Chargement du GLOSSAIRE
    glossaire_path = os.path.join(data_folder, "Glossaire.txt")
    if os.path.exists(glossaire_path):
        try:
            # Essai 1 : Lecture auto (souvent UTF-8)
            con.execute(f"CREATE OR REPLACE VIEW glossaire AS SELECT * FROM read_csv_auto('{glossaire_path}')")
        except:
            try:
                # Essai 2 : Forçage Latin-1 (pour fichiers Windows/Excel)
                con.execute(f"CREATE OR REPLACE VIEW glossaire AS SELECT * FROM read_csv_auto('{glossaire_path}', encoding='latin-1')")
            except Exception as e:
                st.warning(f"Impossible de lire le Glossaire : {e}")
        
        schema_info.append("glossaire")

    return con, ", ".join(schema_info)

# Initialisation de la base
con, table_list = init_db()

# --- INTERFACE ---
st.title("🤖 Assistant Données Territoires")
st.caption(f"🚀 Moteur DuckDB actif sur {len(table_list.split(','))} tables.")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Bonjour ! Je suis connecté à vos données locales. Posez-moi une question sur un territoire."}]

# Affichage historique
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "data" in msg:
            st.dataframe(msg["data"])

# --- LOGIQUE DE CHAT ---
if prompt := st.chat_input("Ex: Part des familles monoparentales à Vincennes ?"):
    
    # 1. Afficher message user
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Cerveau : GPT génère le SQL
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🧠 *Lecture du glossaire et réflexion...*")
        
        try:
            # A. PRÉPARATION DU CONTEXTE GLOSSAIRE (Colonnes B, E, F)
            glossaire_txt = ""
            if con:
                try:
                    # On charge tout le glossaire en DataFrame
                    df_gloss = con.execute("SELECT * FROM glossaire").df()
                    
                    # --- MAPPAGE DES COLONNES (B, E, F) ---
                    # Excel Col B = Index 1 | Col E = Index 4 | Col F = Index 5
                    try:
                        # Si les entêtes existent (recommandé)
                        cols_to_keep = ["Onglet", "Nom au sein de la base de données", "Intitulé détaillé"]
                        # On filtre si les noms existent
                        valid_cols = [c for c in cols_to_keep if c in df_gloss.columns]
                        
                        if len(valid_cols) == 3:
                            df_context = df_gloss[valid_cols]
                        else:
                            # FALLBACK : On prend par position (Index 1, 4, 5)
                            # Attention : Python commence à 0. Donc B=1, E=4, F=5
                            df_context = df_gloss.iloc[:, [1, 4, 5]]
                            df_context.columns = ["Table_SQL", "Nom_Colonne", "Description"]
                            
                    except Exception:
                        # Si tout échoue, on prend tout (mais c'est plus lourd)
                        df_context = df_gloss
                    
                    # Conversion en texte CSV léger pour GPT
                    glossaire_txt = df_context.to_csv(index=False, sep="|")
                    
                except Exception as e:
                    # Si pas de glossaire, on continue sans (mais GPT sera moins précis)
                    st.warning(f"Glossaire non chargé : {e}")

            # B. LE PROMPT SYSTÈME
            system_prompt = f"""
            Tu es un expert Data Analyst connecté à une base DuckDB.
            
            OBJECTIF :
            Tu dois transformer la question de l'utilisateur en une requête SQL DuckDB valide.
            
            1. ANALYSE LE GLOSSAIRE CI-DESSOUS :
            Chaque ligne contient : Table SQL | Nom de la colonne variable | Description du contenu.
            
            --- DÉBUT GLOSSAIRE ---
            {glossaire_txt}
            --- FIN GLOSSAIRE ---
            
            2. RÈGLES :
            - Cherche dans la colonne 'Description' (ou 'Intitulé détaillé') le concept qui correspond à la question.
            - Utilise la 'Table_SQL' (ou Onglet) et le 'Nom_Colonne' correspondants.
            - La colonne géographique s'appelle toujours 'libgeo' (ou vérifie 'LIBGEO').
            - Utilise ILIKE pour la ville : WHERE libgeo ILIKE '%Vincennes%'
            - Ne réponds QUE le code SQL pur (pas de ```sql, pas de texte).
            """

            # C. APPEL GPT
            response_sql = client.chat.completions.create(
                model="gpt-4o", # Utilise gpt-4o pour gérer le contexte long du glossaire
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            
            sql_query = response_sql.choices[0].message.content.replace("```sql", "").replace("```", "").strip()
            
            # Debug : Voir ce que GPT a choisi
            with st.expander("Voir la requête générée"):
                st.code(sql_query, language="sql")

            # D. EXÉCUTION
            if con:
                df_result = con.execute(sql_query).df()
                
                if not df_result.empty:
                    # Analyse du résultat
                    analysis = client.chat.completions.create(
                        model="gpt-5.2-2025-12-11",
                        messages=[
                            {"role": "system", "content": "Tu es un expert territoires. Fais une phrase de réponse claire avec le chiffre."},
                            {"role": "user", "content": f"Question: {prompt}\nDonnées: {df_result.to_string()}"}
                        ]
                    )
                    final_response = analysis.choices[0].message.content
                    message_placeholder.markdown(final_response)
                    st.dataframe(df_result)
                    st.session_state.messages.append({"role": "assistant", "content": final_response, "data": df_result})
                else:
                    msg = "Aucun résultat trouvé (Tableau vide). Vérifiez le nom de la ville."
                    message_placeholder.warning(msg)
                    st.session_state.messages.append({"role": "assistant", "content": msg})
            else:
                st.error("Erreur connexion DB")

        except Exception as e:
            st.error(f"Erreur technique : {e}")