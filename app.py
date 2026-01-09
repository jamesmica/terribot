import streamlit as st
import openai
import duckdb
import pandas as pd
import os

# --- CONFIGURATION ---
st.set_page_config(page_title="Ithea Data Assistant", layout="centered")

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
    con = duckdb.connect(database=':memory:')
    
    data_folder = "data"
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        st.error(f"Le dossier '{data_folder}' n'existe pas.")
        return None, ""

    files = [f for f in os.listdir(data_folder) if f.endswith('.parquet')]
    schema_info = []

    for f in files:
        # Nettoyage du nom
        table_name = f.replace('.parquet', '').replace('-', '_').replace(' ', '_').lower()
        file_path = os.path.join(data_folder, f)
        
        # Ajout des guillemets pour protéger les noms comme "all"
        con.execute(f'CREATE OR REPLACE VIEW "{table_name}" AS SELECT * FROM \'{file_path}\'')
        schema_info.append(table_name)

    # Chargement Glossaire
    glossaire_path = os.path.join(data_folder, "Glossaire.txt")
    if os.path.exists(glossaire_path):
        try:
            con.execute(f"CREATE OR REPLACE VIEW glossaire AS SELECT * FROM read_csv_auto('{glossaire_path}')")
            schema_info.append("glossaire")
        except:
            try:
                con.execute(f"CREATE OR REPLACE VIEW glossaire AS SELECT * FROM read_csv_auto('{glossaire_path}', encoding='latin-1')")
                schema_info.append("glossaire")
            except Exception as e:
                st.warning(f"Erreur lecture glossaire : {e}")
        
    return con, table_list_str(schema_info)

def table_list_str(schema_list):
    return ", ".join(schema_list)

con, table_list = init_db()

# --- FONCTION DE RÉSOLUTION DE VILLE (Code INSEE) ---
def find_insee_code(con, city_name):
    """
    Cherche le code INSEE (1ère colonne) correspondant au nom de la ville.
    On scanne quelques tables courantes pour trouver une correspondance.
    """
    if not con: return None, None
    
    # Tables susceptibles de contenir les noms de villes (à adapter selon vos données)
    # On prend toutes les tables sauf le glossaire
    candidate_tables = [t[0] for t in con.execute("SHOW TABLES").fetchall() if t[0] != 'glossaire']
    
    clean_city = city_name.strip().replace("'", "''") # Sécurité SQL basique
    
    for table in candidate_tables:
        try:
            # On cherche une colonne qui ressemble à "libgeo", "lib_geo", "nom", "commune"
            cols = [c[0] for c in con.execute(f'DESCRIBE "{table}"').fetchall()]
            col_geo_name = next((c for c in cols if c.lower() in ['libgeo', 'lib_geo', 'commune', 'nom_com']), None)
            
            if col_geo_name:
                # On tente de trouver la ville
                # On récupère la 1ère colonne (Code INSEE) et le nom
                col_id = cols[0] # La consigne dit que la 1ère colonne est l'ID
                
                query = f"""
                SELECT "{col_id}" 
                FROM "{table}" 
                WHERE "{col_geo_name}" ILIKE '{clean_city}' 
                LIMIT 1
                """
                res = con.execute(query).fetchone()
                if res:
                    return res[0], col_id # Retourne (94080, 'codgeo')
        except:
            continue
            
    return None, None

# --- INTERFACE ---
st.title("🤖 Assistant Données Territoires")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Bonjour ! Je suis prêt. Posez-moi une question sur une ville."}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "data" in msg:
            st.dataframe(msg["data"])

# --- LOGIQUE DE CHAT ---
if prompt := st.chat_input("Ex: Part des moins de 3 ans à Vincennes ?"):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            # ÉTAPE 0 : Identification de la ville dans le prompt
            # On demande à GPT d'extraire juste le nom de la ville pour notre recherche technique
            city_extract = client.chat.completions.create(
                model="gpt-3.5-turbo", # Rapide et pas cher pour ça
                messages=[{"role": "system", "content": "Extrais uniquement le nom de la ville mentionnée dans le texte. Si aucune ville, réponds 'None'."}, {"role": "user", "content": prompt}]
            ).choices[0].message.content.strip()
            
            code_insee = None
            col_id_name = "column_1"
            
            if city_extract and city_extract != "None":
                message_placeholder.markdown(f"📍 *Recherche du code INSEE pour {city_extract}...*")
                code_insee, col_id_name = find_insee_code(con, city_extract)
            
            if code_insee:
                st.success(f"Ville identifiée : {city_extract} (Code : {code_insee})")
            else:
                # Si on ne trouve pas le code, on laissera GPT faire un ILIKE classique
                st.caption("Code INSEE non trouvé automatiquement, passage en mode recherche textuelle.")

            # ÉTAPE 1 : Filtrage du Glossaire (RAG)
            glossaire_extract = ""
            if con:
                message_placeholder.markdown("🧠 *Analyse du glossaire...*")
                try:
                    df_gloss = con.execute("SELECT * FROM glossaire").df()
                    # Fallback colonnes (adapter selon votre fichier réel)
                    if len(df_gloss.columns) >= 6:
                        df_search = df_gloss.iloc[:, [1, 4, 5]] # Onglet, Variable, Intitulé
                    else:
                        df_search = df_gloss
                    
                    keywords = [w.lower() for w in prompt.split() if len(w) > 3]
                    if keywords:
                        mask = df_search.iloc[:, -1].astype(str).str.lower().apply(lambda x: any(k in x for k in keywords))
                        df_filtered = df_search[mask].head(15)
                        glossaire_extract = df_filtered.to_csv(index=False, sep="|")
                    else:
                        glossaire_extract = df_search.head(10).to_csv(index=False, sep="|")
                except Exception as e:
                    st.warning(f"Glossaire partiel : {e}")

            # ÉTAPE 2 : Prompt Système Stratégique
            base_instruction = ""
            if code_insee:
                base_instruction = f"""
                INFO CRUCIALE : Le code INSEE de la ville est '{code_insee}'.
                LA PREMIÈRE COLONNE de chaque table est l'identifiant géographique.
                
                TA MISSION :
                1. Utilise le code INSEE '{code_insee}' dans le WHERE sur la 1ère colonne de la table.
                   Exemple : SELECT ... FROM table WHERE "{col_id_name}" = '{code_insee}'
                2. N'utilise PAS de filtre sur le texte 'libgeo' si tu as le code.
                """
            else:
                base_instruction = """
                INFO : Code INSEE non trouvé. Filtre sur la colonne 'libgeo' avec ILIKE.
                Exemple : WHERE libgeo ILIKE '%Ville%'
                """

            system_prompt = f"""
            Tu es un expert SQL DuckDB.
            
            {base_instruction}
            
            EXTRAIT DU GLOSSAIRE PERTINENT :
            {glossaire_extract}
            
            RÈGLES :
            1. Trouve la table et la colonne de données dans l'extrait du glossaire.
            2. Si tu dois faire un calcul (ex: part %), fais-le en SQL.
            3. Réponds UNIQUEMENT le code SQL (pas de markdown).
            """

            # ÉTAPE 3 : Génération SQL
            response_sql = client.chat.completions.create(
                model="gpt-5.2-2025-12-11",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            
            sql_query = response_sql.choices[0].message.content.replace("```sql", "").replace("```", "").strip()
            
            with st.expander("Voir la requête SQL"):
                st.code(sql_query, language="sql")

            # ÉTAPE 4 : Exécution
            if con:
                df_result = con.execute(sql_query).df()
                
                if not df_result.empty:
                    analysis = client.chat.completions.create(
                        model="gpt-5.2-2025-12-11",
                        messages=[
                            {"role": "system", "content": "Tu es un expert data. Fais une phrase de réponse précise."},
                            {"role": "user", "content": f"Question: {prompt}\nRésultat: {df_result.to_string()}"}
                        ]
                    )
                    final_response = analysis.choices[0].message.content
                    message_placeholder.markdown(final_response)
                    st.dataframe(df_result)
                    st.session_state.messages.append({"role": "assistant", "content": final_response, "data": df_result})
                else:
                    message_placeholder.warning("Aucune donnée trouvée.")
            
        except Exception as e:
            st.error(f"Erreur : {e}")