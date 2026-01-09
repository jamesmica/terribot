import streamlit as st
import openai
import duckdb
import pandas as pd
import os

# --- CONFIGURATION ---
st.set_page_config(page_title="Ithea Data Assistant", layout="centered")

# Récupération de la clé API (soit via secrets, soit en dur pour tester)
# Pour tester vite, tu peux mettre "sk-..." à la place de st.secrets si tu n'as pas configuré secrets.toml
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
        table_name = f.replace('.parquet', '').replace('-', '_').replace(' ', '_').lower()
        file_path = os.path.join(data_folder, f)
        # Création d'une VUE (ne charge pas la RAM, lit directement le fichier)
        con.execute(f"CREATE OR REPLACE VIEW {table_name} AS SELECT * FROM '{file_path}'")
        schema_info.append(table_name)
    
    # 2. Chargement du GLOSSAIRE (si présent)
    glossaire_path = os.path.join(data_folder, "Glossaire.txt")
    if os.path.exists(glossaire_path):
        try:
            # DuckDB est très fort pour détecter le format CSV/TXT tout seul
            con.execute(f"CREATE OR REPLACE VIEW glossaire AS SELECT * FROM read_csv_auto('{glossaire_path}')")
            schema_info.append("glossaire (Contient la liste des indicateurs)")
        except Exception as e:
            st.warning(f"Erreur chargement glossaire: {e}")

    return con, ", ".join(schema_info)

# Initialisation
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
        # Si le message contient des résultats (dataframe), on les affiche
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
        message_placeholder.markdown("🧠 *Réflexion et recherche de la bonne table...*")
        
        try:
            # A. On demande à GPT de trouver la table et faire le SQL
            # On lui donne un aperçu du glossaire pour qu'il choisisse la bonne table
            glossaire_sample = ""
            try:
                if con: glossaire_sample = con.execute("SELECT * FROM glossaire LIMIT 20").df().to_string()
            except: pass

            system_prompt = f"""
            Tu es un expert SQL DuckDB. 
            Tu as accès à des tables locales : {table_list}.
            
            Voici un extrait du GLOSSAIRE pour t'aider à choisir la bonne table :
            {glossaire_sample}

            RÈGLES :
            1. Trouve la table pertinente.
            2. La colonne géographique s'appelle souvent 'libgeo', 'LIBGEO', ou 'commune'.
            3. Génère uniquement une requête SQL valide (pas de Markdown).
            4. Utilise ILIKE pour les villes : WHERE libgeo ILIKE '%Vincennes%'
            """

            response_sql = client.chat.completions.create(
                model="gpt-4o", # Ou gpt-3.5-turbo
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            
            sql_query = response_sql.choices[0].message.content.replace("```sql", "").replace("```", "").strip()
            
            # Affichage debug du SQL (optionnel, pour vérifier)
            with st.expander("Voir la requête SQL générée"):
                st.code(sql_query, language="sql")

            # B. Exécution locale (DuckDB)
            if con:
                df_result = con.execute(sql_query).df()
                
                if not df_result.empty:
                    # C. Interprétation du résultat par GPT
                    analysis = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "Tu es un analyste. Résume ces données en une phrase simple et claire pour l'utilisateur."},
                            {"role": "user", "content": f"Question: {prompt}\nDonnées: {df_result.to_string()}"}
                        ]
                    )
                    final_response = analysis.choices[0].message.content
                    
                    # Affichage
                    message_placeholder.markdown(final_response)
                    st.dataframe(df_result)
                    
                    # Sauvegarde historique
                    st.session_state.messages.append({"role": "assistant", "content": final_response, "data": df_result})
                else:
                    msg = "J'ai exécuté la requête mais je n'ai trouvé aucun résultat (tableau vide). Vérifiez l'orthographe de la ville."
                    message_placeholder.markdown(msg)
                    st.session_state.messages.append({"role": "assistant", "content": msg})
            else:
                st.error("Erreur de connexion DuckDB.")

        except Exception as e:
            error_msg = f"Une erreur technique est survenue : {e}"
            message_placeholder.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
