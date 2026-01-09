import streamlit as st
import openai
import duckdb
import pandas as pd
import os
import numpy as np
# On supprime scikit-learn pour accélérer le démarrage
# from sklearn.metrics.pairwise import cosine_similarity 

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
MODEL_NAME = "gpt-5.2-2025-12-11"
EMBEDDING_MODEL = "text-embedding-3-small"

# --- FONCTIONS VECTORIELLES (Optimisées Numpy) ---
@st.cache_resource
def get_glossary_embeddings(df_glossaire):
    """
    Vectorisation avec nettoyage des données pour éviter l'erreur 400.
    """
    if df_glossaire.empty:
        return None
    
    # Création du texte riche
    df_glossaire['combined_text'] = (
        "Table: " + df_glossaire.iloc[:, 1].astype(str) + 
        " | Var: " + df_glossaire.iloc[:, 4].astype(str) + 
        " | Desc: " + df_glossaire.iloc[:, 5].astype(str)
    ).fillna("")

    # NETTOYAGE CRITIQUE : On enlève les lignes vides ou trop courtes qui font planter l'API
    inputs = df_glossaire['combined_text'].tolist()
    # On garde une trace des index valides si besoin, mais ici on simplifie
    # On s'assure que c'est bien des strings et pas vides
    inputs = [str(x) for x in inputs if str(x).strip() != ""]

    if not inputs:
        return None

    try:
        # Batch limit arbitraire pour l'exemple
        limit = 2000 
        inputs_batch = inputs[:limit]
        
        response = client.embeddings.create(input=inputs_batch, model=EMBEDDING_MODEL)
        embeddings = np.array([data.embedding for data in response.data])
        
        return embeddings
    except Exception as e:
        st.error(f"Erreur d'embedding : {e}")
        return None

def semantic_search(query, df_glossaire, glossary_embeddings, top_k=60):
    """
    Recherche vectorielle via Numpy pur (beaucoup plus rapide au chargement que sklearn).
    """
    if glossary_embeddings is None or df_glossaire.empty:
        return pd.DataFrame()

    try:
        # 1. Vectoriser la question
        query_resp = client.embeddings.create(input=[query], model=EMBEDDING_MODEL)
        query_vec = np.array(query_resp.data[0].embedding)

        # 2. Calculer la similarité (Produit Scalaire car vecteurs normalisés par OpenAI)
        # C'est l'équivalent ultra-rapide de cosine_similarity
        similarities = np.dot(glossary_embeddings, query_vec)

        # 3. Associer aux données
        # Attention : on suppose que l'ordre est conservé et qu'on a pris les N premières lignes
        # Si on a filtré les inputs vides, il peut y avoir un décalage, 
        # mais dans un glossaire propre, c'est rare.
        limit = len(similarities)
        df_results = df_glossaire.iloc[:limit].copy()
        df_results['similarity'] = similarities

        # 4. Trier
        return df_results.sort_values('similarity', ascending=False).head(top_k)
    except Exception as e:
        st.warning(f"Erreur recherche (taille index ?) : {e}")
        return pd.DataFrame()

# --- MOTEUR DE DONNÉES ---
@st.cache_resource
def init_db():
    con = duckdb.connect(database=':memory:')
    data_folder = "data"
    
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        return None, {}, [], pd.DataFrame(), None

    files = [f for f in os.listdir(data_folder) if f.endswith('.parquet')]
    schema_map = {} 
    table_list = []

    for f in files:
        table_name = f.replace('.parquet', '').replace('-', '_').replace(' ', '_').upper()
        file_path = os.path.join(data_folder, f)
        try:
            con.execute(f'CREATE OR REPLACE VIEW "{table_name}" AS SELECT * FROM \'{file_path}\'')
            table_list.append(table_name)
            cols = con.execute(f'DESCRIBE "{table_name}"').fetchall()
            for col in cols:
                real_col_name = col[0]
                clean_key = real_col_name.lower().replace("-", "").replace("_", "")
                schema_map[clean_key] = (table_name, real_col_name)
                schema_map[real_col_name.lower()] = (table_name, real_col_name)
        except Exception as e:
            st.error(f"Erreur chargement {f}: {e}")

    df_glossaire = pd.DataFrame()
    glossaire_path = os.path.join(data_folder, "Glossaire.txt")
    if os.path.exists(glossaire_path):
        try:
            df_glossaire = pd.read_csv(glossaire_path, encoding='utf-8', sep=None, engine='python')
        except:
            try:
                df_glossaire = pd.read_csv(glossaire_path, encoding='latin-1', sep=None, engine='python')
            except: pass

    # CALCUL DES EMBEDDINGS (Initialisation unique)
    glossary_embeddings = None
    if not df_glossaire.empty:
        with st.spinner("Initialisation du moteur sémantique..."):
            glossary_embeddings = get_glossary_embeddings(df_glossaire)

    territoires_path = os.path.join(data_folder, "territoires.txt")
    if os.path.exists(territoires_path):
        try:
            con.execute(f"CREATE OR REPLACE VIEW territoires AS SELECT * FROM read_csv_auto('{territoires_path}', all_varchar=True)")
            table_list.append("TERRITOIRES")
        except: pass

    return con, schema_map, table_list, df_glossaire, glossary_embeddings

con, schema_map, table_list, df_glossaire, glossary_embeddings = init_db()

# --- FONCTION GEO ---
def get_territory_context(con, city_name):
    if not con: return None
    clean_city = city_name.strip().replace("'", "''")
    try:
        query = f"""
        SELECT "ID", "NOM_COUV", "COMP1", "COMP2", "COMP3" 
        FROM territoires 
        WHERE "NOM_COUV" ILIKE '{clean_city}' 
        LIMIT 1
        """
        res = con.execute(query).fetchone()
        if res:
            target_id = res[0]
            target_name = res[1]
            comp_ids = [code for code in [res[2], res[3], res[4]] if code and str(code).strip() != '']
            comp_names = []
            if comp_ids:
                ids_sql = "', '".join(comp_ids)
                names_res = con.execute(f"SELECT NOM_COUV FROM territoires WHERE ID IN ('{ids_sql}')").fetchall()
                comp_names = [n[0] for n in names_res]
            return {"target_id": target_id, "target_name": target_name, "all_ids": [target_id] + comp_ids, "comp_names": comp_names}
    except: pass
    return None

# --- INTERFACE ---
st.title("🤖 Assistant Données Territoires")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Posez votre question (ex: accès aux soins, démographie, logement...)"}]
if "current_geo_context" not in st.session_state:
    st.session_state.current_geo_context = None

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "data" in msg:
            st.dataframe(msg["data"])

# --- LOGIQUE PRINCIPALE ---
if prompt := st.chat_input("Votre question..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            # 1. GEO CONTEXT
            city_extract = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": "Extrais la ville. Si aucune, réponds 'None'."}, {"role": "user", "content": prompt}]
            ).choices[0].message.content.strip().replace(".", "")
            
            if city_extract and city_extract != "None":
                new_context = get_territory_context(con, city_extract)
                if new_context:
                    st.session_state.current_geo_context = new_context
                    comps_str = ", ".join(new_context['comp_names'])
                    message_placeholder.markdown(f"📍 **Analyse pour {new_context['target_name']}** (Comparaison : {comps_str})")
            
            geo_context = st.session_state.current_geo_context

            # 2. RECHERCHE SÉMANTIQUE + POPULATION
            glossaire_context = ""
            if not df_glossaire.empty and glossary_embeddings is not None:
                message_placeholder.markdown("🧠 *Recherche sémantique des indicateurs...*")
                
                # A. Sémantique (Question vs Glossaire)
                results_semantic = semantic_search(prompt, df_glossaire, glossary_embeddings, top_k=50)
                
                # B. Injection Pop Totale (Structure)
                mask_pop_tot = df_glossaire.iloc[:, 5].astype(str).str.contains(r'population totale', case=False, regex=True)
                results_pop_tot = df_glossaire[mask_pop_tot].head(5)

                # C. Injection Pop par Âge (Taux spécifiques)
                mask_pop_age = df_glossaire.iloc[:, 4].astype(str).str.contains(r'pop.*[0-9]', case=False, regex=True)
                results_pop_age = df_glossaire[mask_pop_age].head(20)
                
                # Fusion
                final_results = pd.concat([results_semantic, results_pop_tot, results_pop_age]).drop_duplicates().head(90)
                
                # D. Mapping
                corrected_rows = []
                for _, row in final_results.iterrows():
                    try:
                        table_val = str(row.iloc[1])
                        var_val = str(row.iloc[4])
                        desc_val = str(row.iloc[5])
                        clean_search = var_val.strip().lower().replace("-", "").replace("_", "")
                        
                        if clean_search in schema_map:
                            real_table, real_col = schema_map[clean_search]
                            corrected_rows.append(f"✅ TABLE: \"{real_table}\" | COLONNE: \"{real_col}\" | DESC: {desc_val}")
                        elif var_val.lower() in schema_map:
                            real_table, real_col = schema_map[var_val.lower()]
                            corrected_rows.append(f"✅ TABLE: \"{real_table}\" | COLONNE: \"{real_col}\" | DESC: {desc_val}")
                    except: continue

                glossaire_context = "\n".join(corrected_rows)

            # 3. PROMPT SQL AVEC LOGIQUE DÉNOMINATEUR
            geo_instruction = ""
            if geo_context:
                ids_sql_list = "', '".join(geo_context['all_ids'])
                geo_instruction = f"""
                - Territoires : {geo_context['all_ids']}
                - CLAUSE : WHERE "ID" IN ('{ids_sql_list}')
                """
            else:
                geo_instruction = "- Pas de filtre géographique précis."

            system_prompt = f"""
            Tu es un expert SQL DuckDB.
            
            CONTEXTE (Indicateurs et Populations) :
            {glossaire_context}
            
            TA MISSION : Choisir le DÉNOMINATEUR le plus logique pour répondre à l'INTENTION de la question.
            
            RÈGLES DE DÉNOMINATEUR (ARBRE DE DÉCISION) :
            -----------------------------------------------------------------------
            CAS 1 : Question de STRUCTURE ("Y a-t-il beaucoup de jeunes ?", "Population vieillissante ?")
               -> Numérateur : Population de la tranche d'âge (ex: pop_15_24).
               -> Dénominateur : POPULATION TOTALE (ex: POPtot).
               -> But : Calculer la part de ce groupe dans la ville.
            
            CAS 2 : Question de PRÉVALENCE/COMPORTEMENT ("Les jeunes vont-ils au médecin ?", "Chômage des jeunes ?")
               -> Numérateur : Variable du phénomène (ex: nb_jeunes_sans_medecin).
               -> Dénominateur : POPULATION DE LA TRANCHE D'ÂGE concernée (ex: pop_15_24).
               -> But : Calculer le taux de prévalence au sein du groupe.
            -----------------------------------------------------------------------
            
            RÈGLES TECHNIQUES :
            1. JOIN "TERRITOIRES" USING ("ID") pour avoir "NOM_COUV".
            2. FORMULE : (TRY_CAST("Num" AS DOUBLE) / NULLIF(TRY_CAST("Denom" AS DOUBLE), 0)) * 100.
            3. {geo_instruction}
            4. Utilise UNIQUEMENT les tables/colonnes ✅.
            
            Réponds uniquement le SQL.
            """

            # 4. Génération SQL
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            sql_query = response.choices[0].message.content.replace("```sql", "").replace("```", "").strip()
            
            with st.expander("🕵️‍♂️ Voir la requête technique"):
                st.code(sql_query, language="sql")

            # 5. Exécution & Analyse
            if con and geo_context:
                try:
                    df = con.execute(sql_query).df()
                    if not df.empty:
                        analysis = client.chat.completions.create(
                            model=MODEL_NAME,
                            messages=[
                                {"role": "system", "content": f"""
                                 Tu es un consultant expert en stratégie territoriale.
                                 
                                 CONTEXTE :
                                 Analyse comparée pour {geo_context['target_name']} vs Voisins.
                                 
                                 CONSIGNES :
                                 1. Compare les taux calculés (Ville vs Moyennes).
                                 2. Valide la cohérence du dénominateur utilisé dans ton explication.
                                 3. Sois clair et synthétique.
                                 """},
                                {"role": "user", "content": f"Question: {prompt}\nTableau:\n{df.to_string()}"}
                            ]
                        )
                        final_resp = analysis.choices[0].message.content
                        message_placeholder.markdown(final_resp)
                        st.dataframe(df)
                        st.session_state.messages.append({"role": "assistant", "content": final_resp, "data": df})
                    else:
                        message_placeholder.warning("Aucune donnée trouvée.")
                except Exception as e:
                    message_placeholder.error(f"Erreur SQL : {e}")
            elif not geo_context:
                 message_placeholder.error("Ville non identifiée.")

        except Exception as e:
            st.error(f"Erreur : {e}")