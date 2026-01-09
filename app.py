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
    initial_sidebar_state="expanded" # La sidebar est ouverte par défaut
)

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stChatInput {padding-bottom: 20px;}
    .stDataFrame {border: 1px solid #f0f2f6; border-radius: 5px;}
    /* header {visibility: hidden;}  <-- CETTE LIGNE CACHIT LE BOUTON DE LA SIDEBAR */
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
    st.caption("Intelligence Territoriale v0.1")
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
    st.info("💡 **Comparaison :** Je fusionne tous les territoires (Cibles + Voisins + France) dans une seule liste.")

client = openai.OpenAI(api_key=api_key)
MODEL_NAME = "gpt-5.2-2025-12-11"
EMBEDDING_MODEL = "text-embedding-3-small"

# --- 4. FONCTIONS VECTORIELLES ---
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

# --- 5. MOTEUR DE DONNÉES ---
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
            # 1. Inspection du schéma
            temp_view = f"temp_{table_name}"
            con.execute(f"CREATE OR REPLACE VIEW {temp_view} AS SELECT * FROM read_parquet('{file_path}') LIMIT 1")
            cols_info = con.execute(f"DESCRIBE {temp_view}").fetchall()
            
            # 2. Construction requête CAST
            col_defs = []
            for col_name, col_type, _, _, _, _ in cols_info:
                if col_name.upper() in ['ID', 'CODGEO', 'CODE_GEO']:
                    col_defs.append(f"CAST(\"{col_name}\" AS VARCHAR) AS \"{col_name}\"")
                else:
                    col_defs.append(f"\"{col_name}\"")
            
            select_stmt = ", ".join(col_defs)
            
            # 3. Vue finale sur FICHIER
            con.execute(f"CREATE OR REPLACE VIEW \"{table_name}\" AS SELECT {select_stmt} FROM read_parquet('{file_path}')")
            con.execute(f"DROP VIEW {temp_view}")

            # Mapping
            for col in cols_info:
                real_col = col[0]
                clean_key = real_col.lower().replace("-", "").replace("_", "")
                schema_map[clean_key] = (table_name, real_col)
                schema_map[real_col.lower()] = (table_name, real_col)
                
        except Exception as e:
            try:
                con.execute(f'CREATE OR REPLACE VIEW "{table_name}" AS SELECT * FROM read_parquet(\'{file_path}\')')
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

# --- 6. INTELLIGENCE GÉOGRAPHIQUE ---
def normalize_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")
    text = text.replace('-', ' ').replace("'", " ").replace("’", " ")
    text = text.replace("st ", "saint ").replace("ste ", "sainte ")
    return text.strip()

def search_territory_smart(con, input_str):
    clean_raw = input_str.strip()
    
    if clean_raw.isdigit() and len(clean_raw) <= 3: 
        query = f"SELECT ID, NOM_COUV, COMP1, COMP2, COMP3 FROM territoires WHERE ID = 'D{clean_raw}' LIMIT 1"
        try:
            res = con.execute(query).fetchone()
            if res: return res
        except: pass

    norm_input = normalize_text(clean_raw)
    sql_clean_col = "lower(replace(replace(replace(replace(NOM_COUV, '-', ' '), '''', ' '), '’', ' '), 'St ', 'Saint '))"
    
    query = f"""
    SELECT ID, NOM_COUV, COMP1, COMP2, COMP3 
    FROM territoires 
    WHERE "ID" = '{clean_raw}' OR {sql_clean_col} LIKE '%{norm_input}%'
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
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Extrais TOUTES les entités géographiques. JSON: {\"lieux\": [\"Cergy\", \"93\"]}. Si aucun, {\"lieux\": []}."},
                {"role": "user", "content": rewritten_prompt}
            ],
            response_format={"type": "json_object"}
        )
        data = json.loads(extraction.choices[0].message.content)
        lieux_cites = data.get("lieux", [])
        
        with st.expander("🌍 Debug: Extraction Territoires (IA)", expanded=False):
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

    with st.expander("🔍 Debug: Résolution Territoires (DB)", expanded=False):
        st.table(pd.DataFrame(debug_search))

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
    is_hierarchical = any(k in prompt_lower for k in ["toutes les", "tous les", "liste des"])
    
    if is_hierarchical and primary_res:
        type_filter = ""
        if any(k in prompt_lower for k in ["commune", "ville"]):
            type_filter = "length(t.ID) = 5" 
            display_suffix = " (Communes)"
        elif any(k in prompt_lower for k in ["epci", "interco"]):
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
        "display_context": f"{display_name}{display_suffix}"
    }

# --- 7. UI ---
st.title("🗺️ Terribot")
st.markdown("#### L'expert des données territoriales")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Bonjour ! Quel territoire souhaitez-vous analyser ?"}]
if "current_geo_context" not in st.session_state:
    st.session_state.current_geo_context = None

for msg in st.session_state.messages:
    avatar = "🤖" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if "data" in msg: st.dataframe(msg["data"], use_container_width=True)

# --- 8. TRAITEMENT ---
inject_placeholder_animation()

if prompt := st.chat_input("Posez votre question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"): st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        
        try:
            # A. REFORMULATION (Mémoire conversationnelle RENFORCÉE)
            history_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-4:]])
            current_geo_name = st.session_state.current_geo_context['target_name'] if st.session_state.current_geo_context else ""
            
            reformulation = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"""
                    Tu es un expert en reformulation de questions pour l'analyse de données.
                    CONTEXTE GÉOGRAPHIQUE ACTUEL : '{current_geo_name}'
                    
                    TES OBJECTIFS :
                    1. Rendre la question de l'utilisateur totalement autonome (compréhensible sans l'historique).
                    2. Si l'utilisateur dit "ramène à la population", "et pour le chômage ?", "compare avec X", tu DOIS réintégrer le SUJET de la conversation précédente (ex: les crimes, les impôts, les jeunes).
                    3. Si aucun lieu n'est cité, réinjecte '{current_geo_name}'.
                    
                    EXEMPLES :
                    - User: "Et en 2020 ?" (Sujet précédent: Chômage à Paris) -> "Quel était le taux de chômage à Paris en 2020 ?"
                    - User: "Ramène ça à la population" (Sujet précédent: Crimes à Fontenay) -> "Quels sont les taux de criminalité et délits par habitant à Fontenay-sous-Bois ?"
                    """},
                    {"role": "user", "content": f"Historique de la conversation :\n{history_text}\n\nDernière question à reformuler : {prompt}"}
                ]
            )
            rewritten_prompt = reformulation.choices[0].message.content
            
            with st.expander("🗣️ Debug: Reformulation", expanded=False):
                st.write(f"**Original:** {prompt}")
                st.write(f"**Reformulé:** {rewritten_prompt}")

            # B. GEO SCOPE
            new_context = analyze_territorial_scope(con, rewritten_prompt)
            
            if new_context:
                st.session_state.current_geo_context = new_context
                message_placeholder.info(f"📍 **Périmètre :** {new_context['display_context']}")
                
                with st.expander("📍 Debug: Périmètre Final (IDs)", expanded=False):
                    st.write(f"**IDs retenus:** {new_context['all_ids']}")
                    if new_context['parent_clause']:
                        st.write(f"**Clause Enfants:** {new_context['parent_clause']}")
                        
            elif not st.session_state.current_geo_context:
                message_placeholder.warning("⚠️ Je ne détecte pas de territoire. Précisez une ville.")
                st.stop()
            
            geo_context = st.session_state.current_geo_context

            # C. RAG
            glossaire_context = ""
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
                    glossaire_context = "\n".join(rows)
                    
                    with st.expander("📚 Debug: Variables Trouvées (RAG)", expanded=False):
                        st.text(glossaire_context[:2000] + "...")

            # D. SQL (SÉCURISÉ)
            ids_sql = ", ".join([f"'{str(i)}'" for i in geo_context['all_ids']])
            parent_clause = geo_context.get('parent_clause', '')
            
            system_prompt = f"""
            Tu es Terribot.
            CONTEXTE DONNÉES :
            {glossaire_context}
            
            SCHEMA TABLE "TERRITOIRES" (alias t) :
            - "ID" (VARCHAR)
            - "NOM_COUV" (VARCHAR)
            
            MISSION : Répondre à "{rewritten_prompt}" via SQL.
            
            RÈGLES CRITIQUES :
            1. PÉRIMÈTRE GÉOGRAPHIQUE :
               - IDs : {geo_context['all_ids']}
               - CLAUSE OBLIGATOIRE : WHERE (t."ID" IN ({ids_sql}) {parent_clause})
               - 🚨 INTERDIT : Ne fais PAS de UNION ALL complexe. Fais UNE seule requête simple qui sélectionne les colonnes demandées pour les territoires filtrés.
               - 🚨 INTERDIT : Ne fais PAS de ORDER BY sur des colonnes calculées ou des alias complexes. Trie simplement par t."NOM_COUV".
            
            2. JOIN :
               - JOIN "TERRITOIRES" t USING ("ID").
               - SELECT t."NOM_COUV", ...
            
            3. CASTING :
               - DONNÉES -> TRY_CAST("col" AS DOUBLE).
            
            Réponds uniquement le SQL.
            """

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": rewritten_prompt}],
                temperature=0
            )
            sql_query = response.choices[0].message.content.replace("```sql", "").replace("```", "").strip()
            
            with st.expander("🛠️ Debug: Requête SQL Générée", expanded=False):
                st.code(sql_query, language="sql")

            # E. EXECUTION
            if con:
                try:
                    df = con.execute(sql_query).df()
                    
                    with st.expander("📊 Debug: Données Brutes (DataFrame)", expanded=False):
                        st.dataframe(df)

                    if not df.empty:
                        with st.status("Analyse en cours...", expanded=True) as status:
                            analysis = client.chat.completions.create(
                                model=MODEL_NAME,
                                messages=[
                                    {"role": "system", "content": f"Tu es Terribot. Tu viens d'interroger la base. Analyse les chiffres pour l'utilisateur : {rewritten_prompt}. Sois clair, compare les territoires trouvés."},
                                    {"role": "user", "content": df.to_string()}
                                ]
                            )
                            status.update(label="Terminé", state="complete", expanded=False)
                        
                        final_resp = analysis.choices[0].message.content
                        message_placeholder.markdown(final_resp)
                        st.dataframe(df, use_container_width=True)
                        st.session_state.messages.append({"role": "assistant", "content": final_resp, "data": df})
                    else:
                        message_placeholder.warning("Aucune donnée trouvée.")
                except Exception as e:
                    message_placeholder.error("Erreur technique.")
                    with st.expander("Debug Erreur"): st.write(e)

        except Exception as e:
            st.error(f"Erreur : {e}")