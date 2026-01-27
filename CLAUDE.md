# CLAUDE.md - Terribot AI Assistant Guide

## Project Overview

**Terribot** is a French territorial analysis chatbot built with Streamlit that helps local government officials and agents analyze socioeconomic data about French territories (communes, EPCIs, departments, regions).

The application uses:
- **Streamlit** for the web interface
- **OpenAI API** (GPT models) for natural language understanding and SQL generation
- **DuckDB** for in-memory data querying
- **Parquet files** as the primary data source (~75 files with French statistical data)
- **Folium/Streamlit-Folium** for map visualizations

## Repository Structure

```
terribot/
├── app.py                 # Main application (~5300 lines, monolithic)
├── compare_logs.py        # Log comparison utility for tracking regressions
├── download_data.py       # Script to download parquet files (bypasses LFS issues)
├── requirements.txt       # Python dependencies
├── data/                  # Parquet data files + Glossaire.txt + territoires.txt
│   ├── *.parquet          # Statistical data tables (75 files)
│   ├── Glossaire.txt      # Variable metadata (columns, descriptions, sources)
│   └── territoires.txt    # Territory reference data (INSEE codes, names)
├── logs/                  # Session logs with git metadata and metrics
│   └── session_*.txt      # Timestamped session logs
├── .streamlit/
│   └── config.toml        # Streamlit theme configuration
├── .devcontainer/
│   └── devcontainer.json  # GitHub Codespaces configuration
├── DATA_SETUP.md          # Instructions for data file setup
├── FIX_LFS_ISSUE.md       # Documentation about Git LFS quota fix
└── README_LOGS.md         # Documentation about the logging system
```

## Key Architecture Components

### 1. Database Layer (`get_db_connection`)
- Creates DuckDB in-memory connection
- Loads all parquet files as views dynamically
- Creates tables for `glossaire` (variable metadata) and `territoires` (geographic reference)
- Initializes Full-Text Search (FTS) index on glossaire
- Stores schemas in `st.session_state.db_schemas`

### 2. Territory Resolution Pipeline
The app resolves user-mentioned territories through multiple stages:

1. **`check_territory_mentioned`** - Uses AI to detect if a territory is mentioned
2. **`search_territory_smart`** - SQL-based search with fuzzy matching (Jaro-Winkler)
3. **`get_broad_candidates`** - Gets candidate territories for disambiguation
4. **`ai_validate_territory`** - AI selects the best match among candidates
5. **`ai_fallback_territory_search`** - Web search fallback for unknown territories
6. **`build_geo_context_from_id`** - Builds geographic context with parent hierarchies

### 3. Variable Search (RAG System)
**`hybrid_variable_search`** combines:
- Semantic search using OpenAI embeddings (`text-embedding-3-small`)
- Full-text search on glossaire
- DuckDB keyword matching

Returns relevant variables from the glossaire for SQL generation.

### 4. SQL Generation Pipeline
1. **`generate_and_fix_sql`** - Generates SQL with automatic retry on errors
2. Injects table schemas on failures for self-correction
3. Uses strict output format: columns must be named `ID`, `NOM_COUV`

### 5. Visualization Components
- **`get_chart_configuration`** - AI selects columns and formats for charts
- **`ai_enhance_formats`** - AI enhances number formatting (percent, currency, etc.)
- **`auto_plot_data`** - Renders Vega-Lite charts
- **`render_epci_choropleth`** - Renders Folium choropleth maps

### 6. Response Streaming
Uses OpenAI streaming API to generate natural language responses about the data.

## Important Conventions

### Debug Logging
Use the `_dbg(label, **kwargs)` function for all debug output:
```python
_dbg("geo.search_smart.enter", input_str=input_str)
_dbg("sql.exec.result", empty=df.empty, rows=len(df))
```

Log labels follow the pattern: `module.function.stage`

### Print Statements
Use prefixed print statements for pipeline tracking:
```python
print("[TERRIBOT][PIPE] 🌍 analyze_territorial_scope()")
print("[TERRIBOT][DB] ✅ FTS index created")
print("[TERRIBOT][GEO] ✅ search_territory_smart results: X")
```

### SQL Column Naming
- Always use double quotes for column names: `t."column-name"`
- Output columns must be named exactly `ID` and `NOM_COUV`
- Never alias ID or NOM_COUV columns

### Number Formatting
The app uses French number formatting:
- Thousands separator: space (` `)
- Decimal separator: comma (`,`)
- Example: `1 234,56`

### Territory ID Formats
- Communes: 4-5 digits (e.g., `4112`, `75115`)
- EPCIs: 9 digits (SIREN code, e.g., `200068468`)
- Departments: `D` + number (e.g., `D4`, `D2A`, `D971`)
- Regions: `R` + number (e.g., `R11`, `R93`)
- France: `FR`

## Common Patterns

### Session State Keys
```python
st.session_state.messages          # Chat history
st.session_state.current_geo_context  # Current territory context
st.session_state.valid_tables_list # List of available tables
st.session_state.db_schemas        # Table schemas dict
st.session_state.current_viz_data  # Data for sidebar visualization
st.session_state.debug_data        # Debug information
```

### OpenAI API Calls
```python
response = client.responses.create(
    model=MODEL_NAME,  # Currently "gpt-5.2"
    input=build_messages(system_prompt, user_prompt),
    temperature=0,
)
text = extract_response_text(response)
```

For streaming:
```python
stream = client.responses.create(..., stream=True)
full_text = message_placeholder.write_stream(stream_response_text(stream))
```

### Error Handling in Pipeline
The main pipeline catches exceptions and provides user-friendly error messages:
- Timeout errors
- Rate limit errors
- API connection errors
- SQL/DuckDB errors

## Development Workflow

### Running Locally
```bash
streamlit run app.py
```

The app runs on port 8501 by default.

### Environment Variables
- `OPENAI_API_KEY` - Required for AI functionality
- Can also be set via Streamlit secrets (`.streamlit/secrets.toml`)

### Comparing Sessions
Use the log comparison tool to track regressions:
```bash
python compare_logs.py logs/session_before.txt logs/session_after.txt
```

### Data Setup
If parquet files are missing or show as LFS pointers:
```bash
python download_data.py
```

## Code Style Guidelines

### Language
- Code comments and variable names: primarily French
- User-facing text: French
- Documentation (CLAUDE.md, etc.): English or French depending on context

### Functions
- Functions are defined at module level (no classes except for dataclasses)
- Heavy use of `@st.cache_resource` and `@st.cache_data` for caching
- Type hints used sparingly

### Error Handling
- Use try/except with specific error messages
- Log errors with `_dbg("label.error", error=str(e))`
- Provide user-friendly error messages via `st.error()`

## Model Configuration

Current model: `gpt-5.2` (defined in `MODEL_NAME`)
Embedding model: `text-embedding-3-small` (defined in `EMBEDDING_MODEL`)

## Known Limitations

1. **Monolithic Structure**: All code is in `app.py` (~5300 lines)
2. **No Unit Tests**: Testing is done through session logs comparison
3. **LFS Dependency**: Data files require special handling (see DATA_SETUP.md)
4. **French-only**: The application is designed for French territories only

## Quick Reference - Key Functions

| Function | Line | Purpose |
|----------|------|---------|
| `get_db_connection()` | 221 | Initialize DuckDB with all data |
| `hybrid_variable_search()` | 1521 | RAG search for relevant variables |
| `analyze_territorial_scope()` | 2877 | Resolve territory from user input |
| `generate_and_fix_sql()` | 1288 | Generate SQL with auto-retry |
| `get_chart_configuration()` | 648 | AI-powered chart configuration |
| `auto_plot_data()` | 3744 | Render Vega-Lite charts |
| `render_epci_choropleth()` | 3147 | Render Folium maps |
| `ai_validate_territory()` | 2146 | AI territory disambiguation |
| `style_df()` | 1173 | Format DataFrame for display |

## Commit Message Conventions

Based on recent commits, use these prefixes:
- `fix:` / `Fix:` - Bug fixes
- `feat:` - New features
- `revert:` - Reverts
- Descriptive messages in French or English

## Testing Changes

Before submitting changes:
1. Run the app locally: `streamlit run app.py`
2. Test with sample queries (examples in the placeholder animation)
3. Check session logs in `logs/` directory
4. Compare metrics with previous sessions if available

## Useful Sample Queries

From the placeholder animation in app.py:
- "Fais un etat des lieux de la precarite a Manosque"
- "Toulouse est-elle une ville vieillissante ?"
- "Compare la situation sociale de Mulhouse avec Strasbourg"
- "Qui sont les habitants de Limoges Metropole ?"
