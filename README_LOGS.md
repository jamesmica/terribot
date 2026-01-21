# Configuration des Logs GitHub pour Terribot

Ce document explique comment configurer Terribot pour qu'il enregistre automatiquement les fichiers de logs sur GitHub à chaque session.

## Fonctionnement

À chaque démarrage de l'application, un nouveau fichier de log est créé avec le format `session_YYYY-MM-DD_HH-MM-SS.txt`.

Lorsque l'application se termine, le fichier de log est automatiquement poussé vers le dossier `logs/` de votre repository GitHub.

## Configuration

### Étape 1 : Créer un token GitHub

1. Allez sur [GitHub Settings > Tokens](https://github.com/settings/tokens)
2. Cliquez sur "Generate new token" > "Generate new token (classic)"
3. Donnez un nom au token (ex: "Terribot Logs")
4. Sélectionnez les permissions suivantes :
   - ✅ **repo** (toutes les sous-permissions)
5. Cliquez sur "Generate token"
6. **Copiez le token immédiatement** (vous ne pourrez plus le voir après)

### Étape 2 : Configurer les secrets Streamlit

#### En local

1. Copiez le fichier d'exemple :
   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   ```

2. Éditez `.streamlit/secrets.toml` et remplacez les valeurs :
   ```toml
   GITHUB_TOKEN = "ghp_votre_token_ici"
   GITHUB_REPO = "votre-username/votre-repo"
   OPENAI_API_KEY = "votre_cle_api_openai"
   ```

3. Le fichier `.streamlit/secrets.toml` est automatiquement ignoré par git (défini dans `.gitignore`)

#### Sur Streamlit Cloud

1. Allez sur votre application sur [Streamlit Cloud](https://share.streamlit.io/)
2. Cliquez sur "Settings" > "Secrets"
3. Ajoutez les secrets suivants :
   ```toml
   GITHUB_TOKEN = "ghp_votre_token_ici"
   GITHUB_REPO = "votre-username/votre-repo"
   OPENAI_API_KEY = "votre_cle_api_openai"
   ```
4. Cliquez sur "Save"

### Étape 3 : Vérification

Lancez l'application :

```bash
streamlit run app.py
```

À la fin de la session (quand vous fermez l'application), vous devriez voir dans les logs :

```
[TERRIBOT] 📤 Envoi du log vers GitHub...
[TERRIBOT][GITHUB] ✅ Log poussé vers GitHub: logs/session_YYYY-MM-DD_HH-MM-SS.txt
```

Vérifiez sur GitHub que le fichier a bien été créé dans le dossier `logs/`.

## Dépannage

### Erreur : "GITHUB_TOKEN ou GITHUB_REPO manquant"

- Vérifiez que vous avez bien créé le fichier `.streamlit/secrets.toml`
- Vérifiez que les clés sont correctement nommées (sensible à la casse)

### Erreur : "401 Unauthorized"

- Votre token GitHub est invalide ou expiré
- Créez un nouveau token et mettez à jour `secrets.toml`

### Erreur : "403 Forbidden"

- Le token n'a pas les permissions nécessaires
- Recréez un token avec la permission **repo** complète

### Erreur : "404 Not Found"

- Le nom du repository est incorrect
- Vérifiez le format : `username/repo` (ex: `jamesmica/terribot`)

### Les logs ne sont pas poussés

- Vérifiez que l'application se termine proprement (pas de crash)
- Consultez les logs du terminal pour voir les messages d'erreur

## Sécurité

⚠️ **Important** :
- Ne commitez **JAMAIS** votre fichier `secrets.toml`
- Ne partagez **JAMAIS** votre token GitHub
- Si votre token est exposé, révoquez-le immédiatement sur GitHub

## Branche par défaut

Par défaut, les logs sont poussés sur la branche `main`. Si votre branche principale s'appelle différemment (ex: `master`), vous pouvez modifier la ligne dans `app.py` :

```python
payload = {
    "message": f"Add log file {log_filename}",
    "content": content_base64,
    "branch": "main"  # Changez ici si nécessaire
}
```
