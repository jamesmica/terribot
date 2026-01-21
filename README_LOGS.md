# 📝 Système de Suivi et Logs - Terribot

Ce document explique le système de logs enrichi de Terribot, conçu pour suivre l'évolution du code, de la qualité des réponses et du raisonnement au fil des versions et des pull requests.

## 🎯 Objectifs

Le système de logs permet de suivre :
1. **L'évolution du code** : Métadonnées git (commit, branche, auteur) dans chaque log
2. **L'évolution de la qualité** : Métriques de performance (temps, succès/échecs, appels API)
3. **Les étapes du raisonnement** : Logs détaillés du pipeline de traitement

## 📄 Format des Logs

Chaque session génère un fichier `logs/session_YYYY-MM-DD_HH-MM-SS.txt` avec :

### 1. En-tête avec métadonnées Git

```
================================================================================
SESSION LOG - TERRIBOT
================================================================================
Session started: 2026-01-21 19:45:32

--- GIT METADATA ---
Commit:        bec5b11 (bec5b11a2f3c4d5e6f7g8h9i0j1k2l3m4n5o6p7)
Branch:        claude/fix-github-logs-creation-iAS0d
Commit Author: Claude
Commit Date:   2026-01-21 19:43:25 +0100
Commit Msg:    revert: Simplification du système de logs
Local Changes: No
================================================================================
```

### 2. Logs de session

Tous les prints et messages de debug du pipeline :
- `[TERRIBOT][PIPE]` : Étapes du pipeline principal
- `[TERRIBOT][DBG]` : Debug détaillé avec paramètres
- `[TERRIBOT][SQL]` : Exécution et validation SQL
- `[TERRIBOT][GITHUB]` : Opérations GitHub (si applicable)

### 3. Pied de page avec métriques

```
================================================================================
SESSION METRICS
================================================================================
Session Duration:     45.23s
SQL Queries:          3 (✅ 3 / ❌ 0)
API Calls:            5
Responses Generated:  2
================================================================================
```

## 🔧 Utilisation

### Générer des logs

Les logs sont créés automatiquement à chaque lancement :

```bash
streamlit run app.py
```

Le fichier de log est créé dans `logs/` avec l'horodatage du lancement.

### Consulter un log

```bash
cat logs/session_2026-01-21_19-45-32.txt
```

Ou ouvrir dans un éditeur de texte.

### Comparer deux logs

Utilisez le script `compare_logs.py` pour comparer deux sessions :

```bash
python compare_logs.py logs/session_2026-01-21_10-00-00.txt logs/session_2026-01-21_11-00-00.txt
```

#### Exemple de sortie

```
================================================================================
COMPARAISON DE LOGS TERRIBOT
================================================================================

📁 FICHIERS
  Log 1: session_2026-01-21_10-00-00.txt
  Log 2: session_2026-01-21_11-00-00.txt

🔍 MÉTADONNÉES GIT
--------------------------------------------------------------------------------
  Commit               ⚠️  Différent
    Log 1: abc1234
    Log 2: def5678
  Branche              ✓ Identique: main
  Date commit          ⚠️  Différent
    Log 1: 2026-01-21 10:00:00
    Log 2: 2026-01-21 11:00:00
  Message commit       ⚠️  Différent
    Log 1: feat: Amélioration du RAG
    Log 2: fix: Correction SQL

📊 MÉTRIQUES DE PERFORMANCE
--------------------------------------------------------------------------------
  Durée session            42.5s → 38.2s  📉 -4.3s (-10.1%)
  Requêtes SQL             5 → 4  📉 -1 (-20.0%)
  SQL succès               5 → 4  📉 -1 (-20.0%)
  SQL erreurs              0 → 0  = =0 (0.0%)
  Appels API               8 → 7  📉 -1 (-12.5%)
  Réponses générées        2 → 2  = =0 (0.0%)

🤖 COMPORTEMENT DE TERRIBOT
--------------------------------------------------------------------------------
  Géolocalisation          2 → 2  (identique)
  RAG/Recherche            2 → 2  (identique)
  Config graphique         2 → 2  (identique)
  Réponses stream          2 → 2  (identique)
  Pipelines complétés      2 → 2  (identique)

  ⚠️  Warnings              1 → 0  (-1)
  ❌ Erreurs               0 → 0  (identique)
  ✅ Succès                12 → 13  (+1)
```

## 📊 Métriques Trackées

Le système track automatiquement :

| Métrique | Description | Où c'est tracké |
|----------|-------------|-----------------|
| **SQL Queries** | Nombre de requêtes SQL exécutées | À chaque `con.execute()` dans le pipeline principal |
| **SQL Success/Errors** | Succès vs échecs SQL | Try/except autour des exécutions SQL |
| **API Calls** | Appels à l'API OpenAI | Après chaque `client.chat.completions.create()` |
| **Responses Generated** | Réponses streamées générées | Après chaque streaming de réponse |
| **Session Duration** | Durée totale de la session | Calculée automatiquement à la fin |

## 🔄 Workflow de Suivi

### 1. Développement d'une nouvelle fonctionnalité

```bash
# 1. Créer une branche
git checkout -b feature/nouvelle-fonctionnalite

# 2. Faire vos modifications
# ...

# 3. Tester et générer des logs
streamlit run app.py
# Utiliser l'application, les logs sont créés automatiquement

# 4. Consulter les logs
cat logs/session_2026-01-21_XX-XX-XX.txt

# 5. Comparer avec la version précédente
python compare_logs.py logs/session_old.txt logs/session_new.txt
```

### 2. Suivi de l'évolution entre PRs

```bash
# 1. Logs sur la branche main avant PR
git checkout main
streamlit run app.py
# → logs/session_main_before.txt

# 2. Logs sur la branche feature après modifications
git checkout feature/ma-feature
streamlit run app.py
# → logs/session_feature.txt

# 3. Comparer les deux
python compare_logs.py logs/session_main_before.txt logs/session_feature.txt

# 4. Commiter les logs si pertinent
git add logs/session_feature.txt
git commit -m "docs: Ajout logs de test pour feature X"
```

### 3. Analyse de régression

Si une PR dégrade les performances :

```bash
# Comparer les logs avant/après merge
python compare_logs.py logs/session_before_merge.txt logs/session_after_merge.txt
```

Vous verrez immédiatement :
- ⬆️ Augmentation du temps de réponse
- ⬆️ Augmentation des appels API
- ⬆️ Augmentation des erreurs SQL
- Etc.

## 📈 Bonnes Pratiques

### 1. Commiter les logs importants

Ne commitez que les logs pertinents (tests significatifs) :

```bash
# Bon : log d'un test complet de validation
git add logs/session_validation_complete.txt

# Éviter : logs de debug local
# (ne pas commiter tous les logs)
```

### 2. Nommer les sessions de test

Renommez les logs importants pour faciliter le suivi :

```bash
# Après une session de test importante
mv logs/session_2026-01-21_10-00-00.txt logs/test_rag_improvement_v1.txt
git add logs/test_rag_improvement_v1.txt
```

### 3. Créer des benchmarks

Établissez des logs de référence pour chaque fonctionnalité majeure :

```
logs/
  benchmarks/
    benchmark_geo_simple.txt      # Requête géo simple
    benchmark_geo_complex.txt     # Requête géo complexe
    benchmark_sql_basic.txt       # SQL de base
    benchmark_sql_aggregation.txt # SQL avec agrégations
```

### 4. Analyser régulièrement

Avant chaque merge vers main :

```bash
# Comparer avec le benchmark de référence
python compare_logs.py logs/benchmarks/benchmark_geo_simple.txt logs/session_current.txt
```

## 🛠️ Extension du Système

Le système est extensible. Pour ajouter de nouvelles métriques :

### 1. Modifier la classe PerformanceMetrics

Dans `app.py`, ajoutez une nouvelle métrique :

```python
class PerformanceMetrics:
    def __init__(self):
        # ... métriques existantes
        self.custom_metric = 0

    def log_custom_metric(self, value):
        self.custom_metric += value

    def get_summary(self):
        summary = {
            # ... métriques existantes
            'custom_metric': self.custom_metric
        }
        return summary
```

### 2. Tracker la métrique dans le code

```python
# Quelque part dans app.py
metrics.log_custom_metric(1)
```

### 3. Mettre à jour le footer

Dans `DualLogger.write_footer()`, ajoutez l'affichage :

```python
footer += f"Custom Metric:        {metrics_summary['custom_metric']}\n"
```

### 4. Mettre à jour le comparateur

Dans `compare_logs.py`, ajoutez le parsing et la comparaison.

## 📚 Cas d'Usage

### Cas 1 : Amélioration du RAG

**Objectif** : Vérifier que le nouveau système RAG améliore la pertinence

```bash
# Avant amélioration
python compare_logs.py logs/before_rag_improvement.txt logs/after_rag_improvement.txt
```

**Attendu** :
- ⬇️ Moins d'appels API (meilleur contexte)
- ⬆️ Même nombre de réponses générées
- ⬇️ Moins d'erreurs SQL (meilleures variables trouvées)

### Cas 2 : Optimisation des performances

**Objectif** : Réduire le temps de réponse

```bash
python compare_logs.py logs/before_optimization.txt logs/after_optimization.txt
```

**Attendu** :
- ⬇️ Durée de session réduite
- = Même qualité de réponses
- = Même nombre de succès

### Cas 3 : Debugging d'une régression

**Symptôme** : Les utilisateurs rapportent plus d'erreurs

```bash
# Comparer les logs récents avec les anciens
python compare_logs.py logs/stable_version.txt logs/current_version.txt
```

**Indicateurs** :
- ⬆️ Augmentation des erreurs SQL
- ⬆️ Augmentation des warnings
- ⬇️ Diminution des succès

## 🔍 Debugging avec les Logs

Les logs contiennent des informations détaillées pour le debugging :

```bash
# Chercher toutes les erreurs SQL
grep "❌" logs/session_XXX.txt

# Voir les requêtes SQL générées
grep "sql.exec.about_to_run" logs/session_XXX.txt

# Suivre le pipeline complet d'une requête
grep "\[TERRIBOT\]\[PIPE\]" logs/session_XXX.txt

# Voir les appels à l'IA de géolocalisation
grep "geo.ai_validate" logs/session_XXX.txt
```

## 💡 Tips

1. **Garder des logs de référence** : Sauvegardez des logs "gold standard" pour chaque type de requête
2. **Automatiser les comparaisons** : Créez des scripts pour comparer automatiquement les nouvelles versions avec les références
3. **Utiliser git pour le versioning** : Les logs commitables permettent de voir l'évolution historique
4. **Ne pas sur-commiter** : Sélectionnez seulement les logs significatifs pour éviter de polluer le repo

## 🚀 Prochaines Améliorations Possibles

- Dashboard web pour visualiser l'évolution des métriques
- Tests automatisés qui comparent les métriques avec des seuils
- Export des métriques en JSON pour analyse automatisée
- Intégration avec GitHub Actions pour valider les PRs automatiquement
