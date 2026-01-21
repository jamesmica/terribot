#!/usr/bin/env python3
"""
Script de comparaison de logs Terribot
Permet de comparer deux sessions pour suivre l'évolution du code et de la qualité des réponses
"""

import re
import sys
from pathlib import Path
from typing import Dict, Any, Optional


def parse_log_metadata(log_content: str) -> Dict[str, Any]:
    """Parse les métadonnées git d'un fichier de log"""
    metadata = {}

    # Parser la section GIT METADATA
    commit_match = re.search(r'Commit:\s+(\w+)\s+\(([a-f0-9]+)\)', log_content)
    if commit_match:
        metadata['commit_short'] = commit_match.group(1)
        metadata['commit_full'] = commit_match.group(2)

    branch_match = re.search(r'Branch:\s+(.+)', log_content)
    if branch_match:
        metadata['branch'] = branch_match.group(1).strip()

    author_match = re.search(r'Commit Author:\s+(.+)', log_content)
    if author_match:
        metadata['commit_author'] = author_match.group(1).strip()

    date_match = re.search(r'Commit Date:\s+(.+)', log_content)
    if date_match:
        metadata['commit_date'] = date_match.group(1).strip()

    msg_match = re.search(r'Commit Msg:\s+(.+)', log_content)
    if msg_match:
        metadata['commit_message'] = msg_match.group(1).strip()

    changes_match = re.search(r'Local Changes:\s+(.+)', log_content)
    if changes_match:
        metadata['has_local_changes'] = 'Yes' in changes_match.group(1)

    # Parser la date de session
    session_match = re.search(r'Session started:\s+(.+)', log_content)
    if session_match:
        metadata['session_date'] = session_match.group(1).strip()

    return metadata


def parse_log_metrics(log_content: str) -> Dict[str, Any]:
    """Parse les métriques de performance d'un fichier de log"""
    metrics = {}

    # Parser SESSION METRICS
    duration_match = re.search(r'Session Duration:\s+([\d.]+)s', log_content)
    if duration_match:
        metrics['duration_seconds'] = float(duration_match.group(1))

    sql_match = re.search(r'SQL Queries:\s+(\d+)\s+\(✅\s+(\d+)\s+/\s+❌\s+(\d+)\)', log_content)
    if sql_match:
        metrics['sql_total'] = int(sql_match.group(1))
        metrics['sql_success'] = int(sql_match.group(2))
        metrics['sql_errors'] = int(sql_match.group(3))

    api_match = re.search(r'API Calls:\s+(\d+)', log_content)
    if api_match:
        metrics['api_calls'] = int(api_match.group(1))

    responses_match = re.search(r'Responses Generated:\s+(\d+)', log_content)
    if responses_match:
        metrics['responses_generated'] = int(responses_match.group(1))

    return metrics


def parse_log_behavior(log_content: str) -> Dict[str, Any]:
    """Analyse le comportement détaillé de Terribot dans les logs"""
    behavior = {}

    # Compter les étapes du pipeline
    behavior['pipeline_geo'] = log_content.count('[TERRIBOT][PIPE] 🌍 analyze_territorial_scope()')
    behavior['pipeline_rag'] = log_content.count('[TERRIBOT][PIPE] 📚 RAG hybrid_variable_search()')
    behavior['pipeline_chart'] = log_content.count('[TERRIBOT][PIPE] 📈 get_chart_configuration()')
    behavior['pipeline_stream'] = log_content.count('[TERRIBOT][PIPE] 📝 Streaming response start')
    behavior['pipeline_done'] = log_content.count('[TERRIBOT][PIPE] ✅ Pipeline done')

    # Compter les erreurs et warnings
    behavior['warnings'] = log_content.count('⚠️')
    behavior['errors'] = log_content.count('❌')
    behavior['successes'] = log_content.count('✅')

    # Extraire les requêtes SQL (premières lignes)
    sql_queries = re.findall(r'\[TERRIBOT\]\[DBG\] sql\.exec\.about_to_run :: sql=(.*?)(?=\[TERRIBOT\]|\Z)', log_content, re.DOTALL)
    behavior['sql_queries_count'] = len(sql_queries)

    return behavior


def compare_logs(log1_path: Path, log2_path: Path):
    """Compare deux fichiers de logs et affiche les différences"""

    # Lire les fichiers
    try:
        with open(log1_path, 'r', encoding='utf-8') as f:
            log1_content = f.read()
    except FileNotFoundError:
        print(f"❌ Fichier introuvable: {log1_path}")
        return

    try:
        with open(log2_path, 'r', encoding='utf-8') as f:
            log2_content = f.read()
    except FileNotFoundError:
        print(f"❌ Fichier introuvable: {log2_path}")
        return

    # Parser les métadonnées
    meta1 = parse_log_metadata(log1_content)
    meta2 = parse_log_metadata(log2_content)

    # Parser les métriques
    metrics1 = parse_log_metrics(log1_content)
    metrics2 = parse_log_metrics(log2_content)

    # Parser le comportement
    behavior1 = parse_log_behavior(log1_content)
    behavior2 = parse_log_behavior(log2_content)

    # Afficher la comparaison
    print("=" * 80)
    print("COMPARAISON DE LOGS TERRIBOT")
    print("=" * 80)
    print()

    print("📁 FICHIERS")
    print(f"  Log 1: {log1_path.name}")
    print(f"  Log 2: {log2_path.name}")
    print()

    print("🔍 MÉTADONNÉES GIT")
    print("-" * 80)

    def compare_field(label: str, val1: Any, val2: Any):
        if val1 == val2:
            print(f"  {label:20s} ✓ Identique: {val1}")
        else:
            print(f"  {label:20s} ⚠️  Différent")
            print(f"    Log 1: {val1}")
            print(f"    Log 2: {val2}")

    compare_field("Commit", meta1.get('commit_short'), meta2.get('commit_short'))
    compare_field("Branche", meta1.get('branch'), meta2.get('branch'))
    compare_field("Date commit", meta1.get('commit_date'), meta2.get('commit_date'))
    compare_field("Message commit", meta1.get('commit_message'), meta2.get('commit_message'))
    print()

    print("📊 MÉTRIQUES DE PERFORMANCE")
    print("-" * 80)

    def compare_metric(label: str, val1: Optional[float], val2: Optional[float], unit: str = ""):
        if val1 is None or val2 is None:
            print(f"  {label:25s} Données manquantes")
            return

        diff = val2 - val1
        pct = (diff / val1 * 100) if val1 != 0 else 0

        if diff > 0:
            symbol = "📈"
            sign = "+"
        elif diff < 0:
            symbol = "📉"
            sign = ""
        else:
            symbol = "="
            sign = "="

        print(f"  {label:25s} {val1}{unit} → {val2}{unit}  {symbol} {sign}{diff}{unit} ({pct:+.1f}%)")

    compare_metric("Durée session", metrics1.get('duration_seconds'), metrics2.get('duration_seconds'), "s")
    compare_metric("Requêtes SQL", metrics1.get('sql_total'), metrics2.get('sql_total'))
    compare_metric("SQL succès", metrics1.get('sql_success'), metrics2.get('sql_success'))
    compare_metric("SQL erreurs", metrics1.get('sql_errors'), metrics2.get('sql_errors'))
    compare_metric("Appels API", metrics1.get('api_calls'), metrics2.get('api_calls'))
    compare_metric("Réponses générées", metrics1.get('responses_generated'), metrics2.get('responses_generated'))
    print()

    print("🤖 COMPORTEMENT DE TERRIBOT")
    print("-" * 80)

    def compare_count(label: str, val1: int, val2: int):
        diff = val2 - val1
        if diff > 0:
            print(f"  {label:25s} {val1} → {val2}  (+{diff})")
        elif diff < 0:
            print(f"  {label:25s} {val1} → {val2}  ({diff})")
        else:
            print(f"  {label:25s} {val1} → {val2}  (identique)")

    compare_count("Géolocalisation", behavior1.get('pipeline_geo', 0), behavior2.get('pipeline_geo', 0))
    compare_count("RAG/Recherche", behavior1.get('pipeline_rag', 0), behavior2.get('pipeline_rag', 0))
    compare_count("Config graphique", behavior1.get('pipeline_chart', 0), behavior2.get('pipeline_chart', 0))
    compare_count("Réponses stream", behavior1.get('pipeline_stream', 0), behavior2.get('pipeline_stream', 0))
    compare_count("Pipelines complétés", behavior1.get('pipeline_done', 0), behavior2.get('pipeline_done', 0))
    print()
    compare_count("⚠️  Warnings", behavior1.get('warnings', 0), behavior2.get('warnings', 0))
    compare_count("❌ Erreurs", behavior1.get('errors', 0), behavior2.get('errors', 0))
    compare_count("✅ Succès", behavior1.get('successes', 0), behavior2.get('successes', 0))
    print()

    print("=" * 80)


def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_logs.py <log1.txt> <log2.txt>")
        print()
        print("Exemple:")
        print("  python compare_logs.py logs/session_2026-01-21_10-00-00.txt logs/session_2026-01-21_11-00-00.txt")
        sys.exit(1)

    log1_path = Path(sys.argv[1])
    log2_path = Path(sys.argv[2])

    compare_logs(log1_path, log2_path)


if __name__ == "__main__":
    main()
