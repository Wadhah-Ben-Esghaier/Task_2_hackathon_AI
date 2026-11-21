# Task 2 — Model Equality Testing

Les scripts de départ et les réponses de référence viennent du [dépôt officiel sur Hugging Face](https://huggingface.co/datasets/SprintML/model_equality/tree/main).

## Premier essai dans VS Code

Dans le terminal PowerShell du projet, saisis la clé d’équipe à l’invite masquée, puis lance le script de pratique.

```powershell
$task2Secret = Read-Host "Cle API de l'equipe" -AsSecureString
$env:MODEL_EQUALITY_API_KEY = [System.Net.NetworkCredential]::new('', $task2Secret).Password
Remove-Variable task2Secret
.\.venv\Scripts\python.exe query_reference.py
```

Le script utilise le prompt de `reference_prompts.txt` et ajoute les réponses des endpoints de pratique dans `reference_answers.jsonl`. La clé est disponible dans la session PowerShell en cours.

Pour évaluer un CSV de prédictions localement, lance `.\.venv\Scripts\python.exe score_practice.py mon_essai.csv`. Le CSV de prédictions utilise les colonnes `endpoint,family,model`.

## Recréer l’environnement si besoin

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install "requests>=2.28"
```

`requests` suffit pour exécuter les scripts fournis. `requirements.txt` liste aussi des bibliothèques d’analyse facultatives.

## Endpoints évalués et soumission

`task_template.py` contient la fonction `solve()`. Le réglage `QUERY` contrôle l’envoi de prompts aux endpoints évalués à partir de `prompts.txt`. Pour envoyer tes résultats, indique le chemin de ton CSV final dans `SUBMISSION_FILE` dans `submission.py`.
