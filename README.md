# "Maintenance-Predictive-Detection-de-Pannes-Machines-par-ML"  
detection de pannes machines rares (0,09% des cas) à partir de 9 capteurs industriels. Comparaison de 3 algorithmes (Logistic Regression, XGBoost, Random Forest) avec feature engineering temporel avancé et explicabilité SHAP.


🎯 Contexte & Problématique
Objectif métier : Anticiper les pannes machines avant qu'elles surviennent pour réduire les arrêts non planifiés et les coûts de maintenance curative.
Problème ML : Classification binaire sur séries temporelles fortement déséquilibrées — seulement 0,09 % de pannes dans le dataset.
Métrique cible : Maximiser le Rappel et la PR-AUC (Precision-Recall AUC) — particulièrement informative sur classes déséquilibrées. Arbitrage via le F2-Score pour limiter les faux positifs.

🏗️ Architecture du Pipeline
[Données capteurs]      [Prétraitement]        [Feature Engineering]    [Modélisation]
124 494 observations →  Nettoyage         →   Lags (J-1, J-7)      →  Logistic Regression
9 métriques (metric1    Suppression            Moyennes mobiles (7j)    XGBoost (scale_pos_weight)
à metric9)              doublons               Volatilité (std 7j)      Random Forest + SMOTE
0.09% de pannes         metric8 retirée        Variations (diff)        TimeSeriesSplit CV
                        (corrélée à metric7)   Split chronologique      SHAP Explicabilité
                                               80% train / 20% test

🗂️ Structure du Dépôt
predictive-maintenance/
│
├── notebook/
│   └── predictive_maintenance.ipynb   # Notebook principal
│
├── data/
│   └── machine_data.csv               # Données capteurs (voir lien ci-dessous)
│
├── slides/
│   └── Predictive_Maintenance.pdf     # Slides de présentation (30 pages)
│
├── outputs/
│   └── model_comparison.csv           # Tableau comparatif des 6 modèles
│
├── requirements.txt
└── README.md

📋 Données
CaractéristiqueValeurObservations124 494Colonnes12 (dont 9 capteurs)Capteursmetric1 à metric9Taux de pannes0,09 % (classes fortement déséquilibrées)PériodeDonnées journalières multi-machines

📥 Dataset similaire disponible sur Kaggle — Predictive Maintenance.


🔬 Étapes du Projet
1️⃣ Analyse Exploratoire (EDA)

Répartition des classes (échelle log) : 99,91 % normal / 0,09 % panne
Distribution des pannes par mois (pics en janvier et mai) et par jour de la semaine (lundi)
Matrice de corrélation des métriques → metric7 et metric8 fortement corrélées

2️⃣ Nettoyage des Données

Doublons : 1 supprimé
Valeurs manquantes : aucune détectée
Corrélation : metric8 dupliquée de metric7 → retirée
Outliers : conservés — signal potentiel de panne

3️⃣ Feature Engineering Temporel
À partir des 8 métriques conservées, construction de variables dérivées :
FeatureDescriptionIntérêtLags (J-1, J-7)Valeur décalée d'1 et 7 joursComparaison avec l'historique récentMoyennes mobiles (7j)Moyenne glissante sur 7 joursIsolation des dérives de fondVolatilité (std 7j)Écart-type glissant sur 7 joursDétection d'instabilité croissanteVariations (diff)Différence jour J vs J-1Mesure de l'accélération des changements

💡 Analyse pre-panne : jusqu'à J-15 la machine est stable, dérive entre J-15 et J-5, effondrement brutal dans les 5 derniers jours.

4️⃣ Préparation pour la Modélisation

Tri chronologique pour respecter la structure temporelle
Split chronologique 80/20 : train = données les plus anciennes, test = données les plus récentes
Suppression des variables date et device des features
Gestion du déséquilibre : scale_pos_weight (XGBoost) et SMOTE (LR, RF)

5️⃣ Modélisation — 3 Algorithmes × 2 Seuils = 6 Modèles

Validation croisée : TimeSeriesSplit (respect de la structure temporelle)
Métrique d'évaluation : PR-AUC
Optimisation du seuil : F1-Score vs F2-Score (favorise le rappel)


📊 Résultats
Tableau Comparatif Final
ModèleSeuilPrécision (panne)Recall (panne)F1 (panne)F2 (panne)PR-AUCROC-AUCXGB_F2 ⭐0.7400.0950.1900.1270.1590.0370.763XGB_F10.5000.0320.1900.0550.0960.0370.763RF_F20.6570.2500.1430.1820.1560.0330.837RF_F10.6710.1670.0950.1210.1040.0320.839LogReg_F10.9990.0460.3330.0810.1480.0150.679LogReg_F20.9990.0460.3330.0810.1480.0150.679
🏆 Modèle Retenu : XGB_F2
Critères de sélection : PR-AUC (priorité) → F2-Score → Recall → Précision
Matrice de confusion (XGB_F1 sur test) :
Prédit : NormalPrédit : PanneRéel : Normal23 207 ✅121 ❌Réel : Panne17 ❌4 ✅
Variables les Plus Influentes (SHAP)
Top 3 : num__day_of_week · num__month · num__metric4 — les variables temporelles et les statistiques glissantes de metric4 et metric7 dominent la prédiction.

🛠️ Stack Technique
CoucheOutilsMLXGBoost · RandomForest · LogisticRegression (Scikit-learn)DéséquilibreSMOTE (imbalanced-learn) · scale_pos_weightValidationTimeSeriesSplit · PR-AUC · F1/F2-ScoreExplicabilitéSHAP (summary plot + waterfall faux négatifs)Feature EngineeringPandas (rolling, shift, diff)VisualisationMatplotlib · Seaborn

🚀 Lancement
bashpip install -r requirements.txt
jupyter notebook notebook/predictive_maintenance.ipynb

📦 Dépendances
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
shap
matplotlib
seaborn
jupyter

💡 Compétences Démontrées

Feature engineering temporel : lags, moyennes mobiles, volatilité glissante sur séries chronologiques industrielles
Gestion du déséquilibre extrême : SMOTE, scale_pos_weight, optimisation du seuil de décision
Validation rigoureuse sur séries temporelles : TimeSeriesSplit (pas de data leakage)
Comparaison multi-modèles : 6 configurations évaluées sur PR-AUC, F1, F2, ROC-AUC
Explicabilité ML : analyse SHAP globale (summary plot) et locale (waterfall sur faux négatifs)
Raisonnement métier : arbitrage coût faux négatif vs faux positif, recommandations actionnables


📌 Recommandations Métier

Utiliser le score prédit comme indicateur de risque de maintenance
Ajuster le seuil d'alerte selon le coût relatif des faux négatifs (panne non détectée) vs faux positifs (alerte inutile)
Surveiller les variables SHAP prioritaires : day_of_week, month, metric4, metric7
Prioriser les inspections des équipements à score de risque élevé
Réentraîner le modèle périodiquement selon l'évolution des équipements


⚠️ Limites du Projet

Résultats dépendants de la qualité et du volume de données de pannes (21 pannes sur le test)
Le comportement des machines peut évoluer → nécessité de monitoring et recalibration
Une mise en production nécessiterait un pipeline de scoring en temps réel et un système d'alertes


📌 Pistes d'Amélioration

 Tester LSTM ou autres modèles séquentiels pour mieux capturer les dynamiques temporelles
 Ajouter une fenêtre de pré-alerte (prédire la panne J-3 avant qu'elle survienne)
 Déployer le modèle via une API (FastAPI + MLflow)
 Intégrer un tableau de bord de monitoring temps réel (Streamlit / Grafana)
