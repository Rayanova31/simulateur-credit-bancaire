import gradio as gr
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 1. Entraînement du modèle
url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/credit.csv"
df = pd.read_csv(url)
X = df[['age', 'amount', 'months_loan_duration']]
y = df['default']
modele = RandomForestClassifier(random_state=42)
modele.fit(X, y)

# 2. Création de la fonction métier (ce que fait le bouton)
def predire_risque(age, montant, duree):
    # On met en forme les données reçues de l'interface
    donnees_client = pd.DataFrame([[age, montant, duree]], columns=['age', 'amount', 'months_loan_duration'])

    # Le modèle prédit (0 = 1er élément de la liste des prédictions)
    prediction = modele.predict(donnees_client)[0]

    if prediction == 1:
        return "✅ Risque Faible : Profil validé. Crédit Accordé !"
    else:
        return "❌ Risque Élevé : Profil à risque. Crédit Refusé !"

# 3. Création de l'interface visuelle Gradio
profils_types = [
    [22, 12000, 48],  # Profil 1 : Jeune étudiant, gros prêt très long -> Risqué !
    [45, 2000, 12],   # Profil 2 : Cadre expérimenté, petit prêt court -> Sécurisé
    [32, 5000, 24]    # Profil 3 : Le client moyen standard
]
interface = gr.Interface(
    fn = predire_risque, # La fonction à appeler quand on clique
    inputs=[
        gr.Number(label="Âge du client", value=30),
        gr.Number(label="Montant demandé ($)", value=5000),
        gr.Number(label="Durée du prêt (mois)", value=24)
    ],
    outputs=gr.Textbox(label="Décision de l'Intelligence Artificielle"),
    title="💳 Simulateur d'Octroi de Crédit Bancaire",
    description="Ce simulateur utilise l'IA pour évaluer le risque d'un dossier client."
    examples=profils_types
)

# 4. Lancement avec création d'un lien public (share=True)
interface.launch(server_name="0.0.0.0", server_port=10000)