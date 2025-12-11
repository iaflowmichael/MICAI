import os
import streamlit as st
from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai_tools import SerperDevTool\nfrom crewai.tools import BaseTool
from crewai.tools import BaseTool
from dotenv import load_dotenv
from typing import Any, Type
from pydantic import BaseModel, Field

# ===================================================================================
# CONFIGURATION INITIALE ET SÉCURITÉ
# ===================================================================================

# Charger les variables d'environnement (pour GOOGLE_API_KEY)
load_dotenv()

# Vérifier la présence de la clé API
if "GOOGLE_API_KEY" not in os.environ:
    st.error("Erreur : La variable d'environnement GOOGLE_API_KEY n'est pas configurée.")
    st.stop()

# Configuration du LLM (Gemini)
llm = ChatGoogleGenerativeAI(model="gemini-pro",
                             verbose=True,
                             temperature=0.5,
                             google_api_key=os.getenv("GOOGLE_API_KEY"))

# ===================================================================================
# OUTILS ET BASES DE CONNAISSANCE
# ===================================================================================

# 1. Outil de Recherche Web (pour l'Expert Recherche)
# Nous utilisons SerperDevTool comme exemple. L'utilisateur devra configurer SERPER_API_KEY.
# Pour l'instant, nous le laissons désactivé pour ne pas bloquer le déploiement.
# search_tool = SerperDevTool()

# 2. Outil de Lecture de Méthodologie (RAG sur fichier local)
class MethodologieInput(BaseModel):
    """Inputs pour l'outil de lecture de méthodologie."""
    question: str = Field(description="La question spécifique à poser à la méthodologie pour extraire les directives.")

class MethodologieReader(BaseTool):
    name: str = "Expert_Methodologie_Reader"
    description: str = "Un outil pour lire et extraire des directives spécifiques du manuel de méthodologie interne (methodologie.txt)."
    args_schema: Type[BaseModel] = MethodologieInput

    def _run(self, question: str) -> str:
        try:
            with open("methodologie.txt", "r", encoding="utf-8") as f:
                methodologie_content = f.read()
        except FileNotFoundError:
            return "Erreur: Le fichier methodologie.txt est introuvable."

        # Utiliser le LLM pour extraire la réponse pertinente du contenu
        prompt = f"""
        En tant qu'Expert Méthodologie, tu dois répondre à la question suivante en te basant UNIQUEMENT sur le texte fourni ci-dessous.
        Ta réponse doit être une directive claire et concise pour les autres agents.

        Question: {question}

        ---
        TEXTE DE LA MÉTHODOLOGIE :
        {methodologie_content}
        ---
        """
        # Utiliser le LLM pour le raisonnement RAG
        response = llm.invoke(prompt)
        return response.content

methodologie_reader = MethodologieReader()

# ===================================================================================
# DÉFINITION DES AGENTS (LA CREW)
# ===================================================================================

# Le Chef d'Orchestre (Agent de Décision)
chef_orchestre = Agent(
    role='Chef d\'Orchestre et Stratège de Contenu',
    goal='Analyser la mission, déterminer si une recherche web est nécessaire, et orchestrer la création de contenu selon la méthodologie.',
    backstory=(
        "Je suis le cerveau de MICAI. Mon rôle est de garantir que chaque contenu est aligné avec la stratégie globale. "
        "Je suis le seul à décider si une information externe est requise. Je délègue ensuite aux experts."
    ),
    llm=llm,
    tools=[methodologie_reader], # Il a accès à la méthodologie pour la stratégie
    verbose=True,
    allow_delegation=True
)

# L'Expert Rédacteur (Agent d'Exécution)
redacteur = Agent(
    role='Expert Rédacteur et Styliste',
    goal='Rédiger le post final en respectant scrupuleusement le brief, le ton, et le format de la plateforme demandée.',
    backstory=(
        "Je suis la plume de MICAI. Je transforme les directives et les faits en un contenu engageant, aéré et optimisé pour la plateforme cible (LinkedIn, Facebook, etc.)."
    ),
    llm=llm,
    tools=[], # Pas d'outils, il se concentre sur l'écriture
    verbose=True
)

# L'Expert Recherche (Agent d'Information)
# Nous le définissons ici, mais il ne sera utilisé que si le Chef d'Orchestre le décide.
# Pour l'instant, il n'a pas d'outil de recherche actif pour éviter de bloquer le déploiement.
# Il servira de placeholder pour la logique de décision.
expert_recherche = Agent(
    role='Expert en Recherche et Vérification de Faits',
    goal='Trouver des faits, statistiques ou exemples récents pour enrichir le contenu, uniquement si le Chef d\'Orchestre le demande.',
    backstory=(
        "Je suis le détective de MICAI. Je garantis la véracité et la pertinence des informations externes. "
        "Je n'agis que sur ordre du Chef d'Orchestre."
    ),
    llm=llm,
    tools=[], # Temporairement sans outil de recherche actif
    verbose=True
)

# ===================================================================================
# LOGIQUE DE LA CREW (MODE AGENT AUTONOME)
# ===================================================================================

def creer_crew(mission: str, plateforme: str) -> Crew:
    """Crée et configure la CrewAI pour une mission spécifique."""

    # Tâche 1: Analyse de la Mission et Stratégie (par le Chef d'Orchestre)
    tache_strategie = Task(
        description=(
            f"Analyser la mission : '{mission}' pour la plateforme '{plateforme}'. "
            "Déterminer si une recherche web est nécessaire pour la véracité des faits ou l'actualité. "
            "Utiliser l'outil Expert_Methodologie_Reader pour extraire les directives de la méthodologie. "
            "Le résultat doit être un brief créatif détaillé, incluant la décision de recherche web (OUI/NON) et les directives de la méthodologie."
        ),
        expected_output="Un brief créatif structuré, incluant la stratégie, le ton, la structure (P-A-S ou AIDA), et la décision claire (OUI ou NON) concernant la nécessité d'une recherche web.",
        agent=chef_orchestre
    )

    # Tâche 2: Recherche d'Information (Conditionnelle - par l'Expert Recherche)
    # Cette tâche sera exécutée SEULEMENT si la Tâche 1 le demande.
    tache_recherche = Task(
        description=(
            "Si le brief créatif de la Tâche 1 indique OUI pour la recherche web, trouver 1 à 2 faits, statistiques ou exemples récents pertinents pour la mission. "
            "Si le brief indique NON, répondre simplement 'AUCUNE RECHERCHE NÉCESSAIRE'."
        ),
        expected_output="Une liste de faits/statistiques avec leurs sources, OU la phrase 'AUCUNE RECHERCHE NÉCESSAIRE'.",
        agent=expert_recherche
    )

    # Tâche 3: Rédaction Finale (par l'Expert Rédacteur)
    tache_redaction = Task(
        description=(
            f"Rédiger le post final pour la plateforme {plateforme} en utilisant le brief créatif de la Tâche 1 et les faits de la Tâche 2. "
            "Respecter la structure, le ton, et le format de la plateforme (sauts de ligne, emojis, etc.) comme spécifié dans la méthodologie."
        ),
        expected_output="Le texte complet du post, prêt à être publié, avec les hashtags appropriés.",
        agent=redacteur
    )

    # Création de la Crew
    crew = Crew(
        agents=[chef_orchestre, expert_recherche, redacteur],
        tasks=[tache_strategie, tache_recherche, tache_redaction],
        process=Process.sequential, # Les tâches s'exécutent dans l'ordre
        verbose=2 # Très important pour voir la réflexion de l'agent
    )
    return crew

# ===================================================================================
# INTERFACE STREAMLIT (LE FRONT-END)
# ===================================================================================

st.set_page_config(page_title="MICAI - Votre Double Numérique", layout="wide")

# Définition des couleurs pour le branding (utilisé dans le CSS)
BRAND_COLOR_START = "#9F7AEA" # Mauve
BRAND_COLOR_END = "#FF4D6D"   # Pourpre/Rose

# CSS pour le branding (bouton dégradé, etc.)
st.markdown(f"""
<style>
    /* Titre et icône */
    .st-emotion-cache-10trblm {{
        color: {BRAND_COLOR_START};
    }}
    /* Bouton Lancer la Mission */
    .stButton>button {{
        background: linear-gradient(to right, {BRAND_COLOR_START}, {BRAND_COLOR_END});
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        transition: 0.3s;
    }}
    .stButton>button:hover {{
        opacity: 0.9;
    }}
    /* Conteneur principal pour le mode nuit/jour */
    .main {{
        background-color: var(--background-color);
    }}
    /* Pour les messages de l'agent (réflexion) */
    .st-emotion-cache-1cypq8p {{
        background-color: #262730; /* Fond sombre pour la réflexion */
        border-left: 5px solid {BRAND_COLOR_START};
        padding: 10px;
        border-radius: 5px;
    }}
</style>
""", unsafe_allow_html=True)

st.title("MICAI 🤖 - Le Double Numérique")

# Initialisation de l'historique de la conversation
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage de l'historique
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ===================================================================================
# LOGIQUE DES DEUX MODES
# ===================================================================================

# Utilisation d'un conteneur pour la barre latérale (Mode Agent Autonome)
with st.sidebar:
    st.header("Mode Agent Autonome")
    st.markdown("---")
    
    plateforme = st.selectbox(
        "Plateforme Cible :",
        ("LinkedIn", "Facebook", "Instagram")
    )
    
    mission_autonome = st.text_area(
        "Décrivez la Mission (ex: 'Analyse le marché de l'IA et rédige un post LinkedIn')",
        height=150
    )
    
    # Le bouton qui lance le Mode Agent Autonome
    if st.button("Lancer la Mission Autonome"):
        if mission_autonome:
            st.session_state.messages.append({"role": "user", "content": f"MISSION AUTONOME LANCÉE pour {plateforme}: {mission_autonome}"})
            
            # Affichage de la réflexion en temps réel
            with st.spinner("MICAI réfléchit... (Voir le terminal pour la réflexion détaillée)"):
                
                # Création et exécution de la Crew
                crew = creer_crew(mission_autonome, plateforme)
                
                # Lancement de la Crew
                resultat_final = crew.kickoff(inputs={'mission': mission_autonome, 'plateforme': plateforme})
                
                # Affichage du résultat final
                st.session_state.messages.append({"role": "assistant", "content": f"**MISSION TERMINÉE !**\n\n{resultat_final}"})
                st.experimental_rerun()
        else:
            st.warning("Veuillez décrire la mission.")

# Le champ de saisie principal (Mode Chat)
if prompt := st.chat_input("Discutez avec MICAI (Mode Chat)..."):
    # Ajouter le message de l'utilisateur à l'historique
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Afficher le message de l'utilisateur
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Logique du Mode Chat (simple appel à Gemini)
    with st.chat_message("assistant"):
        with st.spinner("MICAI réfléchit..."):
            # Ici, on pourrait ajouter l'outil de recherche web pour le mode chat
            # Mais pour l'instant, on fait un simple appel pour la conversation
            response = llm.invoke(prompt)
            st.markdown(response.content)
            st.session_state.messages.append({"role": "assistant", "content": response.content})

# ===================================================================================
# NOTE IMPORTANTE POUR L'UTILISATEUR
# ===================================================================================
st.sidebar.markdown("---")
st.sidebar.info("⚠️ **ACTION REQUISE :** Pour que MICAI fonctionne, vous devez ajouter votre clé API Gemini dans les Secrets de votre dépôt GitHub. Voir les instructions de l'agent.")
