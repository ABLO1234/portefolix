import streamlit as st
import pandas as pd
import numpy as np
import scipy.optimize as sco
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import smtplib
from email.message import EmailMessage
import re # Importation de re au niveau global
from bs4 import BeautifulSoup
import os
import warnings
# Pour le modèle ARIMA




# --- Fonctions de modélisation financière ---

# Fonction pour récupérer les tickers du CAC 40 depuis Wikipedia
@st.cache_data(ttl=86400) # Cache les données pendant 24 heures (un jour)
def tickers_taker():
    url = "https://en.wikipedia.org/wiki/CAC_40"
    try:
        tables = pd.read_html(url)
    except Exception as e:
        st.error(f"Erreur lors de la lecture de la page Wikipedia : {e}")
        raise ValueError("Impossible de récupérer la liste des tickers du CAC 40. Vérifiez votre connexion Internet ou l'URL.")

    df = None
    for table in tables:
        # Cherche une table qui contient à la fois "Ticker" ou "Ticker symbol" et "Company" ou "Company Name"
        # et idéalement "Sector"
        if ("Ticker" in table.columns or "Ticker symbol" in table.columns) and \
           ("Company" in table.columns or "Company Name" in table.columns):
            df = table.copy()
            break # Trouvé la table, on sort de la boucle

    if df is None:
        raise ValueError("Impossible de trouver une table pertinente contenant les tickers du CAC 40 sur la page Wikipedia. Le format de la page a peut-être changé.")

    # Harmonisation des noms de colonnes
    if "Ticker symbol" in df.columns:
        df.rename(columns={"Ticker symbol": "Ticker"}, inplace=True)
    if "Company Name" in df.columns:
        df.rename(columns={"Company Name": "Company"}, inplace=True)

    # Filtrer pour s'assurer que les colonnes essentielles existent
    expected_columns = ["Ticker", "Company"]
    if "Sector" in df.columns:
        expected_columns.append("Sector")
    if "Link" in df.columns: # Si la colonne Link existe, on la garde aussi
        expected_columns.append("Link")

    if not all(col in df.columns for col in expected_columns if col != "Link"):
        st.warning("La table trouvée ne contient pas toutes les colonnes attendues (Ticker, Company). Tentative de continuer avec les colonnes disponibles.")
        # Filtrer seulement les colonnes qui existent réellement
        df = df[[col for col in expected_columns if col in df.columns]]


    # Nettoyage des tickers: Ajout de .PA pour les tickers français si absent
    # Ceci est une heuristique et pourrait nécessiter un ajustement selon les données de yfinance
    # On évite d'ajouter .PA si le ticker contient déjà un suffixe boursier (ex: .AS, .MI, .DE, .L)
    suffix_pattern = r'\.(PA|AS|MI|DE|L|LS|BR|VX|CO|ST|HE|MC|TL|VI|IR|PR|VI|IS)$'
    df['Ticker'] = df['Ticker'].apply(lambda x: x.upper().strip() + ".PA" if not pd.isna(x) and not re.search(suffix_pattern, x.upper().strip()) else x.upper().strip())

    return df[expected_columns].dropna(subset=['Ticker'])


# Fonction pour récupérer les données historiques des tickers
# Retourne un DataFrame aplati des prix de clôture ajustés
@st.cache_data(ttl=3600) # Cache les données pendant 1 heure
def get_historical_data(tickers, start_date, end_date):
    data = pd.DataFrame()
    failed_tickers_info = {} # Dictionnaire pour stocker les tickers échoués et la raison
    
    if not tickers:
        st.warning("Aucun ticker fourni pour le téléchargement de données.")
        return None

    for ticker in tickers:
        try:
            # Télécharge seulement 'Adj Close' pour simplifier la structure
            ticker_data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if not ticker_data.empty:
                if 'Close' in ticker_data.columns:
                    data[ticker] = ticker_data['Close']
                else:
                    failed_tickers_info[ticker] = "Missing 'Close' column"
            else:
                failed_tickers_info[ticker] = "Empty data returned by yfinance"
        except Exception as e:
            failed_tickers_info[ticker] = f"Error during download: {e}"
            
    if data.empty:
        error_messages = [f"{t} ({reason})" for t, reason in failed_tickers_info.items()]
        st.error(f"Aucune donnée historique n'a pu être téléchargée pour les tickers suivants : {', '.join(error_messages)}. Veuillez vérifier les symboles ou raccourcir la période sélectionnée.")
        return None

    # Enlève les colonnes de tickers qui ont échoué
    for ticker in failed_tickers_info:
        if ticker in data.columns:
            data.drop(columns=[ticker], inplace=True)

    if failed_tickers_info:
        warning_messages = [f"{t} ({reason})" for t, reason in failed_tickers_info.items()]
        st.warning(f"Impossible de télécharger les données pour les tickers suivants : {', '.join(warning_messages)}. Ils seront ignorés dans l'analyse.")

    return data.dropna() # Supprime les lignes avec des NaN après le merge des tickers


# Fonction pour calculer les statistiques du portefeuille
@st.cache_data(ttl=3600)
def portfolio_stats(weights, mean_returns, cov_matrix, risk_free_rate):
    # Assurez-vous que les poids, mean_returns et cov_matrix sont des numpy arrays ou pandas Series/DataFrames
    # pour éviter les erreurs de dimension
    if isinstance(weights, pd.Series):
        weights = weights.values
    if isinstance(mean_returns, pd.Series):
        mean_returns = mean_returns.values
    if isinstance(cov_matrix, pd.DataFrame):
        cov_matrix = cov_matrix.values

    # Vérifiez la dimension des poids
    if weights.ndim > 1:
        weights = weights.flatten()
    
    # Vérifiez que les dimensions correspondent
    if len(weights) != len(mean_returns):
        st.error("Mismatch between weights and mean_returns dimensions in portfolio_stats. This indicates an internal error in data processing.")
        return 0, 0, 0 # Retourne des valeurs nulles en cas d'erreur de dimension

    try:
        returns = np.sum(mean_returns * weights) * 252 # Annualisé
        std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252) # Annualisé
        sharpe_ratio = (returns - risk_free_rate) / std_dev
        return returns, std_dev, sharpe_ratio
    except Exception as e:
        st.error(f"Erreur lors du calcul des statistiques du portefeuille : {e}. Cela peut être dû à des données non numériques ou à des dimensions incorrectes.")
        return 0, 0, 0 # Retourne des valeurs nulles en cas d'erreur


# Fonction pour minimiser la volatilité du portefeuille (objectif pour l'optimisation)
@st.cache_data(ttl=3600)
def minimize_volatility(weights, mean_returns, cov_matrix, risk_free_rate):
    return portfolio_stats(weights, mean_returns, cov_matrix, risk_free_rate)[1]


# Fonction pour optimiser le portefeuille
@st.cache_data(ttl=3600)
def optimize_portfolio(tickers, start_date, end_date, risk_free_rate):
    data = get_historical_data(tickers, start_date, end_date)

    if data is None or data.empty:
        st.error("L'optimisation ne peut pas être effectuée car aucune donnée valide n'a été récupérée pour les tickers sélectionnés. Référez-vous aux messages d'erreur de collecte de données.")
        return None, None, None, None, None

    # Calcule les rendements logarithmiques pour l'optimisation
    log_returns = np.log(data / data.shift(1)).dropna()

    if log_returns.empty:
        st.error("Pas assez de données pour calculer les rendements ou la matrice de covariance après la suppression des valeurs manquantes. Veuillez ajuster la période ou les actifs (ex: raccourcir la période).")
        return None, None, None, None, None

    mean_returns = log_returns.mean()
    cov_matrix = log_returns.cov()

    num_assets = len(mean_returns)
    if num_assets == 0:
        st.error("Aucun actif avec des données valides pour l'optimisation après le calcul des rendements.")
        return None, None, None, None, None
    elif num_assets == 1:
        # Si un seul actif, le poids optimal est 1
        optimal_weights = np.array([1.0])
        optimal_weights_df = pd.DataFrame({
            'Ticker': mean_returns.index,
            'Optimal Weight': np.round(optimal_weights, 5)
        })
        rendement, volatilite, sharpe = portfolio_stats(optimal_weights, mean_returns, cov_matrix, risk_free_rate)
        return optimal_weights_df, rendement, volatilite, sharpe, data


    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) # La somme des poids doit être 1
    bounds = tuple((0, 1) for _ in range(num_assets)) # Les poids doivent être entre 0 et 1

    initial_weights = np.array([1. / num_assets] * num_assets)

    try:
        optimal = sco.minimize(
            minimize_volatility,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            args=args
        )
    except Exception as e:
        st.error(f"Erreur lors de l'exécution de l'optimisation (scipy.optimize) : {e}. Cela peut être dû à des données insuffisantes, à des problèmes de convergence de l'algorithme ou à des valeurs non valides dans les rendements/covariance.")
        return None, None, None, None, data


    if not optimal.success:
        st.error(f"L'optimisation a échoué : {optimal.message}. Il se peut que le problème ne puisse pas être résolu avec les contraintes données ou que les données soient problématiques (ex: tous les rendements sont identiques).")
        return None, None, None, None, data

    new_weights = optimal.x

    # Correspondance ticker -> nom de société pour un affichage plus convivial
    tickers_map = {}
    try:
        tickers_info = tickers_taker()
        tickers_map = dict(zip(tickers_info['Ticker'], tickers_info['Company']))
    except ValueError: # Si tickers_taker échoue, on continue sans les noms de société
        st.warning("Impossible de récupérer les noms de société pour les tickers. Affichage des tickers bruts.")

    weights_df = pd.DataFrame({
        'Ticker': mean_returns.index,
        'Optimal Weight': np.round(new_weights, 5)
    })
    weights_df['Company'] = weights_df['Ticker'].map(tickers_map).fillna(weights_df['Ticker']) # Utilise le ticker si pas de nom de société

    rendement, volatilite, sharpe = portfolio_stats(new_weights, mean_returns, cov_matrix, risk_free_rate)

    return weights_df, rendement, volatilite, sharpe, data


@st.cache_data(ttl=3600)
def calcul_performance(ticker, start_date, end_date):
    data = get_historical_data([ticker], start_date, end_date) # Utilisez get_historical_data pour un seul ticker

    if data is None or data.empty or ticker not in data.columns:
        # Erreur déjà gérée dans get_historical_data, juste retourner None ici
        return None

    returns = data[ticker].pct_change().dropna()

    if returns.empty:
        st.warning(f"Pas assez de données pour calculer le rendement pour {ticker} sur la période donnée. Les rendements sont peut-être tous NaN.")
        return None

    mean_return = returns.mean() 
    mean_return_an = mean_return * 252  # Annualisé
    return float(np.round(mean_return, 3)), float(np.round(mean_return_an, 3))

@st.cache_data(ttl=3600)
def monte_carlo_simulation(start_price, mu, sigma, n_days=252, n_simulations=100, actif ="Actif"):
    simulations = np.zeros((n_days, n_simulations))
    for i in range(n_simulations):
        prices = [start_price]
        for _ in range(1, n_days):
            shock = np.random.normal(loc=mu/252, scale=sigma/np.sqrt(252))
            price = prices[-1] * np.exp(shock)
            prices.append(price)
        simulations[:, i] = prices

    fig = go.Figure()
    for i in range(n_simulations):
        fig.add_trace(go.Scatter(y=simulations[:, i], mode='lines', line=dict(width=1), opacity=0.3))
    fig.update_layout(title=f"Simulation Monte Carlo de {actif}", xaxis_title="Jours", yaxis_title="Prix")
    return fig

# Predictio
@st.cache_data(ttl=3600)
def predict_with_arima(ticker, start_date_str, end_date_str, forecast_days=7, order=(5,1,0)):
    from statsmodels.tsa.arima.model import ARIMA
    data = get_historical_data([ticker], start_date_str, end_date_str)

    if data is None or data.empty or ticker not in data.columns:
        return {'success': False, 'message': f"Impossible de récupérer les données historiques pour {ticker}. Vérifiez le ticker et la période."}

    series = data[ticker].dropna()

    if len(series) < sum(order) + 1:
        return {'success': False, 'message': f"Pas assez de données historiques pour entraîner le modèle ARIMA pour {ticker} avec l'ordre {order}. Minimum de {sum(order) + 1} jours requis."}
    
    try:
        model = ARIMA(series, order=order)
        model_fit = model.fit()
    except Exception as e:
        return {'success': False, 'message': f"Erreur lors de l'entraînement du modèle ARIMA pour {ticker}: {e}. Essayez de modifier l'ordre (p,d,q) ou la période de données."}

    last_date = series.index[-1]
    forecast_dates_candidate = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days * 2, freq='D')
    forecast_index = forecast_dates_candidate[forecast_dates_candidate.dayofweek < 5][:forecast_days]

    try:
        forecast_result = model_fit.get_forecast(steps=forecast_days)
        forecast_mean = forecast_result.predicted_mean
    except Exception as e:
        return {'success': False, 'message': f"Erreur lors de la génération de la prévision ARIMA pour {ticker}: {e}. Le modèle peut être instable."}

    if len(forecast_mean) != len(forecast_index):
        st.warning(f"ARIMA a prédit {len(forecast_mean)} étapes au lieu de {forecast_days}. Ajustement de l'index des prévisions.")
        forecast_index = forecast_index[:len(forecast_mean)]

    forecast_mean.index = forecast_index
    
    historical_trace = go.Scatter(
        x=series.index,
        y=series.tolist(),
        mode='lines',
        name='Historique'
    )

    forecast_trace = go.Scatter(
        x=forecast_mean.index,
        y=forecast_mean.tolist(),
        mode='lines',
        name=f'Prévision ARIMA ({forecast_days} jours)',
        line=dict(dash='dash', color='red')
    )
    
    plot_data = [historical_trace, forecast_trace]

    fig = go.Figure(data=plot_data)
    fig.update_layout(
        title=f"Prédiction du prix de {ticker} avec ARIMA {order}",
        xaxis_title="Date",
        yaxis_title="Prix de clôture",
        hovermode='x unified'
    )

    return {
        'success': True,
        'plot': fig,
        'forecast_days': forecast_days,
        'future_prices': list(zip([d.strftime('%Y-%m-%d') for d in forecast_mean.index], np.round(forecast_mean.tolist(), 2).tolist())),
        'message': f"Prévision générée pour {ticker} sur {forecast_days} jours avec ARIMA {order}."
    }


# Fonction pour l'envoi d'e-mail
# ATTENTION: NE JAMAIS CODER EN DUR VOTRE MOT DE PASSE EN PRODUCTION.
# Utilisez st.secrets pour une gestion sécurisée des identifiants.
# Ex: votre_mdp = st.secrets["EMAIL_PASSWORD"]
@st.cache_data(ttl=3600)
def send_mail(mail, weights_df, rendement, volatilite, sharpe):
    weights_html = weights_df.to_html(index=False, border=1, justify='center')
    rendement_percent = rendement * 100
    volatilite_percent = volatilite * 100
    sharpe_rounded = round(sharpe, 2)

    msg = EmailMessage()
    msg['Subject'] = "Nouveau portefeuille optimal !"
    msg['From'] = 'styftang1@gmail.com' # Remplacez par votre adresse d'envoi
    msg['To'] = mail

    msg.set_content(
        f"Bonjour,\n\n"
        f"Voici le nouveau portefeuille optimal calculé :\n"
        f"Rendement annuel attendu : {rendement_percent:.2f}%\n"
        f"Volatilité annuelle : {volatilite_percent:.2f}%\n"
        f"Ratio de Sharpe : {sharpe_rounded:.2f}\n\n"
        f"Poids des actifs :\n{weights_df.to_string(index=False)}\n\n" # Utilise to_string pour la version texte
        "Cordialement,\nVotre application de portefeuille"
    )

    msg.add_alternative(f"""
        <html>
            <body>
                <p>Bonjour,</p>
                <p>Voici le nouveau portefeuille optimal calculé :</p>
                {weights_html}
                <p><strong>Rendement annuel attendu :</strong> {rendement_percent:.2f}%</p>
                <p><strong>Volatilité annuelle :</strong> {volatilite_percent:.2f}%</p>
                <p><strong>Ratio de Sharpe :</strong> {sharpe_rounded:.2f}</p>
                <p>Cordialement,<br>Votre application de portefeuille</p>
            </body>
        </html>
    """, subtype='html')

    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        # Assurez-vous d'utiliser st.secrets pour le mot de passe en production
        # Pour le test local, vous pouvez utiliser la variable ci-dessous, mais ce n'est PAS SÉCURISÉ.
        #password = "72228699Gmail" # Votre mot de passe de l'application Gmail ou mot de passe de compte
        password = st.secrets.get("72228699Gmail") # PRÉFÉRÉ: Récupérer depuis les secrets de Streamlit

        if not password:
            st.error("Le mot de passe de l'e-mail n'est pas configuré. Veuillez utiliser st.secrets.")
            raise ValueError("Mot de passe email non configuré.")

        server.ehlo()
        server.login('abdoulayetangara722@gmail.com', password) 
        if server.send_message(msg) :
            st.info("L'e-mail a été envoyé avec succès.")
            server.quit()
        else:
            st.error("L'envoi de l'e-mail a échoué. Veuillez vérifier les paramètres du serveur SMTP ou l'adresse e-mail.")
    except Exception as e:
        st.error(f"Erreur lors de l'envoi de l'e-mail. Vérifiez l'adresse, le mot de passe (si vous utilisez GMail, activez l'authentification à deux facteurs et générez un mot de passe d'application) ou les paramètres du serveur SMTP : {e}")





# ===================== 🔐 AUTHENTIFICATION =====================
USERS = {
    "admin": "ablo",
    "analyste": "portefeuille2025",
    "mahamane" : "korobara",
    "salimata" : "jolie",
    "kader" : "bassoro",
    "haoussa" : "korobara",
    "garba" : "garbiss"
}

import base64
from io import BytesIO

def image_to_base64(img):
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return img_str

def login_interface():
    from PIL import Image

    image = Image.open("style/logo.png")
    st.markdown(
        """
        <div style="text-align: center;">
            <img src="data:image/png;base64,%s" width="350"/>
            <p style="font-style: italic; color: rgba(0, 0, 0, 0.6);">
            Vos actifs boursiers sont à portée de main !<br>
            Gérez votre portefeuille d'investissement avec PortfoliX, l'application qui optimise vos choix financiers.<br>
        </p>
        </div>
        """ % (image_to_base64(image)),
        unsafe_allow_html=True
    )

    st.sidebar.markdown("<h2 style='text-align: left; color: #173d1d;'>🔐 Connexion requise</h2>", unsafe_allow_html=True)
    username = st.sidebar.text_input("Nom d'utilisateur")
    password = st.sidebar.text_input("Mot de passe", type="password")

    if st.sidebar.button("Se connecter"):
        if USERS.get(username) == password:
            st.session_state["authenticated"] = True
            st.session_state["user"] = username
            st.success(f"Bienvenue {username} !")
            #st.experimental_rerun()
        else:
            st.error("Identifiants incorrects.")

    st.sidebar.markdown(
        """
        <p style='text-align: left; font-size: 12px; color: gray;'>
            © 2025 Abdoulaye Tangara. Tous droits réservés.
        </p>
        """,
        unsafe_allow_html=True
    )

# --- Application Streamlit ---

def main():
    st.set_page_config(layout="wide", page_title="PortfoliX", page_icon = "style/blazon.png")
   
    
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        login_interface()      
        return
  

    import requests
    from bs4 import BeautifulSoup
    from datetime import datetime
    import urllib.parse

    def get_temperature(city):
        city_query = urllib.parse.quote_plus(f"météo {city}")
        url = f"https://www.google.com/search?q={city_query}"
        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        try:
            response = requests.get(url, headers=headers, timeout=5)
            soup = BeautifulSoup(response.text, "html.parser")
            temp = soup.find("span", attrs={"id": "wob_tm"})
            if temp:
                return temp.text.strip()
            else:
                return "N/A"
        except Exception as e:
            return "Erreur"

    # 🔄 Appel dynamique
    temperature = get_temperature("Paris")

    # 💡 Barre d’en-tête stylisée
    st.markdown(f"""
    <div id="header-bar" style="background-color:#004080; color:white; padding:10px; border-radius:6px; display:flex; justify-content:space-between; font-size:0.9rem;">
        <div class="title">Gestion de portefeuille d'investissement</div>
        <div class="datetime">{datetime.now().strftime('%A %d %B %Y - %H:%M:%S')}</div>
        <div class="weather">🌤️ Bamako | {temperature}°C</div>
    </div>
    """, unsafe_allow_html=True)




    with open("style/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    #st.title("Gestion de portefeuille d'investissement")
    st.write("Cette application permet d'optimiser un portefeuille d'investissement basé sur les entreprises du CAC 40.")
    st.markdown("---")

    # === SECTION TICKERS CAC 40 ===
    st.sidebar.header("Les entreprises cotées du CAC 40")
    tickers_df = pd.DataFrame()
    try:
        tickers_df = tickers_taker()
        st.sidebar.dataframe(tickers_df, use_container_width=True)
    except ValueError as e:
        st.sidebar.error(str(e))
        st.stop() # Arrête l'exécution si les tickers ne peuvent pas être récupérés
    st.sidebar.markdown("---")

    # === SECTION TENDANCE HISTORIQUE INDIVIDUELLE ===
    st.subheader("Tendance historique d’un actif")
    selected_indiv = st.selectbox(
        "Sélectionnez un actif pour voir son historique",
        options=tickers_df['Ticker'].tolist(),
        index=0
    )


    if selected_indiv:

        # Récupère les données historiques pour le ticker sélectionné
        df_indiv_data = get_historical_data([selected_indiv], datetime.now() - timedelta(days=365 * 5), datetime.now())

        if df_indiv_data is not None and selected_indiv in df_indiv_data.columns and not df_indiv_data[selected_indiv].empty:
            df_plot = df_indiv_data[selected_indiv].dropna()

            annuel, rendement = calcul_performance(
                    selected_indiv,
                    datetime.now() - timedelta(days=365 * 5),
                    datetime.now()
                )
            
            if rendement is not None:

                if rendement > 0:
                    st.info(f"**{selected_indiv}** a un rendement annuel de **{annuel:.2%}**.")
                    st.success(f"**{selected_indiv}** a un rendement journalier positif de **{rendement:.2%}**.")
                else:
                    st.info(f"**{selected_indiv}** a un rendement annuel de **{annuel:.2%}**.")
                    st.warning(f"**{selected_indiv}** a un rendement journalier négatif de **{rendement:.2%}**.")

            col1, col2 = st.columns(2)
            with col1:
                
                fig = go.Figure(data=[go.Scatter(x=df_plot.index, y=df_plot.values, mode='lines', name=selected_indiv)])
                fig.update_layout(title=f"Cours de clôture ajusté de {selected_indiv}", xaxis_title="Date", yaxis_title="Prix")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                
                fig_candlestick = go.Figure(data=[go.Candlestick(
                    x=df_indiv_data.index,
                    open=df_indiv_data[selected_indiv].values,
                    high=df_indiv_data[selected_indiv].values,
                    low=df_indiv_data[selected_indiv].values,
                    close=df_indiv_data[selected_indiv].values
                )])
                fig_candlestick.update_layout(title=f"Graphique en chandelier de {selected_indiv}", xaxis_title="Date", yaxis_title="Prix")
                st.plotly_chart(fig_candlestick, use_container_width=True,axxis_visible=True)
        
                        # Simulation Monte Carlo basée sur cet actif
            mu = annuel
            sigma = df_plot.pct_change().std() * np.sqrt(252)
            fig_mc = monte_carlo_simulation(df_plot.iloc[-1], mu, sigma, actif=selected_indiv)
            st.plotly_chart(fig_mc, use_container_width=True)

        else:
            st.error(f"Aucune donnée disponible pour l'actif sélectionné : {selected_indiv}. Il se peut que le symbole ne soit pas valide ou que la période ne contienne pas de données. Vérifiez les messages d'avertissement ci-dessus.")
         # --- Nouvelle section pour ARIMA avec les sliders en deux colonnes dans la sidebar ---
        st.sidebar.header(f"Prévision de prix pour {selected_indiv} (Modèle ARIMA)")
            
            # Utilisation de st.columns pour organiser les sliders dans la sidebar
        col_arima_1, col_arima_2 = st.sidebar.columns(2)

        with col_arima_1:
            forecast_days_arima = st.slider("Nombre de jours de prévision", 7, 60, 30, key="forecast_days_arima")
            
        with col_arima_2:
            p_arima = st.slider("Ordre p (AR)", 0, 10, 5, key="p_arima")
            d_arima = st.slider("Ordre d (I)", 0, 2, 1, key="d_arima")
            q_arima = st.slider("Ordre q (MA)", 0, 10, 0, key="q_arima")
            
        arima_order = (p_arima, d_arima, q_arima)

        if st.sidebar.button(f"Générer la prévision ARIMA"):
            with st.spinner("Génération de la prévision ARIMA..."):
                arima_result = predict_with_arima(
                        selected_indiv,
                        (datetime.now() - timedelta(days=365 * 2)).strftime('%Y-%m-%d'),
                        datetime.now().strftime('%Y-%m-%d'),
                        forecast_days=forecast_days_arima,
                        order=arima_order
                    )

                if arima_result['success']:
                    st.subheader(f"Prévision de prix pour {selected_indiv} (Modèle ARIMA)")
                    st.plotly_chart(arima_result['plot'], use_container_width=True)
                    st.write(f"Prix futurs prévus pour {selected_indiv} :")
                    for date_str, price in arima_result['future_prices']:
                        st.write(f"- **{date_str}**: {price:.2f} €")
                else:
                    st.error(arima_result['message'])
        st.sidebar.markdown("---") # Séparateur après les options ARIMA


    # === SECTION OPTIMISATION DE PORTEFEUILLE ===
    st.sidebar.header("⚙️ Paramètres du portefeuille")
    # Utiliser les tickers du CAC 40 comme valeurs par défaut pour l'optimisation
    # Assurez-vous que ces tickers sont bien les versions .PA/.AS de yfinance si applicable
    all_cac_tickers = tickers_df['Ticker'].tolist()
    # Sélectionne les 10 premiers tickers disponibles pour l'optimisation par défaut
    default_selected_tickers_for_opt = all_cac_tickers[:min(len(all_cac_tickers), 10)] 

    selected_tickers_for_opt = st.sidebar.multiselect(
        "Sélectionnez les symboles boursiers à inclure dans le portefeuille",
        options=all_cac_tickers,
        default=default_selected_tickers_for_opt
    )

    end_date_opt = st.sidebar.date_input("Date de fin", datetime.now())
    start_date_opt = st.sidebar.date_input("Date de début des données pour l'optimisation", value=end_date_opt - timedelta(days=365 * 5))
    risk_free_rate_opt = st.sidebar.slider("Taux sans risque annuel (%)", 0.0, 10.0, 2.0, 0.1) / 100
    amount = st.sidebar.number_input("💰 Montant à investir (€)", min_value=0.0, value=100000.0, step=1000.0)

    if st.sidebar.button("✨ Optimiser le Portefeuille"):
        if not selected_tickers_for_opt:
            st.warning("⚠️ Veuillez sélectionner au moins un symbole boursier pour l'optimisation.")
            return

        with st.spinner("Optimisation en cours... Cela peut prendre un certain temps si beaucoup d'actifs sont sélectionnés."):
            weights_df, rendement, volatilite, sharpe, full_data_opt = optimize_portfolio(
                selected_tickers_for_opt, start_date_opt, end_date_opt, risk_free_rate_opt
            )

        if all(x is not None for x in [weights_df, rendement, volatilite, sharpe, full_data_opt]):
            st.subheader("Résultat de l'optimisation du portefeuille")
            st.dataframe(weights_df.set_index('Ticker'))

            st.write(f"**Rendement annuel attendu :** {rendement:.2%}")
            st.write(f"**Volatilité annuelle :** {volatilite:.2%}")
            st.write(f"**Ratio de Sharpe :** {sharpe:.2f}")

            st.markdown("---")
            st.subheader("💰 Distribution de l'investissement")
            total_invest = st.number_input("Somme à investir (€)", min_value=0.0, value=100000.0, step=1000.0)

            if total_invest > 0:
                dist = weights_df.copy()
                dist['Invested Amount (€)'] = np.round(dist['Optimal Weight'] * total_invest, 2)
                st.dataframe(dist.set_index('Ticker'))

                # Graphique en secteurs pour la distribution
                fig_pie = go.Figure(data=[go.Pie(
                    labels=dist['Company'], # Utilise le nom de la société pour les labels
                    values=dist['Invested Amount (€)'],
                    hole=0.3 # Pour un graphique en beignet
                )])
                fig_pie.update_layout(title_text="Répartition de l'investissement par actif")
                st.plotly_chart(fig_pie, use_container_width=True)

                # Monte Carlo pour la simulation de l'actif principal
                        # ================== Suivi des performances en temps réel ==================
            st.subheader("📈 Suivi en temps réel de l’investissement")

            latest_prices = {}
            current_prices = {}

            for ticker in dist['Ticker']:
                series = full_data_opt[ticker].dropna()
                if not series.empty:
                    latest_prices[ticker] = series.iloc[-1]

            today = datetime.now().date()
            try:
                yf_data = yf.download(dist['Ticker'].tolist(), start=today - timedelta(days=5), end=today + timedelta(days=1), progress=False)
                if 'Close' in yf_data.columns:
                    current_data = yf_data['Close']
                    if isinstance(current_data, pd.Series):
                        current_data = current_data.to_frame()
                    for ticker in dist['Ticker']:
                        if ticker in current_data.columns:
                            latest_value = current_data[ticker].dropna()
                            if not latest_value.empty:
                                current_prices[ticker] = latest_value.iloc[-1]
            except Exception as e:
                st.warning(f"Erreur lors de la récupération des prix actuels : {e}")

            gain_loss_data = []
            total_gain = 0.0
            for _, row in dist.iterrows():
                ticker = row['Ticker']
                amount_invested = row['Invested Amount (€)']
                old_price = latest_prices.get(ticker)
                new_price = current_prices.get(ticker)

                if old_price and new_price:
                    quantity = amount_invested / old_price
                    new_value = quantity * new_price
                    gain = new_value - amount_invested
                    gain_loss_data.append({
                        "Ticker": ticker,
                        "Ancien Prix (€)": round(old_price, 2),
                        "Prix Actuel (€)": round(new_price, 2),
                        "Gain/Perte (€)": round(gain, 2),
                        "Variation (%)": round((gain / amount_invested) * 100, 2)
                    })
                    total_gain += gain

            if gain_loss_data:
                gain_df = pd.DataFrame(gain_loss_data).set_index("Ticker")
                st.dataframe(gain_df)

                if total_gain >= 0:
                    st.success(f"📈 Gain actuel estimé : **{total_gain:.2f} €**")
                else:
                    st.error(f"📉 Perte actuelle estimée : **{total_gain:.2f} €**")
            else:
                st.warning("Impossible de récupérer les prix actuels pour les actifs sélectionnés.")

            st.markdown("---")
            st.subheader("📧 Envoyer le portefeuille par e-mail")
            email_to_send = st.text_input("Votre adresse e-mail")
            if st.button("Envoyer le portefeuille"):
                if email_to_send and re.match(r"[^@]+@[^@]+\.[^@]+", email_to_send):
                    send_portfolio_email(email_to_send, weights_df, rendement, volatilite, sharpe)
                else:
                    st.error("Veuillez entrer une adresse e-mail valide.")
        else:
            st.error("L'optimisation du portefeuille n'a pas pu être réalisée avec les paramètres fournis. Veuillez vérifier les messages d'erreur ci-dessus.")

                
            else:
                st.info("Veuillez entrer un montant à investir pour voir la distribution.")
        else:
            st.error("L'optimisation du portefeuille n'a pas pu aboutir. Veuillez vérifier les actifs sélectionnés ou la période de données. Des messages d'erreur spécifiques peuvent apparaître ci-dessus.")

        # === SECTION ENVOI D'EMAIL ===
        st.sidebar.header("📧 Envoi des résultats par e-mail")

        email = st.sidebar.text_input("Adresse e-mail", placeholder="ex: nom@exemple.com", key="email_input")

        # Créer une clé d’état pour le bouton s’il n’existe pas
        if "email_sent" not in st.session_state:
            st.session_state["email_sent"] = False


        if st.sidebar.button("📤 Envoyer les résultats par e-mail"):
            if email.strip():
                if 'weights_df' in locals() and weights_df is not None:
                    try:
                        if not st.session_state["email_sent"]:
                            send_mail(email, weights_df, rendement, volatilite, sharpe)
                            st.success("📧 Les résultats ont été envoyés par e-mail.")
                            st.session_state["email_sent"] = True
                        else:
                            st.info("📨 L'e-mail a déjà été envoyé dans cette session.")
                    except Exception as e:
                        st.error(f"❌ Erreur lors de l'envoi de l'e-mail : {e}")
                else:
                    st.warning("⚠️ Veuillez d'abord optimiser un portefeuille.")
            else:
                st.warning("⚠️ Veuillez entrer une adresse e-mail valide.")
        else :
            if st.session_state.get("email_sent", False):
                st.info("📨 Les résultats ont déjà été envoyés par e-mail dans cette session. Vous pouvez les consulter dans votre boîte de réception.")
            else:
                st.info("📧 Entrez une adresse e-mail pour recevoir les résultats de l'optimisation du portefeuille.")   

    st.sidebar.markdown("---")
    st.sidebar.info("© 2025 Abdoulaye Tangara. Tous droits réservés.")

    if st.sidebar.button("🔓 Se déconnecter"):
        st.session_state["authenticated"] = False
        #st.experimental_rerun()

if __name__ == "__main__":
    main()
