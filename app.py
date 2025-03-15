import streamlit as st
import numpy as np
import pickle

# Charger le modèle et le scaler
with open("/Users/a/Desktop/house price predicitons/house_price_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("/Users/a/Desktop/house price predicitons/scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Vérification du type du modèle et du scaler
print(type(model))  # Devrait afficher : <class 'sklearn.linear_model._base.LinearRegression'>
print(type(scaler))  # Devrait afficher : <class 'sklearn.preprocessing._scaler.StandardScaler'>

# Interface Streamlit
st.title("Prédiction des Prix des Maisons")

# Collecte des caractéristiques
sqft_living = st.number_input("Surface habitable (sqft)", min_value=300, max_value=10000, value=2000)
sqft_lot = st.number_input("Surface totale (sqft)", min_value=500, max_value=50000, value=5000)
sqft_above = st.number_input("Surface au-dessus du sol (sqft)", min_value=300, max_value=10000, value=1500)
sqft_basement = st.number_input("Surface sous-sol (sqft)", min_value=0, max_value=5000, value=500)
bedrooms = st.slider("Nombre de chambres", 1, 10, 3)
bathrooms = st.slider("Nombre de salles de bain", 1, 5, 2)
floors = st.slider("Nombre d'étages", 1, 3, 1)
condition = st.slider("Condition de la maison", 1, 5, 3)

# Fonction pour prédire le prix
def predict_price(features):
    # Convertir les caractéristiques en tableau numpy
    features_array = np.array(features).reshape(1, -1)
    
    # Vérifier que le scaler est bien un StandardScaler
    if not isinstance(scaler, pickle.Unpickler):
        print("Le scaler n'est pas du bon type !")
    
    # Normalisation des caractéristiques
    features_scaled = scaler.transform(features_array)
    
    # Vérifier que le modèle est bien un modèle de régression
    if hasattr(model, 'predict'):
        print("Le modèle est correct.")
    else:
        print("Le modèle n'est pas valide.")
    
    # Prédiction avec le modèle
    price = model.predict(features_scaled)[0]
    return price

# Bouton de prédiction
if st.button("Prédire le prix de la maison"):
    # Collecte des caractéristiques sous forme de liste
    features = [sqft_living, sqft_lot, sqft_above, sqft_basement, bedrooms, bathrooms, floors, condition]
    
    # Effectuer la prédiction
    predicted_price = predict_price(features)
    
    # Afficher le résultat
    st.success(f"🏡 Le prix estimé de la maison est : **{predicted_price:,.2f} $**")
