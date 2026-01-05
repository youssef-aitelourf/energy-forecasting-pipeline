# 🚀 Déploiement sur Streamlit Community Cloud (GRATUIT)

## 📋 Prérequis

- ✅ Compte GitHub (déjà fait)
- ✅ Projet poussé sur GitHub (déjà fait)
- ✅ Fichiers nécessaires présents :
  - `app.py` (application principale)
  - `requirements.txt` (dépendances)
  - `.streamlit/config.toml` (configuration)
  - `models/` (modèles entraînés)

---

## 🎯 Étapes de Déploiement

### 1️⃣ **Créer un Compte Streamlit Community Cloud**

1. Va sur [share.streamlit.io](https://share.streamlit.io)
2. Clique sur **"Sign up with GitHub"**
3. Autorise Streamlit à accéder à ton compte GitHub
4. C'est **100% GRATUIT** ! 🎉

---

### 2️⃣ **Pousser les Nouveaux Fichiers sur GitHub**

```bash
cd "/Users/youssefaitelourf/Desktop/Projets Perso/Pipeline End to End/energy-forecasting-pipeline"
git add .
git commit -m "Add Streamlit UI for energy forecasting"
git push origin main
```

---

### 3️⃣ **Déployer l'Application**

1. **Connexion** : Va sur [share.streamlit.io](https://share.streamlit.io) et connecte-toi

2. **Nouvelle App** : Clique sur **"New app"**

3. **Configuration** :
   ```
   Repository : youssef-aitelourf/energy-forecasting-pipeline
   Branch     : main
   Main file  : app.py
   ```

4. **App URL** : Choisis ton URL (par exemple: `energy-forecast-youssef`)
   - URL finale : `https://energy-forecast-youssef.streamlit.app`

5. **Deploy** : Clique sur **"Deploy!"**

6. **Attendre** : Le déploiement prend 2-3 minutes ⏱️

---

## ✅ Vérification

Une fois déployé, tu verras :

- ✅ **URL publique** : `https://[ton-nom].streamlit.app`
- ✅ **Logs en temps réel** : Pour déboguer si besoin
- ✅ **Redémarrage automatique** : À chaque push GitHub

---

## 🎨 Fonctionnalités de l'App

L'interface Streamlit inclut :

- 🎛️ **Sidebar avec inputs** : 28 paramètres d'entrée
  - Températures (8 pièces)
  - Humidité (8 pièces)
  - Conditions externes (température, pression, vent, etc.)
  - Informations temporelles

- 📊 **Visualisations interactives** :
  - Gauge chart de la prédiction
  - Bar chart des features importantes
  - Métriques en temps réel

- 💾 **Export des résultats** : Téléchargement CSV

- 🎨 **Design professionnel** : 
  - Thème personnalisé
  - Responsive
  - Emojis et icônes

---

## 🔧 Mise à Jour de l'App

Pour mettre à jour l'app après déploiement :

```bash
# Faire des modifications locales
git add .
git commit -m "Update app features"
git push origin main

# L'app se redéploie automatiquement ! 🚀
```

---

## 💡 Astuces

### **Limites Gratuites Streamlit Cloud**
- ✅ 1 app publique gratuite
- ✅ Ressources : 1 GB RAM, 1 CPU
- ✅ Sleep après 7 jours d'inactivité (se réveille automatiquement)
- ✅ Bande passante illimitée

### **Optimisation**
- Cache les données avec `@st.cache_resource` (déjà fait ✅)
- Les modèles sont chargés 1 seule fois
- Performance optimale

### **Monitoring**
- Dashboard Streamlit Cloud pour voir :
  - Logs en temps réel
  - Nombre de visiteurs
  - Temps de chargement
  - Erreurs

---

## 🌐 Partage

Une fois déployé, partage ton app :

- 📧 **Email** : youssefaitelourf@gmail.com
- 💼 **LinkedIn** : Ajoute le lien dans ton profil
- 📄 **CV** : Ajoute l'URL comme projet
- 🐙 **GitHub** : Le lien est automatiquement dans le footer

---

## 🆘 Dépannage

### Problème : "Module not found"
**Solution** : Vérifie que tous les packages sont dans `requirements.txt`

### Problème : "Model not found"
**Solution** : Assure-toi que le dossier `models/` est poussé sur GitHub

### Problème : App trop lente
**Solution** : 
- Utilise `@st.cache_resource` pour charger le modèle
- Évite les opérations lourdes dans la boucle principale

---

## 📞 Support

- 📚 **Documentation** : [docs.streamlit.io](https://docs.streamlit.io)
- 💬 **Forum** : [discuss.streamlit.io](https://discuss.streamlit.io)
- 🐛 **Issues** : [github.com/streamlit/streamlit](https://github.com/streamlit/streamlit)

---

## 🎉 Résultat Final

Tu auras une **app ML déployée gratuitement** avec :

- ✅ URL publique professionnelle
- ✅ Interface interactive et moderne
- ✅ Mise à jour automatique via GitHub
- ✅ Aucun coût
- ✅ Parfait pour portfolio !

**Bonne chance ! 🚀**
