# 🚢 Titanic Survival Predictor

A full-stack machine learning web application that predicts passenger survival probability based on historic demographics. This project features a custom-built, responsive Single Page Application (SPA) frontend paired with a production-ready cloud machine learning API.

---

## 📊 Dataset Information

The machine learning model is trained on the classic **Titanic - Machine Learning from Disaster** dataset from Kaggle. It analyzes demographic and traveling details from 891 passengers to identify correlation patterns linked to survival rates.

### Key Features Used & Preprocessed:
* **Pclass:** Ticket class (1st = Upper/Luxury, 2nd = Middle, 3rd = Lower)
* **Sex:** Passenger biological gender (Mapped and encoded to `0` for Female, `1` for Male)
* **Age:** Age in years (Handled missing data via median imputation)
* **SibSp:** Number of siblings or spouses aboard the Titanic
* **Parch:** Number of parents or children aboard the Titanic
* **Fare:** Passenger fare (numerical ticket price)
* **Embarked:** Port of Embarkation (Encoded dynamically as C = Cherbourg, Q = Queenstown, S = Southampton)

### Target Variable:
* **Survived:** `0` = No (Deceased), `1` = Yes (Survived)

---

## 🛠️ Architecture & Technologies Used

This project demonstrates an end-to-end full-stack deployment pipeline, cleanly decoupling the machine learning inference engine from the client user interface:

* **Machine Learning Stack:** Python, Scikit-Learn, Pandas, NumPy, Matplotlib
* **Backend Inference Engine:** Flask REST API hosted securely on **Render**
* **Frontend User Interface:** Native HTML5, CSS3 Custom Properties (Variables), Vanilla JavaScript (Asynchronous Fetch API for seamless SPA navigation) hosted on **GitHub Pages**

---

## 🚀 Project Workflow & Mechanics
1. **Data Preprocessing:** Cleaning missing data points, dropped uninformative columns (PassengerId, Name, Ticket, Cabin), and handled categorical encoding.
2. **Model Optimization:** Training a predictive classification model using Scikit-Learn to optimize prediction accuracy.
3. **API Deployment:** Wrapping the trained model pipeline into a Flask application endpoint capable of receiving and parsing `POST` JSON payloads.
4. **Client-Side Interactivity:** Capturing user inputs asynchronously, updating the active DOM view instantly without hard page refreshes, and rendering real-time predictions directly from the cloud backend.

---
