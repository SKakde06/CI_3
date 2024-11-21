import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import (
    mean_squared_error, r2_score, 
    accuracy_score, precision_score, 
    recall_score, f1_score, 
    classification_report, 
    silhouette_score
)

# Logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

def create_synthetic_dataset(n_samples=1000):
    """Create a synthetic dataset for customer behavior analysis."""
    np.random.seed(42)
    
    # Generate synthetic data
    data = pd.DataFrame({
        'user_id': range(1, n_samples + 1),
        'age': np.random.randint(18, 65, n_samples),
        'annual_income': np.random.normal(50000, 15000, n_samples),
        'purchase_amount': np.random.normal(250, 100, n_samples),
        'products_purchased': np.random.randint(1, 10, n_samples),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples)
    })
    
    # Add some correlations
    data['purchase_amount'] = data['purchase_amount'] + 0.5 * data['annual_income'] / 1000
    data['purchase_amount'] = np.abs(data['purchase_amount'])
    
    return data

def data_exploration(data):
    """Perform and display data exploration."""
    st.header("Data Exploration")
    
    # Display dataset
    st.subheader("Dataset Overview")
    st.dataframe(data.head())
    
    # Basic statistics
    st.subheader("Basic Statistics")
    st.dataframe(data.describe())
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Income Distribution")
        fig, ax = plt.subplots()
        sns.histplot(data['annual_income'], kde=True, ax=ax)
        st.pyplot(fig)
        plt.close(fig)
    
    with col2:
        st.subheader("Purchase Amount Distribution")
        fig, ax = plt.subplots()
        sns.histplot(data['purchase_amount'], kde=True, ax=ax)
        st.pyplot(fig)
        plt.close(fig)

def regression_analysis(data):
    """Perform regression to predict purchase amount."""
    st.header("Regression Analysis: Predicting Purchase Amount")
    
    # Prepare data
    X = data[['age', 'annual_income', 'products_purchased']]
    y = data['purchase_amount']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regression": RandomForestRegressor(random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {"RMSE": rmse, "R2": r2}
        
        st.subheader(f"{name} Results")
        st.write(f"RMSE: {rmse:.2f}")
        st.write(f"R² Score: {r2:.2f}")
        
        # Scatter plot of predictions
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{name}: Actual vs Predicted")
        st.pyplot(fig)
        plt.close(fig)

def classification_analysis(data):
    """Perform classification to predict region."""
    st.header("Classification Analysis: Predicting Region")
    
    # Prepare data
    X = data[['age', 'annual_income', 'purchase_amount', 'products_purchased']]
    y = data['region']
    
    # Encode target variable
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Models
    models = {
        "Logistic Regression": LogisticRegression(multi_class='ovr', max_iter=1000),
        "Random Forest Classifier": RandomForestClassifier(random_state=42)
    }
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        st.subheader(f"{name} Results")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred, 
                                      target_names=le.classes_))

def clustering_analysis(data):
    """Perform clustering to segment customers."""
    st.header("Clustering Analysis: Customer Segmentation")
    
    # Prepare data
    X = data[['age', 'annual_income', 'purchase_amount', 'products_purchased']]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Clustering algorithms
    clustering_methods = {
        "KMeans": KMeans(n_clusters=3, random_state=42),
        "Agglomerative Clustering": AgglomerativeClustering(n_clusters=3),
        "Gaussian Mixture": GaussianMixture(n_components=3, random_state=42)
    }
    
    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    for name, clusterer in clustering_methods.items():
        st.subheader(f"{name} Clustering")
        
        if name == "Gaussian Mixture":
            clusters = clusterer.fit_predict(X_scaled)
        else:
            clusters = clusterer.fit_predict(X_scaled)
        
        # Compute silhouette score
        try:
            sil_score = silhouette_score(X_scaled, clusters)
            st.write(f"Silhouette Score: {sil_score:.2f}")
        except Exception as e:
            st.write(f"Could not compute silhouette score: {e}")
        
        # Visualization
        fig, ax = plt.subplots()
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
        ax.set_title(f"{name} Clustering Visualization")
        ax.set_xlabel("First Principal Component")
        ax.set_ylabel("Second Principal Component")
        plt.colorbar(scatter, ax=ax)
        st.pyplot(fig)
        plt.close(fig)

def main():
    st.set_page_config(page_title="Customer Behavior ML", layout="wide")
    
    st.title("🚀 Customer Behavior Machine Learning Analysis")
    
    # Generate synthetic dataset
    data = create_synthetic_dataset()
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "Data Exploration", 
        "Regression Analysis", 
        "Classification Analysis", 
        "Clustering Analysis"
    ])
    
    with tab1:
        data_exploration(data)
    
    with tab2:
        regression_analysis(data)
    
    with tab3:
        classification_analysis(data)
    
    with tab4:
        clustering_analysis(data)

if __name__ == "__main__":
    main()
