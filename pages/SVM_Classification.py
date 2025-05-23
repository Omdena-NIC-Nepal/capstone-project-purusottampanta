import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.datasets import make_classification
from joblib import dump, load
import io

if 'model' not in st.session_state:
    # Pre-train a default model
    X, y = make_classification(n_samples=2000, n_features=8, random_state=42)
    scaler = StandardScaler().fit(X)
    model = svm.SVC(kernel='rbf', C=1.0, gamma=0.1, probability=True)
    model.fit(scaler.transform(X), y)
    
    st.session_state.model = model
    st.session_state.scaler = scaler

# Generate synthetic extreme weather data
@st.cache_data
def load_data():
    X, y = make_classification(
        n_samples=2000,
        n_features=8,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        weights=[0.9, 0.1],
        random_state=42,
        flip_y=0.05,
        class_sep=1.5
    )
    feature_names = [
        'Rainfall (mm)', 'Temperature (°C)', 'Humidity (%)',
        'Wind Speed (m/s)', 'Pressure (hPa)', 'Elevation (m)',
        'River Level (m)', 'Soil Moisture'
    ]
    return pd.DataFrame(X, columns=feature_names), pd.Series(y)

def plot_decision_boundary(model, X, y):
    plt.figure(figsize=(10, 6))
    h = .02  # Step size in the mesh
    
    # Create mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Plot decision boundary
    Z = model.predict(np.c_[xx.ravel(), yy.ravel(), np.zeros((xx.ravel().shape[0], X.shape[1]-2))])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    
    # Plot training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('SVM Decision Boundary')
    return plt

def app():
    st.title("⚡ Extreme Event Prediction using SVM")
    
    # Load data
    X, y = load_data()
    feature_names = X.columns.tolist()
    
    # Sidebar controls
    with st.sidebar:
        st.header("Model Configuration")
        kernel_type = st.selectbox(
            "Kernel Type",
            options=['linear', 'rbf', 'poly', 'sigmoid'],
            index=1
        )
        C = st.slider("Regularization (C)", 0.01, 10.0, 1.0)
        gamma = st.slider("Kernel Coefficient (gamma)", 0.001, 1.0, 0.1)
        test_size = st.slider("Test Size (%)", 10, 40, 20)
        random_state = st.number_input("Random State", 0, 100, 42)
        
        # if st.button("Train Model"):
        #     st.session_state.trained = True
        # else:
        #     st.session_state.trained = False


        if st.button("Train New Model"):
            # Train new model with current parameters
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size/100, random_state=random_state, stratify=y
            )
            scaler = StandardScaler().fit(X_train)
            model = svm.SVC(
                kernel=kernel_type,
                C=C,
                gamma=gamma,
                probability=True,
                random_state=random_state
            )
            model.fit(scaler.transform(X_train), y_train)
            
            # Update session state
            st.session_state.model = model
            st.session_state.scaler = scaler
            st.success("Model retrained successfully!")

            st.session_state.trained = True
        else:
            st.session_state.trained = False
            
    # Data preprocessing
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=test_size/100, random_state=random_state, stratify=y
    # )
    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)
    
    # Model training
    if st.session_state.trained:
        # model = svm.SVC(
        #     kernel=kernel_type,
        #     C=C,
        #     gamma=gamma,
        #     probability=True,
        #     random_state=random_state
        # )
        # model.fit(X_train_scaled, y_train)
        
        # # Predictions
        # y_pred = model.predict(X_test_scaled)
        # y_prob = model.predict_proba(X_test_scaled)[:, 1]

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]
        
        # Performance metrics
        accuracy = metrics.accuracy_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred)
        recall = metrics.recall_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred)
        roc_auc = metrics.roc_auc_score(y_test, y_prob)
        
        # Confusion matrix
        cm = metrics.confusion_matrix(y_test, y_pred)
        
        # Display metrics
        st.header("Model Performance")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Accuracy", f"{accuracy:.2%}")
        col2.metric("Precision", f"{precision:.2%}")
        col3.metric("Recall", f"{recall:.2%}")
        col4.metric("F1 Score", f"{f1:.2%}")
        col5.metric("ROC AUC", f"{roc_auc:.2%}")
        
        # Visualizations
        st.subheader("Performance Visualizations")
        
        # Confusion Matrix
        fig1, ax1 = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        st.pyplot(fig1)
        
        # ROC Curve
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        fig2, ax2 = plt.subplots()
        ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
        ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('Receiver Operating Characteristic')
        ax2.legend(loc="lower right")
        st.pyplot(fig2)
        
        # Precision-Recall Curve
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
        
        fig3, ax3 = plt.subplots()
        ax3.plot(recall_curve, precision_curve, color='blue', lw=2)
        ax3.set_xlim([0.0, 1.0])
        ax3.set_ylim([0.0, 1.05])
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.set_title('Precision-Recall Curve')
        st.pyplot(fig3)
        
        # Feature Importance (for linear kernel)
        if kernel_type == 'linear':
            fig4, ax4 = plt.subplots()
            importance = model.coef_[0]
            sns.barplot(x=importance, y=feature_names, ax=ax4)
            ax4.set_title('Feature Importance (Linear Kernel)')
            st.pyplot(fig4)
        
        # Model persistence
        buffer = io.BytesIO()
        dump(model, buffer)
        st.download_button(
            label="Download Trained Model",
            data=buffer,
            file_name="svm_model.joblib",
            mime="application/octet-stream"
        )
    
    # Prediction interface
    st.header("Real-time Prediction")
    with st.form("prediction_form"):
        st.subheader("Input Parameters")
        inputs = []
        cols = st.columns(4)
        for i, feature in enumerate(feature_names):
            with cols[i % 4]:
                inputs.append(st.number_input(
                    label=feature,
                    value=float(X.iloc[0, i]),
                    step=0.1
                ))
        
        if st.form_submit_button("Predict Extreme Event"):
            model = st.session_state.model
            scaler = st.session_state.scaler
            # if 'model' in locals():
            if model:
                input_data = scaler.transform([inputs])
                prediction = model.predict(input_data)
                probability = model.predict_proba(input_data)[0][1]
                if prediction[0] == 1:
                    st.error(f"Extreme Event Likely ({probability:.2%} probability)")
                else:
                    st.success(f"Normal Conditions ({1-probability:.2%} probability)")
            else:
                st.warning("Please train the model first")

if __name__ == "__main__":
    app()