"""
F1 DNF Classification - Streamlit App
Predicts whether an F1 driver will finish a race or DNF (Did Not Finish)
"""

import streamlit as st
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, learning_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    RocCurveDisplay, precision_recall_curve, PrecisionRecallDisplay,
    f1_score, precision_score, recall_score, average_precision_score
)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.inspection import permutation_importance
from sklearn.tree import plot_tree
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="F1 DNF Classification",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Performance optimization: Cache expensive computations
@st.cache_data
def load_kaggle_dataset():
    """Load dataset from Kaggle with caching"""
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "pranay13257/f1-dnf-classification",
        "f1_dnf.csv" 
    )
    return df

@st.cache_data
def clean_dataframe(df):
    """Clean dataframe with caching"""
    df = df.copy()
    # Convert to numeric
    df['points'] = pd.to_numeric(df['points'], errors='coerce')
    df['laps'] = pd.to_numeric(df['laps'], errors='coerce')
    df['milliseconds'] = pd.to_numeric(df['milliseconds'], errors='coerce')
    df['fastestLapSpeed'] = pd.to_numeric(df['fastestLapSpeed'], errors='coerce')
    df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
    df['fastestLap'] = pd.to_numeric(df['fastestLap'], errors='coerce')
    df['fastestLapTime'] = pd.to_numeric(df['fastestLapTime'], errors='coerce')
    
    # Drop columns
    df = df.drop(columns=['milliseconds','fastestLap','rank','fastestLapTime','fastestLapSpeed'])
    
    # Fill missing values
    df['points'] = df.groupby('raceId')['points'].transform(lambda x: x.fillna(x.median()))
    df['points'] = df['points'].fillna(df['points'].median())
    df['laps'] = df.groupby('raceId')['laps'].transform(lambda x: x.fillna(x.median()))
    df['laps'] = df['laps'].fillna(df['laps'].median())
    
    return df

@st.cache_data
def compute_correlation_matrix(df):
    """Compute correlation matrix with caching"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return df[numeric_cols].corr()

st.title("🏎️ F1 DNF Classification Analysis")
st.markdown("---")

# Sidebar for navigation with organized sections
st.sidebar.title("📊 Navigation")
st.sidebar.markdown("---")

# Initialize last_section if not exists
if 'last_section' not in st.session_state:
    st.session_state.last_section = None

# Single section selector with cleaner groups
section = st.sidebar.radio(
    "Select Section:",
    options=[
        "� Data Loading",
        "🧹 Data Cleaning", 
        "� Visualization",
        "⚙️ Feature Engineering",
        "⚖️ Class Imbalance",
        "🎓 Model Training",
        "🏆 Model Comparison",
        "📉 Model Evaluation",
        "📚 Learning Curves",
        "�️ Hyperparameter Tuning",
        "🎯 Feature Importance",
        "🚀 Live Prediction"
    ],
    index=0 if st.session_state.last_section is None else None,
    key="main_nav",
    label_visibility="collapsed"
)

# Remove emojis for section matching
if section:
    section = section.split(" ", 1)[1] if " " in section else section
    st.session_state.last_section = section

# Add section info in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📍 Current Section")
st.sidebar.info(f"**{section}**")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Quick Guide")
with st.sidebar.expander("📖 Workflow"):
    st.markdown("""
    **Recommended Order:**
    1. 📊 Load Data
    2. 🧹 Clean Data
    3. 📈 Visualize
    4. 🎓 Train Model
    5. 🏆 Compare Models
    6. 🚀 Make Predictions
    
    **Advanced:**
    - ⚙️ Feature Engineering
    - ⚖️ Class Imbalance
    - 📚 Learning Curves
    - 🎛️ Hyperparameter Tuning
    - 🎯 Feature Importance
    """)

with st.sidebar.expander("ℹ️ About"):
    st.markdown("""
    **F1 DNF Classification**
    
    Predict whether a Formula 1 driver will finish a race or DNF (Did Not Finish).
    
    **Tech Stack:**
    - Python, Streamlit
    - Scikit-learn
    - Imbalanced-learn
    - Pandas, NumPy
    - Matplotlib, Seaborn, Plotly
    
    **Features:**
    - Multi-model comparison
    - Feature engineering
    - Class imbalance handling
    - Hyperparameter optimization
    - Real-time predictions
    """)

st.sidebar.markdown("---")
st.sidebar.caption("Built with ❤️ using Streamlit")

# Initialize session state for data persistence
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_cleaned' not in st.session_state:
    st.session_state.df_cleaned = None
if 'df_encoded' not in st.session_state:
    st.session_state.df_encoded = None
if 'df_engineered' not in st.session_state:
    st.session_state.df_engineered = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'all_models' not in st.session_state:
    st.session_state.all_models = {}
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}
if 'balanced_models' not in st.session_state:
    st.session_state.balanced_models = {}
if 'balanced_results' not in st.session_state:
    st.session_state.balanced_results = {}
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'engineered_features' not in st.session_state:
    st.session_state.engineered_features = None
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'best_params' not in st.session_state:
    st.session_state.best_params = None
if 'feature_importances' not in st.session_state:
    st.session_state.feature_importances = None

# Add caching flags for performance
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'data_cleaned' not in st.session_state:
    st.session_state.data_cleaned = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# ============================================================================
# SECTION 1: DATA LOADING
# ============================================================================
if section == "Data Loading":
    st.header("📊 Data Loading")
    
    # Callback function to load data
    def load_data_callback():
        """Callback to load dataset - runs before page refresh"""
        try:
            df = load_kaggle_dataset()
            st.session_state.df = df
            st.session_state.data_loaded = True
        except Exception as e:
            st.session_state.load_error = str(e)
    
    # Callback function to reload data
    def reload_data_callback():
        """Callback to reset and reload dataset"""
        st.session_state.df = None
        st.session_state.df_cleaned = None
        st.session_state.df_encoded = None
        st.session_state.data_loaded = False
        if 'load_error' in st.session_state:
            del st.session_state.load_error
    
    # Show load button only if data is not loaded
    if st.session_state.df is None:
        # Warning about button behavior
        st.warning("⚠️ **Note:** If then load button does not work in one go , go to a different page in nav bar and come again.")
        
        # Check if there was an error from callback
        if 'load_error' in st.session_state:
            st.error(f"Error loading dataset: {st.session_state.load_error}")
            st.info("💡 Make sure you have kagglehub installed: `pip install kagglehub`")
            del st.session_state.load_error
        
        # Button with callback - callback runs BEFORE page refresh
        st.button(
            "� Load F1 DNF Dataset from Kaggle", 
            type="primary", 
            use_container_width=True,
            on_click=load_data_callback
        )
    else:
        st.success("✅ Dataset already loaded!")
        st.button(
            "🔄 Reload Dataset",
            on_click=reload_data_callback
        )
    
    # Display data if loaded
    if st.session_state.df is not None:
        df = st.session_state.df
        
        st.subheader("Dataset Overview")
        st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**First 5 rows:**")
            st.dataframe(df.head())
        
        with col2:
            st.write("**Dataset Info:**")
            buffer = []
            buffer.append(f"Total entries: {len(df)}")
            buffer.append(f"Total columns: {len(df.columns)}")
            buffer.append("\nColumn details:")
            for col in df.columns:
                buffer.append(f"- {col}: {df[col].dtype}")
            st.text("\n".join(buffer))
        
        st.subheader("Target Variable Distribution")
        st.write("**Note:** target_finish == 1 means driver finished, 0 means DNF")
        target_counts = df['target_finish'].value_counts()
        st.write(target_counts)
        
        st.subheader("Sample Data from Speed Columns")
        st.dataframe(df[['fastestLap','fastestLapSpeed','fastestLapTime']].sample(5))
        
        # Check for suspicious values in object columns
        st.subheader("Checking Object Columns for Suspicious Values")
        object_cols = df.select_dtypes(include='object').columns
        
        for col in object_cols:
            unique_vals = df[col].unique()
            with st.expander(f"{col}: {len(unique_vals)} unique values"):
                st.write(f"Sample values: {unique_vals[:10]}")
                
                suspicious = df[df[col].isin(['\\N', 'NaN', 'nan', 'NULL', 'null', '', ' '])][col].value_counts()
                if len(suspicious) > 0:
                    st.warning(f"Found suspicious values:")
                    st.write(suspicious)

# ============================================================================
# SECTION 2: DATA CLEANING
# ============================================================================
elif section == "Data Cleaning":
    st.header("🧹 Data Cleaning")
    
    if st.session_state.df is None:
        st.warning("Please load the data first from the 'Data Loading' section.")
    else:
        st.info("💡 Click the button below to clean all data in one step (optimized for performance)")
        
        if st.button("🚀 Clean All Data (Fast)", type="primary"):
            with st.spinner("Cleaning data..."):
                try:
                    df = clean_dataframe(st.session_state.df)
                    st.session_state.df_cleaned = df
                    st.session_state.data_cleaned = True
                    st.success("✅ Data cleaned successfully!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Shape", f"{df.shape[0]} × {df.shape[1]}")
                    with col2:
                        st.metric("Total NaNs", df.isna().sum().sum())
                    with col3:
                        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                    
                    st.write("**Sample of cleaned data:**")
                    st.dataframe(df.head(), use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error cleaning data: {e}")
        
        # Show manual cleaning options in expander
        with st.expander("🔧 Advanced: Manual Step-by-Step Cleaning"):
            df = st.session_state.df.copy()
            
            st.subheader("Step 1: Convert Object Columns to Numeric")
            if st.button("Convert to Numeric"):
                with st.spinner("Converting columns..."):
                    df['points'] = pd.to_numeric(df['points'], errors='coerce')
                    df['laps'] = pd.to_numeric(df['laps'], errors='coerce')
                    df['milliseconds'] = pd.to_numeric(df['milliseconds'], errors='coerce')
                    df['fastestLapSpeed'] = pd.to_numeric(df['fastestLapSpeed'], errors='coerce')
                    df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
                    df['fastestLap'] = pd.to_numeric(df['fastestLap'], errors='coerce')
                    df['fastestLapTime'] = pd.to_numeric(df['fastestLapTime'], errors='coerce')
                    
                    st.success("Conversion complete!")
                    st.write("NaN counts after conversion:")
                    st.write(df.isna().sum()[df.isna().sum() > 0])
            
            st.subheader("Step 2: Drop Columns with Too Many NaNs")
            st.write("Columns to drop: milliseconds, fastestLap, rank, fastestLapTime, fastestLapSpeed")
            
            if st.button("Drop Columns"):
                df = df.drop(columns=['milliseconds','fastestLap','rank','fastestLapTime','fastestLapSpeed'])
                st.success("Columns dropped!")
                st.write(f"New shape: {df.shape}")
            
            st.subheader("Step 3: Fill Missing Values")
            
            if st.button("Fill NaNs in 'points' and 'laps'"):
                with st.spinner("Filling missing values..."):
                    df['points'] = df.groupby('raceId')['points'].transform(lambda x: x.fillna(x.median()))
                    df['points'] = df['points'].fillna(df['points'].median())
                    df['laps'] = df.groupby('raceId')['laps'].transform(lambda x: x.fillna(x.median()))
                    df['laps'] = df['laps'].fillna(df['laps'].median())
                    
                    st.success("Missing values filled!")
                    st.write(f"Remaining NaNs in 'points': {df['points'].isna().sum()}")
                    st.write(f"Remaining NaNs in 'laps': {df['laps'].isna().sum()}")
        
        # Show column info
        if st.session_state.df_cleaned is not None:
            st.divider()
            st.subheader("📊 Cleaned Data Summary")
            df_clean = st.session_state.df_cleaned
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Column Unique Value Counts:**")
                unique_counts = pd.DataFrame({
                    'Column': df_clean.columns,
                    'Unique Values': [df_clean[col].nunique() for col in df_clean.columns],
                    'Data Type': [str(df_clean[col].dtype) for col in df_clean.columns]
                })
                st.dataframe(unique_counts, use_container_width=True, hide_index=True)
            
            with col2:
                st.write("**Missing Values:**")
                missing = df_clean.isna().sum()
                if missing.sum() == 0:
                    st.success("✅ No missing values!")
                else:
                    st.dataframe(missing[missing > 0], use_container_width=True)

# ============================================================================
# SECTION 3: VISUALIZATION
# ============================================================================
elif section == "Visualization":
    st.header("📈 Data Visualization")
    
    if st.session_state.df_cleaned is None:
        st.warning("Please clean the data first from the 'Data Cleaning' section.")
    else:
        df = st.session_state.df_cleaned
        
        # Tabs for better organization
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Correlations", "📈 Class Balance", "🎯 Feature Distributions", "✅ Safe Features"])
        
        with tab1:
            st.subheader("Feature Correlations")
            with st.spinner("Computing correlations..."):
                correlation_matrix = compute_correlation_matrix(df)
                
                fig, ax = plt.subplots(figsize=(12, 10))
                mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
                sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                            center=0, mask=mask, square=True, linewidths=1,
                            cbar_kws={"shrink": 0.8}, ax=ax)
                ax.set_title('Feature Correlations', fontsize=14, fontweight='bold', pad=20)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        
        with tab2:
            st.subheader("DNF vs Finished Distribution")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            counts = df['target_finish'].value_counts()
            colors = ['#e74c3c', '#2ecc71']
            
            ax1.bar(['DNF', 'Finished'], counts.values, color=colors, edgecolor='black', linewidth=2)
            ax1.set_title('DNF vs Finished Distribution', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Count', fontsize=11)
            
            for i, count in enumerate(counts.values):
                ax1.text(i, count + 100, f'{count:,}', ha='center', fontsize=11, fontweight='bold')
            
            ax2.pie(counts.values, labels=['DNF', 'Finished'], autopct='%1.1f%%',
                    colors=colors, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'},
                    explode=(0.05, 0))
            ax2.set_title('Proportion', fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with tab3:
            st.subheader("Feature Distributions: DNF vs Finished")
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Feature Distributions: DNF vs Finished', fontsize=16, fontweight='bold')
            
            features_to_plot = [
                ('grid', 'Starting Grid Position'),
                ('points', 'Points'),
                ('laps', 'Laps Completed'),
                ('positionOrder', 'Position Order'),
                ('year', 'Race Year'),
                ('round', 'Round Number')
            ]
            
            for idx, (feature, title) in enumerate(features_to_plot):
                ax = axes[idx // 3, idx % 3]
                
                ax.hist(df[df['target_finish']==0][feature].dropna(), bins=25, alpha=0.6, 
                        label='DNF', color='#e74c3c', edgecolor='black')
                ax.hist(df[df['target_finish']==1][feature].dropna(), bins=25, alpha=0.6, 
                        label='Finished', color='#2ecc71', edgecolor='black')
            
                ax.set_title(title, fontsize=11, fontweight='bold')
                ax.set_xlabel(feature, fontsize=10)
                ax.set_ylabel('Frequency', fontsize=10)
                ax.legend(loc='upper right', fontsize=9)
                ax.grid(True, alpha=0.3, linestyle='--')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with tab4:
            st.subheader("Safe Features - No Data Leakage")
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Safe Features - No Data Leakage', fontsize=16, fontweight='bold', color='green')
            
            safe_features = ['grid', 'year', 'round', 'circuitId']
            
            for idx, feature in enumerate(safe_features):
                ax = axes[idx // 2, idx % 2]
                
                ax.hist(df[df['target_finish']==0][feature].dropna(), bins=20, alpha=0.6, 
                        label='DNF', color='#e74c3c', edgecolor='black')
                ax.hist(df[df['target_finish']==1][feature].dropna(), bins=20, alpha=0.6, 
                        label='Finished', color='#2ecc71', edgecolor='black')
                
                ax.set_title(f'{feature} Distribution', fontsize=12, fontweight='bold')
                ax.set_xlabel(feature, fontsize=10)
                ax.set_ylabel('Frequency', fontsize=10)
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# ============================================================================
# SECTION 4: FEATURE ENGINEERING
# ============================================================================
elif section == "Feature Engineering":
    st.header("🔧 Advanced Feature Engineering")
    st.write("Create interaction features and polynomial features to improve model performance!")
    
    if st.session_state.df_encoded is None:
        st.warning("⚠️ Please complete the Model Training section first to encode the data.")
    else:
        df = st.session_state.df_encoded.copy()
        
        st.subheader("📊 Current Features")
        base_features = ['grid', 'driverRef_encoded', 'constructorRef_encoded', 'circuitId_encoded', 'raceId_encoded']
        st.write(f"**Base features ({len(base_features)}):**", base_features)
        
        st.markdown("---")
        
        # Interaction Features
        st.subheader("🔀 1. Interaction Features")
        st.write("Create features by combining existing ones to capture relationships")
        
        create_interactions = st.checkbox("Create Interaction Features", value=True)
        
        if create_interactions:
            st.write("**Interactions to create:**")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("- `driver × constructor` - Driver-team synergy")
                st.write("- `grid × driver` - Starting position advantage")
                st.write("- `grid × circuit` - Circuit-specific grid impact")
            
            with col2:
                st.write("- `driver × circuit` - Driver circuit expertise")
                st.write("- `constructor × circuit` - Team circuit performance")
                st.write("- `grid squared` - Non-linear grid effect")
        
        # Polynomial Features
        st.subheader("📈 2. Polynomial Features")
        st.write("Create polynomial combinations for non-linear relationships")
        
        create_poly = st.checkbox("Create Polynomial Features (degree 2)", value=False)
        
        if create_poly:
            st.warning("⚠️ This will significantly increase feature count and training time!")
        
        st.markdown("---")
        
        if st.button("🚀 Apply Feature Engineering"):
            with st.spinner("Creating engineered features..."):
                df_eng = df.copy()
                new_features = []
                
                # Base features
                feature_cols = base_features.copy()
                
                # Create interaction features
                if create_interactions:
                    # Driver × Constructor interaction
                    df_eng['driver_constructor'] = df_eng['driverRef_encoded'] * df_eng['constructorRef_encoded']
                    new_features.append('driver_constructor')
                    
                    # Grid × Driver interaction
                    df_eng['grid_driver'] = df_eng['grid'] * df_eng['driverRef_encoded']
                    new_features.append('grid_driver')
                    
                    # Grid × Circuit interaction
                    df_eng['grid_circuit'] = df_eng['grid'] * df_eng['circuitId_encoded']
                    new_features.append('grid_circuit')
                    
                    # Driver × Circuit interaction
                    df_eng['driver_circuit'] = df_eng['driverRef_encoded'] * df_eng['circuitId_encoded']
                    new_features.append('driver_circuit')
                    
                    # Constructor × Circuit interaction
                    df_eng['constructor_circuit'] = df_eng['constructorRef_encoded'] * df_eng['circuitId_encoded']
                    new_features.append('constructor_circuit')
                    
                    # Grid squared (non-linear effect)
                    df_eng['grid_squared'] = df_eng['grid'] ** 2
                    new_features.append('grid_squared')
                    
                    feature_cols.extend(new_features)
                
                # Create polynomial features
                if create_poly:
                    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
                    X_poly = poly.fit_transform(df_eng[base_features])
                    poly_feature_names = poly.get_feature_names_out(base_features)
                    
                    # Add polynomial features
                    for i, feat_name in enumerate(poly_feature_names):
                        if feat_name not in base_features:  # Skip original features
                            df_eng[feat_name] = X_poly[:, i]
                            new_features.append(feat_name)
                    
                    feature_cols = list(poly_feature_names)
                    if create_interactions:
                        # Add manual interaction features
                        for feat in ['driver_constructor', 'grid_driver', 'grid_circuit', 
                                   'driver_circuit', 'constructor_circuit', 'grid_squared']:
                            if feat not in feature_cols:
                                feature_cols.append(feat)
                
                # Store engineered dataframe
                st.session_state.df_engineered = df_eng
                st.session_state.engineered_features = feature_cols
                
                st.success("✅ Feature engineering complete!")
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Original Features", len(base_features))
                with col2:
                    st.metric("New Features Created", len(new_features))
                with col3:
                    st.metric("Total Features", len(feature_cols))
                
                # Show sample
                st.subheader("📋 Engineered Features Sample")
                st.dataframe(df_eng[feature_cols].head())
                
                # Feature correlation with target
                st.subheader("🎯 New Features Correlation with Target")
                if new_features:
                    correlations = []
                    for feat in new_features:
                        corr = df_eng[feat].corr(df_eng['target_finish'])
                        correlations.append({'Feature': feat, 'Correlation': abs(corr)})
                    
                    corr_df = pd.DataFrame(correlations).sort_values('Correlation', ascending=False)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    colors = ['#2ecc71' if i < 3 else '#3498db' for i in range(len(corr_df))]
                    bars = ax.barh(corr_df['Feature'], corr_df['Correlation'], 
                                  color=colors, edgecolor='black', linewidth=2)
                    ax.set_xlabel('Absolute Correlation with Target', fontsize=11, fontweight='bold')
                    ax.set_title('New Features Importance (by correlation)', fontsize=12, fontweight='bold')
                    ax.grid(axis='x', alpha=0.3, linestyle='--')
                    
                    # Add value labels
                    for bar, val in zip(bars, corr_df['Correlation']):
                        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2, f'{val:.4f}',
                               va='center', fontweight='bold', fontsize=9)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                st.info("💡 **Next Steps:** Use these engineered features in Model Training or Class Imbalance sections!")

# ============================================================================
# SECTION 5: MODEL TRAINING
# ============================================================================
elif section == "Model Training":
    st.header("🤖 Model Training - Baseline Model")
    
    st.info("💡 **In this section, we'll train a baseline Random Forest model** to establish initial performance. Random Forest is a good starting point - it's robust, handles non-linear relationships well, and is less prone to overfitting. After training, you can compare it with other models (Gradient Boosting, Logistic Regression, Decision Tree) in the 'Model Comparison' section to find the best performer!")
    
    if st.session_state.df_cleaned is None:
        st.warning("Please clean the data first from the 'Data Cleaning' section.")
    else:
        # Use encoded dataframe if it exists, otherwise use cleaned dataframe
        if st.session_state.df_encoded is not None:
            df = st.session_state.df_encoded.copy()
        else:
            df = st.session_state.df_cleaned.copy()
        
        st.subheader("Step 1: Target Encoding")
        st.write("Encoding categorical features based on DNF rate using KFold to prevent leakage")
        
        def target_encode(df, column, target, n_splits=5):
            """Safe target encoding using KFold to prevent leakage"""
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            encoded_col = np.zeros(len(df))
            
            for train_idx, val_idx in kf.split(df):
                means = df.iloc[train_idx].groupby(column)[target].mean()
                encoded_col[val_idx] = df.iloc[val_idx][column].map(means)
            
            encoded_col = pd.Series(encoded_col).fillna(df[target].mean())
            return encoded_col
        
        # Check if encoding has been done
        encoding_done = all(col in df.columns for col in ['driverRef_encoded', 'constructorRef_encoded', 'circuitId_encoded', 'raceId_encoded'])
        
        if encoding_done:
            st.success("✅ Target encoding already applied!")
        
        if st.button("Apply Target Encoding", disabled=encoding_done):
            with st.spinner("Encoding features..."):
                df['driverRef_encoded'] = target_encode(df, 'driverRef', 'target_finish')
                df['constructorRef_encoded'] = target_encode(df, 'constructorRef', 'target_finish')
                df['circuitId_encoded'] = target_encode(df, 'circuitId', 'target_finish')
                df['raceId_encoded'] = target_encode(df, 'raceId', 'target_finish')
                
                # Save to session state
                st.session_state.df_encoded = df
                
                st.success("Encoding complete!")
                st.write("Sample of encoded data:")
                st.dataframe(df.sample(5))
        
        st.subheader("Step 2: Feature Selection")
        features = ['grid', 'driverRef_encoded', 'constructorRef_encoded', 'circuitId_encoded', 'raceId_encoded']
        st.write("**Selected features for the model:**", features)
        st.caption("These features represent: starting grid position, driver performance, team performance, circuit characteristics, and race difficulty")
        
        # Check if encoded columns exist
        encoded_cols_exist = all(col in df.columns for col in ['driverRef_encoded', 'constructorRef_encoded', 'circuitId_encoded', 'raceId_encoded'])
        
        if not encoded_cols_exist:
            st.warning("⚠️ Please apply target encoding first before preparing features!")
        
        if st.button("Prepare Features and Target", disabled=not encoded_cols_exist):
            X = df[features]
            y = df['target_finish']
            
            st.success("Features prepared!")
            
            # Verify no data leakage
            st.write("**Correlation with target_finish:**")
            st.write(X.corrwith(y).sort_values(ascending=False))
            
            st.write("**Feature summary:**")
            st.write(X.describe())
            
            # Store in session state
            st.session_state.X = X
            st.session_state.y = y
            st.session_state.features = features
            st.session_state.df_encoded = df
        
        st.subheader("Step 3: Train-Test Split")
        
        features_prepared = 'X' in st.session_state and 'y' in st.session_state
        
        if not features_prepared:
            st.warning("⚠️ Please prepare features first!")
        
        if st.button("Split Data", disabled=not features_prepared):
            X = st.session_state.X
            y = st.session_state.y
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            
            st.success("Data split complete!")
            st.write(f"Training set: {X_train.shape}")
            st.write(f"Test set: {X_test.shape}")
            st.write(f"Class distribution in train:")
            st.write(y_train.value_counts(normalize=True))
        
        st.subheader("Step 4: Train Random Forest Model with Hyperparameter Tuning")
        st.write("**Model:** Random Forest Classifier (Baseline)")
        st.write("**Why Random Forest?** Good starting point for comparison - will train other models in 'Model Comparison' section")
        st.write("**Optimization:** GridSearchCV will test different hyperparameter combinations to find the best settings")
        st.write("**Evaluation Metric:** ROC AUC (Area Under the Receiver Operating Characteristic Curve)")
        st.caption("⏱️ Training time: 1-2 minutes (fast mode) or 7-9 minutes (thorough mode)")
        
        use_full_grid = st.checkbox("Use full parameter grid (slower but more thorough 7-9 minutes)", value=False)
        
        if use_full_grid:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [10, 20, 30],
                'min_samples_leaf': [5, 10, 15],
                'max_features': ['sqrt', 'log2']
            }
            cv_folds = 2
            st.info("📋 **Thorough Grid:** Testing 540 parameter combinations")
        else:
            param_grid = {
                'n_estimators': [100],
                'max_depth': [10],
                'min_samples_split': [20],
                'min_samples_leaf': [10],
                'max_features': ['sqrt']
            }
            cv_folds = 2
            st.info("📋 **Fast Grid:** Using pre-selected optimal parameters")
        
        with st.expander("📖 What do these parameters mean?"):
            st.markdown("""
            - **n_estimators**: Number of decision trees in the forest (more trees = more accurate but slower)
            - **max_depth**: Maximum depth of each tree (controls model complexity)
            - **min_samples_split**: Minimum samples required to split a node (prevents overfitting)
            - **min_samples_leaf**: Minimum samples required at leaf node (smooths predictions)
            - **max_features**: Number of features considered for best split (adds randomness for diversity)
            """)
        
        data_split_done = 'X_train' in st.session_state and 'y_train' in st.session_state
        
        if not data_split_done:
            st.warning("⚠️ Please split the data first!")
        
        if st.button("🚀 Train Random Forest Model with GridSearchCV", disabled=not data_split_done):
            X_train = st.session_state.X_train
            y_train = st.session_state.y_train
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test
            
            with st.spinner("Training Random Forest model... This may take a while..."):
                # Initialize Random Forest classifier
                rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
                
                # Set up GridSearchCV for hyperparameter optimization
                grid_search = GridSearchCV(
                    estimator=rf_base,
                    param_grid=param_grid,
                    cv=cv_folds,
                    scoring='roc_auc',
                    verbose=2,
                    n_jobs=-1
                )
                
                # Train the model
                grid_search.fit(X_train, y_train)
                
                st.success("✅ Random Forest model training complete!")
                
                # Display best parameters found
                st.subheader("🏆 Best Hyperparameters Found")
                st.write(f"**Best Parameters:** {grid_search.best_params_}")
                st.caption("These are the optimal settings found by GridSearchCV")
                
                # Display cross-validation score
                st.write(f"**Best Cross-Validation ROC AUC:** {grid_search.best_score_:.4f}")
                st.caption("Average performance across validation folds during training")
                
                # Get the best model
                rf_model = grid_search.best_estimator_
                st.session_state.model = rf_model
                
                # Make predictions on test set
                y_pred = rf_model.predict(X_test)  # Binary predictions (0 or 1)
                y_pred_proba = rf_model.predict_proba(X_test)[:, 1]  # Probability of finishing (class 1)
                
                st.session_state.y_pred = y_pred
                st.session_state.y_pred_proba = y_pred_proba
                
                # Store in all_models for comparison
                st.session_state.all_models['Random Forest'] = rf_model
                st.session_state.model_results['Random Forest'] = {
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba,
                    'roc_auc': roc_auc_score(y_test, y_pred_proba),
                    'f1': f1_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred),
                    'recall': recall_score(y_test, y_pred),
                    'avg_precision': average_precision_score(y_test, y_pred_proba)
                }
                
                # Display test set performance
                st.subheader("📊 Test Set Performance")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("ROC AUC", f"{roc_auc_score(y_test, y_pred_proba):.4f}")
                    st.caption("Overall prediction quality")
                with col2:
                    st.metric("F1 Score", f"{f1_score(y_test, y_pred):.4f}")
                    st.caption("Balance of precision/recall")
                with col3:
                    st.metric("Precision", f"{precision_score(y_test, y_pred):.4f}")
                    st.caption("Correct positive predictions")
                with col4:
                    st.metric("Recall", f"{recall_score(y_test, y_pred):.4f}")
                    st.caption("Found all actual positives")
                
                st.success("✅ Baseline model saved! Now go to 'Model Comparison' section to train and compare other models (Gradient Boosting often performs better!).")

# ============================================================================
# SECTION 6: CLASS IMBALANCE HANDLING
# ============================================================================
elif section == "Class Imbalance":
    st.header("⚖️ Class Imbalance Handling")
    st.write("Address class imbalance using various resampling techniques.")
    
    if st.session_state.df_cleaned is None:
        st.warning("Please clean the data first from the 'Data Cleaning' section.")
    elif 'X_train' not in st.session_state:
        st.warning("Please complete the Model Training section first to prepare the data.")
    else:
        X_train = st.session_state.X_train
        X_test = st.session_state.X_test
        y_train = st.session_state.y_train
        y_test = st.session_state.y_test
        
        # Display original class distribution
        st.subheader("📊 Original Class Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Training Set:**")
            train_dist = pd.Series(y_train).value_counts()
            st.write(train_dist)
            st.write(f"Imbalance Ratio: {train_dist[0]/train_dist[1]:.2f}:1")
            
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            train_dist.plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4'])
            ax.set_title('Original Training Set Distribution')
            ax.set_xlabel('Class (0=Finished, 1=DNF)')
            ax.set_ylabel('Count')
            ax.set_xticklabels(['Finished', 'DNF'], rotation=0)
            st.pyplot(fig)
        
        st.divider()
        
        # Resampling technique selection
        st.subheader("🔄 Resampling Techniques")
        st.write("Choose a technique to balance the classes:")
        
        technique = st.selectbox(
            "Select Resampling Method:",
            ["None", "SMOTE (Oversampling)", "ADASYN (Adaptive Oversampling)", 
             "Random Undersampling", "SMOTETomek (Combined)"]
        )
        
        if st.button("Apply Resampling"):
            with st.spinner("Resampling data..."):
                try:
                    X_resampled = X_train.copy()
                    y_resampled = y_train.copy()
                    
                    if technique == "SMOTE (Oversampling)":
                        smote = SMOTE(random_state=42)
                        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
                        st.info("✅ Applied SMOTE: Synthetic Minority Oversampling Technique")
                        
                    elif technique == "ADASYN (Adaptive Oversampling)":
                        adasyn = ADASYN(random_state=42)
                        X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)
                        st.info("✅ Applied ADASYN: Adaptive Synthetic Sampling")
                        
                    elif technique == "Random Undersampling":
                        rus = RandomUnderSampler(random_state=42)
                        X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
                        st.info("✅ Applied Random Undersampling")
                        
                    elif technique == "SMOTETomek (Combined)":
                        smotetomek = SMOTETomek(random_state=42)
                        X_resampled, y_resampled = smotetomek.fit_resample(X_train, y_train)
                        st.info("✅ Applied SMOTETomek: Combined Over & Undersampling")
                    
                    # Display resampled distribution
                    st.subheader("📈 Resampled Class Distribution")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**After Resampling:**")
                        resampled_dist = pd.Series(y_resampled).value_counts()
                        st.write(resampled_dist)
                        st.write(f"Imbalance Ratio: {resampled_dist[0]/resampled_dist[1]:.2f}:1")
                        
                    with col2:
                        fig, ax = plt.subplots(figsize=(6, 4))
                        resampled_dist.plot(kind='bar', ax=ax, color=['#95E1D3', '#F38181'])
                        ax.set_title('Resampled Distribution')
                        ax.set_xlabel('Class (0=Finished, 1=DNF)')
                        ax.set_ylabel('Count')
                        ax.set_xticklabels(['Finished', 'DNF'], rotation=0)
                        st.pyplot(fig)
                    
                    # Train models on balanced data
                    st.divider()
                    st.subheader("🎯 Model Performance Comparison")
                    st.write("Training models on original vs. resampled data...")
                    
                    # Train on original data
                    rf_original = RandomForestClassifier(n_estimators=100, random_state=42)
                    rf_original.fit(X_train, y_train)
                    y_pred_original = rf_original.predict(X_test)
                    y_pred_proba_original = rf_original.predict_proba(X_test)[:, 1]
                    
                    # Train on resampled data
                    rf_resampled = RandomForestClassifier(n_estimators=100, random_state=42)
                    rf_resampled.fit(X_resampled, y_resampled)
                    y_pred_resampled = rf_resampled.predict(X_test)
                    y_pred_proba_resampled = rf_resampled.predict_proba(X_test)[:, 1]
                    
                    # Comparison metrics
                    comparison_data = {
                        'Metric': ['ROC AUC', 'F1 Score', 'Precision', 'Recall', 'Average Precision'],
                        'Original Data': [
                            roc_auc_score(y_test, y_pred_proba_original),
                            f1_score(y_test, y_pred_original),
                            precision_score(y_test, y_pred_original),
                            recall_score(y_test, y_pred_original),
                            average_precision_score(y_test, y_pred_proba_original)
                        ],
                        'Resampled Data': [
                            roc_auc_score(y_test, y_pred_proba_resampled),
                            f1_score(y_test, y_pred_resampled),
                            precision_score(y_test, y_pred_resampled),
                            recall_score(y_test, y_pred_resampled),
                            average_precision_score(y_test, y_pred_proba_resampled)
                        ]
                    }
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    comparison_df['Improvement'] = ((comparison_df['Resampled Data'] - comparison_df['Original Data']) / 
                                                   comparison_df['Original Data'] * 100).round(2)
                    
                    st.dataframe(comparison_df.style.format({
                        'Original Data': '{:.4f}',
                        'Resampled Data': '{:.4f}',
                        'Improvement': '{:+.2f}%'
                    }).background_gradient(subset=['Improvement'], cmap='RdYlGn', vmin=-10, vmax=10))
                    
                    # Visual comparison
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Original Data - Confusion Matrix**")
                        cm_original = confusion_matrix(y_test, y_pred_original)
                        fig, ax = plt.subplots(figsize=(5, 4))
                        sns.heatmap(cm_original, annot=True, fmt='d', cmap='Blues', ax=ax)
                        ax.set_title('Original Data')
                        ax.set_xlabel('Predicted')
                        ax.set_ylabel('Actual')
                        st.pyplot(fig)
                    
                    with col2:
                        st.write("**Resampled Data - Confusion Matrix**")
                        cm_resampled = confusion_matrix(y_test, y_pred_resampled)
                        fig, ax = plt.subplots(figsize=(5, 4))
                        sns.heatmap(cm_resampled, annot=True, fmt='d', cmap='Greens', ax=ax)
                        ax.set_title('Resampled Data')
                        ax.set_xlabel('Predicted')
                        ax.set_ylabel('Actual')
                        st.pyplot(fig)
                    
                    # Store results
                    st.session_state.balanced_models = {
                        'original': rf_original,
                        'resampled': rf_resampled,
                        'technique': technique
                    }
                    st.session_state.balanced_results = comparison_df
                    
                    st.success("✅ Resampling complete! Check the performance comparison above.")
                    
                except Exception as e:
                    st.error(f"Error during resampling: {str(e)}")
        
        # Show previous results if available
        if st.session_state.balanced_results is not None:
            st.divider()
            st.subheader("📋 Previous Results")
            st.write(f"**Technique Used:** {st.session_state.balanced_models['technique']}")
            st.dataframe(st.session_state.balanced_results)

# ============================================================================
# SECTION 7: MODEL COMPARISON
# ============================================================================
elif section == "Model Comparison":
    st.header("🏆 Multi-Model Comparison Dashboard")
    st.write("Train and compare multiple ML models to find the best performer!")
    
    if st.session_state.df_cleaned is None:
        st.warning("Please clean the data first from the 'Data Cleaning' section.")
    elif 'X_train' not in st.session_state:
        st.warning("Please complete the Model Training section first to prepare the data.")
    else:
        X_train = st.session_state.X_train
        X_test = st.session_state.X_test
        y_train = st.session_state.y_train
        y_test = st.session_state.y_test
        
        st.subheader("📋 Select Models to Compare")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            train_rf = st.checkbox("Random Forest", value=True)
        with col2:
            train_gb = st.checkbox("Gradient Boosting", value=True)
        with col3:
            train_lr = st.checkbox("Logistic Regression", value=True)
        with col4:
            train_dt = st.checkbox("Decision Tree", value=False)
        
        if st.button("🚀 Train All Selected Models"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            models_to_train = []
            if train_rf:
                models_to_train.append(('Random Forest', RandomForestClassifier(
                    n_estimators=100, max_depth=10, min_samples_split=20,
                    min_samples_leaf=10, max_features='sqrt', random_state=42, n_jobs=-1
                )))
            if train_gb:
                models_to_train.append(('Gradient Boosting', GradientBoostingClassifier(
                    n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
                )))
            if train_lr:
                models_to_train.append(('Logistic Regression', LogisticRegression(
                    max_iter=1000, random_state=42, n_jobs=-1
                )))
            if train_dt:
                models_to_train.append(('Decision Tree', DecisionTreeClassifier(
                    max_depth=10, min_samples_split=20, random_state=42
                )))
            
            total_models = len(models_to_train)
            
            for idx, (name, model) in enumerate(models_to_train):
                status_text.text(f"Training {name}... ({idx+1}/{total_models})")
                
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Store results
                st.session_state.all_models[name] = model
                st.session_state.model_results[name] = {
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba,
                    'roc_auc': roc_auc_score(y_test, y_pred_proba),
                    'f1': f1_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred),
                    'recall': recall_score(y_test, y_pred),
                    'avg_precision': average_precision_score(y_test, y_pred_proba)
                }
                
                progress_bar.progress((idx + 1) / total_models)
            
            status_text.text("✅ All models trained successfully!")
            st.success(f"Successfully trained {total_models} models!")
        
        # Display results if models have been trained
        if st.session_state.model_results:
            st.markdown("---")
            st.subheader("📊 Performance Comparison")
            
            # Create comparison DataFrame
            results_df = pd.DataFrame({
                'Model': list(st.session_state.model_results.keys()),
                'ROC AUC': [r['roc_auc'] for r in st.session_state.model_results.values()],
                'F1 Score': [r['f1'] for r in st.session_state.model_results.values()],
                'Precision': [r['precision'] for r in st.session_state.model_results.values()],
                'Recall': [r['recall'] for r in st.session_state.model_results.values()],
                'Avg Precision': [r['avg_precision'] for r in st.session_state.model_results.values()]
            }).sort_values('ROC AUC', ascending=False)
            
            # Display metrics table with styling
            st.dataframe(
                results_df.style.background_gradient(cmap='RdYlGn', subset=['ROC AUC', 'F1 Score', 'Precision', 'Recall', 'Avg Precision'])
                .format({
                    'ROC AUC': '{:.4f}',
                    'F1 Score': '{:.4f}',
                    'Precision': '{:.4f}',
                    'Recall': '{:.4f}',
                    'Avg Precision': '{:.4f}'
                }),
                use_container_width=True
            )
            
            # Highlight best model
            best_model = results_df.iloc[0]['Model']
            st.success(f"🏆 **Best Model:** {best_model} with ROC AUC = {results_df.iloc[0]['ROC AUC']:.4f}")
            
            # Visual comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("ROC AUC Comparison")
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(results_df))]
                bars = ax.barh(results_df['Model'], results_df['ROC AUC'], color=colors, edgecolor='black', linewidth=2)
                ax.set_xlabel('ROC AUC Score', fontsize=12, fontweight='bold')
                ax.set_title('Model Comparison - ROC AUC', fontsize=14, fontweight='bold')
                ax.grid(axis='x', alpha=0.3, linestyle='--')
                
                # Add value labels
                for i, (bar, val) in enumerate(zip(bars, results_df['ROC AUC'])):
                    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.4f}',
                           va='center', fontweight='bold', fontsize=10)
                
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                st.subheader("F1 Score Comparison")
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = ['#2ecc71' if i == 0 else '#e74c3c' for i in range(len(results_df))]
                bars = ax.barh(results_df['Model'], results_df['F1 Score'], color=colors, edgecolor='black', linewidth=2)
                ax.set_xlabel('F1 Score', fontsize=12, fontweight='bold')
                ax.set_title('Model Comparison - F1 Score', fontsize=14, fontweight='bold')
                ax.grid(axis='x', alpha=0.3, linestyle='--')
                
                # Add value labels
                for i, (bar, val) in enumerate(zip(bars, results_df['F1 Score'])):
                    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.4f}',
                           va='center', fontweight='bold', fontsize=10)
                
                plt.tight_layout()
                st.pyplot(fig)
            
            # ROC Curves Comparison
            st.subheader("📈 ROC Curves - All Models")
            fig, ax = plt.subplots(figsize=(12, 8))
            
            colors_map = {
                'Random Forest': '#2ecc71',
                'Gradient Boosting': '#3498db',
                'Logistic Regression': '#e74c3c',
                'Decision Tree': '#f39c12'
            }
            
            for name, results in st.session_state.model_results.items():
                from sklearn.metrics import roc_curve
                fpr, tpr, _ = roc_curve(y_test, results['y_pred_proba'])
                ax.plot(fpr, tpr, label=f"{name} (AUC = {results['roc_auc']:.4f})",
                       linewidth=3, color=colors_map.get(name, '#95a5a6'))
            
            ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
            ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
            ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
            ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
            ax.legend(loc='lower right', fontsize=11)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Precision-Recall Curves Comparison
            st.subheader("📉 Precision-Recall Curves - All Models")
            fig, ax = plt.subplots(figsize=(12, 8))
            
            for name, results in st.session_state.model_results.items():
                precision, recall, _ = precision_recall_curve(y_test, results['y_pred_proba'])
                ax.plot(recall, precision, label=f"{name} (AP = {results['avg_precision']:.4f})",
                       linewidth=3, color=colors_map.get(name, '#95a5a6'))
            
            ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
            ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
            ax.set_title('Precision-Recall Curves Comparison', fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=11)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Detailed Metrics Radar Chart
            st.subheader("🎯 Multi-Metric Performance Radar")
            
            # Create radar chart data
            metrics = ['ROC AUC', 'F1 Score', 'Precision', 'Recall', 'Avg Precision']
            
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]
            
            for name, results in st.session_state.model_results.items():
                values = [
                    results['roc_auc'],
                    results['f1'],
                    results['precision'],
                    results['recall'],
                    results['avg_precision']
                ]
                values += values[:1]
                
                ax.plot(angles, values, 'o-', linewidth=2, label=name, 
                       color=colors_map.get(name, '#95a5a6'))
                ax.fill(angles, values, alpha=0.15, color=colors_map.get(name, '#95a5a6'))
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics, fontsize=11, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_title('Multi-Metric Performance Comparison', fontsize=14, fontweight='bold', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
            ax.grid(True)
            
            plt.tight_layout()
            st.pyplot(fig)

# ============================================================================
# SECTION 8: MODEL EVALUATION (Enhanced)
# ============================================================================
elif section == "Model Evaluation":
    st.header("📊 Detailed Model Evaluation")
    
    if st.session_state.model is None:
        st.warning("Please train the model first from the 'Model Training' section.")
    else:
        rf_model = st.session_state.model
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        y_pred = st.session_state.y_pred
        y_pred_proba = st.session_state.y_pred_proba
        features = st.session_state.features
        
        # Classification Report
        st.subheader("Classification Report")
        # Classification Report
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, target_names=['DNF', 'Finished'])
        st.text(report)
        
        # Enhanced Metrics Display
        st.subheader("📈 Key Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("ROC AUC", f"{roc_auc_score(y_test, y_pred_proba):.4f}")
        with col2:
            st.metric("F1 Score", f"{f1_score(y_test, y_pred):.4f}")
        with col3:
            st.metric("Precision", f"{precision_score(y_test, y_pred):.4f}")
        with col4:
            st.metric("Recall", f"{recall_score(y_test, y_pred):.4f}")
        
        # Confusion Matrix with enhanced visualization
        st.subheader("Confusion Matrix")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['DNF', 'Finished'], 
                        yticklabels=['DNF', 'Finished'], ax=ax)
            ax.set_title('Confusion Matrix', fontweight='bold')
            ax.set_ylabel('Actual', fontweight='bold')
            ax.set_xlabel('Predicted', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            # Normalized confusion matrix
            fig, ax = plt.subplots(figsize=(8, 6))
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens', 
                        xticklabels=['DNF', 'Finished'], 
                        yticklabels=['DNF', 'Finished'], ax=ax)
            ax.set_title('Normalized Confusion Matrix', fontweight='bold')
            ax.set_ylabel('Actual', fontweight='bold')
            ax.set_xlabel('Predicted', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
        
        # ROC and PR Curves side by side
        st.subheader("📊 ROC & Precision-Recall Curves")
        col1, col2 = st.columns(2)
        
        with col1:
            # ROC Curve
            fig, ax = plt.subplots(figsize=(8, 6))
            RocCurveDisplay.from_estimator(rf_model, X_test, y_test, ax=ax, 
                                          color='#2ecc71', linewidth=3)
            ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
            ax.set_title('ROC Curve', fontsize=12, fontweight='bold')
            ax.legend(loc='lower right')
            ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            # Precision-Recall Curve
            fig, ax = plt.subplots(figsize=(8, 6))
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            avg_precision = average_precision_score(y_test, y_pred_proba)
            
            ax.plot(recall, precision, linewidth=3, color='#e74c3c', 
                   label=f'AP = {avg_precision:.4f}')
            ax.set_xlabel('Recall', fontsize=11, fontweight='bold')
            ax.set_ylabel('Precision', fontsize=11, fontweight='bold')
            ax.set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        
        # Feature Importance
        st.subheader("🎯 Feature Importance Analysis")
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(feature_importance))]
            bars = ax.barh(feature_importance['feature'], feature_importance['importance'], 
                          color=colors, edgecolor='black', linewidth=2)
            ax.set_xlabel('Importance', fontsize=11, fontweight='bold')
            ax.set_title('Feature Importance', fontsize=12, fontweight='bold')
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            
            # Add value labels
            for bar, val in zip(bars, feature_importance['importance']):
                ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.4f}',
                       va='center', fontweight='bold', fontsize=9)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.write("**Feature Importance Table:**")
            st.dataframe(
                feature_importance.style.background_gradient(cmap='Greens', subset=['importance'])
                .format({'importance': '{:.4f}'}),
                use_container_width=True
            )
        
        # Prediction Distribution
        st.subheader("📉 Prediction Probability Distribution")
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        
        # Distribution by actual class
        dnf_probs = y_pred_proba[y_test == 0]
        finished_probs = y_pred_proba[y_test == 1]
        
        axes[0].hist(dnf_probs, bins=50, alpha=0.7, color='#e74c3c', edgecolor='black', label='Actual DNF')
        axes[0].hist(finished_probs, bins=50, alpha=0.7, color='#2ecc71', edgecolor='black', label='Actual Finished')
        axes[0].set_xlabel('Predicted Probability of Finishing', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[0].set_title('Probability Distribution by Actual Class', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Box plot
        axes[1].boxplot([dnf_probs, finished_probs], labels=['Actual DNF', 'Actual Finished'],
                       patch_artist=True,
                       boxprops=dict(facecolor='lightblue', color='black', linewidth=2),
                       medianprops=dict(color='red', linewidth=2),
                       whiskerprops=dict(color='black', linewidth=1.5),
                       capprops=dict(color='black', linewidth=1.5))
        axes[1].set_ylabel('Predicted Probability of Finishing', fontsize=11, fontweight='bold')
        axes[1].set_title('Probability Distribution Box Plot', fontsize=12, fontweight='bold')
        axes[1].grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        st.pyplot(fig)

# ============================================================================
# SECTION 9: LEARNING CURVES
# ============================================================================
elif section == "Learning Curves":
    st.header("📚 Learning Curves Analysis")
    st.write("Visualize model performance as a function of training set size to diagnose bias/variance issues.")
    
    if st.session_state.df_cleaned is None:
        st.warning("Please clean the data first from the 'Data Cleaning' section.")
    elif 'X_train' not in st.session_state:
        st.warning("Please complete the Model Training section first to prepare the data.")
    else:
        X_train = st.session_state.X_train
        X_test = st.session_state.X_test
        y_train = st.session_state.y_train
        y_test = st.session_state.y_test
        
        st.subheader("🎯 Model Selection")
        
        model_choice = st.selectbox(
            "Select Model for Learning Curve Analysis:",
            ["Random Forest", "Gradient Boosting", "Logistic Regression", "Decision Tree"]
        )
        
        # Model configuration
        models_config = {
            "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42)
        }
        
        col1, col2 = st.columns(2)
        with col1:
            cv_folds = st.slider("Cross-Validation Folds:", 3, 10, 5)
        with col2:
            n_jobs = st.slider("Parallel Jobs:", 1, 4, -1)
        
        if st.button("📈 Generate Learning Curves"):
            with st.spinner(f"Generating learning curves for {model_choice}..."):
                try:
                    model = models_config[model_choice]
                    
                    # Define training sizes
                    train_sizes = np.linspace(0.1, 1.0, 10)
                    
                    # Compute learning curves
                    train_sizes_abs, train_scores, test_scores = learning_curve(
                        model, X_train, y_train,
                        cv=cv_folds,
                        n_jobs=n_jobs,
                        train_sizes=train_sizes,
                        scoring='roc_auc',
                        random_state=42
                    )
                    
                    # Calculate mean and std
                    train_mean = np.mean(train_scores, axis=1)
                    train_std = np.std(train_scores, axis=1)
                    test_mean = np.mean(test_scores, axis=1)
                    test_std = np.std(test_scores, axis=1)
                    
                    # Plot learning curves
                    st.subheader("📊 Learning Curve Visualization")
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Plot training scores
                    ax.plot(train_sizes_abs, train_mean, 'o-', color='#FF6B6B', 
                           label='Training Score', linewidth=2)
                    ax.fill_between(train_sizes_abs, 
                                   train_mean - train_std,
                                   train_mean + train_std,
                                   alpha=0.2, color='#FF6B6B')
                    
                    # Plot validation scores
                    ax.plot(train_sizes_abs, test_mean, 'o-', color='#4ECDC4',
                           label='Cross-Validation Score', linewidth=2)
                    ax.fill_between(train_sizes_abs,
                                   test_mean - test_std,
                                   test_mean + test_std,
                                   alpha=0.2, color='#4ECDC4')
                    
                    ax.set_xlabel('Training Set Size', fontsize=12)
                    ax.set_ylabel('ROC AUC Score', fontsize=12)
                    ax.set_title(f'Learning Curves - {model_choice}', fontsize=14, fontweight='bold')
                    ax.legend(loc='lower right', fontsize=10)
                    ax.grid(True, alpha=0.3)
                    ax.set_ylim([0.5, 1.05])
                    
                    st.pyplot(fig)
                    
                    # Interpretation
                    st.divider()
                    st.subheader("🔍 Interpretation Guide")
                    
                    final_gap = train_mean[-1] - test_mean[-1]
                    final_cv_score = test_mean[-1]
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Final Training Score", f"{train_mean[-1]:.4f}")
                    with col2:
                        st.metric("Final CV Score", f"{test_mean[-1]:.4f}")
                    with col3:
                        st.metric("Training-CV Gap", f"{final_gap:.4f}")
                    
                    st.markdown("#### 📋 Diagnostic Analysis:")
                    
                    # High bias diagnosis
                    if final_cv_score < 0.75:
                        st.warning("⚠️ **High Bias (Underfitting)**")
                        st.write("- Both training and CV scores are low")
                        st.write("- **Recommendations:**")
                        st.write("  - Use more complex model (increase depth, features)")
                        st.write("  - Add more features or polynomial features")
                        st.write("  - Reduce regularization strength")
                    
                    # High variance diagnosis
                    elif final_gap > 0.1:
                        st.warning("⚠️ **High Variance (Overfitting)**")
                        st.write("- Large gap between training and CV scores")
                        st.write("- **Recommendations:**")
                        st.write("  - Collect more training data")
                        st.write("  - Reduce model complexity")
                        st.write("  - Apply regularization (L1/L2)")
                        st.write("  - Use feature selection")
                    
                    # Good fit
                    else:
                        st.success("✅ **Good Fit**")
                        st.write("- Training and CV scores are high and close")
                        st.write("- Model generalizes well to unseen data")
                        st.write("- **Next Steps:**")
                        st.write("  - Fine-tune hyperparameters")
                        st.write("  - Try ensemble methods")
                        st.write("  - Validate on test set")
                    
                    # Additional metrics table
                    st.divider()
                    st.subheader("📈 Performance at Different Training Sizes")
                    
                    metrics_df = pd.DataFrame({
                        'Training Size': train_sizes_abs,
                        'Training Score (Mean)': train_mean,
                        'Training Score (Std)': train_std,
                        'CV Score (Mean)': test_mean,
                        'CV Score (Std)': test_std,
                        'Gap': train_mean - test_mean
                    })
                    
                    st.dataframe(metrics_df.style.format({
                        'Training Size': '{:.0f}',
                        'Training Score (Mean)': '{:.4f}',
                        'Training Score (Std)': '{:.4f}',
                        'CV Score (Mean)': '{:.4f}',
                        'CV Score (Std)': '{:.4f}',
                        'Gap': '{:.4f}'
                    }).background_gradient(subset=['CV Score (Mean)'], cmap='RdYlGn', vmin=0.5, vmax=1.0))
                    
                    # Convergence analysis
                    st.divider()
                    st.subheader("🎯 Convergence Analysis")
                    
                    # Check if adding more data would help
                    score_improvement = test_mean[-1] - test_mean[-3]
                    
                    if score_improvement > 0.01:
                        st.info("📊 **More Data Recommended**")
                        st.write(f"CV score improved by {score_improvement:.4f} in recent iterations.")
                        st.write("Collecting more training data may further improve performance.")
                    else:
                        st.success("✅ **Performance Converged**")
                        st.write("Model performance has plateaued. Additional data may not significantly help.")
                        st.write("Focus on feature engineering or model selection instead.")
                    
                except Exception as e:
                    st.error(f"Error generating learning curves: {str(e)}")

# ============================================================================
# SECTION 10: HYPERPARAMETER TUNING
# ============================================================================
elif section == "Hyperparameter Tuning":
    st.header("🎛️ Advanced Hyperparameter Tuning")
    st.write("Optimize model performance using GridSearchCV and RandomizedSearchCV")
    
    if st.session_state.df_cleaned is None:
        st.warning("Please clean the data first from the 'Data Cleaning' section.")
    elif 'X_train' not in st.session_state:
        st.warning("Please complete the Model Training section first to prepare the data.")
    else:
        X_train = st.session_state.X_train
        X_test = st.session_state.X_test
        y_train = st.session_state.y_train
        y_test = st.session_state.y_test
        
        st.subheader("🎯 Model & Search Strategy Selection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_choice = st.selectbox(
                "Select Model:",
                ["Random Forest", "Gradient Boosting", "Logistic Regression"]
            )
        
        with col2:
            search_method = st.selectbox(
                "Search Method:",
                ["Grid Search (Exhaustive)", "Randomized Search (Faster)"]
            )
        
        # Define parameter grids
        param_grids = {
            "Random Forest": {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 5, 10],
                'max_features': ['sqrt', 'log2', None]
            },
            "Gradient Boosting": {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7, 10],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 5],
                'subsample': [0.8, 0.9, 1.0]
            },
            "Logistic Regression": {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['liblinear', 'saga'],
                'max_iter': [1000, 2000, 5000]
            }
        }
        
        # Show parameter grid
        with st.expander("📋 Parameter Grid"):
            st.write(f"**Parameters to tune for {model_choice}:**")
            st.json(param_grids[model_choice])
            
            total_combinations = 1
            for param_values in param_grids[model_choice].values():
                total_combinations *= len(param_values)
            st.info(f"Total combinations: **{total_combinations}**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            cv_folds = st.slider("CV Folds:", 3, 10, 5)
        with col2:
            n_iter = st.slider("Random Iterations:", 10, 100, 50)
        with col3:
            scoring_metric = st.selectbox("Scoring:", ["roc_auc", "f1", "precision", "recall"])
        
        if st.button("🚀 Start Hyperparameter Tuning"):
            with st.spinner(f"Tuning {model_choice} using {search_method}..."):
                try:
                    # Select base model
                    if model_choice == "Random Forest":
                        base_model = RandomForestClassifier(random_state=42)
                    elif model_choice == "Gradient Boosting":
                        base_model = GradientBoostingClassifier(random_state=42)
                    else:
                        base_model = LogisticRegression(random_state=42)
                    
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Perform search
                    if search_method == "Grid Search (Exhaustive)":
                        status_text.text("Performing Grid Search...")
                        search = GridSearchCV(
                            base_model,
                            param_grids[model_choice],
                            cv=cv_folds,
                            scoring=scoring_metric,
                            n_jobs=-1,
                            verbose=1
                        )
                    else:
                        from sklearn.model_selection import RandomizedSearchCV
                        status_text.text("Performing Randomized Search...")
                        search = RandomizedSearchCV(
                            base_model,
                            param_grids[model_choice],
                            n_iter=n_iter,
                            cv=cv_folds,
                            scoring=scoring_metric,
                            n_jobs=-1,
                            random_state=42,
                            verbose=1
                        )
                    
                    progress_bar.progress(30)
                    
                    # Fit search
                    search.fit(X_train, y_train)
                    progress_bar.progress(80)
                    
                    # Get best model
                    best_model = search.best_estimator_
                    best_params = search.best_params_
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Tuning Complete!")
                    
                    # Display results
                    st.success(f"✅ Hyperparameter tuning complete!")
                    
                    st.divider()
                    st.subheader("🏆 Best Parameters Found")
                    
                    # Format best parameters
                    params_df = pd.DataFrame([best_params]).T
                    params_df.columns = ['Value']
                    st.dataframe(params_df.style.set_properties(**{'background-color': '#e8f5e9'}))
                    
                    # Performance comparison
                    st.divider()
                    st.subheader("📊 Performance Comparison")
                    
                    # Default model
                    if model_choice == "Random Forest":
                        default_model = RandomForestClassifier(random_state=42)
                    elif model_choice == "Gradient Boosting":
                        default_model = GradientBoostingClassifier(random_state=42)
                    else:
                        default_model = LogisticRegression(random_state=42)
                    
                    default_model.fit(X_train, y_train)
                    
                    # Predictions
                    y_pred_default = default_model.predict(X_test)
                    y_pred_tuned = best_model.predict(X_test)
                    y_pred_proba_default = default_model.predict_proba(X_test)[:, 1]
                    y_pred_proba_tuned = best_model.predict_proba(X_test)[:, 1]
                    
                    # Metrics
                    comparison_data = {
                        'Metric': ['ROC AUC', 'F1 Score', 'Precision', 'Recall', 'Accuracy'],
                        'Default Model': [
                            roc_auc_score(y_test, y_pred_proba_default),
                            f1_score(y_test, y_pred_default),
                            precision_score(y_test, y_pred_default),
                            recall_score(y_test, y_pred_default),
                            (y_test == y_pred_default).mean()
                        ],
                        'Tuned Model': [
                            roc_auc_score(y_test, y_pred_proba_tuned),
                            f1_score(y_test, y_pred_tuned),
                            precision_score(y_test, y_pred_tuned),
                            recall_score(y_test, y_pred_tuned),
                            (y_test == y_pred_tuned).mean()
                        ]
                    }
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    comparison_df['Improvement'] = ((comparison_df['Tuned Model'] - comparison_df['Default Model']) / 
                                                   comparison_df['Default Model'] * 100)
                    
                    st.dataframe(comparison_df.style.format({
                        'Default Model': '{:.4f}',
                        'Tuned Model': '{:.4f}',
                        'Improvement': '{:+.2f}%'
                    }).background_gradient(subset=['Improvement'], cmap='RdYlGn', vmin=-5, vmax=5))
                    
                    # CV results visualization
                    st.divider()
                    st.subheader("📈 Cross-Validation Results")
                    
                    cv_results = pd.DataFrame(search.cv_results_)
                    top_10 = cv_results.nlargest(10, 'mean_test_score')[['mean_test_score', 'std_test_score', 'rank_test_score']]
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    x_pos = np.arange(len(top_10))
                    ax.bar(x_pos, top_10['mean_test_score'], 
                          yerr=top_10['std_test_score'],
                          color='#4ECDC4', alpha=0.8, capsize=5)
                    ax.set_xlabel('Configuration Rank', fontsize=12)
                    ax.set_ylabel(f'{scoring_metric} Score', fontsize=12)
                    ax.set_title('Top 10 Configurations', fontsize=14, fontweight='bold')
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels([f"#{int(r)}" for r in top_10['rank_test_score']])
                    ax.grid(True, alpha=0.3, axis='y')
                    st.pyplot(fig)
                    
                    # Store best model
                    st.session_state.best_model = best_model
                    st.session_state.best_params = best_params
                    
                    # Download best parameters
                    st.divider()
                    params_json = pd.DataFrame([best_params]).to_json()
                    st.download_button(
                        label="📥 Download Best Parameters",
                        data=params_json,
                        file_name=f"best_params_{model_choice.replace(' ', '_')}.json",
                        mime="application/json"
                    )
                    
                except Exception as e:
                    st.error(f"Error during hyperparameter tuning: {str(e)}")
                    progress_bar.empty()
                    status_text.empty()

# ============================================================================
# SECTION 11: FEATURE IMPORTANCE ANALYSIS
# ============================================================================
elif section == "Feature Importance":
    st.header("🎯 Feature Importance Analysis")
    st.write("Understand which features contribute most to model predictions")
    
    if st.session_state.df_cleaned is None:
        st.warning("Please clean the data first from the 'Data Cleaning' section.")
    elif 'X_train' not in st.session_state:
        st.warning("Please complete the Model Training section first to prepare the data.")
    else:
        X_train = st.session_state.X_train
        X_test = st.session_state.X_test
        y_train = st.session_state.y_train
        y_test = st.session_state.y_test
        features = st.session_state.features
        
        st.subheader("📊 Feature Importance Methods")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_choice = st.selectbox(
                "Select Model:",
                ["Random Forest", "Gradient Boosting", "Decision Tree"]
            )
        
        with col2:
            importance_type = st.selectbox(
                "Importance Method:",
                ["Built-in Feature Importance", "Permutation Importance"]
            )
        
        if st.button("🔍 Analyze Feature Importance"):
            with st.spinner(f"Analyzing features using {importance_type}..."):
                try:
                    # Train model
                    if model_choice == "Random Forest":
                        model = RandomForestClassifier(n_estimators=100, random_state=42)
                    elif model_choice == "Gradient Boosting":
                        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
                    else:
                        model = DecisionTreeClassifier(max_depth=10, random_state=42)
                    
                    model.fit(X_train, y_train)
                    
                    # Get feature importance
                    if importance_type == "Built-in Feature Importance":
                        importances = model.feature_importances_
                        method_desc = "Based on decrease in node impurity (Gini importance)"
                    else:
                        perm_importance = permutation_importance(
                            model, X_test, y_test,
                            n_repeats=10,
                            random_state=42,
                            n_jobs=-1
                        )
                        importances = perm_importance.importances_mean
                        method_desc = "Based on decrease in model score when feature values are randomly shuffled"
                    
                    # Create importance dataframe
                    importance_df = pd.DataFrame({
                        'Feature': features,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False)
                    
                    st.success("✅ Feature importance analysis complete!")
                    
                    st.divider()
                    st.subheader("📈 Feature Importance Rankings")
                    st.info(f"**Method:** {method_desc}")
                    
                    # Display top features
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Bar plot
                        fig, ax = plt.subplots(figsize=(10, 6))
                        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(importance_df)))
                        ax.barh(importance_df['Feature'], importance_df['Importance'], color=colors)
                        ax.set_xlabel('Importance Score', fontsize=12)
                        ax.set_ylabel('Feature', fontsize=12)
                        ax.set_title(f'Feature Importance - {model_choice}', fontsize=14, fontweight='bold')
                        ax.invert_yaxis()
                        ax.grid(True, alpha=0.3, axis='x')
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    with col2:
                        # Top 5 features
                        st.markdown("**🏆 Top 5 Features:**")
                        for idx, row in importance_df.head(5).iterrows():
                            st.metric(
                                label=row['Feature'],
                                value=f"{row['Importance']:.4f}",
                                delta=f"Rank #{idx+1}"
                            )
                    
                    # Feature importance table
                    st.divider()
                    st.subheader("📋 Complete Feature Rankings")
                    
                    importance_df['Percentage'] = (importance_df['Importance'] / importance_df['Importance'].sum() * 100)
                    importance_df['Cumulative %'] = importance_df['Percentage'].cumsum()
                    importance_df['Rank'] = range(1, len(importance_df) + 1)
                    
                    st.dataframe(
                        importance_df[['Rank', 'Feature', 'Importance', 'Percentage', 'Cumulative %']].style.format({
                            'Importance': '{:.4f}',
                            'Percentage': '{:.2f}%',
                            'Cumulative %': '{:.2f}%'
                        }).background_gradient(subset=['Importance'], cmap='YlGn')
                    )
                    
                    # Cumulative importance plot
                    st.divider()
                    st.subheader("📊 Cumulative Importance Analysis")
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(range(1, len(importance_df) + 1), importance_df['Cumulative %'], 
                           marker='o', linewidth=2, color='#FF6B6B')
                    ax.axhline(y=80, color='green', linestyle='--', label='80% Threshold')
                    ax.axhline(y=90, color='orange', linestyle='--', label='90% Threshold')
                    ax.set_xlabel('Number of Features', fontsize=12)
                    ax.set_ylabel('Cumulative Importance (%)', fontsize=12)
                    ax.set_title('Cumulative Feature Importance', fontsize=14, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Feature selection recommendation
                    features_for_80 = (importance_df['Cumulative %'] <= 80).sum()
                    features_for_90 = (importance_df['Cumulative %'] <= 90).sum()
                    
                    st.info(f"""
                    **💡 Feature Selection Recommendations:**
                    - Use **{features_for_80}** features to capture 80% of importance
                    - Use **{features_for_90}** features to capture 90% of importance
                    - Total features: **{len(features)}**
                    """)
                    
                    # Store results
                    st.session_state.feature_importances = importance_df
                    
                except Exception as e:
                    st.error(f"Error analyzing feature importance: {str(e)}")

# ============================================================================
# SECTION 12: LIVE PREDICTION
# ============================================================================
elif section == "Live Prediction":
    st.header("🎯 Interactive DNF Prediction Tool")
    st.write("Enter race parameters to predict DNF probability in real-time!")
    
    if st.session_state.model is None:
        st.warning("⚠️ Please train a model first from the 'Model Training' section.")
    else:
        st.markdown("---")
        
        # Check if we have encoded dataframe to get value ranges
        if st.session_state.df_encoded is not None:
            df = st.session_state.df_encoded
            
            st.subheader("🏁 Race Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📍 Position & Circuit")
                
                # Grid position
                grid_min = int(df['grid'].min())
                grid_max = int(df['grid'].max())
                grid = st.slider("Starting Grid Position", 
                               min_value=grid_min, 
                               max_value=grid_max, 
                               value=10,
                               help="Lower numbers = better starting position")
                
                # Circuit
                circuit_min = float(df['circuitId_encoded'].min())
                circuit_max = float(df['circuitId_encoded'].max())
                circuit_encoded = st.slider("Circuit Reliability Score", 
                                           min_value=circuit_min, 
                                           max_value=circuit_max, 
                                           value=(circuit_min + circuit_max) / 2,
                                           help="Historical finish rate for this circuit")
            
            with col2:
                st.markdown("#### 👤 Driver & Team")
                
                # Driver
                driver_min = float(df['driverRef_encoded'].min())
                driver_max = float(df['driverRef_encoded'].max())
                driver_encoded = st.slider("Driver Reliability Score", 
                                          min_value=driver_min, 
                                          max_value=driver_max, 
                                          value=(driver_min + driver_max) / 2,
                                          help="Historical finish rate for this driver")
                
                # Constructor
                constructor_min = float(df['constructorRef_encoded'].min())
                constructor_max = float(df['constructorRef_encoded'].max())
                constructor_encoded = st.slider("Team Reliability Score", 
                                               min_value=constructor_min, 
                                               max_value=constructor_max, 
                                               value=(constructor_min + constructor_max) / 2,
                                               help="Historical finish rate for this team")
            
            st.markdown("#### 🗓️ Race Context")
            race_min = float(df['raceId_encoded'].min())
            race_max = float(df['raceId_encoded'].max())
            race_encoded = st.slider("Race Difficulty Score", 
                                    min_value=race_min, 
                                    max_value=race_max, 
                                    value=(race_min + race_max) / 2,
                                    help="Historical finish rate for similar races")
            
            st.markdown("---")
            
            # Predict button
            if st.button("🚀 Predict DNF Probability", type="primary", use_container_width=True):
                # Prepare input
                input_data = pd.DataFrame({
                    'grid': [grid],
                    'driverRef_encoded': [driver_encoded],
                    'constructorRef_encoded': [constructor_encoded],
                    'circuitId_encoded': [circuit_encoded],
                    'raceId_encoded': [race_encoded]
                })
                
                # Make prediction
                model = st.session_state.model
                prediction = model.predict(input_data)[0]
                probability = model.predict_proba(input_data)[0]
                
                dnf_prob = probability[0]
                finish_prob = probability[1]
                
                # Display results
                st.markdown("---")
                st.subheader("🎯 Prediction Results")
                
                # Main prediction
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col2:
                    if prediction == 1:
                        st.success("### ✅ WILL FINISH THE RACE")
                        st.metric("Finish Probability", f"{finish_prob*100:.2f}%", 
                                 delta=f"{(finish_prob - dnf_prob)*100:.1f}% advantage")
                    else:
                        st.error("### ❌ LIKELY TO DNF")
                        st.metric("DNF Probability", f"{dnf_prob*100:.2f}%",
                                 delta=f"{(dnf_prob - finish_prob)*100:.1f}% risk", delta_color="inverse")
                
                # Probability breakdown
                st.markdown("#### 📊 Probability Breakdown")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("🏁 Finish Probability", f"{finish_prob*100:.2f}%")
                with col2:
                    st.metric("💥 DNF Probability", f"{dnf_prob*100:.2f}%")
                
                # Visual probability gauge
                fig, ax = plt.subplots(figsize=(10, 2))
                ax.barh([0], [finish_prob], color='#2ecc71', height=0.5, label='Finish')
                ax.barh([0], [dnf_prob], left=[finish_prob], color='#e74c3c', height=0.5, label='DNF')
                ax.set_xlim(0, 1)
                ax.set_ylim(-0.5, 0.5)
                ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
                ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
                ax.set_yticks([])
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.5), ncol=2, fontsize=11)
                ax.set_title('Probability Distribution', fontsize=12, fontweight='bold', pad=20)
                
                # Add percentage labels
                if finish_prob > 0.1:
                    ax.text(finish_prob/2, 0, f'{finish_prob*100:.1f}%', 
                           ha='center', va='center', fontweight='bold', fontsize=12, color='white')
                if dnf_prob > 0.1:
                    ax.text(finish_prob + dnf_prob/2, 0, f'{dnf_prob*100:.1f}%', 
                           ha='center', va='center', fontweight='bold', fontsize=12, color='white')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Input summary
                with st.expander("📋 Input Summary"):
                    st.write("**Race Configuration:**")
                    st.write(f"- Starting Grid Position: **{grid}**")
                    st.write(f"- Driver Reliability Score: **{driver_encoded:.3f}**")
                    st.write(f"- Team Reliability Score: **{constructor_encoded:.3f}**")
                    st.write(f"- Circuit Reliability Score: **{circuit_encoded:.3f}**")
                    st.write(f"- Race Difficulty Score: **{race_encoded:.3f}**")
                
                # Confidence indicator
                confidence = abs(finish_prob - dnf_prob)
                st.markdown("#### 🎚️ Prediction Confidence")
                if confidence > 0.7:
                    st.success(f"**Very High Confidence:** {confidence*100:.1f}%")
                elif confidence > 0.5:
                    st.info(f"**High Confidence:** {confidence*100:.1f}%")
                elif confidence > 0.3:
                    st.warning(f"**Moderate Confidence:** {confidence*100:.1f}%")
                else:
                    st.error(f"**Low Confidence:** {confidence*100:.1f}% - Prediction is uncertain")
                
        else:
            st.error("Encoded data not found. Please complete the Model Training section first.")

# Footer
st.markdown("---")
st.markdown("🏎️ **F1 DNF Classification Analysis** | Built with Streamlit")
st.caption("🚀 Comprehensive ML Dashboard: Multi-Model Comparison • Feature Engineering • Hyperparameter Tuning • Advanced Analytics")
