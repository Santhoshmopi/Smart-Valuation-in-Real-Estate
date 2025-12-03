import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import PartialDependenceDisplay
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="NYC Property Price Predictor - Enhanced",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 1.5rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stAlert {
        border-radius: 10px;
    }
    .insight-box {
        background-color: #f8f9fa;
        border-left: 4px solid #3498db;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'metrics' not in st.session_state:
    st.session_state.metrics = {}
if 'cv_scores' not in st.session_state:
    st.session_state.cv_scores = {}
if 'data_processed' not in st.session_state:
    st.session_state.data_processed = False
if 'best_model_name' not in st.session_state:
    st.session_state.best_model_name = None


def remove_outliers_by_segment(df, column='sale_price'):
    """Remove outliers by borough segment for better handling of NYC variance"""
    df_clean = df.copy()
    removed_count = 0

    if 'borough' in df_clean.columns:
        for borough in df_clean['borough'].unique():
            mask = df_clean['borough'] == borough
            Q1 = df_clean.loc[mask, column].quantile(0.25)
            Q3 = df_clean.loc[mask, column].quantile(0.75)
            IQR = Q3 - Q1

            # Use 3 IQR for NYC's extreme variance (luxury properties)
            lower = Q1 - 3 * IQR
            upper = Q3 + 3 * IQR

            before = len(df_clean)
            df_clean = df_clean[~mask | ((df_clean[column] >= lower) & (df_clean[column] <= upper))]
            removed_count += (before - len(df_clean))
    else:
        # Fallback to global outlier removal
        Q1 = df_clean[column].quantile(0.25)
        Q3 = df_clean[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 3 * IQR
        upper = Q3 + 3 * IQR
        before = len(df_clean)
        df_clean = df_clean[(df_clean[column] >= lower) & (df_clean[column] <= upper)]
        removed_count = before - len(df_clean)

    return df_clean, removed_count


def engineer_features(df):
    """Advanced feature engineering for better predictions"""
    df_new = df.copy()

    # 1. Price-related features
    if 'sale_price' in df_new.columns and 'gross_square_feet' in df_new.columns:
        df_new['price_per_sqft'] = df_new['sale_price'] / (df_new['gross_square_feet'] + 1)

    # 2. Age-related features
    if 'year_built' in df_new.columns:
        current_year = datetime.now().year
        df_new['property_age'] = current_year - df_new['year_built']
        df_new['age_category'] = pd.cut(df_new['property_age'],
                                        bins=[0, 10, 30, 50, 100, 200],
                                        labels=['New', 'Modern', 'Established', 'Old', 'Historic'])
        df_new['age_category_encoded'] = LabelEncoder().fit_transform(df_new['age_category'].astype(str))

    # 3. Size ratios
    if 'land_square_feet' in df_new.columns and 'gross_square_feet' in df_new.columns:
        df_new['building_coverage_ratio'] = df_new['gross_square_feet'] / (df_new['land_square_feet'] + 1)

    # 4. Unit density
    if 'total_units' in df_new.columns and 'gross_square_feet' in df_new.columns:
        df_new['units_per_sqft'] = df_new['total_units'] / (df_new['gross_square_feet'] + 1)

    # 5. Commercial ratio
    if 'commercial_units' in df_new.columns and 'total_units' in df_new.columns:
        df_new['commercial_ratio'] = df_new['commercial_units'] / (df_new['total_units'] + 1)

    # 6. Temporal features from sale_date
    if 'sale_date' in df_new.columns:
        df_new['sale_date'] = pd.to_datetime(df_new['sale_date'])
        df_new['sale_year'] = df_new['sale_date'].dt.year
        df_new['sale_month'] = df_new['sale_date'].dt.month
        df_new['sale_quarter'] = df_new['sale_date'].dt.quarter
        df_new['sale_day_of_week'] = df_new['sale_date'].dt.dayofweek
        df_new['is_weekend'] = (df_new['sale_day_of_week'] >= 5).astype(int)

        # Season
        df_new['season'] = df_new['sale_month'].apply(lambda x:
                                                      'Winter' if x in [12, 1, 2] else
                                                      'Spring' if x in [3, 4, 5] else
                                                      'Summer' if x in [6, 7, 8] else 'Fall')
        df_new['season_encoded'] = LabelEncoder().fit_transform(df_new['season'])

    # 7. Borough-Neighborhood interaction
    if 'borough' in df_new.columns and 'neighborhood' in df_new.columns:
        df_new['borough_neighborhood'] = df_new['borough'] + '_' + df_new['neighborhood']
        df_new['borough_neighborhood_encoded'] = LabelEncoder().fit_transform(
            df_new['borough_neighborhood'].astype(str))

    return df_new


def preprocess_data(df):
    """Comprehensive data preprocessing pipeline with advanced features"""
    with st.spinner('🔄 Processing data...'):
        df_clean = df.copy()

        # Remove exact duplicates
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        st.info(f"✓ Removed {initial_rows - len(df_clean)} duplicate rows")

        # Handle missing values for numeric columns
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)

        # Handle missing values for categorical columns
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'Unknown',
                                     inplace=True)

        # Remove invalid sale prices (0 or negative)
        if 'sale_price' in df_clean.columns:
            before_invalid = len(df_clean)
            df_clean = df_clean[df_clean['sale_price'] > 0]
            st.info(f"✓ Removed {before_invalid - len(df_clean)} invalid sale prices")

        # Feature engineering BEFORE outlier removal
        df_clean = engineer_features(df_clean)

        # Remove outliers by borough segment
        if 'sale_price' in df_clean.columns:
            df_clean, removed = remove_outliers_by_segment(df_clean, 'sale_price')
            st.info(f"✓ Removed {removed} price outliers using borough-based IQR method")

        # Encode categorical variables and store encoders
        label_encoders = {}
        categorical_features = ['borough', 'neighborhood', 'building_class_category',
                                'tax_class_at_present', 'building_class_at_present']

        for col in categorical_features:
            if col in df_clean.columns:
                le = LabelEncoder()
                df_clean[f'{col}_encoded'] = le.fit_transform(df_clean[col].astype(str))
                label_encoders[col] = le

        # Store encoders and processed data
        st.session_state.label_encoders = label_encoders
        st.session_state.df_raw = df

        st.success("✅ Data preprocessing completed successfully!")

        # Display preprocessing summary
        col1, col2, col3 = st.columns(3)
        col1.metric("Original Records", f"{initial_rows:,}")
        col2.metric("Processed Records", f"{len(df_clean):,}")
        col3.metric("Features Created", len(df_clean.columns))

        return df_clean


def train_models_with_tuning(X_train, X_test, y_train, y_test, use_tuning=False):
    """Train multiple ML models with optional hyperparameter tuning"""

    if use_tuning:
        st.info("🔧 Hyperparameter tuning enabled - this may take several minutes...")

        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': GridSearchCV(
                RandomForestRegressor(random_state=42, n_jobs=-1),
                param_grid={
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                },
                cv=3,
                scoring='r2',
                n_jobs=-1
            ),
            'Gradient Boosting': GridSearchCV(
                GradientBoostingRegressor(random_state=42),
                param_grid={
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5],
                    'learning_rate': [0.01, 0.1],
                    'subsample': [0.8, 1.0]
                },
                cv=3,
                scoring='r2',
                n_jobs=-1
            ),
            'XGBoost': GridSearchCV(
                XGBRegressor(random_state=42, n_jobs=-1),
                param_grid={
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1],
                    'subsample': [0.8, 1.0]
                },
                cv=3,
                scoring='r2',
                n_jobs=-1
            )
        }
    else:
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            ),
            'XGBoost': XGBRegressor(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
                n_jobs=-1
            )
        }

    trained_models = {}
    metrics = {}
    cv_scores = {}

    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, (name, model) in enumerate(models.items()):
        status_text.text(f"Training {name}... ({idx + 1}/{len(models)})")

        # Train model
        model.fit(X_train, y_train)

        # Store best estimator if GridSearchCV
        if use_tuning and name != 'Linear Regression':
            trained_models[name] = model.best_estimator_
            st.info(f"✓ {name} best params: {model.best_params_}")
        else:
            trained_models[name] = model

        # Cross-validation scores
        if name == 'Linear Regression' or not use_tuning:
            cv_scores_arr = cross_val_score(trained_models[name], X_train, y_train, cv=5, scoring='r2')
            cv_scores[name] = {
                'mean': cv_scores_arr.mean(),
                'std': cv_scores_arr.std()
            }

        # Make predictions
        y_pred = trained_models[name].predict(X_test)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        metrics[name] = {
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'MAPE': mape
        }

        progress_bar.progress((idx + 1) / len(models))

    status_text.empty()
    progress_bar.empty()

    return trained_models, metrics, cv_scores


def display_model_comparison(metrics, cv_scores):
    """Display comprehensive model comparison with enhanced visualizations"""
    st.markdown('<div class="sub-header">📊 Model Performance Comparison</div>', unsafe_allow_html=True)

    # Create metrics DataFrame
    metrics_df = pd.DataFrame(metrics).T

    # Add CV scores if available
    if cv_scores:
        cv_df = pd.DataFrame(cv_scores).T
        cv_df.columns = ['CV Mean R²', 'CV Std R²']
        metrics_df = pd.concat([metrics_df, cv_df], axis=1)

    # Display metrics table
    st.dataframe(
        metrics_df.style
        .highlight_max(axis=0, subset=['R²'], color='lightgreen')
        .highlight_min(axis=0, subset=['RMSE', 'MAE', 'MAPE'], color='lightgreen')
        .format("{:.2f}"),
        use_container_width=True
    )

    # Find best model
    best_model = max(metrics.items(), key=lambda x: x[1]['R²'])[0]
    st.session_state.best_model_name = best_model

    # Performance insights
    best_r2 = metrics[best_model]['R²']
    best_rmse = metrics[best_model]['RMSE']

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🏆 Best Model", best_model)
    with col2:
        st.metric("R² Score", f"{best_r2:.4f}",
                  help="Proportion of variance explained by the model")
    with col3:
        st.metric("RMSE", f"${best_rmse:,.0f}",
                  help="Root Mean Squared Error in dollars")

    # Create comparison visualizations
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('R² Score Comparison', 'RMSE Comparison',
                        'MAE Comparison', 'MAPE Comparison'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'bar'}]]
    )

    model_names = list(metrics.keys())

    # R² Score
    fig.add_trace(
        go.Bar(x=model_names, y=[metrics[m]['R²'] for m in model_names],
               marker_color='lightseagreen', name='R²'),
        row=1, col=1
    )

    # RMSE
    fig.add_trace(
        go.Bar(x=model_names, y=[metrics[m]['RMSE'] for m in model_names],
               marker_color='indianred', name='RMSE'),
        row=1, col=2
    )

    # MAE
    fig.add_trace(
        go.Bar(x=model_names, y=[metrics[m]['MAE'] for m in model_names],
               marker_color='lightcoral', name='MAE'),
        row=2, col=1
    )

    # MAPE
    fig.add_trace(
        go.Bar(x=model_names, y=[metrics[m]['MAPE'] for m in model_names],
               marker_color='lightsalmon', name='MAPE (%)'),
        row=2, col=2
    )

    fig.update_layout(height=700, showlegend=False, title_text="Comprehensive Model Performance Metrics")
    fig.update_yaxes(title_text="R² Score", row=1, col=1)
    fig.update_yaxes(title_text="RMSE ($)", row=1, col=2)
    fig.update_yaxes(title_text="MAE ($)", row=2, col=1)
    fig.update_yaxes(title_text="MAPE (%)", row=2, col=2)

    st.plotly_chart(fig, use_container_width=True)

    # Model interpretation
    st.markdown("### 📈 Performance Interpretation")

    if best_r2 >= 0.80:
        performance_text = "Excellent"
        color = "green"
    elif best_r2 >= 0.70:
        performance_text = "Good"
        color = "blue"
    elif best_r2 >= 0.60:
        performance_text = "Moderate"
        color = "orange"
    else:
        performance_text = "Needs Improvement"
        color = "red"

    st.markdown(f"""
    <div class="insight-box">
    <strong style="color: #2c3e50;">Model Performance: <span style="color: {color};">{performance_text}</span></strong><br>
    <span style="color: #2c3e50;">The best model ({best_model}) explains <strong>{best_r2 * 100:.1f}%</strong> of the variance in property prices.<br>
    Average prediction error: <strong>${best_rmse:,.0f}</strong> (RMSE)<br>
    Mean Absolute Percentage Error: <strong>{metrics[best_model]['MAPE']:.2f}%</strong></span>
    </div>
    """, unsafe_allow_html=True)


def find_similar_properties(df, input_features, n=5):
    """Find similar properties based on input features"""
    try:
        # Calculate similarity score based on key features
        df_copy = df.copy()

        # Normalize key features for comparison
        feature_cols = ['gross_square_feet', 'year_built', 'total_units']
        available_cols = [col for col in feature_cols if col in df_copy.columns]

        if len(available_cols) == 0:
            return pd.DataFrame()

        # Calculate distance (simple Euclidean for demo)
        distances = []
        for idx, row in df_copy.iterrows():
            dist = 0
            for col in available_cols:
                if col in input_features:
                    dist += ((row[col] - input_features[col]) / (df_copy[col].std() + 1)) ** 2
            distances.append(np.sqrt(dist))

        df_copy['similarity_score'] = distances
        similar = df_copy.nsmallest(n, 'similarity_score')

        # Select relevant columns
        display_cols = ['borough', 'neighborhood', 'gross_square_feet', 'year_built',
                        'total_units', 'sale_price', 'sale_date']
        display_cols = [col for col in display_cols if col in similar.columns]

        return similar[display_cols]
    except Exception as e:
        st.warning(f"Could not find similar properties: {e}")
        return pd.DataFrame()


def prediction_interface(models, feature_names, scaler):
    """Enhanced interactive prediction interface"""
    st.markdown('<div class="sub-header">🔮 Property Price Prediction</div>', unsafe_allow_html=True)

    # Model selection
    col_model, col_help = st.columns([3, 1])
    with col_model:
        selected_model = st.selectbox(
            "Select Model for Prediction",
            list(models.keys()),
            index=list(models.keys()).index(st.session_state.best_model_name) if st.session_state.best_model_name else 0
        )

    with col_help:
        if st.session_state.best_model_name:
            st.info(f"🏆 Best: {st.session_state.best_model_name}")

    st.markdown("### 📝 Enter Property Details")

    # Get available options from processed data
    df = st.session_state.df_processed

    # Property Information Section
    with st.expander("🏠 Property Information", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            boroughs = sorted(df['borough'].unique()) if 'borough' in df.columns else ['Bronx', 'Brooklyn', 'Manhattan',
                                                                                       'Queens', 'Staten Island']
            borough = st.selectbox("Borough", boroughs)

            # Filter neighborhoods by borough
            if 'borough' in df.columns and 'neighborhood' in df.columns:
                neighborhoods = sorted(df[df['borough'] == borough]['neighborhood'].unique())
            else:
                neighborhoods = ['Bathgate']
            neighborhood = st.selectbox("Neighborhood", neighborhoods)

        with col2:
            if 'building_class_category' in df.columns:
                building_classes = sorted(df['building_class_category'].unique())
            else:
                building_classes = ['01 One Family Dwellings', '02 Two Family Dwellings']
            building_class = st.selectbox("Building Class Category", building_classes)

            zip_code = st.number_input("Zip Code", min_value=10001, max_value=11697, value=10457)

        with col3:
            year_built = st.number_input("Year Built", min_value=1800, max_value=2024, value=1950)
            current_year = datetime.now().year
            property_age = current_year - year_built
            st.info(f"Property Age: {property_age} years")

    # Size & Units Section
    with st.expander("📏 Size & Units", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            land_sqft = st.number_input("Land Square Feet", min_value=100, max_value=50000, value=2000)
            gross_sqft = st.number_input("Gross Square Feet", min_value=100, max_value=50000, value=1500)

        with col2:
            residential_units = st.number_input("Residential Units", min_value=0, max_value=100, value=1)
            commercial_units = st.number_input("Commercial Units", min_value=0, max_value=50, value=0)
            total_units = residential_units + commercial_units

        with col3:
            st.info(f"Total Units: {total_units}")
            if gross_sqft > 0:
                st.info(f"Building Coverage: {(gross_sqft / land_sqft) * 100:.1f}%")

    # Prediction Button
    if st.button("🎯 Predict Property Price", type="primary", use_container_width=True):
        try:
            # Get label encoders
            label_encoders = st.session_state.get('label_encoders', {})

            # Encode categorical features
            encodings = {}

            for cat_feature, cat_value in [
                ('borough', borough),
                ('neighborhood', neighborhood),
                ('building_class_category', building_class)
            ]:
                if cat_feature in label_encoders:
                    try:
                        encodings[f'{cat_feature}_encoded'] = label_encoders[cat_feature].transform([cat_value])[0]
                    except:
                        encodings[f'{cat_feature}_encoded'] = 0
                else:
                    encodings[f'{cat_feature}_encoded'] = 0

            # Create base features
            base_features = {
                'land_square_feet': land_sqft,
                'gross_square_feet': gross_sqft,
                'year_built': year_built,
                'residential_units': residential_units,
                'commercial_units': commercial_units,
                'total_units': total_units,
                **encodings
            }

            # Add engineered features
            base_features['property_age'] = property_age
            base_features['building_coverage_ratio'] = gross_sqft / (land_sqft + 1)
            base_features['units_per_sqft'] = total_units / (gross_sqft + 1)
            base_features['commercial_ratio'] = commercial_units / (total_units + 1)

            # Build feature vector
            feature_values = []
            for fname in feature_names:
                if fname in base_features:
                    feature_values.append(base_features[fname])
                else:
                    feature_values.append(0)

            features = np.array([feature_values])

            # Scale and predict
            features_scaled = scaler.transform(features)
            prediction = models[selected_model].predict(features_scaled)[0]
            prediction = max(0, prediction)

            # Display Results
            st.markdown("---")
            st.markdown("### 💰 Prediction Results")

            col_a, col_b, col_c, col_d = st.columns(4)

            with col_a:
                st.metric("Predicted Price", f"${prediction:,.0f}")

            with col_b:
                price_per_sqft = prediction / gross_sqft if gross_sqft > 0 else 0
                st.metric("Price per Sq Ft", f"${price_per_sqft:,.2f}")

            with col_c:
                confidence = st.session_state.metrics[selected_model]['R²'] * 100
                st.metric("Model Confidence", f"{confidence:.1f}%")

            with col_d:
                mape = st.session_state.metrics[selected_model]['MAPE']
                st.metric("Avg Error", f"±{mape:.1f}%")

            # Property Summary
            st.markdown("### 📍 Property Summary")
            summary_col1, summary_col2 = st.columns(2)

            with summary_col1:
                st.markdown(f"""
                **Location Details:**
                - Borough: {borough}
                - Neighborhood: {neighborhood}
                - Building Type: {building_class}
                - Zip Code: {zip_code}
                """)

            with summary_col2:
                st.markdown(f"""
                **Property Specifications:**
                - Total Size: {gross_sqft:,} sq ft
                - Land Size: {land_sqft:,} sq ft
                - Year Built: {year_built} (Age: {property_age} years)
                - Total Units: {total_units} ({residential_units} residential, {commercial_units} commercial)
                """)

            # Price Range Estimation
            rmse = st.session_state.metrics[selected_model]['RMSE']
            lower_bound = max(0, prediction - rmse)
            upper_bound = prediction + rmse

            st.markdown("### 📊 Price Range Estimate")
            st.info(
                f"Based on model uncertainty, the property price is likely between **\${lower_bound:,.0f}** and **\${upper_bound:,.0f}**")
            # Similar Properties
            st.markdown("### 🏘️ Similar Properties in Dataset")
            similar_props = find_similar_properties(df, base_features, n=5)

            if not similar_props.empty:
                st.dataframe(similar_props, use_container_width=True)
            else:
                st.info("No similar properties found for comparison")

            # Feature Values Used
            with st.expander("🔍 Feature Values Used in Prediction"):
                feature_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Value': feature_values
                })
                st.dataframe(feature_df, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
            st.write("Debug Information:")
            st.write(f"Expected features: {feature_names}")
            st.write(f"Number of expected features: {len(feature_names)}")


def analyze_key_factors(df, models, feature_names):
    """Comprehensive analysis of key factors influencing property prices"""
    st.markdown('<div class="sub-header">🔍 Key Factors Analysis</div>', unsafe_allow_html=True)

    if not st.session_state.models_trained:
        st.warning("Train models first to see factor analysis")
        return

    best_model_name = st.session_state.best_model_name
    best_model = models[best_model_name]

    # Feature Importance Analysis
    if hasattr(best_model, 'feature_importances_'):
        st.markdown("### 📊 Feature Importance Ranking")

        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=False)

        # Top 15 features
        top_features = importance_df.head(15)

        fig = px.bar(
            top_features,
            x='Importance',
            y='Feature',
            orientation='h',
            title=f'Top 15 Most Important Features ({best_model_name})',
            color='Importance',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Insights
        st.markdown("### 💡 Key Insights")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Most Influential Factors:**")
            for idx, row in top_features.head(5).iterrows():
                st.markdown(f"- **{row['Feature']}**: {row['Importance']:.4f}")

        with col2:
            st.markdown("**Factor Categories:**")
            location_importance = \
            importance_df[importance_df['Feature'].str.contains('borough|neighborhood|zip', case=False)][
                'Importance'].sum()
            size_importance = importance_df[importance_df['Feature'].str.contains('sqft|square', case=False)][
                'Importance'].sum()
            age_importance = importance_df[importance_df['Feature'].str.contains('age|year', case=False)][
                'Importance'].sum()

            st.markdown(f"- **Location factors**: {location_importance:.2%}")
            st.markdown(f"- **Size factors**: {size_importance:.2%}")
            st.markdown(f"- **Age factors**: {age_importance:.2%}")

    # Borough Analysis
    st.markdown("### 🏙️ Borough-wise Price Analysis")

    if 'borough' in df.columns and 'sale_price' in df.columns:
        borough_stats = df.groupby('borough').agg({
            'sale_price': ['mean', 'median', 'std', 'min', 'max', 'count']
        }).round(2)
        borough_stats.columns = ['Mean Price', 'Median Price', 'Std Dev', 'Min Price', 'Max Price', 'Count']

        st.dataframe(borough_stats.style.format({
            'Mean Price': '${:,.0f}',
            'Median Price': '${:,.0f}',
            'Std Dev': '${:,.0f}',
            'Min Price': '${:,.0f}',
            'Max Price': '${:,.0f}',
            'Count': '{:,}'
        }), use_container_width=True)

        # Visualization
        col1, col2 = st.columns(2)

        with col1:
            fig = px.box(
                df,
                x='borough',
                y='sale_price',
                title='Price Distribution by Borough',
                color='borough'
            )
            fig.update_yaxes(title='Sale Price ($)')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            borough_avg = df.groupby('borough')['sale_price'].mean().sort_values(ascending=False)
            fig = px.bar(
                x=borough_avg.index,
                y=borough_avg.values,
                title='Average Price by Borough',
                labels={'x': 'Borough', 'y': 'Average Price ($)'},
                color=borough_avg.values,
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)

    # Building Class Analysis
    if 'building_class_category' in df.columns and 'sale_price' in df.columns:
        st.markdown("### 🏗️ Building Type Analysis")

        building_stats = df.groupby('building_class_category').agg({
            'sale_price': ['mean', 'count']
        }).round(2)
        building_stats.columns = ['Average Price', 'Count']
        building_stats = building_stats.sort_values('Average Price', ascending=False).head(10)

        fig = px.bar(
            building_stats.reset_index(),
            x='building_class_category',
            y='Average Price',
            title='Top 10 Building Types by Average Price',
            color='Average Price',
            color_continuous_scale='Reds'
        )
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    # Correlation Analysis
    st.markdown("### 🔗 Feature Correlation with Price")

    numeric_features = df[feature_names + ['sale_price']].select_dtypes(include=[np.number])
    correlation = numeric_features.corr()['sale_price'].sort_values(ascending=False).drop('sale_price')

    top_corr = pd.concat([correlation.head(10), correlation.tail(10)])

    fig = go.Figure(go.Bar(
        x=top_corr.values,
        y=top_corr.index,
        orientation='h',
        marker=dict(
            color=top_corr.values,
            colorscale='RdBu',
            cmin=-1,
            cmax=1
        )
    ))
    fig.update_layout(
        title='Top Positive & Negative Correlations with Sale Price',
        xaxis_title='Correlation Coefficient',
        yaxis_title='Feature',
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)


def main():
    # Header
    st.markdown('<div class="main-header">🏢 Smart Valuation in Real Estate</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align: center; color: #7f8c8d; font-size: 1.2rem;">AI-Powered Modelling of NYC Property Sales - Enhanced Version</p>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/Flag_of_New_York_City.svg/1200px-Flag_of_New_York_City.svg.png",
            width=100
        )
        st.title("🗺️ Navigation")
        page = st.radio(
            "Go to",
            ["📊 Data Overview", "🤖 Model Training", "🔮 Price Prediction", "📈 Analysis & Insights"],
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info("""
        **Enhanced NYC Property Price Predictor**

        This tool uses advanced machine learning to predict property prices with:

        ✓ **4 ML Models**
        - Linear Regression
        - Random Forest
        - Gradient Boosting
        - XGBoost

        ✓ **Advanced Features**
        - Borough-based outlier removal
        - 20+ engineered features
        - Hyperparameter tuning
        - Cross-validation
        - Similar property matching

        ✓ **Comprehensive Analysis**
        - Feature importance
        - Price trends
        - Borough comparisons
        - Correlation analysis
        """)

        if st.session_state.models_trained:
            st.success("✅ Models Trained")
            st.metric("Best Model", st.session_state.best_model_name)
            st.metric("R² Score", f"{st.session_state.metrics[st.session_state.best_model_name]['R²']:.3f}")

    # Main Content
    if page == "📊 Data Overview":
        st.header("📊 Data Overview & Preprocessing")

        st.markdown("""
        Upload your NYC property sales dataset to begin. The system will:
        1. Clean and validate data
        2. Remove outliers by borough
        3. Engineer 20+ predictive features
        4. Encode categorical variables
        """)

        uploaded_file = st.file_uploader(
            "📁 Upload NYC Property Sales Dataset (CSV)",
            type=['csv'],
            help="Upload the NYC property sales dataset from data.world"
        )

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)

                st.success(f"✅ Dataset loaded successfully!")

                # Display basic info
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("📝 Total Records", f"{len(df):,}")
                col2.metric("📊 Features", df.shape[1])
                col3.metric("🔢 Numeric Cols", len(df.select_dtypes(include=[np.number]).columns))
                col4.metric("📝 Categorical Cols", len(df.select_dtypes(include=['object']).columns))

                # Data Quality Overview
                st.markdown("### 🔍 Data Quality Overview")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Missing Values:**")
                    missing = df.isnull().sum()
                    missing = missing[missing > 0].sort_values(ascending=False)
                    if len(missing) > 0:
                        st.dataframe(missing.to_frame('Missing Count'), use_container_width=True)
                    else:
                        st.success("No missing values detected!")

                with col2:
                    st.markdown("**Data Types:**")
                    dtypes_df = df.dtypes.astype(str).value_counts().to_frame('Count')
                    dtypes_df.index.name = 'Data Type'
                    st.dataframe(dtypes_df, use_container_width=True)

                # Show sample data
                st.markdown("### 📋 Sample Data")
                st.dataframe(df.head(10), use_container_width=True)

                # Data statistics
                with st.expander("📈 Statistical Summary"):
                    st.dataframe(df.describe(), use_container_width=True)

                # Column information
                with st.expander("ℹ️ Column Information"):
                    col_info = pd.DataFrame({
                        'Column': df.columns,
                        'Type': df.dtypes.astype(str).values,  # Convert dtype objects to strings
                        'Non-Null Count': df.notnull().sum().values,
                        'Unique Values': [df[col].nunique() for col in df.columns]
                    })
                    st.dataframe(col_info, use_container_width=True)

                # Preprocess button
                st.markdown("---")
                if st.button("🔧 Preprocess Data", type="primary", use_container_width=True):
                    df_processed = preprocess_data(df)
                    st.session_state.df_processed = df_processed
                    st.session_state.data_processed = True

                    st.markdown("### ✅ Processed Data Preview")
                    st.dataframe(df_processed.head(), use_container_width=True)

                    # Visualizations
                    col1, col2 = st.columns(2)

                    with col1:
                        if 'sale_price' in df_processed.columns:
                            fig = px.histogram(
                                df_processed,
                                x='sale_price',
                                nbins=50,
                                title='Distribution of Sale Prices (After Processing)',
                                labels={'sale_price': 'Sale Price ($)'}
                            )
                            st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        if 'property_age' in df_processed.columns:
                            fig = px.histogram(
                                df_processed,
                                x='property_age',
                                nbins=30,
                                title='Distribution of Property Age',
                                labels={'property_age': 'Property Age (years)'}
                            )
                            st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"❌ Error loading dataset: {e}")
                st.info("Please ensure your CSV file matches the expected NYC property sales format")

    elif page == "🤖 Model Training":
        st.header("🤖 Machine Learning Model Training")

        if not st.session_state.data_processed:
            st.warning("⚠️ Please load and preprocess data first from the Data Overview page.")
        else:
            df = st.session_state.df_processed

            st.markdown("### 🎯 Training Configuration")

            col1, col2 = st.columns(2)

            with col1:
                test_size = st.slider("Test Set Size (%)", 10, 40, 20) / 100

            with col2:
                use_tuning = st.checkbox(
                    "Enable Hyperparameter Tuning",
                    value=False,
                    help="This will take longer but may improve accuracy"
                )

            # Feature selection
            st.markdown("### 📊 Feature Selection")

            target_col = 'sale_price'

            # Comprehensive feature list
            base_features = [
                'land_square_feet', 'gross_square_feet', 'year_built',
                'residential_units', 'commercial_units', 'total_units'
            ]

            encoded_features = [
                'borough_encoded', 'neighborhood_encoded',
                'building_class_category_encoded'
            ]

            engineered_features = [
                'property_age', 'building_coverage_ratio',
                'units_per_sqft', 'commercial_ratio',
                'sale_year', 'sale_month', 'sale_quarter'
            ]

            # Filter available features
            all_features = base_features + encoded_features + engineered_features
            available_features = [f for f in all_features if f in df.columns]

            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"**Base Features:** {len([f for f in base_features if f in available_features])}")
            with col2:
                st.info(f"**Encoded Features:** {len([f for f in encoded_features if f in available_features])}")
            with col3:
                st.info(f"**Engineered Features:** {len([f for f in engineered_features if f in available_features])}")

            with st.expander("📋 View All Selected Features"):
                st.write(available_features)

            # Prepare data
            X = df[available_features]
            y = df[target_col]

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            # Scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            col1, col2, col3 = st.columns(3)
            col1.metric("🎓 Training Samples", f"{len(X_train):,}")
            col2.metric("🧪 Test Samples", f"{len(X_test):,}")
            col3.metric("📊 Features", len(available_features))

            # Train models button
            st.markdown("---")
            if st.button("🚀 Train All Models", type="primary", use_container_width=True):
                trained_models, metrics, cv_scores = train_models_with_tuning(
                    X_train_scaled, X_test_scaled, y_train, y_test, use_tuning
                )

                st.session_state.models = trained_models
                st.session_state.metrics = metrics
                st.session_state.cv_scores = cv_scores
                st.session_state.models_trained = True
                st.session_state.scaler = scaler
                st.session_state.feature_names = available_features
                st.session_state.X_test = X_test_scaled
                st.session_state.y_test = y_test

                st.success("✅ All models trained successfully!")

                # Display comparison
                display_model_comparison(metrics, cv_scores)

                # Model recommendations
                st.markdown("### 💡 Model Recommendations")
                best_model = st.session_state.best_model_name
                best_r2 = metrics[best_model]['R²']

                if best_r2 >= 0.75:
                    st.success(f"""
                    ✅ **Excellent Performance!** 

                    The {best_model} achieved an R² score of {best_r2:.3f}, indicating strong predictive capability.
                    This model is recommended for production use.
                    """)
                elif best_r2 >= 0.65:
                    st.info(f"""
                    ℹ️ **Good Performance** 

                    The {best_model} achieved an R² score of {best_r2:.3f}.
                    Consider enabling hyperparameter tuning to potentially improve performance.
                    """)
                else:
                    st.warning(f"""
                    ⚠️ **Moderate Performance**

                    The {best_model} achieved an R² score of {best_r2:.3f}.
                    Recommendations:
                    - Enable hyperparameter tuning
                    - Check for data quality issues
                    - Consider additional feature engineering
                    """)

    elif page == "🔮 Price Prediction":
        st.header("🔮 Property Price Prediction")

        if not st.session_state.models_trained:
            st.warning("⚠️ Please train models first from the Model Training page.")
        else:
            prediction_interface(
                st.session_state.models,
                st.session_state.feature_names,
                st.session_state.scaler
            )

    elif page == "📈 Analysis & Insights":
        st.header("📈 Analysis & Insights")

        if not st.session_state.data_processed:
            st.warning("⚠️ Please load and preprocess data first.")
        else:
            df = st.session_state.df_processed

            # Analysis tabs
            tab1, tab2, tab3, tab4 = st.tabs([
                "🔍 Key Factors", "📊 Market Trends", "🏙️ Geographic Analysis", "📉 Price Patterns"
            ])

            with tab1:
                if st.session_state.models_trained:
                    analyze_key_factors(
                        df,
                        st.session_state.models,
                        st.session_state.feature_names
                    )
                else:
                    st.info("Train models first to see key factor analysis")

            with tab2:
                st.markdown("### 📈 Price Trends Over Time")

                if 'sale_year' in df.columns and 'sale_price' in df.columns:
                    yearly_stats = df.groupby('sale_year').agg({
                        'sale_price': ['mean', 'median', 'count']
                    }).reset_index()
                    yearly_stats.columns = ['Year', 'Mean Price', 'Median Price', 'Transaction Count']

                    fig = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=('Average Price Trend', 'Transaction Volume'),
                        row_heights=[0.6, 0.4]
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=yearly_stats['Year'],
                            y=yearly_stats['Mean Price'],
                            mode='lines+markers',
                            name='Mean Price',
                            line=dict(color='blue', width=3)
                        ),
                        row=1, col=1
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=yearly_stats['Year'],
                            y=yearly_stats['Median Price'],
                            mode='lines+markers',
                            name='Median Price',
                            line=dict(color='green', width=3, dash='dash')
                        ),
                        row=1, col=1
                    )

                    fig.add_trace(
                        go.Bar(
                            x=yearly_stats['Year'],
                            y=yearly_stats['Transaction Count'],
                            name='Transactions',
                            marker_color='lightblue'
                        ),
                        row=2, col=1
                    )

                    fig.update_xaxes(title_text="Year", row=2, col=1)
                    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
                    fig.update_yaxes(title_text="Count", row=2, col=1)
                    fig.update_layout(height=700, showlegend=True)

                    st.plotly_chart(fig, use_container_width=True)

                # Monthly seasonality
                if 'sale_month' in df.columns:
                    st.markdown("### 📅 Seasonal Price Patterns")

                    monthly_avg = df.groupby('sale_month')['sale_price'].mean().reset_index()
                    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    monthly_avg['Month Name'] = monthly_avg['sale_month'].apply(lambda x: month_names[int(x) - 1])

                    fig = px.line(
                        monthly_avg,
                        x='Month Name',
                        y='sale_price',
                        title='Average Price by Month',
                        markers=True
                    )
                    fig.update_yaxes(title='Average Price ($)')
                    fig.update_xaxes(title='Month')
                    st.plotly_chart(fig, use_container_width=True)

            with tab3:
                st.markdown("### 🗺️ Geographic Price Distribution")

                if 'borough' in df.columns and 'neighborhood' in df.columns:
                    selected_borough = st.selectbox(
                        "Select Borough for Neighborhood Analysis",
                        sorted(df['borough'].unique())
                    )

                    borough_data = df[df['borough'] == selected_borough]

                    neighborhood_stats = borough_data.groupby('neighborhood').agg({
                        'sale_price': ['mean', 'median', 'count']
                    }).reset_index()
                    neighborhood_stats.columns = ['Neighborhood', 'Mean Price', 'Median Price', 'Count']
                    neighborhood_stats = neighborhood_stats.sort_values('Mean Price', ascending=False).head(20)

                    fig = px.bar(
                        neighborhood_stats,
                        x='Neighborhood',
                        y='Mean Price',
                        title=f'Top 20 Neighborhoods in {selected_borough} by Average Price',
                        color='Mean Price',
                        color_continuous_scale='Plasma'
                    )
                    fig.update_xaxes(tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)

            with tab4:
                st.markdown("### 📊 Price Distribution Patterns")

                col1, col2 = st.columns(2)

                with col1:
                    if 'property_age' in df.columns:
                        fig = px.scatter(
                            df.sample(min(5000, len(df))),
                            x='property_age',
                            y='sale_price',
                            title='Price vs Property Age',
                            opacity=0.5,
                            trendline='lowess'
                        )
                        st.plotly_chart(fig, use_container_width=True)

                with col2:
                    if 'gross_square_feet' in df.columns:
                        fig = px.scatter(
                            df.sample(min(5000, len(df))),
                            x='gross_square_feet',
                            y='sale_price',
                            title='Price vs Size',
                            opacity=0.5,
                            trendline='ols'
                        )
                        st.plotly_chart(fig, use_container_width=True)

                # Price per square foot analysis
                if 'price_per_sqft' in df.columns and 'borough' in df.columns:
                    st.markdown("### 💵 Price per Square Foot by Borough")

                    fig = px.box(
                        df,
                        x='borough',
                        y='price_per_sqft',
                        title='Price per Sq Ft Distribution by Borough',
                        color='borough'
                    )
                    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()