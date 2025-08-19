# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# %%
df = pd.read_csv(r"C:\Users\rosel\Downloads\New folder\Movie Success Prediction\test_data.csv") 
print("Data loaded. Shape:", df.shape)
df.head()

# %%
print("\nMissing values before handling:")
print(df.isnull().sum())

# %%
# Keep numeric columns numeric (coerce weird strings to NaN)
num_like_cols = [
    'num_critic_for_reviews','duration','director_facebook_likes',
    'actor_3_facebook_likes','actor_1_facebook_likes','gross','num_voted_users',
    'cast_total_facebook_likes','facenumber_in_poster','num_user_for_reviews',
    'budget','title_year','actor_2_facebook_likes','movie_facebook_likes'
]
for c in num_like_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# %%
# Create ROI target (percent)
df['ROI_pct'] = np.where(df['budget'] > 0, (df['gross'] - df['budget']) / df['budget'] * 100, np.nan)

# %%
# Drop rows with undefined ROI target
df = df.dropna(subset=['ROI_pct']).copy()

# %%
# Time
df['movie_age'] = 2025 - df['title_year']               # how old the movie is
df['decade']    = (df['title_year'] // 10) * 10         # decade bucket

# Ratios / rates
df['likes_per_actor']   = df['cast_total_facebook_likes'] / df['facenumber_in_poster'].replace(0, np.nan)
df['gross_per_budget']  = df['gross'] / df['budget']
df['reviews_per_vote']  = (df['num_user_for_reviews'] + df['num_critic_for_reviews']) / df['num_voted_users'].replace(0, np.nan)
df['critic_user_ratio'] = df['num_critic_for_reviews'] / df['num_user_for_reviews'].replace(0, np.nan)

# Interactions (simple but useful)
df['duration_x_reviews'] = df['duration'] * df['num_user_for_reviews']

# Genre features (multi-label one-hot)
if 'genres' in df.columns:
    genre_dummies = df['genres'].str.get_dummies(sep='|')
    # Optionally keep only the most common genres to avoid sparsity
    common_genres = genre_dummies.sum().sort_values(ascending=False).head(12).index
    genre_dummies = genre_dummies[common_genres]
    df = pd.concat([df, genre_dummies.add_prefix('genre_')], axis=1)

# Log transforms for skewed columns
for col in ['gross','budget','num_voted_users','movie_facebook_likes','cast_total_facebook_likes']:
    if col in df.columns:
        df[f'log_{col}'] = np.log1p(df[col])

# Clean divisions
df[['likes_per_actor','gross_per_budget','reviews_per_vote','critic_user_ratio']] = (
    df[['likes_per_actor','gross_per_budget','reviews_per_vote','critic_user_ratio']]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
)

# %%
df['target'] = np.where(df['ROI_pct'] > 50, 1, 0)   # 1 = Hit, 0 = Flop

# Remove leakage (post-release features)
leakage_cols = [
    "gross_per_budget", "imdb_score", "reviews_per_vote",
    "critic_user_ratio", "log_movie_facebook_likes", "gross", "log_gross", "ROI_pct"
]

X = df.drop(columns=leakage_cols + ["target"])   # 'target' = Hit/Flop label
y = df["target"]

# %%
from catboost import CatBoostClassifier, Pool

feature_columns = X.columns.tolist()

# Identify Categorical Columns
cat_features = X.select_dtypes(include=['object']).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Create CatBoost Pool
train_pool = Pool(X_train, y_train, cat_features=cat_features)
test_pool = Pool(X_test, y_test, cat_features=cat_features)

# Train Model
clf = CatBoostClassifier(
    iterations=1500,
    learning_rate=0.03,
    depth=8,
    loss_function='Logloss',
    eval_metric='Accuracy',
    random_seed=42,
    verbose=200
)

clf.fit(train_pool, eval_set=test_pool, early_stopping_rounds=100)

# %%
from catboost import cv

cv_params = {
    'iterations': 1500,
    'learning_rate': 0.03,
    'depth': 8,
    'loss_function': 'Logloss',
    'eval_metric': 'Accuracy',
    'random_seed': 42,
    'verbose': 200
}

cv_results = cv(
    pool=train_pool,
    params=cv_params,
    fold_count=5,
    shuffle=True,
    partition_random_seed=42
)

print(cv_results.head())
print("\nMean Accuracy:", cv_results['test-Accuracy-mean'].iloc[-1])

# %%
import shap

explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test)

# Global explanation
shap.summary_plot(shap_values, X_test, feature_names=X_test.columns)

# Local explanation (pick one movie)
i = 0
shap.force_plot(explainer.expected_value, shap_values[i], X_test.iloc[i])

# %%
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

y_pred = clf.predict(X)
ConfusionMatrixDisplay.from_predictions(y, y_pred, cmap="Blues")
plt.show()

# %%
import joblib
joblib.dump(clf, "hit_flop_model.pkl")
joblib.dump(feature_columns, "feature_columns.pkl")
joblib.dump(cat_features, "cat_features.pkl")
clf.save_model("catboost_model.cbm")

# %%
feature_columns = joblib.load("feature_columns.pkl")
cat_features = joblib.load("cat_features.pkl")

# %%
clf = CatBoostClassifier()
clf.load_model("catboost_model.cbm")

# %%
# Create new movie input (must follow same column order!)
new_movie = pd.DataFrame([{
    "budget": 50_000_000,
    "log_budget": np.log1p(50_000_000),
    "runtime": 120,
    "log_runtime": np.log1p(120),
    "popularity": 50,
    "log_popularity": np.log1p(50),
    "genres": "Action",
    "production_companies": "Marvel Studios",
    "release_month": 5,
    "release_day": 20,
    "release_year": 2024,
    "country": "USA"
}])

# Add missing columns with default values
for col in feature_columns:
    if col not in new_movie.columns:
        new_movie[col] = 0   
        
# Reorder columns to match training
new_movie = new_movie[feature_columns]

# Create CatBoost Pool with correct categorical features
new_pool = Pool(new_movie, cat_features=cat_features)

# Predict
print("Prediction:", model.predict(new_pool))

# %%
print("Prediction:", model.predict(new_pool))
print("Probabilities:", model.predict_proba(new_pool))

# %%



