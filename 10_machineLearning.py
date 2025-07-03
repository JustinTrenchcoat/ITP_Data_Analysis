import pandas as pd
from helper import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
import statsmodels.api as sm

# Load Data
final_df = pd.read_pickle("final.pkl")

experiment_df = final_df[final_df['itpNum'].isin([62, 65, 68])].copy()
ocean_df = experiment_df.copy()
ocean_df['Date'] = pd.to_datetime(ocean_df['date'])
first_day = ocean_df["Date"].min()

train_df = ocean_df.query("Date <= 20130503")
test_df = ocean_df.query("Date >  20130503")

train_df = train_df.assign(
    Month=train_df["Date"].apply(lambda x: x.month_name()),
    Days_since=train_df["Date"].apply(lambda x: (x - first_day).days)
)
test_df = test_df.assign(
    Month=test_df["Date"].apply(lambda x: x.month_name()),
    Days_since=test_df["Date"].apply(lambda x: (x - first_day).days)
)

numeric_features = [
    "depth",
    "n_sq",
    "R_rho",
    'lon',
    'lat',
    'Days_since'
]
categorical_features = []  # You’ll add 'Month' later
time_feature = ["Date", 'Month']
target = ['mask_ml']
drop_features = [
    'profileNumber', 'date', 'itpNum', 'mask_sc', 'mask_int', 'mask_cl',
    'turner_angle', 'pressure', "temp", "salinity", 'dT/dZ', 'dS/dZ'
]

def preprocess_features(train_df, test_df, numeric_features, categorical_features, drop_features, target):
    all_features = set(numeric_features + time_feature + drop_features + target)
    if set(train_df.columns) != all_features:
        print("Missing columns", set(train_df.columns) - all_features)
        print("Extra columns", all_features - set(train_df.columns))
        raise Exception("Columns do not match")

    numeric_transformer = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler()
    )
    categorical_transformer = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="missing"),
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    )

    preprocessor = make_column_transformer(
        (numeric_transformer, numeric_features),
        (categorical_transformer, categorical_features),
        ("drop", drop_features),
    )
    preprocessor.fit(train_df)
    ohe_feature_names = (
        preprocessor.named_transformers_["pipeline-2"]
        .named_steps["onehotencoder"]
        .get_feature_names_out(categorical_features)
        .tolist()
    )
    new_columns = numeric_features + ohe_feature_names

    X_train_enc = pd.DataFrame(
        preprocessor.transform(train_df), index=train_df.index, columns=new_columns
    )
    X_test_enc = pd.DataFrame(
        preprocessor.transform(test_df), index=test_df.index, columns=new_columns
    )

    y_train = train_df["mask_sc"]
    y_test = test_df["mask_sc"]

    # Drop rows with missing target in train set
    mask_train = ~y_train.isna()
    X_train_enc = X_train_enc[mask_train]
    y_train = y_train[mask_train]

    # Drop rows with missing target in test set
    mask_test = ~y_test.isna()
    X_test_enc = X_test_enc[mask_test]
    y_test = y_test[mask_test]

    return X_train_enc, y_train, X_test_enc, y_test, preprocessor

X_train_enc, y_train, X_test_enc, y_test, preprocessor = preprocess_features(
    train_df, test_df, 
    numeric_features, 
    categorical_features + ["Month"], 
    drop_features,
    target
)

# Feature Importance Analysis via statsmodels (with correct alignment)
feature_names = preprocessor.get_feature_names_out()
print("Feature Names:", feature_names)

# Apply preprocessor only on the same clean subset
mask_train = ~train_df["mask_sc"].isna()
X_train_processed = preprocessor.transform(train_df[mask_train])
y_train_clean = train_df.loc[mask_train, "mask_sc"]

# Add intercept manually
X_train_processed = sm.add_constant(X_train_processed)
feature_names = np.insert(feature_names, 0, 'Intercept')

# Fit logistic regression with statsmodels
model = sm.Logit(y_train_clean, X_train_processed)
result = model.fit()

# Collect coefficients and p-values
coef_df = pd.DataFrame({
    'coef': result.params,
    'p_value': result.pvalues,
    'conf_lower': result.conf_int()[0],
    'conf_upper': result.conf_int()[1]
}, index=feature_names)

print(coef_df)


import seaborn as sns

# Diagnostics: Target distribution
print("\n✅ Target class distribution:")
print(y_train_clean.value_counts())

# Diagnostics: Dataset uniqueness check
print("\n✅ Unique feature rows vs. total rows:")
print(f"Unique feature rows: {np.unique(X_train_processed, axis=0).shape[0]}")
print(f"Total rows: {len(X_train_processed)}")

# Diagnostics: Simple cross-tab for potential separation in categorical/time features
print("\n✅ Crosstab with Month:")
print(pd.crosstab(train_df.loc[mask_train, "Month"], y_train_clean))


# ✅ Try Regularized Logistic Regression (Handles Separation)
print("\n✅ Fitting Regularized Logistic Regression (handles separation automatically):")
model = sm.Logit(y_train_clean, X_train_processed)
result = model.fit_regularized()

print(result.summary())

# Coefficient table
coef_df = pd.DataFrame({
    'coef': result.params,
})
print("\n✅ Coefficients (Regularized Model):")
print(coef_df)


#######################################################################################
# print("Model performance assessment:\n")
# # confusion matrix:
# ConfusionMatrixDisplay.from_estimator(
#     pipe_lr, test_df, y_test, values_format='d', display_labels=["not staircase", 'staircase']
# )
# plt.show()
# print(classification_report(y_test, pipe_lr.predict(test_df)))
# ###########################################################################################
# #  ROC and AUC metric:
# from sklearn.metrics import roc_curve
# lr_pipe = make_pipeline(preprocessor, LogisticRegression(max_iter=2000))
# lr_pipe.fit(train_df, y_train)


# fpr_imbalanced, tpr_imbalanced, thresholds_imbalanced = roc_curve(y_test, lr_pipe.predict_proba(test_df)[:, 1])
# plt.plot(fpr_imbalanced, tpr_imbalanced, label="ROC Curve")
# plt.xlabel("FPR")
# plt.ylabel("TPR (recall)")

# default_threshold = np.argmin(np.abs(thresholds_imbalanced - 0.5))

# plt.plot(
#     fpr_imbalanced[default_threshold],
#     tpr_imbalanced[default_threshold],
#     "ob",
#     markersize=10,
#     label="threshold 0.5",
# )

# fpr, tpr, thresholds = roc_curve(y_test, pipe_lr.predict_proba(test_df)[:, 1])
# plt.plot(fpr, tpr, label="ROC Curve")
# plt.xlabel("FPR")
# plt.ylabel("TPR (recall)")

# default_threshold = np.argmin(np.abs(thresholds - 0.5))

# plt.plot(
#     fpr[default_threshold],
#     tpr[default_threshold],
#     "or",
#     markersize=10,
#     label="threshold 0.5",
# )
# plt.legend(loc="best")
# plt.show()

