import pandas as pd
from helper import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
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
    'turner_angle', 'pressure', "temp", "salinity", 'dT/dZ', 'dS/dZ', "depth"
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
        OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop='first'),
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
##############################################################################################
# Model Fitting Part
# Feature Importance Analysis via statsmodels (with correct alignment)
feature_names = preprocessor.get_feature_names_out()

# Add intercept manually
X_train_processed = sm.add_constant(X_train_enc)
feature_names = np.insert(feature_names, 0, 'Intercept')
# Fit logistic regression with statsmodels
model = sm.Logit(y_train, X_train_processed)
result = model.fit()

# Collect coefficients and p-values
coef_df = pd.DataFrame({
    'coef': result.params,
    'p_value': result.pvalues,
    'conf_lower': result.conf_int()[0],
    'conf_upper': result.conf_int()[1]
})

print(coef_df)

#################################################################################
# VIF analysis
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame()
vif_data["feature"] = X_train_enc.columns
vif_data["VIF"] = [variance_inflation_factor(X_train_enc.values, i) for i in range(X_train_enc.shape[1])]
print(vif_data)

'''
The VIF shows that Days_since is highly correlated with other features, further investigation is shown below
'''
#####################################################################################
import seaborn as sns
corr = X_train_enc[['Days_since', 'n_sq', 'R_rho', 'lon', 'lat']].corr()
sns.heatmap(corr, annot=True)
plt.show()
# heat map and VIF indicated that Days_since and latitude is hight correlated, but I think it might be due to the lack of data.
# additionally, n_sq, r_rho and depth are highly correlated, considering that, we might need to drop depth for better inference??
######################################################################
# pearson residual
# resid_pearson = model.resid_pearson
# print(resid_pearson)