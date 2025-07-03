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
'''
In the demo case we would only use ITP 62, 65,and 68. and from the visualization we would devide train set and test set based on date.
-------------------------------------------------------
Numerical features:
Depth
Temperature
Salinity
N^2
turner_angle or R-rho
--------------------------------------------------------
Categorical features:
Month
--------------------------------------------------------------------
Target? 
1. layer types:
        Mixed layer
        interface layer
2. staircase types:
        appearance of staircase
        sharp
        mushy
        super mushy
----------------------------------------------------------------------
Time analysis variable:
Date/Days since
'''


final_df = pd.read_pickle("final.pkl")
experiment_df = final_df[final_df['itpNum'].isin([62, 65, 68])].copy()
############################################################
# sanity checks, run it to check if you have loaded everything right:

# print(experiment_df.shape)
# num_unique = experiment_df['itpNum'].nunique()
# print(f"Number of unique profNum values: {num_unique}")

# check_df = experiment_df[experiment_df['profileNumber'] == 30].copy()
# checkDepth = check_df['depth']
# checkTemp = check_df['temp']
# helPlot(checkTemp, checkDepth)

# depth_col = experiment_df['depth']
# n_sq_col = experiment_df['n_sq']

# printBasicStat(depth_col)
# printBasicStat(n_sq_col)

# print(experiment_df['n_sq'].min(), experiment_df['n_sq'].max(), experiment_df['n_sq'].describe())
####################################################################################
ocean_df = experiment_df.copy()
ocean_df['Date'] = pd.to_datetime(ocean_df['date'])
first_day = ocean_df["Date"].min()
# 2012-08-28
last_day = ocean_df["Date"].max()
# 2014-04-14
train_df = ocean_df.query("Date <= 20130503")
test_df = ocean_df.query("Date >  20130503")
train_df = train_df.assign(
    Month=train_df["Date"].apply(lambda x: x.month_name())
    )  # x.month_name() to get the actual string
train_df = train_df.assign(
    Days_since = train_df["Date"].apply(lambda x: (x-first_day).days)
)
test_df = test_df.assign(Month=test_df["Date"].apply(lambda x: x.month_name()))
test_df = test_df.assign(
    Days_since = test_df["Date"].apply(lambda x: (x-first_day).days)
)

numeric_features = [
    "depth",
    "n_sq",
    "R_rho",
    'lon',
    'lat',
    'Days_since'
]
categorical_features = []
time_feature = ["Date", 'Month']
target = ['mask_ml']
drop_features = ['profileNumber',
                'date',
                'itpNum',
                'mask_sc',
                'mask_int',
                'mask_cl',
                'turner_angle',
                'pressure',
                "temp",
                "salinity",
                'dT/dZ',
                'dS/dZ']


def preprocess_features(
    train_df,
    test_df,
    numeric_features,
    categorical_features,
    drop_features,
    target):

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

    return X_train_enc, y_train, X_test_enc, y_test, preprocessor

X_train_enc, y_train, X_test_enc, y_test, preprocessor = preprocess_features(
    train_df, test_df, 
    numeric_features, 
    categorical_features + ["Month"], 
    drop_features,
    target
    )
# print(X_train_enc.head())


def score_lr_print_coeff(preprocessor, train_df, y_train, test_df, y_test, X_train_enc):
    lr_pipe = make_pipeline(preprocessor, LogisticRegression(max_iter=1000))
    lr_pipe.fit(train_df, y_train)
    print("Train score: {:.2f}".format(lr_pipe.score(train_df, y_train)))
    print("Test score: {:.2f}".format(lr_pipe.score(test_df, y_test)))
    lr_coef = pd.DataFrame(
        data=lr_pipe.named_steps["logisticregression"].coef_.flatten(),
        index=X_train_enc.columns,
        columns=["Coef"],
    )
    return lr_coef.sort_values(by="Coef", ascending=False)
score_lr_print_coeff(preprocessor, train_df, y_train, test_df, y_test, X_train_enc)
# ###################################################################
# feaure importance analysis
print('feature importance analysis: \n')
pipe_lr = make_pipeline(preprocessor, LogisticRegression(max_iter=2000, random_state=2, class_weight='balanced'))
pipe_lr.fit(train_df, y_train)

feature_names = (
    numeric_features + categorical_features + ['Month']
)
# Get the coefficients (flattened to 1D list)
coefs = pipe_lr.named_steps["logisticregression"].coef_.flatten().tolist()

# Get the feature names
feature_names = pipe_lr.named_steps["columntransformer"].get_feature_names_out()

# Sanity check
assert len(coefs) == len(feature_names)

# Now build the DataFrame
data = {
    "coefficient": coefs,
    "magnitude": np.abs(coefs),
}

coef_df = pd.DataFrame(data, index=feature_names).sort_values("magnitude", ascending=False)
print(coef_df)
#######################################################################################
print("Model performance assessment:\n")
# confusion matrix:
ConfusionMatrixDisplay.from_estimator(
    pipe_lr, test_df, y_test, values_format='d', display_labels=["not staircase", 'staircase']
)
plt.show()
print(classification_report(y_test, pipe_lr.predict(test_df)))
###########################################################################################
#  ROC and AUC metric:
from sklearn.metrics import roc_curve
lr_pipe = make_pipeline(preprocessor, LogisticRegression(max_iter=2000))
lr_pipe.fit(train_df, y_train)


fpr_imbalanced, tpr_imbalanced, thresholds_imbalanced = roc_curve(y_test, lr_pipe.predict_proba(test_df)[:, 1])
plt.plot(fpr_imbalanced, tpr_imbalanced, label="ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR (recall)")

default_threshold = np.argmin(np.abs(thresholds_imbalanced - 0.5))

plt.plot(
    fpr_imbalanced[default_threshold],
    tpr_imbalanced[default_threshold],
    "ob",
    markersize=10,
    label="threshold 0.5",
)

fpr, tpr, thresholds = roc_curve(y_test, pipe_lr.predict_proba(test_df)[:, 1])
plt.plot(fpr, tpr, label="ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR (recall)")

default_threshold = np.argmin(np.abs(thresholds - 0.5))

plt.plot(
    fpr[default_threshold],
    tpr[default_threshold],
    "or",
    markersize=10,
    label="threshold 0.5",
)
plt.legend(loc="best")
plt.show()

