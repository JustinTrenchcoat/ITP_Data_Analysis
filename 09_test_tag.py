import pandas as pd
from helper import *
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import gsw
import datetime
import re
import seaborn as sns
from matplotlib.colors import SymLogNorm
import matplotlib.colors as colors
import matplotlib.dates as mdates
from matplotlib.colors import Normalize
from scipy.ndimage import uniform_filter1d
import math
'''
In the dataset:
every row is one observation from one profile.
It has to be ordered in time so that time series analysis would work.
'''
# Read in the data:
def readNC(full_path, ls, itp_num):
    ds = nc.Dataset(full_path)
    with ds as dataset:
        # extract variables:
        # the prof is not profile number, but index. FloatID true profile number from each ITP system
        # this would not be used as an input variable
        profN = dataset.variables['FloatID'][:]
        # the nc file mistakenly wrote pressure instead of depth
        depth = dataset.variables['pressure'][:]
        temp = dataset.variables["ct"][:]
        salinity = dataset.variables["sa"][:]
        connect_layer_mask = dataset.variables['mask_cl'][:]
        interface_layer_mask = dataset.variables['mask_int'][:]
        mixed_layer_mask = dataset.variables["mask_ml"][:]
        staircase_mask = dataset.variables["mask_sc"][:]
        dates = dataset.variables["dates"][:]
        lon = dataset.variables["lon"][:]
        lat = dataset.variables["lat"][:]
        date = pd.to_datetime(dates, unit = 's')
        date = date.date
        for i in range(len(profN)):
            mask_cl = connect_layer_mask[i]
            mask_int = interface_layer_mask[i]
            mask_ml = mixed_layer_mask[i]
            mask_sc = staircase_mask[i]
            new_df = pd.DataFrame({
                "profileNumber" : profN[i],
                "depth" : depth[i],
                'temp' : temp[i],
                'date' : date[i],
                "salinity" : salinity[i],
                'mask_cl' : mask_cl,
                'mask_int' : mask_int,
                'mask_ml' : mask_ml,
                "mask_sc" : mask_sc,
                "lon" : lon[i],
                "lat" : lat[i]
            })
            new_df['pressure'] = pressure(depth[i],lat[i])
            new_df['itpNum'] = itp_num
            #############################################################################
            # background infos, do it here because we calculate it per-profile:
            # N Sqaured:
            # Apply centered rolling window smoothing (you can adjust window size)
            temp_smooth = pd.Series(temp[i]).rolling(window=5, center=True).mean().to_numpy()
            salt_smooth = pd.Series(salinity[i]).rolling(window=5, center=True).mean().to_numpy()
            temp_smooth = pd.Series(temp_smooth).bfill().ffill().to_numpy()
            salt_smooth = pd.Series(salt_smooth).bfill().ffill().to_numpy()
            n_sq = gsw.Nsquared(salt_smooth, temp_smooth, new_df['pressure'], lat[i])[0]
            # padding for last value as the function returns only N-1 values
            n_sq_padded = np.append(n_sq, np.nan)
            new_df['n_sq'] = n_sq_padded
            # turner angle and R_rho
            [turner_angle, R_rho,p_mid] = gsw.Turner_Rsubrho(salt_smooth, temp_smooth, new_df['pressure'])
            new_df['turner_angle'] = np.append(turner_angle,np.nan)
            new_df['R_rho'] = np.append(R_rho,np.nan)
            ####################
            ls.append(new_df)
    return ls
#######################################################
# Read Data and save, Only run once                   #
#######################################################
# tagData_dir = 'prod_files'
# df_list = []
# for fileName in tqdm(sorted(os.listdir(tagData_dir)), desc="Processing files"):
#     match = re.search(r'itp(\d+)cormat\.nc', fileName)
#     if match:
#             itp_num = int(match.group(1))
#             full_path = os.path.join(tagData_dir, fileName)
#             df_list = readNC(full_path, df_list, itp_num)
#             final_df = pd.concat(df_list, ignore_index=True)
#             final_df.to_pickle("final.pkl")
#######################################################
#                                                     #
#######################################################
final_df = pd.read_pickle("final.pkl")
test_df = final_df[final_df['itpNum'] == 65].copy()
############################################################
# file checks:
# print(test_df.shape)
# num_unique = test_df['profileNumber'].nunique()
# print(f"Number of unique profNum values: {num_unique}")

# check_df = test_df[test_df['profileNumber'] == 20].copy()
# checkDepth = check_df['depth']
# checkTemp = check_df['temp']
# helPlot(checkTemp, checkDepth)

# depth_col = test_df['depth']
# n_sq_col = test_df['n_sq']

# printBasicStat(depth_col)
# printBasicStat(n_sq_col)

# depth_index = np.where(np.isnan(depth_col))[0]
# print(depth_index)

# n_sq_index = np.where(np.isnan(n_sq_col))[0]
# print(len(n_sq_index))

# print(test_df['n_sq'].min(), test_df['n_sq'].max(), test_df['n_sq'].describe())
####################################################################################
# machine learning part
'''
In the demo case we would only use ITP100. and according to the visualization we would devide train set and test set based on date.
We would make some features on the fly for testing smoothing window_size

-------------------------------------------------------
Numerical features:
Depth
Temperature
Salinity
N^2
turner_angle or R-rho
--------------------------------------------------------
Categorical features:
?
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
Date/Month, dat?
'''
def process_df(df, window_size):
    ls = []
    # test if the loop for every profile works
    unique_profNum = df['profileNumber'].unique()
    for i in unique_profNum:
        df_on_fly = df[df['profileNumber'] == i].copy()
        temp_smooth = uniform_filter1d(df_on_fly['temp'], size=window_size, mode='nearest')
        salt_smooth = uniform_filter1d(df_on_fly['salinity'], size=window_size, mode='nearest')
        pres_smooth = uniform_filter1d(df_on_fly['pressure'], size=window_size, mode='nearest')
        # add new cols:
        n_sq = gsw.Nsquared(salt_smooth, temp_smooth, pres_smooth, df_on_fly['lat'])[0]
        # padding for last value as the function returns only N-1 values
        n_sq_padded = np.append(n_sq, np.nan)
        df_on_fly['smooth_n_sq'] = n_sq_padded
        # turner angle and R_rho
        [turner_angle, R_rho,p_mid] = gsw.Turner_Rsubrho(salt_smooth, temp_smooth, pres_smooth)
        df_on_fly['smooth_turner_angle'] = np.append(turner_angle,np.nan)
        df_on_fly['smooth_R_rho'] = np.append(R_rho,np.nan)
        ls.append(df_on_fly)
    fin_df = pd.concat(ls, ignore_index=True)
    return fin_df
###################################################
# run once
# ocean_df = process_df(test_df, 5)
# ocean_df.to_pickle("ocean.pkl")
####################################################
ocean_df = pd.read_pickle("ocean.pkl")
ocean_df['Date'] = pd.to_datetime(ocean_df['date'])
train_df = ocean_df.query("Date <= 20130503")
test_df = ocean_df.query("Date >  20130503")
train_df = train_df.assign(
    Month=train_df["Date"].apply(lambda x: x.month_name())
    )  # x.month_name() to get the actual string
test_df = test_df.assign(Month=test_df["Date"].apply(lambda x: x.month_name()))
# print(train_df.info())
numeric_features = [
    "depth",
    "smooth_n_sq",
    "smooth_R_rho",
    'lon',
    'lat'
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
                'n_sq',
                'turner_angle',
                'smooth_turner_angle',
                'pressure',
                "temp",
                "salinity",
                'R_rho']
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.impute import SimpleImputer


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

from sklearn.linear_model import LogisticRegression

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
###################################################################
# feaure importance analysis
pipe_lr = make_pipeline(preprocessor, LogisticRegression(max_iter=2000, random_state=2))
pipe_lr.fit(train_df, y_train)

feature_names = (
    numeric_features + categorical_features + ['Month']
)
# Get the coefficients (flattened to 1D list)
coefs = pipe_lr.named_steps["logisticregression"].coef_.flatten().tolist()

# Get the feature names
feature_names = pipe_lr.named_steps["columntransformer"].get_feature_names_out()
print(feature_names)

# Sanity check
assert len(coefs) == len(feature_names)

# Now build the DataFrame
data = {
    "coefficient": coefs,
    "magnitude": np.abs(coefs),
}

coef_df = pd.DataFrame(data, index=feature_names).sort_values("magnitude", ascending=False)
print(coef_df)
