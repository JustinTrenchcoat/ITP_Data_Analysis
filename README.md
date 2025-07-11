# ITP Data Analysis Project

This is my experiment field for ITP data analysis project from UBC WLIURA 2025 Summer. This project is under the research project `S25 WLIURA - Undergraduate Research Asst Arctic Oceanography` with Dr. Stephanie Waterman as the supervisor. 

## List of scripts and their funcionalities:
| Script Name          | Functionalities |
| -----------          | --------------- |
| `00_testField.py`    | test field for new functions|
| `01_pullData.py`     | download and unzip the cormat files into the folder `datasets`, organize the files in ITP system number|
| `02_cleanData.py`    | Comb through every profile in `datasets` folder. Save profiles that matches the "gold standard" into another folder named `goldData`|
| `04_Plot.py`         | Has all kinds of plotting functions|
| `05_analyze.py`      | Calculate the depth difference in all profiles, summarize it and store the result into `depth_differences.pkl`.|
| `06_gridMaker.py`    | Take profiles from `goldData`, interpolate the data into a 0.25 m regular grided dataset, stores them in folder named `gridData`|
| `07_singleReader.py` | Tool for checking individual profile when something goes wrong.|
| `helper.py`          | Place for all helper functions|
| `09_process.py`      | process tagged dataset. It applies gaussian smoothing to the temperature, salinity, pressure and depth to produce new columns: `dS/dZ`, `dT/dZ`, `n_sq`, `turner_angle`, and `R_rho`|
| `10_machineLearning.py`  | a toy logistic time series regression with metrics and feature importance analysis|
 

## Development Log:
2025/05/22:
- [\[Here\]](https://scienceweb.whoi.edu/itp/data/) contains all data needed for the project. Below is a list of dataset folders that we did not use and why:  

| ITP System Number                          | Reason                                                   |
|------------------------------------------- |----------------------------------------------------------|
| 31, 40                                     | Mission in Antarctica                                    |
| 96,106,124,133,135,137,138,139,141,142,143 | Still active                                             |
| 39,45,46                                   | LMP Missions (in lakes)                                  |
| 134                                        | Device not deployed yet                                  |
| 136, 140                                   | Not been retreived yet?                                  |
| 20, 67, 71, 44, 50, 66                     | Received no profiles due to technical difficulties       |
| 93                                         | Directory error, deployed in irrelevant area             |



- `test.py` and `matlabRead.py` have scripts that could read from .mat files. The next step is to filter them through our requirement. The question now is whether we need to convert them into .nc or other format after filtering through them
- Since the dataset is totally different, we might not need the itp package?? Or we can write our own package
- We should also read through the clustering method and figure out how it works.
- in the `test.py`, we might need to pay attention to the data structure when unpacking the `.mat`file variables. the ordering is weird.
------
2025/05/26:
- the dataset is done, and need to be filtered
--------
2025/05/27:
- the cleaned dataset is done. There are in total of 32557 profiles, slightly less than the number of observations from 1-db measurement dataset (they have 35135 profiles)
- There are few code pieces need to be changed to accomodate the new data format:
    - These code pieces need to be deleted or changed as their functioaliy is replaced by other scripts.
        - `convert.py` deleted
        - `filter.py` deleted.
        - `read.py` deleted.
        - `unpack_matlab.py` renamed to `singleProfileReader.py` to test new functions on single profile.
    - There scripts need to be modified. Their functionality is still important.
        - `helper.py` remained same. Will add more helper functions.
        - `omniPlot.py` renamed to `04_Plot.py`, changed functionality
    - These are kept unchanged for now, but maybe a better name:
        - `pull.py` changed to `01_pullData.py`
        - `test.py` changed to `00_testField.py`
        - `number_check.py` changed to `03_checkData.py`
        - `matlabRead.py` changed to `02_cleanData.py`

- we need to look into the delta z of each profile. Maybe we need to make a constant resolution.
------------------------
2025/05/28:
- We need to get the dz values for each profile, summarize it into a histogram. This would be done in a new script called `05_analyze.py`
--------------------------
2025/05/29:
- The histogram is out, and 0.25 grid sounds like a reasonable resolution.
- Need to look into what had caused the drastic depth difference in the data. 
- from the experiment in testfield, the data structure is indeed weird: each measurement in one profile is not strictly arranged in time:
    - take `D:\EOAS\ITP_Data_Analysis\datasets\itp41cormat\cor1391.mat` for example, the first entry in its depth is 412.52 instead of 200
    - sort() does the work
- Task: refilter the dataset to get the "golden standard" data:
    - Profile taken in the beaufort Gyre
    - Deepest measurement is taken at least 2m deeper than where the AW temperature max appear.
---------------------------
2025/05/30:
- The golden standard dataset is out: we have in total of 23366 profiles.
- Max depth difference in range of 200 to 600 meters below the sea is 88m.
- Most (99%) of the abnormal depth difference occurs in ITP# 41. `05_analyze.py` looks into the dataset and would produce a summary.
----------------------------
2025/06/02:
- `06_gridMaker.py` will interpolate on a 0.25 meter grid, and save everything in a different folder called "gridData"
------------------------------
2025/06/03:
- `02_cleanData.py` mostly fixed, still need to find out a way to make it write into another folder.
--------------------------------
2025/06/04:
- `02_cleanData.py` done, new dataset would be collected and cleaned once the WHOI database has confirmed that ITP#93 has profiles.
---------------------------------
2025/06/05:
- `03_checkData.py` has been deprecated. 
- In the staricase detection algorithm, the `te_adj` and `sa_adj` were further converted with gsw functions, do we really need that?
---------------------------------
2025/06/06:
- `06_gridMaker` successfully transforms .mat into 0.25m gridded dataframe. 
- reorganized `05_analyze.py` and `06_gridMaker.py` with helper function
- Minor changes made to `Makefile`, `02_cleanData.py` and `helper.py`
---------------------------------
2025/06/09:
- `08_SOM.py` is on the way to implement Self-organizing Map to cluster single profile for assessing the quality of clustering
- considering adding a function for `06_gridMaker.py` so that it also produces `.mat` format files for easier matlab process. Note: it does not convert to v7.3 MATLAB files.
- few issue occured when experimenting file format converting, modified `02_cleanData.py` to address the issue.
- will fix more tomorrow, check 121/0604
-------------------------
2025/06/10:
- fix `06_gridMaker.py` wrong logic
- fixed `02_cleanData.py`: improved filtering logic. 
- debug log shows that these profiles are having issues. These profiles are not included in the gridded dataset:

| ITP Number \ Profile Number                                  | Reason                                         |
|--------------------------------------------------------------|------------------------------------------------|
| itp114cormat\cor3783                                         | Abnormal profile measurement around 450 m      |
| itp115cormat\cor0327                                         | Contains only measurement from 575 m to 750 m |
| itp115cormat\cor0534,0564,0582,0589,0593                     | Abnormal depth measurement                     |
| itp32cormat\cor0063                                          | Abnormal depth measurement                     |
| itp1cormat\cor1802                                           | Contains only measurement at around 700 m      |
| itp21cormat\cor0420                                          | Contains only measurement from 500 m to 750 m  |
| itp21cormat\cor0536                                          | Contains only measurement from 400 m to 450 m  |
| itp21cormat\cor0544                                          | Contains only measurement from 582.5 m to 602.5 m |
| itp21cormat\cor0700                                          | Contains only measurement from 387.5 m to 407.5 m |
| itp3cormat\cor1523                                           | Contains only measurement from 675 m to 710 m  |
| itp3cormat\cor1525                                           | Contains only measurement from 580 m to 680 m  |
| itp3cormat\cor1526                                           | Contains only measurement from 640 m to 740 m  |
| itp5cormat\cor0975                                           | Contains only measurement from 450 m to 750 m  |
| itp5cormat\cor0976                                           | Contains only measurement from 575 m to 750 m  |
| itp8cormat\cor0779                                           | Contains only measurement from 600 m to 700 m  |
| itp8cormat\cor0780                                           | Contains only measurement from 640 m to 710 m  |
| itp8cormat\cor0850                                           | Contains only measurement from 680 m to 750 m  |
----------------------------
2025/06/12:
- added `StaircaseAlgorithm.py` for reference. Will start toy analysis algorithm
----------------------------
2025/06/16:
- files in `gridDataMat` can only be loaded with `loadmat` function.
- `09_test_tag.py` reproduces staircaseAlgorithm with minor changes
-----------------------------
2025/06/17:
- The algorothm would be a time series analysis with SHAP analysis on the feature importance.
----------------------------
2025/06/18:
- `09_test_tag.py` will now make a dataframe for time series analysis
------------------------------
2025/06/23:
- `09_test_tag.py` makes transect plot for specific values, but unclear on how to skip nan value (How does nan get in there??)
-------------------------------
2025/07/02:
- `04_Plot.py` now makes the transect plots.
- `09_test_tag.py` would apply a time series analysis of logistic regression on ITP 62, 65 and 68 with metrics. ROC/AUC is needed, other metrics too
-------------------------------
2025/07/03:
- renamed 09 script, took out machine learning part, and put it in `10_machineLearning.py`
- The algorithm has very low precision on positive values, which is a problem. changing the positive class weight to 10 raised the precision from 0.06 to 0.22, imporoved recall and f1-score from 0 to 0.72 and 0.34
- keep raising the weight might reduce the precision, weight of 10 is good enough
- ROC curve shows that the current performance (with weighted class) is not as good as expected. To raise the TPR to 80%, the false positive rate would have to be raised to beyond 20%
