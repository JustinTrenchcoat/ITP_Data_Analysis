# ITP Data Analysis Project

This is my experiment field for ITP data analysis project from UBC WLIURA 2025 Summer. This project is under the research project `S25 WLIURA - Undergraduate Research Asst Arctic Oceanography` with Dr. Stephanie Waterman as the supervisor. 

## Development Log:
- we have figured out that the  [\[here\]](https://scienceweb.whoi.edu/itp/data/) contains all data we need. We should look for the `itp####cormat` files. 
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
- Task: refilter the dataset to get the "golden standard" data: find the AW temp max and go deeper than that. 
---------------------------
2025/05/30:
- The golden standard dataset is out: we have in total of 23366 profiles.
- Max depth difference in range of 200 to 600 meters below the sea is 88m. It is due to a measurement
- Most (99%) of the abnormal depth difference occurs in ITP# 41. `05_analyze.py` looks into the dataset and would produce a summary.
----------------------------
2025/06/02:
- `06_gridMaker.py` will interpolate on a 0.25 meter grid, and save everything in a different folder called "gridData"