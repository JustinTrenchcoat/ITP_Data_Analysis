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

- 