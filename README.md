# ITP Data Analysis Project

This is my experiment field for ITP data analysis project from UBC WLIURA 2025 Summer. This project is under the research project `S25 WLIURA - Undergraduate Research Asst Arctic Oceanography` with Dr. Stephanie Waterman as the supervisor. 

## Development Log:
- we have figured out that the  [\[here\]](https://scienceweb.whoi.edu/itp/data/) contains all data we need. We should look for the `itp####cormat` files. 
- `test.py` and `matlabRead.py` have scripts that could read from .mat files. The next step is to filter them through our requirement. The question now is whether we need to convert them into .nc or other format after filtering through them
- Since the dataset is totally different, we might not need the itp package?? Or we can write our own package
- We should also read through the clustering method and figure out how it works.
- in the `test.py`, we might need to pay attention to the data structure when unpacking the `.mat`file variables. the ordering is weird.