# Notebooks
This directory contains the notebooks that were used to analyze the results from CORAL. The code is split into four parts:

* __AcademicLabels.ipynb__: Analyzes CORAL's predictions on notebooks with known associations to academic publications
* __GORC-Dask.ipynb__: Performs a regular expression search across GORC to find these aforementioned associations.
* __LabelAnalysis.ipynb__: Analyzes CORAL's agreements with human annotators. 
* __MiscViz.ipynb__: Creates a visualization of CORAL's performance compared with a battery of baselines.  

Note that two files, `gorc_github_refs.csv` and `results_with_gorc.csv` are already provided as tarballs. You may use the provided scripts to generate them on your own, or you can uncompress them.
If something seems like it's missing, please reach out to the authors or raise an issue on this repo.
