# BDanskin_SciAdv_2023
Code associated with the manuscript for Science Advances.  
This repository contains code and processed data necessary to regenerate figures from:  
> Danskin B.P. et al. Exponential history integration with diverse temporal scales in retrosplenial cortex supports hyperbolic behavior (2023). Under review for Science Advances.  
  
All necessary analysis functions are contained in the associated notebook, or in `bdanskin.py`.  
Data deposited to Dryad.
  
## Repository layout  
. BDanskin_SciAdvances_2023 ### source folder  
|-- `py_code/` ### expected location of bdanskin.py  
|-- `chr2_optogenetics/` ### zipped directory containing pkl files for analysis and plotting  
|-- `hattori_datasets_behavior/` ### nested zipped directory containing pkl files for analysis and plotting  
|-- `hattori_datasets_xarray/` ### nested zipped directory containing netcdf files in the xarray format; imaging and behavior data for analysis  
|-- `hattori_datasets_xarray_cellfits/` ### zipped directory containing netcdf files in the xarray format; the outputs of analysis and used for plotting  
|-- `bdanskin_analysis_all.ipynb` ### notebook file to run all analysis  
|-- `bdanskin_plot_figure_1.ipynb` ### notebook file to load analysis outputs and plot Fig 1  
|-- `bdanskin_plot_figure_2.ipynb` ### notebook file to load analysis outputs and plot Fig 2  
|-- `bdanskin_plot_figure_3.ipynb` ### notebook file to load analysis outputs and plot Fig 3  
|-- `bdanskin_plot_figure_4.ipynb` ### notebook file to load analysis outputs and plot Fig 4  
