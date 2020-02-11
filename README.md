# zfish_mutants_analysis
Code used in the paper "Genetic control of collective behavior in zebrafish"

These 3 notebooks used to loop through and process/generate results for all lines.  These need to be run in order.  First run notebook 1 to read in the raw .mat tracking files and output smoothed and filtered .pkl files that are easier to work with.  Then run notebook 2 to calculate various quantities used in the analysis, and to save files with input-output structure to be used in models fits.  Then run notebook 3 to fit the model.
  1 - Data import and smoothing.ipynb
  2 - Process group quantities and model io.ipynb
  3 - Run model fits.ipynb

These 3 notebooks are used to create the figures in the paper.  They import processed and saved results, which are included in the 'savedresults' folder.  Thus, these notebooks run very quickly, and the figures can be re-created without running notebooks 1,2,3 described above.
  Fig2-3 - Data.ipynb
  Fig3-example-trajectories.ipynb
  Fig4-Model.ipynb

These three files and included by the above notebooks, and contain calculation, plotting, and model fit functions:
  datafunctions.py
  headingchange_models.py
  modelfit_plotfns.py
