# Lowered inter-stimulus discriminability hurts incremental, but not working memory-dependent contributions to learning

### To run the code in these scripts, you will need to download the dataset from [OSF](https://osf.io/f4hst/).
Once you download and unzip the file, you should see two folders named *data* and *fits*. Please place these in the same folder that this README is in, such that you can see fits, data, helper_functions, and models at the same time. This will allow the code to properly access the data.

Here is a brief description of the folders. Each folder has a README file to further explain its contents. 

- **anova.R**: contains the code to conduct the ANOVAs reported in the manuscript
- **helper_functions/**: contains various functions used to fit, plot, and otherwise wrangle the data.
- **models/**: contains functions to fit models
- **analysis_scripts.m**: contains example code for how to collect and fit data, as well as code used to analyze model fits and generate figures. 
- **plot_figures.m**: contains code to fit all figures found in the main manuscript as well as the Supplementary Materials. 

Human and simulated data, as well as fits, can be found at https://osf.io/f4hst/.

To create Figures 15-17, you will need an installation of [SPM](https://www.fil.ion.ucl.ac.uk/spm/). If you have this, please add its path at the top of plot_figures in the designated space. If you do not, this code will be skipped.
