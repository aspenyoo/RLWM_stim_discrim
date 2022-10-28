# models

Here is a brief description of the folders and files contained within the models directory. 

Generally, there are two types of functions, simulate*.m and calc_LL_(model).m. Perhaps unsurprisingly, the simulate*.m functions simulate data and the calc_LL_(model).m functions calculate the log-likelihood of data given (model) and parameters.

Each function contains more notes about input and output variables. 

Other files:
- interferencecon: parameter constraints in optimization for the condition-specific WM-interference models (fit on Exp 2 data and shown in Supplementary materials). 
- loadfittingparams: a function that give optimization variables like upper and lower bounds, parameter constraints, and fixes parameters based on model name. 
- simulatetest*.m: functions used to simulate the test phase data only (used for exp 2)


