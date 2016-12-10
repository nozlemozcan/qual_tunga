# QUALIFICATION EXAM
## December 2016
## Question by Tunga Güngör
## Turkish Sentiment Analysis

### how to use system

The entry point is the Matlab file. It then calls python file.<br />
The parameters for Matlab:<br />

1. classifier_choice<br />
   values: 'NB' for multinomial naive bayes classifier<br />
           'SVM' for SVM classifier<br />

2. stem_flag<br />
   values: 0 for no stemming<br />
           1 for do stemming<br />

3. featureset_type<br />
   values: 1 for binary bow features<br />
           2 for binary bow features + SentiTurkNet features<br />
           3 for only SentiTurkNet features<br />

4. python_exe_str<br />
   The full path and filename for python.exe file<br />
   
5. input_filename<br />
   The full path and filename for txt input file<br />
   
6. output_folder<br />
   The path for output folder that is used for intermediate files and output statistics file<br />

7. STN_filename<br />
   The full path and filename for txt SentiTurkNet input file<br />


### files and folders

1. my_classification.m<br />
   The classification part is implemented in Matlab. The entrance point of the system is in Matlab. The python part is called from m file. <br />

2. my_preprocess.py<br />
   The preprocessing, feature vector preparation and train-test set preparation parts are implemented with python.<br />
   Python version is 3.5.1.<br />

3. dataset_reviews.xlsx and dnmAll_reviews.txt<br />
   Dataset files<br />
   txt file is the input for the system.<br />
   The structure of txt file is that one sample is found in every line and the input and the class code is separated by \t (tab) character.<br />
   
4. my_STN_dict.txt<br />
   txt form of STN file. I converted to this form for ease of parsing.<br />
   
5. snowballstemmer<br />
   Turkish stemmer <br />
   https://pypi.python.org/pypi/snowballstemmer<br />

6. sample_outputs<br />
   Output files that are used for experiment results in the report.<br />
   
7. example_calls.txt<br />
   The call commands for the system for the experiments presented in the report.<br />
