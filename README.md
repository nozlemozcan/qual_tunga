## QUALIFICATION EXAM
## December 2016
## Question by Tunga Güngör
## Turkish Sentiment Analysis

# how to use system

The entry point is the Matlab file. It then calls python file.
The parameters for Matlab:
1. classifier_choice
   values: 'NB' for multinomial naive bayes classifier
           'SVM' for SVM classifier

2. stem_flag
   values: 0 for no stemming
           1 for do stemming

3. featureset_type
   values: 1 for binary bow features
           2 for binary bow features + SentiTurkNet features
           3 for only SentiTurkNet features

4. python_exe_str
   The full path and filename for python.exe file
   
5. input_filename
   The full path and filename for txt input file
   
6. output_folder
   The path for output folder that is used for intermediate files and output statistics file

7. STN_filename
   The full path and filename for txt SentiTurkNet input file


# files and folders

1. my_classification.m
   The classification part is implemented in Matlab. The entrance point of the system is in Matlab. The python part is called from m file. 

2. my_preprocess.py
   The preprocessing, feature vector preparation and train-test set preparation parts are implemented with python.
   Python version is 3.5.1.

3. dataset_reviews.xlsx and dnmAll_reviews.txt
   Dataset files
   txt file is the input for the system.
   The structure of txt file is that one sample is found in every line and the input and the class code is separated by \t (tab) character.
   
4. my_STN_dict.txt
   txt form of STN file. I converted to this form for ease of parsing.
   
5. snowballstemmer
   Turkish stemmer 
   https://pypi.python.org/pypi/snowballstemmer

6. sample_outputs
   Output files that are used for experiment results in the report.
   
7. example_calls.txt
   The call commands for the system for the experiments presented in the report.
