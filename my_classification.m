function my_classification( classifier_choice, stem_flag, featureset_type, python_exe_str, input_filename, output_folder, STN_filename )

% input parameters
% classifier_choice --> 'NB': naivebayes, 'SVM': svm
% stem_flag --> 0: no stemming, 1: do stemming
% featureset_type --> 1: bow, 2: bow+STN, 3:STN

% call python for preprocessing
[status, result] = my_run_python_part(stem_flag, featureset_type, python_exe_str, input_filename, output_folder, STN_filename);
if status==0
    disp('python success!!!')
elseif status~=0
    status
    disp('python error!!!')
    disp(result)
    return
end

% read train-test pairs from output_folder
[ ...
train0,train1,train2,train3,train4,train5,train6,train7,train8,train9, ...
trainclass0,trainclass1,trainclass2,trainclass3,trainclass4,trainclass5,trainclass6,trainclass7,trainclass8,trainclass9, ...
test0,test1,test2,test3,test4,test5,test6,test7,test8,test9, ...
testclass0,testclass1,testclass2,testclass3,testclass4,testclass5,testclass6,testclass7,testclass8,testclass9 ...
] = my_read_inputs(output_folder);

% 10-fold CV
all_conf_mat = zeros(3,3,10);
all_accuracy = zeros(1,10);
for f=0:9
    curr_trainset = strcat('train',num2str(f));
    curr_testset = strcat('test',num2str(f));
    curr_trainset_labels = strcat('trainclass',num2str(f));
    curr_testset_labels = strcat('testclass',num2str(f));
    
    if strcmp(classifier_choice,'NB') == 1
        exp_NB = strcat('my_classify_with_NB(',curr_trainset,',',curr_trainset_labels,',',curr_testset,',',curr_testset_labels,')');
        [curr_conf_mat, curr_accuracy] = eval(exp_NB);
    elseif strcmp(classifier_choice,'SVM') == 1
        exp_SVM = strcat('my_classify_with_SVM(',curr_trainset,',',curr_trainset_labels,',',curr_testset,',',curr_testset_labels,')');
        [curr_conf_mat, curr_accuracy] = eval(exp_SVM);
    end
    all_accuracy(f+1) = curr_accuracy;
    all_conf_mat(:,:,f+1) = curr_conf_mat;    
end

my_output_results(output_folder, classifier_choice, all_accuracy,all_conf_mat);

% ---------------------- my functions -------------------------------------

    function [s, r] = my_run_python_part(stem_flag, featureset_type, python_exe_str, input_filename, output_folder, STN_filename)
        python_commandStr = [python_exe_str,' ','"E:/doktora/yeterlilik/TAKEHOME/Tunga/code/my_1preprocess_python/my_preprocess.py" ',num2str(stem_flag),' ',num2str(featureset_type),' ',input_filename,' ',output_folder,' ',STN_filename];
        [s, r] = system(python_commandStr);
    end

    % read train - test docs
    function [ ...
            train0,train1,train2,train3,train4,train5,train6,train7,train8,train9, ...
            trainclass0,trainclass1,trainclass2,trainclass3,trainclass4,trainclass5,trainclass6,trainclass7,trainclass8,trainclass9, ...
            test0,test1,test2,test3,test4,test5,test6,test7,test8,test9, ...
            testclass0,testclass1,testclass2,testclass3,testclass4,testclass5,testclass6,testclass7,testclass8,testclass9 ...
            ] = my_read_inputs(output_folder)
        
        formatSpec = '%s';
        for i = 0:9
            train_input_sample_bow_filename = strcat(output_folder,'\my_train_bow',num2str(i),'.csv');
            train_input_sample_label_filename = strcat(output_folder,'\my_train_label',num2str(i),'.txt');
            test_input_sample_bow_filename = strcat(output_folder,'\my_test_bow',num2str(i),'.csv');
            test_input_sample_label_filename = strcat(output_folder,'\my_test_label',num2str(i),'.txt');
            
            curr_train_bow = csvread(train_input_sample_bow_filename);
            curr_test_bow = csvread(test_input_sample_bow_filename);
            
            train_fileID_forlabels = fopen(train_input_sample_label_filename,'r');
            curr_train_labels = fscanf(train_fileID_forlabels,formatSpec);
            curr_train_labels = transpose(curr_train_labels);
            fclose(train_fileID_forlabels);

            test_fileID_forlabels = fopen(test_input_sample_label_filename,'r');
            curr_test_labels = fscanf(test_fileID_forlabels,formatSpec);
            curr_test_labels = transpose(curr_test_labels);
            fclose(test_fileID_forlabels);
            
            train_bow = strcat('train',num2str(i));
            test_bow = strcat('test',num2str(i));
            train_labels = strcat('trainclass',num2str(i));
            test_labels = strcat('testclass',num2str(i));
            
            exp1 = strcat(train_bow,' = curr_train_bow;');
            exp2 = strcat(test_bow,' = curr_test_bow;');
            exp3 = strcat(train_labels,' = curr_train_labels;');
            exp4 = strcat(test_labels,' = curr_test_labels;');

            eval(exp1);
            eval(exp2);
            eval(exp3);
            eval(exp4);
        end     
        
    end


    function [precision_overall,precision_p,precision_n,precision_o, ...
            recall_overall,recall_p,recall_n,recall_o] = my_find_prediction_statistics(cm)
            
        % precision = tp_c/total_predicted_as_c
        precision_p = cm(1,1)/sum(cm(:,1));
        precision_n = cm(2,2)/sum(cm(:,2));
        precision_o = cm(3,3)/sum(cm(:,3));
        precision_overall = (precision_p+precision_n+precision_o)/3;
        
        % recall = tp_c/total_gold_as_c
        recall_p = cm(1,1)/sum(cm(1,:));
        recall_n = cm(2,2)/sum(cm(2,:));
        recall_o = cm(3,3)/sum(cm(3,:));
        recall_overall = (recall_p+recall_n+recall_o)/3;
        
    end

    function [curr_conf_mat, curr_accuracy] = my_classify_with_NB(curr_trainset,curr_trainset_labels,curr_testset,curr_testset_labels)
        my_model_NB = fitcnb(curr_trainset,curr_trainset_labels,'DistributionNames','mn');
        [curr_predicted_labels,~,~] = predict(my_model_NB,curr_testset);
        
        curr_accuracy = mean(curr_predicted_labels == curr_testset_labels);
        [curr_conf_mat,~] = confusionmat(curr_testset_labels,curr_predicted_labels,'order',['p';'n';'o']);
    end

    function [curr_conf_mat, curr_accuracy] = my_classify_with_SVM(curr_trainset,curr_trainset_labels,curr_testset,curr_testset_labels)
        my_classifier = templateSVM('KernelFunction','linear','ClassNames',['p','n','o']); % 'polynomial' 'linear' 'gaussian' 'rbf'
        my_model_SVM = fitcecoc(curr_trainset,curr_trainset_labels,'Learners',my_classifier);
        [curr_predicted_labels,~,~] = predict(my_model_SVM,curr_testset);
        
        curr_accuracy = mean(curr_predicted_labels == curr_testset_labels);
        [curr_conf_mat,~] = confusionmat(curr_testset_labels,curr_predicted_labels,'order',['p';'n';'o']);
    end

    function my_output_results(output_folder, classifier_choice, all_accuracy, all_conf_mat)
        output_stats_filename = strcat(output_folder,strcat('\my_output_stats',classifier_choice,'.txt'));
        if exist(output_stats_filename, 'file')==2
          delete(output_stats_filename);
        end
        output_fid = fopen(output_stats_filename,'wt');
        
        fprintf(output_fid,'Classifier : %s \n',classifier_choice);
        fprintf(output_fid,'             10-fold cross-validation \n');
        
        my_accuracy = (sum(all_accuracy))/10;
        fprintf(output_fid,'Accuracy : %f \n',my_accuracy); 
        disp(strcat('Accuracy : ',num2str(my_accuracy)))
        
        my_conf_mat = zeros(3,3);
        for i = 1:3
            for j = 1:3
                for k = 1:10
                    my_conf_mat(i,j) = my_conf_mat(i,j)+all_conf_mat(i,j,k);
                end
            end
        end
        my_conf_mat = my_conf_mat/10;      
        
        fprintf(output_fid,'Confusion Matrix : (Gold/Prediction)\n');
        fprintf(output_fid,'  P     N       O\n');
        classes=['P','N','O'];
        for ii = 1:size(my_conf_mat,1)
            fprintf(output_fid,[classes(ii),' ']);
            fprintf(output_fid,'%g\t',my_conf_mat(ii,:));
            fprintf(output_fid,'\n');
        end
        
        [my_precision_overall,my_precision_p,my_precision_n,my_precision_o, ...
            my_recall_overall,my_recall_p,my_recall_n,my_recall_o] = my_find_prediction_statistics(my_conf_mat);

        fprintf(output_fid,'Overall Precision : %f \n',my_precision_overall);
        fprintf(output_fid,'Overall Recall : %f \n',my_recall_overall);
        fprintf(output_fid,'Precision P : %f \n',my_precision_p);
        fprintf(output_fid,'Precision N : %f \n',my_precision_n);
        fprintf(output_fid,'Precision O : %f \n',my_precision_o);
        fprintf(output_fid,'Recall P : %f \n',my_recall_p);
        fprintf(output_fid,'Recall N : %f \n',my_recall_n);
        fprintf(output_fid,'Recall O : %f \n',my_recall_o);
        
        fclose(output_fid);
    end
end

