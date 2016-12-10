"""
QUALIFICATION 2016
Nuriye Özlem Özcan Şimşek
2013800072
QUESTION: TUNGA GÜNGÖR
SUBJECT: Turkish Sentiment Analysis
"""
import string
import sys, os
import random
import math
import shutil
import re
import collections
import traceback
import datetime
import snowballstemmer as sb


def my_print(text):
    print(text)


def print_error(text):
    print('Error: ', text)


def write_list(r, filename):
    if os.path.exists(filename):
        os.remove(filename)
    input_filename = open(filename, "a", encoding='utf-8')
    for a in r:
        input_filename.write(str(a)+'\n')
    input_filename.close()


def write_list_as_csv(r, filename):
    if os.path.exists(filename):
        os.remove(filename)
    input_filename = open(filename, "a", encoding='utf-8')
    for a in r:
        input_filename.write(str(a).replace('[','').replace(']','')+'\n')
    input_filename.close()


def my_separate_samples(read_input_lines, stem_flag):

    input_splitted_list = []
    input_class_list = []

    if stem_flag == '1':
        print('stemmer')
        my_stemmer = sb.stemmer('turkish')

    for curr_line in read_input_lines:
        curr_line2 = curr_line.lower()
        exclude = string.punctuation
        curr_line3 = ''.join(ch for ch in curr_line2 if ch not in exclude)
        curr_line4 = curr_line3.split('\t')
        curr_sample = curr_line4[0].split()
        curr_sample = list(set(curr_sample))
        curr_class = curr_line4[1].replace('\n','')

        if stem_flag == '1':
            stemmed_curr_sample = []
            for wt in curr_sample:
                if len(wt) > 5:
                    stemmed_curr_sample.append(my_stemmer.stemWord(wt))
                else :
                    stemmed_curr_sample.append(wt)
            curr_sample = stemmed_curr_sample

        input_splitted_list.append(curr_sample)
        input_class_list.append(curr_class)

    return input_splitted_list,input_class_list




def my_read_input(input_filename,stem_flag):
    my_print("my_read_input")

    words = []
    train_input_split_list = []
    train_input_class_list = []
    test_input_split_list = []
    test_input_class_list = []

    # read input file
    with open(input_filename, 'r', encoding='windows-1254') as current_opened_file:
        read_input_lines = current_opened_file.readlines()
    current_opened_file.close()

    # separate samples
    [input_splitted_list, input_class_list] = my_separate_samples(read_input_lines,stem_flag)

    # shuffle
    input_all = list(zip(input_splitted_list, input_class_list))
    random.shuffle(input_all)
    random.shuffle(input_all)
    r_input_splitted_list,r_input_class_list = zip(*input_all)

    # separate into train-test pairs (10-fold) and find vocabulary
    N = len(r_input_splitted_list)  # all input size
    n = math.floor(N/10)  # fold size

    all_indexes = list(range(N))
    for f in range(10):  # starts from 0 upto 9
        testset_indexes = list(range(f*n,(f+1)*n))
        if f == 9:
            testset_indexes = list(range(f*n,N))
        trainset_indexes = list(set(all_indexes) - set(testset_indexes))
        trainset = [r_input_splitted_list[i] for i in trainset_indexes]
        trainset_labels = [r_input_class_list[i] for i in trainset_indexes]
        testset = [r_input_splitted_list[i] for i in testset_indexes]
        testset_labels = [r_input_class_list[i] for i in testset_indexes]
        train_input_split_list.append(trainset)
        train_input_class_list.append(trainset_labels)
        test_input_split_list.append(testset)
        test_input_class_list.append(testset_labels)

        words_train = []
        for t in trainset:
            for wt in t:
                if wt not in words_train:
                    words_train.append(wt)
        words.append(words_train)

    return train_input_split_list, train_input_class_list, test_input_split_list, test_input_class_list, words


def my_convert_to_bow(input_split_list,words):

    input_bow_list = []

    for f in range(10):
        curr_words = words[f]
        word_count = len(curr_words)
        curr_sample_list = input_split_list[f]
        curr_input_bow_list=[]
        for s in curr_sample_list:
            curr_bow = [0] * word_count
            i=0
            for w in curr_words:
                if w in s:
                    curr_bow[i] = 1
                i = i+1
            curr_input_bow_list.append(curr_bow)
        input_bow_list.append(curr_input_bow_list)

    return input_bow_list


def my_convert_with_sentiturknet(input_split_list, words, my_dict):

    # sentiturknet features:
    #   1: p labelled word count
    #   2: n labelled word count
    #   3: o labelled word count
    #   4: p polarity sum
    #   5: n polarity sum
    #   6: o polarity sum

    input_list = []

    for f in range(10):
        curr_words = words[f]
        word_count = len(curr_words)
        curr_sample_list = input_split_list[f]
        curr_input_bow_list=[]
        for s in curr_sample_list:
            curr_flist = [0] * 6
            for w in curr_words:
                if w in s:
                    if w in my_dict.keys():
                        curr_word_pollabel = my_dict[w]['pollabel'];
                        if curr_word_pollabel == 'p':
                            curr_flist[0] = curr_flist[0]+1
                        elif curr_word_pollabel == 'n':
                            curr_flist[1] = curr_flist[1]+1
                        else:
                            curr_flist[2] = curr_flist[2]+1

                        curr_flist[3] = curr_flist[3]+my_dict[w]['polpos']
                        curr_flist[4] = curr_flist[4]+my_dict[w]['polneg']
                        curr_flist[5] = curr_flist[5]+my_dict[w]['polobj']
            curr_input_bow_list.append(curr_flist)
        input_list.append(curr_input_bow_list)

    return input_list


def read_STN_dict(STN_filename):
    my_dict={}

    for line in open(STN_filename):
        k,v1,v2,v3,v4=line.split(";")
        my_dict[k] = {"pollabel":v1,"polneg":float(v2),"polobj":float(v3),"polpos":float(v4)}

    return my_dict

def check_words_with_dict(words,my_dict):

    count=0
    for wl in words:
        for w in wl:
            if w in my_dict.keys():
                count=count+1

    if count>0:
        print('VAR '+str(count))
    else:
        print('YOK')

# Define a main() function that manages requests
def main():

    try:
        start_time = datetime.datetime.now()
        main_exception_message = 'HOHOHO'

        stem_flag = sys.argv[1]
        featureset_type = sys.argv[2]  # 1:bow 2:bow+sentiturknet 3:sentiturknet
        input_filename = sys.argv[3]
        output_folder = sys.argv[4]
        STN_filename = sys.argv[5]

        # parse input file and break into 10 folds train - test pairs
        [train_input_split_list, train_input_class_list,test_input_split_list, test_input_class_list, words] = my_read_input(input_filename,stem_flag)
        # convert word lists to featureset
        if featureset_type == '1':
            train_input_bow_list = my_convert_to_bow(train_input_split_list, words)
            test_input_bow_list = my_convert_to_bow(test_input_split_list, words)
        elif featureset_type == '2':
            print("sentiturknet 2")
            train_input_bow_list=[]
            test_input_bow_list=[]
            my_dict_STN = read_STN_dict(STN_filename)
            check_words_with_dict(words,my_dict_STN)
            train1 = my_convert_to_bow(train_input_split_list, words)
            test1 = my_convert_to_bow(test_input_split_list, words)
            train2 = my_convert_with_sentiturknet(train_input_split_list, words, my_dict_STN)
            test2 = my_convert_with_sentiturknet(test_input_split_list, words, my_dict_STN)

            for f in range(10):
                train_len = len(train1[f])
                curr_train = []
                for i in range(train_len):
                    curr_train.append([train1[f][i], train2[f][i]])
                train_input_bow_list.append(curr_train)

                test_len = len(test1[f])
                curr_test = []
                for j in range(test_len):
                    curr_test.append([test1[f][j], test2[f][j]])
                test_input_bow_list.append(curr_test)

        else:
            print("sentiturknet 3")
            my_dict_STN = read_STN_dict(STN_filename)
            check_words_with_dict(words,my_dict_STN)
            train_input_bow_list = my_convert_with_sentiturknet(train_input_split_list, words, my_dict_STN)
            test_input_bow_list = my_convert_with_sentiturknet(test_input_split_list, words, my_dict_STN)


        # write out train - test pairs and vocabulary
        for i1 in range(len(train_input_bow_list)):
            output_bow_filename = output_folder+"/my_train_bow"+str(i1)+".csv"
            write_list_as_csv(train_input_bow_list[i1], output_bow_filename)

        for i2 in range(len(test_input_bow_list)):
            output_bow_filename = output_folder+"/my_test_bow"+str(i2)+".csv"
            write_list_as_csv(test_input_bow_list[i2], output_bow_filename)

        for j1 in range(len(train_input_class_list)):
            output_label_filename = output_folder+"/my_train_label"+str(j1)+".txt"
            write_list(train_input_class_list[j1], output_label_filename)

        for j2 in range(len(test_input_class_list)):
            output_label_filename = output_folder+"/my_test_label"+str(j2)+".txt"
            write_list(test_input_class_list[j2], output_label_filename)

        for k in range(len(words)):
            output_vocabulary_filename = output_folder+"/my_vocabulary"+str(k)+".txt"
            write_list(words[k], output_vocabulary_filename)

        my_print(main_exception_message)
        finish_time = datetime.datetime.now()
        time_difference = finish_time - start_time
        my_print("time: " + str(time_difference))
    except Exception:
        print("Exception in user code:")
        print("-"*60)
        traceback.print_exc(file=sys.stdout)
        print("-"*60)
        print(traceback._cause_message, file=sys.stderr)
        sys.exit(1)

# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()
