from os import listdir
import re

folders = ['part1', 'part2', 'part3', 'part4', 'part5', 'part6', 'part7', 'part8', 'part9', 'part10']


def get_term_fre_of_folder(foldername):
    dic_fol_tf = {}
    files =  listdir('./5_que_data/' + foldername)
    #files = ['1051942legit190.txt']
    lines = []
    for fil in files:
        f = open('./5_que_data/' + foldername + '/'+fil, 'r')
        lines = f.read().replace('\n', ' ')
        tokens = lines.split(' ')[1:]
        tokens = [int(x) for x in tokens if x]
        #print tokens

        dic_term_fre = {}
        for term in tokens:
            if term not in dic_term_fre.keys():
                dic_term_fre[term] = 1
            else:
                dic_term_fre[term] += 1
        dic_fol_tf[fil] = dic_term_fre
    return dic_fol_tf

def get_data_set():
    dataset = []
    for folder in folders:
        dataset.append(get_term_fre_of_folder(folder))
    return dataset

def get_vocabulary(folds):
    vocab = []
    for fold in folds:
        for fil in fold.keys():
            vocab += fold[fil].keys()
    vocab = list(set(vocab))
    return vocab

def get_class_frm_filname(filename):
    match = re.search(r'spmsg', filename)
    if match:
        return 0
    elif re.search(r'legit', filename):
        return 1
    else:
        return 'illegal file name'

def separate_files(fol_list):
    spams = []
    legits = []
    for fold in fol_list:
        for fil in fold.keys():
            if get_class_frm_filname(fil) == 0:
                spams.append(fold[fil])
            else:
                legits.append(fold[fil])
    return spams,legits

def get_merged_dic(vocabulary, list_dic):
    freq = dict.fromkeys(vocabulary, 0)
    for dic in list_dic:
        for key in dic.keys():
            freq[key] += dic[key]
    return freq

def calc_prob(xi, N, d):
    alpha = .1
    #performing additive smoothing
    return  (xi+alpha)/float(N+alpha*d)

def calc_class_priors(class_0_fre, class_1_fre, vocabulary):
    total_fre = {}
    class_0_priors = {}
    class_1_priors = {}
    vocab_len = len(vocabulary)

    for key in vocabulary:
        total_fre[key] = class_0_fre[key] + class_1_fre[key]
        class_0_priors[key] = calc_prob(class_0_fre[key], total_fre[key], vocab_len)
        class_1_priors[key] = calc_prob(class_1_fre[key], total_fre[key], vocab_len)
        temp = class_0_priors[key]+class_1_priors[key]
        class_0_priors[key] = class_0_priors[key]/(temp)
        class_1_priors[key] = class_1_priors[key]/(temp)
    return class_0_priors, class_1_priors

def train_function(dataset, fol_list):
    #fol_list = dataset[:8]
    vocabulary = get_vocabulary(fol_list)
    spams, legits = separate_files(fol_list)

    prob_cls_0 = len(spams)/float(len(spams)+len(legits))
    prob_cls_1 = len(legits)/float(len(spams)+len(legits))

    spams_fre = get_merged_dic(vocabulary, spams)
    legits_fre = get_merged_dic(vocabulary, legits)

    spam_priors, legit_priors = calc_class_priors(spams_fre, legits_fre, vocabulary)

    return [spam_priors, legit_priors, prob_cls_0, prob_cls_1]

def test_function(fol_list, priors_0, priors_1, prob_cls_0, prob_cls_1):
    #print fol_list
    #print(len(fol_list[0].keys()))
    #print(len(fol_list[1].keys()))
    cnt = 1
    for fold in fol_list:
        for fil in fold.keys():
            true_class = get_class_frm_filname(fil)
            #print true_class
            cls0_prior = 1
            cls1_prior = 1
            for key in fold[fil].keys():
                if key in priors_0.keys():
                    cls0_prior *= priors_0[key]
                    #as vocab for both classes is same
                    cls1_prior *= priors_1[key]
                else:
                    #presently if word in test set is not in trained vocab then take its prior as 1
                    # later try smooothing
                    cls0_prior *= cls0_prior*1
                    cls1_prior *= cls1_prior*1
            cls0_prior = cls0_prior*prob_cls_0
            cls1_prior = cls1_prior*prob_cls_1

            if cls0_prior>cls1_prior:
                if 0 == true_class:
                    cnt += 1
            else:
                if 1 == true_class:
                    cnt += 1
    print cnt/float(len(fol_list[0].keys()) + len(fol_list[0].keys()))

dataset = get_data_set()

def run_five_folds():
    temp_data = dataset
    for i in xrange(0,10,2):
        folds = temp_data[:8]
        params = train_function(temp_data, folds)
        test_function(temp_data[-2:], params[0], params[1], params[2], params[3])
        temp_data = temp_data[-2:] + temp_data[:8]

run_five_folds()

#print separate_files(fol_list)[0]
#print separate_files(fol_list)[1]
#print dataset
#print (vocabulary)
#print spam_priors
#print legit_priors
#print params[0]
#print params[1]
#print params[2], params[3]