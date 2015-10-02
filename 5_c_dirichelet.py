from os import listdir
import re
import math
import numpy as np
folders = ['part1', 'part2', 'part3', 'part4', 'part5', 'part6', 'part7', 'part8', 'part9', 'part10']
from sklearn.metrics import precision_recall_fscore_support,precision_recall_curve
from matplotlib import pylab as plt
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
    vocab_len = len(vocabulary)
    max_wrd = np.max(np.asarray(class_0_fre.keys()))
    #print max_wrd
    spam_array = np.zeros(max_wrd+1)
    legit_array = np.zeros(max_wrd+1)

    for key in vocabulary:
        spam_array[key] = class_0_fre[key]
        legit_array[key] = class_1_fre[key]

    alpha = 1*np.ones(24748)
    prior_cl0 =  np.divide((alpha + spam_array),(np.ones(24748)*np.sum(alpha+spam_array)))
    para_cls_1 = 2*np.ones(24748) + legit_array
    prior_cl1 =  np.random.dirichlet(alpha=para_cls_1)

    return prior_cl0, prior_cl1

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


def get_freq_spam_ham(fol_list):
    vocabulary = get_vocabulary(fol_list)
    spams, legits = separate_files(fol_list)

    prob_cls_0 = len(spams)/float(len(spams)+len(legits))
    prob_cls_1 = len(legits)/float(len(spams)+len(legits))

    spams_fre = get_merged_dic(vocabulary, spams)
    legits_fre = get_merged_dic(vocabulary, legits)

    spam_priors, legit_priors = calc_class_priors(spams_fre, legits_fre, vocabulary)

    return [spam_priors, legit_priors]

def norm_pred_prob(lis_prob):
    prob_array = np.asarray(lis_prob)
    prob_array = np.subtract(prob_array,np.min(prob_array))
    prob_array = np.divide(prob_array, np.max(prob_array))
    return prob_array

def new_test_function(fol_list, priors_0, priors_1):
    #print fol_list
    #print(len(fol_list[0].keys()))
    #print(len(fol_list[1].keys()))
    true_label = []
    predicted_label = []
    predicted_prob = []
    cnt = 1
    for fold in fol_list:
        for fil in fold.keys():
            true_class = get_class_frm_filname(fil)
            true_label.append(true_class)
            #print true_class
            cls0_prior = 0
            cls1_prior = 0
            for key in fold[fil].keys():
                if priors_0[key] != 0:
                    cls0_prior += math.log(priors_0[key])
                    #as vocab for both classes is same
                    cls1_prior += math.log(priors_1[key])
                else:
                    #presently if word in test set is not in trained vocab then take its prior as 1
                    # later try smooothing
                    cls0_prior += math.log(1)
                    cls1_prior += math.log(1)
            #cls0_prior = cls0_prior*prob_cls_0
            #cls1_prior = cls1_prior*prob_cls_1

            if cls0_prior>cls1_prior:
                predicted_label.append(0)
                #predicted_prob.append(cls0_prior)
                #predicted_prob.append(cls0_prior/float(cls0_prior+cls1_prior))
                predicted_prob.append(cls0_prior-cls1_prior)
                if 0 == true_class:
                    cnt += 1
            elif cls0_prior<cls1_prior:
                predicted_label.append(1)
                #predicted_prob.append(cls1_prior)
                #predicted_prob.append(cls1_prior/float(cls0_prior+cls1_prior))
                predicted_prob.append(cls0_prior-cls1_prior)

                if 1 == true_class:
                    cnt += 1
            else:
                print 'equal priors'
    predicted_prob = norm_pred_prob(predicted_prob)
    predicted_prob = np.ones(len(predicted_label)) - predicted_prob
    print cnt/float(len(fol_list[0].keys()) + len(fol_list[0].keys()))
    return true_label, predicted_label, predicted_prob


dataset = get_data_set()

def run_five_folds():
    temp_data = dataset
    pc = np.zeros(2)
    rc = np.zeros(2)
    fc = np.zeros(2)
    for i in xrange(0,10,2):
        folds = temp_data[:8]
        params = get_freq_spam_ham(folds)
        true, predicted, pred_prob = new_test_function(temp_data[-2:], params[0], params[1])
        p, r, f,z = precision_recall_fscore_support(true, predicted, labels=[0,1])
        pc += p
        rc += r
        fc += f
        temp_data = temp_data[-2:] + temp_data[:8]
    print pc/float(5), rc/float(5), fc/float(5)
#run_five_folds()


def plot_five_foldsPR():
    temp_data = dataset
    pc = np.zeros(2)
    rc = np.zeros(2)
    fc = np.zeros(2)
    for i in xrange(0,10,2):
        folds = temp_data[:8]
        params = get_freq_spam_ham(folds)
        true, predicted, pred_prob = new_test_function(temp_data[-2:], params[0], params[1])
        p, r, f,z = precision_recall_fscore_support(true, predicted, labels=[0,1])
        pc += p
        rc += r
        fc += f
        temp_data = temp_data[-2:] + temp_data[:8]
        prec, reca,abc = precision_recall_curve(true, pred_prob)
    print pc/float(5), rc/float(5), fc/float(5)
    plt.plot(reca, prec)
    plt.axis([0, 1, 0, 1.1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curve for Spam Class in Dirichelet')
    print np.mean(np.asarray(true))
    plt.show()
    #print abc
plot_five_foldsPR()



#print separate_files(fol_list)[0]
#print separate_files(fol_list)[1]
#print dataset
#print (vocabulary)
#print spam_priors
#print legit_priors
#print params[0]
#print params[1]
#print params[2], params[3]