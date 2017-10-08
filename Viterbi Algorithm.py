#! /usr/bin/python3
import numpy as np
from HW6 import *

# function to calculate maximum likelhood trigram estimate given the 3 tags and
# the map of trigrams and bigram as keys, and their counts as values
def compute_trigram_estimate(y_i_minus_2,y_i_minus_1,y_i, count_map):
    trigram_tuple = (y_i_minus_2,y_i_minus_1,y_i)
    bigram_tuple = (y_i_minus_2,y_i_minus_1)
    if trigram_tuple in count_map:
        count_of_trigram = count_map[trigram_tuple]
    else:
        count_of_trigram = 0
    if bigram_tuple in count_map:
        count_of_bigram = count_map[bigram_tuple]
    else:
        count_of_bigram = 0
    if count_of_bigram == 0 or count_of_trigram == 0:
        inf =  1000
        inf = inf*-1
        return inf
    else:
        qmle = float(count_of_trigram)/float(count_of_bigram)
        return np.log2(qmle)


# function to create a map of trigram and bigram as keys and their counts as values to be used for trigram estimation
def populate_bigram_trigram_counts(count_file):
    count_map = {}
    with open(count_file) as f:
        for line in f:
            words = line.split(' ')
            size = len(words)
            words[size-1] = words[size-1].replace("\n", "")
            if words[1] == '1-GRAM' or words[1] =='WORDTAG':
                continue
            elif words[1] =='2-GRAM':
                count_map[(words[2],words[3])] = int( words[0])
            else:
                count_map[(words[2],words[3],words[4])] = int( words[0])
    return count_map

# function to create dictionary of tags as keys and they summed counts as values
# and to create a dictionary woth words as keys, and a list of tuples of their possible tags and counts
def map_counts_tag_to_word(count_file):
    word_with_tag_map = {}
    tag_count_map = {}
    with open(count_file) as f:
        for line in f:
            words = line.split(' ')
            if words[1] == "WORDTAG":
                word = words[3].replace("\n", "")
                if word not in word_with_tag_map:
                    word_with_tag_map[word] = [(words[2],int(words[0]))]
                else:
                    word_with_tag_map[word].append((words[2],int(words[0])))
                if words[2] not in tag_count_map:
                    tag_count_map[words[2]] = int(words[0])
                else:
                    tag_count_map[words[2]] = tag_count_map[words[2]] + int(words[0])
    return word_with_tag_map,tag_count_map

# function to compute emission based on the overall tag count and count of the tag of a particular word
def compute_emission(word_tag_count,tag_count):
    if tag_count > 0:
        e =  float(word_tag_count)/float(tag_count)
        return e
    else:
        return 0

# function to create list of possible tags and word-tag counts
def get_tag_dictionary_and_word_given_tag_counts(word,word_with_tag_map,count_file, word_given_tag =None):
    tag_list = []
    word_given_tag_counts = {}
    if word in word_with_tag_map:
        for tag_count_tuple in  word_with_tag_map[word]:
            tag , count = tag_count_tuple
            tag_list.append(tag)
            word_given_tag_counts[(word,tag)] = count
    else:
        if count_file == 'ner_grouped.counts':
            word_category = find_category(word)
        else:
            word_category = '_RARE_'
        for tag_count_tuple in  word_with_tag_map[word_category]:
            tag , count = tag_count_tuple
            tag_list.append(tag)
            word_given_tag_counts[(word,tag)] = count

    if word_given_tag is None:
        return tag_list
    else:
        return tag_list , word_given_tag_counts


def viterbi_algorithm(sentence):
    X = sentence
    n = len(X)
    Y = ['']*n

    # this value will be changed depending what count file is to be used.
    # Use ner_grouped.counts to run algorithm on bucketed/grouped (into categories other than _RARE_) data
    count_file = "4_1.txt"

    word_with_tag_map , tag_count_map = map_counts_tag_to_word(count_file)
    count_map = populate_bigram_trigram_counts(count_file)
    pi = {}
    bp = {}
    pi[(-1,'*','*')] = 0 # the index is -1 because our sentence words start from index 0

    # iterating over the sentence and populating the pi and bp (backpointers) values
    for k in range(0, n):
        if k==0:
            tag_dict_u = ['*']
            tag_dict_v ,word_tag_count_v = get_tag_dictionary_and_word_given_tag_counts(X[k],word_with_tag_map,count_file, True)
        else:
            tag_dict_u = get_tag_dictionary_and_word_given_tag_counts(X[k-1],word_with_tag_map, count_file)
            tag_dict_v,word_tag_count_v = get_tag_dictionary_and_word_given_tag_counts(X[k],word_with_tag_map,count_file,True)

        for u in tag_dict_u:
            for v in tag_dict_v:
                if k==0 or k==1:
                    tag_dict_w = ['*']
                else:
                    tag_dict_w = get_tag_dictionary_and_word_given_tag_counts(X[k-2],word_with_tag_map,count_file)
                max_val = np.finfo(np.float64).min
                max_w = None
                for w in tag_dict_w:
                    a =  pi[(k-1,w,u)]
                    b =  compute_trigram_estimate(w,u,v, count_map)
                    c =  np.log2(compute_emission(word_tag_count_v[(X[k],v)],tag_count_map[v]))
                    val = a+ b+ c
                    if val > max_val:
                        max_val = val
                        max_w = w
                pi[(k,u,v)] = max_val
                bp[(k,u,v)] = max_w
    if n > 1:
        tag_dict_u = get_tag_dictionary_and_word_given_tag_counts(X[n-2],word_with_tag_map, count_file)
    else:
        tag_dict_u = ['*']
    tag_dict_v = get_tag_dictionary_and_word_given_tag_counts(X[n-1],word_with_tag_map, count_file)
    max_val = np.finfo(np.float64).min
    for u in tag_dict_u:
        for v in tag_dict_v:
            a =  pi[(n-1,u,v)]
            b =  compute_trigram_estimate(u,v,'STOP', count_map)
            val = a+b
            if val > max_val:
                max_val = val
                if n > 1:
                    Y[n-2] = u
                    Y[n-1] = v
                else:
                    Y[n-1] = v

    if n > 2:
        for k in reversed(range(0,n-2)):
            Y[k] = bp[(k+2,Y[k+1],Y[k+2])]

    max_log_probabilities = []
    for k in range(0,n):
        if k==0:
            max_log_probabilities.append(pi[(k,'*',Y[k])])
        else:
            max_log_probabilities.append(pi[(k,Y[k-1],Y[k])])

    return X,Y,max_log_probabilities


dev_file = 'ner_dev.dat'
final_output = ""
current_sentence = []
with open(dev_file) as f:
    for line in f:
        if len(line.strip())==0:
            # the viterbi algorithm function is called when we have a complete sentence i.e. when our iterator encounters a blank line
            X, Y, probability = viterbi_algorithm(current_sentence)
            for i in range(0,len(X)):
                final_output = final_output+X[i]+ " "+ Y[i] + " "+ str(probability[i]) +'\n'
            final_output = final_output+'\n'
            current_sentence = []
            continue
        else:
            line = line.replace("\n", "")
            current_sentence.append(line)

# the output is cummulated in the variable 'final_output' and is printed in the output file at the end
with open("5_2.txt", 'w+') as file:
    file.write(final_output)