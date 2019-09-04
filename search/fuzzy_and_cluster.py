# -*- coding: utf-8 -*-
import pandas as pd
from fuzzywuzzy import fuzz
from fuzzywuzzy import process 
from fuzzyfinder import fuzzyfinder
#import jieba
#import jieba.posseg as pseg
#import jieba.analyse as anls
import pickle

index_dic = {}
question_str = None
what = None
why = None
how = None
what_str = None
question_index = None
searched_items = None
history_filter = []
filter_dic = None
def filter_by_zhaoxin(what, why, how):
    key = ['更换','检查','清洗','调节','拆卸','拆装']
    print (key)
    t1,t2,t3 = what, why, how
    what = []
    why = []
    how = []
    for i,x in enumerate(t3):
        if type(x) is float:
            continue
        for k in key:
            if k in x:
                what.append(t1[i])
                why.append(t2[i])
                how.append(t3[i])
    return what, why, how

def load_data():
    global question_index, what, why, how, index_dic,what_str
    csv_data = pd.read_csv('../data/crm_issue_csv.txt', '|')
    what = csv_data.loc[:,'a001']
    why = csv_data.loc[:,'a002']
    how = csv_data.loc[:,'a003'] 
    what = list(what)
    why = list(why)
    how = list(how)
    #what,why,how = filter_by_zhaoxin(what, why, how)

    with open('../search/index_dic.pickle', "rb") as fp: 
        index_dic = pickle.load(fp) 

    question_index = {x[1]:x[0] for x in enumerate(what)}
    original_count = len(what)
    simplified_what = list(set(what))
    print ('原始问题个数为：', original_count, '重复问题描述个数为：',original_count-len(simplified_what))
    print ('index count ', len(index_dic))
    global what_str
    what_str = [str(x) for x in simplified_what]

def fuzzy_match_by_key(keyword):
    keyword = "".join(keyword.split())
    print (keyword)
    history_filter.append(keyword)
    searched_items = fuzzyfinder(keyword, what_str)
    searched_items = list(searched_items)
    #searched_items.sort(key=lambda x: len(x))
    searched_items.sort(key=lambda x: len(x), reverse=True)
    return searched_items
    #print (searched_items)

def get_filter(searched_items):
    global history_filter
    new_filter_dic = {}
    print ('history filter', history_filter)
    i = 0
    print (index_dic)
    for r in searched_items:
        i += 1
        print (i, len(searched_items))
        if r in index_dic:
            terms = index_dic[r]
            print(terms)
            for t in terms:
                if t is None:
                    continue
                b_in = False
                for f in history_filter:
                    if t in f:
                        b_in = True
                        break
                if b_in:
                    continue
                else:
                    if t not in new_filter_dic:
                        new_filter_dic[t] = []
                    new_filter_dic[t].append(r) 
    #kv_list = sorted(new_filter_dic.items(), key=lambda x: len(x[1]), reverse=True)
    kv_list = sorted(new_filter_dic.items(), key=lambda x: len(x[1]), reverse=False)
    new_filter_dic = {}
    for kv in kv_list:
        new_filter_dic[kv[0]] = kv[1]

    print (new_filter_dic)

    return new_filter_dic

def filter_search(searched_items, selected_filter):
    global filter_dic,question_index,what, why,how
    if selected_filter != '':
        searched_items = filter_dic[selected_filter]
    filter_dic = get_filter(searched_items)
    print ('filter dic debug', filter_dic)
    print (type(filter_dic))
    if selected_filter != '':
        history_filter.append(selected_filter)
    print ('historical filter ', history_filter)

    n = len(searched_items)
    print ('searched ', n)
    elem_list = []
    for x in searched_items:
        elem_list.append(
                            {
                                "title": what[question_index[x]],
                                "description": how[question_index[x]],
                                "url": why[question_index[x]]
                            }
                        )
    print (len(filter_dic))
    a = filter_dic.keys()
    a = list(a)
    print ('key',len(a))
    result = {
                "total": n,
                "results": elem_list,
                "filter_count":len(filter_dic),
                "filters":list(filter_dic.keys())
            }
    return result

if __name__ == "__main__":
    print ("start")
