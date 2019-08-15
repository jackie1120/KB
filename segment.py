# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pickle
from collections import Counter
import string
from zhon import hanzi
import re
import jieba
import jieba.posseg as pseg
import jieba.analyse as anls

pd.set_option('display.max_columns', None)

def cal_keyword_frequency(csv_data, key_dic):
    i = 0
    print ('to cal keyword frequence, data length is ', len(csv_data))
    print ('type', type(csv_data))
    if type(csv_data) == type([]):
        for x in csv_data:
            if i % 1000 == 0:
                print (i)
            i += 1
            if type(x) == type([]):
                for y in x:
                    if y not in key_dic:
                        key_dic[y] = 1
                    else:
                        key_dic[y] += 1
            else:
                if x not in key_dic:
                    key_dic[x] = 1
                else:
                    key_dic[x] += 1

def combine_dict(dic1,dic2):
    for k in dic2:
        if k not in dic1:
            dic1[k] = dic2[k]
        else:
            dic1[k] += dic2[k]

    return dic1

def dump_key_dic():
    csv_data_small = pd.read_csv('../data/small.csv')
    print ('small data count is ', len(csv_data_small))
    csv_data_small = csv_data_small.drop_duplicates()
    print ('after remove duplicate, small data count is ', len(csv_data_small))
    data1 = np.array(csv_data_small).tolist()
    complex_key_dic =   {}
    simple_key_dic =    {}
    machine_dic =       {}
    issue_dic =         {}
    company_dic =       {}
    person_dic =        {}
    date_dic =          {}
    question_dic =      {}
    #    cal_keyword_frequency(data1, key_dic)
    columns = csv_data_small.columns.values.tolist()
    for c in columns:
        print (c)
        #if c != 'prcomp':
            #continue
        column_data = list(csv_data_small[c])
        #result = Counter(column_data)
        column_dic = pd.value_counts(column_data)
        #column_dic = dict(column_dic)
        column_dic = column_dic.to_dict()
        if c == 'a001':
            question_dic = combine_dict(question_dic, column_dic)
        if c == 'a001' or c == 'a002' or c == 'a003':
            complex_key_dic = combine_dict(complex_key_dic, column_dic)
            print ('after combine complex dic len ', len(complex_key_dic))
        else:
            simple_key_dic = combine_dict(simple_key_dic, column_dic)
            print ('after combine simple dic len ', len(complex_key_dic))
        if c == 'division_desc'             \
            or c  == 'prsys_desc'           \
            or c  == 'process_type_desc'    \
            or c == 'prpart_desc'           \
            or c == 'prcomp_desc':
            machine_dic = combine_dict(machine_dic, column_dic)
        if c == 'prlose_desc':
            issue_dic = combine_dict(issue_dic, column_dic)
    print ('complex dic len ', len(complex_key_dic))
    print ('simple dic len ', len(simple_key_dic))

    csv_data_history = pd.read_csv('../data/crm_issue_csv.txt', '|')
    print ('historical data count is ', len(csv_data_history))
    csv_data_history = csv_data_history.drop_duplicates()
    print ('after remove duplicate, history data count is ', len(csv_data_history))
    columns = csv_data_history.columns.values.tolist()
    for c in columns:
        print (c)
        #if c != 'prcomp':
            #continue
        column_data = list(csv_data_history[c])
        #result = Counter(column_data)
        column_dic = pd.value_counts(column_data)
        #column_dic = dict(column_dic)
        column_dic = column_dic.to_dict()
        print ('len ', len( column_dic))
        #print (c,column_dic)
        #print ('len ', len( column_dic))
        if c == 'a001':
            question_dic = combine_dict(question_dic, column_dic)
        if c == 'a001' or c == 'a002' or c == 'a003':
            complex_key_dic = combine_dict(complex_key_dic, column_dic)
            print ('after combine len ', len(complex_key_dic))
        else:
            simple_key_dic = combine_dict(simple_key_dic, column_dic)
            print ('after combine simple dic len ', len(complex_key_dic))
        if c == 'division_desc'             \
            or c  == 'dist_desc'            \
            or c  == 'process_type_desc'    \
            or c  == 'srvtype_desc'         \
            or c  == 'prsys_desc'           \
            or c == 'prpart_desc'           \
            or c == 'prcomp_desc'           \
            or c == 'prpos_desc'           \
            or c == 'prclass_desc':
            #print (c,column_dic)
            machine_dic = combine_dict(machine_dic, column_dic)
        if c == 'prlose_desc':
            issue_dic = combine_dict(issue_dic, column_dic)

    print ('complex dic len ', len(complex_key_dic))
    print ('simple dic len ', len(simple_key_dic))

    kv_list = sorted(complex_key_dic.items(),key=lambda item:item[1], reverse = True)
    complex_key_dic = {}
    for kv in kv_list:
        complex_key_dic[kv[0]] = kv[1]
    with open("./complex_key.pickle", "wb") as fp:
        pickle.dump(complex_key_dic, fp, protocol = pickle.HIGHEST_PROTOCOL)      

    complex_key_dic = pd.DataFrame(kv_list)
    complex_key_dic.to_csv('./complex_key.csv')

    kv_list = sorted(simple_key_dic.items(),key=lambda item:item[1], reverse = True)
    simple_key_dic = {}
    for kv in kv_list:
        simple_key_dic[kv[0]] = kv[1]
    with open("./simple_key.pickle", "wb") as fp:
        pickle.dump(simple_key_dic, fp, protocol = pickle.HIGHEST_PROTOCOL)      

    simple_key_dic = pd.DataFrame(kv_list)
    simple_key_dic.to_csv('./simple_key.csv')

    pd.DataFrame(machine_dic.items()).to_csv('./machine.csv', index=0)
    pd.DataFrame(issue_dic.items()).to_csv('./issue.csv', index=0)

    with open("./question_dic.pickle", "wb") as fp:
        pickle.dump(question_dic, fp, protocol = pickle.HIGHEST_PROTOCOL)      

#dump_key_dic()
#exit()

#machine = pd.read_csv('./machine_unique.csv')
#machine = machine['0']
#print (machine.shape[0])
#machine = machine.drop_duplicates()
#machine.to_csv('./machine_unique_2.csv', index = 0)
#print (machine.shape[0])
#print (machine.head())
#issue = pd.read_csv('./issue_unique.csv')
#issue = issue['0']
#print (issue.shape[0])
#issue = issue.drop_duplicates()
#issue.to_csv('./issue_unique_2.csv', index = 0)
#print (issue.shape[0])
#print (issue.head())
#exit()

def is_chinese(key):
    if type(key) == type(0):
        return False
    if type(key) == type(0.1):
        return False
    b_is = False
    for c in key:
        if '\u4e00' <= c <= '\u9fff':
            b_is = True
    return b_is

def remove_not_chinese(in_path_dic, out_path_dic, log_path_dic):
    key_dic = {}
    with open(in_path_dic, "rb") as fp: 
        key_dic = pickle.load(fp) 
    print ('load key dic cont is ', len(key_dic))
    key_dic_processed = {}
    for k in key_dic:
        if is_chinese(k) == False:
            print ('illegal key ', k)
        else:
            key_dic_processed[k] = key_dic[k]
    with open(out_path_dic, "wb") as fp:
        pickle.dump(key_dic_processed, fp, protocol = pickle.HIGHEST_PROTOCOL)      
    print ('after remove illegal key, dic count is ', len(key_dic_processed))
    key_dic = pd.DataFrame(list(key_dic_processed.items()))
    key_dic.to_csv(log_path_dic)

#remove_not_chinese('./simple_key.pickle', './simple_key_processed.pickle', './simple_key_processed_log.csv')
#remove_not_chinese('./complex_key.pickle', './complex_key_processed.pickle', './complex_key_processed_log.csv')
#remove_not_chinese('./question_dic.pickle', './question_dic_processed.pickle', './question_dic_processed_log.csv')
#exit()

punc = string.punctuation
zhon_punc = hanzi.punctuation
punc = punc + zhon_punc
def remove_exceptions():
    for c in "()（）":
        print (c)
        punc = punc.replace(c,'')
#print (punc)
#print (list(punc))
def split_by_punc(k):
    result = re.split(r"["+punc+"\s]\s*",k)
    return result

def separate_sentence(src,tar,log):
    key_dic = {}
    with open(src, "rb") as fp: 
        key_dic = pickle.load(fp) 
    complex_dic = {}
    for k in key_dic:
        ks = split_by_punc(k)
        for x in ks:
            if len(x)>0:
                if x in complex_dic:
                    complex_dic[x] += key_dic[k]
                else:
                    complex_dic[x] = key_dic[k]
    with open(tar, "wb") as fp:
        pickle.dump(complex_dic, fp, protocol = pickle.HIGHEST_PROTOCOL)      
    complex_dic = pd.DataFrame(list(complex_dic.items()))
    complex_dic.to_csv(log)

#separate_sentence('./complex_key_processed.pickle', 'complex_key_separated.pickle', 'complex_key_separated.csv')
#separate_sentence('./question_dic_processed.pickle', 'question_dic_separated.pickle', 'question_dic_separated.csv')

def find_best_semantic_seg(s, key_list):
    key_dic = {}
    for k in key_list:
        if k[0] not in key_dic:
            key_dic[k[0]] = [k]
        else:
            key_dic[k[0]].append(k)

    dp = []
    for i in range(len(s)+1):
        dp.append([])
        for j in range(len(s)+1):
            dp[i].append('')

    for k in key_list:
        dp[k[0]][k[1]]=k[2]

    for i in range(1, len(s)):
        for j in range(len(s)-i):
            if dp[j][i] != '':
                break
            for k in range(i):
                if dp[j][k] != '' and dp[j+k][i+j] != '':
                    dp[j][i] = dp[j][k]+'|'+dp[j+k][i+j]
                    break

    result = [[i,j,y] for i,x in enumerate(dp) for j,y in enumerate(x)]
    result = sorted(result, key=lambda x:(x[0],100000 - (x[1]-x[0])))

    if len(result) == 0:
        return []
    start = None
    temp = []
    for i,x in enumerate(result):
        if x[2] == '':
            continue
        if x[2] != '' and start == None:
            start = x
            continue
        if x[0] >= start[0] and x[1] <= start[1]:
            continue
        else:
            temp.append(start)
            start = x
    if start != None and start not in temp:
        temp.append(start)
    print ('best semantic seg ', temp)

    return temp
    

def find_key_in_dict(sentence, dictionary):
    result = []
    for i in range(len(sentence)):
        for j in range(i+1, len(sentence)+1):
            if sentence[i:j] in dictionary:
                result.append([i,j,sentence[i:j]])
    temp = result
    result = []
    for i,a in enumerate(temp):
        b_in = False
        for j,b in enumerate(temp):
            if i == j:
                continue
            if a in b:
                b_in = True
                break
        if b_in == False:
            result.append(a)

    return result

def find_pattern_by_pos(s, semantic_seg, semantic_flag):
    pos_index = []
    pos_flag = []
    start = 0
    pattern = ''
    total_seg = []
    print ('input', semantic_seg, semantic_flag)

    word_flags = pseg.cut(s)
    for word, flag in word_flags:
        pos_index.append((start, start+len(word)))
        start += len(word)
        pos_flag.append((flag, word))
    print ('pos seg ', pos_flag)

    semantic_seg_merge = []
    start = None
    for i in range(len(semantic_seg)):
        print ('debug', i, start)
        if start == None:
            start = list(semantic_seg[i])
            continue
        else:
            if semantic_seg[i][0] == start[1]:
                start[1] = semantic_seg[i][1]
                start[2] += semantic_seg[i][2]
            else:
                semantic_seg_merge.append(start)
                start = list(semantic_seg[i])
    print ('debug', start)
    if start !=  None:
        semantic_seg_merge.append(start)

    print ('merge seg ', semantic_seg_merge)
    print ('segmantic seg ', semantic_seg)
    for p in pos_index:
        for s in semantic_seg_merge:
            if s[0] > p[0] and s[0] < p[1] or s[1] > p[0] and s[1] < p[1]:
                print ('conflict by pos', s, p, semantic_seg, pos_index)

    print ('semantic seg', semantic_seg)
    semantic = ''
    start = 0
    for i,p in enumerate(pos_index):
        b_in = False
        for s in semantic_seg_merge:
            if p[0] >= s[0] and p[1] <= s[1]:
                b_in = True
                break
        if start < len(semantic_seg) and p[0] > semantic_seg[start][0]:
            semantic = semantic_seg[start][2]
            pattern += ('|'+semantic_flag[semantic])
            total_seg.append(semantic)
            start += 1
        if b_in == False:
            pattern += ('|'+pos_flag[i][0])
            total_seg.append(pos_flag[i][1])
    if start < len(semantic_seg):
        semantic  = semantic_seg[start][2]
        if semantic not in total_seg:
            pattern += ('|'+semantic_flag[semantic])
            total_seg.append(semantic)

    return pattern[1:],total_seg

def remove_semantic_conflict(seg):
        seg = sorted(seg, key=lambda x:(x[0], x[1]-x[0]))
        if len(seg) == 0:
            print ('can not find machine and issue semantic ')
            return []
        else:
            print ('total seg is  :', seg)
        
        print ('to remove semantic confilict ', seg)
        remove_conflict_seg = []
        b_previous_tie_added = False
        for j in range(1,len(seg)):
            if seg[j][0] > seg[j-1][0] and seg[j][0] < seg[j-1][1] or seg[j][1] > seg[j-1][0] and seg[j][1] < seg[j-1][1]:
                print('conflict by semantic', seg[j], seg[j-1], b_previous_tie_added, len(seg[j]), len(seg[j-1]))
                if len(seg[j][2]) < len(seg[j-1][2]):
                    if b_previous_tie_added == False:
                        remove_conflict_seg.append(seg[j-1])
                        b_previous_tie_added = True
                    else:
                        b_previous_tie_added = False
                print (remove_conflict_seg)
            else:
                 remove_conflict_seg.append(seg[j-1])
        if seg[-1] not in remove_conflict_seg and b_previous_tie_added == False:
            remove_conflict_seg.append(seg[-1])

        return remove_conflict_seg

def semantic_segment(s, machine_seg, issue_seg):
    machine_seg = sorted(machine_seg, key=lambda x:(x[0], x[1]))
    issue_seg = sorted(issue_seg, key=lambda x:(x[0], x[1]))
    if len(machine_seg) != 0 and len(issue_seg) == 0:
        machine_seg = find_best_semantic_seg(s, machine_seg)
        return machine_seg
    if len(issue_seg) != 0 and len(machine_seg) == 0:
        issue_seg = find_best_semantic_seg(s, issue_seg)
        return issue_seg
    if len(issue_seg) == 0 and len(machine_seg) == 0:
        return []

    if machine_seg[-1][1] > issue_seg[0][0]:
        print ('warning machine seg and issue seg is disordered ', machine_seg, issue_seg, s)
    best_seg_size = 0
    best_seg = None
    for machine in machine_seg:
        temp_machine_seg = []
        for x in machine_seg:
            if x[1] <= machine[1]:
                temp_machine_seg.append(x)
        temp_issue_seg = []
        for y in issue_seg:
            if y[0] >= machine[1]:
                temp_issue_seg.append(y)

        temp_machine_seg = find_best_semantic_seg(s, temp_machine_seg)
        temp_issue_seg = find_best_semantic_seg(s, temp_issue_seg)
        
        temp_seg = temp_machine_seg + temp_issue_seg
        temp_seg_size = len('.'.join([x[2] for x in temp_seg]))
        if temp_seg_size > best_seg_size:
            best_seg_size = temp_seg_size
            best_seg = temp_seg
    print ('best semantic seg is  ', best_seg)

    best_seg = remove_semantic_conflict(best_seg)

    return best_seg

def find_pattern_by_dict(sentence_dic):
    machine = pd.read_csv('./machine_now.csv')
    #print (machine.head())
    machine = machine['0'].tolist()
    #print (machine)
    issue = pd.read_csv('./issue_now.csv')
    #print (issue.head())
    issue = issue['0'].tolist()
    #print (issue)
    key_dic = {}
    with open(sentence_dic, "rb") as fp: 
        key_dic = pickle.load(fp) 

    pattern_dic = {}
    i = 0
    for s in key_dic:
        i += 1
        print (i)
        machine_seg = find_key_in_dict(s, machine)    
        print ('machine seg is ', machine_seg)
        issue_seg = find_key_in_dict(s, issue)    
        print ('issue seg is ', issue_seg)

        seg = semantic_segment(s, machine_seg, issue_seg)
        semantic_flag = {}
        for t in seg:
            if '|' in t[2]:
                keyword = t[2].split('|')[0]
                t[2] = t[2].replace('|','')
            else:
                keyword = t[2]
            if keyword in machine:
                semantic_flag[t[2]]='machine'
            if keyword in issue:
                semantic_flag[t[2]]='issue'
        print ('semantic flag ', semantic_flag)
        pattern,final_seg = find_pattern_by_pos(s,seg, semantic_flag)
        print (pattern, final_seg)
        if pattern not in pattern_dic:
            pattern_dic[pattern] = [final_seg]
        else:
            pattern_dic[pattern].append(final_seg)

    pattern_dic = dict(sorted(pattern_dic.items(),key=lambda x:len(x[1]), reverse=True))
    with open("./pattern_dic", "w") as fp:
        for k in pattern_dic:
            fp.write(k+':\n')
            for x in pattern_dic[k]:
                fp.write(str(x)+'\n')
            fp.write('\n')

    pattern_dic = dict(sorted(pattern_dic.items(),key=lambda x:x[1], reverse=True))
    print (pattern_dic)
    with open("./pattern_dic.pickle", "wb") as fp:
        pickle.dump(pattern_dic, fp, protocol = pickle.HIGHEST_PROTOCOL)      
    pattern_df = pd.DataFrame(list(pattern_dic.items()))
    pattern_df.to_csv('./pattern_dic_log.csv')
    valid = len(pattern_dic['machine|issue']
                + pattern_dic['machine|issue|issue']
                + pattern_dic['machine|issue|a']
                + pattern_dic['vn|machine|issue']
                + pattern_dic['machine|issue|x']
                + pattern_dic['machine|v|issue']
                + pattern_dic['m|machine|issue']
                + pattern_dic['machine|vn|issue']
                + pattern_dic['n|v|machine|issue']
                + pattern_dic['v|machine|machine|issue']
                + pattern_dic['machine|a|issue']
                + pattern_dic['v|machine|issue'])
    print ('complete ration = ', valid/1.0/i, 'total is ', i)

def deal_with_pattern(pattern, seg_index, pattern_dic_path, semantic1, semantic2):
    with open(pattern_dic_path, "rb") as fp: 
        pattern_dic = pickle.load(fp) 
    key_list = pattern_dic[pattern]

    temp = []
    for keys in key_list:
        temp.append(''.join(keys[:seg_index])+'\n')
    temp = list(set(temp))
    with open(semantic1, "w") as fp:
        for t in temp:
            fp.write(t)

    temp = []
    for keys in key_list:
        temp.append(''.join(keys[seg_index:])+'\n')
    temp = list(set(temp))
    with open(semantic2, "w") as fp:
        for t in temp:
            fp.write(t)

#find_pattern_by_dict('./question_dic_separated.pickle')

#deal_with_pattern('machine|v|n', 1, './pattern_dic.pickle', './new_machine', './new_issue')
#deal_with_pattern('machine|v', 1, './pattern_dic.pickle', './new_machine', './new_issue')
#deal_with_pattern('n|issue', 1, './pattern_dic.pickle', './new_machine', './new_issue')
#deal_with_pattern('machine|d|v', 1, './pattern_dic.pickle', './new_machine', './new_issue')
#deal_with_pattern('machine|v|v', 1, './pattern_dic.pickle', './new_machine', './new_issue')
#deal_with_pattern('machine|n|issue', 2, './pattern_dic.pickle', './new_machine', './new_issue')
#deal_with_pattern('machine|a', 1, './pattern_dic.pickle', './new_machine', './new_issue')
#deal_with_pattern('n|machine|issue', 2, './pattern_dic.pickle', './new_machine', './new_issue')
#deal_with_pattern('machine|n|v', 1, './pattern_dic.pickle', './new_machine', './new_issue')
#deal_with_pattern('v|n', 0, './pattern_dic.pickle', './new_machine', './new_issue')
#deal_with_pattern('machine|v|issue', 1, './pattern_dic.pickle', './new_machine', './new_issue')
#deal_with_pattern('machine|v|issue', 2, './pattern_dic.pickle', './new_machine', './new_issue')
#deal_with_pattern('machine|n', 2, './pattern_dic.pickle', './new_machine', './new_issue')
#deal_with_pattern('machine|d|issue', 1, './pattern_dic.pickle', './new_machine', './new_issue')
#deal_with_pattern('machine|n|issue', 1, './pattern_dic.pickle', './new_machine', './new_issue')
#deal_with_pattern('machine|v|vn', 1, './pattern_dic.pickle', './new_machine', './new_issue')
#deal_with_pattern('machine|issue|n', 1, './pattern_dic.pickle', './new_machine', './new_issue')
#deal_with_pattern('v|issue', 1, './pattern_dic.pickle', './new_machine', './new_issue')
#deal_with_pattern('issue|issue', 1, './pattern_dic.pickle', './new_machine', './new_issue')
#deal_with_pattern('machine|v|machine|issue', 3, './pattern_dic.pickle', './new_machine', './new_issue')
#deal_with_pattern('v|v|machine', 1, './pattern_dic.pickle', './new_machine', './new_issue')
#deal_with_pattern('machine|v|machine', 1, './pattern_dic.pickle', './new_machine', './new_issue')
#deal_with_pattern('machine|d|vn', 1, './pattern_dic.pickle', './new_machine', './new_issue')
#deal_with_pattern('v|n|issue', 2, './pattern_dic.pickle', './new_machine', './new_issue')
#deal_with_pattern('machine|issue|v', 1, './pattern_dic.pickle', './new_machine', './new_issue')
#deal_with_pattern('machine|n|vn', 1, './pattern_dic.pickle', './new_machine', './new_issue')
#deal_with_pattern('machine|d', 1, './pattern_dic.pickle', './new_machine', './new_issue')

def find_topo_by_ne(sentence_dic):
    machine = pd.read_csv('./machine_now.csv')
    machine = machine['0'].tolist()
    issue = pd.read_csv('./issue_now.csv')
    issue = issue['0'].tolist()
    key_dic = {}
    with open(sentence_dic, "rb") as fp: 
        key_dic = pickle.load(fp) 

    ne_machine_dic = {}
    ne_issue_dic = {}
    i = 0
    for s in key_dic:
        i += 1
        print (i)
        #if i == 1000:
            #break

        machine_seg = find_key_in_dict(s, machine)    
        print ('machine seg is ', machine_seg)
        issue_seg = find_key_in_dict(s, issue)    
        print ('issue seg is ', issue_seg)
        seg = semantic_segment(s, machine_seg, issue_seg)
        semantic_flag = {}
        for t in seg:
            if '|' in t[2]:
                keyword = t[2].split('|')[0]
                t[2] = t[2].replace('|','')
            else:
                keyword = t[2]
            if keyword in machine:
                semantic_flag[t[2]]='machine'
            if keyword in issue:
                semantic_flag[t[2]]='issue'
        print ('semantic flag ', semantic_flag)
        pattern,final_seg = find_pattern_by_pos(s,seg, semantic_flag)
        pattern = pattern.split('|')
        if 'machine' in pattern:
            machine_index = pattern.index('machine')
        else:
            machine_index = -1
        if 'issue' in pattern:
            issue_index = pattern.index('issue')
        else:
            issue_index = -1
        print (pattern, final_seg, machine_index, issue_index)
        if machine_index != -1 and issue_index != -1:
            ne_machine = final_seg[machine_index]
            ne_issue = final_seg[issue_index]
            if ne_machine not in ne_machine_dic:
                ne_machine_dic[ne_machine] = [ne_issue]
            else:
                if ne_issue not in ne_machine_dic[ne_machine]:
                    ne_machine_dic[ne_machine].append(ne_issue)

            if ne_issue not in ne_issue_dic:
                ne_issue_dic[ne_issue] = [ne_machine]
            else:
                if ne_machine not in ne_issue_dic[ne_issue]:
                    ne_issue_dic[ne_issue].append(ne_machine)

    kv_list = sorted(ne_machine_dic.items(),key=lambda item:len(item[1]), reverse = True)
    ne_machine_dic = {}
    for kv in kv_list:
        ne_machine_dic[kv[0]] = kv[1]

    kv_list = sorted(ne_issue_dic.items(),key=lambda item:len(item[1]), reverse = True)
    ne_issue_dic = {}
    for kv in kv_list:
        ne_issue_dic[kv[0]] = kv[1]

    print (ne_machine_dic)
    with open("./kb_machine.txt", "w") as fp:
        for k in ne_machine_dic:
            fp.write('<ne>'+k+':\n')
            for x in ne_machine_dic[k]:
                fp.write('\t<attribute>'+str(x)+'\n')
            fp.write('\n')
    print (ne_issue_dic)
    with open("./kb_issue.txt", "w") as fp:
        for k in ne_issue_dic:
            fp.write('<ne>'+k+':\n')
            for x in ne_issue_dic[k]:
                fp.write('\t<attribute>'+str(x)+'\n')
            fp.write('\n')

find_topo_by_ne('./question_dic_separated.pickle')
