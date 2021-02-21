
import numpy as np
import pandas as pd
from datetime import date
import random as random
import os
import cPickle as pickle
import matplotlib.pyplot as plt


root_path = os.getcwd()
print(root_path)
DATASET_PATH = os.path.join(root_path, 'mimic')
def Read_csv_fromPath(csv_path):
    result = pd.read_csv(csv_path,sep=',')
    return result
file_name = 'PATIENT_VISIT_182.csv'
FILE_PATH = os.path.join(DATASET_PATH, file_name)
data = Read_csv_fromPath(FILE_PATH)
print(data.head())

data['date']=list(map(lambda x:map(int,x.split('-')),data.admittime))
data['icd9']=list(map(lambda x:map(int,x.split(',')),data.icd9_code))
data['deltatime']=np.zeros(data.shape[0]).astype(int)

# calculate time
p = -1 # patient's subject id
for i in range(data.shape[0]):
    if data['subject_id'][i] != p: # new patient
        p = data['subject_id'][i]
    else:
        payIntList = list(data['date'][i])
        y, m, d = payIntList
        y0,m0,d0 = data['date'][i-1]
        data.loc[i,'deltatime'] = (date(y,m,d)-date(y0,m0,d0)).days


visit_file = [] # icd9 code
patient_id = [] # patient id
admission_id = [] # admission id 
duration = [] # duration

p_id = -1 # patient's subject id

for i in range(data.shape[0]):
    # new patient
    if data['subject_id'][i] != p_id:
        # for previous patient, add visits to visit_file
        if p_id != -1:
            visit_file.append(visits)
            admission_id.append(ad_id)
            duration.append(t)
            
        p_id = data['subject_id'][i]
        patient_id.append(p_id)
        
        visits = [map(lambda x:x-1,data['icd9'][i])]
        ad_id = [data['hadm_id'][i]]
        t = [data['deltatime'][i]]
        
    # same patient
    else: 
        visits.append(map(lambda x:x-1,data['icd9'][i]))
        ad_id.append(data['hadm_id'][i])
        t.append(data['deltatime'][i])
        
# add the last patient's visit        
visit_file.append(visits)
admission_id.append(ad_id)
duration.append(t)
pNum = len(visit_file)
title = ['train','valid','test']
proportion = [0.8,0.1,0.1]
train_test_proportion = 0.8

# generate random index for train, valid, test set
random.seed(10000)
random_index = random.sample(range(pNum),pNum)

# divide dataset into train, valid, test set for training data
trainingNum_index = []
trainingNum_index.append([0,int(pNum*train_test_proportion*proportion[0])]) # train
trainingNum_index.append([trainingNum_index[0][1],\
                          trainingNum_index[0][1]+\
                          int(pNum*train_test_proportion*proportion[1])]) # valid
trainingNum_index.append([trainingNum_index[1][1],\
                          int(pNum*train_test_proportion)]) # test

# test dataset
testNum_index = [int(pNum*train_test_proportion),pNum]
print trainingNum_index
print testNum_index


# generate train, valid, test visit/label/duration files
# visit file is the same as label file
for t in range(3):
    indexRange = trainingNum_index[t]
    f_tmp = [] # visit/label data
    d_tmp = [] # duration data
    for i in range(indexRange[0],indexRange[1]):
        f_tmp.append(visit_file[random_index[i]])
        d_tmp.append(duration[random_index[i]])
    pickle.dump(f_tmp,open('data/visit_182.'+ title[t],'w'))
    pickle.dump(f_tmp,open('data/label_182.'+ title[t],'w'))
    pickle.dump(d_tmp,open('data/duration.'+ title[t],'w'))

# test dataset
indexRange = testNum_index
f_tmp = [] # visit/label data
d_tmp = [] # duration data
for i in range(indexRange[0],indexRange[1]):
    f_tmp.append(visit_file[random_index[i]])
    d_tmp.append(duration[random_index[i]])
pickle.dump(f_tmp,open('data/test_182','w'))
pickle.dump(d_tmp,open('data/test_duration','w'))
print len(f_tmp)


# generate full set of training data
indexRange = [trainingNum_index[0][0],trainingNum_index[2][1]]
print indexRange
f_tmp = [] # visit/label data
d_tmp = [] # duration data
for i in range(indexRange[0],indexRange[1]):
    f_tmp.append(visit_file[random_index[i]])
    d_tmp.append(duration[random_index[i]])
pickle.dump(f_tmp,open('data/visit_182','w'))
pickle.dump(d_tmp,open('data/train_duration','w'))
print len(f_tmp)

# get visit file (e.g. data in 'icd9') according to admission_id
def getVisitFile(data,admission_id,column = 'icd9'):
    visit_file = []
    p_id = 0
    v_id = 0
    index = -1

    for adm in data['hadm_id']:
        index += 1

        if v_id == 0 and admission_id[p_id][v_id] == adm:
            visits = [map(lambda x:x-1,data.loc[index,column])]
        elif admission_id[p_id][v_id] == adm:
            visits.append(map(lambda x:x-1, data.loc[index,column]))
        else:
            continue

        if v_id == len(admission_id[p_id])-1:
            p_id += 1
            v_id = 0
            visit_file.append(visits)
        else:
            v_id += 1
    return visit_file

def outputFile(visit_file,trainingNum_index,testNum_index,file_name):
    # generate train, valid, test visit/label/duration files
    # visit file is the same as label file
    for t in range(3):
        indexRange = trainingNum_index[t]
        f_tmp = [] # visit/label data
        d_tmp = [] # duration data
        for i in range(indexRange[0],indexRange[1]):
            f_tmp.append(visit_file[random_index[i]])
        pickle.dump(f_tmp,open('data/visit_'+ file_name +\
                               '.'+ title[t],'w'))
        pickle.dump(f_tmp,open('data/label_'+ file_name +\
                               '.'+ title[t],'w'))

    # test dataset
    indexRange = testNum_index
    f_tmp = [] # visit/label data
    d_tmp = [] # duration data
    for i in range(indexRange[0],indexRange[1]):
        f_tmp.append(visit_file[random_index[i]])
    pickle.dump(f_tmp,open('data/test_'+ file_name,'w'))

    # visits with full set of codes
FULL_VISIT = 'PATIENT_VISIT_FULL.csv'
FULL_VISIT_Path = os.path.join(DATASET_PATH, FULL_VISIT)
data_full = Read_csv_fromPath(FULL_VISIT_Path)
data_full['icd9']=map(lambda x:map(int,x.split(',')),data_full.icd9_code)

visit_file_full = getVisitFile(data_full,admission_id)

################ skip this step when not generating cv files ################
# generate train, valid, test visit/label files for rnn cross validation
# visit file is the same as label file
K_FOLD = 2
training_size = int(pNum*train_test_proportion)

# divide dataset into train, valid, test set for training data
trainingNum_index = []
trainingNum_index.append([0,int(training_size*train_test_proportion*\
                                proportion[0])]) # train
trainingNum_index.append([trainingNum_index[0][1],\
                          trainingNum_index[0][1]+\
                          int(training_size*train_test_proportion*\
                              proportion[1])]) # valid
trainingNum_index.append([trainingNum_index[1][1],\
                          int(training_size*train_test_proportion)]) # test

# test dataset
testNum_index = [int(training_size*train_test_proportion),training_size]

print trainingNum_index
print testNum_index

# fold 1
for t in range(3):
    indexRange = trainingNum_index[t]
    f_tmp = [] # visit/label data
    for i in range(indexRange[0],indexRange[1]):
        f_tmp.append(visit_file_full[random_index[i]])
    pickle.dump(f_tmp,open('data/rnn-cv/visit_cv_1.'+ title[t],'w'))
    pickle.dump(f_tmp,open('data/rnn-cv/label_cv_1.'+ title[t],'w'))

# test dataset
indexRange = testNum_index
f_tmp = [] # visit/label data
for i in range(indexRange[0],indexRange[1]):
    f_tmp.append(visit_file[random_index[i]])
pickle.dump(f_tmp,open('data/rnn-cv/test_cv_1','w'))
print len(f_tmp)

# fold 2
for t in range(3):
    indexRange = trainingNum_index[t]
    f_tmp = [] # visit/label data
    for i in range(indexRange[0],indexRange[1]):
        f_tmp.append(visit_file_full[random_index[::-1][i]])
    pickle.dump(f_tmp,open('data/rnn-cv/visit_cv_2.'+ title[t],'w'))
    pickle.dump(f_tmp,open('data/rnn-cv/label_cv_2.'+ title[t],'w'))

# test dataset
indexRange = testNum_index
f_tmp = [] # visit/label data
for i in range(indexRange[0],indexRange[1]):
    f_tmp.append(visit_file[random_index[::-1][i]])
pickle.dump(f_tmp,open('data/rnn-cv/test_cv_2','w'))
print len(f_tmp)

outputFile(visit_file_full,trainingNum_index,testNum_index,'full')

# visits with sub set of codes, 250
data_250 = pd.read_csv('mimic/PATIENT_VISIT_250.csv',sep=',')
data_250['icd9']=map(lambda x:map(int,x.split(',')),data_250.icd9_code)


visit_file_250 = getVisitFile(data_250,admission_id)
outputFile(visit_file_250,trainingNum_index,testNum_index,'250')

note_file = open('data/note_topic_50','rb')
note_topic = pickle.load(note_file)
note_file.close()
print note_topic.shape
note_topic.head(10)
print note_topic['topics'].shape
print len(note_topic['topics'][0])

# get topic file according to admission_id
def getTopicFile(data,admission_id,column = 'topics'):
    info_dict = {}
    
    for i in range(data.shape[0]):
        info_dict[data.loc[i,'hadm_id']] = data.loc[i,column]
    
    visit_file = []
    p_id = 0
    v_id = 0
    
    n_feature = len(data.loc[0,column])
    while p_id < len(admission_id):
        if v_id == 0:
            info = [list(np.zeros(n_feature))]
            if admission_id[p_id][v_id] in info_dict.keys():
                info = [list(info_dict[admission_id[p_id][v_id]])]
        elif admission_id[p_id][v_id] in info_dict.keys():
            info.append(list(info_dict[admission_id[p_id][v_id]]))
        else:
            info.append(list(np.zeros(n_feature)))

        if v_id == len(admission_id[p_id])-1:
            p_id += 1
            v_id = 0
            visit_file.append(info)
        else:
            v_id += 1
    return visit_file

topic_file = getTopicFile(note_topic,admission_id,column = 'topics')

def roundTo(l,digit=4):
    return map(lambda x:np.round(x,digit),l)

topic_file = map(lambda l:roundTo(l,6),topic_file)
print len(topic_file),np.sum(map(len,topic_file))
info_dict1 = {}
    
for i in range(note_topic.shape[0]):
    info_dict1[note_topic.loc[i,'hadm_id']] = note_topic.loc[i,'topics']
print admission_id[0]
print info_dict1[194023],info_dict1[161087]
file_name = open('data/visit_file_topic_'+ str(len(note_topic['topics'][0])),'w')
pickle.dump(topic_file,file_name)
file_name.close()


outputFile(topic_file,trainingNum_index,testNum_index,'topic_'+str(len(note_topic['topics'][0])))

# extract tfidf 
note_file = open('data/note_tfidf_200','rb')
note_tfidf = pickle.load(note_file)
note_file.close()
print note_tfidf.shape
note_tfidf.head(10)

tfidf_file = getTopicFile(note_tfidf,admission_id,column = 'tfidf')
def roundTo(l,digit=4):
    return map(lambda x:np.round(x,digit),l)

tfidf_file = map(lambda l:roundTo(l,6),tfidf_file)

print len(tfidf_file),np.sum(map(len,tfidf_file))

print '######################## End of processing dataset ###########################'


file_name = open('data/visit_file_tfidf_'+ str(len(note_tfidf['tfidf'][0])),'w')
pickle.dump(tfidf_file,file_name)
file_name.close()

outputFile(tfidf_file,trainingNum_index,testNum_index,'tfidf_'+str(len(note_tfidf['tfidf'][0])))