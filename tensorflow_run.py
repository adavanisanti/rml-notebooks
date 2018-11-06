import warnings
warnings.simplefilter(action='ignore')

import pandas as pd
import numpy as np
import re
import os

import time #cpu time
import psutil #memory usage
#tensorflow
import tensorflow as tf

#Scikitlearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file
from sklearn.datasets import dump_svmlight_file
from scipy.sparse import coo_matrix,csr_matrix,lil_matrix
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
####################Splitting Data File################
def get_data(file):
    data = load_svmlight_file(file)
    return data[0], data[1]

input_file=os.getcwd()+'/a9a'

X,y=get_data(input_file)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dump_svmlight_file(X_train, y_train,'train_file')#%80
dump_svmlight_file(X_test, y_test,'test_file')#%20
print('Data file split.')
#######################################################
learning_rate =  0.001
max_iter = 100
batch_size = 100

train_file=os.getcwd()+'/train_file'
test_file=os.getcwd()+'/test_file'

class DataSet(object):
    def __init__(self):
        self.iter = 0
        self.epoch_pass = 0

    def get_data(file):
        data = load_svmlight_file(file)
        return data[0], data[1]    
    
    def load(self, file):
        X, y=get_data(file)
        self.feature_num=X.shape[1] #X.shape[1]
        self.ins_num =X.shape[0] #equivalent to .shape[0] in csr format
        self.y = list(y)
        self.feature_ids = list(X.indices) #equivalent to .indices in csr format
        self.feature_values = list(X.data) #equivalent to .data in csr format
        self.ins_feature_interval = list(X.indptr)
        self.ins_feature_interval_diff = [j-i for i, j in zip(X_indptr[:-1], X_indptr[1:])] #equivalent to .indptr in csr format
    

    def mini_batch(self, batch_size):
        begin = self.iter #begins as 0 as defined above
        end = self.iter #starts with 0 as defined above
        if self.iter + batch_size > self.ins_num: #if 0 + batchsize(10) > ins_num(1) defined in def load
            end = self.ins_num #set end to be ins_num(1) 
            self.iter = 0 #set iter to 0
            self.epoch_pass += 1 #add +1 to epoch_pass 
        else:
            end += batch_size#add batch size to end, which should be equal to batch size
            #print('end:',end)
            #print('batch_size:',batch_size)
            self.iter = end#set self.iter to batch size
            #(begin, end)setting bounds moving across batch length in data
            #print((begin, end))#slicing action of data (0, 15) (15, 30) (30, 45) (60,...
        return self.slice(begin, end)

    def slice(self, begin, end):
        sparse_index = []
        sparse_ids = []
        sparse_values = []
        sparse_shape = []
        max_feature_num = 0
        for i in range(begin, end):#within range begin, end
        #           15,461,906,1351          0,446,891,1338 =~15 supposed to be length of token
        #          (token length + range number +1) - (token length + range number)
            feature_num = self.ins_feature_interval[i + 1] - self.ins_feature_interval[i]
            if feature_num > max_feature_num:
                max_feature_num = feature_num
                                #       0,446,891,1338                  15,461,906,1351
            #(token length + range number) ,       (token length + range number +1)
            #self.max_ins_feature_interval-len(self.feature_ids)<-------------------
            #print(self.ins_feature_interval[i],self.ins_feature_interval[i + 1])
            #print(self.feature_ids[i])
            for j in range(self.ins_feature_interval[i], self.ins_feature_interval[i + 1]):
            #print(j,(len(self.feature_ids)))
            #print(self.ins_feature_interval[i + 1])
            #print(range(self.ins_feature_interval[i], self.ins_feature_interval[i + 1]))
            #15 vals:                  0  , 0-14,446-460,891-905
                sparse_index.append([i - begin, j - self.ins_feature_interval[i]]) # index must be accent
                #[0, 0]-[0, 14][0, 0]-[0, 12]
                #print([i - begin, j - self.ins_feature_interval[i]])
                sparse_ids.append(self.feature_ids[j])
                sparse_values.append(self.feature_values[j])
        sparse_shape.append(end - begin)
        #print(end - begin)#<-----30
        sparse_shape.append(max_feature_num)
        #print(max_feature_num)#<-----15
        #       Creates array shape of 30,1 of y values  (30, 1)
        y = np.array(self.y[begin:end]).reshape((end - begin, 1))
        #begin:0,30,60,90,120,150,180,210 intervals of 30
        #end: 30,60,90,120,150,180,210,240

        #            0            0                       0                 30            30
        #print(len(sparse_index), len(sparse_ids), len(sparse_values), len(sparse_shape), len(y))
        return (sparse_index, sparse_ids, sparse_values, sparse_shape, y)

class BinaryLogisticRegression(object):
    def __init__(self, feature_num):
        self.feature_num = feature_num
        self.sparse_index = tf.placeholder(tf.int64)
        self.sparse_ids = tf.placeholder(tf.int64)
        self.sparse_values = tf.placeholder(tf.float32)
        self.sparse_shape = tf.placeholder(tf.int64)
        self.w = tf.Variable(tf.random_normal([self.feature_num, 1], stddev=0.1))
        self.y = tf.placeholder("float", [None, 1])

    def forward(self):
        return tf.nn.embedding_lookup_sparse(self.w,tf.SparseTensor(self.sparse_index, self.sparse_ids, self.sparse_shape),tf.SparseTensor(self.sparse_index, self.sparse_values, self.sparse_shape),combiner="sum")

mem_baseline1=psutil.virtual_memory() #  physical memory usage
print('Here is the memory baseline prior to importing data:\n\n',mem_baseline1)

start = time.time()

train_set = DataSet()
train_set.load(train_file)

end = time.time()
exec_time1=(end - start)
print('\nThe time taken to import and prepare training data for Tensorflow is:',exec_time1)

start = time.time()

test_set = DataSet()
test_set.load(test_file)
feature_num=test_set.feature_num

end = time.time()
exec_time2=(end - start)
print('\nThe time taken to import and prepare testing data for Tensorflow is:',exec_time2)

mem_baseline2=psutil.virtual_memory() #  physical memory usage
print('Here is the memory usage after importing data:\n',mem_baseline2)

model = BinaryLogisticRegression(feature_num)

y = model.forward()

loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=model.y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

probability_output = tf.nn.sigmoid(y)

session = tf.Session()
init_all_variable = tf.global_variables_initializer()
init_local_variable = tf.local_variables_initializer()
session.run([init_all_variable, init_local_variable])

num_passes=1
start = time.time()
end_list=[]
for i in range(0,num_passes):
    while train_set.epoch_pass < max_iter:
        sparse_index, sparse_ids, sparse_values, sparse_shape, mb_y = train_set.mini_batch(batch_size)
        
        _, loss_, prob_out = session.run([optimizer, loss, probability_output],
                                         feed_dict={model.sparse_index: sparse_index,
                                                    model.sparse_ids: sparse_ids,
                                                    model.sparse_values: sparse_values,
                                                    model.sparse_shape: sparse_shape,
                                                    model.y: mb_y})
        
    end = time.time()
    exec_time=(end - start)
    end_list.append(exec_time) 
    #save endlist not exec_time
    try:
        auc = roc_auc_score(mb_y, prob_out)
        print("epoch: ", train_set.epoch_pass, " ROC AUC score is: ", auc)

    except:
        print('\nValueError: Only one class present in y_true. ROC AUC score is not defined in that case.\n')
        print(mb_y.T)
        print(prob_out.T,'\n')

print('\nThe average time taken to execute logistic regression for '+str(num_passes)+' full passes of',max_iter,'iterations took',np.array(end_list).mean(),'seconds with a standard deviation of +- '+str(np.array(end_list).std()))

bench_list=[str(mem_baseline1),str(exec_time1),str(exec_time2),str(mem_baseline2),str(np.array(end_list).mean())]
with open('bench_times.txt', 'w') as f:
    for item in bench_list:
        f.write("%s\n" % item)
