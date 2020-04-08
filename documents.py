#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import numpy as np
import pandas as pd
import string
import re
import nltk
import random
from sklearn.svm import SVC
from collections import Counter
# nltk.download('stopwords')
# nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import accuracy_score
from statistics import mode 



class Cluster:
    
    vectorize = None
    vectorized_train_data = None
    train_data_labels = None
    K = 5
    iterations = 25
    def merge_files_data(self,data_path):
        train_data_list = []
        files = os.listdir(data_path)
        count=0
#         files = files[:100]
        for entry in files:
            file_name = entry
#             print(file_name)
            c_filename = data_path+"/"+file_name
            cluster_val = file_name.split("_")[1].split(".")[0]
            f = open(c_filename, "r", encoding='latin1')
            file_contents = f.read()
            a_contents = np.empty((0))
            a_contents = np.append(a_contents,file_contents)
            a_contents = np.append(a_contents,cluster_val)
            train_data_list = np.append(train_data_list, a_contents)

        train_data_array = np.reshape(np.asarray(train_data_list) , (len(files),2))
        return train_data_array, files
    
    def preprocess_data(self,train_data_frm):
               
        rows,cols = train_data_frm.shape
              
        ''' removing punctuations'''
        for i, rows in train_data_frm.iterrows():
            val = train_data_frm.iat[i,0]
            fval = val.translate(val.maketrans('','', string.punctuation))
            train_data_frm.at[i, 'Col1'] = fval
        
       
        ''' removing numbers'''
        for i, rows in train_data_frm.iterrows():
            val = train_data_frm.iat[i,0]
            fval = re.sub(r'\d+', '', val)
            train_data_frm.at[i, 'Col1'] = fval
            
        ''' converting to lowercase'''
        for i, rows in train_data_frm.iterrows():
            val = train_data_frm.iat[i,0]
            fval = val.lower()
            train_data_frm.at[i, 'Col1'] = fval
            
        '''removing stop words'''
        stop_words = set(stopwords.words('english'))
        for i, rows in train_data_frm.iterrows():
            val = train_data_frm.iat[i,0]
            tokens = word_tokenize(val)
            fval = [i for i in tokens if not i in stop_words]
            ffval = " "
            train_data_frm.at[i, 'Col1'] = ffval.join(fval)
            
        '''removing whitespaces'''
        for i, rows in train_data_frm.iterrows():
            val = train_data_frm.iat[i,0]
            train_data_frm.at[i, 'Col1'] = val.strip()
                                  
        return train_data_frm
    
    def train_validation_split(self,data_frm,validation_data_size):
        if isinstance(validation_data_size, float):
            validation_data_size=round(validation_data_size * len(data_frm))
        indices=data_frm.index.tolist()
        valid_indices=random.sample(indices, validation_data_size)
        valid_datafrm=data_frm.loc[valid_indices]
        train_datafrm=data_frm.drop(valid_indices)
        return train_datafrm, valid_datafrm
    
   
    def prepare_data(self,data_frm):
        data_labels = data_frm.iloc[:,-1]
        data_frm = data_frm.iloc[:,:-1]
        return data_frm, data_labels
        
        
    def run_kmeans_algo(self, files):
        n = self.vectorized_train_data.shape[0]
        n1 = self.vectorized_train_data.shape[1]
        train_data_array = self.vectorized_train_data.toarray()
        indices=[]
        for j in range(self.K): 
            indices.append(random.randint(0,n)) 

        cluster_centroids_li = []

        for i in range(self.K):
            cluster_centroids_li.append(train_data_array[indices[i]])
        cluster_centroids = np.asarray(cluster_centroids_li)
        
       
        cluster_dict = {}
        cluster_files = {}
        cluster_labels = {}
        for it in range(self.iterations):
            #print("iteration ", it)
            cluster_centroid_dis = []
            cluster_dict = {0:[] , 1:[], 2:[], 3:[], 4:[]}
            cluster_files = {0:[] , 1:[], 2:[], 3:[], 4:[]}
            cluster_labels = {0:[] , 1:[], 2:[], 3:[], 4:[]}
            
            for j in range(n):
                curr = []
                min_dist_clust = np.finfo(np.float64).max
                min_clust_num = None
                for i in range(self.K):
                    ed_dis = np.linalg.norm(train_data_array[j] - cluster_centroids[i])
                    val = ed_dis < min_dist_clust
                    if val:
                        min_dist_clust = ed_dis
                        val1 = min_clust_num
                        min_clust_num = i
                data_val = train_data_array[j]
                cluster_dict[min_clust_num].append(data_val)
                data_val = files[j]
                cluster_files[min_clust_num].append(files[j])
                data_val = str(self.train_data_labels[j])
                cluster_labels[min_clust_num].append(str(self.train_data_labels[j]))

            updated_mean = []
            
            for i in range(self.K):

                cl_array = np.asarray(cluster_dict[i])
                cl_mean = cl_array.mean(axis=0)
                updated_mean.append(cl_mean)
            
            cluster_centroids = np.asarray(updated_mean)
        #cluster_final=[]
        #final_count=0
        #for i in range(5):
        #    mod_val=mode(cluster_labels[i])
         #   cluster_final.append(mod_val)
          #  mod_count=cluster_labels[i].count(mod_val)
           # final_count=final_count+mod_count
        final_dict_files = {}
        for k,v in cluster_files.items():
            for file_name in v:
                final_dict_files[file_name] = k
        return final_dict_files
                
            
            
    def cluster(self,data_path):
        train_data_array,files = self.merge_files_data(data_path)
        train_data_frm = pd.DataFrame({'Col1': train_data_array[:, 0], 'Col2': train_data_array[:, 1]})
        train_data_frm = self.preprocess_data(train_data_frm)
        train_data_frm, self.train_data_labels = self.prepare_data(train_data_frm)
        self.vectorize = TfidfVectorizer()
        train_data_frm = train_data_frm.values.flatten()
        self.vectorized_train_data = self.vectorize.fit_transform(train_data_frm)
        self.train_data_labels = self.train_data_labels.values.flatten()
        self.vectorized_train_data = self.vectorized_train_data
        final_dict = self.run_kmeans_algo(files)
        return final_dict
      


# In[ ]:


#cluster_algo = Cluster()
# You will be given path to a directory which has a list of documents. You need to return a list of cluster labels for those documents
#predictions = cluster_algo.cluster('/home/jyoti/Documents/SMAI/assign2/Q6/dataset') 
#print(predictions)


# In[ ]:




