# coding:utf-8
import math
import random
import operator
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from statistics import mean
from matplotlib.pylab import mpl
import pathlib
import csv




class KmeansPP(object):
    clusters=list()
    centroids=list()
    
    
    def __init__(self,Matrix,K_val):
        self.matrix=None
        self.total_len=None
        self.K_val=K_val
        self.clusters=[None]*K_val
        self.df = pd.DataFrame(Matrix).T.fillna(0)
        self.df = self.df.transpose()

    def cosine(self,qry_TFIDF:dict,abs_TFIDF:dict):
        #calculate the Cosine distance of two articles
        num=0
        deno_abs=0
        deno_qry=0
        for key,value in abs_TFIDF.items():
            if key in qry_TFIDF.keys():
                num+=qry_TFIDF[key]*abs_TFIDF[key]
                
        for key,value in abs_TFIDF.items():
            deno_abs+=abs_TFIDF[key]**2
        for key,value in qry_TFIDF.items():
            deno_qry+=qry_TFIDF[key]**2

        deno=math.sqrt(deno_abs)*math.sqrt(deno_qry)
        if deno!=0:
            return 1-num/deno
        else:
            return 1

    def init(self):#Initilize k centroids for Kmeans++ Algorithm
        print("Initiating")
        self.centroids=[None]*self.K_val
        self.centroids[0]=random.randint(0,self.total_len-1)#Randomly choose the first point
        cur_k=1
        while cur_k < self.K_val:
            distance=0
            #print("calculating ",cur_k," k")    
            for i in range(len(self.matrix)):
                if i not in self.centroids:
                    cur_dis=0#For every point not already choosen as centroids
                    #print("for point ",i)
                    for j in self.centroids:#Calculate cumilative distance with those already in centroids
                        if j != None:
                            #print("centroid ",j)
                            cur_dis+=self.cosine(self.matrix[i],self.matrix[j])          
                        if cur_dis>distance:
                            distance=cur_dis
                            self.centroids[cur_k]=i                        
            cur_k+=1
            
    def centroiding(self):
        new_centroids_list=list()       
        for cluster in self.clusters:#3 clusers,each countains a list of numbers
            cluster_len=len(cluster)
            new_centroid=dict()
            for i in range(0,cluster_len):
                for key,value in self.matrix[cluster[i]].items():
                    if key in new_centroid.keys():
                        new_centroid[key]+=value
                    else:
                        new_centroid[key]=value
            for key,value in new_centroid.items():
                new_centroid[key]=new_centroid[key]/cluster_len
            new_centroids_list.append(new_centroid)
        self.centroids=new_centroids_list


    def first_cluster(self):
        #Cluster for the first time, now the self.centroids stores 3 numbers
        #From the second time on, the self.centroids would be storing 3 dictionaries
        
        for j in range(len(self.clusters)):
            self.clusters[j]=list()
            self.clusters[j].append(self.centroids[j])


        for i in range(self.total_len):
            dis=1
            cluster=self.K_val-1
            for j in range(len(self.centroids)):
                cur_dis=self.cosine(self.matrix[i],self.matrix[self.centroids[j]])
                cur_cluster = j
                if cur_dis < dis:
                    dis=cur_dis
                    cluster=cur_cluster
            
            if i not in self.clusters[cluster]:
                self.clusters[cluster].append(i)  
                
                
    def further_cluster(self):
        #Note from this time on , self,centroids would be storing 3 dictionaries!      
        iteration=0
        while iteration<100:#Stop when iterate for 100 times
            change=0#Stop when there is no change            
            self.centroiding()
        
            for i in range(self.total_len):
                dis=1
                cluster_index=self.K_val-1
                
                for j in range(len(self.centroids)):                   
                    cur_dis=self.cosine(self.matrix[i],self.centroids[j])
                    cur_cluster_index = j                   
                    if cur_dis < dis:
                        dis=cur_dis
                        cluster_index=cur_cluster_index
                
                if i not in self.clusters[cluster_index]:
                    change+=1
                    for cluster in self.clusters:
                        if i in cluster:
                            cluster.remove(i)
                    self.clusters[cluster_index].append(i)
        
           
            
            if change==0:
                return
            else:
                iteration+=1

    def PCA_decompose(self):
        pca = PCA(n_components=0.9)
        pca.fit(self.df)
        self.df_new = pca.transform(self.df)
        self.df_new=pd.DataFrame(self.df_new)
        self.matrix=self.df_new
        self.matrix= self.matrix.to_dict("records")
        self.total_len=len(self.matrix)
        




    
        
class Visulization(object):
    def __init__(self,matrix,clusters,title,file_names):
        self.matrix=matrix
        self.df = pd.DataFrame(matrix).T.fillna(0)
        self.df = self.df.transpose()
        self.clusters=clusters
        self.title=title
        self.len=len(self.clusters)
        self.clusters_files=None
        self.file_names=file_names
        #Transfer original list-dict two dimension 2-d structure to pandas dataframe
    
    def listToString(self,s): 
        str1 = ""       
        for ele in s: 
            str1 += ele  
            str1+="/"
        return str1
        
    def topic_extract(self):
        self.topics=list()
        for i in range(self.len):
            
            new_mat=dict()
            
            for j in self.clusters[i]:#add documents of one cluster up
                for key,value in self.matrix[j].items():
                    if key in new_mat:
                        new_mat[key]+=self.matrix[j][key]
                    else :
                        new_mat[key]=self.matrix[j][key]
            #sort the added document and take the highest few
            sorted_mat=sorted(new_mat.items(), key=operator.itemgetter(1))
            sorted_mat.reverse()
            
            cur_topic=list()
            if len(sorted_mat)>0:               
                for x in range(8):#choose 8 words with highest overall tfidf score as topics
                    cur_topic.append(sorted_mat[x][0])  
                cur_topic=self.listToString(cur_topic)
            else:
                cur_topic.append(None)  
            self.topics.append(cur_topic)
            
        return

    def doc_names(self):
        self.clusters_files=list()
        for topic_index in range(self.len):
            cluster_files=list()
            for doc_index in self.clusters[topic_index]:
                cluster_files.append(self.file_names[doc_index])
            self.clusters_files.append(cluster_files)



    def PCA_decompose(self):
        pca = PCA(n_components=2)
        pca.fit(self.df)

        self.df_new = pca.transform(self.df)

        self.df_new=pd.DataFrame(self.df_new)
        
        self.df_new.columns = ["PC1","PC2"]
        self.df_new["clusters_db"]=None
        
        ind=0
        for index,row in self.df_new.iterrows():#Store the clustering info
            for j in range(len(self.clusters)):
                if ind in self.clusters[j]:
                    self.df_new.loc[ind,"clusters_db"]=j
                    row["clusters_db"]=j
            ind+=1
                  
        

            
    def PCA_print(self):
        '''
        fig = plt.figure()
        ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
        '''
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        
        ax1.set_title(self.title)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        arr=self.df_new.to_numpy()
        plt.scatter(arr[:, 0], arr[:, 1], marker='o', c=arr[:,2], cmap='jet')
        figtext=""
        for text in self.topics:
            figtext=figtext+text+"\n"

        plt.figtext(0.5, 0.01, figtext, ha="center", fontsize=6, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})

        

        mpl.rcParams['font.sans-serif'] = ['SimHei']

        mpl.rcParams['axes.unicode_minus'] = False
        #self.df_new.plot.scatter("PC1", "PC2",c="clusters_db",colormap="jet")
        plt.savefig(f'fig_k={self.len}.png')
    
    def save(self):
        write_topic=open(f"topics_simple_k={str(self.len)}.txt",'w',encoding="utf8")
        write_topic.write("topics for each Cosine cluster are: \n")
        for i in range(self.len):
            write_topic.write(self.topics[i]+"\n")
            for j in range(20):
                write_topic.write(self.clusters_files[i][j])
                write_topic.write("\n")
            write_topic.write("\n")
        write_topic.close()


        cluster_topic=open(f"topics_comprehensive_l={str(self.len)}.txt",'w',encoding="utf8")
        for i in range(self.len):
            
            cluster_topic.write(self.topics[i]+"\n")
            for doc_names in self.clusters_files[i]:
                cluster_topic.write(doc_names)
                cluster_topic.write("\n")
            cluster_topic.write("\n")
        cluster_topic.close()

        
        
    
    
    
def driver(data,k,file_names):
    
#----------Start Cosine Kmeans---------#
    print(f"Class Kmeans++ with {str(k)} clusters begins\n") 
    clusters=KmeansPP(data,k)
    clusters.PCA_decompose()   
    clusters.init()
    print("Initiailize Cosine Kmeans++ with ",clusters.K_val," centroids ",clusters.centroids)
    clusters.first_cluster() 
    print("Iterating, this is going to be slow")
    clusters.further_cluster()
    
   

#--------------Start plotting for Cos--------------------#
    Cos_pca=Visulization(data,clusters.clusters,"Keans++ PCA w/ Cos Distance",file_names)
    Cos_pca.topic_extract()
    Cos_pca.PCA_decompose()
    Cos_pca.doc_names()
    print("Visualizing")
    Cos_pca.save()
    Cos_pca.PCA_print()
    

def main():
    print("reading TFIDF.csv, this is very slow")
    data=pd.read_csv("TFIDF.csv")
    data.rename( columns={'Unnamed: 0':'doc_name'}, inplace=True )
    print("processing data, SLOW")
    file_names=list(data["doc_name"])
    data.drop(columns=["doc_name","Unnamed: 1"], inplace=True)
    data= data.to_dict("records")

    for k in [2,3,5,8,10]:
        driver(data,k,file_names)


main()


  

        
        
    

    
    
    
