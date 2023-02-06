# coding:utf-8  

import pandas as pd
import pathlib
import math
import csv
import random


class Tf_Idf(object):
    def __init__(self,dirname,filename,samplel_num):
        self.dirname=dirname
        self.filename=filename
        self.len=None
        self.corpus=self.reading_file() 
        self.IDF=None
        self.TFIDF=None
        self.max=samplel_num


    def reading_file(self):
        #read file from designated path and read it as split lines
        #self.corpus=list(),with each element being one article, words segmented with " "
        segmented = open(pathlib.Path(__file__).parent.parent / self.dirname / self.filename, "r",encoding='utf8')

        corpus=segmented.read().splitlines()
        corpus=random.sample(corpus,2000)
        
        self.len=len(corpus)
        for index in range(self.len):
            corpus[index]=corpus[index].split(" ")
        return corpus
    
    def IDF_score(self,articleList:list)->dict:
        #takes a list of articles as input and return a dictionary contains each word's IDF-Score
        articles_len=len(articleList)
        IDF=dict()
        for i in range(0,articles_len):
            #print("calculating IDF"+str(i))
            words_len=len(articleList[i])
            for j in range(1, words_len):
                The_word=articleList[i][j]
                if The_word in IDF.keys():
                    IDF[The_word].add(i)
                elif The_word not in IDF.keys():
                    IDF[The_word]=set()
                    #There will be no duplicates in set so we can just use it and take length
                    IDF[The_word].add(i)    
        for key , value in IDF.items():
            IDF[key]=math.log2(articles_len/float(len(IDF[key])))
        return IDF

    def get_idf2(self):
        #get overall idf score
        tempcorpus=list()
        for i in range(self.len):
            tempcorpus.append(self.corpus[i][1:])    
        IDF=self.IDF_score(tempcorpus)
        self.IDF=IDF

    def get_idf(self):
        IDF=self.IDF_score(self.corpus)
        self.IDF=IDF

    def write_idf(self):
        #write into IDF.csv
        with open('IDF.csv', 'w',encoding="utf8") as csvfile:
            writer = csv.DictWriter(csvfile,fieldnames=self.IDF.keys())
            writer.writeheader()
            writer.writerow(self.IDF)

    def TF_score(self,wordsList :list) -> dict:
        #takes a list of words and returns a dictionary that contains each words' TF score
        #adjust by the length of the article
        words_len=len(wordsList)
        TF=dict()
        for i in range(0,words_len):
            if wordsList[i] not in TF:
                TF[wordsList[i]]=1
            elif wordsList[i] in TF:
                TF[wordsList[i]]+=1

        for key,value in TF.items():
            
                TF[key]=value/(float(words_len))
            
                
            
        return TF

    def get_TFIDF(self):
        #get TF score for all articles
        #self.TF[title] refer to a dictionary that contains TF_score of this article
        self.TFIDF=dict()
        for i in range(self.len):

            if i%100==0: print(f"processing {str(i)} documents")
            title=self.corpus[i][0]
            self.TFIDF[title]=self.TF_score(self.corpus[i][1:])          
            
            for key in self.IDF.keys():               
                try:
                    self.TFIDF[title][key]=self.TFIDF[title][key]*self.IDF[key]               
                except :
                    self.TFIDF[title][key]=0
            
        


    def write_TFIDF(self):
        '''
        with open('TFIDF.csv', 'w',encoding="utf8") as csvfile:
            writer = csv.DictWriter(csvfile,fieldnames=self.IDF.keys())
            writer.writeheader()
            for TFIDF in self.TFIDF:
                writer.writerow(TFIDF)
        '''
        print("Processing data, it is going to take a while, you can leave it running and go do something eles")
        TFIDF_DF=pd.DataFrame.from_dict(self.TFIDF)    
        TFIDF_DF=TFIDF_DF.transpose()
        print("Writing to TFIDF.csv, still long time to wait")
        TFIDF_DF.to_csv("TFIDF.csv")
        

    


def driver(sample_num):
    TFIDF=Tf_Idf("article","segmented.txt",sample_num)
    TFIDF.reading_file()
    TFIDF.get_idf()
    TFIDF.get_TFIDF()
    TFIDF.write_idf()
    TFIDF.write_TFIDF()

driver(2000)


