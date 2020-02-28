# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 13:57:34 2020

@author: cgasca
"""

import numpy as np
import pandas as pd
import json
from sklearn.feature_extraction.text import CountVectorizer 
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer

#leer documentos
#tokenizar
#quitar stopwords
#lematizar

#calcular frecuencias (matriz bow)

#entrenar modelo
#guardar en json
class Modelo_entrenado:
  def __init__(self, p_phishing, p_ham,prior_phishing,prior_ham):
    self.p_phishing = p_phishing
    self.p_ham = p_ham
    self.prior_phishing = prior_phishing
    self.prior_ham = prior_ham
    

    
def getTokens(corpus):
    '''Recibe lista de texto y devuelve texto tokenizado sin digitos ni puntuacion y sin stopwords'''
    tokenizer = RegexpTokenizer(r'\w+')
    corpus_tokenized = []
    stopwords = getStopWords("stopwords-es.txt")
    for c in corpus:
        tokens = tokenizer.tokenize(c)
        text = []
        for t in tokens:    
            t = t.lower()
            if t.isalpha() and t not in stopwords:
                text.append(t)
            #else:
                #print(t)
        corpus_tokenized.append(text)
    return corpus_tokenized

def getStopWords(filename):
    '''Recibe path de archivo con stopword y devuelve lista de stopwords'''
    f = open(filename,"r",encoding='utf-8')
    stopwords = f.readlines()
    for i in range(len(stopwords)):
        stopwords[i] = stopwords[i].replace("\n",'')
    return stopwords

def get_bagOfWords(corpus_tokenized):
    '''Recibe emails dividido en tokens y devuelve modelo BOW'''
    corpus_tokenized_text = []
    for c in corpus_tokenized:
        '''Cada email divido en tokens lo concatena de nuevo'''
        corpus_tokenized_text.append(' '.join(c))
    vectorizer = CountVectorizer(max_features=500)
    term_document_matrix = vectorizer.fit_transform(corpus_tokenized_text).todense()
    dic = vectorizer.vocabulary_
    return term_document_matrix,dic


def fillWithZero(term_bow):
    '''Llena dos diccionarios con las mismas llaves de term_bow con puros 0'''
    p_phishing = {}
    p_ham = {}
    for i in range(len(term_bow)):
        p_phishing[getKeyByValue(term_bow,i)] = 0
        p_ham[getKeyByValue(term_bow,i)] = 0
    return p_phishing,p_ham

def train(X_train,y_train,totalOfEmails,totalOfPhishing,totalOfHam,term_bow,smoothing=True):
    '''Calcula las probabilidades y regresa diccionario de probabilidades condicionales y los prior probabilities'''
    A_p = 0
    A_h = 0
    lambdaSmoothing = 0
    
    if smoothing:
        A_p,A_h = smoothing_(X_train,y_train,len(term_bow))
        lambdaSmoothing = 1    
    prior_phishing = (totalOfPhishing + lambdaSmoothing)/(totalOfEmails + (len(set(y))*lambdaSmoothing))
    prior_ham = (totalOfHam + lambdaSmoothing) / (totalOfEmails + (len(set(y))*lambdaSmoothing))
    p_phishing,p_ham = fillWithZero(term_bow)
    for i in range(totalOfEmails):
        for j in range(len(term_bow)):
            value = X_train.item(i,j)
            if y_train[i]=="phishing":
                p_phishing[getKeyByValue(term_bow,j)] += value
            else:
                p_ham[getKeyByValue(term_bow,j)] += value
                
    for key,value in term_bow.items():
        p_phishing[key] = (p_phishing[key] + lambdaSmoothing) / (totalOfPhishing + A_p*lambdaSmoothing)
        p_ham[key] = (p_ham[key] + lambdaSmoothing) / (totalOfHam + A_h*lambdaSmoothing)
        

    return p_phishing,p_ham,prior_phishing,prior_ham


            
def smoothing_(X,y,total_of_terms):
    k_p = 0
    k_h = 0
    for i in range(len(X)):
        if y[i] == "phishing":
            for j in range(total_of_terms):
                if X.item(i,j)>0:
                    k_p += 1
        else:
            for j in range(total_of_terms):
                if X.item(i,j)>0:
                    k_h += 1
            
            
    return k_p,k_h
                
    
def test(email_tokenized,p_phishing,p_ham,prior_phishing,prior_ham):
    '''Hace prueba de un email diferente al del corpus'''
    conditional_probability_phishing = 1
    conditional_probability_ham = 1
    for e in email_tokenized:
        try:
            if p_phishing[e] != 0:
                conditional_probability_phishing *= p_phishing[e]
            if p_ham[e] != 0:
                conditional_probability_ham *= p_ham[e]
            
        except KeyError:
            print(e)
    
    print(prior_phishing*conditional_probability_phishing)
    print(prior_ham*conditional_probability_ham)
    result_phishing = prior_phishing*conditional_probability_phishing
    result_ham = prior_ham*conditional_probability_ham
    if result_phishing > result_ham:
        return "phishing"
    else:
        return "ham"
    

def getKeyByValue(term_bow,value):
    '''Regresa la llave que le corresponde a un valor en un diccionario'''
    k = ""
    for key in term_bow.keys():    # for name, age in dictionary.iteritems():  (for Python 2.x)
        if term_bow[key] == value:
            k = key
    return k
     

def normalization(corpus):
    stemmer = SnowballStemmer('spanish')
    corpus_tokenized = getTokens(corpus)
    stemmed_text = []
    for c in corpus_tokenized:
        stemmed_text.append([stemmer.stem(i) for i in c])
    return stemmed_text

    
if __name__ == '__main__':
    corpus = ["Ha ganado 500,000 dólares de la Lotería Nacional del Reino Unido, responda para reclamar su precio.",
          "Tu Apple ID ha sido bloqueada por razones de seguridad, para desbloquear debes verificar tu identidad.",
          "Buenas tardes, miércoles 18:00 me queda bien, ¿cuando empezamos?",
          "Saludos, por medio del presente correo les estoy adjuntando el formato de la primera práctica. Hasta pronto."]
    etiquetas=["phishing","phishing","ham","ham"]
    
    #corpus y etiquetas deben leerse de un archivo .csv
    
    
    corpus_normalized = normalization(corpus) #normalizacion de correos (tokenizacion, eliminacion de stopwords, [lemmatizacion,stemming,tfidf])
    term_document_matrix,term_bow = get_bagOfWords(corpus_normalized) #bag of words y diccionario de terminos
    X =  term_document_matrix
    y = etiquetas
    p_phishing,p_ham,prior_phishing,prior_ham = train(X,y,4,2,2,term_bow) #modelo entrenado, diccionario de P(X=""|Y=phishing), diccionario de P(X=""|y=ham), p(y=phishing),p(y=ham)
    
    modelo_entrenado = Modelo_entrenado(p_phishing,p_ham,prior_phishing,prior_ham)
    modelo_entrenado_json = json.dumps(modelo_entrenado.__dict__)
    file_json=open("modelo_entrenado_json.json",'w')
    file_json.write(modelo_entrenado_json)
    file_json.close()
    
    #*********************************test*****************************************
    prueba = ["Saludos, nos vemos el miércoles para desbloquear su correo"]
    prueba_tokens = normalization(prueba)
    print(test(prueba_tokens[0],p_phishing,p_ham,prior_phishing,prior_ham)) #imprime phishing o ham
   


    

#
#print(term_document_matrix.item(1,11))

#print(getKeyByValue(term_bow,1))






