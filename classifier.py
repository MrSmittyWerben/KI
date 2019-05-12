#!/usr/bin/env python

import sys, os, argparse, json
import numpy as np
import pickle


"""
  "News Classifier" 
  -------------------------
  This is a small interface for document classification. Implement your own Naive Bayes classifier 
  by completing the class 'NaiveBayesDocumentClassifier' below.

  To run the code, 

  1. place the files 'train.json' and 'test.json' in the current folder.

  2. train your model on 'train.json' by calling > python classifier.py --train 

  3. apply the model to 'test.json' by calling > python classifier.py --apply

"""


class NaiveBayesDocumentClassifier:

    
    def __init__(self):

        """ The classifier should store all its learned information
            in this 'model' object. Pick whatever form seems appropriate
            to you. Recommendation: use 'pickle' to store/load this model! """
        self.path = "./dict.pickle"
        if(os.path.isfile(self.path)):
            file = open(str(self.path),"rb")
            self.model = pickle.load(file)
        else:
            self.model = None

        
    def train(self, features, labels):
        """
        trains a document classifier and stores all relevant
        information in 'self.model'.

        @type features: dict
        @param features: Each entry in 'features' represents a document
                         by its so-called bag-of-words vector. 
                         For each document in the dataset, 'features' contains 
                         all terms occurring in the document and their frequency
                         in the document:
                         {
                           'doc1.html':
                              {
                                'the' : 7,   # 'the' occurs seven times
                                'world': 3, 
                                ...
                              },
                           'doc2.html':
                              {
                                'community' : 2,
                                'college': 1, 
                                ...
                              },
                            ...
                         }
        @type labels: dict
        @param labels: 'labels' contains the class labels for all documents
                       in dictionary form:
                       {
                           'doc1.html': 'arts',       # doc1.html belongs to class 'arts'
                           'doc2.html': 'business',
                           'doc3.html': 'sports',
                           ...
                       }
        """
        vocab = set()  # Alle Woerter ohne Duplikate aus allen Artikeln
        for k, v in features.items():
            for k1, v1 in v.items():
                vocab.add(k1)

        priors = {}  # Liste mit allen Labeln und derer Wahrscheinlichkeiten
        for k, v in labels.items():
            if priors.get(v, 0) != 0:
                priors[v] += 1
            else:
                priors[v] = 1


        for k, v in features.items():
            for term in vocab:
                if term in v.keys():
                    v.update({term: 1})
                else:
                    v[term] = 0


        bow_labled = {}
        for k, v in features.items():
            if labels[k] not in bow_labled.keys():
                bow_labled[labels[k]] = v
            else:
                for k1, v1 in v.items():
                    bow_labled[labels[k]][k1] += v1

        for k,v in bow_labled.items():
            for k1,v1 in v.items():
                v[k1]= v1/priors.get(k,1)
                if v[k1] == 0:
                    v[k1] = 10**-100 #da Wsh. nie 0 ist

        for k, v in priors.items():
            priors[k] = float(v / len(labels))

        for c in priors:
            print(c, bow_labled[c]["food"])

        data = [bow_labled,priors]
        file = open(self.path,"wb")
        pickle.dump(data, file)
        file.close()





        
    def apply(self, features):
        """
        applies a classifier to a set of documents. Requires the classifier
        to be trained (i.e., you need to call train() before you can call apply()).

        @type features: dict
        @param features: see above (documentation of train())

        @rtype: dict
        @return: For each document in 'features', apply() returns the estimated class.
                 The return value is a dictionary of the form:
                 {
                   'doc1.html': 'arts',
                   'doc2.html': 'travel',
                   'doc3.html': 'sports',
                   ...
                 }
        """

        bow_labled, priors = self.model

        result = {}
        for doc_name, words_in_doc in features.items():
            argmax = {}
            for prior, prior_value in priors.items():
                arg = 0
                arg = np.log(prior_value)
                for term in words_in_doc: #prev. words_in_doc
                    if term in bow_labled[prior]: #prev. bow_labled[prior]
                        arg += np.log(bow_labled[prior][term])
                    #wörter welche nicht in bow_labeld sind -> ignorieren
                    else:
                        arg += np.log(1 - bow_labled[prior][term])

                argmax.update({prior : arg})
            result.update({doc_name: sorted(argmax.items(), key=lambda kv: kv[1], reverse=True)[0][0] })

        return result



if __name__ == "__main__":

    # parse command line arguments (no need to touch)
    parser = argparse.ArgumentParser(description='A document classifier.')
    parser.add_argument('--train', help="train the classifier", action='store_true')
    parser.add_argument('--apply', help="apply the classifier (you'll need to train or load"\
                                        "a trained model first)", action='store_true')
    parser.add_argument('--inspect', help="get some info about the learned model",
                        action='store_true')

    args = parser.parse_args()

    classifier = NaiveBayesDocumentClassifier()

    def read_json(path):
        with open(path) as f:
            data = json.load(f)['docs']
            features,labels = {},{}
            for f in data:
                features[f] = data[f]['tokens']
                labels[f]   = data[f]['label']
        return features,labels
    
    if args.train:
        features,labels = read_json('train_filtered.json')
        classifier.train(features, labels)

    if args.apply:
        features,labels = read_json('test_filtered.json')
        result = classifier.apply(features)

        failures = 0
        for doc_name, label in result.items():
            doc_label = doc_name.split("/")[2]
            if label != doc_label:
                failures += 1
                print(doc_name, "!=", label)
            else:
                print(doc_name,"==", label)

        succes_rate = (1-(failures/len(result)))*100

        print("Erfolgsrate:", succes_rate)




            
    
