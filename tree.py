# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 05:45:19 2017

@author: Abhay
"""
import nltk

tree1 = nltk.Tree('NP', ['Alice'])
tree2 = nltk.Tree('m', ['Alice'])
tree3 = nltk.Tree('n', ['Abhay'])
tree4 = nltk.Tree('s', [tree1,tree2,tree3])

def traverse(tree41):
    try:
        tree41.label()
    except AttributeError:
        print(tree41)
    else:
        # Now we know that t.node is defined
        j = tree41.label()
        if(j == "n"):
            print(tree41)
        #print('(', tree41.label)
        for child in tree41:
            traverse(child)
        #print (')')

#tree4.draw()  
#if(tree4.label()=="s"):
#    print(tree4.leaves())
#print(tree4.label())
#print(tree4.leaves())
traverse(tree4)