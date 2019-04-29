#!/usr/bin/env python

"""
Stand-alone script to process directory with xmls to obtain 
Probabilistic BiasRobot estimates on the full text of a clinical trial
"""

# Author:   Wim Otte <w.m.otte@umcutrecht.nl>

import os
import sys
import pandas as pd
import pickle
import argparse

from robotreviewer.robots.prob_bias_robot import ProbBiasRobot
from robotreviewer.robots.prob_rct_robot import RCTRobot
from robotreviewer.textprocessing.pdfreader import PdfReader
from robotreviewer.textprocessing.tokenizer import nlp


#############################################

###
# Read xml content.
##
def read_xml( xml_file ):
	f = open( xml_file, 'r' )
	xml = f.read()
	f.close()

	return( xml )

###
# Get articles
##
def get_articles( xml_files ):
	reader = PdfReader()
	articles = []
	for xml_file in xml_files:
		xml = read_xml( xml_file )
		xml_text = reader.parse_xml( xml )
		articles.append( xml_text )
		   
	return( articles )

###
# Return list of xmls
##
def get_list_of_xmls( dir_name ):
	xml_names = filter( lambda x: x.endswith( '.xml' ), os.listdir( dir_name ) )
	xml_files = [ dir_name + "/" + s for s in xml_names ]

	return xml_files

###
# Preparing text
##
def prepare_articles( articles ):
    parsed_articles = []
    out_articles = []
    for doc in nlp.pipe( ( d.get('text', u'') for d in articles), batch_size = 1, n_threads = 6 ):
        parsed_articles.append( doc )

    for article, parsed_text in zip( articles, parsed_articles ):
        article._spacy['parsed_text'] = parsed_text
        out_articles.append( article )

    return( articles )

###
# Get bias probabilies
##
def classify_articles( articles, xml_files ):

    # get probabilistic bias robot
    robot = ProbBiasRobot()
    
    # annotate and convert to data.frame
    df_all = pd.DataFrame()
    
    for article, xml_file in zip( articles, xml_files ):
        print( xml_file )
        df = pd.DataFrame( robot.annotate( article, xml_file ) )
        df_all = pd.concat( [ df_all, df ] )

    return( df_all )


###############################################

def main():
    # argument parser
    parser = argparse.ArgumentParser( description = 'Determine probabilistic bias in clinical trial xmls' )
    parser.add_argument( '-i','--indir', help = 'Input directory - all xmls will be parsed', required = True )
    parser.add_argument( '-o','--outfile', help = 'Output csv file with xml names and corresponding results', required = True )
    parsed = parser.parse_args()
    
    # get input/output info
    indir = parsed.indir
    outfile = parsed.outfile

    # get list of xmls
    print( '*** Get list of xmls ***' )
    xml_files = get_list_of_xmls( indir )

    # get articles
    print( '*** Get articles ***' )
    articles = get_articles( xml_files )

    # prepare with tokenizer etc.
    print( '*** Prepare articles ***' )
    prep_articles = prepare_articles( articles )
    
    # prepare with tokenizer etc.
    print( '*** Determine Bias ***' )
    class_articles = classify_articles( prep_articles, xml_files )

    # TODO
    print( class_articles )

    # write data.frame to csv
    print( '*** Write csv output ***' )
    class_articles.to_csv( outfile, index = False, header = True )


###############################################

# run main
if __name__== "__main__":
    main()

