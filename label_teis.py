#!/usr/bin/env python

"""
Stand-alone script to process directory with pdfs to obtain 
Probabilistic BiasRobot estimates on the full text of a clinical trial
"""

# Author:   Wim Otte <w.m.otte@umcutrecht.nl>

import os
import sys
import pandas as pd
import pickle
import argparse



#############################################

###
# Read pdf as binary file
##
def read_binary_pdf( pdffile ):
    f = open( pdffile, 'rb' )
    pdf_text = f.read()
    f.close()
    return( pdf_text )

###
# Get articles
##
def get_articles( pdffiles ):
    blobs = []
    for pdffile in pdffiles:
        blobs.append( read_binary_pdf( pdffile ) )

    # convert binary pdfs to articles
    pdf_reader = PdfReader()
    articles = pdf_reader.convert_batch( blobs )
    return( articles )

###
# Return list of pdfs
##
def get_list_of_pdfs( dir_name ):
	pdf_names = filter( lambda x: x.endswith( '.pdf' ), os.listdir( dir_name ) )
	pdf_files = [ dir_name + "/" + s for s in pdf_names ]
	return pdf_files

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
def classify_articles( articles, pdffiles ):

    # get probabilistic bias robot
    robot = ProbBiasRobot()
    
    # annotate and convert to data.frame
    df_all = pd.DataFrame()
    
    for article, pdffile in zip( articles, pdffiles ):
        print( pdffile )
        df = pd.DataFrame( robot.annotate( article, pdffile ) )
        df_all = pd.concat( [ df_all, df ] )

    return( df_all )


###############################################

def main():
    # argument parser
    parser = argparse.ArgumentParser( description = 'Determine probabilistic bias in clinical trial pdfs' )
    parser.add_argument( '-i','--indir', help = 'Input directory - all pdfs will be parsed', required = True )
    parser.add_argument( '-o','--outfile', help = 'Output csv file with pdf names and corresponding results', required = True )
    parsed = parser.parse_args()

    from robotreviewer.robots.prob_bias_robot import ProbBiasRobot
    from robotreviewer.textprocessing.pdfreader import PdfReader
    from robotreviewer.textprocessing.tokenizer import nlp
    
    # get input/output info
    indir = parsed.indir
    outfile = parsed.outfile

    # get list of pdfs
    print( '*** Get list of pdfs ***' )
    pdffiles = get_list_of_pdfs( indir )

    # get articles
    print( '*** Get articles ***' )
    articles = get_articles( pdffiles )

    # prepare with tokenizer etc.
    print( '*** Prepare articles ***' )
    prep_articles = prepare_articles( articles )
    
    # classify articles
    print( '*** Classify articles ***' )
    class_articles = classify_articles( prep_articles, pdffiles )

    # write data.frame to csv
    print( '*** Write csv output ***' )
    class_articles.to_csv( outfile, index = False, header = True )

###############################################

# run main
if __name__== "__main__":
    main()

