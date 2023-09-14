#!/usr/bin/env python3

#Adapted from https://github.com/facebookresearch/DrQA/blob/master/scripts/retriever/build_db.py
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""A script to build the tf-idf document matrices for retrieval."""
import os
import pathlib
from drqascripts.retriever.build_tfidf import *
try:
    from log import get_logger
except:
    from .log import get_logger

if __name__ == '__main__':
    logger = get_logger("DrQA Build TFIDF")

    logger.info("Build TF-IDF matrix")

    parser = argparse.ArgumentParser()
    parser.add_argument('--db_path', type=str, default="./data/wikipedia.db",
                        help='Path to sqlite db holding document texts')
    parser.add_argument('--out_path', type=str, default="./data/tfidf.npz",
                        help='Directory for saving output files')
    parser.add_argument('--ngram', type=int, default=2,
                        help=('Use up to N-size n-grams '
                              '(e.g. 2 = unigrams + bigrams)'))
    parser.add_argument('--hash-size', type=int, default=int(math.pow(2, 24)),
                        help='Number of buckets to use for hashing ngrams')
    parser.add_argument('--tokenizer', type=str, default='simple',
                        help=("String option specifying tokenizer type to use "
                              "(e.g. 'corenlp')"))
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of CPU processes (for tokenizing, etc)')
    args = parser.parse_args()

    tfidf_builder = TfIdfBuilder(args,'sqlite', {'db_path': args.db_path})
    logging.info('Counting words...')
    count_matrix, doc_dict = tfidf_builder.get_count_matrix()

    logger.info('Making tfidf vectors...')
    tfidf_mat = tfidf_builder.get_tfidf_matrix(count_matrix)

    logger.info('Getting word-doc frequencies...')
    freqs = tfidf_builder.get_doc_freqs(count_matrix)

    logger.info('Saving to {}'.format(args.out_path))
    metadata = {
        'doc_freqs': freqs,
        'tokenizer': args.tokenizer,
        'hash_size': args.hash_size,
        'ngram': args.ngram,
        'doc_dict': doc_dict
    }
    retriever.utils.save_sparse_csr(args.out_path, tfidf_mat, metadata)
