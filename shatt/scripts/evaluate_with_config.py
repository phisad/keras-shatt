#!/usr/bin/env python
'''
Created on 01.03.2019

@author: Philipp
'''

from argparse import ArgumentParser

from shatt.configuration import Configuration
from shatt.evaluation import to_model_dir
from shatt.evaluation.categorical_capabilities import categorical_capability_from_config
from shatt.evaluation.categorical_embeddings import compute_word_embeddings, \
    categorical_knearest_neighbors_from_config
from shatt.scripts import OPTION_DRY_RUN
from shatt.evaluation.results.categorical import compute_categorical_matches_from_model_dir_at_k


def main():
    parser = ArgumentParser("Start the model using the configuration.ini")
    parser.add_argument("command", help="""One of [capabilities, embeddings, neighbors, matches, all]. 
                        capabilities: Analyse how many categories are incorporated in caption corpus and generated caption results for [gt, fixed, boxes, alternating].
                        embeddings: Compute the embeddings for all words within the vocabulary. 
                        neighbors: Compute the k-nearest neighbors for the categories. This requires embeddings.
                        matches: Compute how many categories are matched by the the experimental result captions. This requires neighbors and capabilities. 
                        """)
    parser.add_argument("-c", "--configuration", help="Determine a specific configuration to use. If not specified, the default is used.")
    parser.add_argument("-m", "--model_dir", help="The path to the model directory.")
    parser.add_argument("-k", "--k_neighbors", type=int, default=5, help="The number of nearest neighbors to involve")
    parser.add_argument("-s", "--strict", action="store_true", default=False, help="Strict-mode: Caption must contain both words in case of two word categories (default: false)")
    parser.add_argument("-t", "--tight", action="store_true", default=False, help="Tight-mode: All neighbors@k must NOT be contained in the control caption (default: false)")
    parser.add_argument("-e", "--epoch", type=int, help="The epoch to evaluate e.g. to determine the name of result files or checkpoint 'shatt.epoch.h5'.")
    parser.add_argument("-n", "--experiment_name", help="The experiment name. One of [fixed, semi, dynamic].")
    
    run_opts = parser.parse_args()
    
    if run_opts.configuration:
        config = Configuration(run_opts.configuration)
    else:
        config = Configuration()
    
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.getGpuDevices())
        
    if not run_opts.model_dir:
        raise Exception("Cannot evaluate, when not model path is given. Please provide the path to the model and retry.")
    
    if run_opts.command in ["capabilities", "all"]:
        print("\nAnalyse how many categories are incorporated in caption corpus and generated caption results for [gt, fixed, boxes, alternating]")
        if not run_opts.epoch:
            raise Exception("Cannot analyze categorical capabilities, when no epoch is given. Please provide the epoch and retry.")
        model_dir = to_model_dir(run_opts.model_dir)
        categorical_capability_from_config(config, model_dir, run_opts.epoch, experiment_name= "attention_" + run_opts.experiment_name)
        
    if run_opts.command in ["embeddings", "all"]:
        print("\nCompute the embeddings for all words within the vocabulary.")
        if not run_opts.epoch:
            raise Exception("Cannot compute embeddings, when no epoch is given. Please provide the epoch and retry.")
        model_dir = to_model_dir(run_opts.model_dir)
        compute_word_embeddings(model_dir, run_opts.epoch)
        
    if run_opts.command in ["neighbors", "all"]:
        print("\nCompute the k-nearest neighbors (k={}) for the categories. This requires embeddings.".format(run_opts.k_neighbors))
        model_dir = to_model_dir(run_opts.model_dir)
        categorical_knearest_neighbors_from_config(config, model_dir, k=run_opts.k_neighbors)

    if run_opts.command in ["matches", "all"]:
        print("\nCompute how many categories are matched by the the experimental result captions. This requires neighbors and capabilities.")
        if not run_opts.epoch:
            raise Exception("Cannot compute matches, when no epoch is given. Please provide the epoch and retry.")
        if not run_opts.experiment_name:
            raise Exception("Cannot compute matches, when no experiment name is given. Please provide the value and retry.")
        model_dir = to_model_dir(run_opts.model_dir)
        dataset_dir = config.getDatasetTextDirectoryPath()
        if run_opts.strict:
            print("Strict-mode: Caption must contain both words in case of two word categories")
        if run_opts.tight:
            print("Tight-mode: All neighbors@k must NOT be contained in the control caption")
        else:
            print("Loose-mode: Only the box category must NOT be contained in the control caption")
        compute_categorical_matches_from_model_dir_at_k(dataset_dir, 
                                                        model_dir, 
                                                        run_opts.epoch,
                                                        experiment_name= "attention_" + run_opts.experiment_name, 
                                                        do_strict=run_opts.strict, 
                                                        do_tight=run_opts.tight, 
                                                        k_list=[1, run_opts.k_neighbors])

        
if __name__ == '__main__':
    main()
    
