#!/usr/bin/env python
'''
Created on 01.03.2019

@author: Philipp
'''

from argparse import ArgumentParser
from shatt.configuration import Configuration
from shatt.scripts import OPTION_DRY_RUN
from pathlib import Path
from shatt.experiments.fixed_attention import start_fixed_attention_from_config
from shatt.experiments.semi_fixed_attention import start_semi_fixed_attention_from_config
from shatt.experiments.dynamic_attention import start_dynamic_attention_from_config
from shatt.experiments.extern.extern_semi_attention import start_extern_semi_fixed_attention_from_config
from shatt.experiments.extern.extern_fixed_attention import start_extern_fixed_attention_from_config
from shatt.experiments.extern.extern_dynamic_attention import start_extern_dynamic_attention_from_config


def main():
    parser = ArgumentParser("Start the model using the configuration.ini")
    parser.add_argument("command", help="""One of ["fixed", "semi", "dynamic","extern-fixed","extern-semi","extern-dynamic"]. 
                        fixed: Perform the experiment using fixed attention based on bounding boxes. This experiment requires bounding box files.
                        """)
    parser.add_argument("-c", "--configuration", help="Determine a specific configuration to use. If not specified, the default is used.")
    parser.add_argument("-m", "--path_to_model", help="The absolute path to the model.")
    parser.add_argument("-s", "--split_name", help="""The split name to perform the experiment on. One of [train, validate, test_dev, test, trainval].
                        """)
    parser.add_argument("-t", "--target_path", help="Directory where the result file is stored (default: user home)")
    parser.add_argument("-d", "--dryrun", action="store_true")
    parser.add_argument("-w", "--input_weight", type=int, help="The input weight factor for dynamic attention")
    parser.add_argument("-i", "--input_iterations", type=int, help="The amount of iterations to fix for semi fixed attention")
    
    run_opts = parser.parse_args()
    
    if run_opts.configuration:
        config = Configuration(run_opts.configuration)
    else:
        config = Configuration()
    config[OPTION_DRY_RUN] = run_opts.dryrun
    
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.getGpuDevices())
        
    if run_opts.command in ["fixed", "semi", "dynamic","extern-fixed","extern-semi","extern-dynamic"]:
        if not run_opts.path_to_model:
            raise Exception("Cannot experiment, when not model path is given. Please provide the path to the model and retry.")
        if not run_opts.split_name:
            raise Exception("Cannot experiment, when no split name is given. Please provide the split name and retry.")
        target_split = run_opts.split_name
        if not target_split:
            print("Error: Cannot experiment, when not both source and target split name are given: {}".format(run_opts.split_name))
            raise Exception("Please provide the split name like '-s train test' and retry.")
        
        target_path = str(Path.home())
        if run_opts.target_path:
            target_path = run_opts.target_path
        print("Target directory for the result files:", target_path)
        
        if run_opts.command == "fixed":             
            start_fixed_attention_from_config(config, run_opts.path_to_model, target_split, target_path)
            
        if run_opts.command == "extern-fixed":             
            start_extern_fixed_attention_from_config(config, run_opts.path_to_model, target_split, target_path)
        
        if run_opts.command == "semi":                        
            if not run_opts.input_iterations:
                raise Exception("Cannot experiment semi, when no input_iterations is given. Please provide the value and retry.")
            input_iterations = run_opts.input_iterations
            start_semi_fixed_attention_from_config(config, run_opts.path_to_model, target_split, target_path, input_iterations)
            
        if run_opts.command == "extern-semi":                        
            if not run_opts.input_iterations:
                raise Exception("Cannot experiment semi, when no input_iterations is given. Please provide the value and retry.")
            input_iterations = run_opts.input_iterations
            start_extern_semi_fixed_attention_from_config(config, run_opts.path_to_model, target_split, target_path, input_iterations)

        if run_opts.command == "dynamic":           
            if not run_opts.input_weight:
                raise Exception("Cannot experiment dynamic, when no input_weight is given. Please provide the value and retry.")  
            input_weight = run_opts.input_weight
            start_dynamic_attention_from_config(config, run_opts.path_to_model, target_split, target_path, input_weight)
            
        if run_opts.command == "extern-dynamic":           
            if not run_opts.input_weight:
                raise Exception("Cannot experiment dynamic, when no input_weight is given. Please provide the value and retry.")  
            input_weight = run_opts.input_weight
            start_extern_dynamic_attention_from_config(config, run_opts.path_to_model, target_split, target_path, input_weight)
        
if __name__ == '__main__':
    main()
    
