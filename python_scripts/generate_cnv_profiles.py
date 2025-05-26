#%% Import libraries
import pandas as pd
import numpy as np
import os
import random
import argparse
from collections import Counter
from samtools_utils import *

# np.random.seed(17321)
# random.seed(17321)

BASEDIR = "/data1/shahs3/users/sunge/cnv_simulator"
DATADIR = os.path.join(BASEDIR, "data")

#%% Functions

def copy_state_map(state = None, copy = None):
    """
    Map copy states to their corresponding copy numbers.
    """
    copy_state_map = {0:-2, 1:-1, 2:0, 3:1, 4:2, 5:3, 6:4, 7:5, 8:6, 9:7,
                      10:8, 11:9, 12:10, 13:11, 14:12}
    state_copy_map = {-2:0, -1:1, 0:2, 1:3, 2:4, 3:5, 4:6, 5:7, 6:8, 7:9,
                      8:10, 9:11, 10:12, 11:13, 12:14}
    
    if (state != None) and (state in state_copy_map):
        return state_copy_map[state]
    elif (copy != None) and (copy in copy_state_map):
        return copy_state_map[copy]
    else:
        raise ValueError(f"Invalid state {state} or copy {copy} for copy_state_map")
    

def parse_cnv_profile_input(input_path):
    """
    Parse input CNV file, validate columns, and convert to appropriate types.
    """
    try:
        profile_input = pd.read_csv(input_path, sep="\t")
        # Validate columns
        required_columns = [
            "num_clones", "cells_per_clone_lst", "num_cnvs_per_clone",
            "chr_lst", "min_cnv_size", "max_cnv_size", "possible_copy_numbers"
        ]
        for col in required_columns:
            if col not in profile_input.columns:
                raise ValueError(f"Missing required column: {col}")
            
        profile_input['num_clones'] = profile_input['num_clones'].astype(int)
        profile_input['cells_per_clone_lst'] = profile_input['cells_per_clone_lst'].astype(str).str.split(",")
        profile_input['cells_per_clone_lst'] = profile_input['cells_per_clone_lst'].apply(lambda x: [int(i) for i in x])
        profile_input['num_cnvs_per_clone'] = profile_input['num_cnvs_per_clone'].astype(str).str.split(",")
        profile_input['num_cnvs_per_clone'] = profile_input['num_cnvs_per_clone'].apply(lambda x: [int(i) for i in x])
        profile_input['chr_lst'] = profile_input['chr_lst'].astype(str).str.split(",")
        profile_input['min_cnv_size'] = profile_input['min_cnv_size'].astype(int)
        profile_input['max_cnv_size'] = profile_input['max_cnv_size'].astype(int)
        profile_input['possible_copy_numbers'] = profile_input['possible_copy_numbers'].astype(str).str.split(",")
        profile_input['possible_copy_numbers'] = profile_input['possible_copy_numbers'].apply(lambda x: [int(i) for i in x])

        # If chromosome list only contains 0, then include all chromosomes
        all_chr_lst = [str(i) for i in range(1, 23)] + ["X", "Y"]
        profile_input['chr_lst'] = profile_input['chr_lst'].apply(
            lambda x: all_chr_lst.copy() if x == ["0"] else x
        )
    except Exception as e:
        raise ValueError(f"Error reading or validating input file: {e}")    
    
    return profile_input

def fill_in_cnv_profile(cnv_profile, chr_lengths_df, chr_lst):
    filled_cnv_profile_lst = []
    for clone in cnv_profile['clone'].unique():
        for chr in chr_lst:
            clone_chr_cnv = cnv_profile[(cnv_profile['clone'] == clone) & (cnv_profile['chr'] == chr)]
            chr_length = chr_lengths_df[chr_lengths_df['chr'] == chr]['length'].values[0]
            if len(clone_chr_cnv) == 0: # no CNVs for this clone and chromosome
                filled_cnv_profile_lst.append({"clone": clone, "chr": chr, "start": 0, "end": chr_length,
                                               "copy_number": 2, "state": 0, "size": chr_length})
            else:
                current_chr_pos = 0
                for _, row in clone_chr_cnv.iterrows():
                    if current_chr_pos < row['start']:
                        filled_cnv_profile_lst.append({"clone": clone, "chr": chr, "start": current_chr_pos,
                                                       "end": row['start'], "copy_number": 2, "state": 0,
                                                       "size": row['start'] - current_chr_pos})
                    filled_cnv_profile_lst.append({"clone": clone, "chr": chr, "start": row['start'],
                                                   "end": row['end'], "copy_number": row['copy_number'],
                                                   "state": row['state'], "size": row['size']})
                    current_chr_pos = row['end']
                if current_chr_pos < chr_length:
                    filled_cnv_profile_lst.append({"clone": clone, "chr": chr, "start": current_chr_pos,
                                                   "end": chr_length, "copy_number": 2, "state": 0,
                                                   "size": chr_length - current_chr_pos})
    filled_cnv_profile = pd.DataFrame(filled_cnv_profile_lst)
    filled_cnv_profile = filled_cnv_profile.sort_values(by=["clone", "chr", "start"]).reset_index(drop=True)

    return filled_cnv_profile
                    

def make_cnv_profile(row, chr_lengths_df):
    """
    Generates CNV profile given input parameters
    """
    
    # Check whether select number of CNVs fit in selected chromosomes
    total_chr_length = chr_lengths_df[chr_lengths_df['chr'].isin(row['chr_lst'])]['length'].sum()
    for num_cnvs in row['num_cnvs_per_clone']:
        total_max_cnv_size = num_cnvs * row['max_cnv_size']
        if total_max_cnv_size > total_chr_length:
            raise ValueError(f"Total CNV size exceeds chromosome length for {row['test_id']}.")
        
    # Filter selected chromosomes
    select_chr_lengths_df = chr_lengths_df[chr_lengths_df['chr'].isin(row['chr_lst'])]

    # Create CNV profile
    cnv_profile = pd.DataFrame(columns=["clone", "chr", "start", "end", "copy_number", "state"])
    for clone in range(row['num_clones']):
        logger.info(f"Generating CNV profile for clone {clone} of {row['num_clones']}")
        logger.debug(f"Selected chromosomes: {row['chr_lst']}")
        logger.debug(f"Contains {row['cells_per_clone_lst'][clone]} cells")

        clone_cnv_count = row['num_cnvs_per_clone'][clone]
        logger.info(f"Number of CNVs for clone {clone}: {clone_cnv_count}")

        # Assign number of CNVs per chromosome based on chromosome lengths
        num_cnvs_per_chr = Counter(np.random.choice(row['chr_lst'], 
                                                    size = clone_cnv_count,
                                                    p = select_chr_lengths_df['length'] / total_chr_length))
        for chr in row['chr_lst']:
            if chr not in num_cnvs_per_chr:
                num_cnvs_per_chr[chr] = 0
        num_cnvs_per_chr = dict(sorted(num_cnvs_per_chr.items(), key=lambda x: x[0]))

        # Randomly sample CNV start and end positions for each chromosome
        for chr, num_cnvs in num_cnvs_per_chr.items():
            chr_cnv_df = pd.DataFrame(columns=["clone", "chr", "start", "end", "copy_number", "state"])
            chr_length = select_chr_lengths_df[select_chr_lengths_df['chr'] == chr]['length'].values[0]

            for _ in range(num_cnvs_per_chr[chr]):
                chr_length = select_chr_lengths_df[select_chr_lengths_df['chr'] == chr]['length'].values[0]
                # Sample CNV start and end until no overlap with existing CNVs
                while True:
                    cnv_start = random.randint(0, chr_length - row['max_cnv_size'])
                    cnv_size = random.randint(row['min_cnv_size'], row['max_cnv_size'])
                    cnv_end = cnv_start + cnv_size

                    overlaps = False
                    for _, chr_cnv_row in chr_cnv_df.iterrows():
                        existing_start = chr_cnv_row['start']
                        existing_end = chr_cnv_row['end']
                        if not (cnv_end < existing_start or cnv_start > existing_end):
                            overlaps = True
                            break
                    if not overlaps:
                        break

                cnv_copy_number = random.choice(row['possible_copy_numbers'])
                cnv_state = copy_state_map(copy = cnv_copy_number)
                chr_cnv_df = pd.concat([chr_cnv_df,
                                        pd.DataFrame([{"clone": clone, "chr": chr, "start": cnv_start, "end": cnv_end,
                                                      "copy_number": cnv_copy_number, "state": cnv_state, "size": cnv_size}])],
                                        ignore_index=True)
            # Add CNVs to CNV profile
            cnv_profile = pd.concat([cnv_profile, chr_cnv_df], ignore_index=True)
                    
    logger.info(f"CNV profile for clone {clone} generated successfully.")

    # Sort CNV profile by clone, chromosome and start position
    cnv_profile = cnv_profile.sort_values(by=["clone", "chr", "start"]).reset_index(drop=True)

    # Fill in missing CNVs with copy number 2
    cnv_profile = fill_in_cnv_profile(cnv_profile, chr_lengths_df, row['chr_lst'])

    logger.info(f"Filled CNV profile for clone {clone} generated successfully.")

    return cnv_profile

#%% Main function

def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic CNV profiles.")
    parser.add_argument("-i", "--input_path", type = str, required = True, 
                        help = "Path to tsv file with specification of CNV profiles to make (# clones, # cells per clone, chromosomes to include, etc.).")
    parser.add_argument("-p", "--profile_name", type = str, required = True,
                        help = "Name of the CNV profile.")
    parser.add_argument("-o", "--output_path", type = str, required = True,
                        help = "Path to output directory.")
    parser.add_argument("-v", "--verbose", action = "store_true",
                        help = "Verbose output.")
    return parser.parse_args()

def main(input_path, output_path, profile_name = None):
    profile_input = parse_cnv_profile_input(input_path)
    logger.info("Parsed CNV profile input successfully.")

    chr_lengths_df = pd.read_csv(f"{DATADIR}/chr_lengths.csv")
    
    if profile_name == None:
        for _, row in profile_input.iterrows():
            new_cnv_profile = make_cnv_profile(row, chr_lengths_df)
            cnv_profile_path = f"{output_path}/{row['test_id']}_cnv_profile.tsv"
            new_cnv_profile.to_csv(cnv_profile_path, sep="\t", index=False)
    else:
        row = profile_input[profile_input['test_id'] == profile_name].iloc[0]
        new_cnv_profile = make_cnv_profile(row, chr_lengths_df)
        cnv_profile_path = f"{output_path}/{profile_name}_cnv_profile.tsv"
        new_cnv_profile.to_csv(cnv_profile_path, sep="\t", index=False)

    logger.info("CNV profiles generated successfully.")

    return

if __name__ == "__main__":
    args = parse_args()
    input_path = args.input_path
    output_path = args.output_path
    verbose = args.verbose
    profile_name = args.profile_name

    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info("Starting CNV profile generation.")
    logger.info(f"Input path: {input_path}")
    logger.info(f"Output path: {output_path}")

    os.makedirs(output_path, exist_ok=True)
    logger.info(f"Output directory created: {output_path}")

    main(input_path, output_path, profile_name)