#%% Import libraries
import pandas as pd
import numpy as np
import os
import random
import argparse
from collections import Counter
from samtools_utils import *

np.random.seed(17321)
random.seed(17321)

BASEDIR = f"/data1/shahs3/users/sunge/cnv_simulator"
DATADIR = f"{BASEDIR}/data"

#%% Functions

def select_groups(all_normal_cells, cells_per_clone, profile):
    """
    Select groups of cells based on the number of cells required for each clone.
    """
    logger = logging.getLogger(__name__)

    group_cell_counts = all_normal_cells['sample_group'].value_counts().sort_index().reset_index()

    # Select groups based on number of cells required, including baseline and gains
    total_required_cells = 0
    for clone in range(len(cells_per_clone)):
        clone_profile = profile[profile['clone'] == clone]
        clone_added_copies = clone_profile['state'].sum()
        clone_required_cells = cells_per_clone[clone] * clone_added_copies
        if clone_required_cells > 0:
            total_required_cells += clone_required_cells
            logger.debug(f"Clone {clone}: {clone_added_copies} added copies, {clone_required_cells} required cells")

    total_available_cells = group_cell_counts['count'].sum()
    if total_available_cells < total_required_cells:
        raise ValueError(f"Not enough cells available. Required: {total_required_cells}, Available: {total_available_cells}")
    
    # Select groups such that their total combined cell count > total required cells
    selected_groups = []
    selected_count = 0
    while selected_count <= total_required_cells:
        group = group_cell_counts['sample_group'].sample(n = 1).values[0]
        if group not in selected_groups:
            selected_groups.append(group)
            selected_count += group_cell_counts[group_cell_counts['sample_group'] == group]['count'].values[0]
        if len(selected_groups) == len(group_cell_counts):
            raise ValueError(f"Not enough groups to select from. Required: {total_required_cells}, Available: {total_available_cells}")
    
    logger.debug(f"Selected groups: {selected_groups}")
    logger.debug(f"Total selected cells: {selected_count}")

    return selected_groups

def get_region_read_count(bam_dir, region, unique_groups, cells_df, NCORES = 32):
    group_to_cells = cells_df.groupby("sample_group")['cell_barcode'].apply(list).to_dict()
    n_reads = 0
    for group in unique_groups:
        bam_path = f"{bam_dir}/group_{group}_merged.bam"
        if group in group_to_cells:
            n_reads += samtools_get_indexed_read_count(bam_path, region, 
                                                       cell_barcodes = group_to_cells[group],
                                                       NCORES = NCORES)
    return n_reads


def sample_additional_reads(n_current_reads, n_current_cells, n_required_reads,
                            pool_cells, region, pool_groups, bam_dir, NCORES = 32):
    while_loop_counter = 0
    while n_current_reads < n_required_reads:
        n_required_cells = int(n_required_reads / (n_current_reads / n_current_cells))

        current_new_cells = pool_cells.sample(n = n_required_cells)
        n_current_reads = get_region_read_count(bam_dir, region, pool_groups, 
                                                current_new_cells, NCORES = NCORES)
        while_loop_counter += 1
        logger.debug(f"Current new cells reads: {n_current_reads}, Current new cells: {n_required_cells}")
        logger.debug(f"While loop counter: {while_loop_counter}")
        if while_loop_counter > 10:
            raise ValueError(f"Not enough reads in region {region}. {n_current_reads} reads available, {n_required_reads} reads required.")
    return current_new_cells


def assign_cells_to_profile(profile, profile_params, all_normal_cells, group_name, bam_dir,
                            allow_dup = False, NCORES = 32):
    logger = logging.getLogger(__name__)
    
    cells_per_clone = list(map(int, profile_params['cells_per_clone_lst'].split(",")))
    if group_name == "all":
        group_cells = all_normal_cells
    elif group_name == None:
        picked_groups = select_groups(all_normal_cells, cells_per_clone, profile)
        group_cells = all_normal_cells[all_normal_cells['sample_group'].isin(picked_groups)]
    else:
        group_cells = pd.read_csv(f"{DATADIR}/{group_name}.csv")
        group_cells = group_cells.assign(sample_group = group_name)
    unique_groups = group_cells['sample_group'].unique() # All groups of selected cells
    logger.debug(f"Unique groups selected: {unique_groups}")

    cell_profile_rows = []

    # Pick baseline cells
    baseline_cell_count = np.sum(cells_per_clone)
    baseline_cells = group_cells.sample(n = baseline_cell_count)
    logger.debug(f"Number of baseline cells: {baseline_cell_count}")
    baseline_cells_set = set(baseline_cells['cell_barcode'].values)
    if allow_dup == False:
        unused_cells = group_cells[~group_cells['cell_barcode'].isin(baseline_cells_set)]
        logger.debug(f"Number of unused cells: {len(unused_cells)}")
    else:
        unused_cells = group_cells
    cell_profile_rows.append(baseline_cells)

    # Assign baseline cells to clones
    shuffled_baseline_cells = baseline_cells.sample(frac = 1).reset_index(drop = True)
    current_index = 0
    for clone in range(len(cells_per_clone)):
        clone_baseline_cells = shuffled_baseline_cells.iloc[current_index : current_index + cells_per_clone[clone]]
        cell_profile_rows.append(clone_baseline_cells)
        current_index += cells_per_clone[clone]
    if current_index != baseline_cell_count:
        raise ValueError(f"Not all baseline cells were assigned to clones. {current_index} cells assigned, {len(baseline_cells)} cells expected.")
    logger.debug(f"Assigned baseline cells to clones: {cells_per_clone}")

    # Assign additional cells for CNVs
    for index, row in profile.iterrows():
        clone_baseline_cells = cell_profile_rows[int(row['clone'] + 1)]
        if row['state'] <= 0:
            cell_profile_rows.append(clone_baseline_cells)
        else:
            region = f"{int(row['chr'])}:{int(row['start'])}-{int(row['end'])}"
            n_available_reads = get_region_read_count(bam_dir, region, unique_groups, unused_cells, NCORES = NCORES)
            n_clone_reads = get_region_read_count(bam_dir, region, unique_groups, clone_baseline_cells, NCORES = NCORES)
            n_required_reads = (n_clone_reads / 2) * row['state']
            logger.debug(f"Region: {region}, Available reads: {n_available_reads}, Required reads: {n_required_reads}")

            if n_available_reads < n_required_reads:
                logger.info(f"Not enough reads in region {region} for clone {row['clone']} in selecting groups. {n_available_reads} reads available, {n_required_reads} reads required. Trying all groups.")
                n_available_all_reads = get_region_read_count(bam_dir, region, all_normal_cells['sample_group'].unique(), 
                                                              group_cells, NCORES = NCORES)
                if n_available_all_reads < n_required_reads:
                    raise ValueError(f"Not enough reads in region {region} for clone {row['clone']} in all groups. {n_available_all_reads} reads available, {n_required_reads} reads required.")
                else:
                    raise ValueError(f"Not enough reads in region {region} for clone {row['clone']} in selected groups. Try running on group_name = all.")
            else:
                n_new_cells = int(len(clone_baseline_cells) * row['state'])
                new_cells = unused_cells.sample(n = n_new_cells)
                n_new_cells_reads = get_region_read_count(bam_dir, region, unique_groups, new_cells, NCORES = NCORES)
                logger.debug(f"New cells reads: {n_new_cells_reads}, New cells: {n_new_cells}")

                if n_new_cells_reads >= n_required_reads:
                    cell_profile_rows.append(new_cells)
                    if allow_dup == False:
                        unused_cells = unused_cells[~unused_cells['cell_barcode'].isin(new_cells['cell_barcode'])]
                    logger.debug(f"Sufficient initial cells, assigned {n_new_cells} new cells for clone {row['clone']}")
                else:
                    logger.debug(f"Not enough initial cells, need to sample more cells for clone {row['clone']}")
                    current_new_cells = sample_additional_reads(n_new_cells_reads, n_new_cells, n_required_reads,
                                                                unused_cells, region, unique_groups, bam_dir, NCORES = NCORES)
                    cell_profile_rows.append(current_new_cells)
                    logger.debug(f"Assigned {len(current_new_cells)} new cells for clone {row['clone']}")
                    if allow_dup == False:
                        unused_cells = unused_cells[~unused_cells['cell_barcode'].isin(current_new_cells['cell_barcode'])]
    profile_cell_lst = []
    profile_group_lst = []
    for i in range(len(cell_profile_rows)):
        barcode_str = '"' + ",".join(cell_profile_rows[i]['cell_barcode'].values) + '"'
        profile_cell_lst.append(barcode_str)
        sample_group_str = '"' + ",".join(cell_profile_rows[i]['sample_group'].values.astype(str)) + '"'
        profile_group_lst.append(sample_group_str)
    
    logger.debug(f"Cell list length: {len(profile_cell_lst)}, length of profile: {len(profile)}")
    baseline_clone_lst = [-1] + [i for i in range(len(cells_per_clone))]
    baseline_cell_assign_profile = pd.DataFrame({"clone": baseline_clone_lst,
                                                 "chr": [0] * len(baseline_clone_lst), 
                                                 "start": [0] * len(baseline_clone_lst),
                                                 "end": [0] * len(baseline_clone_lst),
                                                 "copy_number": [2] * len(baseline_clone_lst),
                                                 "state": [0] * len(baseline_clone_lst),
                                                 "size": [0] * len(baseline_clone_lst)})
    cell_assign_profile = pd.concat([baseline_cell_assign_profile, profile.copy()], ignore_index = True)
    cell_assign_profile = cell_assign_profile.assign(cell_barcode = profile_cell_lst,
                                                     sample_group = profile_group_lst)
    logger.info(f"Assigned cells for profile {profile_params['test_id']}")
    return cell_assign_profile


#%% Main function

def parse_args():
    parser = argparse.ArgumentParser(description="Assign cells for gains in CNV profile")
    parser.add_argument("--profile_name", type=str, required = True, help="Path to CNV profile file")
    parser.add_argument("--profile_params_path", type = str, required = True, help="Path to file with CNV profile parameters")
    parser.add_argument("--group_name", type = str, default = None, help="Name of the group to pick cells from")
    parser.add_argument("--bam_dir", type = str, required = None, help="Directory where BAM files are stored")
    parser.add_argument("--allow_dup", type = str, default = False, help="Whether to strictly enforce no duplicate cell barcodes")
    parser.add_argument("--input_dir", type = str, required = True, help="Directory where input CNV profiles are stored")
    parser.add_argument("--output_dir", type = str, required = True, help="Directory to save output files to")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--ncores", type = int, default = 32, help="Number of cores to use for processing")

    return parser.parse_args()

def main(profile_name, profile_params_path, group_name, bam_dir, allow_dup,
         input_dir, output_dir, NCORES = 32):
    logger = logging.getLogger(__name__)
    if not os.path.exists(input_dir):
        raise ValueError(f"Input directory {input_dir} does not exist.")
    
    # Load the CNV profile parameters
    logger.debug(f"Loading profile parameters from {profile_params_path}")
    profile_params = pd.read_csv(profile_params_path, sep = "\t")
    if profile_name not in profile_params['test_id'].values:
        raise ValueError(f"Profile name {profile_name} not found in profile parameters file.")
    profile_params = profile_params[profile_params['test_id'] == profile_name].iloc[0]

    # Load the CNV profile
    logger.debug(f"Loading CNV profile from {input_dir}/{profile_name}_cnv_profile.tsv")
    profile_path = f"{input_dir}/{profile_name}_cnv_profile.tsv"
    if not os.path.exists(profile_path):
        raise ValueError(f"Profile file {profile_path} does not exist.")
    profile = pd.read_csv(profile_path, sep = "\t")

    # Load the cell barcodes
    logger.debug(f"Loading cell barcodes from {DATADIR}/all_normal_cells.csv")
    all_normal_cells = pd.read_csv(f"{DATADIR}/all_normal_cells.csv")

    logger.info(f"Assigning cells to {profile_name} profile")
    cell_assign_profile = assign_cells_to_profile(profile, profile_params, all_normal_cells, group_name, bam_dir, 
                                                  allow_dup = allow_dup, NCORES = NCORES)
    
    # Save the assigned cells to a file
    output_path = f"{output_dir}/{profile_name}_cell_profile.tsv"
    cell_assign_profile.to_csv(output_path, sep = "\t", index = False, quoting = 3)
    logger.info(f"Assigned cells saved to {output_path}")

    return

if __name__ == "__main__":
    args = parse_args()
    profile_name = args.profile_name
    profile_params_path = args.profile_params_path
    group_name = args.group_name
    bam_dir = args.bam_dir
    allow_dup = args.allow_dup
    input_dir = args.input_dir
    output_dir = args.output_dir
    verbose = args.verbose
    NCORES = args.ncores

    # Check if the output directory exists, if not create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    logger.info(f"Running with the following parameters:")
    logger.info(f"Profile name: {profile_name}")
    logger.info(f"Profile parameters path: {profile_params_path}")
    logger.info(f"Group name: {group_name}")
    logger.info(f"BAM directory: {bam_dir}")
    logger.info(f"Allow duplicate cell barcodes: {allow_dup}")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Number of cores: {NCORES}")
    logger.info(f"Verbose logging: {verbose}")
    
    logger.info(f"Assigning cells to {profile_name} profile")

    main(profile_name, profile_params_path, group_name, bam_dir, allow_dup, 
         input_dir, output_dir, NCORES = NCORES)