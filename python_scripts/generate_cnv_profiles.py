#%% Import libraries
import pandas as pd
import numpy as np
import os
import random
import argparse
from collections import Counter

np.random.seed(17321)
random.seed(17321)

### Script for generating CNV profiles for testing

#%%
DATADIR = "/data1/shahs3/users/sunge/cnv_simulator/data"

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
    

def filter_overlapping_cnv(cnv_profile):
    """
    Filter overlapping CNVs from the CNV profile.

    Parameters:
    cnv_profile (pd.DataFrame): DataFrame containing CNV profile.
        clone - Clone number, ranging from 0 to num_clones
        chr - Chromosome number
        start - Start position of CNV
        end - End position of CNV
        copy_number - Copy number of CNV
        state - State of CNV
    """
    filtered_cnv_profile_lst = []

    print(f"Unique clones: {cnv_profile['clone'].unique()}")

    num_merged_cnvs = 0
    for clone in cnv_profile['clone'].unique():
        for chr in cnv_profile[cnv_profile['clone'] == clone]['chr'].unique():
            clone_chr_cnv = cnv_profile[(cnv_profile['clone'] == clone) & (cnv_profile['chr'] == chr)]
            if len(clone_chr_cnv) > 0:
                # Initialize the first CNV as the current CNV
                current_cnv = clone_chr_cnv.iloc[0]
                for _, row in clone_chr_cnv.iloc[1:].iterrows():
                    # Check if the current CNV overlaps with the next CNV
                    if row['start'] <= current_cnv['end']:
                        # Merge the two CNVs
                        merged_cnv = current_cnv.copy()
                        merged_cnv['end'] = max(current_cnv['end'], row['end'])
                        filtered_cnv_profile_lst.append(merged_cnv)
                        num_merged_cnvs += 1
                    else:
                        # Add the current CNV to the filtered profile
                        filtered_cnv_profile_lst.append(current_cnv)
                        # Update the current CNV to the next CNV
                        current_cnv = row

                # Add the last CNV to the filtered profile
                filtered_cnv_profile_lst.append(current_cnv)
    
    filtered_cnv_profile = pd.DataFrame(filtered_cnv_profile_lst)
    filtered_cnv_profile = filtered_cnv_profile.sort_values(by = ['clone', 'chr', 'start']).reset_index(drop = True)
    print(f"Number of merged CNVs: {num_merged_cnvs}")
    return filtered_cnv_profile

def fill_in_cnv_profile(cnv_profile, chr_lengths_df, chr_lst):
    """
    Fill in the CNV profile with missing chromosomes and copy numbers.

    Parameters:
    cnv_profile (pd.DataFrame): DataFrame containing CNV profile.
        clone - Clone number, ranging from 0 to num_clones.
        chr - Chromosome number
        start - Start position of CNV
        end - End position of CNV
        copy_number - Copy number of CNV
        state - State of CNV
    chr_lengths_df (pd.DataFrame): DataFrame containing chromosome lengths.
        chr - Chromosome number
        length - Length of chromosome
    chr_lst (list): List of chromosomes that are included

    Returns:
    filled_cnv_profile (pd.DataFrame): DataFrame containing filled CNV profile.
    """
    # Fill in missing chromosomes and copy numbers
    all_clones = cnv_profile['clone'].unique()
    
    filled_cnv_profile_lst = []
    for clone in all_clones:
        for chr in chr_lst:
            clone_chr_cnv = cnv_profile[(cnv_profile['clone'] == clone) & (cnv_profile['chr'] == chr)]
            if len(clone_chr_cnv) == 0:
                # If there are no CNVs for this chromosome, add a new row with copy number 0
                chr_length = chr_lengths_df[chr_lengths_df['chr'] == chr]['length'].values[0]
                filled_cnv_profile_lst.append({'clone': clone, 'chr': chr, 'start': 0, 'end': chr_length,
                                               'state': 0, 'copy_number': 2, 'size': chr_length})
            else:
                # If there are CNVs for this chromosome, fill in gaps
                current_chr_pos = 0
                for index, row in clone_chr_cnv.iterrows():
                    if current_chr_pos < row['start']:
                        # Add a new row for the gap
                        filled_cnv_profile_lst.append({'clone': clone, 'chr': chr, 'start': current_chr_pos,
                                                       'end': row['start'], 'state': 0, 'copy_number': 2, 
                                                       'size': row['start'] - current_chr_pos})
                    # Add the current CNV
                    filled_cnv_profile_lst.append({'clone': clone, 'chr': chr, 'start': row['start'],
                                                   'end': row['end'], 'state': row['state'], 
                                                   'copy_number': row['copy_number'], 'size': row['size']})
                    current_chr_pos = row['end']
                # Add a new row for the end of the chromosome
                chr_length = chr_lengths_df[chr_lengths_df['chr'] == chr]['length'].values[0]
                if current_chr_pos < chr_length:
                    filled_cnv_profile_lst.append({'clone': clone, 'chr': chr, 'start': current_chr_pos,
                                                   'end': chr_length, 'state': 0, 'copy_number': 2, 
                                                   'size': chr_length - current_chr_pos})
    # Create a DataFrame from the filled CNV profile list
    filled_cnv_profile = pd.DataFrame(filled_cnv_profile_lst)
    # Sort the filled CNV profile by clone, chromosome, and start position
    filled_cnv_profile = filled_cnv_profile.sort_values(by = ['clone', 'chr', 'start']).reset_index(drop = True)

    return filled_cnv_profile
    

def generate_cnv_profile(num_clones, num_cnvs_per_clone, chr_lst, chr_lengths_df,
                         possible_copy_numbers = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                         min_cnv_size = 2000000, max_cnv_size = 10000000):
    """
    Generate a CNV profile for a given number of clones and cells per clone.

    Parameters:
    num_clones (int): Number of clones.
    num_cnvs_per_clone (lst): List of number of CNVs per clone.
    chr_lst (list): List of chromosomes.
    chr_lengths_df (pd.DataFrame): DataFrame containing chromosome lengths.
    min_cnv_size (int): Minimum size of CNVs (default = 2Mb).
    max_cnv_size (int): Maximum size of CNVs (default = 10Mb).

    Returns:
    cnv_profile (pd.DataFrame): DataFrame containing CNV profile.
        clone - Clone number, ranging from 0 to num_clones.
    """

    # Check if cnvs can fit in chromosomes
    total_chr_length = chr_lengths_df[chr_lengths_df['chr'].isin(chr_lst)]['length'].sum()
    for num_cnvs in num_cnvs_per_clone:
        total_max_cnv_size = num_cnvs * max_cnv_size
        if total_max_cnv_size > total_chr_length:
            raise ValueError(f"Total CNV size {total_max_cnv_size} exceeds total chromosome length {total_chr_length}")
    
    # Filter chromosome lengths
    select_chr_lengths_df = chr_lengths_df[chr_lengths_df['chr'].isin(chr_lst)]
    
    cnv_profile = pd.DataFrame(columns=['clone', 'chr', 'start', 'end', 'copy_number', 'state'])
    for clone in range(0, num_clones):
        print(f"Generating CNV profile for clone {clone}")
        print(f"Number of CNVs: {num_cnvs_per_clone[clone]}")
        clone_cnv_count = num_cnvs_per_clone[clone]

        #Account for edge case of when clone_cnv_count = 0
        if clone_cnv_count == 0:
            chr_length = select_chr_lengths_df[select_chr_lengths_df['chr'] == chr_lst[0]]['length'].values[0]
            cnv_profile = pd.concat([cnv_profile, 
                                     pd.DataFrame([{'clone': clone, 'chr': chr_lst[0], 'start': 0, 'end': chr_length,
                                                   'state': 0, 'copy_number': 2, 'size': chr_length}])],
                                                   ignore_index=True)

        # Assign number of cnvs to each chromosome
        num_cnvs_per_chr = Counter(np.random.choice(chr_lst, size = clone_cnv_count,
                                                    p = select_chr_lengths_df['length'] / total_chr_length))
        num_cnvs_per_chr = dict(sorted(num_cnvs_per_chr.items(), key=lambda x: str(x[0])))

        # Generate CNV profile
        for chr in num_cnvs_per_chr.keys():
            for cnv in range(num_cnvs_per_chr[chr]):
                # Generate random start and end positions for CNV
                chr_length = select_chr_lengths_df[select_chr_lengths_df['chr'] == chr]['length'].values[0]
                cnv_start = random.randint(0, chr_length - max_cnv_size)
                cnv_end = cnv_start + random.randint(min_cnv_size, max_cnv_size)
                cnv_size = int(cnv_end - cnv_start)

                # Generate random copy number and state
                cnv_copy_number = random.choice(possible_copy_numbers)
                cnv_state = copy_state_map(copy = cnv_copy_number)

                cnv_profile = pd.concat([cnv_profile, pd.DataFrame([{'clone': clone, 'chr': chr, 'start': cnv_start, 'end': cnv_end, 'size': cnv_size,
                                                                    'state': cnv_state, 'copy_number': cnv_copy_number}])],
                                        ignore_index=True)
    
    # Sort cnv_profile by clone, chr, and start position
    cnv_profile = cnv_profile.sort_values(by = ['clone', 'chr', 'start']).reset_index(drop = True)
    # Filter overlapping CNVs
    cnv_profile = filter_overlapping_cnv(cnv_profile)
    # Fill in missing chromosomes and copy numbers
    cnv_profile = fill_in_cnv_profile(cnv_profile, chr_lengths_df, chr_lst)

    return cnv_profile


def get_required_copy_cell_count(cnv_profile, cells_per_clone_lst, verbose = True):
    """ 
    Get the total number of added copies and the total number of cells required
    for the CNV profile.
    """
    num_clones = len(cells_per_clone_lst)

    max_added_copies = cnv_profile[cnv_profile["clone"] != "-1"]["state"].max()
    # for clone in range(num_clones):
    #     clone_added_copies = cnv_profile[(cnv_profile["clone"] == clone) & \
    #                                      (cnv_profile["state"] > 0)]["state"].max()
    #     total_added_copies = max(total_added_copies, clone_added_copies * cells_per_clone_lst[clone])

    total_required_cells = (np.max(cells_per_clone_lst) * max_added_copies) + np.sum(cells_per_clone_lst)

    if verbose:
        print(f"Total added copies: {max_added_copies}")
        print(f"Total required cells: {total_required_cells}")

    return max_added_copies, total_required_cells


def select_groups(all_normal_cells, num_clones, cnv_profile, cells_per_clone_lst):
    """
    Select groups of cells from all_normal_cells such that the total number of cells
    in the selected groups exceeds the total number of cells required for the CNV profile.

    Parameters:
    all_normal_cells (pd.DataFrame): DataFrame containing all normal cells.
        sample_group - Group number of the cell
        cell_id - Cell ID
        cell_barcode - Cell barcode
        sample_name - Sample name
        sample_group - Sample group
    cnv_profile (pd.DataFrame): DataFrame containing CNV profile.
        clone - Clone number, ranging from 1 to num_clones. 0 is used for baseline cells
        chr - Chromosome number
        start - Start position of CNV
        end - End position of CNV
        copy_number - Copy number of CNV
        state - State of CNV
    cells_per_clone_lst (list): List of number of cells per clone.
        clone - Clone number, ranging from 1 to num_clones. 0 is used for baseline cells

    Returns:
    selected_groups (list): List of selected groups.
        sample_group - Group number of the cell
    """

    group_cell_counts = all_normal_cells["sample_group"].value_counts().sort_index().reset_index()

    # Compute total number of additional cells required
    _, total_required_cells = get_required_copy_cell_count(cnv_profile, cells_per_clone_lst)

    # Check if there are enough cells in the groups
    total_available_cells = np.sum(group_cell_counts["count"])
    if total_available_cells < total_required_cells:
        raise ValueError(f"Not enough cells in the groups to assign {total_required_cells} cells. "
                         f"Only {total_available_cells} cells available.")

    # Select random groups such that their combined count exceeds total_required_cells
    selected_groups = []
    selected_count = 0
    while selected_count <= total_required_cells:
        group = group_cell_counts["sample_group"].sample(n = 1).values[0]
        if group not in selected_groups:
            selected_groups.append(int(group))
            selected_count += group_cell_counts[group_cell_counts["sample_group"] == group]["count"].values[0]
        if len(selected_groups) >= len(group_cell_counts):
            break

    print(f"Selected groups: {selected_groups}")
    print(f"Total cells in selected groups: {selected_count}")
    return selected_groups


def assign_cell_clones(all_normal_cells, num_clones, cnv_profile, cells_per_clone_lst, group_name = None,
                       verbose = False):
    """
    List cells to use for each CNV region in each clone

    Parameters:
    all_normal_cells (pd.DataFrame): DataFrame containing all normal cells.
        sample_group - Group number of the cell
        cell_id - Cell ID
        cell_barcode - Cell barcode
        sample_name - Sample name
        sample_group - Sample group
    num_clones (int): Number of clones.
    cnv_profile (pd.DataFrame): DataFrame containing CNV profile.
        clone - Clone number, ranging from 1 to num_clones. 0 is used for baseline cells
        chr - Chromosome number
        start - Start position of CNV
        end - End position of CNV
        copy_number - Copy number of CNV
        state - State of CNV
    cells_per_clone_lst (list): List of number of cells per clone.

    Returns:
    cell_cnv_profile (pd.DataFrame): DataFrame containing CNV profile with cell barcodes.
        clone - Clone number, ranging from 1 to num_clones. 0 is used for baseline cells
        chr - Chromosome number
        start - Start position of CNV
        end - End position of CNV
        copy_number - Copy number of CNV
        state - State of CNV
        cell_barcode - Cell barcodes for each CNV region, joined by ","
    """

    if group_name == None:
        picked_groups = select_groups(all_normal_cells, num_clones, cnv_profile, cells_per_clone_lst)
        group_cells = all_normal_cells[all_normal_cells["sample_group"].isin(picked_groups)]
    else:
        group_cells = pd.read_csv(f"{DATADIR}/{group_name}.csv")
        group_cells = group_cells.assign(sample_group = group_name)
        _, total_required_cells = get_required_copy_cell_count(cnv_profile, cells_per_clone_lst, verbose = True)
        if len(group_cells) < total_required_cells:
            raise ValueError(f"Not enough cells in the groups to assign {total_required_cells} cells. "
                             f"Only {len(group_cells)} cells available.")

    # Add baseline cells (copy number 0)
    baseline_cell_count = np.sum(cells_per_clone_lst)
    baseline_cells = group_cells.sample(n = baseline_cell_count)

    # Assign clones to cells
    # Group cnv_profile by clone and loop through every clone
    used_cells = set(baseline_cells["cell_id"].values)
    clone_cell_barcodes_lst = []
    clone_cell_groups_lst = []
    for index, row in cnv_profile.iterrows():
        # if copy number is 0, -1, or -2, only baseline cells
        if row['state'] <= 0:
            clone_cell_barcodes = list(baseline_cells["cell_barcode"])
            clone_cell_groups = list(baseline_cells["sample_group"])
        else:
            num_new_cells = int((row['state']) * cells_per_clone_lst[int(row['clone'])])
            # Check if there are enough cells in the group
            if len(group_cells[~group_cells["cell_id"].isin(used_cells)]) < num_new_cells:
                raise ValueError(f"Not enough cells in group {row['clone']} to assign clone cells")
            new_clone_cells = group_cells[~group_cells["cell_id"].isin(used_cells)].sample(n = num_new_cells)

            clone_cells = new_clone_cells.copy()
            clone_cells = clone_cells.sort_values(by = ['sample_group', 'cell_barcode'])

            clone_cell_barcodes = list(clone_cells["cell_barcode"])
            clone_cell_groups = list(clone_cells["sample_group"])
        clone_cell_barcodes_lst.append(",".join(clone_cell_barcodes))
        clone_cell_groups_lst.append(",".join(map(str, clone_cell_groups)))
    
    cell_cnv_profile = cnv_profile.copy()
    cell_cnv_profile = cell_cnv_profile.assign(cell_barcode = clone_cell_barcodes_lst,
                                               cell_group = clone_cell_groups_lst)
    # Add a new row for baseline cells with clone -1
    baseline_row = {
        'clone': -1, 'chr': 0, 'start': 0, 'end': 0,
        'state': 0, 'copy_number': 0, 'size': 0,
        'cell_barcode': ",".join(baseline_cells["cell_barcode"]),
        'cell_group': ",".join(map(str, baseline_cells["sample_group"]))
    }
    baseline_clone_cells_df = pd.DataFrame([baseline_row])

    # Assign baseline cells to a clone
    shuffled_baseline_cells = baseline_cells.sample(frac=1).reset_index(drop=True)
    current_index = 0
    for clone in range(num_clones):
        clone_baseline_cells = shuffled_baseline_cells.iloc[current_index:current_index + cells_per_clone_lst[clone], ]
        clone_row = {
            'clone': clone, 'chr': 0, 'start': 0, 'end': 0,
            'state': 0, 'copy_number': 0, 'size': 0,
            'cell_barcode': ",".join(clone_baseline_cells["cell_barcode"]),
            'cell_group': ",".join(map(str, clone_baseline_cells["sample_group"]))
        }
        baseline_clone_cells_df = pd.concat([baseline_clone_cells_df, pd.DataFrame([clone_row])], ignore_index=True)
        current_index += cells_per_clone_lst[clone]
    if current_index != len(shuffled_baseline_cells):
        raise ValueError(f"Not all baseline cells were assigned to clones. {current_index} cells assigned, {len(shuffled_baseline_cells)} expected.")

    cell_cnv_profile = pd.concat([baseline_clone_cells_df, cell_cnv_profile], ignore_index=True)

    return cell_cnv_profile

def make_single_cnv_profiles(profile_input_row, all_normal_cells, chr_lengths_df, group_name = None):
    """
    Generate CNV profiles for any line of input. 
    Randomly assigns breakpoints and copy numbers to each clone, then assigns cells to use.
    """
    # Extract parameters from the input row
    test_id = profile_input_row["test_id"]
    num_clones = profile_input_row["num_clones"]
    cells_per_clone_lst = profile_input_row["cells_per_clone_lst"]
    num_cnvs_per_clone = profile_input_row["num_cnvs_per_clone"]
    chr_lst = profile_input_row["chr_lst"]
    min_cnv_size = profile_input_row["min_cnv_size"]
    max_cnv_size = profile_input_row["max_cnv_size"]
    possible_copy_numbers = profile_input_row["possible_copy_numbers"]

    print(f"Generating new CNV profile for {test_id} with {num_clones} clones and {cells_per_clone_lst} cells per clone")

    # Generate CNV profile
    cnv_profile = generate_cnv_profile(num_clones, num_cnvs_per_clone, chr_lst, chr_lengths_df,
                                       possible_copy_numbers = possible_copy_numbers,
                                       min_cnv_size = min_cnv_size, max_cnv_size = max_cnv_size)

    # Assign cells to clones
    cell_clones_df = assign_cell_clones(all_normal_cells, num_clones, cnv_profile, cells_per_clone_lst, group_name = group_name, verbose = True)

    return cell_clones_df

#%%
def main(profile_input, cnv_profile_name = None, group_name = None):
    """
    Main function to generate CNV profiles for testing.
    Parameters:
    profile_input (pd.DataFrame): DataFrame containing CNV profile input.
        test_id - Test ID for the CNV profile
        num_clones - Number of clones
        cells_per_clone_lst - List of number of cells per clone
        num_cnvs_per_clone - List of number of CNVs per clone
        chr_lst - List of chromosomes
        min_cnv_size - Minimum size of CNVs
        max_cnv_size - Maximum size of CNVs
        possible_copy_numbers - List of possible copy numbers for CNVs
    cnv_profile_name (str): Name of the CNV profile to generate. If None, generate profiles for all input rows.
    group_name (str): Name of the group to select cells from. If None, select random groups.
    """

    chr_lengths_df = pd.read_csv(f"{DATADIR}/chr_lengths.csv")
    chr_lengths_df['chr'] = chr_lengths_df['chr'].astype(str)

    all_normal_cells = pd.read_csv(f"{DATADIR}/all_normal_cells.csv")

    if cnv_profile_name == None:
        # Make profiles for all input rows
        for _, row in profile_input.iterrows():
            cell_cnv_profile = make_single_cnv_profiles(row, all_normal_cells, chr_lengths_df)
            cnv_profile_path = f"{DATADIR}/small_cnv_profiles/{row['test_id']}_cnv_profile.tsv"
            cell_cnv_profile.to_csv(cnv_profile_path, sep = "\t", index = False)
    else:
        # Select the CNV profile from the input
        profile_input_row = profile_input[profile_input["test_id"] == cnv_profile_name]
        if len(profile_input_row) == 0:
            raise ValueError(f"CNV profile {cnv_profile_name} not found in the input file.")
        
        print(profile_input_row["possible_copy_numbers"])

        # Load the CNV profile from the specified path
        cnv_profile_path = f"{DATADIR}/small_cnv_profiles/{cnv_profile_name}_cnv_profile.tsv"
        # If CNV profile doens't exist, generate based on input row
        if not os.path.exists(cnv_profile_path):
            cell_cnv_profile = make_single_cnv_profiles(profile_input_row, all_normal_cells, chr_lengths_df, group_name = group_name)
            cnv_profile_path = f"{DATADIR}/small_cnv_profiles/{cnv_profile_name}_{group_name}_cnv_profile.tsv"
            cell_cnv_profile.to_csv(cnv_profile_path, sep = "\t", index = False)
        # Otherwise, load CNV profile and assign cells
        else:
            cnv_profile = pd.read_csv(cnv_profile_path, sep = "\t")
            # Remove the first row (baseline cells) and existing cell barcodes
            cnv_profile = cnv_profile[cnv_profile["clone"] != -1]
            cnv_profile = cnv_profile.drop(columns=["cell_barcode", "cell_group"])

            # Get the parameters from the profile input
            cells_per_clone_lst = profile_input[profile_input["test_id"] == cnv_profile_name]["cells_per_clone_lst"].values[0]
            num_clones = len(cells_per_clone_lst)
            
            # Assign cells to clones
            cell_clones_df = assign_cell_clones(all_normal_cells, num_clones, cnv_profile, cells_per_clone_lst, group_name = group_name)
            print(cell_clones_df)

            # Save the CNV profile with cell barcodes
            new_cnv_profile_path = f"{DATADIR}/small_cnv_profiles/{cnv_profile_name}_{group_name}_cnv_profile.tsv"
            cell_clones_df.to_csv(new_cnv_profile_path, sep = "\t", index = False)

    return

def parse_cnv_profile_input(input_path):
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
            
        profile_input["num_clones"] = profile_input["num_clones"].astype(int)
        profile_input["cells_per_clone_lst"] = profile_input["cells_per_clone_lst"].astype(str).str.split(",")
        profile_input["cells_per_clone_lst"] = profile_input["cells_per_clone_lst"].apply(lambda x: [int(i) for i in x])
        profile_input["num_cnvs_per_clone"] = profile_input["num_cnvs_per_clone"].astype(str).str.split(",")
        profile_input["num_cnvs_per_clone"] = profile_input["num_cnvs_per_clone"].apply(lambda x: [int(i) for i in x])
        profile_input["chr_lst"] = profile_input["chr_lst"].astype(str).str.split(",")
        profile_input["min_cnv_size"] = profile_input["min_cnv_size"].astype(int)
        profile_input["max_cnv_size"] = profile_input["max_cnv_size"].astype(int)
        profile_input["possible_copy_numbers"] = profile_input["possible_copy_numbers"].astype(str).str.split(",")
        profile_input["possible_copy_numbers"] = profile_input["possible_copy_numbers"].apply(lambda x: [int(i) for i in x])

        # If chromosome list only contains 0, then include all chromosomes
        all_chr_lst = [str(i) for i in range(1, 23)] + ["X", "Y"]
        profile_input["chr_lst"] = profile_input["chr_lst"].apply(
            lambda x: all_chr_lst.copy() if x == ["0"] else x
        )
    except Exception as e:
        raise ValueError(f"Error reading or validating input file: {e}")    
    
    return profile_input

def parse_args():
    parser = argparse.ArgumentParser(description="Generate CNV profiles for testing.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input file.")
    parser.add_argument("--cnv_profile_name", type=str, default=None, help="Name of the CNV profile to generate.")
    parser.add_argument("--group_name", type=str, default=None, help="Name of the group to select cells from.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    input_path = args.input_path
    cnv_profile_name = args.cnv_profile_name
    group_name = args.group_name

    profile_input = parse_cnv_profile_input(input_path)

    print(f"Generating CNV profiles for {len(profile_input)} input rows.")
    print(f"CNV profile name: {cnv_profile_name}")
    print(f"Group name: {group_name}")

    main(profile_input, cnv_profile_name = cnv_profile_name, group_name = group_name)
