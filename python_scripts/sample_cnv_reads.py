#%%
import os
import sys
import logging

env_name = os.environ.get("CONDA_DEFAULT_ENV")
print(f"Current conda environment: {env_name}")

print(sys.executable)

import pandas as pd
import numpy as np
import pysam
import argparse
import subprocess
import tempfile
import shutil
from samtools_utils import *

BASEDIR = "/data1/shahs3/users/sunge/cnv_simulator"

#%%

##########################################################################
### General Functions
##########################################################################

def get_group_bam_dict(profile_df):
    """
    Get a dictionary of group names and their corresponding bam file paths.
    """
    group_bam_dict = {}
    unique_groups = profile_df['sample_group'].str.split(",").explode().unique()

    for group in unique_groups:
        # Get the bam file path for the group
        if not group.isdigit():
            bam_path = f"{BASEDIR}/data/normal_cell_bams/{group}.bam"
        else:
            bam_path = f"{BASEDIR}/data/normal_cell_bams/group_{group}_merged.bam"
        if not os.path.exists(bam_path):
            raise ValueError(f"BAM file does not exist: {bam_path}")
        group_bam_dict[group] = bam_path

    return group_bam_dict


def assign_new_barcodes(bam_path, baseline_clone_cell_barcodes, output_bam_path):
    
    bamfile = pysam.AlignmentFile(bam_path, "rb")
    out_bam = pysam.AlignmentFile(output_bam_path, "wb", template=bamfile)

    new_cell_assignments = {}
    for i, read in enumerate(bamfile):
        read_cell = read.get_tag("CB")
        if read_cell not in new_cell_assignments:
            new_cb = np.random.choice(baseline_clone_cell_barcodes)
            new_cell_assignments[read_cell] = new_cb
        else:
            new_cb = new_cell_assignments[read_cell]
        read.set_tag("CB", new_cell_assignments[read_cell])
        out_bam.write(read)
    bamfile.close()
    out_bam.close()
    return


def get_sample_read_counts(profile_row, new_group_cell_dict, baseline_cell_reads_dict,
                           clone_cell_barcodes, group_bam_dict, NCORES = 32):
    """
    Get the number of reads to sample from the baseline bam file and the group bam files for a clone in a region.
    
    Parameters:
    profile_row: Row in the profile dataframe for the region
    new_group_cell_dict: Dictionary of groups and their corresponding cell barcodes (new cells)
    baseline_cell_reads_dict: Dictionary of cell barcodes and their corresponding read IDs in the baseline bam file
    clone_cell_barcodes: List of cell barcodes for the clone
    group_bam_dict: Dictionary of groups and their corresponding bam file paths

    Returns:
    n_clone_baseline_reads: # reads for clone in baseline bam file for this region
    n_clone_additional_reads: # reads to sample from group bam file for clone in this region
    n_clone_additional_reads_per_group: # reads to sample from each group bam file for clone in this region
    n_clone_frac_reads_per_group: # fraction of reads to sample from each group bam file for clone in this region
    """

    logger = logging.getLogger(__name__)
    logger.info(f"Getting sample read counts for profile row {profile_row.chr}:{profile_row.start}-{profile_row.end}")

    logger.debug(f"Profile row: {profile_row}")
    logger.debug(f"chr: {profile_row.chr}, Start: {profile_row.start}, End: {profile_row.end}, Copy number: {profile_row.copy_number}, State: {profile_row.state}, Clone: {profile_row.clone}")

    # Compute proportion of additional cells in each group
    group_cell_proportions = np.array([len(new_group_cell_dict[group]) for group in new_group_cell_dict], dtype = float)
    group_cell_proportions /= group_cell_proportions.sum()
    logger.info(f"Group proportions: {group_cell_proportions}")

    # Figure out how many additional reads to sample based on number of reads in baseline bam file in region for clone
    # Number of reads in baseline clone total in region
    n_clone_baseline_reads = sum(len(reads) for cb, reads in baseline_cell_reads_dict.items() if cb in clone_cell_barcodes)
    n_clone_additional_reads = (n_clone_baseline_reads // 2) * profile_row.state
    logger.info(f"Number of baseline reads for clone in region: {n_clone_baseline_reads}")
    logger.info(f"Number of additional reads to sample for clone in region: {n_clone_additional_reads}")

    # Calculate the number of additional reads to sample from each group based on proportion of cells from each group
    n_clone_additional_reads_per_group_array = np.round(n_clone_additional_reads * group_cell_proportions).astype(int)
    n_clone_additional_reads_per_group = dict(zip(new_group_cell_dict.keys(), n_clone_additional_reads_per_group_array))

    # Get number of reads in each group bam file for this region
    group_max_reads = {}
    for group in new_group_cell_dict:
        total_reads = samtools_get_indexed_read_count(group_bam_dict[group],
                                                       region = f"{profile_row.chr}:{profile_row.start}-{profile_row.end}",
                                                       cell_barcodes = new_group_cell_dict[group],
                                                       NCORES = NCORES)
        logger.info(f"Total reads in group {group}: {total_reads}")
        group_max_reads[group] = total_reads
    n_combined_max_reads = sum(group_max_reads.values())
    logger.info(f"Total reads in all groups: {n_combined_max_reads}")
    if n_combined_max_reads < n_clone_additional_reads:
        raise ValueError(f"Total reads in all groups {n_combined_max_reads} is less than the number of additional reads to sample {n_clone_additional_reads}.")

    # Allocate reads to each group based on the number of cells in each group randomly
    initial_alloc = dict(zip(new_group_cell_dict.keys(),
                             n_clone_additional_reads_per_group_array))
    
    # Check overflow and redistribute to different groups
    capped_alloc = {}
    overflow_reads = 0
    for group, requested in initial_alloc.items():
        max_available = group_max_reads[group]
        if requested <= max_available:
            capped_alloc[group] = requested
        else:
            capped_alloc[group] = max_available
            overflow_reads += requested - max_available

    while overflow_reads > 0:
        # Find groups with extra reads available
        eligible_groups = {g: group_max_reads[g] - capped_alloc[g]
                           for g in capped_alloc
                           if group_max_reads[g] > capped_alloc[g]}
        if not eligible_groups:
            # logger.warning(f"{overflow_reads} reads could not be redistributed to any group.")
            raise ValueError(f"{overflow_reads} reads could not be redistributed to any group.")
        total_extra_capacity = sum(eligible_groups.values())

        # Reassign reads to groups based on their capacity
        reassign = {
            g: min(overflow_reads, int(overflow_reads * (cap / total_extra_capacity)))
            for g, cap in eligible_groups.items()
        }
        for g, n in reassign.items():
            allocatable = min(n, group_max_reads[g] - capped_alloc[g])
            capped_alloc[g] += allocatable
            overflow_reads -= allocatable
            if overflow_reads <= 0:
                break
        logger.debug(f"Reassigned {reassign} reads to groups {capped_alloc}")
        if sum(reassign.values()) == 0:
            logger.warning(f"Could not reassign any reads. Overflow reads: {overflow_reads}")
            break
    
    n_clone_additional_reads_per_group = capped_alloc

    # Check that the total number of additional reads is equal to the number of reads to sample
    if sum(n_clone_additional_reads_per_group.values()) != n_clone_additional_reads:
        logger.info(f"Total number of additional reads does not match: {sum(n_clone_additional_reads_per_group.values())} != {n_clone_additional_reads}")
        # raise ValueError(f"Total number of additional reads does not match: {sum(n_clone_additional_reads_per_group.values())} != {n_clone_additional_reads}")
    
    frac_reads_per_group = {}
    for group, reads in n_clone_additional_reads_per_group.items():
        max_reads = group_max_reads[group]
        if max_reads > 0:
            frac = reads / max_reads
            if np.isnan(frac):
                logger.debug(f"Number of reads for group {group} is {reads}, max reads is {max_reads}. Setting fraction to 0.")
                frac = 0.0
            frac_reads_per_group[group] = frac
        else:
            logger.warning(f"Group {group} has no reads in the region {profile_row.chr}:{profile_row.start}-{profile_row.end}. Setting fraction to 0.")
            frac_reads_per_group[group] = 0.0

    return n_clone_baseline_reads, n_clone_additional_reads, n_clone_additional_reads_per_group, frac_reads_per_group


def remove_reads_from_baseline_bam(baseline_bam_path, profile_row, profile_row_index, clone_row, 
                                   TMP_BAMDIR, NCORES = 32):
    logger = logging.getLogger(__name__)
    logger.info(f"Removing reads from baseline bam file for profile row {profile_row_index}")

    clone_cell_barcodes = clone_row.cell_barcode.split(",")
    
    out_bam_path = f"{TMP_BAMDIR}/row_{profile_row_index}_{profile_row.chr}_{profile_row.start}_{profile_row.end}_tmp.bam"
    
    if profile_row.copy_number == 0:
        # No bam file for this region
        logger.info(f"No reads for this region: {profile_row.chr}:{profile_row.start}-{profile_row.end}")
        logger.info(f"Shouldn't get here anyway?")
        # samtools_sample_reads(baseline_bam_path, out_bam_path, 0.0,
        #                       region = f"{profile_row.chr}:{profile_row.start}-{profile_row.end}",
        #                       cell_barcodes = clone_cell_barcodes,
        #                       NCORES = NCORES)
    elif profile_row.copy_number == 1:
        # Remove half reads from this region in the baseline bam file
        logger.info(f"Removing half reads for this region: {profile_row.chr}:{profile_row.start}-{profile_row.end}")
        samtools_sample_reads(baseline_bam_path, out_bam_path, 0.5,
                              region = f"{profile_row.chr}:{profile_row.start}-{profile_row.end}",
                              cell_barcodes = clone_cell_barcodes,
                              NCORES = NCORES)
    elif profile_row.copy_number == 2:
        # Select all reads from this region in the baseline bam file
        logger.info(f"Selecting all reads for this region: {profile_row.chr}:{profile_row.start}-{profile_row.end}")
        samtools_get_reads(baseline_bam_path, out_bam_path,
                           region = f"{profile_row.chr}:{profile_row.start}-{profile_row.end}",
                           cell_barcodes = clone_cell_barcodes,
                           NCORES = NCORES)
    else:
        ValueError(f"Invalid copy number: {profile_row.copy_number}")

    if os.path.exists(out_bam_path):
        n_reads = samtools_get_indexed_read_count(out_bam_path)
        logger.info(f"Number of reads in temporary bam file: {n_reads}")
    else:
        logger.info(f"Temporary bam file does not exist: {out_bam_path}")
        n_reads = 0

    return out_bam_path, n_reads
    

def add_reads_to_baseline_bam(baseline_bam_path, group_bam_dict, 
                              profile_row, profile_row_index, clone_row,
                              TMP_BAMDIR, NCORES = 32):
    logger = logging.getLogger(__name__)
    logger.info(f"Adding reads to baseline bam file for profile row {profile_row_index}")

    logger.info(profile_row)
    logger.info(profile_row.chr)
    logger.info(profile_row.start)
    logger.info(profile_row.end)

    # Check that copy number is greater than 0
    if profile_row.copy_number <= 0:
        raise ValueError(f"Copy number must be greater than 0: {profile_row.copy_number}")

    # Get the cell barcodes for the clone
    clone_cell_barcodes = clone_row.cell_barcode.split(",")
    baseline_cell_reads_dict, _ = samtools_get_cell_reads(baseline_bam_path,
                                                          region = f"{profile_row.chr}:{profile_row.start}-{profile_row.end}",
                                                          NCORES = NCORES)
    missing_clone_barcodes = [cb for cb in clone_cell_barcodes if cb not in baseline_cell_reads_dict]
    if missing_clone_barcodes:
        logger.info(clone_cell_barcodes)
        logger.info(baseline_cell_reads_dict)
        logger.warning(f"Missing cell barcodes in baseline BAM: {missing_clone_barcodes}")
    # TODO: Selected cells may not have reads in the region
    # TODO: Selected cells may not contain enough reads in the region in comparison to the number of reads in baseline cells

    # Get all necessary groups for this region, assign cell barcodes to groups
    new_groups = profile_row.sample_group.split(",")
    new_cell_barcodes = profile_row.cell_barcode.split(",")
    new_group_cell_dict = {} # group: [cell_barcode in row]
    for group, cell_barcode in zip(new_groups, new_cell_barcodes):
        if group not in group_bam_dict:
            raise ValueError(f"BAM file for group {group} does not exist.")
        new_group_cell_dict.setdefault(group, []).append(cell_barcode)

    # Get the number of reads in the baseline bam file for this region
    n_baseline_reads, n_additional_reads, n_additional_reads_per_group, frac_reads_per_group = get_sample_read_counts(
        profile_row, new_group_cell_dict, baseline_cell_reads_dict, clone_cell_barcodes, group_bam_dict)

    logger.info(f"Region {profile_row.chr}:{profile_row.start}-{profile_row.end} has {n_baseline_reads} reads in baseline bam file")
    logger.info(f"Sampling {n_additional_reads} additional reads")
    logger.info(f"Number of additional reads per group: {n_additional_reads_per_group}")

    # For each group, get the bam file and sample reads from the group bam file
    tmp_group_bam_paths = [] # 1 temporary bam file per group
    for group, frac_reads in frac_reads_per_group.items():
        logger.info(f"Sampling {frac_reads} fraction from group {group}")
        region_str = f"{profile_row.chr}:{profile_row.start}-{profile_row.end}"

        # Sample reads from the group bam file
        sampled_bam_path = f"{TMP_BAMDIR}/row_{profile_row_index}_{profile_row.chr}_{profile_row.start}_{profile_row.end}_{group}_tmp.bam"
        samtools_sample_reads(group_bam_dict[group], sampled_bam_path, frac_reads,
                              region = region_str, cell_barcodes = new_group_cell_dict[group],
                              NCORES = NCORES)
        tmp_group_bam_paths.append(sampled_bam_path)

    logger.info(f"Temporary BAM file created: {sampled_bam_path}")
    
    # Merge the temporary bam files into a single bam file
    merged_bam_path = f"{TMP_BAMDIR}/row_{profile_row_index}_{profile_row.chr}_{profile_row.start}_{profile_row.end}_merged_tmp.bam"
    if len(tmp_group_bam_paths) < 2:
        n_tmp_reads = samtools_get_indexed_read_count(tmp_group_bam_paths[0])
        logger.info(f"# reads in merged bam: {n_tmp_reads}, # additional reads: {n_additional_reads}")
        shutil.move(tmp_group_bam_paths[0], merged_bam_path)
    else:
        samtools_merge(tmp_group_bam_paths, merged_bam_path, NCORES = NCORES)
        logger.info(f"Merged BAM file created: {merged_bam_path}")

    # Check that the number of reads in the merged bam file is equal to the number of additional reads
    n_merged_reads = samtools_get_indexed_read_count(merged_bam_path)
    logger.info(f"Number of reads in merged bam file: {n_merged_reads}")

    # Remove the temporary bam files
    for tmp_bam_path in tmp_group_bam_paths:
        if os.path.exists(tmp_bam_path):
            os.remove(tmp_bam_path)
            logger.info(f"Temporary BAM file deleted: {tmp_bam_path}")

    # Replace the cell barcodes in the merged bam file with the new cell barcodes
    renamed_bam_path = f"{TMP_BAMDIR}/row_{profile_row_index}_{profile_row.chr}_{profile_row.start}_{profile_row.end}_renamed_tmp.bam"
    assign_new_barcodes(merged_bam_path, clone_cell_barcodes, renamed_bam_path)
    
    logger.info(f"Temporary BAM file with renamed cell barcodes created: {renamed_bam_path}")

    # Remove the merged bam file
    os.remove(merged_bam_path)
    logger.info(f"Merged BAM file deleted: {merged_bam_path}")
    
    # Get original reads from baseline bam file
    baseline_clone_path = f"{TMP_BAMDIR}/row_{profile_row_index}_{profile_row.chr}_{profile_row.start}_{profile_row.end}_baseline_tmp.bam"
    samtools_sample_reads(baseline_bam_path, baseline_clone_path, 1.0,
                          region = f"{profile_row.chr}:{profile_row.start}-{profile_row.end}",
                          cell_barcodes = clone_cell_barcodes)
    
    out_bam_path = f"{TMP_BAMDIR}/row_{profile_row_index}_{profile_row.chr}_{profile_row.start}_{profile_row.end}_tmp.bam"
    # Merge the renamed bam file with the baseline bam file
    samtools_merge([baseline_clone_path, renamed_bam_path], out_bam_path, NCORES = NCORES)
    logger.info(f"Final BAM file created: {out_bam_path}")
    # Remove the renamed bam file
    os.remove(renamed_bam_path)
    logger.info(f"Renamed BAM file deleted: {renamed_bam_path}")
    # Remove the baseline bam file
    os.remove(baseline_clone_path)
    logger.info(f"Baseline BAM file deleted: {baseline_clone_path}")

    return out_bam_path, n_merged_reads


def normalize_final_bam(baseline_bam_path, bam_path, out_path, n_final_reads, NCORES = 32):
    logger = logging.getLogger(__name__)
    logger.info(f"Normalizing final bam file: {bam_path}")

    n_baseline_reads = samtools_get_entire_read_count(baseline_bam_path)
    
    sample_frac = n_baseline_reads / n_final_reads
    logger.info(f"Number of reads in baseline bam file: {n_baseline_reads}")
    logger.info(f"Number of reads in final bam file: {n_final_reads}")
    logger.info(f"Sample fraction: {sample_frac}")
    if sample_frac > 1:
        logger.info(f"Sample fraction is greater than 1: {sample_frac}. Keeping unnormalized reads.")
        shutil.copy(bam_path, out_path)
    else:
        samtools_sample_reads(bam_path, out_path, sample_frac, NCORES = NCORES)

    logger.info(f"Final BAM file created: {out_path}")

    return


def sample_cnv_reads(profile_dir, profile_name, group_name, NCORES = 32):
    logger = logging.getLogger(__name__)
    logger.info(f"Sampling CNV reads for profile: {profile_name}, group: {group_name}")

    # Get list of all groups, read in bam files, dictionary of group:bam path
    if group_name == "":
        prefix = f"{profile_name}"
    else:
        prefix = f"{profile_name}_{group_name}"
    profile_path = f"{profile_dir}/{prefix}_cell_profile.tsv"
    profile_df = pd.read_csv(profile_path, sep="\t", quoting = 3)
    profile_df['cell_barcode'] = profile_df['cell_barcode'].str.strip('"')
    profile_df['sample_group'] = profile_df['sample_group'].str.strip('"')
    group_bam_dict = get_group_bam_dict(profile_df)

    # Get list of baseline cells, make bam file containing all reads from baseline cells
    baseline_bam_path = f"{profile_dir}/{profile_name}_baseline_cells.bam"
    if not os.path.exists(baseline_bam_path):
        raise ValueError(f"Baseline BAM file does not exist: {baseline_bam_path}")
    
    TMP_BAMDIR = f"{profile_dir}/{prefix}_intermediate_bams"
    os.makedirs(TMP_BAMDIR, exist_ok=True)

    cnv_profile_df = profile_df[(profile_df['clone'] != -1) & (profile_df['chr'] != 0)]
    
    # For every row in the profile df (except for first row, which is baseline cells)
    all_tmp_bam_paths = []
    total_reads = 0
    for index, row in cnv_profile_df.iterrows():
        # Get the clone row
        cur_row_clone = row['clone']
        clone_row = profile_df.iloc[cur_row_clone + 1]
        cur_copy_number = int(row['copy_number'])

        logger.info(f"Processing row {index} for clone {cur_row_clone} with copy number {cur_copy_number}")

        if cur_copy_number == 0:
            logger.info(f"Skipping row {index} with copy number 0")
            continue
        elif cur_copy_number <= 2:
            tmp_bam_path, readcount = remove_reads_from_baseline_bam(baseline_bam_path, row, index, clone_row, TMP_BAMDIR)
            logger.info(f"Temporary BAM file created: {tmp_bam_path}")
            logger.info(f"Number of reads in temporary BAM file: {readcount}")
        else:
            tmp_bam_path, readcount = add_reads_to_baseline_bam(baseline_bam_path, group_bam_dict, row, index, clone_row, TMP_BAMDIR)
        all_tmp_bam_paths.append(tmp_bam_path)
        total_reads += readcount

    # Merge all the temporary bam files into a single bam file
    merged_bam_path = f"{profile_dir}/{prefix}_unnormalized_cnv.bam"
    if len(all_tmp_bam_paths) < 2:
        n_tmp_reads = samtools_get_indexed_read_count(all_tmp_bam_paths[0])
        logger.info(f"# reads in merged bam: {n_tmp_reads}")
        shutil.move(all_tmp_bam_paths[0], merged_bam_path)
    else:
        samtools_merge(all_tmp_bam_paths, merged_bam_path, NCORES = NCORES)
        logger.info(f"Sum of reads in temporary bam files: {total_reads}")
        logger.info(f"Merged BAM file created: {merged_bam_path}")

    # Remove all files in intermediate bam directory
    logger.info(f"Deleting temporary BAM files in {TMP_BAMDIR}")
    if os.path.exists(TMP_BAMDIR):
        shutil.rmtree(TMP_BAMDIR)
        logger.info(f"Temporary BAM directory deleted: {TMP_BAMDIR}")

    # Sample original number of reads from the modified bam file
    logger.info(f"Sampling {total_reads} reads from merged bam file")
    final_bam_path = f"{profile_dir}/{prefix}_final_cnv.bam"
    normalize_final_bam(baseline_bam_path, merged_bam_path, final_bam_path, 
                        total_reads, NCORES = NCORES)

    # Sort the final bam file
    logger.info(f"Sorting final BAM file: {final_bam_path}")
    sorted_bam_path = f"{profile_dir}/{prefix}_final_sorted_cnv.bam"
    samtools_sort(final_bam_path, sorted_bam_path, NCORES = NCORES)

    # Index the final bam file
    logger.info(f"Indexing final BAM file: {sorted_bam_path}")
    samtools_index(sorted_bam_path, NCORES = NCORES)

    logger.info("DONE!")
    return


#%%

def parse_args():
    parser = argparse.ArgumentParser(description="Sample CNV reads from BAM files.")
    parser.add_argument("--profile_dir", type=str, required=True, help="Path to the profile file.")
    parser.add_argument("--profile_name", type=str, required=True, help="Name of the profile.")
    parser.add_argument("--group_name", type=str, default="", help="Name of the group.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument("-c", "--ncores", type=int, default=32, help="Number of cores to use.")
    return parser.parse_args()


def main():
    args = parse_args()
    profile_dir = args.profile_dir
    profile_name = args.profile_name
    group_name = args.group_name
    VERBOSE = args.verbose
    NCORES = args.ncores

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if VERBOSE else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    sample_cnv_reads(profile_dir, profile_name, group_name, NCORES = NCORES)


if __name__ == "__main__":
    main()
