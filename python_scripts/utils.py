import pandas as pd
import numpy as np
import os
import sys
import pysam
import isabl_cli as ii
import subprocess
import logging
import tempfile

##################################################################
# ISABEL UTILS
##################################################################

def get_data_paths(aliquot_id_lst):
    """
    Get bam and gc metrics paths for a given aliquot_id.
    """
    data_path_dict = {"aliquot_id": [], "bam_path": [], "gc_metrics_path": []}
    for aliquot_id in aliquot_id_lst:
        # Get the bam path from the filtered cell table
        search_results = ii.get_analyses(
            application__name__in=['MONDRIAN-ALIGNMENT'],
            targets__aliquot__identifier=aliquot_id,
            status='SUCCEEDED',
            limit=100,
        )

        if (search_results == False) or (len(search_results) == 0):
            print(f"Search for {aliquot_id} returned {len(search_results)} results. Please check the analysis.")
            continue
        else:
            bam_path = search_results[0].results.bam
            gc_metrics_path = search_results[0].results.gc_metrics
        
        # Check if the bam path is valid
        if not os.path.exists(bam_path):
            raise ValueError(f"BAM file does not exist: {bam_path}")
        
        # Append the bam path to the list
        data_path_dict["aliquot_id"].append(aliquot_id)
        data_path_dict["bam_path"].append(bam_path)
        data_path_dict["gc_metrics_path"].append(gc_metrics_path)
    
    return data_path_dict

##################################################################
# SAMTOOLS UTILS
##################################################################

def samtools_read_id_filter(bam_path, out_path, read_ids, logger, NCORES = 16, VERBOSE = False):
    """
    Saves new bam file with only the specified read IDs.
    """
    with tempfile.NamedTemporaryFile(mode = "w", delete = False) as tmp:
        for rid in read_ids:
            tmp.write(f"{rid}\n")
        tmp_path = tmp.name
        if VERBOSE:
            logger.debug(f"Temporary file created: {tmp_path}")

    # Print number of read IDs to be filtered
    if VERBOSE:
        logger.debug(f"Number of read IDs to filter: {len(read_ids)}")
        logger.debug(f"Read IDs: {read_ids[:10]}...")

    try:
        cmd = [
            "samtools", "view", "-@", str(NCORES),
            "-b", "-N", tmp_path, bam_path,
            "-o", out_path
        ]
        if VERBOSE:
            logger.debug(f"Executing command: {' '.join(cmd)}")
        subprocess.run(cmd, check = True)
    finally:
        os.remove(tmp_path)
        if VERBOSE:
            logger.debug(f"Temporary file deleted: {tmp_path}")
    
    return out_path


def samtools_get_cell_reads(bam_path, logger, chr = None, start = None, end = None,
                            NCORES = 16, VERBOSE = False):
    """
    Get the reads in a specific region sorted by cell barcode.

    Parameters:
    ----------
    bam_path (str): Path to source BAM file.
    chr (str): Chromosome name
    start (int): Start position of the region.
    end (int): End position of the region.

    Returns:
    -------
    cell_reads_dict: Dictionary with cell barcodes as keys and lists of read IDs as values.
    """
    cmd = [
        "samtools", "view", "-@", str(NCORES),
        bam_path
    ]
    if chr and start and end:
        cmd.append(f"{chr}:{start}-{end}")
    try:
        if VERBOSE:
            logger.debug(f"Executing command: {' '.join(cmd)}")
        region_reads_result = subprocess.run(cmd, stdout=subprocess.PIPE, check = True, text = True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing command: {' '.join(cmd)}")
        logger.error(f"Error message: {e.stderr}")
        raise

    cell_reads_dict = {}
    reads_cell_dict = {}
    for line in region_reads_result.stdout.splitlines():
        fields = line.strip().split("\t")
        read_id = fields[0]
        cb_tag = next((f for f in fields[11:] if f.startswith("CB:Z:")), None)
        if cb_tag:
            cell_barcode = cb_tag.split(":")[2]
            cell_reads_dict.setdefault(cell_barcode, []).append(read_id)
            reads_cell_dict[read_id] = cell_barcode
    return cell_reads_dict, reads_cell_dict


def samtools_get_group_read_count(group_bam_dict, chr, start, end):
    """
    Get the number of reads in the group_bam_dict for given region.
    """
    group_bam_length_dict = {}
    for group, bam_path in group_bam_dict.items():
        group_bam_length_dict[group] = samtools_get_indexed_read_count(bam_path, chr, start, end)
    return group_bam_length_dict


def samtools_get_indexed_read_count(bam_path, chr, start, end, logger, cell_barcodes = None,
                                    NCORES = 16, VERBOSE = False):
    """
    Get the number of reads in a specific region.

    Parameters:
    ----------
    bam_path (str): Path to source BAM file.
    chr (str): Chromosome name
    start (int): Start position of the region.
    end (int): End position of the region.

    Returns:
    -------
    read_count: Number of reads in the specified region.
    """
    cmd = [
        "samtools", "view", "-@", str(NCORES),
        "-c"
    ]

    if cell_barcodes:
        for cb in cell_barcodes:
            cmd.extend(["-d", f"CB:{cb}"])

    cmd.append(bam_path)
    cmd.append(f"{chr}:{start}-{end}")

    try:
        if VERBOSE:
            logger.debug(f"Executing command: {' '.join(cmd)}")
        region_reads_result = subprocess.run(cmd, stdout=subprocess.PIPE, check = True, text = True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing command: {' '.join(cmd)}")
        logger.error(f"Error message: {e}")
        raise

    read_count = int(region_reads_result.stdout.strip())
    return read_count


def samtools_get_unindexed_read_count(bam_path, logger, VERBOSE = False):
    """
    Get the number of reads in a specific region.

    Parameters:
    ----------
    bam_path (str): Path to source BAM file.
    chr (str): Chromosome name
    start (int): Start position of the region.
    end (int): End position of the region.

    Returns:
    -------
    read_count: Number of reads in the specified region.
    """
    cmd = f"samtools view {bam_path} | wc -l"
    try:
        if VERBOSE:
            logger.debug(f"Executing command: {cmd}")
        region_reads_result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, check = True, text = True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing command: {cmd}")
        logger.error(f"Error message: {e}")
        raise
    read_count = int(region_reads_result.stdout.strip())
    return read_count


def samtools_get_entire_read_count(bam_path, logger):
    """
    Returns the total number of mapped reads in a BAM file using `samtools idxstats`.

    Parameters:
    ----------
    bam_path (str): Path to the BAM file (must be indexed with .bai)

    Returns:
    -------
    int: Total number of mapped reads
    """
    cmd = ["samtools", "idxstats", bam_path]
    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        total_mapped_reads = 0
        for line in result.stdout.strip().split("\n"):
            fields = line.split("\t")
            if len(fields) >= 3:
                mapped = int(fields[2])
                total_mapped_reads += mapped
        return total_mapped_reads
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running samtools idxstats: {e.stderr}")
        raise


def samtools_sample_reads(bam_path, out_path, frac_reads, logger, region = None, cell_barcodes = None,
                          seed=5091130, NCORES = 16, VERBOSE = False):
    """
    Sample a fraction of reads from a specific region of a BAM file.
    
    Parameters:
    - bam_path: Input BAM file
    - out_path: Output BAM file
    - frac_reads: Fraction of reads to sample (0.0 to 1.0)
    - region: Genomic region string, e.g., 'chr1:10000-50000'. If None, samples from the whole file.
    - cell_barcodes: List of cell barcodes to sample from. If None, samples from all reads.
    - seed: Random seed for reproducibility
    """

    cmd = [
        "samtools", "view", "-@", str(NCORES),
        "--subsample-seed", str(seed),
        "--subsample", str(frac_reads),
        "-b"
    ]

    if cell_barcodes:
        for cb in cell_barcodes:
            cmd.extend(["-d", f"CB:{cb}"])

    cmd.append(bam_path)

    if region:
        cmd.append(region)

    cmd.extend(["-o", out_path])

    try:
        if VERBOSE:
            logger.debug(f"Executing command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing command: {' '.join(cmd)}")
        logger.error(f"Error message: {e}")
        raise
    return out_path


def samtools_merge(bam_paths, out_path, logger, NCORES = 16, VERBOSE = False):
    """
    Merge multiple BAM files into a single BAM file.

    Parameters:
    ----------
    bam_paths (list): List of paths to the BAM files to merge.
    out_path (str): Path to the output merged BAM file.

    Returns:
    -------
    out_path: Path to the merged BAM file.
    """
    cmd = [
        "samtools", "merge", "-@", str(NCORES),
        out_path,
        *bam_paths
    ]
    try:
        if VERBOSE:
            logger.debug(f"Executing command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing command: {' '.join(cmd)}")
        logger.error(f"Error message: {e.stderr.strip()}")
        raise
    return out_path

def samtools_index(bam_path, logger, NCORES = 16, VERBOSE = False):
    """
    Index a BAM file.

    Parameters:
    ----------
    bam_path (str): Path to the BAM file to index.

    Returns:
    -------
    None
    """
    cmd = [
        "samtools", "index", "-@", str(NCORES),
        bam_path
    ]
    try:
        if VERBOSE:
            logger.debug(f"Executing command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing command: {' '.join(cmd)}")
        logger.error(f"Error message: {e}")
        raise

def samtools_sort(bam_path, out_path, logger, NCORES = 16, VERBOSE = False):
    """
    Sort a BAM file.

    Parameters:
    ----------
    bam_path (str): Path to the BAM file to sort.
    out_path (str): Path to the output sorted BAM file.

    Returns:
    -------
    None
    """
    cmd = [
        "samtools", "sort", "-@", str(NCORES),
        "-o", out_path,
        bam_path
    ]
    try:
        if VERBOSE:
            logger.debug(f"Executing command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing command: {' '.join(cmd)}")
        logger.error(f"Error message: {e}")
        raise
    return out_path