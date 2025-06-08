import os
import subprocess
import logging
import tempfile
from pathlib import Path

##################################################################
# SAMTOOLS UTILS
##################################################################

logger = logging.getLogger(__name__)

def samtools_read_id_filter(bam_path, out_path, read_ids, NCORES = 16):
    """
    Saves new bam file with only the specified read IDs.
    """
    with tempfile.NamedTemporaryFile(mode = "w", delete = False) as tmp:
        for rid in read_ids:
            tmp.write(f"{rid}\n")
        tmp_path = tmp.name
        logger.debug(f"Temporary file created: {tmp_path}")

    # Print number of read IDs to be filtered
    logger.debug(f"Number of read IDs to filter: {len(read_ids)}")
    logger.debug(f"Read IDs: {read_ids[:10]}...")

    try:
        cmd = [
            "samtools", "view", "-@", str(NCORES),
            "-b", "-N", tmp_path, bam_path,
            "-o", out_path
        ]
        logger.debug(f"Executing command: {' '.join(cmd)}")
        subprocess.run(cmd, check = True)
    finally:
        os.remove(tmp_path)
        logger.debug(f"Temporary file deleted: {tmp_path}")
    
    return out_path


def samtools_get_cell_reads(bam_path, region = None, NCORES = 32):
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
    if region:
        cmd.append(region)
    try:
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


def samtools_get_group_read_count(group_bam_dict, region):
    """
    Get the number of reads in the group_bam_dict for given region.
    """
    group_bam_length_dict = {}
    for group, bam_path in group_bam_dict.items():
        group_bam_length_dict[group] = samtools_get_indexed_read_count(bam_path, region)
    return group_bam_length_dict


def samtools_get_indexed_read_count(bam_path, region = None, cell_barcodes = None, NCORES = 32):
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

    if region:
        cmd.append(region)

    try:
        logger.debug(f"Executing command: {' '.join(cmd)}")
        region_reads_result = subprocess.run(cmd, stdout=subprocess.PIPE, check = True, text = True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing command: {' '.join(cmd)}")
        logger.error(f"Error message: {e}")
        raise

    read_count = int(region_reads_result.stdout.strip())
    return read_count


def samtools_get_unindexed_read_count(bam_path):
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
        logger.debug(f"Executing command: {cmd}")
        region_reads_result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, check = True, text = True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing command: {cmd}")
        logger.error(f"Error message: {e}")
        raise
    read_count = int(region_reads_result.stdout.strip())
    return read_count


def samtools_get_entire_read_count(bam_path):
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


def samtools_sample_reads(bam_path, out_path, frac_reads, region = None, cell_barcodes = None,
                          seed=5091130, NCORES = 32):
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
        logger.debug(f"Executing command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing command: {' '.join(cmd)}")
        logger.error(f"Error message: {e}")
        raise
    return out_path


def samtools_get_reads(bam_path, out_path, region = None, cell_barcodes = None,
                       NCORES = 32):
    """
    Subset all reads from a specific region of a BAM file with specified barcodes.
    
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
        logger.debug(f"Executing command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing command: {' '.join(cmd)}")
        logger.error(f"Error message: {e}")
        raise
    return out_path


def _run_samtools_merge(bam_paths, out_path, NCORES):
    cmd = [
        "samtools", "merge", "-@", str(NCORES), "-c", "-p", "--no-PG",
        out_path,
        *bam_paths
    ]
    try:
        logger.debug(f"Executing command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing command: {' '.join(cmd)}")
        logger.error(f"Error message: {e.stderr.strip()}")
        raise
    return out_path


def samtools_merge(bam_paths, out_path, NCORES=32, batch_size=10, _tmpdir_obj=None):
    """
    Merge multiple BAM files into a single BAM file using samtools.

    Parameters:
    ----------
    bam_paths (list): List of paths to the BAM files to merge.
    out_path (str): Path to the output merged BAM file.
    NCORES (int): Number of threads to use for merging.
    batch_size (int): Number of BAM files to merge at once. Default is 10.
    _tmpdir_obj (TemporaryDirectory): Optional existing temp directory object.

    Returns:
    -------
    out_path: Path to the merged BAM file.
    """
    if len(bam_paths) == 0:
        raise ValueError("No BAM files provided for merging.")

    created_temp_dir = False
    if _tmpdir_obj is None:
        _tmpdir_obj = tempfile.TemporaryDirectory()
        created_temp_dir = True

    tmpdir = Path(_tmpdir_obj.name)
    round_num = 0
    current_paths = bam_paths

    while len(current_paths) > 1:
        logger.info(f"Batching round {round_num + 1}: merging {len(current_paths)} files.")
        next_paths = []
        for i in range(0, len(current_paths), batch_size):
            batch = current_paths[i:i + batch_size]
            # Decide if this is the final merge
            is_final_round = len(current_paths) <= batch_size
            batch_out = out_path if is_final_round else str(tmpdir / f"merged_round{round_num}_batch{i // batch_size}.bam")
            logger.info(f"Merging batch {i // batch_size + 1} of {len(batch)} files in round {round_num}")
            _run_samtools_merge(batch, batch_out, NCORES)
            next_paths.append(batch_out)
        current_paths = next_paths
        round_num += 1

    if created_temp_dir:
        _tmpdir_obj.cleanup()

    return out_path


def samtools_index(bam_path, NCORES = 32):
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
        logger.debug(f"Executing command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing command: {' '.join(cmd)}")
        logger.error(f"Error message: {e}")
        raise

def samtools_sort(bam_path, out_path, NCORES = 32):
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
        logger.debug(f"Executing command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing command: {' '.join(cmd)}")
        logger.error(f"Error message: {e}")
        raise
    return out_path


def _run_picard_merge(bam_paths, out_path):
    """
    Run Picard's MergeSamFiles to merge multiple BAM files into one.

    Parameters:
    ----------
    bam_paths (list): List of paths to the BAM files to merge.
    out_path (str): Path to the output merged BAM file.

    Returns:
    -------
    out_path: Path to the merged BAM file.
    """
    input_args = " ".join(f"INPUT={bam}" for bam in bam_paths)

    picard_path = os.path.expanduser("~/packages/picard.jar")

    cmd = (
        f"java -Xmx128g -jar {picard_path} MergeSamFiles "
        f"{input_args} OUTPUT={out_path} USE_THREADING=true VALIDATION_STRINGENCY=LENIENT"
    )

    try:
        logger.info(f"Merging {len(bam_paths)} BAM files into {out_path} using Picard.")
        logger.debug(f"Executing command: {cmd}")
        subprocess.run(cmd, check=True, shell=True, text=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing command: {cmd}")
        logger.error(f"Error message: {e}")
        raise
    
    return out_path

def check_bam_integrity(bam_paths):
    for bam in bam_paths:
        # quickcheck first
        result = subprocess.run(["samtools", "quickcheck", bam])
        if result.returncode != 0:
            raise RuntimeError(f"Corrupted BAM file detected by quickcheck: {bam}")
        # try reading the header
        result = subprocess.run(["samtools", "view", "-H", bam], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise RuntimeError(f"Corrupted BAM file detected by view -H: {bam}\n{result.stderr.decode()}")

def picard_merge(bam_paths, out_path, batch_size=10, NCORES=32, _tmpdir_obj=None):
    """
    Merges multiple BAM files using Picard's MergeSamFiles, batching if needed.

    Parameters:
    -----------
    bam_paths (list): List of paths to BAM files to merge.
    out_path (str): Path to the final merged BAM file.
    batch_size (int): Maximum number of files to merge in one call to Picard.
    _tmpdir_obj (tempfile.TemporaryDirectory or None): Optional existing temporary directory.

    Returns:
    --------
    out_path (str): Path to the merged BAM file.
    """
    logger.info("Starting Picard merge.")

    check_bam_integrity(bam_paths)

    created_temp_dir = False
    if _tmpdir_obj is None:
        _tmpdir_obj = tempfile.TemporaryDirectory()
        created_temp_dir = True

    tmpdir = Path(_tmpdir_obj.name)
    intermediate_paths = list(bam_paths)
    round_num = 0

    while len(intermediate_paths) > batch_size:
        logger.info(f"Batching round {round_num + 1}: merging {len(intermediate_paths)} files.")
        new_intermediate_paths = []
        for i in range(0, len(intermediate_paths), batch_size):
            batch = intermediate_paths[i:i + batch_size]
            batch_out = tmpdir / f"merged_round{round_num}_batch_{i // batch_size}.bam"
            logger.debug(f"Merging batch {i // batch_size + 1} with {len(batch)} files.")
            # Using samtools for internal batches
            _run_samtools_merge(batch, str(batch_out), NCORES)
            new_intermediate_paths.append(str(batch_out))
        intermediate_paths = new_intermediate_paths
        round_num += 1

    for batch_out in new_intermediate_paths:
        result = subprocess.run(["samtools", "quickcheck", batch_out])
        if result.returncode != 0:
            raise RuntimeError(f"Corrupted BAM file detected: {batch_out}")

    # Final merge
    logger.info(f"Final merge of {len(intermediate_paths)} BAM files into {out_path}.")
    _run_picard_merge(intermediate_paths, out_path)

    if created_temp_dir:
        _tmpdir_obj.cleanup()

    return out_path



##################################################################
### BEDTOOLS UTILS
##################################################################

def get_genomic_windows(bin_size, chr_lengths_path, out_path):
    """
    Generate genomic windows of a specified size for each chromosome.

    Parameters:
    ----------
    bin_size (int): Size of the bins in base pairs.
    chr_lengths_path (str): Path to the file containing chromosome lengths.
    out_path (str): Path to the output bed file.

    Returns:
    -------
    None
    """
    cmd = [
        "bedtools", "makewindows",
        "-g", chr_lengths_path,
        "-w", str(bin_size)
    ]
    
    with open(out_path, 'w') as out_file:
        subprocess.run(cmd, stdout=out_file, check=True)
    return out_path


def get_bin_coverage(bam_path, bin_bed_path, out_path):
    """
    Calculate the coverage of each bin in a BAM file.

    Parameters:
    ----------
    bam_path (str): Path to the BAM file.
    bin_bed_path (str): Path to the bed file containing genomic windows.
    out_path (str): Path to the output bed file with coverage information.

    Returns:
    -------
    None
    """
    cmd = [
        "bedtools", "coverage",
        "-a", bin_bed_path,
        "-b", bam_path,
        "-counts",
        "-g", bin_bed_path
    ]
    
    with open(out_path, 'w') as out_file:
        subprocess.run(cmd, stdout=out_file, check=True)
    return out_path