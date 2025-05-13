#%%
import os
import sys
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

BASEDIR = "/data1/shahs3/users/sunge/cnv_simulator"
BAMDIR = f"{BASEDIR}/synthetic_bams"

VERBOSE = True
NCORES = 32

#%%

def samtools_select_reads(bam_path, out_path, region = None, cell_barcodes = None):
    """
    Select reads from a BAM file based on cell barcodes and region.
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
        if VERBOSE:
            print(f"Executing command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {' '.join(cmd)}")
        print(f"Error message: {e}")
        raise

    # Index the output BAM file
    index_cmd = ["samtools", "index", "-@", str(NCORES), out_path]
    try:
        if VERBOSE:
            print(f"Indexing command: {' '.join(index_cmd)}")
        subprocess.run(index_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error indexing BAM file: {' '.join(index_cmd)}")
        print(f"Error message: {e}")
        raise
    return out_path


def mosdepth_run(bam_path, output_prefix):    
    cmd = [
        "/home/sunge/conda_envs/scanpy_env/bin/mosdepth",
        "--no-per-base",
        "--fast-mode",
        "--threads", str(NCORES),
        "--by", "1000",
        output_prefix,
        bam_path
    ]
    subprocess.run(cmd, check=True)
    if VERBOSE:
        print(f"mosdepth command: {' '.join(cmd)}")

    return

def main():
    test_name = "minitest_c3_9_sg0_500cells"

    bam_path = f"{BAMDIR}/{test_name}_cnv.bam"
    profile_path = f"{BASEDIR}/data/small_cnv_profiles/{test_name}_cnv_profile.tsv"

    profile_df = pd.read_csv(profile_path, sep="\t")
    
    for clone in profile_df["clone"].unique():
        clone_row = profile_df.loc[(profile_df["clone"] == clone) & (profile_df["chr"] == 0)]
        clone_cell_barcodes = clone_row["cell_barcode"].values[0].split(",")

        output_prefix = f"{BAMDIR}/{test_name}_cnv_depth/clone{clone}"
        tmp_bam_path = f"{BAMDIR}/{test_name}_cnv_depth/tmp.bam"

        # Select reads from the BAM file based on cell barcodes
        selected_bam_path = samtools_select_reads(bam_path, tmp_bam_path, cell_barcodes=clone_cell_barcodes)
        # Run mosdepth on the selected BAM file
        mosdepth_run(selected_bam_path, output_prefix)
        # Clean up the temporary BAM file
        if os.path.exists(tmp_bam_path):
            os.remove(tmp_bam_path)
        else:
            print(f"Temporary BAM file {tmp_bam_path} does not exist.")
    return

def parse_args():
    parser = argparse.ArgumentParser(description="Compute read depths from BAM files.")
    parser.add_argument("--test_name", type=str, required=True, help="Name of the test.")
    return parser.parse_args()

if __name__ == "__main__":
    main()