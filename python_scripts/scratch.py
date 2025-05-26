import os
import sys
import logging

import pandas as pd
import numpy as np
import pysam
import shutil
from samtools_utils import *

BASEDIR = "/data1/shahs3/users/sunge/cnv_simulator"
profile_name = "minitest_c2_1"

BAMDIR = f"{BASEDIR}/synthetic_bams_2/{profile_name}"

cnv_profile_df = pd.read_csv(f"{BAMDIR}/{profile_name}_cell_profile.tsv",
                             sep="\t", header=0)
data_cnv_profile_df = cnv_profile_df.loc[cnv_profile_df["chr"] != 0]
clone_cnv_profile_df = cnv_profile_df.loc[cnv_profile_df["chr"] == 0]
clone_cell_count_lst = [
    len(row["cell_barcode"].split(","))
    for _, row in clone_cnv_profile_df.iloc[1:].iterrows()
]
print(clone_cnv_profile_df)

clone_cell_id_dict = {}
for _, clone in enumerate(clone_cnv_profile_df["clone"].unique()):
    if clone != -1:
        clone_cell_id_dict[f"clone{clone}"] = set(
            clone_cnv_profile_df.loc[clone_cnv_profile_df["clone"] == clone, "cell_barcode"].values[0].split(",")
        )
print(clone_cell_id_dict.keys())

baseline_bam_path = f"{BAMDIR}/{profile_name}_baseline_cells.bam"
# synthetic_bam_path = f"{BAMDIR}/{profile_name}_unnormalized_cnv.bam"
synthetic_bam_path = f"{BAMDIR}/{profile_name}_final_sorted_cnv.bam"
del_region = "1:72429122-76860719"

print(f"Baseline BAM path: {baseline_bam_path}")
baseline_region_counts = samtools_get_indexed_read_count(baseline_bam_path,
                                                         del_region,
                                                         cell_barcodes = clone_cell_id_dict["clone1"])
print(f"Baseline read count in {del_region}: {baseline_region_counts}")

print(f"Synthetic BAM path: {synthetic_bam_path}")
synthetic_region_counts = samtools_get_indexed_read_count(synthetic_bam_path,
                                                         del_region,
                                                         cell_barcodes = clone_cell_id_dict["clone1"])
print(f"Synthetic read count in {del_region}: {synthetic_region_counts}")