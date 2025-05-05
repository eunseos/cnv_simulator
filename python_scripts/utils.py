import pandas as pd
import numpy as np
import os
import sys
import pysam
import isabl_cli as ii


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

