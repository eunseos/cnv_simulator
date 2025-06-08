import pyBigWig

def filter_bigwig_chroms(input_bw_path, output_bw_path, keep_chroms):
    bw = pyBigWig.open(input_bw_path)

    print("Chromosomes in input BigWig:", bw.chroms().keys())
    
    # Get chrom sizes from the input BigWig itself
    chrom_sizes = {chrom: bw.chroms()[chrom] for chrom in keep_chroms if chrom in bw.chroms()}
    
    # Collect all intervals
    output_bw = pyBigWig.open(output_bw_path, "w")
    output_bw.addHeader(list(chrom_sizes.items()))

    for chrom in keep_chroms:
        if chrom in bw.chroms():
            intervals = bw.intervals(chrom)
            if intervals:
                output_bw.addEntries(
                    [chrom]*len(intervals),
                    [start for start, end, val in intervals],
                    ends=[end for start, end, val in intervals],
                    values=[val for start, end, val in intervals]
                )
    
    bw.close()
    output_bw.close()

# Example usage
filter_bigwig_chroms("/data1/shahs3/users/sunge/cnv_simulator/data/refs_2/wgEncodeUwRepliSeqMcf7WaveSignalRep1.bigWig",
                     "/data1/shahs3/users/sunge/cnv_simulator/data/refs_2/wgEncodeUwRepliSeqMcf7WaveSignalRep1_chr1_chr2.bigWig", 
                     ["chr1", "chr2"])
