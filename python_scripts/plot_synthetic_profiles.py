import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sp
import argparse

BASEDIR = "/data1/shahs3/users/sunge/cnv_simulator"
DATADIR = f"{BASEDIR}/data"
BAMDIR = f"{BASEDIR}/synthetic_bams_2"

copynumber_colors = {
    "0": "#2C78B2",  # Dark blue
    "1": "#94C4DB",  # Light blue
    "2": "#C5C5C5",  # Gray
    "3": "#FCC484",  # Light orange
    "4": "#FA8154",  # Orange
    "5": "#DD4031",  # Red-orange
    "6": "#A9000D",  # Dark red
    "7": "#8C033A",  # Burgundy
    "8": "#6A1B9A",  # Medium purple
    "9": "#4A148C",  # Dark purple
    "10": "#38006B",  # Very dark purple
    "11": "#2E0057",  # Deep purple
    "12": "#240043",  # Almost black purple
    "13": "#1A002F",  # Darkest purple
    "14": "#12001C",  # Near-black purple
}

def plot_true_cnv(cnv_profile_df, n_cells_per_clone, centromere_pos, FIGDIR = None):
    num_clones = len(cnv_profile_df["clone"].unique())

    chr_lengths = cnv_profile_df.groupby("chr")["end"].max().sort_index()
    chr_offsets = chr_lengths.cumsum().shift(fill_value = 0).to_dict()
    chr_labels = {pos: chr for chr, pos in chr_offsets.items()}

    plot_df = cnv_profile_df.copy()
    plot_df.loc[:, "genome_start"] = plot_df.apply(
        lambda row: row["start"] + chr_offsets[row["chr"]], axis=1)
    plot_df.loc[:, "genome_end"] = plot_df.apply(
        lambda row: row["end"] + chr_offsets[row["chr"]], axis=1)

    fig, ax = plt.subplots(nrows = len(plot_df["clone"].unique()),
                        ncols = 1,
                        figsize = (10, 2 * num_clones))

    for i, clone in enumerate(plot_df["clone"].unique()):
        clone_df = plot_df[plot_df["clone"] == clone]
        for _, row in clone_df.iterrows():
            ax[i].plot(
                [row["genome_start"], row["genome_end"]],
                [row["state"], row["state"]],
                color = copynumber_colors[str(min(14, row["copy_number"]))],
                alpha=0.7,
            )
        for offset in chr_offsets.values():
            ax[i].axvline(offset, color = "grey", linestyle = "--", linewidth = 0.5)
        for _, cent_row in centromere_pos.iterrows():
            if cent_row["chrom"] not in chr_offsets:
                continue
            cent_start = cent_row["chromStart"] + chr_offsets[cent_row["chrom"]]
            cent_end = cent_row["chromEnd"] + chr_offsets[cent_row["chrom"]]
            ax[i].axvspan(cent_start, cent_end, color = "lightgrey", alpha = 0.5, label = "Centromere" if i == 0 else None)

        ax[i].set_xticks(
            ticks = list(chr_offsets.values()),
            labels = list(chr_labels.values()),
            ha = "center"
        )

        ax[i].set_ylabel("State")
        ax[i].set_title(f"Clone {clone} ({n_cells_per_clone[i]} cells)")
        ax[i].set_ylim(-2, 14)

    plt.tight_layout()

    if FIGDIR is not None:
        fig.savefig(f"{FIGDIR}/true_cnv_profile.png", dpi = 300, bbox_inches = "tight")
        print(f"Saved figure to {FIGDIR}/true_cnv_profile.png")
    else:
        plt.show()

def plot_read_depth_heatmap(read_depth_mat, clone_cell_idx_dict, clone_lst, bins, bin_size, bin_mult, chr_lst, 
                            bin_range = None, prefix = "", vmax = 100, FIGDIR = None):

    clone_heatmap_data_lst = []
    row_colors = []

    clone_color_palette = sns.color_palette("tab10", n_colors = len(clone_lst))

    for clone in clone_lst:
        clone_idxs = sorted(clone_cell_idx_dict[f"clone{clone}"])
        chr_heatmap_data_lst = []
        for chr in chr_lst:
            if bin_range is not None:
                chr_bin_idxs = bins[(bins["chrom"] == chr) & (bins["start"] >= bin_range[0]) & (bins["end"] <= bin_range[1])].index.tolist()
            else:
                chr_bin_idxs = bins[bins["chrom"] == chr].index.tolist()
            chr_heatmap_data = read_depth_mat.tocsr()[chr_bin_idxs, :][:, clone_idxs].toarray()
            nrows = chr_heatmap_data.shape[0]
            if nrows % bin_mult != 0:
                chr_heatmap_data = chr_heatmap_data[:nrows - nrows % bin_mult, :]
            chr_heatmap_data = chr_heatmap_data.reshape(-1, bin_mult, len(clone_idxs)).sum(axis = 1)
            chr_heatmap_data_lst.append(chr_heatmap_data.T)
        clone_heatmap_data = np.concatenate(chr_heatmap_data_lst, axis = 1)
        clone_heatmap_data_lst.append(clone_heatmap_data)
        row_colors.extend([clone_color_palette[int(clone)]] * clone_heatmap_data.shape[0])

    heatmap_data = np.concatenate(clone_heatmap_data_lst, axis = 0)

    chr_start_pos = []
    pos = 0
    for chr in chr_lst:
        n_bins = len(bins[bins["chrom"] == chr])
        chr_start_pos.append(pos)
        pos += n_bins // bin_mult

    print(f"vmax = {vmax}")
    weighted_vmax = vmax * bin_mult

    plt.figure(figsize = (10, 5 * len(chr_lst)))
    cluster_plt = sns.clustermap(
        heatmap_data,
        cmap = "viridis",
        cbar_pos = (1.03, 0.6, 0.03, 0.2),
        cbar_kws = {"label": "Read depth"},
        row_cluster = False,
        col_cluster = False,
        dendrogram_ratio = (0.01, 0.01),
        xticklabels = False,
        yticklabels = False,
        row_colors = row_colors,
        vmax = weighted_vmax
    )
    cluster_plt.ax_heatmap.set_ylabel("Cells")
    cluster_plt.ax_heatmap.set_xlabel("Bins")
    cluster_plt.ax_heatmap.set_xticks(chr_start_pos)
    cluster_plt.ax_heatmap.set_xticklabels(chr_lst)

    for xpos in chr_start_pos:
        cluster_plt.ax_heatmap.axvline(x = xpos, color = "white", linestyle = "-", linewidth = 1)

    if FIGDIR is not None:
        fig_name = f"{prefix}read_depth_heatmap_bin{bin_size // 1000 * bin_mult}kb.png"
        cluster_plt.fig.savefig(f"{FIGDIR}/{fig_name}", dpi = 300, bbox_inches = "tight")
    else:
        plt.show()


def plot_read_depth_scatter(read_depth_mat, clone_cell_idx_dict, clone_lst, bins, bin_size, bin_mult, chr_lst,
                            bin_range = None, prefix = "", FIGDIR = None):

    fig, ax = plt.subplots(figsize=(10, 3 * len(clone_lst)), nrows = len(clone_lst), ncols = len(chr_lst), squeeze = False)

    for i, clone in enumerate(clone_lst):
        clone_idxs = sorted(clone_cell_idx_dict[f"clone{clone}"])
        for j, chr in enumerate(chr_lst):
            if bin_range is not None:
                chr_bin_idxs = bins[(bins["chrom"] == chr) & (bins["start"] >= bin_range[0]) & (bins["end"] <= bin_range[1])].index.tolist()
            else:
                chr_bin_idxs = bins[bins["chrom"] == chr].index.tolist()
            chr_read_depth = read_depth_mat.tocsr()[chr_bin_idxs, :][:, clone_idxs].toarray()
            nrows = chr_read_depth.shape[0]
            if nrows % bin_mult != 0:
                chr_read_depth = chr_read_depth[:nrows - nrows % bin_mult, :]
            chr_read_depth = chr_read_depth.reshape(-1, bin_mult, len(clone_idxs)).sum(axis=1)

            mean_chr_read_depth = chr_read_depth.mean(axis = 1)
            ax[i, j].scatter(
                np.arange(len(mean_chr_read_depth)),
                mean_chr_read_depth,
                s=3
            )
            ax[i, j].set_yscale("log")
            ax[i, j].set_title(f"Clone {clone} - chr{chr}")
            ax[i, j].set_xlabel("Bins")
            ax[i, j].set_ylabel("Log(Read Depth)")
            ax[i, j].set_ylim(1, 10000)
    
    plt.tight_layout()

    if FIGDIR is not None:
        fig_name = f"{prefix}_read_depth_scatter_bin{bin_size // 1000 * bin_mult}kb.png"
        fig.savefig(f"{FIGDIR}/{fig_name}", dpi = 300, bbox_inches = "tight")
        print(f"Saved figure to {FIGDIR}/{fig_name}")
    else:
        plt.show()
    
def parse_args():
    parser = argparse.ArgumentParser(description="Plot synthetic CNV profiles and read depths.")
    parser.add_argument("--test_name", type=str, required=True, help="Name of the CNV profile.")
    parser.add_argument("--bin_size", type=int, default=5000, help="Size of bins in base pairs.")
    parser.add_argument("--min_bin_start", type=int, default=0, help="Minimum bin start position.")
    parser.add_argument("--max_bin_end", type=int, default=0, help="Maximum bin end position.")
    return parser.parse_args()


def main():
    args = parse_args()
    test_name = args.test_name
    bin_size = args.bin_size
    min_bin_start = args.min_bin_start
    max_bin_end = args.max_bin_end

    if max_bin_end == 0:
        bin_range = None
    else:
        bin_range = (min_bin_start, max_bin_end)

    # Set up figure directory
    FIGDIR = f"{BAMDIR}/{test_name}/figs"
    if not os.path.exists(FIGDIR):
        os.makedirs(FIGDIR)

    # Read in the CNV profile
    cnv_profile_df = pd.read_csv(f"{BAMDIR}/{test_name}/{test_name}_cell_profile.tsv", sep="\t")
    cnv_profile_df["chr"] = cnv_profile_df["chr"].astype(str)


    data_cnv_profile_df = cnv_profile_df.loc[cnv_profile_df["chr"] != "0"]
    clone_cnv_profile_df = cnv_profile_df.loc[cnv_profile_df["chr"] == "0"]
    clone_cell_count_lst = [
        len(row["cell_barcode"].split(","))
        for _, row in clone_cnv_profile_df.iloc[1:].iterrows()
    ]
    print(f"Number of clones: {len(clone_cell_count_lst)}")
    print(f"Number of cells in each clone: {clone_cell_count_lst}")

    chr_lst = data_cnv_profile_df["chr"].unique().tolist()
    clone_lst = sorted(data_cnv_profile_df["clone"].unique().tolist())

    print(chr_lst, clone_lst)

    # Get cell barcode to index mapping
    baseline_barcodes_str = cnv_profile_df[cnv_profile_df["clone"] == -1]["cell_barcode"].iloc[0]
    baseline_barcodes = [x.strip() for x in baseline_barcodes_str.split(",")]
    cell_idx_map= {cell_barcode: i for i, cell_barcode in enumerate(baseline_barcodes)}

    # Get cell barcodes for each clone
    clone_cell_id_dict = {}
    clone_cell_idx_dict = {}
    for _, clone in enumerate(clone_cnv_profile_df["clone"].unique()):
        if clone != -1:
            clone_cell_id_dict[f"clone{clone}"] = set(
                clone_cnv_profile_df.loc[clone_cnv_profile_df["clone"] == clone, "cell_barcode"].values[0].split(",")
            )

            clone_cell_idx_dict[f"clone{clone}"] = set(
                cell_idx_map[cb] for cb in clone_cell_id_dict[f"clone{clone}"] if cb in cell_idx_map
            ) 

    # Load read depth matrix
    read_depth_path = f"{BAMDIR}/{test_name}/{test_name}_final_sorted_cnv.bam.{bin_size}_read_depth.npz"
    read_depth_mat = sp.load_npz(read_depth_path)
    print(f"Read depth matrix shape: {read_depth_mat.shape}")

    # Load bin labels
    gene_windows_path = f"{DATADIR}/genome_{bin_size // 1000}kb_bins.bed"
    bins = []
    with open(gene_windows_path) as f:
        for line in f:
            chrom, start, end = line.strip().split()[:3]
            bins.append((chrom, int(start), int(end)))
    bins = pd.DataFrame(bins, columns=["chrom", "start", "end"])

    # Get centromere locations
    centromere_pos = pd.read_csv(f"{DATADIR}/genome_gaps.tsv", sep = "\t")
    centromere_pos = centromere_pos[centromere_pos["type"] == "centromere"]
    centromere_pos["chrom"] = centromere_pos["chrom"].str.replace("chr", "")
    centromere_pos = centromere_pos[["chrom", "chromStart", "chromEnd"]].reset_index(drop=True)

    # Plot true CNV profile
    plot_true_cnv(data_cnv_profile_df, clone_cell_count_lst, centromere_pos, FIGDIR)

    # Plot read depth heatmap
    for bin_mult in [1, 2, 10, 20]:
        plot_read_depth_heatmap(
            read_depth_mat, 
            clone_cell_idx_dict, 
            clone_lst, 
            bins, 
            bin_size, 
            bin_mult = bin_mult,  # Change this to adjust binning
            chr_lst = chr_lst,
            bin_range = bin_range,
            prefix = f"{test_name}_",
            vmax = 30,
            FIGDIR = FIGDIR
        )

        plot_read_depth_scatter(
            read_depth_mat, 
            clone_cell_idx_dict, 
            clone_lst, 
            bins, 
            bin_size, 
            bin_mult = bin_mult,
            chr_lst = chr_lst,
            bin_range = bin_range,
            prefix = f"{test_name}_",
            FIGDIR = FIGDIR
        )

if __name__ == "__main__":
    main()