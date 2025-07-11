################################################################################
## CINner Test Run
################################################################################

library(CINner)
set.seed(1)

cell_lifespan <- 30
T_0 <- list(30, "year")
T_end <- list(50, "year")

Table_sample <- data.frame(
  Sample_ID = "SA01",
  Cell_count = 300,
  Age_sample = 50
)

T_tau_step <- cell_lifespan / 2

CN_bin_length <- 500000

selection_model <- "chrom-arm-and-driver-gene-selection"

prob_CN_whole_genome_duplication <- 0.5e-4
prob_CN_missegregation <- 2e-4
prob_CN_chrom_arm_missegregation <- 2e-4
prob_CN_focal_amplification <- 0
prob_CN_focal_deletion <- 0

prob_neutral_CN_whole_genome_duplication <- 0
prob_neutral_CN_missegregation <- 0
prob_neutral_CN_chrom_arm_missegregation <- 0
prob_neutral_CN_focal_amplification <- 0
prob_neutral_CN_focal_deletion <- 0

model_CN_focal_amplification_length <- "geom"
prob_CN_focal_amplification_length <- 10
model_CN_focal_deletion_length <- "beta"
prob_CN_focal_deletion_length_shape_1 <- 1
prob_CN_focal_deletion_length_shape_2 <- 7

rate_driver <- 0
rate_passenger <- 1e-11

bound_driver <- 3
bound_average_ploidy <- 4.5
bound_maximum_CN <- 8
bound_homozygosity <- 0

table_population_dynamics <- cbind(
  vec_time = T_0[[1]]:T_end[[1]],
  vec_cell_count = 10000 / (1 + exp(-0.3 * ((T_0[[1]]:T_end[[1]]) - 20)))
)

model_variables <- BUILD_general_variables(
  cell_lifespan = cell_lifespan,
  T_0 = T_0, T_end = T_end, T_tau_step = T_tau_step,
  Table_sample = Table_sample,
  CN_bin_length = CN_bin_length,
  prob_CN_whole_genome_duplication = prob_CN_whole_genome_duplication,
  prob_CN_missegregation = prob_CN_missegregation,
  prob_CN_chrom_arm_missegregation = prob_CN_chrom_arm_missegregation,
  prob_CN_focal_amplification = prob_CN_focal_amplification,
  prob_CN_focal_deletion = prob_CN_focal_deletion,
  prob_neutral_CN_whole_genome_duplication = prob_neutral_CN_whole_genome_duplication,
  prob_neutral_CN_missegregation = prob_neutral_CN_missegregation,
  prob_neutral_CN_chrom_arm_missegregation = prob_neutral_CN_chrom_arm_missegregation,
  prob_neutral_CN_focal_amplification = prob_neutral_CN_focal_amplification,
  prob_neutral_CN_focal_deletion = prob_neutral_CN_focal_deletion,
  model_CN_focal_amplification_length = model_CN_focal_amplification_length,
  model_CN_focal_deletion_length = model_CN_focal_deletion_length,
  prob_CN_focal_amplification_length = prob_CN_focal_amplification_length,
  prob_CN_focal_deletion_length_shape_1 = prob_CN_focal_deletion_length_shape_1,
  prob_CN_focal_deletion_length_shape_2 = prob_CN_focal_deletion_length_shape_2,
  rate_driver = rate_driver,
  rate_passenger = rate_passenger,
  selection_model = selection_model,
  bound_driver = bound_driver,
  bound_average_ploidy = bound_average_ploidy,
  bound_homozygosity = bound_homozygosity,
  table_population_dynamics = table_population_dynamics
)

table_arm_selection_rates <- data.frame(
  Arm_ID = c(
    paste(model_variables$cn_info$Chromosome, "p", sep = ""),
    paste(model_variables$cn_info$Chromosome, "q", sep = "")
  ),
  Chromosome = rep(model_variables$cn_info$Chromosome, 2),
  Bin_start = c(
    rep(1, length(model_variables$cn_info$Chromosome)),
    model_variables$cn_info$Centromere_location + 1
  ),
  Bin_end = c(
    model_variables$cn_info$Centromere_location,
    model_variables$cn_info$Bin_count
  ),
  s_rate = runif(2 * length(model_variables$cn_info$Chromosome), 1 / 1.2, 1.2)
)


table_gene_selection_rates <- data.frame(
  Gene_ID = c("Driver_1", "Driver_2", "Driver_3"),
  Gene_role = c("TSG", "ONCOGENE", "TSG"),
  s_rate = runif(3, 1, 1.2),
  Chromosome = c(1, 2, 3),
  Bin = c(10, 20, 30)
)

model_variables <- BUILD_driver_library(
  model_variables = model_variables,
  table_arm_selection_rates = table_arm_selection_rates,
  table_gene_selection_rates = table_gene_selection_rates
)

CN_matrix <- BUILD_cn_normal_autosomes(model_variables$cn_info)

drivers <- list("Driver_1")

model_variables <- BUILD_initial_population(
  model_variables = model_variables,
  cell_count = 20,
  CN_matrix = CN_matrix,
  drivers = list()
)

model_variables <- CHECK_model_variables(model_variables)

CINner_simulations <- simulator_full_program(
  model = model_variables,
  n_simulations = 8,
  stage_final = 4,
  compute_parallel = TRUE
)



sim_save_path <- "/data1/shahs3/users/sunge/cnv_simulator/CINner_simulations/def_8sims_v1.rds"
saveRDS(CINner_simulations, file = sim_save_path)

CINner_simulations <- readRDS(sim_save_path)

for (i in seq_along(CINner_simulations)) {
  sample_name <- paste0("cinner_def1_", i)
  save_dir <- paste0("/data1/shahs3/users/sunge/cnv_simulator/synthetic_bams_2/", sample_name)
  if (!dir.exists(save_dir)) {
    dir.create(save_dir, recursive = TRUE)
  }
  
  sim_cp_profile <- CINner_simulations[[i]][["sample"]][["cn_profiles_long"]]
  sim_cell_clone <- CINner_simulations[[i]][["sample"]][["table_cell_clone"]]
  
  write.csv(sim_cp_profile,
            paste0(save_dir, "/", sample_name, "_raw_cnv_profile.csv"),
            row.names = FALSE, quote = FALSE)
  write.csv(sim_cell_clone,
            paste0(save_dir, "/", sample_name, "_cell_clone.csv"),
            row.names = FALSE, quote = FALSE)
}








