################################################################################
## Clean CINner Profile
################################################################################

library(tidyverse)
library(ggplot2)


BASEDIR <- "/data1/shahs3/users/sunge/cnv_simulator/"
sample_name <- "cinner_def1_1"
sample_dir <- file.path(BASEDIR, "synthetic_bams_2", sample_name)
fig_dir <- file.path(sample_dir, "figs")
if (!dir.exists(fig_dir)) {
  dir.create(fig_dir, recursive = TRUE)
}

init_profile <- read.csv(file.path(sample_dir,
                                   paste0(sample_name, "_cnv_profile.csv")))

ggplot(init_profile, aes(x = copy)) +
  geom_histogram(binwidth = 1, fill = "skyblue") +
  facet_wrap(~ chr, ncol = 5) +
  theme_minimal() +
  labs(title = "Copy number histogram per chromosome",
       x = "Copy number",
       y = "Count")
ggsave(file.path(fig_dir, "cp_histogram_per_chr.png"),
       width = 10, height = 8)


