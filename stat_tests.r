# Script to perform statistical tests
#
# Author: Bjarne Schultze     last modified: 02.10.2024
# -----------------------------------------------------
# Empty environment
rm(list=ls())

# Load rstatix to calculate effect sizes
library(rstatix)

# Set alpha level
alpha <- 0.05

# Define a function that prevents SymmetryProblem errors due to no non-zero
# elements
wilcox_effsize_save <- function(data, subst, conf_level = 0.95) {
  # Check if there is any data point different form zero
  data_subset <- eval(parse(text = sprintf("data$%s", subst)))
  if (any(data_subset != 0)) {
    # If so, calculate effect size
    form <- sprintf("%s ~ 1", subst)
    eff_size <- wilcox_effsize(data, formula(form),
                               conf.level = conf_level)$effsize
  } else {
    # If there are only zeros, save NaN
    eff_size <- NaN
  }
  return(eff_size)
}
# Define a function that performs a Wilcoxon signed rank test and if no data
# are given returns a test result list with NaNs
wilcox_save <- function(data, conf_level = 0.95) {
  # Check if there is any data point different form zero
  if (length(data) > 0) {
    # If so, calculate test
    wilc_res <- wilcox.test(data, conf.level = conf_level, conf.int = TRUE)
  } else {
    # If there are only zeros, create fake results list
    wilc_res = list()
    wilc_res$p.value = NaN
    wilc_res$conf.int = c(NaN, NaN)
    attr(wilc_res$conf.int, "conf.level") = NaN
  }
  return(wilc_res)
}

# Set the experiment groups
exp_groups <- c("pIP10", "TN1A", "vPR13", "vMS12-SS3", "vPR13_ctrl", 
                "CsChrimson_ctrl")

#####---------------------------------------------------------------------------
# Test the changes in signal amounts averaged across stimulus intensities

# Allocate empty lists to store the results and conditions
test_res <- c()
test_res_confint = c()
test_res_covprob = c()
test_res_cond <- c()
test_res_grp <- c()
test_res_signal <- c()
eff_size <- c()

# Iterate over the experiment groups
for (exp_grp in exp_groups){
  # Load the data that was averaged per individual
  data <- read.csv(file =
                     sprintf("E:/MT/additional_files/%s_changes_to_test.csv",
                             exp_grp))

  # Subset male and male+female data
  data_m <- subset(data, data$condition == "m")
  data_mf <- subset(data, data$condition == "mf")

  # Perform Wilcoxon Signed Rank tests for all signal types
  wilc_p_m <- wilcox_save(data_m$pulse, conf_level = 1 - alpha)
  wilc_s_m <- wilcox_save(data_m$sine, conf_level = 1 - alpha)
  wilc_v_m <- wilcox_save(data_m$vib, conf_level = 1 - alpha)
  # Repeat for the male+female condition
  wilc_p_mf <- wilcox_save(data_mf$pulse, conf_level = 1 - alpha)
  wilc_s_mf <- wilcox_save(data_mf$sine, conf_level = 1 - alpha)
  wilc_v_mf <- wilcox_save(data_mf$vib, conf_level = 1 - alpha)
  
  # Calculate effect sizes
  effs_p_m <- wilcox_effsize_save(data_m, "pulse", conf_level = 1 - alpha)
  effs_s_m <- wilcox_effsize_save(data_m, "sine", conf_level = 1 - alpha)
  effs_v_m <- wilcox_effsize_save(data_m, "vib", conf_level = 1 - alpha)
  effs_p_mf <- wilcox_effsize_save(data_mf, "pulse", conf_level = 1 - alpha)
  effs_s_mf <- wilcox_effsize_save(data_mf, "sine", conf_level = 1 - alpha)
  effs_v_mf <- wilcox_effsize_save(data_mf, "vib", conf_level = 1 - alpha)

  # Append p values to array
  test_res <- append(test_res, c(wilc_p_m$p.value,
                                 wilc_s_m$p.value,
                                 wilc_v_m$p.value,
                                 wilc_p_mf$p.value,
                                 wilc_s_mf$p.value,
                                 wilc_v_mf$p.value))
  # Append effect sizes to array
  eff_size <- append(eff_size, c(effs_p_m,
                                 effs_s_m,
                                 effs_v_m,
                                 effs_p_mf,
                                 effs_s_mf,
                                 effs_v_mf))
  # Create confidence intervals as texts 
  cint_p <- sprintf("[%0.2f,%0.2f]", wilc_p_m$conf.int[1],
                       wilc_p_m$conf.int[2])
  cint_s <- sprintf("[%0.2f,%0.2f]", wilc_s_m$conf.int[1],
                       wilc_s_m$conf.int[2])
  cint_v <- sprintf("[%0.2f,%0.2f]", wilc_v_m$conf.int[1],
                       wilc_v_m$conf.int[2])
  cint_p_mf <- sprintf("[%0.2f,%0.2f]", wilc_p_mf$conf.int[1],
                  wilc_p_mf$conf.int[2])
  cint_s_mf <- sprintf("[%0.2f,%0.2f]", wilc_s_mf$conf.int[1],
                       wilc_s_mf$conf.int[2])
  cint_v_mf <- sprintf("[%0.2f,%0.2f]", wilc_v_mf$conf.int[1],
                       wilc_v_mf$conf.int[2])
  # Append confidence intervals
  test_res_confint <- append(test_res_confint, c(cint_p, cint_s, cint_v, 
                                                 cint_p_mf, cint_s_mf, 
                                                 cint_v_mf))
  
  # Append coverage probabilities
  test_res_covprob <- append(test_res_covprob, 
                             c(attributes(wilc_p_m$conf.int)$conf.level,
                               attributes(wilc_s_m$conf.int)$conf.level,
                               attributes(wilc_v_m$conf.int)$conf.level,
                               attributes(wilc_p_mf$conf.int)$conf.level,
                               attributes(wilc_s_mf$conf.int)$conf.level,
                               attributes(wilc_v_mf$conf.int)$conf.level))
  
  # Create array with the corresponding conditions
  test_res_grp <- append(test_res_grp,
                         c(rep(exp_grp, 6)))
  test_res_cond <- append(test_res_cond,
                          c(rep("m", 3), rep("mf", 3)))
  test_res_signal <- append(test_res_signal, c("pulse", "sine", "vib",
                                               "pulse", "sine", "vib"))

  # Write an overview of the results to a text file
  cat("\nWilcoxon Signed Rank Tests for", exp_grp,
      "\n==========================================",
      "\nSolitary male",
      "\n \t\t pulse \t\t\t sine \t\t\t vibration",

      "\n p val \t\t", round(wilc_p_m$p.value, 4),
      "\t\t", round(wilc_s_m$p.value, 4),
      "\t\t\t", round(wilc_v_m$p.value, 4),

      "\n conf.int \t", sprintf("[%0.2f,%0.2f]", wilc_p_m$conf.int[1],
                                wilc_p_m$conf.int[2]),
      "\t\t", sprintf("[%0.2f,%0.2f]", wilc_s_m$conf.int[1],
                      wilc_s_m$conf.int[2]),
      "\t\t", sprintf("[%0.2f,%0.2f]", wilc_v_m$conf.int[1],
                      wilc_v_m$conf.int[2]),

      "\n cover.prob \t", attributes(wilc_p_m$conf.int)$conf.level,
      "\t\t\t", attributes(wilc_s_m$conf.int)$conf.level,
      "\t\t\t", attributes(wilc_v_m$conf.int)$conf.level,

      "\n eff size \t", round(effs_p_m, 4),
      "\t\t", round(effs_s_m, 4),
      "\t\t\t", round(effs_v_m, 4),

      "\n------------------------------------------",
      "\nMale + female",
      "\n \t\t pulse \t\t\t sine \t\t\t vibration",
      "\n p val \t\t", round(wilc_p_mf$p.value, 4),
      "\t\t\t", round(wilc_s_mf$p.value, 4),
      "\t\t", round(wilc_v_mf$p.value, 4),

      "\n conf.int \t", sprintf("[%0.2f,%0.2f]", wilc_p_mf$conf.int[1],
                                wilc_p_mf$conf.int[2]),
      "\t", sprintf("[%0.2f,%0.2f]", wilc_s_mf$conf.int[1],
                    wilc_s_mf$conf.int[2]),
      "\t\t", sprintf("[%0.2f,%0.2f]", wilc_v_mf$conf.int[1],
                      wilc_v_mf$conf.int[2]),

      "\n cover.prob \t", attributes(wilc_p_mf$conf.int)$conf.level,
      "\t\t\t", attributes(wilc_s_mf$conf.int)$conf.level,
      "\t\t\t", attributes(wilc_v_mf$conf.int)$conf.level,

      "\n eff size \t", round(effs_p_mf, 4),
      "\t\t", round(effs_s_mf, 4),
      "\t\t", round(effs_v_mf, 4),

      "\n\nPerformed at:", as.character(Sys.time()), "\n",
      file = sprintf("E:/MT/additional_files/test_results_%s.txt", exp_grp))
}

# Combine p values and effect sizes in a data frame
all_test_res <- data.frame("condition" = test_res_cond, 
                           "group" = test_res_grp,
                           "signal" = test_res_signal,
                           "pvalue" = test_res,
                           "conf_int" = test_res_confint,
                           "cover_prob" = test_res_covprob,
                           "eff_size" = eff_size)
# Save the test results (csv format)
write.csv(all_test_res,
          "E:/MT/additional_files/test_results_global_changes.csv",
          row.names = FALSE)

#####---------------------------------------------------------------------------
# Tests for all possible signal type changes. The counts during and before 
# stimulation are tested with Wilcoxon signed rank tests as the data are paired.
# Effect sizes are computed, if possible. 

# Allocate empty lists to store the results and conditions for all tests
all_chgs_tests <- c()
all_chgs_confint <- c()
all_chgs_conf_l <- c()
all_chgs_cond <- c()
all_chgs_grp <- c()
all_chgs_changes <- c()
all_chgs_effsize <- c()

# Iterate over the experiment groups
for (exp_grp in exp_groups){
  # Load the data with counts for all possible signal type changes
  data_chg <- read.csv(file =
                         sprintf("E:/MT/additional_files/%s_all_changes.csv",
                                 exp_grp))
  # Extract the condition (m/mf) and period (dur/pre)
  condition <- data_chg$condition
  period <- data_chg$X
  
  # Iterate over column indices, i.e. the different signal changes 
  # (first is the period)
  for (ichange in seq(2, length(data_chg[0, ])-1)){
    
    
    # Subset the data and create data frames with the period (dur/pre) and the 
    # data, separately for m and mf condition
    all_current_data = data.frame("period" = subset(data_chg$X, 
                                                    condition == "m"), 
                                  "dat" = subset(data_chg[,ichange], 
                                                 condition == "m"))
    
    all_current_data_mf = data.frame("period" = subset(data_chg$X, 
                                                       condition == "mf"), 
                                     "dat" = subset(data_chg[,ichange], 
                                                    condition == "mf"))
    
    # Test the current signal change (male condition)
    if (any(!is.na(all_current_data$dat))) {
      wilc_allchg <- wilcox.test(dat ~ period, all_current_data,
                                 conf.int = TRUE, 
                                 paired = TRUE)
      # Calculate the effect size
      if (any(all_current_data$dat != 0)) {
        effs_all_chg <- wilcox_effsize(all_current_data, 
                                       dat ~ period, conf.level = 1 - alpha, 
                                       paired = TRUE)$effsize
      } else {
        # Store NaN if effect size cannot be calculated
        effs_all_chg = NaN
      }
    } else {
      # If there is no data, create fake results list
      wilc_allchg = list()
      wilc_allchg$p.value = NaN
      wilc_allchg$conf.int = c(NaN, NaN)
      attr(wilc_allchg$conf.int, "conf.level") = NaN
      # Store NaN if effect size cannot be calculated
      effs_all_chg = NaN
    }
    
    # Test for the male-female data
    if (any(!is.na(all_current_data_mf$dat))){
      wilc_allchg_mf <- wilcox.test(dat ~ period, all_current_data_mf,
                                    conf.int = TRUE, 
                                    paired = TRUE)
      
      # Calculate effect size for male-female condition, if any non-zero 
      # elements
      if (any(all_current_data_mf$dat != 0)) {
        effs_all_chg_mf <- wilcox_effsize(all_current_data_mf, 
                                          dat ~ period, conf.level = 1 - alpha, 
                                          paired = TRUE)$effsize
      }else{
        # Store NaN if effect size cannot be calculated
        effs_all_chg_mf = NaN
      }
      
    } else {
      # If there is no data, create fake results list
      wilc_allchg_mf = list()
      wilc_allchg_mf$p.value = NaN
      wilc_allchg_mf$conf.int = c(NaN, NaN)
      attr(wilc_allchg_mf$conf.int, "conf.level") = NaN
      # Store NaN if effect size cannot be calculated
      effs_all_chg_mf = NaN
    }
    
    
    # Append p values to array
    all_chgs_tests <- append(all_chgs_tests, c(wilc_allchg$p.value,
                                               wilc_allchg_mf$p.value))
    
    # Format the confidence intervals as texts
    cint = sprintf("[%0.2f,%0.2f]", wilc_allchg$conf.int[1], 
                   wilc_allchg$conf.int[2])
    cint_mf = sprintf("[%0.2f,%0.2f]", wilc_allchg_mf$conf.int[1], 
                      wilc_allchg_mf$conf.int[2])
    # Append the confidence intervals
    all_chgs_confint <- append(all_chgs_confint, c(cint, cint_mf))
    
    # Append the reached coverage probability
    all_chgs_conf_l <- append(all_chgs_conf_l, 
                             c(attributes(wilc_allchg$conf.int)$conf.level,
                               attributes(wilc_allchg_mf$conf.int)$conf.level)) 
    
    # Append effect sizes to array
    all_chgs_effsize <- append(all_chgs_effsize, c(effs_all_chg,
                                                   effs_all_chg_mf))
    # Create array with the corresponding conditions
    all_chgs_cond <- append(all_chgs_cond,
                            c("m", "mf"))
    all_chgs_grp <- append(all_chgs_grp,
                           c(rep(exp_grp, 2)))
    # Append the type of change
    all_chgs_changes <- append(all_chgs_changes, 
                               c(rep(colnames(data_chg)[ichange], 2)))
  }
}

# Combine the data into a data frame
all_changes_results = data.frame("change" = all_chgs_changes, 
                                 "condition" = all_chgs_cond,
                                 "group" = all_chgs_grp,
                                 "pvalue" = all_chgs_tests,
                                 "eff_size" = all_chgs_effsize,
                                 "conf_int" = all_chgs_confint, 
                                 "cover_prob" = all_chgs_conf_l)

# Save the test results (csv format)
write.csv(all_changes_results,
          "E:/MT/additional_files/test_results_all_changes.csv",
          row.names = FALSE)

#####---------------------------------------------------------------------------
# Test the changes in certain tracking metrics upon activation with Wilcoxon 
# Signed Rank tests against zero

# Allocate empty lists to store the results and conditions
test_res <- c()
test_res_confint = c()
test_res_covprob = c()
test_res_cond <- c()
test_res_grp <- c()
test_res_type <- c()
eff_size <- c()

# Iterate over the experiment groups
for (exp_grp in exp_groups){
  # Load the data that was averaged per individual
  data <- read.csv(file =
                     sprintf("E:/MT/additional_files/%s_changes_tracking.csv",
                             exp_grp))
  
  # Subset male and male+female data
  data_m <- subset(data, data$condition == "m")
  data_mf <- subset(data, data$condition == "mf")
  
  # Perform Wilcoxon Signed Rank tests (male)
  wilc_velo <- wilcox_save(data_m$change_velocity, conf_level = 1 - alpha)
  wilc_wings <- wilcox_save(data_m$change_wingangle, conf_level = 1 - alpha)

  # Tests for male-female condition
  wilc_velo_m <- wilcox_save(data_mf$change_velocity, 
                             conf_level = 1 - alpha)
  wilc_velo_f <- wilcox_save(data_mf$change_velocity_f, 
                             conf_level = 1 - alpha)
  wilc_v_wings_m <- wilcox_save(data_mf$change_wingangle, 
                                conf_level = 1 - alpha)
  wilc_v_dist <- wilcox_save(data_mf$change_distance, 
                             conf_level = 1 - alpha)
  
  # Calculate effect sizes
  effs_velo <- wilcox_effsize_save(data_m, "change_velocity", 
                                   conf_level = 1 - alpha)
  effs_wings <- wilcox_effsize_save(data_m, "change_wingangle", 
                                    conf_level = 1 - alpha)
  effs_velo_m <- wilcox_effsize_save(data_mf, "change_velocity", 
                                     conf_level = 1 - alpha)
  effs_velo_f <- wilcox_effsize_save(data_mf, "change_velocity_f", 
                                     conf_level = 1 - alpha)
  effs_wings_m <- wilcox_effsize_save(data_mf, "change_wingangle", 
                                      conf_level = 1 - alpha)
  effs_dist <- wilcox_effsize_save(data_mf, "change_distance", 
                                   conf_level = 1 - alpha)
  
  # Append p values to array
  test_res <- append(test_res, c(wilc_velo$p.value,
                                 wilc_wings$p.value,
                                 wilc_velo_m$p.value,
                                 wilc_velo_f$p.value,
                                 wilc_v_wings_m$p.value,
                                 wilc_v_dist$p.value))
  # Append effect sizes to array
  eff_size <- append(eff_size, c(effs_velo,
                                 effs_wings,
                                 effs_velo_m,
                                 effs_velo_f,
                                 effs_wings_m,
                                 effs_dist))
  # Create confidence intervals as texts 
  cint_p <- sprintf("[%0.2f,%0.2f]", wilc_velo$conf.int[1],
                    wilc_velo$conf.int[2])
  cint_s <- sprintf("[%0.2f,%0.2f]", wilc_wings$conf.int[1],
                    wilc_wings$conf.int[2])
  cint_v <- sprintf("[%0.2f,%0.2f]", wilc_velo_m$conf.int[1],
                    wilc_velo_m$conf.int[2])
  cint_p_mf <- sprintf("[%0.2f,%0.2f]", wilc_velo_f$conf.int[1],
                       wilc_velo_f$conf.int[2])
  cint_s_mf <- sprintf("[%0.2f,%0.2f]", wilc_v_wings_m$conf.int[1],
                       wilc_v_wings_m$conf.int[2])
  cint_v_mf <- sprintf("[%0.2f,%0.2f]", wilc_v_dist$conf.int[1],
                       wilc_v_dist$conf.int[2])
  # Append confidence intervals
  test_res_confint <- append(test_res_confint, c(cint_p, cint_s, cint_v, 
                                                 cint_p_mf, cint_s_mf, 
                                                 cint_v_mf))
  
  # Append coverage probabilities
  test_res_covprob <- append(test_res_covprob, 
                             c(attributes(wilc_velo$conf.int)$conf.level,
                               attributes(wilc_wings$conf.int)$conf.level,
                               attributes(wilc_velo_m$conf.int)$conf.level,
                               attributes(wilc_velo_f$conf.int)$conf.level,
                               attributes(wilc_v_wings_m$conf.int)$conf.level,
                               attributes(wilc_v_dist$conf.int)$conf.level))
  
  # Create array with the corresponding conditions
  test_res_grp <- append(test_res_grp,
                         c(rep(exp_grp, 6)))
  test_res_cond <- append(test_res_cond,
                          c(rep("m", 2), rep("mf", 4)))
  test_res_type <- append(test_res_type, c("change_velocity", 
                                             "change_wingangle", 
                                             "change_velocity",
                                             "change_velocity_f", 
                                             "change_wingangle", 
                                             "change_distance"))
}

# Combine p values and effect sizes in a data frame
all_test_res <- data.frame("condition" = test_res_cond, 
                           "group" = test_res_grp,
                           "chg_type" = test_res_type,
                           "pvalue" = test_res,
                           "conf_int" = test_res_confint,
                           "cover_prob" = test_res_covprob,
                           "eff_size" = eff_size)
# Save the test results (csv format)
write.csv(all_test_res,
          "E:/MT/additional_files/test_results_tracking_changes.csv",
          row.names = FALSE)
