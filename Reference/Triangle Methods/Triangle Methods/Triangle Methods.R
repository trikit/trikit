setwd("//cna.com/shared/CAOA.W/_CARP/ReserveVariabilityProject/December 2016/Data/Loss Triangles")
library("ChainLadder")
library(moments)
library(xlsx)

################################## Mack Method ##########################################
# ######## One Group Mack Method Investigation #######
# 
# Group <- "S01"
# 
# dat_pd <- read.csv(paste("Paid Loss Triangle - ", Group, ".csv", sep = ""))
# rownames(dat_pd) <- dat_pd[,1]
# # remove the first column after making it as rownames
# dat_pd <- dat_pd[,-1]
# dat_pd[round(dat_pd,0) == 0] <- NA
# # rename the columns as development period so that the plots below will work
# colnames(dat_pd) <- seq(1, dim(dat_pd)[2], 1)
# # add $1 to the zero entries so that the Mack Method will run.
# #dat_pd <- dat_pd + 1
# MackOut <- MackChainLadder(dat_pd, est.sigma = "Mack", tail = TRUE)
# # investigate the output
# MackOut
# # Pull out the information needed for Mack Example:
# # loss development factors
# DevFac <- as.data.frame(MackOut$f)
# # sigma^2
# SigSq <- as.data.frame(MackOut$sigma^2)
# # Mack S.E.
# MackSE <- as.data.frame(MackOut$Mack.S.E)
# # Mack Output by AY
# MackbyAY <- as.data.frame(summary(MackOut)$ByOrigin)
# # Mack Output total
# MackTot <- as.data.frame(summary(MackOut)$Totals)
# # LogNormal Percentiles
# ResMean<- summary(MackOut)$Totals[4,]
# # calculate SE
# ResSD <- summary(MackOut)$Totals[5,]
# # calculate CV
# ResCV <- ResSD/ResMean
# # calculate sigma for LogNormal
# LogN_Sigma <- sqrt(log(1+ResCV^2))
# # calculate miu for LogNormal
# LogN_Miu <- log(ResMean) - (LogN_Sigma^2)/2
# LogN_Percentile <- as.data.frame(qlnorm(seq(0.05, 0.95, 0.05), LogN_Miu, LogN_Sigma))
# # Normal Quantile:
# Norm_Quantile <- as.data.frame(qnorm(seq(0.05, 0.95, 0.05), ResMean, ResSD))
# # Look at the full triangle
# MackOut$FullTriangle
# # check Mack assumptions by plotting
# plot(MackOut)
# # plot the development, including the forecase and estimated SE by AY
# plot(MackOut, lattice = TRUE)


Group <- c("S01", "S02" , "S03", "S04", "S05", "S06","S07", "S08", "S09", "S10", "S11",
           "S12", "S13", "S14", "S15", "S16", "S17", "S18", "S19", "S20", "S21", "S22",
           "S23", "S24", "S25", "S26")

# create a function to replace the lower triangle to NAs for Mack Method.
clean_tri <- function(tri){
  for (i in 2:30){
    tri[i,(30-i+2):30] <- NA
  }
}

# initialize the data frame to carry the final output of percentiles
LogN_Percentile <- data.frame(Percentiles = seq(0.05, 0.95, 0.05))
ResMean <- c()
ResSD <- c()
ResCV <- c()
LogN_Miu <- c()
LogN_Sigma <- c()

for (i in 1:length(Group)){
  
  dat_pd <- read.csv(paste("Paid Loss Triangle - ", Group[i], ".csv", sep = ""))
  # treat the first column as rownames so that the function can run on the data
  rownames(dat_pd) <- dat_pd[,1]
  # remove the first column after making it as rownames
  dat_pd <- dat_pd[,-1]
  # rename the columns as development period so that the plots below will work
  colnames(dat_pd) <- seq(1, dim(dat_pd)[2], 1)
  # change the zero entries to NA in order for Mack method to run
  dat_pd[round(dat_pd,0) == 0] <- NA
  # for the AYs that have the entire row NA, change them to 0.01. Otherwise, the Mack method won't run
  dat_pd[rowSums(is.na(dat_pd[1:30,]))==30,] <- 0.01
  # clean up the triangle by setting the lower triangle to NA.
  clean_tri(dat_pd)
  # Run Mack function: Group 6 & 11 flatten out in the matured ages, so not fitting a tail
  # ifelse(Group[i] %in% c("S06", "S11"), MackOut <- MackChainLadder(dat_pd, est.sigma = "Mack", tail = FALSE),
  #        MackOut <- MackChainLadder(dat_pd, est.sigma = "Mack", tail = TRUE))
  MackOut <- MackChainLadder(dat_pd, est.sigma = "Mack", tail = FALSE)
  # calculate the reserve
  ResMean[i]<- summary(MackOut)$Totals[4,]
  # calculate SE
  ResSD[i] <- summary(MackOut)$Totals[5,]
  # calculate CV
  ResCV[i] <- ResSD[i]/ResMean[i]
  # calculate sigma for LogNormal
  LogN_Sigma[i] <- sqrt(log(1+ResCV[i]^2))
  # calculate miu for LogNormal
  LogN_Miu[i] <- log(ResMean[i]) - (LogN_Sigma[i]^2)/2
  # calculate LogNormal quantiles
  LogN_Percentile[, i+1] <- qlnorm(seq(0.05, 0.95, 0.05), LogN_Miu[i], LogN_Sigma[i])
  colnames(LogN_Percentile)[i+1] <- Group[i]
  # # calculate Normal percentiles
  # Norm_Percentile[, i+1] <- qnorm(seq(0.05, 0.95, 0.05), ReserveMean, ReserveSE)
  # colnames(Norm_Percentile)[i+1] <- Group[i]
  print(paste("Finish running ", Group[i], sep = ""))
}

Mack_Output <- rbind(ResMean, ResSD, ResCV, LogN_Percentile[,-1])
row.names(Mack_Output) <- c("Mean", "Std Dev", "CV", seq(0.05, 0.95, 0.05))
write.csv(Mack_Output, "//cna.com/shared/CAOA.W/_CARP/ReserveVariabilityProject/December 2016/Triangle Methods/Mack Method Summary.csv")



################################## Bootstrap Method ##########################################

###### One group investigation ######
# Group <- "S01"
# dat_pd <- read.csv(paste("Paid Loss Triangle - ", Group, ".csv", sep = ""))
# dat_inc <- read.csv(paste("Incurred Loss Triangle - ", Group, ".csv", sep = ""))
# # treat the first column as rownames so that the function can run on the data
# rownames(dat_pd) <- dat_pd[,1]
# rownames(dat_inc) <- dat_inc[,1]
# # remove the first column after making it as rownames
# dat_pd <- dat_pd[,-1]
# dat_pd <- dat_pd + 1
# dat_inc <- dat_inc[,-1]
# dat_inc <- dat_inc + 1
# # rename the columns as development period so that the plots below will work
# colnames(dat_pd) <- seq(1, dim(dat_pd)[2], 1)
# colnames(dat_inc) <- seq(1, dim(dat_inc)[2], 1)
# # Run bootstrap function
# BootOut_pd <- BootChainLadder(dat_pd, R = 25000, process.distr = "gamma")
# BootOut_inc <- BootChainLadder(dat_inc, R = 25000, process.distr = "gamma")
# # Output information needed for the Bootstrap example:
# # LDF:
# DevFac <- as.data.frame(BootOut_inc$f)
# # Residuals:
# CLRes <- as.data.frame(BootOut_inc$ChainLadder.Residuals)
# # IBNR Triangle:
# IBNRTri <- as.data.frame(BootOut_inc$IBNR.Triangles[, , 1])
# # simulated claim:
# simClaim <- as.data.frame(BootOut_inc$simClaims[, , 1])
# # Incurred Bootstrap Output by AY
# Boot_Inc_byAY <- as.data.frame(summary(BootOut_inc)$ByOrigin)
# # Paid to date by AY
# Boot_pd_byAY <- as.data.frame(summary(BootOut_pd)$ByOrigin[,1])
# # standard error
# ResSD_inc <- summary(BootOut_inc)$Totals[4,]
# # calculate CV
# ResCV_inc <- ResSD_inc/ResMean_inc
# # calculate bootstrap quantiles
# Boot_Per_inc <- quantile(BootOut_inc, seq(0.05, 0.95, 0.05))$Totals + 
#   summary(BootOut_inc)$Totals[1,] - summary(BootOut_pd)$Totals[1,]



Group <- c("S01", "S02" , "S03", "S04", "S05", "S06","S07", "S08", "S09", "S10", "S11",
           "S12", "S13", "S14", "S15", "S16", "S17", "S18", "S19", "S20", "S21", "S22",
           "S23", "S24", "S25", "S26")

# initialize the data frame to carry the final output of percentiles
Boot_Per_pd <- data.frame(Percentiles = seq(0.05, 0.95, 0.05))
Boot_Per_inc <- data.frame(Percentiles = seq(0.05, 0.95, 0.05))
ResMean_pd <- c()
ResMean_inc <- c()
ResSD_pd <- c()
ResSD_inc <- c()
ResCV_pd <- c()
ResCV_inc <- c()

system.time(
  for (i in 1:length(Group)){
    set.seed(12345)
    print(paste("start group", Group[i]))
    dat_pd <- read.csv(paste("Paid Loss Triangle Bootstrap - ", Group[i], ".csv", sep = ""))
    dat_inc <- read.csv(paste("Incurred Loss Triangle Bootstrap - ", Group[i], ".csv", sep = ""))
    # treat the first column as rownames so that the function can run on the data
    rownames(dat_pd) <- dat_pd[,1]
    rownames(dat_inc) <- dat_inc[,1]
    # remove the first column after making it as rownames
    dat_pd <- dat_pd[,-1]
    dat_inc <- dat_inc[,-1]
    # rename the columns as development period so that the plots below will work
    colnames(dat_pd) <- seq(1, dim(dat_pd)[2], 1)
    colnames(dat_inc) <- seq(1, dim(dat_inc)[2], 1)
    # change the zero entries to NA in order for bootstrap to fit Gamma
    dat_pd[dat_pd == 0] <- NA
    dat_inc[dat_inc == 0] <- NA
    # Run bootstrap function
    BootOut_pd <- BootChainLadder(dat_pd, R = 25000, process.distr = "gamma")
    BootOut_inc <- BootChainLadder(dat_inc, R = 25000, process.distr = "gamma")
    # calculate the reserve
    ResMean_pd[i] <- summary(BootOut_pd)$Totals[2,] - summary(BootOut_pd)$Totals[1,] #Est. Ultiamte - Paid
    ResMean_inc[i] <- summary(BootOut_inc)$Totals[2,] - summary(BootOut_pd)$Totals[1,] 
                      # Est. Ult - Paid (paid from the paid bootstrap model)
    # # Monitor Output
    # print(paste("Incurred Ultimate:",summary(BootOut_inc)$Totals[2,]))
    # print(paste("Paid to Date:", summary(BootOut_pd)$Totals[1,]))
    # print(paste("Incurred Reserve Mean:",ResMean_inc[i]))
    
    # calculate SE
    ResSD_pd[i] <- summary(BootOut_pd)$Totals[4,]
    ResSD_inc[i] <- summary(BootOut_inc)$Totals[4,]
    # calculate CV
    ResCV_pd[i] <- ResSD_pd[i]/ResMean_pd[i]
    ResCV_inc[i] <- ResSD_inc[i]/ResMean_inc[i]
    # calculate bootstrap quantiles
    Boot_Per_pd[, i+1] <- quantile(BootOut_pd, seq(0.05, 0.95, 0.05))$Totals
    colnames(Boot_Per_pd)[i+1] <- Group[i]
    Boot_Per_inc[, i+1] <- quantile(BootOut_inc, seq(0.05, 0.95, 0.05))$Totals + 
                                    summary(BootOut_inc)$Totals[1,] - summary(BootOut_pd)$Totals[1,]
                                    # quantile for IBNR + Case (dervied by subtracting paid from incurred)
    colnames(Boot_Per_inc)[i+1] <- Group[i]
    rm(dat_pd, dat_inc, BootOut_inc, BootOut_pd)
  })

Boot_Output_pd <- rbind(ResMean_pd, ResSD_pd, ResCV_pd, Boot_Per_pd[,-1])
row.names(Boot_Output_pd) <- c("Mean", "Std Dev", "CV", seq(0.05, 0.95, 0.05))
write.csv(Boot_Output_pd, "//cna.com/shared/CAOA.W/_CARP/ReserveVariabilityProject/December 2016/Triangle Methods/Paid Bootstrap Method Summary.csv")

Boot_Output_inc <- rbind(ResMean_inc, ResSD_inc, ResCV_inc, Boot_Per_inc[,-1])
row.names(Boot_Output_inc) <- c("Mean", "Std Dev", "CV", seq(0.05, 0.95, 0.05))
write.csv(Boot_Output_inc, "//cna.com/shared/CAOA.W/_CARP/ReserveVariabilityProject/December 2016/Triangle Methods/Incurred Bootstrap Method Summary.csv")


