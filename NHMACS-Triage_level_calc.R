# Header ----------------------------------------------------------------------------
#
# Tom Hartka, MD, MS
# Created: 9/20/2017
#
# Data set for ML test using NHAMCS 2014 data.  Determining 
# "correct" triage level for patients.
#
#

# Importing and view data structure ----------------------------------------------------------

# set working directory
setwd("~/Research/Triage-Machine Learning/NHAMC")

# read in csv with NHAMCS 2014 data, previously exported from STATA in text form
NHAMCS2014_text <- read.csv("ed2014_text_data.csv", stringsAsFactors = FALSE)
View(NHAMCS2014_text)

# read in STATA file 
library(haven)
NHAMCS2014 <- read_dta("~/Research/Triage-Machine Learning/NHAMC/ed2014-stata.dta")
View(NHAMCS2014)

# database structure
str(NHAMCS2014)
summary(NHAMCS2014)
names(NHAMCS2014)
nrow(NHAMCS2014)
ncol(NHAMCS2014)
dim(NHAMCS2014)

# head and tail
head(NHAMCS2014_text)
tail(NHAMCS2014_text, n = 2)


# Examine data ------------------------------------------------------------

# access age
NHAMCS2014$AGE
summary(NHAMCS2014$AGE)
hist(NHAMCS2014$AGE)
str(NHAMCS2014$AGE)

# find age over 85 years
subset(NHAMCS2014, AGE > 85)
View(subset(NHAMCS2014, AGE > 85))

# show hispanic patients by text or code
View(subset(NHAMCS2014_text, ETHUN == "Hispanic or Latino"))
View(subset(NHAMCS2014, ETHUN == 1))

# show black/hispanics by text
View(subset(NHAMCS2014_text, ETHUN == "Hispanic or Latino" & RACEUN == "Black/African American Only"))
  
# show patients with CC of mouth bleeding, eye discharge, or ankle swelling
View(subset(NHAMCS2014_text, RFV1 %in% c("Mouth bleeding","Discharge from eye","Swelling of ankle")))
  
# subset AGE and CC for patients over 90
OVER90ADULTS <- subset(NHAMCS2014_text, AGE > 90 & AGE != "Under one year", c(AGE, RFV1))
View(OVER90ADULTS)
remove(OVER90ADULTS)

# subset inclusive columns for patients over 90
View(subset(NHAMCS2014_text, AGE > 90 & AGE != "Under one year", VMONTH:AGER))

# create over 90 year flag
NHAMCS2014$OVER90YR <- ifelse(NHAMCS2014$AGE > 90, 1, 0)
View(subset(NHAMCS2014, OVER90YR == 1)) 

# create seen time, first make integers out of strings then add
#   -required removing leading 0's to work properly, extra code added
NHAMCS2014$ARRTIMEINT <- strtoi(substr(NHAMCS2014$ARRTIME,regexpr("[^0]",NHAMCS2014$ARRTIME),nchar(NHAMCS2014$ARRTIME)), base = 0L)
NHAMCS2014$WAITTIMEINT <- strtoi(substr(NHAMCS2014$WAITTIME,regexpr("[^0]",NHAMCS2014$WAITTIME),nchar(NHAMCS2014$WAITTIME)), base = 0L)

NHAMCS2014$SEENTIMEINT <- NHAMCS2014$ARRTIMEINT + NHAMCS2014$WAITTIMEINT
View(subset(NHAMCS2014, OVER90YR == 1, c(AGE, RFV1, ARRTIME, ARRTIMEINT, WAITTIME, WAITTIMEINT, SEENTIMEINT))) 

# remove column
NHAMCS2014$SEENTIMEINT <- NULL

# view age data characteristics
table(NHAMCS2014$AGE)
mean(NHAMCS2014$AGE)
summary(NHAMCS2014$AGE)

# triage level 1 characteristics
table(NHAMCS2014$DIEDED)
table(NHAMCS2014$CPR)
table(NHAMCS2014$ENDOINT)
table(NHAMCS2014$BPAP)
table(NHAMCS2014$DOA)
table(NHAMCS2014$ADMIT == 2)

#triage level
table(NHAMCS2014$IMMEDR)


# Create .csv with Variables available at time of triage -----------------------------------
NHAMCS2014_TriageVars <- NHAMCS2014[,0] 
NHAMCS2014_TriageVars$VMONTH <- NHAMCS2014$VMONTH
NHAMCS2014_TriageVars$VDAYR <- NHAMCS2014$VDAYR
NHAMCS2014_TriageVars$ARRTIME <- NHAMCS2014$ARRTIME
NHAMCS2014_TriageVars$AGE <- NHAMCS2014$AGE
NHAMCS2014_TriageVars$AGER <- NHAMCS2014$AGER
NHAMCS2014_TriageVars$AGEDAYS <- NHAMCS2014$AGEDAYS
NHAMCS2014_TriageVars$RESIDNCE <- NHAMCS2014$RESIDNCE
NHAMCS2014_TriageVars$SEX <- NHAMCS2014$SEX
NHAMCS2014_TriageVars$ETHIM <- NHAMCS2014$ETHIM
NHAMCS2014_TriageVars$ETHUN <- NHAMCS2014$ETHUN
NHAMCS2014_TriageVars$RACEUN <- NHAMCS2014$RACEUN
NHAMCS2014_TriageVars$RACER <- NHAMCS2014$RACER
NHAMCS2014_TriageVars$RACERETH <- NHAMCS2014$RACERETH
NHAMCS2014_TriageVars$ARREMS <- NHAMCS2014$ARREMS
NHAMCS2014_TriageVars$AMBTRANSFER <- NHAMCS2014$AMBTRANSFER
NHAMCS2014_TriageVars$TEMPF <- NHAMCS2014$TEMPF
NHAMCS2014_TriageVars$PULSE <- NHAMCS2014$PULSE
NHAMCS2014_TriageVars$RESPR <- NHAMCS2014$RESPR
NHAMCS2014_TriageVars$BPSYS <- NHAMCS2014$BPSYS
NHAMCS2014_TriageVars$BPDIAS <- NHAMCS2014$BPDIAS
NHAMCS2014_TriageVars$POPCT <- NHAMCS2014$POPCT
NHAMCS2014_TriageVars$IMMEDR <- NHAMCS2014$IMMEDR #this is the triage level, not including
NHAMCS2014_TriageVars$PAINSCALE <- NHAMCS2014$PAINSCALE
NHAMCS2014_TriageVars$SEEN72 <- NHAMCS2014$SEEN72
NHAMCS2014_TriageVars$RFV1 <- NHAMCS2014$RFV1
NHAMCS2014_TriageVars$RFV2 <- NHAMCS2014$RFV2
NHAMCS2014_TriageVars$RFV3 <- NHAMCS2014$RFV3
NHAMCS2014_TriageVars$RFV4 <- NHAMCS2014$RFV4
NHAMCS2014_TriageVars$RFV5 <- NHAMCS2014$RFV5
NHAMCS2014_TriageVars$RFV13D <- NHAMCS2014$RFV13D
NHAMCS2014_TriageVars$RFV23D <- NHAMCS2014$RFV23D
NHAMCS2014_TriageVars$RFV33D <- NHAMCS2014$RFV33D
NHAMCS2014_TriageVars$RFV43D <- NHAMCS2014$RFV43D
NHAMCS2014_TriageVars$RFV53D <- NHAMCS2014$RFV53D
NHAMCS2014_TriageVars$EPISODE <- NHAMCS2014$EPISODE
NHAMCS2014_TriageVars$INJURY <- NHAMCS2014$INJURY
NHAMCS2014_TriageVars$INJR1 <- NHAMCS2014$INJR1
NHAMCS2014_TriageVars$INJR2 <- NHAMCS2014$INJR2
NHAMCS2014_TriageVars$INJPOISAD <- NHAMCS2014$INJPOISAD
NHAMCS2014_TriageVars$INJPOISADR1 <- NHAMCS2014$INJPOISADR1
NHAMCS2014_TriageVars$INJPOISADR2 <- NHAMCS2014$INJPOISADR2
NHAMCS2014_TriageVars$INJURY72 <- NHAMCS2014$INJURY72
NHAMCS2014_TriageVars$INTENT <- NHAMCS2014$INTENT
NHAMCS2014_TriageVars$INJDETR <- NHAMCS2014$INJDETR
NHAMCS2014_TriageVars$INJDETR1 <- NHAMCS2014$INJDETR1
NHAMCS2014_TriageVars$INJDETR2 <- NHAMCS2014$INJDETR2
NHAMCS2014_TriageVars$CANCER <- NHAMCS2014$CANCER
NHAMCS2014_TriageVars$ETOHAB <- NHAMCS2014$ETOHAB
NHAMCS2014_TriageVars$ALZHD <- NHAMCS2014$ALZHD
NHAMCS2014_TriageVars$CEBVD <- NHAMCS2014$CEBVD
NHAMCS2014_TriageVars$CKD <- NHAMCS2014$CKD
NHAMCS2014_TriageVars$COPD <- NHAMCS2014$COPD
NHAMCS2014_TriageVars$CHF <- NHAMCS2014$CHF
NHAMCS2014_TriageVars$CAD <- NHAMCS2014$CAD
NHAMCS2014_TriageVars$DEPRN <- NHAMCS2014$DEPRN
NHAMCS2014_TriageVars$DIABTYP1 <- NHAMCS2014$DIABTYP1
NHAMCS2014_TriageVars$DIABTYP2 <- NHAMCS2014$DIABTYP2
NHAMCS2014_TriageVars$DIABTYP0 <- NHAMCS2014$DIABTYP0
NHAMCS2014_TriageVars$ESRD <- NHAMCS2014$ESRD
NHAMCS2014_TriageVars$HPE <- NHAMCS2014$HPE
NHAMCS2014_TriageVars$EDHIV <- NHAMCS2014$EDHIV
NHAMCS2014_TriageVars$HYPLIPID <- NHAMCS2014$HYPLIPID
NHAMCS2014_TriageVars$HTN <- NHAMCS2014$HTN
NHAMCS2014_TriageVars$OBESITY <- NHAMCS2014$OBESITY
NHAMCS2014_TriageVars$OSA <- NHAMCS2014$OSA
NHAMCS2014_TriageVars$OSTPRSIS <- NHAMCS2014$OSTPRSIS
NHAMCS2014_TriageVars$SUBSTAB <- NHAMCS2014$SUBSTAB
NHAMCS2014_TriageVars$NOCHRON <- NHAMCS2014$NOCHRON
NHAMCS2014_TriageVars$TOTCHRON <- NHAMCS2014$TOTCHRON

# Determine calculated ESI triage level ---------------------------------------

# Calculate resources used per patient
# create clean fields (ie -9 is changed to 0)
NHAMCS2014$TOTDIAGC <- ifelse(NHAMCS2014$TOTDIAG == -9, 0, NHAMCS2014$TOTDIAG)
NHAMCS2014$TOTPROCC <- ifelse(NHAMCS2014$TOTPROC == -9, 0, NHAMCS2014$TOTPROC)
NHAMCS2014$MEDGIVEDC <- ifelse(NHAMCS2014$MEDGIVED == -9, 0, NHAMCS2014$MEDGIVED)
NHAMCS2014$RESOURCES <- (NHAMCS2014$TOTDIAGC + NHAMCS2014$TOTPROCC + NHAMCS2014$MEDGIVEDC)
table(NHAMCS2014$RESOURCES)

# Clean vital signs
NHAMCS2014$PULSEC <- ifelse(NHAMCS2014$PULSE == -9, 90, NHAMCS2014$PULSE)
NHAMCS2014$RESPRC <- ifelse(NHAMCS2014$RESPR == -9, 90, NHAMCS2014$RESPR)
NHAMCS2014$POPCTC <- ifelse(NHAMCS2014$POPCT == -9, 90, NHAMCS2014$POPCT)

# Determine VS abnormalities based on age and set flag
NHAMCS2014$ABNORMALVS <- 0
NHAMCS2014$ABNORMALVS <- ifelse(NHAMCS2014$AGE>8 & (NHAMCS2014$PULSEC >100 | NHAMCS2014$RESPRC >20 | NHAMCS2014$POPCTC < 92),1,NHAMCS2014$ABNORMALVS)
NHAMCS2014$ABNORMALVS <- ifelse((NHAMCS2014$AGE %in% c(3:8))& (NHAMCS2014$PULSEC >140 | NHAMCS2014$RESPRC >30 | NHAMCS2014$POPCTC < 92),1,NHAMCS2014$ABNORMALVS)
NHAMCS2014$ABNORMALVS <- ifelse((NHAMCS2014$AGE %in% c(1:3) | NHAMCS2014$AGEDAYS>90)& (NHAMCS2014$PULSEC >160 | NHAMCS2014$RESPRC >40 | NHAMCS2014$POPCTC < 92),1,NHAMCS2014$ABNORMALVS)
NHAMCS2014$ABNORMALVS <- ifelse((NHAMCS2014$AGEDAYS %in% c(1:3)) & (NHAMCS2014$PULSEC >180 | NHAMCS2014$RESPRC >50 | NHAMCS2014$POPCTC < 92),1,NHAMCS2014$ABNORMALVS)
NHAMCS2014_TriageVars$ABNORMALVS <- NHAMCS2014$ABNORMALVS
table(NHAMCS2014$ABNORMALVS)

# Find triage level 5s
# These patient should have no resources used
# Diagnostic services ordered [DIAGSCRN] = 0
# AND Procedures performed [PROC] = 0
# AND Number of meds given in ED [NUMGIV] = 0
NHAMCS2014_TriageVars$TRIAGELEVEL <- ifelse(NHAMCS2014$RESOURCES == 0, 5, -1)
table(NHAMCS2014_TriageVars$TRIAGELEVEL)

# Find triage level 4s
# Total diagnostic procedures [TOTDIAG] + Total procedures [TOTPROC] + Med given in ed [MEDGIVED] = 1
# Need to calculate MEDGIVED.  If any med is given this flag is 1, otherwise 0
NHAMCS2014$MEDGIVED <- ifelse(NHAMCS2014$NUMGIV > 0, 1, 0)
NHAMCS2014_TriageVars$TRIAGELEVEL <- ifelse(NHAMCS2014$RESOURCES == 1, 4, NHAMCS2014_TriageVars$TRIAGELEVEL) 
table(NHAMCS2014_TriageVars$TRIAGELEVEL)

# Assign triage level 3s to all visits with more than 1 resources used
NHAMCS2014_TriageVars$TRIAGELEVEL <- ifelse(NHAMCS2014$RESOURCES > 1, 3, NHAMCS2014_TriageVars$TRIAGELEVEL) 
table(NHAMCS2014_TriageVars$TRIAGELEVEL)

# Move up those with VS abnormalities to level 2
# any patient admitted to the ICU, OR, or cath lab
# intubated/BiPAP also should be at least 2
# ESI says pain level >7
NHAMCS2014_TriageVars$TRIAGELEVEL <- ifelse(NHAMCS2014$ABNORMALVS == 1, 2, NHAMCS2014_TriageVars$TRIAGELEVEL) 
NHAMCS2014_TriageVars$TRIAGELEVEL <- ifelse(NHAMCS2014$ADMIT %in% c(1,3,5), 2, NHAMCS2014_TriageVars$TRIAGELEVEL) 
NHAMCS2014_TriageVars$TRIAGELEVEL <- ifelse(NHAMCS2014$ENDOINT == 1, 2, NHAMCS2014_TriageVars$TRIAGELEVEL) 
table(NHAMCS2014$PAINSCALE)

# Move up those with certain conidtions to level 1
# These include patient who died or need CPR
NHAMCS2014_TriageVars$TRIAGELEVEL <- ifelse(NHAMCS2014$CPR == 1, 1, NHAMCS2014_TriageVars$TRIAGELEVEL) 
NHAMCS2014_TriageVars$TRIAGELEVEL <- ifelse(NHAMCS2014$DOA == 1, 1, NHAMCS2014_TriageVars$TRIAGELEVEL) 
NHAMCS2014_TriageVars$TRIAGELEVEL <- ifelse(NHAMCS2014$DIEDED == 1, 1, NHAMCS2014_TriageVars$TRIAGELEVEL) 

# Examine overall trends
table(NHAMCS2014_TriageVars$TRIAGELEVEL)
hist(NHAMCS2014_TriageVars$TRIAGELEVEL)
table(NHAMCS2014$IMMEDR)
hist(NHAMCS2014$IMMEDR)
table(NHAMCS2014$IMMEDR, NHAMCS2014_TriageVars$TRIAGELEVEL)
table(NHAMCS2014$IMMEDR, NHAMCS2014$DIEDED)

# Write out files ---------------------------------------------------------

#remove(NHAMCS2014_TriageVars)  #remove to reload triage file
write.csv(NHAMCS2014_TriageVars, "NHAMCS2014_TriageVars.csv")
write.csv(NHAMCS2014, "NHAMCS2014.csv")
write.csv(NHAMCS2014_text, "NHAMCS2014_text.csv")

