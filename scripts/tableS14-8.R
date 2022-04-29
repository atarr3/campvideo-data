library("lme4")
library("stargazer")

here::i_am(file.path("scripts", "tableS14-8.R"))

# read in regression data
data.full <- read.csv(here::here("data", "issueconv.csv"),
                      stringsAsFactors=F)

# split into predictions and wmp
data.pred <- data.full[data.full$wmp == 0, ]
data.wmp <- data.full[data.full$wmp == 1, ]

# perform regressions (random-effects model)
fit.pred <- lm(convergence ~ comp + total_disb_vep + diff_disb_vep + negative + 
                 log_vap + year2012 + consensual + owned + salience, data=data.pred)
fit.wmp <- lm(convergence ~ comp + total_disb_vep + diff_disb_vep + negative + 
                 log_vap + year2012 + consensual + owned + salience, data=data.wmp)

stargazer(fit.wmp, fit.pred, align=T, 
          dep.var.labels=c("Issue Convergence"),
          covariate.labels=c("Competitiveness", "Total Spending/VEP",
                             "Diff. in Spending/VEP", 
                             "% Negative Ads", "VAP (logged)","Year 2012",
                             "Consensual", "Owned", "Salience"),
          column.labels=c("WMP Coding", "Automated Coding"),
          intercept.bottom=T, keep.stat=c("n"), model.numbers=F, 
          out=here::here("tables", "tableS14-8.txt"))
