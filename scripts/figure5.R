library(dplyr)

# set up root location
here::i_am(file.path("scripts", "figure5.R"))

# issue names
issues <- c('prsment','gbush','reagan','gophse','demhse','gopsen','demsen','congmt',
            'democrats','republicans','mention1','mention2','mention3','mention4',
            'mention5','mention6','mention7','mention8','mention14','mention15',
            'mention16','mention17','issue10','issue11','issue12','issue13',
            'issue14','issue15','issue16','issue17','issue18','issue19','issue20',
            'issue21','issue22','issue23','issue30','issue31','issue32','issue33',
            'issue34','issue35','issue36','issue37','issue38','issue39','issue40',
            'issue41','issue42','issue43','issue50','issue51','issue52','issue53',
            'issue54','issue55','issue56','issue57','issue60','issue61','issue62',
            'issue63','issue64','issue65','issue66','issue67','issue68','issue69',
            'issue70','issue71','issue72','issue80','issue82','issue83','issue84',
            'issue90','issue91','issue92','issue93','issue94','issue95','issue96',
            'issue98')

# read in WMP data
issue.wmp <- read.csv(here::here("data", "wmp", "wmp_final.csv"),
                      stringsAsFactors=F)

# merge issue30 (abortion) and issue58 (women's health)
issue.wmp$issue30 <- as.integer(issue.wmp$issue30 | issue.wmp$issue58)

# merge issue53 (healthcare) and issue59 (obamacare)
# NOTE: issue59 only labeled in 2014 data
issue.wmp$issue53 <- as.integer(issue.wmp$issue53 | issue.wmp$issue59)
issue.wmp$issue53[is.na(issue.wmp$issue53)] <- 0

# subset data to issue columns
issue.wmp <- issue.wmp[, c('creative', issues)]

# convert non-binary data to binary
issue.wmp[, 2:11] <- matrix(as.integer(!((issue.wmp[, 2:11] == 0) | 
                                         (issue.wmp[, 2:11] == '0') |
                                         (issue.wmp[, 2:11] == 'No'))
                                      ),
                            ncol=10)

# read in MTurk data
issue.mturk <- read.csv(here::here("data", "mturk", "issue_mturk.csv"), 
                        stringsAsFactors=F)
n = nrow(issue.mturk)

# populate MTurk dataframe with WMP labels
issue.mturk$wmp = 0

for (i in seq(1, n, 5)) {
  issue.mturk$wmp[i:(i+4)] <- issue.wmp[issue.wmp$creative == issue.mturk$creative[i], 
                                        issue.mturk$issue[i]]
}


# bar plots for MTurk agreement
# get relevant data
y.wmp <- issue.mturk$wmp[seq(1, n, 5)]
y.pred <- issue.mturk$pred[seq(1, n, 5)]
agree <- (issue.mturk %>% group_by(creative, issue) 
                      %>% summarise(total=sum(pred == mturk), .groups = 'drop'))$total

fname <- here::here("figs/figure5.pdf")

# set up figure
pdf(fname, width=8.5, height=5)
par(mfrow=c(2, 2), mar=c(3,4,2,2), las=1)

for (i in 0:1) {
  for (j in 0:1) {
    # plot titles
    if (i == 0 & j == 0) {
      title <- "Both WMP and Automated Codings Give No"
    } else if (i == 0 & j == 1) {
      title <- "Only Automated Coding Gives Yes"
    } else if (i == 1 & j == 0) {
      title <- "Only WMP Coding Gives Yes"
    } else {
      title <- "Both WMP and Automated Codings Give Yes"
    }
    # subset data
    sub <- agree[y.wmp == i & y.pred == j]
    
    # plot
    barplot(table(factor(sub, levels=0:5)) / length(sub) * 100, ylim=c(0, 100),
            main=title, axes=F)
    
    # y-axis
    axis(2, at=seq(0, 100, 20), labels=c(seq(0, 80, 20), "100 (%)"))
    
    # floating text
    text(2, 80, paste('No. of videos:', length(sub)), adj=0)
    text(2, 72, paste('Mean:', sprintf('%.2f', mean(sub))), adj=0)
    text(2, 64, paste('St. dev.:', sprintf('%.2f', sd(sub))), adj=0)
  }
}

# clear figure
dev.off()
