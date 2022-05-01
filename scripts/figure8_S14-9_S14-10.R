# working directory check
here::i_am(file.path("scripts", "figure8_S14-9_S14-10.R"))

# read in WMP data
mood.wmp <- read.csv(here::here("data", "wmp", "wmp_final.csv"),
                     stringsAsFactors=F)

# read in MTurk data
mood.mturk <- read.csv(here::here("data", "mturk", "mood_mturk.csv"),
                       stringsAsFactors=F)
n = nrow(mood.mturk)

# subset WMP data down to MTurk sample
mood.wmp.sample <- mood.wmp[match(unique(mood.mturk$creative),
                                  mood.wmp$creative), ]

# sort data by uid for compatibility with tapply
mood.mturk <- mood.mturk[order(mood.mturk$uid), ]
mood.wmp.sample <- mood.wmp.sample[order(mood.wmp.sample$uid), ]

# MTurk agreement with automated prediction indicator
agree.music1.pred <- mood.mturk$music1_mturk == mood.mturk$music1_pred
agree.music2.pred <- mood.mturk$music2_mturk == mood.mturk$music2_pred
agree.music3.pred <- mood.mturk$music3_mturk == mood.mturk$music3_pred

# agreement counts, note tapply sorts according to uid
agree.music1.counts <- tapply(agree.music1.pred, mood.mturk$uid, sum)
agree.music2.counts <- tapply(agree.music2.pred, mood.mturk$uid, sum)
agree.music3.counts <- tapply(agree.music3.pred, mood.mturk$uid, sum)

# bar plots for MTurk agreement
for (t in 1:3){
  # get relevant data
  if (t == 1) {
    y.wmp <- mood.wmp.sample$music1
    y.pred <- mood.mturk$music1_pred[seq(1, n, 5)]
    agree <- agree.music1.counts
    fname <- here::here("figs", "figure8.pdf")
  } else if (t == 2) {
    y.wmp <- mood.wmp.sample$music2
    y.pred <- mood.mturk$music2_pred[seq(1, n, 5)]
    agree <- agree.music2.counts
    fname <- here::here("figs", "figureS14-9.pdf")
  } else {
    y.wmp <- mood.wmp.sample$music3
    y.pred <- mood.mturk$music3_pred[seq(1, n, 5)]
    agree <- agree.music3.counts
    fname <- here::here("figs", "figureS14-10.pdf")
  }
  
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
      barplot(table(factor(sub, levels=0:5)) / nrow(sub) * 100, ylim=c(0, 100),
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
}
