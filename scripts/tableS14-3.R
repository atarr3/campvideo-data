library("readtext")
library("quanteda")
library("quanteda.sentiment")

# working directory
here::here("scripts", "tableS14-3.R")

# create corpus
transcripts <- corpus(readtext(here::here("data" ,"intermediate", "*", 
                                          "transcript.txt"), 
                               cache = FALSE))

# LSD
data("data_dictionary_LSD2015", package = "quanteda.sentiment")

# polarity-based sentiment
polarity(data_dictionary_LSD2015) <- list(pos = c("positive", "neg_negative"), 
                                          neg = c("negative", "neg_positive"))

sentiment <- transcripts %>% textstat_polarity(data_dictionary_LSD2015)

# predict ad tone w/ LSD
LSD.all <- data.frame(uid=sapply(sentiment$doc_id, dirname),
                      tone=as.numeric(sentiment$sentiment >= 0)
                     )
LSD.all <- LSD.all[order(LSD.all$uid), ]

# read in WMP data
tone.wmp <- read.csv(here::here("data", "wmp", "wmp_final.csv"), stringsAsFactors=F)
tone.wmp <- tone.wmp[order(tone.wmp$uid), ]

# drop "N/A" and "CONTRAST"
tone.wmp.sub <- tone.wmp[!(tone.wmp$tonecmag %in% c("N/A", "CONTRAST")), 
                         c("uid", "tonecmag")]

# subset LSD predictions
LSD <- LSD.all[LSD.all$uid %in% tone.wmp.sub$uid, "tone"]

# convert labels to 0/1
WMP <- as.integer(tone.wmp.sub$tonecmag %in% c("POSITIVE", "POS"))

# confusion matrix (full sample)
cm.full <- format(round(table(LSD, WMP) / length(LSD) * 100, 2), nsmall=2)

# read in predictions
tone.pred <- read.csv(here::here("results", "negativity_results.csv"), 
                      stringsAsFactors=F)

# subset to test set for non-linear SVM
tone.pred <- tone.pred[tone.pred$feature == "text" & tone.pred$model == "nsvm" &
                       tone.pred$train == 0, c("uid", "tone")]

# confusion matrix (test set)
cm.test <- format(round(table(LSD, WMP) / length(LSD) * 100, 2), nsmall=2)

# save
cat("LSD Results (Full)\n", file=here::here("tables", "tableS14-3.txt"))
cat("------------------\n", file=here::here("tables", "tableS14-3.txt"), append=T)
sink(file=here::here("tables", "tableS14-3.txt"), append=T)
print(cm.full)
sink(file=NULL)
cat("\n", file=here::here("tables", "tableS14-3.txt"), append=T)
cat("LSD Results (Test)\n", file=here::here("tables", "tableS14-3.txt"), append=T)
cat("------------------\n", file=here::here("tables", "tableS14-3.txt"), append=T)
sink(file=here::here("tables", "tableS14-3.txt"), append=T)
print(cm.test)
sink(file=NULL)
cat("\n", file=here::here("tables", "tableS14-3.txt"), append=T)