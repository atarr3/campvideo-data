suppressPackageStartupMessages(library("quanteda"))
library("quanteda.sentiment", warn.conflicts = FALSE)

# root directory
ROOT = ".."

# read in transcripts
features <- read.csv(file.path(ROOT, "data", "auxiliary", "features_full.csv"),
                     stringsAsFactors=F)

# create corpus
transcripts <- corpus(features$transcript, docnames=features$uid)

# LSD
data("data_dictionary_LSD2015", package = "quanteda.sentiment")

# polarity-based sentiment
polarity(data_dictionary_LSD2015) <- list(pos = c("positive", "neg_negative"), 
                                          neg = c("negative", "neg_positive"))

sentiment <- transcripts %>% textstat_polarity(data_dictionary_LSD2015)

# predict ad tone w/ LSD
LSD.all <- data.frame(uid=sentiment$doc_id,
                      tone=as.numeric(sentiment$sentiment >= 0)
                     )
LSD.all <- LSD.all[order(LSD.all$uid), ]

# read in WMP data
tone.wmp <- read.csv(file.path(ROOT, "data", "wmp", "wmp_final.csv"), stringsAsFactors=F)
tone.wmp <- tone.wmp[order(tone.wmp$uid), ]

# drop "N/A" and "CONTRAST"
tone.wmp.sub <- tone.wmp[!(tone.wmp$tonecmag %in% c("N/A", "CONTRAST")), 
                         c("uid", "tonecmag")]

# subset LSD predictions
LSD <- LSD.all[LSD.all$uid %in% tone.wmp.sub$uid, "tone"]

# convert labels to 0/1
WMP <- as.integer(tone.wmp.sub$tonecmag %in% c("POSITIVE", "POS"))

# confusion matrix (full sample)
cm.full <- format(round(table(WMP, LSD) / length(LSD) * 100, 2), nsmall=2)

# read in predictions
tone.pred <- read.csv(file.path(ROOT, "data", "negativity_results.csv"), 
                      stringsAsFactors=F)

# subset to test set for non-linear SVM
tone.pred <- tone.pred[tone.pred$feature == "text" & tone.pred$model == "nsvm" &
                       tone.pred$train == 0, c("uid", "tone")]

# confusion matrix (test set)
LSD <- LSD.all[match(tone.pred$uid, LSD.all$uid), "tone"]
WMP <- WMP[match(tone.pred$uid, tone.wmp.sub$uid)]
cm.test <- format(round(table(WMP, LSD) / length(LSD) * 100, 2), nsmall=2)

# confusion matrix (our)
AUTO <- tone.pred$tone
cm.ours <- format(round(table(WMP, AUTO) / length(LSD) * 100, 2), nsmall=2)

# save
cat("LSD Results (Full)\n", file=file.path(ROOT, "results", "tables", "tableS14-7.txt")) # overwrites existing file
sink(file=file.path(ROOT, "results", "tables", "tableS14-7.txt"), append=T)
cat("------------------\n")
print(cm.full)
cat("\n")
cat("LSD Results (Test)\n")
cat("------------------\n")
print(cm.test)
cat("\n")
cat("LSD Results (Ours)\n")
cat("------------------\n")
print(cm.ours)
sink(file=NULL)