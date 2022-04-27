library("readtext")
library("quanteda")
library("quanteda.sentiment")

# working directory
setwd("E:/Users/Alex/Desktop/ITH_Matched")

# create corpus
transcripts <- corpus(readtext("../../OneDrive/transcripts/*.txt", cache = FALSE))

# LSD
data("data_dictionary_LSD2015", package = "quanteda.sentiment")

# polarity-based sentiment
polarity(data_dictionary_LSD2015) <- list(pos = c("positive", "neg_negative"), 
                                          neg = c("negative", "neg_positive"))

sentiment <- transcripts %>% textstat_polarity(data_dictionary_LSD2015)

# predict ad tone and save
tone <- as.numeric(sentiment$sentiment >= 0)
ids <- rep("", nrow(sentiment))

for (i in 1:length(ids)) {
  ids[i] <- sub(pattern = "(.*)\\..*$", replacement = "\\1", sentiment$doc_id[i])
}

df <- data.frame(uid = ids, tone = tone)

write.csv(df, "../lsd_tone_pp.csv", row.names=F)