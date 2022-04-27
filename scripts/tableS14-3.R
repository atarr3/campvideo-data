library("readtext")
library("quanteda")
library("quanteda.sentiment")

# working directory
here::here("scripts/lsd.R")

# create corpus
transcripts <- corpus(readtext(here::here("data/intermediate/*/transcript.txt"), 
                               cache = FALSE))

# LSD
data("data_dictionary_LSD2015", package = "quanteda.sentiment")

# polarity-based sentiment
polarity(data_dictionary_LSD2015) <- list(pos = c("positive", "neg_negative"), 
                                          neg = c("negative", "neg_positive"))

sentiment <- transcripts %>% textstat_polarity(data_dictionary_LSD2015)

# predict ad tone w/ LSD
LSD <- as.numeric(sentiment$sentiment >= 0)
ids <- sapply(sentiment$doc_id, dirname)


# read in WMP data
tone.wmp <- read.csv(here::here("data/wmp/final.csv"), stringsAsFactors=F)
WMP <- tone.wmp[match(ids, tone.wmp$uid), "tonecmag"]

# confusion matrix
cm <- format(round(table(LSD, WMP) / length(LSD) * 100, 2), nsmall=2)

# save
write.table(cm, file=here::here("tables/tableS14-3.txt"))