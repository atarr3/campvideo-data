library(readstata13)

# set up file location
here::i_am(file.path("scripts", "preprocess_CMAG.R"))

# function for selecting columns
subset.cmag <- function(data) {
  # election metadata columns
  meta <- c("creative", "year", "race", "category", "categorystate", 
            "affiliation", "cand_id", "tgt_id")
  issues <- colnames(data)[grep("issue[0-9]+(?!.*_txt)", colnames(data), perl=T)]
  other <- c("tonecmag", "vid", "f_mention", "f_picture", "o_mention", "o_picture",
             "prsment", "gbush", "reagan", "gophse", "demhse", "gopsen", "demsen",
             "congmt", "democrats", "republicans", "music0", "music1", "music2",
             "music3", "mention1", "mention2", "mention3", "mention4", "mention5",
             "mention6", "mention7", "mention8", "mention14", "mention15", 
             "mention16", "mention17")
  cols <- c(meta, other, issues)
  
  return(data[, cols])
}

# read in matches file
matches <- read.csv(here::here("data", "matches", "matches.csv"), 
                    stringsAsFactors=F)

# load WMP data and drop duplicates
hou12 <- read.dta13(here::here("data", "wmp", "wmp-house-2012-v1.1.dta"), 
                    convert.factors=F)
hou12 <- hou12[!duplicated(hou12$creative), ]
sen12 <- read.dta13(here::here("data", "wmp", "wmp-senate-2012-v1.1.dta"), 
                    convert.factors=F)
sen12 <- sen12[!duplicated(sen12$creative), ]
gov12 <- read.dta13(here::here("data", "wmp", "wmp-gov-2012-v1.1.dta"), 
                    convert.factors=F)
gov12 <- gov12[!duplicated(gov12$creative), ]
wmp12 <- rbind(hou12, sen12, gov12)

pre12 <- read.dta13(here::here("data", "wmp", "wmp-pres-2012-v1.2_compress.dta"), 
                    convert.factors=F)
pre12 <- pre12[!duplicated(pre12$creative), ]

hou14 <- read.dta13(here::here("data", "wmp", "wmp-house-2014-v1.0.dta"), 
                    convert.factors=F)
hou14 <- hou14[!duplicated(hou14$creative), ]
sen14 <- read.dta13(here::here("data", "wmp", "wmp-senate-2014-v1.0.dta"), 
                    convert.factors=F)
sen14 <- sen14[!duplicated(sen14$creative), ]
gov14 <- read.dta13(here::here("data", "wmp", "wmp-gov-2014-v1.1.dta"), 
                    convert.factors=F)
gov14 <- gov14[!duplicated(gov14$creative), ]
wmp14 <- rbind(hou14, sen14, gov14)

# add year variable
wmp12$year <- 2012
pre12$year <- 2012
wmp14$year <- 2014

# variable edit for 2012 data (`party` -> `affiliation`)
colnames(wmp12)[which(colnames(wmp12) == 'party')] <- 'affiliation'

# subset to columns used in project
wmp12.sub <- subset.cmag(wmp12)
pre12.sub <- subset.cmag(pre12)
wmp14.sub <- subset.cmag(wmp14)

# merge data frames
merged <- merge(rbind(wmp12.sub, pre12.sub), wmp14.sub, all=T)

# drop duplicates (some videos appear multiple times across different election cycles)
merged <- merged[!duplicated(merged$creative), ]

# copy data over to matches dataframe for future us in table1
matches.cols <- c("year", "race", "affiliation", "cand_id")
matches[, matches.cols] = merged[match(matches$creative, merged$creative), 
                                 matches.cols]

# fix some entries with cand_id = " "
matches[matches$creative == "HOUSE/NE02 TERRY&NRCC OBAMACARE",
                 "cand_id"] = "Terry_Lee"
matches[matches$creative == "HOUSE/NY01 DCCC&BISHOP TOXIC WASTE",
                 "cand_id"] = "Bishop_Tim"
matches[matches$creative == "HOUSE/NY24 DCCC&MAFFEI KRISTINA SHERMAN",
                 "cand_id"] = "Maffei_Dan"
matches[matches$creative == "HOUSE/TN04 TNDP&SHERRELL DOMESTIC VIOLENCE",
                 "cand_id"] = "Sherrell_Lenda"
matches[matches$creative == "HOUSE/WV02 NRCC&MOONEY LEAVE A MESSAGE",
                 "cand_id"] = "Mooney_Alex"
matches[matches$creative == "USSEN/NC HAGAN&DSCC IMPORTANT ISSUES",
                 "cand_id"] = "Hagan_Kay"
matches[matches$creative == "USSEN/NC HAGAN&DSCC RECORD",
                 "cand_id"] = "Hagan_Kay"

# subset merged to matched videos only
final.ind <- matches[!(matches$uid %in% c("NoMatch", "NoChannel")), ]$creative
final <- merged[merged$creative %in% final.ind, ]

# sort by `creative`
final <- final[order(final$creative), ]

# save processed files
write.csv(matches, here::here("data", "matches", "matches_processed.csv"), 
          row.names = F)
write.csv(final, here::here("data", "wmp", "wmp_final.csv"), row.names = F)