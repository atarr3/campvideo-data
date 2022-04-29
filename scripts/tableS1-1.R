library(stringr)
library(xtable)

# set up file location
here::i_am(file.path("scripts", "tableS1-1.R"))

# video/candidate counts for all videos
CMAGCover <- function(df, year=NULL, race=NULL, affiliation=NULL) {
  # null conditions
  if (is.null(year)) {year <- unique(df$year)}
  if (is.null(race)) {race <- unique(df$race)}
  if (is.null(affiliation)) {affiliation <- unique(df$affiliation)}
  
  # subset data
  sub <- df[df$year %in% year & df$race %in% race & df$affiliation %in% affiliation, ]
  
  # get video count
  sub.vidcount <- length(sub$creative)
  # get candidate count
  sub.candcount <- length(unique(sub$cand_id))
  
  # output
  out <- c(vid=sub.vidcount, can=sub.candcount)
  
  return(out)
}

# function that returns raw coverage
RawCover <- function(df, race=NULL){
  # null condition
  if (is.null(race)) {race <- unique(df$race)}
  
  # number of candidates = number of channels
  chan <- nrow(unique(df[, c("yt_district", "candidate")]))
  
  # subset data
  sub <- df[df$race %in% race, ]
  
  # video counts by length
  vid15 <- nrow(df[(df$length >= 10) & (df$length <= 20), ])
  vid30 <- nrow(df[(df$length >= 25) & (df$length <= 35), ])
  vid60 <- nrow(df[(df$length >= 55) & (df$length <= 65), ])
  vidall <- sum(vid15, vid30, vid60)
  
  # build row and return
  out <- c(chan, vid15, vid30, vid60, vidall)
  return(out)
}

# read in matches
matches <- read.csv(here::here("data", "matches", "matches_processed.csv"), 
                    stringsAsFactors=F)

# read in YT metadata
yt.info <- read.csv(here::here("data", "matches", "ytinfo.csv"),
                    stringsAsFactors=F) 

# create video-candidate coverage table
canvid.cmag <- rbind(CMAGCover(matches, race='PRESIDENT'),
                     CMAGCover(matches, year=2012, race='US HOUSE'),
                     CMAGCover(matches, year=2012, race='US SENATE'),
                     CMAGCover(matches, year=2012, race='GOVERNOR'),
                     CMAGCover(matches, year=2014, race='US HOUSE'),
                     CMAGCover(matches, year=2014, race='US SENATE'),
                     CMAGCover(matches, year=2014, race='GOVERNOR')
                    )


# ## create unique ids to subset raw videos by WMP records
# ytid <- paste(substr(yt.info$race, 4, 7), yt.info$yt_district, yt.info$candidate, sep = '_')
# 
# ## function to recover last names of candidates
# recover.candname <- function(df){
#   out <- rep(NA, nrow(df))
#   partyind <- df$affiliation
#   out[partyind == 'REPUBLICAN'] <- sapply(str_split(df$rep[partyind == 'REPUBLICAN'], ','), head, 1)
#   out[partyind == 'DEMOCRAT'] <- sapply(str_split(df$dem[partyind == 'DEMOCRAT'], ','), head, 1)
#   out[is.na(out)] <- sapply(str_split(df$third[is.na(out)], ','), head, 1)
#   return(out)
# } 
# 
# ## candidate identifier database to serve as reference
# cmid12 <- unique(paste(2012,
#                        dm12$cdmatch,
#                        recover.candname(dm12), sep = '_'))
# cmid12p <- unique(paste(2012,
#                         dm12p$cdmatch,
#                         recover.candname(dm12p), sep = '_'))
# cmid14 <- unique(paste(2014,
#                        dm14$cdmatch,
#                        recover.candname(dm14), sep = '_'))
# cmid <- c(cmid12, cmid12p, cmid14)
# 
# ## limit candidates to those reported by the CMAG
# cmindex <- ytid %in% cmid
# yt.info <- yt.info[cmindex, ]
# ytid <- ytid[cmindex]

rawytnum <- rbind(RawCover(yt.info[yt.info$race == 'pre2012', ]),
                  RawCover(yt.info[yt.info$race == 'hou2012', ]),
                  RawCover(yt.info[yt.info$race == 'sen2012', ]),
                  RawCover(yt.info[yt.info$race == 'gov2012', ]),
                  RawCover(yt.info[yt.info$race == 'hou2014', ]),
                  RawCover(yt.info[yt.info$race == 'sen2014', ]),
                  RawCover(yt.info[yt.info$race == 'gov2014', ]))

## appendix table 1 configuration
rawytyear <- c(rep(2012, 4), rep(2014, 3))
rawytrace <- c('President', 'House', 'Senate', 'Governor', 'House', 'Senate', 'Governor')
rawyttab <- as.data.frame(cbind(rawytyear, rawytrace,
                                canvid.cmag$can,
                                rawytnum[, 1],
                                round(rawytnum[, 1] / canvid.cmag$can * 100, 1),
                                rawytnum[, -1]))
colnames(rawyttab) <- c('Year', 'Office', 'All Candidates', 'Found Channels', 'Percentage', '15-sec.', '30-sec.', '60-sec.', 'All Videos')
rawyttab <- rbind(rawyttab, c('', 'Total', colSums(apply(rawyttab[, -c(1:2)], 2, as.numeric))))
rawyttab$Percentage[8] <- round(as.numeric(rawyttab$`Found Channels`[8]) / as.numeric(rawyttab$`All Candidates`[8]), 3) * 100
print(xtable(rawyttab, caption = 'Summary of Channels and Videos Recovered from YouTube'),
      caption.placement = 'top', include.rownames = F, digits = c(0, 0, 0, 0, 1, 0, 0, 0, 0))