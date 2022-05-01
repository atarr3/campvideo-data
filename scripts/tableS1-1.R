library(xtable)

# working directory check
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

# function that returns YouTube coverage, broken down by video length
YoutubeCover <- function(df, race=NULL){
  # null condition
  if (is.null(race)) {race <- unique(df$race)}
  
  # subset data
  sub <- df[df$race %in% race, ]
  
  # number of candidates = number of channels
  chan <- nrow(unique(sub[, c("yt_district", "candidate")]))
  
  # video counts by length
  vid15 <- sum(sub$length >= 10 & sub$length <= 20)
  vid30 <- sum(sub$length >= 25 & sub$length <= 35)
  vid60 <- sum(sub$length >= 55 & sub$length <= 65)
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
canvid.cmag <- as.data.frame(canvid.cmag)

# create video-length coverage table
canvid.yt <- rbind(YoutubeCover(yt.info, 'pre2012'),
                   YoutubeCover(yt.info, 'hou2012'),
                   YoutubeCover(yt.info, 'sen2012'),
                   YoutubeCover(yt.info, 'gov2012'),
                   YoutubeCover(yt.info, 'hou2014'),
                   YoutubeCover(yt.info, 'sen2014'),
                   YoutubeCover(yt.info, 'gov2014')
                  )
canvid.yt <- as.data.frame(canvid.yt)

## appendix table 1 configuration
yt.year <- c(rep(2012, 4), rep(2014, 3))
yt.race <- c('President', 'House', 'Senate', 'Governor', 'House', 'Senate', 'Governor')
yt.tab <- as.data.frame(cbind(yt.year, yt.race,
                                canvid.cmag$can,
                                canvid.yt[, 1],
                                round(canvid.yt[, 1] / canvid.cmag$can * 100, 1),
                                canvid.yt[, -1]))
colnames(yt.tab) <- c('Year', 'Office', 'All Candidates', 'Found Channels', 'Percentage', '15-sec.', '30-sec.', '60-sec.', 'All Videos')
yt.tab <- rbind(yt.tab, c('', 'Total', colSums(apply(yt.tab[, -c(1:2)], 2, as.numeric))))
yt.tab$Percentage[8] <- round(as.numeric(yt.tab$`Found Channels`[8]) / as.numeric(yt.tab$`All Candidates`[8]), 3) * 100

# print output to tableS1-1.txt
print(xtable(yt.tab, 
             caption = 'Summary of Channels and Videos Recovered from YouTube',
             digits = c(0, 0, 0, 0, 0, 1, 0, 0, 0, 0)),
      caption.placement = 'top', include.rownames = F,
      file=here::here("tables", "tableS1-1.txt"))