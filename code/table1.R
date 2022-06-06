options(xtable.comment = FALSE)
library(xtable)

# root directory
ROOT = ".."

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

# video/candidate counts for matched YouTube videos
YTCover <- function(df, year=NULL, race=NULL, affiliation=NULL) {
  # null conditions
  if (is.null(year)) {year <- unique(df$year)}
  if (is.null(race)) {race <- unique(df$race)}
  if (is.null(affiliation)) {affiliation <- unique(df$affiliation)}
  
  # subset data
  sub <- df[df$year %in% year & df$race %in% race & df$affiliation %in% affiliation, ]
  
  # total videos for found channels
  sub.chanvid <- length(sub$creative[sub$uid != 'NoChannel'])
  # total candidates for found channels
  sub.chancan <- length(unique(sub$cand_id[sub$uid != 'NoChannel']))
  # total acquired videos
  sub.acqvid <- length(sub$creative[!(sub$uid %in% c('NoChannel', 'NoMatch'))])
  # total candidates for acquired videos
  sub.acqcan <- length(unique(sub$cand_id[!(sub$uid %in% c('NoChannel', 'NoMatch'))]))
  
  # output
  out <- c(chanvid=sub.chanvid, chancan=sub.chancan,
           acqvid=sub.acqvid, acqcan=sub.acqcan)
  
  return(out)
}

# read in matches
matches <- read.csv(file.path(ROOT, "data", "auxiliary", "matches_processed.csv"), 
                    stringsAsFactors=F)

# video-candidate coverage table
canvid.cmag <- rbind(CMAGCover(matches, race='PRESIDENT'),
                     CMAGCover(matches, year=2012, race='US HOUSE'),
                     CMAGCover(matches, year=2012, race='US SENATE'),
                     CMAGCover(matches, year=2012, race='GOVERNOR'),
                     CMAGCover(matches, year=2014, race='US HOUSE'),
                     CMAGCover(matches, year=2014, race='US SENATE'),
                     CMAGCover(matches, year=2014, race='GOVERNOR')
                    )
canvid.cmag <- as.data.frame(canvid.cmag)

canvid.yt <- rbind(YTCover(matches, race='PRESIDENT'),
                   YTCover(matches, year=2012, race='US HOUSE'),
                   YTCover(matches, year=2012, race='US SENATE'),
                   YTCover(matches, year=2012, race='GOVERNOR'),
                   YTCover(matches, year=2014, race='US HOUSE'),
                   YTCover(matches, year=2014, race='US SENATE'),
                   YTCover(matches, year=2014, race='GOVERNOR')
                  )
canvid.yt <- as.data.frame(canvid.yt)

# coverage table (all)
rawytyear <- c(rep(2012, 4), rep(2014, 3))
rawytrace <- c('President', 'House', 'Senate', 'Governor', 'House', 'Senate', 'Governor')
vidtab <- as.data.frame(cbind(canvid.cmag$vid, canvid.yt$acqvid, round(canvid.yt$acqvid / canvid.cmag$vid, 3) * 100))
cantab <- as.data.frame(cbind(canvid.cmag$can, canvid.yt$chancan, round(canvid.yt$chancan / canvid.cmag$can, 3) * 100,
                              canvid.yt$acqcan, round(canvid.yt$acqcan / canvid.cmag$can, 3) * 100))
vidtab <- cbind(rawytyear, rawytrace, vidtab)
colnames(vidtab) <- c('Year', 'Office', 'WMP Total', 'Matches Found', 'Percentage Found')
cantab <- cbind(rawytyear, rawytrace, cantab)
colnames(cantab) <- c('Year', 'Office', 'WMP Total', 'Channel Found', 'Percentage Channel Found', 'Match > 0 Found', 'Percentage Match > 0 Found')
vidtab <- rbind(vidtab, c('', 'Total', colSums(apply(vidtab[, -c(1:2)], 2, as.numeric))))
vidtab$`Percentage Found`[nrow(vidtab)] <- round(as.numeric(vidtab$`Matches Found`)[nrow(vidtab)] / as.numeric(vidtab$`WMP Total`)[nrow(vidtab)] * 100, 1)
cantab <- rbind(cantab, c('', 'Total', colSums(apply(cantab[, -c(1:2)], 2, as.numeric))))
cantab$`Percentage Channel Found`[nrow(cantab)] <- round(as.numeric(cantab$`Channel Found`)[nrow(cantab)] / as.numeric(cantab$`WMP Total`)[nrow(cantab)] * 100, 1)
cantab$`Percentage Match > 0 Found`[nrow(cantab)] <- round(as.numeric(cantab$`Match > 0 Found`)[nrow(cantab)] / as.numeric(cantab$`WMP Total`)[nrow(cantab)] * 100, 1)

# video-candidate tables by party
canvid.cmag.r <- rbind(
        CMAGCover(matches, race='PRESIDENT', affiliation='REPUBLICAN'),
        CMAGCover(matches, year=2012, race='US HOUSE', affiliation='REPUBLICAN'),
        CMAGCover(matches, year=2012, race='US SENATE', affiliation='REPUBLICAN'),
        CMAGCover(matches, year=2012, race='GOVERNOR', affiliation='REPUBLICAN'),
        CMAGCover(matches, year=2014, race='US HOUSE', affiliation='REPUBLICAN'),
        CMAGCover(matches, year=2014, race='US SENATE', affiliation='REPUBLICAN'),
        CMAGCover(matches, year=2014, race='GOVERNOR', affiliation='REPUBLICAN')
                      )
canvid.cmag.r <- as.data.frame(canvid.cmag.r)

canvid.yt.r <- rbind(
          YTCover(matches, race='PRESIDENT', affiliation='REPUBLICAN'),
          YTCover(matches, year=2012, race='US HOUSE', affiliation='REPUBLICAN'),
          YTCover(matches, year=2012, race='US SENATE', affiliation='REPUBLICAN'),
          YTCover(matches, year=2012, race='GOVERNOR', affiliation='REPUBLICAN'),
          YTCover(matches, year=2014, race='US HOUSE', affiliation='REPUBLICAN'),
          YTCover(matches, year=2014, race='US SENATE', affiliation='REPUBLICAN'),
          YTCover(matches, year=2014, race='GOVERNOR', affiliation='REPUBLICAN')
                    )
canvid.yt.r <- as.data.frame(canvid.yt.r)

canvid.cmag.d <- rbind(
         CMAGCover(matches, race='PRESIDENT', affiliation='DEMOCRAT'),
         CMAGCover(matches, year=2012, race='US HOUSE', affiliation='DEMOCRAT'),
         CMAGCover(matches, year=2012, race='US SENATE', affiliation='DEMOCRAT'),
         CMAGCover(matches, year=2012, race='GOVERNOR', affiliation='DEMOCRAT'),
         CMAGCover(matches, year=2014, race='US HOUSE', affiliation='DEMOCRAT'),
         CMAGCover(matches, year=2014, race='US SENATE', affiliation='DEMOCRAT'),
         CMAGCover(matches, year=2014, race='GOVERNOR', affiliation='DEMOCRAT')
                      )
canvid.cmag.d <- as.data.frame(canvid.cmag.d)

canvid.yt.d <- rbind(
           YTCover(matches, race='PRESIDENT', affiliation='DEMOCRAT'),
           YTCover(matches, year=2012, race='US HOUSE', affiliation='DEMOCRAT'),
           YTCover(matches, year=2012, race='US SENATE', affiliation='DEMOCRAT'),
           YTCover(matches, year=2012, race='GOVERNOR', affiliation='DEMOCRAT'),
           YTCover(matches, year=2014, race='US HOUSE', affiliation='DEMOCRAT'),
           YTCover(matches, year=2014, race='US SENATE', affiliation='DEMOCRAT'),
           YTCover(matches, year=2014, race='GOVERNOR', affiliation='DEMOCRAT')
                    )
canvid.yt.d <- as.data.frame(canvid.yt.d)

# coverage table by party
vidtabpty <- data.frame(
                   R_Total=canvid.cmag.r$vid, 
                   R_Found=canvid.yt.r$acqvid, 
                   R_Percentage=format(round(canvid.yt.r$acqvid / 
                                             canvid.cmag.r$vid * 100, 1), nsmall=1),
                   D_Total=canvid.cmag.d$vid,
                   D_Found=canvid.yt.d$acqvid,
                   D_Percentage=format(round(canvid.yt.d$acqvid / 
                                             canvid.cmag.d$vid * 100, 1), nsmall=1)
                   )

totals <- data.frame(
                   R_Total=sum(canvid.cmag.r$vid), 
                   R_Found=sum(canvid.yt.r$acqvid), 
                   R_Percentage=format(round(sum(canvid.yt.r$acqvid) / 
                                             sum(canvid.cmag.r$vid) * 100, 1), nsmall=1),
                   D_Total=sum(canvid.cmag.d$vid),
                   D_Found=sum(canvid.yt.d$acqvid),
                   D_Percentage=format(round(sum(canvid.yt.d$acqvid) / 
                                             sum(canvid.cmag.d$vid) * 100, 1), nsmall=1)
                 )
vidtabpty <- rbind(vidtabpty, totals)

# print output to table1.txt
print(xtable(cbind(vidtab, vidtabpty),
             caption='Number and Percentage of Matched Videos Recovered',
             digits=c(0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1)),
      caption.placement='top', include.rownames=F, 
      file=file.path(ROOT, "results", "tables", "table1.txt"))