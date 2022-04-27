## written in R version 4.1.3
## presupposes source files in /Source
options(stringsAsFactors = F)
library(stringr)
library(xtable)

ytinfo <- read.csv('Source/ytinfo.csv') # youtube video-specific data compiled by authors

## following three files are taken from Preprocess_Replication.R
dm12 <- read.csv('Source/dm12.csv')
dm12p <- read.csv('Source/dm12p.csv')
dm14 <- read.csv('Source/dm14.csv')

## function that returns CMAG video and candidate totals
GetCMAGCover <- function(df, condition = NULL){
  out <- vector()
  if(is.null(condition)){ # if no condition is specified
    out <- as.data.frame(rbind(out, c(vid = length(unique(df$creative)), can = length(unique(df$cand_id)))))
  }
  else{ # if condition is specified
    conditionset <- sort(unique(df[condition])[, 1])
    for(i in 1:length(conditionset)){
      sub <- df[df[condition] == conditionset[i], ]
      subvid <- length(unique(sub$creative))
      subcan <- length(unique(sub$cand_id))
      out <- as.data.frame(rbind(out, c(vid = subvid, can = subcan)))
    }
    rownames(out) <- conditionset
  }
  return(out)
}

## function that returns YouTube video and candidate totals
GetYTCover <- function(df, condition = NULL){
  out <- vector()
  if(is.null(condition)){ # if no condition is specified
    out <- as.data.frame(rbind(out,
                               c(chanvid = length(unique(df$creative[df$uid != 'NoChannel'])),
                                 chancan = length(unique(df$cand_id[df$uid != 'NoChannel'])),
                                 acqvid = length(unique(df$creative[(df$uid != 'NoChannel') & (df$uid != 'NoMatch')])),
                                 acqcan = length(unique(df$cand_id[(df$uid != 'NoChannel') & (df$uid != 'NoMatch')])))))
  }
  
  else{ # if condition is specified
    conditionset <- sort(unique(df[condition])[, 1])
    for(i in 1:length(conditionset)){
      sub <- df[df[condition] == conditionset[i], ]
      subchanvid <- length(unique(sub$creative[sub$uid != 'NoChannel']))
      subchancan <- length(unique(sub$cand_id[sub$uid != 'NoChannel']))
      subacqvid <- length(unique(sub$creative[(sub$uid != 'NoChannel') & (sub$uid != 'NoMatch')]))
      subacqcan <- length(unique(sub$cand_id[(sub$uid != 'NoChannel' & sub$uid != 'NoMatch')]))
      out <- as.data.frame(rbind(out, c(chanvid = subchanvid, chancan = subchancan,
                                        acqvid = subacqvid, acqcan = subacqcan)))
    }
    rownames(out) <- conditionset
  }
  
  return(out)
}

## create the video- and candidate-coverage table
cmagcanvid <- rbind(GetCMAGCover(dm12p),
                    GetCMAGCover(dm12, 'house')[2, ],
                    GetCMAGCover(dm12, 'senate')[2, ],
                    GetCMAGCover(dm12, 'gov')[2, ],
                    GetCMAGCover(dm14, 'house')[2, ],
                    GetCMAGCover(dm14, 'senate')[2, ],
                    GetCMAGCover(dm14, 'gov')[2, ])
ytcanvid <- rbind(GetYTCover(dm12p),
                  GetYTCover(dm12, 'house')[2, ],
                  GetYTCover(dm12, 'senate')[2, ],
                  GetYTCover(dm12, 'gov')[2, ],
                  GetYTCover(dm14, 'house')[2, ],
                  GetYTCover(dm14, 'senate')[2, ],
                  GetYTCover(dm14, 'gov')[2, ])

## create unique ids to subset raw videos by WMP records
ytid <- paste(substr(ytinfo$race, 4, 7), ytinfo$yt_district, ytinfo$candidate, sep = '_')

## function to recover last names of candidates
recover.candname <- function(df){
  out <- rep(NA, nrow(df))
  partyind <- df$affiliation
  out[partyind == 'REPUBLICAN'] <- sapply(str_split(df$rep[partyind == 'REPUBLICAN'], ','), head, 1)
  out[partyind == 'DEMOCRAT'] <- sapply(str_split(df$dem[partyind == 'DEMOCRAT'], ','), head, 1)
  out[is.na(out)] <- sapply(str_split(df$third[is.na(out)], ','), head, 1)
  return(out)
} 

## candidate identifier database to serve as reference
cmid12 <- unique(paste(2012,
                       dm12$cdmatch,
                       recover.candname(dm12), sep = '_'))
cmid12p <- unique(paste(2012,
                        dm12p$cdmatch,
                        recover.candname(dm12p), sep = '_'))
cmid14 <- unique(paste(2014,
                       dm14$cdmatch,
                       recover.candname(dm14), sep = '_'))
cmid <- c(cmid12, cmid12p, cmid14)

## limit candidates to those reported by the CMAG
cmindex <- ytid %in% cmid
ytinfo <- ytinfo[cmindex, ]
ytid <- ytid[cmindex]

## table 1 configuration
rawytyear <- c(rep(2012, 4), rep(2014, 3))
rawytrace <- c('President', 'House', 'Senate', 'Governor', 'House', 'Senate', 'Governor')
vidtab <- as.data.frame(cbind(cmagcanvid$vid, ytcanvid$acqvid, round(ytcanvid$acqvid / cmagcanvid$vid, 3) * 100))
cantab <- as.data.frame(cbind(cmagcanvid$can, ytcanvid$chancan, round(ytcanvid$chancan / cmagcanvid$can, 3) * 100,
                              ytcanvid$acqcan, round(ytcanvid$acqcan / cmagcanvid$can, 3) * 100))
vidtab <- cbind(rawytyear, rawytrace, vidtab)
colnames(vidtab) <- c('Year', 'Office', 'WMP Total', 'Matches Found', 'Percentage Found')
cantab <- cbind(rawytyear, rawytrace, cantab)
colnames(cantab) <- c('Year', 'Office', 'WMP Total', 'Channel Found', 'Percentage Channel Found', 'Match > 0 Found', 'Percentage Match > 0 Found')
vidtab <- rbind(vidtab, c('', 'Total', colSums(apply(vidtab[, -c(1:2)], 2, as.numeric))))
vidtab$`Percentage Found`[nrow(vidtab)] <- round(as.numeric(vidtab$`Matches Found`)[nrow(vidtab)] / as.numeric(vidtab$`WMP Total`)[nrow(vidtab)] * 100, 2)
cantab <- rbind(cantab, c('', 'Total', colSums(apply(cantab[, -c(1:2)], 2, as.numeric))))
cantab$`Percentage Channel Found`[nrow(cantab)] <- round(as.numeric(cantab$`Channel Found`)[nrow(cantab)] / as.numeric(cantab$`WMP Total`)[nrow(cantab)] * 100, 2)
cantab$`Percentage Match > 0 Found`[nrow(cantab)] <- round(as.numeric(cantab$`Match > 0 Found`)[nrow(cantab)] / as.numeric(cantab$`WMP Total`)[nrow(cantab)] * 100, 2)

## create the tables for partisanship
cmagcanvidr <- rbind(GetCMAGCover(dm12p[dm12p$affiliation == 'REPUBLICAN', ]),
                     GetCMAGCover(dm12[dm12$affiliation == 'REPUBLICAN', ], 'house')[2, ],
                     GetCMAGCover(dm12[dm12$affiliation == 'REPUBLICAN', ], 'senate')[2, ],
                     GetCMAGCover(dm12[dm12$affiliation == 'REPUBLICAN', ], 'gov')[2, ],
                     GetCMAGCover(dm14[dm14$affiliation == 'REPUBLICAN', ], 'house')[2, ],
                     GetCMAGCover(dm14[dm14$affiliation == 'REPUBLICAN', ], 'senate')[2, ],
                     GetCMAGCover(dm14[dm14$affiliation == 'REPUBLICAN', ], 'gov')[2, ])

cmagcanvidd <- rbind(GetCMAGCover(dm12p[dm12p$affiliation == 'DEMOCRAT', ]),
                     GetCMAGCover(dm12[dm12$affiliation == 'DEMOCRAT', ], 'house')[2, ],
                     GetCMAGCover(dm12[dm12$affiliation == 'DEMOCRAT', ], 'senate')[2, ],
                     GetCMAGCover(dm12[dm12$affiliation == 'DEMOCRAT', ], 'gov')[2, ],
                     GetCMAGCover(dm14[dm14$affiliation == 'DEMOCRAT', ], 'house')[2, ],
                     GetCMAGCover(dm14[dm14$affiliation == 'DEMOCRAT', ], 'senate')[2, ],
                     GetCMAGCover(dm14[dm14$affiliation == 'DEMOCRAT', ], 'gov')[2, ])

ytcanvidr <- rbind(GetYTCover(dm12p[dm12p$affiliation == 'REPUBLICAN', ]),
                   GetYTCover(dm12[dm12$affiliation == 'REPUBLICAN', ], 'house')[2, ],
                   GetYTCover(dm12[dm12$affiliation == 'REPUBLICAN', ], 'senate')[2, ],
                   GetYTCover(dm12[dm12$affiliation == 'REPUBLICAN', ], 'gov')[2, ],
                   GetYTCover(dm14[dm14$affiliation == 'REPUBLICAN', ], 'house')[2, ],
                   GetYTCover(dm14[dm14$affiliation == 'REPUBLICAN', ], 'senate')[2, ],
                   GetYTCover(dm14[dm14$affiliation == 'REPUBLICAN', ], 'gov')[2, ])

ytcanvidd <- rbind(GetYTCover(dm12p[dm12p$affiliation == 'DEMOCRAT', ]),
                   GetYTCover(dm12[dm12$affiliation == 'DEMOCRAT', ], 'house')[2, ],
                   GetYTCover(dm12[dm12$affiliation == 'DEMOCRAT', ], 'senate')[2, ],
                   GetYTCover(dm12[dm12$affiliation == 'DEMOCRAT', ], 'gov')[2, ],
                   GetYTCover(dm14[dm14$affiliation == 'DEMOCRAT', ], 'house')[2, ],
                   GetYTCover(dm14[dm14$affiliation == 'DEMOCRAT', ], 'senate')[2, ],
                   GetYTCover(dm14[dm14$affiliation == 'DEMOCRAT', ], 'gov')[2, ])

vidtabpty <- as.data.frame(
  cbind(cbind(cmagcanvidr$vid, ytcanvidr$acqvid, round(ytcanvidr$acqvid / cmagcanvidr$vid * 100, 1)),
        cbind(cmagcanvidd$vid, ytcanvidd$acqvid, round(ytcanvidd$acqvid / cmagcanvidd$vid * 100, 1))))
colnames(vidtabpty) <- c('R_Total', 'R_Found', 'R_Percentage', 'D_Total', 'D_Found', 'D_Percentage')
vidtabpty <- rbind(vidtabpty, colSums(vidtabpty))
vidtabpty$R_Percentage[8] <- round(vidtabpty$R_Found[8] / vidtabpty$R_Total[8], 3) * 100
vidtabpty$D_Percentage[8] <- round(vidtabpty$D_Found[8] / vidtabpty$D_Total[8], 3) * 100


print(xtable(cbind(vidtab, vidtabpty),
             caption = 'Number and Percentage of Matched Videos Recovered'),
      caption.placement = 'top', digits = c(0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1), include.rownames = F)