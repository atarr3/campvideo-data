## written in R version 4.1.3
## presupposes source files in /Source
options(stringsAsFactors = F)
library(readstata13)
library(stringr)

## load auxiliary data
elecinfo <- read.csv('Source/elecinfo.csv') # election-specific data compiled by authors
deduplist <- read.csv('Source/dedup_list.csv') # list of (near-)duplicates
matchlist <- read.table('Source/matchlist.txt', # list of matches
                        sep = '\t', quote = '')
colnames(matchlist) <- c('vidfile', 'uid')


## load WMP data with select variables and only keep unique entries
## then merge and fix variable names
hou12 <- read.dta13('Source/wmp-house-2012-v1.1.dta')[, c('creative', 'vidfile', 'l', 'categorystate', 'district', 'election', 'house', 'senate', 'gov', 'sponsor', 'sponsorcmag', 'party', 'cand_id', 'spanish')]
hou12 <- hou12[!duplicated(hou12), ]
hou12$racelab <- 'hou2012'
sen12 <- read.dta13('Source/wmp-senate-2012-v1.1.dta')[, c('creative', 'vidfile', 'l', 'categorystate', 'district', 'election', 'house', 'senate', 'gov', 'sponsor', 'sponsorcmag', 'party', 'cand_id', 'spanish')]
sen12 <- sen12[!duplicated(sen12), ]
sen12$racelab <- 'sen2012'
gov12 <- read.dta13('Source/wmp-gov-2012-v1.1.dta')[, c('creative', 'vidfile', 'l', 'categorystate', 'district', 'election', 'house', 'senate', 'gov', 'sponsor', 'sponsorcmag', 'party', 'cand_id', 'spanish')]
gov12 <- gov12[!duplicated(gov12), ]
gov12$racelab <- 'gov2012'
wmp12 <- rbind(rbind(hou12, sen12), gov12)

pre12 <- read.dta13('Source/wmp-pres-2012-v1.2_compress.dta')[, c('creative', 'vidfile', 'l', 'categorystate', 'district', 'election', 'sponsorwmp', 'sponsorcmag', 'affiliation', 'cand_id', 'spanish')]
pre12 <- pre12[!duplicated(pre12), ]
pre12$racelab <- 'pre2012'

hou14 <- read.dta13('Source/wmp-house-2014-v1.0.dta')[, c('creative', 'vidfile', 'l', 'categorystate', 'district', 'election', 'house', 'senate', 'gov', 'sponsor', 'sponsorcmag', 'affiliation', 'cand_id', 'spanish')]
hou14 <- hou14[!duplicated(hou14), ]
hou14$racelab <- 'hou2014'
sen14 <- read.dta13('Source/wmp-senate-2014-v1.0.dta')[, c('creative', 'vidfile', 'l', 'categorystate', 'district', 'election', 'house', 'senate', 'gov', 'sponsor', 'sponsorcmag', 'affiliation', 'cand_id', 'spanish')]
sen14 <- sen14[!duplicated(sen14), ]
sen14$racelab <- 'sen2014'
gov14 <- read.dta13('Source/wmp-gov-2014-v1.1.dta')[, c('creative', 'vidfile', 'l', 'categorystate', 'district', 'election', 'house', 'senate', 'gov', 'sponsor', 'sponsorcmag', 'affiliation', 'cand_id', 'spanish')]
gov14 <- gov14[!duplicated(gov14), ]
gov14$racelab <- 'gov2014'
wmp14 <- rbind(rbind(hou14, sen14), gov14)

## additional variable adjustments
colnames(wmp12)[which(colnames(wmp12) == 'party')] <- 'affiliation'
wmp12 <- wmp12[wmp12$election == 'GENERAL' &
                 (wmp12$sponsor == 1 | wmp12$sponsor == 3) &
                 wmp12$spanish == 0, ]
wmp12 <- wmp12[rowSums(is.na(wmp12)) < ncol(wmp12), ]
wmp12 <- wmp12[!(wmp12$racelab == 'gov2012' & wmp12$categorystate == 'WI'), ] # remove WI recall election

colnames(pre12)[which(colnames(pre12) == 'sponsorwmp')] <- 'sponsor'
wmp12p <- pre12[pre12$election == 'GENERAL' &
                  (pre12$sponsor == 'Candidate' | pre12$sponsor == 'Cand & Party') &
                  (pre12$affiliation == 'DEMOCRAT' | pre12$affiliation == 'REPUBLICAN') &
                  pre12$spanish == 0, ]
wmp12p <- wmp12p[rowSums(is.na(wmp12p)) < ncol(wmp12p), ]
wmp12p <- wmp12p[wmp12p$cand_id == 'Obama_Barack' | wmp12p$cand_id == 'Romney_Mitt', ]
wmp12p <- wmp12p[wmp12p$vidfile != 'PRES_RNC_WHERE_DID_ALL_THE_MONEY_GO', ] # remove one erroneous entry

wmp14 <- wmp14[wmp14$election == 'GENERAL' &
                 (wmp14$sponsor == 1 | wmp14$sponsor == 3) &
                 wmp14$spanish == 0, ]
wmp14 <- wmp14[rowSums(is.na(wmp14)) < ncol(wmp14), ]

## create unique identifiers based on the state/CD number abbreviations
wmp12$cdmatch <- NA
for(i in 1:nrow(wmp12)){
  if(wmp12$district[i] != 'N/A'){wmp12$cdmatch[i] <- paste(wmp12$categorystate[i], rep(0, 2 - nchar(wmp12$district[i])), wmp12$district[i], sep = '')}
  else{wmp12$cdmatch[i] <- wmp12$categorystate[i]}
}

wmp12p$cdmatch <- 'POUS'

wmp14$cdmatch <- NA
for(i in 1:nrow(wmp14)){
  if(wmp14$district[i] != 'N/A'){wmp14$cdmatch[i] <- paste(wmp14$categorystate[i], rep(0, 2 - nchar(wmp14$district[i])), wmp14$district[i], sep = '')}
  else{wmp14$cdmatch[i] <- wmp14$categorystate[i]}
}

## attach election information
wmp12 <- merge.data.frame(wmp12, elecinfo, by = c('creative', 'racelab'))
wmp12p <- merge.data.frame(wmp12p, elecinfo, by = c('creative', 'racelab'))
wmp14 <- merge.data.frame(wmp14, elecinfo, by = c('creative', 'racelab'))

## merge the matching results to the compiled list
matched12 <- merge.data.frame(matchlist, wmp12, by = 'vidfile', all = T)
matched12 <- matched12[!is.na(matched12$creative), ]
matched12$uid[is.na(matched12$uid)] <- 'NoChannel'
matched12$uid[matched12$uid == ''] <- 'NoMatch'

matched12p <- merge.data.frame(matchlist, wmp12p, by = 'vidfile', all = T)
matched12p <- matched12p[!is.na(matched12p$creative), ]
matched12p$uid[is.na(matched12p$uid)] <- 'NoChannel'
matched12p$uid[matched12p$uid == ''] <- 'NoMatch'

matched14 <- merge.data.frame(matchlist, wmp14, by = 'vidfile', all = T)
matched14 <- matched14[!is.na(matched14$creative), ]
matched14$uid[is.na(matched14$uid)] <- 'NoChannel'
matched14$uid[matched14$uid == ''] <- 'NoMatch'

## deduplication function
Do_Dedup <- function(matcheddf, roster){
  matcheddf['vidfile'] <- apply(matcheddf['vidfile'], 2, function(x){gsub('\'', '', x)})
  correctionind <- which(matcheddf$vidfile %in% roster$sub) ## indices for correction
  matcheddf$deletion <- 0
  
  for(i in 1:length(correctionind)){
    ## retrieve rows with duplicate videos
    subvidfile <- matcheddf$vidfile[correctionind[i]]
    mainrow <- which(matcheddf$vidfile == roster$main[roster$sub == subvidfile])
    subrow <- which(matcheddf$vidfile == subvidfile)
    
    if(length(mainrow) > 0){ # execute only if duplicates are actually recorded in data
      
      if(matcheddf$uid[mainrow] %in% c('NoMatch', 'NoChannel')){ # if main entry doesn't have a match
        
        if(!(matcheddf$uid[mainrow] %in% c('NoMatch', 'NoChannel'))){ # and if sub entry has a match
          matcheddf[mainrow, ] <- matcheddf[subrow, ] # replace main with sub
          matcheddf$deletion[subrow] <- 1 # flag original sub for deletion
          roster$sub[roster$sub == subvidfile] <- roster$main[roster$sub == subvidfile]
          roster$main[roster$sub == subvidfile] <- subvidfile # exchange main and sub in roster
        }
        
        else{ # if neither main or sub has a match
          matcheddf$deletion[subrow] <- 1 # flag sub for deletion
        }
      }
      
      else{ # if main entry has a match
        matcheddf$deletion[subrow] <- 1 # flag sub for deletion
      }
    }
  }
  
  
  # purge data frame with updated roster
  matcheddf <- matcheddf[matcheddf$deletion == 0, -which(colnames(matcheddf) == 'deletion')]
  out <- list(matcheddf, roster)
  return(out)
}

## perform deduplication
dedupmatched12 <- Do_Dedup(matched12, deduplist)
# manual removal for two video groups since the original variable name does not distinguish length
matched12p <- matched12p[!(matched12p$creative %in% c('PRES/DNC&OBAMA THE CHOICE 60',
                                                      'PRES/DNC&OBAMA WOMEN SPEAK JENNI')), ]
dedupmatched12p <- Do_Dedup(matched12p, dedupmatched12[[2]])
dedupmatched14 <- Do_Dedup(matched14, dedupmatched12p[[2]])

dm12 <- dedupmatched12[[1]]
dm12p <- dedupmatched12p[[1]]
dm14 <- dedupmatched14[[1]]
dmroster <- dedupmatched14[[2]]

## fix missing entries in the original file
fix12ref <- which(is.na(dm12$cand_id) | dm12$cand_id %in% c('', ' '))
for(i in fix12ref){
  datasub <- dm12[dm12$racelab == dm12$racelab[i] &
                    dm12$categorystate == dm12$categorystate[i] &
                    dm12$district == dm12$district[i] &
                    dm12$affiliation == dm12$affiliation[i] &
                    dm12$vidfile != dm12$vidfile[i], ]
  dm12$cand_id[i] <- names(sort(table(datasub$cand_id), decreasing = T)[1])
}

fix14ref <- which(is.na(dm14$cand_id) | dm14$cand_id %in% c('', ' '))
for(i in fix14ref){
  datasub <- dm14[dm14$racelab == dm14$racelab[i] &
                    dm14$categorystate == dm14$categorystate[i] &
                    dm14$district == dm14$district[i] &
                    dm14$affiliation == dm14$affiliation[i] &
                    dm14$vidfile != dm14$vidfile[i], ]
  dm14$cand_id[i] <- names(sort(table(datasub$cand_id), decreasing = T)[1])
}

## write source files to .csv for coverage table
write.csv(dm12, 'Source/dm12.csv', row.names = F)
write.csv(dm12p, 'Source/dm12p.csv', row.names = F)
write.csv(dm14, 'Source/dm14.csv', row.names = F)
write.csv(dmroster, 'Source/dmroster.csv', row.names = F)