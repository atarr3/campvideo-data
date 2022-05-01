# Prediction

## Data
Full replication of the [Validation](#Validation) step requires the human-coded labels provided by WMP and the YouTube videos for the ads in MP4 format. Unfortunately, we cannot share either of this data publicly.
- The WMP data can be purchased [here](https://mediaproject.wesleyan.edu/dataaccess/). Our study used the 2012 Presidential, 2012 Non-Presidential, and 2014 data. The data is distributed across 7 Stata files, one for each year and race type (House, Senate, Governor, President). These files should be placed in the [``data/wmp``](data/wmp) folder.
- We provide a list of the YouTube Video IDs used in [``data/matches/matches.csv``](data/matches/matches.csv) under the `uid` variable. Users able to obtain these videos should place them in the [``data/videos``](data/videos) folder, with each video file titled ``<YouTubeID>.mp4``. ``<YouTubeID>`` is the unique YouTube video ID.
