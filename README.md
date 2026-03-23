# DSA210_PROJECT

Project Proposal


The main question of my project is whether home advantage in football is actually real or whether it is a myth. I will focus on the top five European leagues: the Premier League, La Liga, Serie A, Bundesliga, and Ligue 1. I want to see if playing at home really gives teams a clear advantage, and also whether this advantage changes under different circumstances.


I will get the main data from publicly available football match datasets from GitHub for these five leagues. These datasets include information such as match dates, home and away teams, goals, and match results. Besides the main match data, I also want to enrich the project with some additional variables like stadium capacity, the difference between the COVID period and normal seasons, match-day weather conditions, and each team’s form in their last three matches and I will collect the datas about these topics from websites like Kaggle, Open-Meteo Historical Weather API and Wikipedia ( I will going to examine 2 or 3 of them).


I will collect the data by downloading the datasets in CSV format or extracting them from the websites like Wikipedia and then cleaning and combining them with the help of Python. For the enrichment part, I will match stadium capacity data to the teams or stadiums, add a variable showing whether a match was played during the COVID period or in a normal season, and connect weather data to the city and date of each match. I will calculate the recent form variable from the match results dataset by checking how many points each team got in its previous three matches.


The main dataset will be match-based and will include several thousand observations from the top five leagues, so it should be large enough for analysis. The main variables will be match result, goals, teams, date, and league, while the additional variables, to enrich the main data, will be stadium capacity, COVID times compared to normal period, weather on match day, and recent team form. With this dataset, I hope to understand not only whether home advantage exists, but also in which situations it becomes stronger.

