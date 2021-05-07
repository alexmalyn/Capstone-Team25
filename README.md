# Capstone-Team25
Deep Learning Market Analysis

Deep Learning Market Analysis
Team Number S21-25


Team members: Adil Rashid, Alex Malynovsky, Aryaman Narayanan,
Rishabh Chari, Steven Negron
Adviser(s): Roy Yates
Submitted in partial fulfillment of the requirements for senior design project Electrical and Computer Engineering Department Rutgers University, Piscataway, NJ 08854


Abstract— Humans are prone to making mistakes, whether it be in their personal lives or on Wall Street. Unlike computers, we are unable to process information the same way that computers can; completely impartial and neutral. Today, this advantage is being utilized in a variety of ways on the stock market in order to aid investors in turning a profit. However, a lot of the methods being used include well-known algorithmic approaches tailored to the observation of certain market indicators and increase ROI for the investor. 
	Our team aims to adapt a relatively recent deep learning architecture not originally tailored for time series analysis to learn from historic data of every stock within the S&P 500 in order to make large-scale trend predictions without any reinforcement. Our objective with this project is to use and modify an adversarial architecture and apply it to a time series problem to see if we can extract any temporal value from the model’s ability.


	Keywords— Deep Learning, Market Analysis, Generative Adversarial Networks (GAN), Sentiment Analysis, Natural Language Processing


INTRODUCTION
 Technical analysis is a type of financial analysis that attempts to extract temporal patterns within market data and corresponding indicators in order to determine trends as well as attempt to forecast on a market’s performance. Technical analysis in the context of stock markets has been used for decades, making such fields rife with data to analyze and study.
However, not many efforts have been made in an attempt to develop a model that analyzes both stock and intrinsic values. In the field of deep learning, data is key, and without enough data, a model will not learn to generalize given new samples of data it has not seen before. This is especially true in the problem space of time series analysis. It becomes very difficult to teach a model to generalize temporal features when most models cannot see beyond a scope of their training regimen.
Despite this, the flexibility and customization of deep learning allows for models to train in any fashion or dimension. In this report, we discuss the approach taken in order to adapt a specific deep learning architecture to time series analysis and how the dataset is preprocessed as well as fit upon in order to maximize the financial value the project could provide.


TECHNICAL EVOLUTION & ARCHITECTURE
The technical evolution of this project resides in a three step process: Data Collection, Model Training, and Output Preparation. 
	For the first step in this procedure, all the necessary data is prepared as an input to our hybrid GAN. Although the data streams are multifaceted, they can be classified into two major categories: Technical Data and Fundamental Data. The former describes all of the values obtained from the Yahoo Finance API, containing all of the data we require to analyse and classify the stocks within the S&P 500, whereas the latter describes the quantified sentiment analysis performed by our Natural Language Processing (NLP) algorithm obtained from our configured headline scraper. 
Next, all these data streams are funneled into a subsystem within the GAN, handling any data collections that require preprocessing. As it would be highly impractical and inefficient to train the neural network on all technical indicators within a stock, this preprocessing subsystem performs a light procedure on the stock for a singular timeslot and outputs a more digestible analysis-driven format for our GAN to interpret. As the sentiment analyses are already outputting comprehensive values for the GAN, they are fed directly to the GAN’s discriminator. Now, to explain the GAN [1], the system works like any other optimization problem seeking to minimize a loss function.

	In this case, we have two loss functions to optimize, the generator’s and the discriminator’s function. The generator of the GAN is the subcomponent that produces fake data, or predictions in our case, to match real data. The way this part of the GAN learns is through back-propagation of the generator’s performance in fooling the discriminator. Similarly, the discriminator is trained by its own performance. For example, if it fails to classify fake data is real or vice versa, the discriminator is penalized in its training. Eventually, we see an equilibrium reached between the two subcomponents where it is reasonably difficult for the discriminator to see the difference between fake and real sample data. 
However, our adapted GAN is structured somewhat differently. Training involves the same concepts, but the addition of certain shortcuts for our dataset allow the model to capture more features in the discriminator portion of the training. 
First, the data is preprocessed in a rolling window fashion which is typical for time series regression problems:

Then each of these windows are fed through the generator network which consists of an LSTM layer, which extracts temporal dynamics and feeds the sub-sample into a Dense network with a dimension equal to the number of features. By this setup, we can hope to achieve an optimization of forecasting t+1 or one unit time ahead of the window's time period. So, an array of dimension (1,num_features) is extracted that represents the forecast. In order for the discriminator to properly discriminate whether the prediction is real or fake, the architecture concatenates this prediction back to the original window data, or essentially the same data that was fed into the generator. Thus, the 'predicted data' in the image above is 5 rows of real data concatenated with a 6th row that represents the prediction from the generator. Rows, of course, are just units of time in a time series problem. Similar to a typical GAN, we have to also feed real data into the discriminator, so our 'real data' is 6 rows of real data from the same time period. Labels are provided respectively (1 - real, 0 - fake), and the discriminator is given insight into the time series in order to train smarter.






























Discriminator, Generator, GAN Neural Models

	There are 3 main methods of getting articles and computing sentiment analysis. These methods are Data Collection, Data Storage, and Data Manipulation. Each method integrates multiple tools that are used to successfully complete each method. 
The Data Collection process involves a few considerations that need to be accounted for. The first consideration is where to find articles and what type of data needs to be collected. This consideration is important as the choices made in regards to the source of the articles and the type of data that is being scrapped has direct consequences to the overall results of the sentiment analysis being produced. There are many different sources that can be used to scrape articles. We decided to use a news aggregate site, specifically Google News. The reasons why we chose Google News was because it has articles from many different sources and it has an RSS feed which we used to scrape data from. However, there is a restriction by  using Google News for the source of our articles. The restriction is that by using Google News, we are restricted to only scraping the article’s headlines, not the full article text. The reason for this is because Google News holds only the article headlines, published date, and the source of the source of the article. If we wanted full text we would need to make scrapers for each individual webpage which would have greatly reduced the amount of articles we could scrape.  The second consideration is how to separate the sentiment and scrape articles according to that requirement.  We decided to compute daily sentiment because it allows for a lot more articles to be scrapped compared to weekly articles. 
Previously stated was the use of RSS feeds of Google News for scraping. RSS feeds ,also called Really Simple Syndication, allow for web pages and files to be put into XML notation so a computer can access its contents easily. For Google News the RSS feed holds important information that we need such as every article’s headline and publish date. The way we choose which articles are selected is by using a date parameter in the URL of the feed. We can specify which date range the feed to choose what days are to be included in the sentiment analysis. To extract the data from the RSS feed we use a third party package called Feedparser [2] , which allows us to pick and choose what data we want to extract from the RSS feed. We set the extracted data, which is the headline, published date, and the URL of each article in the feed, into a Python dictionary. The headline will then be processed to compute sentiment analysis, while the date and URL are unmodified and stored into a separate dictionary to be used later.  
Once we have the scraped article headline we pass it into a sentiment analysis tool. The tool we used to compute the sentiment is VaderSentiment [2]. VaderSentiment is a rule based sentiment analysis tool used to compute the sentiment of text. Rule based sentiment analysis is just one form of sentiment analysis, and it relies on counting the number of “rules” in the text. These rules are specified by the authors of the VaderSentiment tool and act as a guideline of how sentiment should be attributed based on these rules. For example, if a rule is that “great” is a very positive word then when Vader sees  “great” in the input text it knows to scale the sentiment score more positively. 
	One reason why Vader was chosen was because of its lack of preprocessing needed. Vader itself does most of the preprocessing work for us and will be able to remove items such as punctuation and stop words. Therefore, we need to input text and Vader will make sure only the appropriate items in that text are taken into consideration for sentiment analysis. The outputs of Vader are 4 values, how  negative, neutral, positive the text is on a scale from 0 to 1 , and the compound sentiment which is a normalized calculation from -1 to 1 where -1 means that text is very negative, +1 means that the text is very positive and 0 means the text is neutral. 
	For each article headline we pass it through Vader to get the 4 sentiment values and then we store the headline, the published date, the URL, and these 4 values into their own dictionary to be stored into MongoDB Atlas, This process is the Data Storage process. 
	MongoDB Atlas is a web based database using the same structure as MongoDB. The hierarchy  of the components of MongoDB Atlas is Clusters -> Databases -> Collections -> Documents -> Fields. Clusters are the highest part of the architecture and one cluster will hold all the data we need as it allows for a cap of 512MB of data for the free tier of MongoDB. Databases are separated by GICS Sector, so one Database holds all the article information for all the stocks corresponding to one GICS Sector. For example, the Communication Services Sector has its own Database holding articles for every stock in that sector. Next are Collections, which hold all the article information for each stock in a specific GICS sector. Facebook (FB) is one stock in the Communication Services sector, so one collection holds all the sentiment results for all the articles related to FB. Finally the architecture has Documents and Fields. Documents are what holds each article related to a stock. So one document will hold one article related to a stock. Each document has 7 fields, which are the 3 parts of data scraped from the RSS feed and the 4 sentiment analysis results. 
	Once we have populated this database with all the data we need, we then move onto the final process in this system which is Data Manipulation. This process involves taking each collection in a database and transforming it into a daily sentiment table for each stock. 
The last step in this procedure involves preparing our outputs and establishing baseline comparisons with the trends our model evaluates. Since we will have individual stock performance at our disposal, we’d be able to provide detailed analyses of any possible correlations between stocks as well as GICS sectors and sub-industries.



METHODS & RESULTS
After the week that it took to train our entire dataset, we received some promising results in trend prediction compared to naive or baseline forecasts. Although getting exact prices proved to be difficult for the model to capture in enough epochs, the generator (which is the component that generates synthetic forecasts) proves to be very apt at capturing the derivatives of our features. As a result, we used this to our advantage, calculating short windows of trend, or simply derivative, to compare to our baseline metrics.
Mean Absolute Error Comparisons
RMSE Comparisons
As seen above, the model achieves much lower error with respect to the authentic future data compared to more naive forecasts such as Flat and Slope forecasts. 


COST ANALYSIS
 	The project was built entirely through python packages and libraries such as scikit learn, keras, VaderLexicon, Pandas dataframe, numpy etc. The neural framework was built using keras on the tensorflow backend. The entire stock data was collected from Yahoo Finance free of charge and the entire project was software based, making the total cost for the project turn out to be $0. 


CONCLUSION
Throughout the course of the entire project, we faced many hurdles along the way. To prepare data for training, we had to deal with lagged NaN gaps, missing data and feature engineering the right technical indicators during the data preprocessing phase. The team had limited time/computing constraints, and Generative Adversarial Network (GAN) are inherently very demanding. To optimize GAN training time to a feasible length in order to train several hundred models in the time/computing constraint we have whilst still maintaining moderate high level trend accuracy was challenging. The tuning of all the network’s hyperparameters were taken into consideration with an empirical approach based on the results and visualizations which took  a week to properly enhance. The goal with this project was to use a relatively new deep learning neural framework to try and predict stock prices. The GAN is usually used in image generation and the goal was to observe how effective it would be when working with time series data. Our deep learning framework predicts trends for the next trading day which would be highly valuable to investment programs such as mutual funds and give them a slight edge when making decisions. The model also allows a broader perspective on the market and how each sub industry performs against each other. Given the proper computing power and larger datasets with higher density, this model would only improve at its prediction and accuracy, making it an even more favorable tool for any financial analyst.



REFERENCES
Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio, “Generative Adversarial Networks,” Universite de Montréal, June 2014.
Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.





