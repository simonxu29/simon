#### Start Of Efficient Frontier ####

# Bring in dependencies
library("httr")
library("jsonlite")
library("keyring")
library("dplyr")
library("ggplot2")
library("plotly")

# Get & Format Price History Data

tickers = c("NVDA","KKR","TMUS","TSLA")

for(i in 1:length(tickers)){
  
  reqUrl <- paste0("https://api.stockdata.org/v1/data/eod?symbols=",
                   tickers[i],
                   "&date_from=2023",
                   "&interval=month",
                   "&sort=asc&api_token=",
                   keyring::key_get("stock_data_key")
  )
  
  priceHistoryRes <- GET(reqUrl)
  
  tickerPriceData <- data.frame(fromJSON(rawToChar(priceHistoryRes$content))$data)
  
  if(i == 1){
    alldata <- tickerPriceData[ ,c("date","close")]
  
  }else{
    alldata <- cbind(alldata,tickerPriceData$close)
  }
}


colnames(alldata) <- c("Date",tickers)




# Get Daily Returns and Summary Data

expectedReturns = NULL
standardDeviations = NULL

for(e in tickers){
  
  newColumnName = paste0(e, "return")
  
  alldata <- alldata %>%
    
    mutate(!!newColumnName := (get(e) - lag(get(e))) / lag(get(e)) )
  
  expectedReturns <- cbind(expectedReturns,mean(alldata[-1,newColumnName]))
  
  standardDeviations <- cbind(standardDeviations,sd(alldata[-1,newColumnName]))
  
}

colnames(expectedReturns) <- tickers
colnames(standardDeviations) <- tickers

variances <- standardDeviations**2
modifiedSharpeRatios <- expectedReturns / standardDeviations



# Calculate X an Variance-Covariance Matrices

xDf <- alldata[-1,-(1:(length(tickers)+1))]

colnames(xDf) <- tickers

for (e in tickers) {
  xDf <- xDf %>%
    mutate(!!e :=get(e) - expectedReturns[1,e])
}

xMatrix <- data.matrix(xDf)

xMatrixTranspose <- t(xMatrix)


varCovar <- (xMatrixTranspose %*% xMatrix) / (nrow(xDf) -1)


# Expected Return and Volatility For Equally weighted portfolio


equalPortfolio <- xDf[1,]

for (e in tickers) {
  equalPortfolio <- equalPortfolio %>%
  mutate(!!e := 1/length(tickers))
  
  
}

weights <- data.matrix(equalPortfolio)

equalPortfolio$expectedReturns <- sum(weights * expectedReturns)

equalPortfolio$volatility <- sqrt((weights %*% varCovar) %*% t(weights))

equalPortfolio$sharpeRatio <- equalPortfolio$expectedReturns / equalPortfolio$volatility


# Simulate Multiple Portfolio Weights

numOfPortfolios <- 10000

multipleWeight <- xDf[(1:numOfPortfolios),]

for (e in tickers) {
  multipleWeight <- multipleWeight %>%
    mutate(!!e := runif(numOfPortfolios))
}


multipleWeight$totalOfRandoms <- rowSums(multipleWeight)

weightColNames <- c()

for (e in tickers) {
  newcolumnname <- paste0(e,"weight")
    weightColNames <- c(weightColNames,newcolumnname)
    
    multipleWeight <- multipleWeight %>%
      mutate(!!newcolumnname := get(e) / totalOfRandoms)
}



# Expected Return and Volatility For Different Weighted portfolio

for (i in 1:nrow(multipleWeight)) {
  weights <- data.matrix(multipleWeight[i,weightColNames])
  
  multipleWeight[i,("expectedReturns")] <- sum(weights * expectedReturns)
  
  multipleWeight[i,("volatility")] <- sqrt((weights %*% varCovar) %*% t(weights))
  

}

multipleWeight$sharpeRatio <- multipleWeight$expectedReturns / multipleWeight$volatility


multipleWeight[,c(weightColNames,"expectedReturns","volatility")] <- round(multipleWeight[,c(weightColNames,"expectedReturns","volatility")] * 100, 4)

# Generate Interactive Efficeint Frontier Chart

generalPlot <- function(data,  knownaes){
  match_aes <- intersect(names(data),knownaes)
  my_aes_list <- purrr::set_names(purrr::map(match_aes, rlang::sym), match_aes)
  my_aes <- rlang::eval_tidy(quo(aes(!!!my_aes_list)))
  return(my_aes)
}


graph <- ggplot(multipleWeight, aes(x=volatility, y=expectedReturns)) +
  geom_point(aes(color=sharpeRatio))+
  generalPlot(multipleWeight, weightColNames)+
  scale_colour_gradient(low= "red",high = "blue") +
  theme_classic()+
  theme(axis.title = element_text(size = 12),
        plot.title = element_text(size = 16),
        axis.line = element_line(color = "white"),
        text = element_text(color = "white"),
        panel.background = element_rect(fill = "black"),
        plot.background = element_rect(fill = "black"),
        legend.background = element_rect(fill = "black"),
)+
  xlab("Volatility (%)")+
  ylab("Expected Daily Return (%)")+
  ggtitle("Efficient Frontier (Modern Portfolio Theory)")


ggplotly(graph)




#### End Of Efficient Frontier ####