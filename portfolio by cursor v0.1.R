# 加载必要的包
library(tidyverse)
library(PortfolioAnalytics)
library(PerformanceAnalytics)
library(quadprog)
library(ggplot2)

# 读取CSV文件
# 假设您的CSV文件名为'stock_returns.csv'
stock_data <- read.csv("Simulation.csv", header = TRUE, row.names = 1)

# 计算每支股票的年化收益率和波动率
annual_returns <- colMeans(stock_data) * 12
annual_risk <- apply(stock_data, 2, sd) * sqrt(12)

# 创建性能指标数据框
performance_df <- data.frame(
  Symbol = names(annual_returns),
  Return = annual_returns,
  Risk = annual_risk,
  Sharpe = annual_returns / annual_risk
)

# 选择表现最好的5支股票（基于夏普比率）
top_5_stocks <- performance_df %>%
  arrange(desc(Sharpe)) %>%
  head(5)

# 提取这5支股票的收益数据
selected_returns <- stock_data[, top_5_stocks$Symbol]

# 计算有效前沿
# 设置投资组合参数
n_portfolios <- 1000
weights_matrix <- matrix(NA, nrow = n_portfolios, ncol = 5)
portfolio_returns <- numeric(n_portfolios)
portfolio_risk <- numeric(n_portfolios)

# 模拟不同权重的组合
set.seed(123)
for(i in 1:n_portfolios) {
  weights <- runif(5)
  weights <- weights/sum(weights)
  weights_matrix[i,] <- weights
  portfolio_returns[i] <- sum(weights * annual_returns[top_5_stocks$Symbol])
  portfolio_risk[i] <- sqrt(t(weights) %*% cov(selected_returns) %*% weights)
}

# 创建有效前沿图
portfolio_df <- data.frame(
  Return = portfolio_returns,
  Risk = portfolio_risk
)

# 绘制有效前沿
ggplot(portfolio_df, aes(x = Risk, y = Return)) +
  geom_point(alpha = 0.5, color = "blue") +
  geom_point(data = top_5_stocks, aes(x = Risk, y = Return), color = "red", size = 3) +
  geom_text(data = top_5_stocks, aes(x = Risk, y = Return, label = Symbol), 
            vjust = -1, size = 3) +
  theme_minimal() +
  labs(title = "投资组合有效前沿",
       x = "风险 (标准差)",
       y = "预期收益率")

# 计算最优投资组合（最高夏普比率）
rf_rate <- 0.02 
sharpe_ratios <- (portfolio_returns - rf_rate) / portfolio_risk
optimal_portfolio <- which.max(sharpe_ratios)

# 输出投资组合统计信息
cat("\n投资组合统计分析：\n")
cat("\n1. 选中的五支股票：\n")
print(top_5_stocks)

cat("\n2. 最优投资组合权重：\n")
optimal_weights <- data.frame(
  Stock = top_5_stocks$Symbol,
  Weight = weights_matrix[optimal_portfolio,]
)
print(optimal_weights)

cat("\n3. 最优投资组合表现：\n")
cat("预期年化收益率：", portfolio_returns[optimal_portfolio], "\n")
cat("投资组合风险：", portfolio_risk[optimal_portfolio], "\n")
cat("夏普比率：", sharpe_ratios[optimal_portfolio], "\n")

# 计算相关性矩阵
correlation_matrix <- cor(selected_returns)
print("\n4. 相关性矩阵：")
print(correlation_matrix)