# Load necessary libraries
library(forecast)
library(zoo)
library(knitr)
library(ggplot2)
library(tseries)

# Load data and rename columns correctly
passenger.data <- read.csv("AirPassengers.csv", stringsAsFactors = FALSE)
colnames(passenger.data) <- c("Month", "Passengers")  # Rename column properly

# Convert "Month" column to Date format
passenger.data$Month <- as.Date(paste0(passenger.data$Month, "-01"), format="%Y-%m-%d")
passenger.data$Passengers <- as.numeric(passenger.data$Passengers)
passenger.data$Passengers[is.na(passenger.data$Passengers)] <- median(passenger.data$Passengers, na.rm=TRUE)

# Create the time series object
passenger.ts <- ts(passenger.data$Passengers, start=c(1949, 1), frequency=12)
passenger.data <- passenger.data[order(passenger.data$Month), ]

# Define start and end period manually
start_year <- as.numeric(format(min(passenger.data$Month), "%Y"))
start_month <- as.numeric(format(min(passenger.data$Month), "%m"))
end_year <- as.numeric(format(max(passenger.data$Month), "%Y"))
end_month <- as.numeric(format(max(passenger.data$Month), "%m"))

# Create the time series object
passenger.ts <- ts(passenger.data$Passengers, 
                   start=c(start_year, start_month), 
                   end=c(end_year, end_month),
                   frequency=12)

# Plot the time series
autoplot(passenger.ts) +
  ggtitle("Monthly Airline Passengers (1949-1960)") +
  xlab("Year") + ylab("Number of Passengers") +
  theme_minimal()

# Plot the time series data
autoplot(passenger.ts) +
  ggtitle("Monthly Airline Passengers (1949-1960)") +
  xlab("Year") + ylab("Number of Passengers") +
  theme_minimal()

# Decompose the time series (multiplicative)
decomposed <- decompose(passenger.ts, type = "multiplicative")
autoplot(decomposed) +
  ggtitle("Decomposition of Air Passengers Time Series") +
  theme_minimal()

# Perform ADF test to check stationarity
adf.test(passenger.ts)

# Apply log transformation to stabilize variance
log_passenger.ts <- log(passenger.ts)
log_decomposed <- decompose(log_passenger.ts, type = "additive")

# Plot decomposition after log transformation
autoplot(log_decomposed) +
  ggtitle("Decomposition of Log Transformed Air Passengers Time Series") +
  theme_minimal()

# DATA PARTITIONING
# Define training split (80% training)
train_length <- floor(0.8 * length(log_passenger.ts))
train.ts <- window(log_passenger.ts, end=c(1949 + (train_length/12), (train_length %% 12)))
valid.ts <- window(log_passenger.ts, start=c(1949 + (train_length/12), (train_length %% 12) + 1))

# Plot training vs validation sets
autoplot(train.ts) + autolayer(valid.ts, series="Validation", color="red") +
  ggtitle("Training & Validation Sets for Air Passengers") +
  xlab("Year") + ylab("Log Passengers") +
  theme_minimal()

# HOLT-WINTERS MODEL
hw_model <- ets(train.ts, model="AAA")
hw_forecast <- forecast(hw_model, h=length(valid.ts))
summary(hw_model)

# Check residuals for diagnostic validation
checkresiduals(hw_model)

# Plot Holt-Winters forecast vs validation data
autoplot(hw_forecast) +
  autolayer(valid.ts, series="Validation Data", color="red") +
  ggtitle("Holt-Winters Forecast vs Validation Data") +
  xlab("Year") + ylab("Log Passengers") +
  theme_minimal()

# ARIMA MODEL
auto_arima_model <- auto.arima(train.ts)
arima_forecast <- forecast(auto_arima_model, h=length(valid.ts))
summary(auto_arima_model)

# Check residuals for ARIMA diagnostic validation
checkresiduals(auto_arima_model)

# Plot ARIMA forecast vs validation data
autoplot(arima_forecast) +
  autolayer(valid.ts, series="Validation Data", color="red") +
  ggtitle("ARIMA Forecast vs Validation Data") +
  xlab("Year") + ylab("Log Passengers") +
  theme_minimal()

# MODEL COMPARISON
arima_coefs <- auto_arima_model$coef

# Ensure AR, MA terms exist safely
ar_terms <- if(length(grep("ar", names(arima_coefs))) > 0) {
  paste(round(arima_coefs[grep("ar", names(arima_coefs))], 4), collapse=", ")
} else { "None" }

ma_terms <- if(length(grep("ma", names(arima_coefs))) > 0) {
  paste(round(arima_coefs[grep("ma", names(arima_coefs))], 4), collapse=", ")
} else { "None" }

# Ensure Holt-Winters smoothing parameters exist
alpha_val <- ifelse(!is.null(hw_model$par[1]), hw_model$par[1], NA)
beta_val  <- ifelse(!is.null(hw_model$par[2]), hw_model$par[2], NA)
gamma_val <- ifelse(!is.null(hw_model$par[3]), hw_model$par[3], NA)

# Create Holt-Winters model dataframe
hw_metrics <- data.frame(
  Model = "Holt-Winters",
  RMSE = hw_rmse,
  MAPE = hw_mape,
  Alpha = round(alpha_val, 4),
  Beta = round(beta_val, 4),
  Gamma = round(gamma_val, 4),
  AR = "N/A",
  MA = "N/A",
  AIC = round(hw_model$aic, 2),
  BIC = "N/A"
)

# Create ARIMA model dataframe
arima_metrics <- data.frame(
  Model = "ARIMA",
  RMSE = arima_rmse,
  MAPE = arima_mape,
  Alpha = "N/A",
  Beta = "N/A",
  Gamma = "N/A",
  AR = ar_terms,
  MA = ma_terms,
  AIC = round(auto_arima_model$aic, 2),
  BIC = round(auto_arima_model$bic, 2)
)

# Combine model comparison into one table
model_comparison <- rbind(hw_metrics, arima_metrics)
print(model_comparison)

# FINAL FORECASTING WITH ARIMA
final_forecast <- forecast(auto_arima_model, h=12)

# Plot forecast in log scale
autoplot(final_forecast) +
  ggtitle("Final ARIMA Forecast for Air Passengers (Log Scale)") +
  xlab("Year") + ylab("Log Passengers") +
  theme_minimal()

# Convert actual data and forecasted values back to original scale
original_ts <- exp(log_passenger.ts)
final_forecast_original <- final_forecast  # Copy forecast data

# Transform only the forecasted mean to avoid exponentiation of extreme negatives
final_forecast_original$mean <- exp(final_forecast$mean)
final_forecast_original$lower <- exp(pmax(final_forecast$lower, -10))  # Limit extreme negatives
final_forecast_original$upper <- exp(final_forecast$upper)

# Ensure final transformed values make sense
autoplot(original_ts) +
  autolayer(final_forecast_original$mean, series="ARIMA Forecast", color="blue") +
  ggtitle("Final ARIMA Forecast in Original Scale") +
  xlab("Year") + ylab("Number of Passengers") +
  theme_minimal()













