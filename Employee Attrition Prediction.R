# Load necessary libraries
library(dplyr)  # For data manipulation
library(ggplot2)  # For visualization
library(caret)  # For machine learning models
library(pROC)  # For ROC curve
library(shiny)  # For building web applications

# Read the data
data <- read.csv("HR_Analytics.csv")  # Assuming the data is in a CSV file

# Check column names
colnames(data)

# Select relevant columns
relevant_cols <- c("Attrition", "BusinessTravel", "Department", "EducationField", 
                   "Gender", "JobRole", "MaritalStatus", "OverTime", 
                   "Age", "DailyRate", "DistanceFromHome", 
                   "Education", "EnvironmentSatisfaction", "HourlyRate", 
                   "JobInvolvement", "JobLevel", "JobSatisfaction", "MonthlyIncome", 
                   "MonthlyRate", "NumCompaniesWorked", "PercentSalaryHike", 
                   "PerformanceRating", "StockOptionLevel", 
                   "TotalWorkingYears", "TrainingTimesLastYear", "WorkLifeBalance", 
                   "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion", 
                   "YearsWithCurrManager")

data <- data[, relevant_cols]

# Data preprocessing
# Convert factors to characters
factor_cols <- c("Attrition", "BusinessTravel", "Department", "EducationField", 
                 "Gender", "JobRole", "MaritalStatus", "OverTime")
data[factor_cols] <- lapply(data[factor_cols], as.character)

# Convert categorical variables to factors
data[factor_cols] <- lapply(data[factor_cols], as.factor)

# Split the data into training and testing sets
set.seed(123)  # Set seed for reproducibility
train_index <- createDataPartition(data$Attrition, p = 0.7, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Modeling - Logistic Regression
# Train the model
model <- glm(Attrition ~ ., data = train_data, family = "binomial")

# Predictions
predictions <- predict(model, newdata = test_data, type = "response")

# Convert probabilities to binary predictions
threshold <- 0.5
binary_predictions <- ifelse(predictions > threshold, "Yes", "No")

# Model evaluation
cm <- confusionMatrix(factor(binary_predictions, levels = levels(test_data$Attrition)), test_data$Attrition)
print(cm)

# ROC curve
roc_obj <- roc(test_data$Attrition, as.numeric(predictions))
auc_score <- auc(roc_obj)
print(paste("AUC Score:", auc_score))

# Plot ROC curve
plot(roc_obj, main = "ROC Curve", col = "blue", lwd = 2)
lines(x = c(0, 1), y = c(0, 1), col = "red", lty = 2, lwd = 2)
legend("bottomright", legend = paste("AUC =", round(auc_score, 2)), col = "blue", lwd = 2)

# Shiny Dashboard for Prediction
ui <- fluidPage(
  titlePanel("Employee Attrition Prediction"),
  sidebarLayout(
    sidebarPanel(
      width = 12,  # Increased width of the sidebar panel further
      fluidRow(
        column(3,
               # Input fields
               selectInput("BusinessTravel", "Business Travel", choices = unique(data$BusinessTravel), width = "100%"),
               selectInput("Department", "Department", choices = unique(data$Department), width = "100%"),
               selectInput("EducationField", "Education Field", choices = unique(data$EducationField), width = "100%"),
               selectInput("Gender", "Gender", choices = unique(data$Gender), width = "100%")
        ),
        column(3,
               # Input fields continued
               selectInput("JobRole", "Job Role", choices = unique(data$JobRole), width = "100%"),
               selectInput("MaritalStatus", "Marital Status", choices = unique(data$MaritalStatus), width = "100%"),
               selectInput("OverTime", "Over Time", choices = unique(data$OverTime), width = "100%"),
               selectInput("EnvironmentSatisfaction", "Environment Satisfaction", choices = unique(data$EnvironmentSatisfaction), width = "100%")
        ),
        column(3,
               # Input fields continued
               numericInput("Age", "Age", value = 30, min = 18, max = 65, width = "100%"),
               numericInput("DailyRate", "Daily Rate", value = 500, min = 0, max = 2000, width = "100%"),
               numericInput("DistanceFromHome", "Distance From Home (miles)", value = 5, min = 0, max = 30, width = "100%"),
               numericInput("Education", "Education Level", value = 3, min = 1, max = 5, width = "100%")
        ),
        column(3,
               # Input fields continued
               numericInput("MonthlyIncome", "Monthly Income", value = 5000, min = 1000, max = 20000, width = "100%"),
               numericInput("TotalWorkingYears", "Total Working Years", value = 10, min = 0, max = 40, width = "100%"),
               numericInput("YearsAtCompany", "Years at Company", value = 5, min = 0, max = 20, width = "100%"),
               numericInput("HourlyRate", "Hourly Rate", value = 20, min = 0, max = 100, width = "100%")
        )
      ),
      fluidRow(
        column(3,
               # Input fields continued
               numericInput("JobInvolvement", "Job Involvement", value = 3, min = 1, max = 4, width = "100%"),
               numericInput("JobLevel", "Job Level", value = 2, min = 1, max = 5, width = "100%"),  # Include JobLevel
               numericInput("JobSatisfaction", "Job Satisfaction", value = 3, min = 1, max = 4, width = "100%"),
               numericInput("MonthlyRate", "Monthly Rate", value = 8000, min = 1000, max = 25000, width = "100%")
        ),
        column(3,
               # Input fields continued
               numericInput("NumCompaniesWorked", "Number of Companies Worked", value = 2, min = 0, max = 10, width = "100%"),
               numericInput("PercentSalaryHike", "Percent Salary Hike", value = 15, min = 0, max = 30, width = "100%"),
               numericInput("PerformanceRating", "Performance Rating", value = 3, min = 1, max = 4, width = "100%"),
               numericInput("StockOptionLevel", "Stock Option Level", value = 0, min = 0, max = 3, width = "100%")
        ),
        column(3,
               # Input fields continued
               numericInput("TrainingTimesLastYear", "Training Times Last Year", value = 2, min = 0, max = 6, width = "100%"),
               numericInput("WorkLifeBalance", "Work Life Balance", value = 3, min = 1, max = 4, width = "100%"),
               numericInput("YearsInCurrentRole", "Years in Current Role", value = 3, min = 0, max = 20, width = "100%"),
               numericInput("YearsSinceLastPromotion", "Years Since Last Promotion", value = 1, min = 0, max = 15, width = "100%")
        ),
        column(3,
               # Input fields continued
               numericInput("YearsWithCurrManager", "Years with Current Manager", value = 3, min = 0, max = 20, width = "100%")
        )
      ),
      actionButton("predictButton", "Predict Attrition", width = "100%")
    ),
    mainPanel(
      textOutput("prediction"),  # Prediction output
      plotOutput("plot"),  # Bar plot
      fluidRow(
        column(6, selectInput("x_variable", "X-Axis Variable", choices = c("Age", "DailyRate", "DistanceFromHome",
                                                                           "Education", "HourlyRate", "MonthlyIncome",
                                                                           "TotalWorkingYears", "YearsAtCompany"), width = "100%")),
        column(6, selectInput("y_variable", "Y-Axis Variable", choices = c("JobRole", "EnvironmentSatisfaction",
                                                                           "JobInvolvement", "JobLevel", "JobSatisfaction",
                                                                           "NumCompaniesWorked", "PercentSalaryHike",
                                                                           "PerformanceRating", "StockOptionLevel",
                                                                           "TrainingTimesLastYear", "WorkLifeBalance",
                                                                           "YearsInCurrentRole", "YearsSinceLastPromotion",
                                                                           "YearsWithCurrManager"), width = "100%"))
      )
    )
  )
)

server <- function(input, output) {
  # Function to make predictions
  predict_attrition <- function(business_travel, department, education_field, gender,
                                job_role, marital_status, overtime, environment_satisfaction,
                                age, daily_rate, distance_from_home, education, monthly_income,
                                total_working_years, years_at_company, hourly_rate, job_involvement,
                                job_level, job_satisfaction, monthly_rate, num_companies_worked,
                                percent_salary_hike, performance_rating, stock_option_level,
                                training_times_last_year, work_life_balance, years_in_current_role,
                                years_since_last_promotion, years_with_curr_manager) {
    new_data <- data.frame(
      BusinessTravel = factor(business_travel, levels = unique(data$BusinessTravel)),
      Department = factor(department, levels = unique(data$Department)),
      EducationField = factor(education_field, levels = unique(data$EducationField)),
      Gender = factor(gender, levels = unique(data$Gender)),
      JobRole = factor(job_role, levels = unique(data$JobRole)),
      MaritalStatus = factor(marital_status, levels = unique(data$MaritalStatus)),
      OverTime = factor(overtime, levels = unique(data$OverTime)),
      EnvironmentSatisfaction = as.integer(environment_satisfaction),  # Change to integer
      Age = age,
      DailyRate = daily_rate,
      DistanceFromHome = distance_from_home,
      Education = education,
      MonthlyIncome = monthly_income,
      TotalWorkingYears = total_working_years,
      YearsAtCompany = years_at_company,
      HourlyRate = hourly_rate,
      JobInvolvement = job_involvement,
      JobLevel = job_level,
      JobSatisfaction = job_satisfaction,
      MonthlyRate = monthly_rate,
      NumCompaniesWorked = num_companies_worked,
      PercentSalaryHike = percent_salary_hike,
      PerformanceRating = performance_rating,
      StockOptionLevel = stock_option_level,
      TrainingTimesLastYear = training_times_last_year,
      WorkLifeBalance = work_life_balance,
      YearsInCurrentRole = years_in_current_role,
      YearsSinceLastPromotion = years_since_last_promotion,
      YearsWithCurrManager = years_with_curr_manager
    )
    
    prediction <- predict(model, newdata = new_data, type = "response")
    ifelse(prediction > threshold, "Predicted Attrition: Yes", "Predicted Attrition: No")
  }
  
  # Output prediction result
  output$prediction <- renderText({
    if(input$predictButton > 0) {
      predict_attrition(input$BusinessTravel, input$Department, input$EducationField,
                        input$Gender, input$JobRole, input$MaritalStatus, input$OverTime,
                        input$EnvironmentSatisfaction, input$Age, input$DailyRate,
                        input$DistanceFromHome, input$Education, input$MonthlyIncome,
                        input$TotalWorkingYears, input$YearsAtCompany, input$HourlyRate,
                        input$JobInvolvement, input$JobLevel, input$JobSatisfaction,
                        input$MonthlyRate, input$NumCompaniesWorked, input$PercentSalaryHike,
                        input$PerformanceRating, input$StockOptionLevel, input$TrainingTimesLastYear,
                        input$WorkLifeBalance, input$YearsInCurrentRole, input$YearsSinceLastPromotion,
                        input$YearsWithCurrManager)
    }
  })
  
  output$plot <- renderPlot({
    if(input$predictButton > 0) {
      ggplot(data, aes_string(x = input$x_variable, y = input$y_variable, fill = "Attrition")) +
        geom_bar(stat = "identity") +
        labs(title = "Attrition Distribution by Selected Variables")  # Add title here
    }
  })
  
  
}

shinyApp(ui = ui, server = server)


