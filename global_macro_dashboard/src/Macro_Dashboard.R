library(shiny)
library(shinydashboard)
library(data.table)
library(ggplot2)
library(dplyr)
library(scales)
library(plotly)


# LOAD & PREPARE DATA

Macro_Data <- fread('C:\\Users\\pears\\OneDrive\\Desktop\\MACRO\\Daily.csv')
Macro_Data$Date <- as.Date(Macro_Data$Date)

# Automatically determine date range

latest_date <- max(Macro_Data$Date, na.rm = TRUE)
earliest_date <- min(Macro_Data$Date, na.rm = TRUE)

# Z-score scaling

Macro_Data_Scaled_0 <- Macro_Data[, 1]
Macro_Data_Scaled_1 <- Macro_Data[, -1]
Macro_Data_Scaled <- as.data.frame(scale(Macro_Data_Scaled_1))
Macro_Data_Scaled <- cbind(Macro_Data_Scaled_0, Macro_Data_Scaled)
Macro_Data_Scaled$Date <- as.Date(Macro_Data_Scaled$Date)

Variable_List <- names(Macro_Data)
Daily_Dataframe <- Macro_Data

# USER INTERFACE

ui <- dashboardPage(
  skin = "blue",
  
  dashboardHeader(
    title = span("Macro Dashboard", class = "header-title"),
    titleWidth = 250
  ),
  
  dashboardSidebar(
    width = 80,
    disable = TRUE
  ),
  
  dashboardBody(
    tags$div(id = "right-ribbon"),
    
    # STYLES
    
    tags$style(HTML("
      .main-header { background:#000 !important; border-bottom:3px solid #007BFF !important; }
      .main-header .navbar { display:none !important; }
      .main-header .logo {
        background:#000 !important; color:#fff !important; width:100% !important;
        left:0; right:0; position:fixed; text-align:center; font-weight:700;
        font-size:22px; letter-spacing:1px; height:50px; line-height:50px;
        border:0 !important; z-index:1101;
      }
      .main-header .logo .header-title { color:#fff !important; }
      .content-wrapper, .right-side { background:#F7F8FA; margin-left:80px; margin-right:80px; padding-top:60px; }
      .main-sidebar { background:#000 !important; width:80px !important; }
      #right-ribbon{ position:fixed; top:0; right:0; width:80px; height:100%; background:#000; z-index:1100; }
      .box.box-solid>.box-header{ color:#fff!important; background:#0A2342!important; border-color:#0A2342!important; }
      .box.box-solid>.box-header .box-title{ color:#fff!important; font-weight:600; }
      .box.box-solid{ border:1px solid #0A2342!important; }
      .box{ box-shadow:0 2px 4px rgba(0,0,0,0.1); }
      .small-box{ min-height:130px!important; }
      .small-box h3{ font-size:26px; font-weight:700; margin:0 0 6px 0; }
      .small-box p{ font-size:13px; margin:0; }
      .form-group{ margin-bottom:6px; }
      .selectize-input{ min-height:34px; font-size:13px; }
      .selectize-dropdown-content{ font-size:13px; }
    ")),
    

    # Controls

    fluidRow(
      box(width=12, solidHeader=TRUE,
          fluidRow(
            column(width=2, dateInput("date_1", "Start", value=earliest_date)),
            column(width=2, dateInput("date_2", "End", value=latest_date)),
            column(width=2, selectInput("chart_var1", "Chart 1", Variable_List, "S_and_P_Close")),
            column(width=2, selectInput("chart_var2", "Chart 2", Variable_List, "FTSE_Close")),
            column(width=2, selectInput("chart_var3", "Chart 3", Variable_List, "Nasdaq_Close"))
          ),
          fluidRow(
            column(width=2, selectInput("chart_var4", "Chart 4", Variable_List, "Gold")),
            column(width=2, selectInput("chart_var5", "Chart 5", Variable_List, "Brent")),
            column(width=2, selectInput("var_1", "Blue Variable", Variable_List, "Nasdaq_Close")),
            column(width=2, selectInput("var_2", "Red Variable", Variable_List, "S_and_P_Close")),
            column(width=2, selectInput("var_3", "Green Variable", Variable_List, "FTSE_Close"))
          )
      )
    ),
    

    # COLLAPSIBLE VALUE BOX PANEL

    box(title="📊 Market Summary", solidHeader=TRUE, width=12, collapsible=TRUE, collapsed=TRUE,
        fluidRow(
          tags$h4("📈 Major Indices", style="margin-left:15px; font-weight:600;"),
          valueBoxOutput("vbox_sp", width=3),
          valueBoxOutput("vbox_nasdaq", width=3),
          valueBoxOutput("vbox_ftse", width=3),
          valueBoxOutput("vbox_nikkei", width=3)
        ),
        hr(),
        fluidRow(
          tags$h4("💰 Treasury Yields", style="margin-left:15px; font-weight:600;"),
          valueBoxOutput("vbox_y3", width=2),
          valueBoxOutput("vbox_y2", width=2),
          valueBoxOutput("vbox_y5", width=2),
          valueBoxOutput("vbox_y10", width=2),
          valueBoxOutput("vbox_y20", width=2)
        ),
        hr(),
        fluidRow(
          tags$h4("💹 FX Rates", style="margin-left:15px; font-weight:600;"),
          valueBoxOutput("vbox_fx1", width=3),
          valueBoxOutput("vbox_fx2", width=3),
          valueBoxOutput("vbox_fx3", width=3),
          valueBoxOutput("vbox_fx4", width=3)
        ),
        hr(),
        fluidRow(
          tags$h4("⚙️ Commodities", style="margin-left:15px; font-weight:600;"),
          valueBoxOutput("vbox_brent", width=2),
          valueBoxOutput("vbox_wti", width=2),
          valueBoxOutput("vbox_gold", width=2),
          valueBoxOutput("vbox_silver", width=2),
          valueBoxOutput("vbox_copper", width=2)
        )
    ),
    

    # CHARTS ROW 1

    fluidRow(
      box(width=4, height="300px", title=textOutput("chart_yield_title"), solidHeader=TRUE, plotlyOutput("yield_curve_plot", height="260px")),
      box(width=4, height="300px", title=textOutput("chart1_title"), solidHeader=TRUE, plotlyOutput("line", height="260px")),
      box(width=4, height="300px", title=textOutput("chart2_title"), solidHeader=TRUE, plotlyOutput("line_2", height="260px"))
    ),
    

    # CHARTS ROW 2

    fluidRow(
      box(width=4, height="300px", title=textOutput("chart3_title"), solidHeader=TRUE, plotlyOutput("line_3a", height="260px")),
      box(width=4, height="300px", title=textOutput("chart4_title"), solidHeader=TRUE, plotlyOutput("line_5", height="260px")),
      box(width=4, height="300px", title=textOutput("chart5_title"), solidHeader=TRUE, plotlyOutput("line_6", height="260px"))
    ),
    

    # COMPARISON GRAPH

    fluidRow(
      box(width=12, title=textOutput("comparison_title"), solidHeader=TRUE, plotlyOutput("comparison_plot", height="420px"))
    )
  )
)



# SERVER

server <- function(input, output) {
  
  clean_name <- function(x) gsub("_Close", "", x)
  
  # Dynamic titles
  
  output$chart_yield_title <- renderText("Chart 1 — Yield Curve (3M–20Y)")
  output$chart1_title <- renderText({ paste("Chart 2 —", clean_name(input$chart_var1)) })
  output$chart2_title <- renderText({ paste("Chart 3 —", clean_name(input$chart_var2)) })
  output$chart3_title <- renderText({ paste("Chart 4 —", clean_name(input$chart_var3)) })
  output$chart4_title <- renderText({ paste("Chart 5 —", clean_name(input$chart_var4)) })
  output$chart5_title <- renderText({ paste("Chart 6 —", clean_name(input$chart_var5)) })
  
  output$comparison_title <- renderText({
    paste("Z-Score Comparison:", clean_name(input$var_1), "(Blue) vs",
          clean_name(input$var_2), "(Red) vs", clean_name(input$var_3), "(Green)")
  })
  
  make_line <- function(var, col){
    ggplot(Macro_Data %>% filter(between(Date, input$date_1, input$date_2)),
           aes_string(x="Date", y=var)) +
      geom_line(color=col, size=0.8) +
      labs(y="Value", x="") +
      theme_minimal()
  }
  
  # Charts
  
  output$line   <- renderPlotly({ ggplotly(make_line(input$chart_var1,"steelblue")) })
  output$line_2 <- renderPlotly({ ggplotly(make_line(input$chart_var2,"darkorange")) })
  output$line_3a <- renderPlotly({ ggplotly(make_line(input$chart_var3,"forestgreen")) })
  output$line_5 <- renderPlotly({ ggplotly(make_line(input$chart_var4,"goldenrod")) })
  output$line_6 <- renderPlotly({ ggplotly(make_line(input$chart_var5,"deeppink")) })
  
  # Yield curve
  
  output$yield_curve_plot <- renderPlotly({
    yields <- c("Three_Month_Yield", "Two_Year_Yield", "Five_Year_Yield", "Ten_Year_Yield", "Twenty_Year_Yield")
    latest <- tail(Macro_Data[, ..yields], 1)
    
    maturities <- c("3M","2Y","5Y","10Y","20Y")
    df <- data.frame(
      Maturity = factor(maturities, levels=maturities),
      Yield = as.numeric(latest[1, ])
    )
    
    p <- ggplot(df, aes(x=Maturity, y=Yield, group=1)) +
      geom_line(color="#007BFF", size=1.1) +
      geom_point(size=2, color="#007BFF") +
      labs(y="Yield (%)", x=NULL) +
      theme_minimal()
    
    ggplotly(p)
  })
  
  # Comparison
  
  output$comparison_plot <- renderPlotly({
    p <- ggplot(Macro_Data_Scaled %>%
                  filter(between(Date,input$date_1,input$date_2)), aes(x=Date)) +
      geom_line(aes_string(y=input$var_1), color="steelblue", size=0.8) +
      geom_line(aes_string(y=input$var_2), color="darkred", size=0.8) +
      geom_line(aes_string(y=input$var_3), color="darkgreen", size=0.8) +
      labs(y=NULL, x=NULL) +
      theme_minimal()
    ggplotly(p)
  })
  
  # Value Boxes
  
  make_vbox <- function(val, sub, col, iconname)
    valueBox(val, subtitle=sub, color=col, icon=icon(iconname), width=3)
  
  output$vbox_sp      <- renderValueBox({ make_vbox(round(tail(na.omit(Daily_Dataframe$S_and_P_Close),1),2),"S&P 500","blue","chart-line") })
  output$vbox_nasdaq  <- renderValueBox({ make_vbox(round(tail(na.omit(Daily_Dataframe$Nasdaq_Close),1),2),"Nasdaq","purple","chart-area") })
  output$vbox_ftse    <- renderValueBox({ make_vbox(round(tail(na.omit(Daily_Dataframe$FTSE_Close),1),2),"FTSE 100","teal","chart-bar") })
  output$vbox_nikkei  <- renderValueBox({ make_vbox(round(tail(na.omit(Daily_Dataframe$Nikkei_Close),1),2),"Nikkei 225","red","globe-asia") })
  
  output$vbox_y3  <- renderValueBox({ make_vbox(paste0(round(tail(na.omit(Daily_Dataframe$Three_Month_Yield),1),2),"%"),"3-Month","navy","percent") })
  output$vbox_y2  <- renderValueBox({ make_vbox(paste0(round(tail(na.omit(Daily_Dataframe$Two_Year_Yield),1),2),"%"),"2-Year","navy","percent") })
  output$vbox_y5  <- renderValueBox({ make_vbox(paste0(round(tail(na.omit(Daily_Dataframe$Five_Year_Yield),1),2),"%"),"5-Year","navy","percent") })
  output$vbox_y10 <- renderValueBox({ make_vbox(paste0(round(tail(na.omit(Daily_Dataframe$Ten_Year_Yield),1),2),"%"),"10-Year","navy","percent") })
  output$vbox_y20 <- renderValueBox({ make_vbox(paste0(round(tail(na.omit(Daily_Dataframe$Twenty_Year_Yield),1),2),"%"),"20-Year","navy","percent") })
  
  output$vbox_fx1 <- renderValueBox({ make_vbox(round(tail(na.omit(Daily_Dataframe$GBPUSD_Close),1),4),"GBP/USD","green","pound-sign") })
  output$vbox_fx2 <- renderValueBox({ make_vbox(round(tail(na.omit(Daily_Dataframe$EURUSD_Close),1),4),"EUR/USD","green","euro-sign") })
  output$vbox_fx3 <- renderValueBox({ make_vbox(round(tail(na.omit(Daily_Dataframe$USDJPY_Close),1),2),"USD/JPY","green","yen-sign") })
  output$vbox_fx4 <- renderValueBox({ make_vbox(round(tail(na.omit(Daily_Dataframe$USDCHF_Close),1),4),"USD/CHF","green","dollar-sign") })
  
  output$vbox_brent  <- renderValueBox({ make_vbox(paste0("$",round(tail(na.omit(Daily_Dataframe$Brent),1),2)),"Brent Crude","black","fire") })
  output$vbox_wti    <- renderValueBox({ make_vbox(paste0("$",round(tail(na.omit(Daily_Dataframe$WTI),1),2)),"WTI Crude","black","oil-can") })
  output$vbox_gold   <- renderValueBox({ make_vbox(paste0("$",round(tail(na.omit(Daily_Dataframe$Gold),1),2)),"Gold","yellow","gem") })
  output$vbox_silver <- renderValueBox({ make_vbox(paste0("$",round(tail(na.omit(Daily_Dataframe$Silver),1),2)),"Silver","light-blue","ring") })
  output$vbox_copper <- renderValueBox({ make_vbox(paste0("$",round(tail(na.omit(Daily_Dataframe$Copper),1),2)),"Copper","orange","coins") })
}


# RUN APP

shinyApp(ui=ui, server=server)
