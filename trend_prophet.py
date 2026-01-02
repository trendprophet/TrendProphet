import streamlit as st
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import pandas as pd
from datetime import datetime, timedelta

# 1. Page Configuration
st.set_page_config(
    page_title="TrendProphet | Financial Forecaster", 
    page_icon="📈", 
    layout="wide"
)

# 2. Caching Function
@st.cache_data(ttl=3600)
def load_data(ticker):
    try:
        t = yf.Ticker(ticker)
        df = t.history(period="max")
        if df.empty:
            return None
        df.index = df.index.tz_localize(None)
        return df
    except Exception:
        return None

# 3. Sidebar
with st.sidebar:
    st.title("📈 TrendProphet")
    
    # Added a help tooltip to explain international suffixes
    ticker = st.text_input(
        "Ticker Symbol", 
        "AAPL", 
        help="Use suffixes for international markets: .L (London), .DE (Frankfurt), .T (Tokyo)."
    ).upper()
    
    full_df = load_data(ticker)
    
    if full_df is not None:
        earliest_date = full_df.index.min().date()
        latest_date = full_df.index.max().date()
        total_days = (latest_date - earliest_date).days
        
        st.divider()
        st.subheader("Settings")
        
        # NEW: Date Slider
        # We use a slider representing 'Days from Earliest Available Data'
        days_offset = st.slider(
            "Training Start Date (Slider)",
            min_value=0,
            max_value=total_days - 30, # Leave at least 30 days for training
            value=0, # Default to the very beginning
            help="Slide right to ignore older historical data."
        )
        
        # Convert slider integer back to a displayable date
        training_start = earliest_date + timedelta(days=days_offset)
        st.caption(f"Start Date: **{training_start}**")
        
        horizon = st.slider(
            "Forecast Horizon (Days)", 30, 365, 90,
            help="Slide right to increase forecast horizon."
        )
        
        run_button = st.button("Generate Forecast", use_container_width=True, type="primary")
    else:
        st.error("Ticker not found. Please check the symbol.")
        run_button = False

    st.divider()
    st.write("☕ **Support this project**")
    
    # 1. Buy Me a Coffee Button
    bmc_link = 'https://www.buymeacoffee.com/trendprophet' # Remember to update this!
    button_html = f'''
        <a href="{bmc_link}" target="_blank">
            <img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" 
                 alt="Buy Me A Coffee" 
                 style="height: 40px !important;width: 150px !important;" >
        </a>
    '''
    st.markdown(button_html, unsafe_allow_html=True)

    # 2. Improved Visitor Badge Section
    st.write("") # Spacer
    # Using an HTML <img> tag instead of Markdown "!" syntax is often more reliable in Streamlit
    badge_html = '''
        <img src="https://visitor-badge.laobi.icu/badge?page_id=trendprophet-v1&left_color=grey&right_color=orange" 
             alt="visitor badge">
    '''
    st.markdown(badge_html, unsafe_allow_html=True)

# 4. Main Dashboard
st.title(f"Analysis: {ticker}")

if run_button:
    with st.spinner(f"Analyzing {ticker} patterns..."):
        filtered_df = full_df.loc[str(training_start):].copy()
        
        if len(filtered_df) > 20:
            df_prophet = filtered_df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

            m = Prophet()
            m.fit(df_prophet)
            future = m.make_future_dataframe(periods=horizon)
            forecast = m.predict(future)

            # Metrics
            col1, col2, col3 = st.columns(3)
            current_price = float(df_prophet['y'].iloc[-1])
            col1.metric("Latest Price", f"{current_price:.2f}")
            col2.metric("Training Points", len(df_prophet))
            col3.metric("Training Since", training_start.strftime('%Y-%m-%d'))

            # Tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Forecast", "📂 Data", "🔍 Trends", "📜 About & Legal"])
            
            with tab1:
                st.plotly_chart(plot_plotly(m, forecast), use_container_width=True)
                st.subheader("Statistical Summary")
                start_pred = forecast['yhat'].iloc[-horizon] 
                end_pred = forecast['yhat'].iloc[-1]        
                perc_change = ((end_pred - start_pred) / start_pred) * 100
                
                trend_label = "Bullish (Upward)" if perc_change > 0 else "Bearish (Downward)"
                trend_color = "green" if perc_change > 0 else "red"
                
                st.markdown(f"""
                Based on historical patterns from **{training_start}** to today, the model identifies a **{trend_label}** trend for the next **{horizon}** days. 
                
                * **Projected Change:** <span style="color:{trend_color}; font-weight:bold;">{perc_change:.2f}%</span>
                * **Conservative Estimate:** {forecast['yhat_lower'].iloc[-1]:.2f}
                * **Optimistic Estimate:** {forecast['yhat_upper'].iloc[-1]:.2f}
                
                _Note: The shaded area represents the 80% confidence interval._
                """, unsafe_allow_html=True)
            
            with tab2:
                st.subheader(f"Full {horizon}-Day Forecast Data")
                forecast_export = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(horizon)
                st.dataframe(forecast_export, use_container_width=True)
                csv = forecast_export.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Forecast as CSV", data=csv, file_name=f'{ticker}_forecast.csv', mime='text/csv')
            
            with tab3:
                st.plotly_chart(plot_components_plotly(m, forecast), use_container_width=True)

            with tab4:
                st.header("About TrendProphet")
                st.write("""
                **TrendProphet** is an interactive forecasting dashboard built to provide objective, 
                statistical insights into historical market movements using advanced time-series analysis. 

                **The Developer:**
                I am a developer passionate about making complex financial modeling accessible. 
                This tool was built using Python and Meta's Prophet library to help users 
                visualize data-driven trends without the noise of typical financial media.
                """)
                
                st.subheader("⚖️ Legal Disclaimer")
                st.info("""
                **This tool is for educational and informational purposes only.**
                - **Not Financial Advice:** All data is generated by statistical models. Do not use this as the basis for trading.
                - **No Liability:** The developer is not responsible for financial decisions made based on this tool.
                - **Data Accuracy:** Stock markets are volatile; historical performance does not guarantee future results.
                """)
                
                st.subheader("📚 Credits & Attributions")
                st.markdown("""
                - **Data Source:** [Yahoo Finance](https://finance.yahoo.com) via `yfinance`.
                - **Forecasting Engine:** [Prophet](https://facebook.github.io/prophet/) by Meta Open Source.
                - **Development Partner:** [Gemini 3](https://gemini.google.com) by Google.
                - **UI Framework:** [Streamlit](https://streamlit.io).
                - **Visualization:** [Plotly](https://plotly.com).
                """)
                st.divider()
                st.caption("Version 1.0 | Developed on macOS Big Sur")
            
            st.divider()
            st.caption("Disclaimer: Statistical tool for historical data analysis. Not financial advice.")
        else:
            st.warning("Not enough historical data in the selected range.")
else:
    # --- UPDATED LANDING PAGE TIPS ---
    st.info("👈 Configure settings in the sidebar and click **'Generate Forecast'** to begin.")
    
    st.subheader("Global Market Search Tips")
    st.write("TrendProphet supports major global exchanges via Yahoo Finance tickers:")
    
    # Displaying international examples in a neat table
    market_data = {
        "Exchange": ["US Markets", "London (LSE)", "Frankfurt (XETRA)", "Tokyo (TSE)", "Crypto"],
        "Suffix": ["(No suffix)", ".L", ".DE", ".T", "-USD"],
        "Example": ["AAPL, TSLA", "BP.L, VOD.L", "SAP.DE, BMW.DE", "7203.T (Toyota)", "BTC-USD, ETH-USD"]
    }
    # Display the table without the 0, 1, 2, 3 index column
    st.dataframe(pd.DataFrame(market_data), hide_index=True, use_container_width=True)
    
    st.markdown("""
    > **💡** You can search for a symbol on [Yahoo Finance](https://finance.yahoo.com) 
    > to find the exact ticker string (e.g., 'AstraZeneca' is `AZN.L`).
    """)