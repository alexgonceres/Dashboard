#streamlit run domestic_consumption_dashboard.py

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np  # Add this import at the top of your script
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

def load_data():
    global df
    df = pd.read_csv('weather_and_lights_with_percentage_and_kitchen.csv')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    
    # Ensure all relevant columns are numeric; coerce any non-numeric values to NaN
    numeric_columns = ['Total final consumption / Mean values [W]', 'Kitchen', 'light_amount', 
                       'Fridge_E', 'LCD_TV', 'Games_Console', 'Cell_Phone', 
                       'Washing_machine', 'Computer_desk', 'heating', "Unknown"]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    
    # Drop rows with missing Timestamp or necessary numeric columns
    df.dropna(subset=['Timestamp'] + numeric_columns, inplace=True)
    
    # Extract month, day, and hour for filtering, set Timestamp as index
    df['month'] = df['Timestamp'].dt.month
    df['day'] = df['Timestamp'].dt.day
    df['hour'] = df['Timestamp'].dt.hour
    df.set_index('Timestamp', inplace=True)

def create_daily_distribution_chart():
    """Display daily energy distribution as a stacked bar chart."""
    activity_columns = ['Kitchen', 'light_amount', 'Fridge_E', 'LCD_TV', 'Games_Console', 
                        'Cell_Phone', 'Washing_machine', 'Computer_desk', 'heating', "Unknown"]
    
    min_date = df.index.min().date()
    max_date = df.index.max().date()
    
    start_date, end_date = st.slider(
        'Select date range for daily distribution',
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date)
    )
    
    date_filtered_data = df[(df.index.date >= start_date) & (df.index.date <= end_date)]
    
    fig, ax = plt.subplots(figsize=(13.5, 7.2))  # Reduced size by 10%
    date_filtered_data[activity_columns].resample('D').sum().plot(kind='bar', stacked=True, ax=ax)
    ax.set_title('Daily Energy Distribution')
    ax.set_xlabel('Date')
    ax.set_ylabel('Energy Consumption (W)')
    ax.legend(activity_columns, loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    st.pyplot(fig, use_container_width=True)

def create_hourly_distribution_for_month():
    """Display average hourly energy distribution for a selected month."""
    activity_columns = ['Kitchen', 'light_amount', 'Fridge_E', 'LCD_TV', 'Games_Console', 
                        'Cell_Phone', 'Washing_machine', 'Computer_desk', 'heating', "Unknown"]
    
    selected_month = st.selectbox('Select month for hourly distribution', range(1, 13))
    monthly_data = df[df['month'] == selected_month]
    
    hourly_distribution = monthly_data.groupby('hour')[activity_columns].mean()

    fig, ax = plt.subplots(figsize=(13.5, 7.2))  # Reduced size by 10%
    hourly_distribution.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title(f'Average Hourly Energy Distribution for Month {selected_month}')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Average Energy Consumption (W)')
    ax.legend(activity_columns, loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    st.pyplot(fig, use_container_width=True)

def create_detailed_daily_breakdown():
    """Display detailed breakdown for a specific day in stacked bar form."""
    activity_columns = ['Kitchen', 'light_amount', 'Fridge_E', 'LCD_TV', 'Games_Console', 
                        'Cell_Phone', 'Washing_machine', 'Computer_desk', 'heating', "Unknown"]
    
    selected_month = st.selectbox('Select month', range(1, 13), key="daily_month")
    selected_day = st.selectbox('Select day', df[df['month'] == selected_month]['day'].unique(), key="daily_day")

    daily_data = df[(df['month'] == selected_month) & (df['day'] == selected_day)]

    fig, ax = plt.subplots(figsize=(13.5, 7.2))  # Reduced size by 10%
    daily_data[activity_columns].plot(kind='bar', stacked=True, ax=ax)
    ax.set_title(f'Energy Distribution for {selected_month}/{selected_day}')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Consumption (W)')
    ax.legend(activity_columns, loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    st.pyplot(fig, use_container_width=True)











import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def add_pie_chart(pie_size=3.5, font_size=12, dpi=300, label_distance=1.4, threshold=2.0):
    """
    Display annual and monthly energy distribution with conditional percentage placement.
    """
    
    activity_columns = [
        'Kitchen', 'light_amount', 'Fridge_E', 'LCD_TV', 'Games_Console',
        'Cell_Phone', 'Washing_machine', 'Computer_desk', 'heating', "Unknown"
    ]

    def plot_pie(ax, totals, percentages, labels, title):
        """
        Helper function to plot a pie chart with conditional label placement.
        """
        wedges, texts, autotexts = ax.pie(
            totals,
            startangle=90,
            wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
            autopct=lambda pct: f'{pct:.1f}%' if pct > threshold else '',
            textprops={'fontsize': font_size - 6, 'color': 'white'},
        )

        # Collect outside labels info
        label_positions = []  # Store adjusted positions to avoid overlap

        for i, wedge in enumerate(wedges):
            percentage = percentages.iloc[i]
            theta = (wedge.theta1 + wedge.theta2) / 2.0  # Midpoint angle of the slice

            if percentage <= threshold:
                # Calculate label angle and position to avoid overlap
                theta_rad = np.deg2rad(theta)
                x = label_distance * np.cos(theta_rad)
                y = label_distance * np.sin(theta_rad)

                # Adjust position to prevent overlap
                while any(abs(y - pos[1]) < 0.1 for pos in label_positions):
                    y += 0.1  # Increment y slightly to avoid overlap

                label_positions.append((x, y))

                ha = 'right' if x < 0 else 'left'
                va = 'center'

                # Draw line from pie slice to label
                x0 = wedge.r * np.cos(theta_rad)
                y0 = wedge.r * np.sin(theta_rad)

                ax.annotate(
                    f'{percentage:.1f}%',
                    xy=(x0, y0),  # Point at the edge of the slice
                    xytext=(x, y),  # Position of the percentage text
                    ha=ha,
                    va=va,
                    fontsize=font_size - 6,
                    arrowprops={
                        'arrowstyle': '-',  # Straight line
                        'connectionstyle': 'arc3,rad=0',  # Straight connection
                        'color': 'black',
                        'linewidth': 0.7,
                    }
                )

        # Add legend with activity names only
        ax.legend(
            activity_columns,
            loc='center left',
            bbox_to_anchor=(1.25, 0.5),
            borderaxespad=0,
            fontsize=font_size - 6
        )
        
        ax.set_title(title, fontsize=font_size)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        ax.set_box_aspect(1)  # Ensure aspect ratio is 1:1

    # --- Annual Energy Distribution ---
    st.write("### Annual Energy Distribution")
    annual_totals = df[activity_columns].sum()
    percentages = 100 * annual_totals / annual_totals.sum()
    labels = activity_columns

    fig1, ax1 = plt.subplots(figsize=(pie_size, pie_size), dpi=dpi)
    plot_pie(ax1, annual_totals, percentages, labels, "Annual Energy Distribution (%)")
    plt.subplots_adjust(left=0.1, right=0.7)  # Adjust layout for legend alignment
    fig1.tight_layout()
    st.pyplot(fig1)

    # --- Monthly Energy Distribution ---
    st.write("### Monthly Energy Distribution")
    selected_month = st.selectbox(
        'Select month for pie chart distribution', range(1, 13), key="monthly_pie"
    )
    monthly_totals = df[df['month'] == selected_month][activity_columns].sum()
    percentages_monthly = 100 * monthly_totals / monthly_totals.sum()
    labels_monthly = activity_columns

    fig2, ax2 = plt.subplots(figsize=(pie_size, pie_size), dpi=dpi)
    plot_pie(ax2, monthly_totals, percentages_monthly, labels_monthly, f"Monthly Energy Distribution for Month {selected_month} (%)")
    plt.subplots_adjust(left=0.1, right=0.7)  # Ensure same layout adjustment
    fig2.tight_layout()
    st.pyplot(fig2)

































def add_monthly_distribution_charts():
    """Display 12 small charts for monthly energy distribution together."""
    activity_columns = ['Kitchen', 'light_amount', 'Fridge_E', 'LCD_TV', 'Games_Console', 
                        'Cell_Phone', 'Washing_machine', 'Computer_desk', 'heating', "Unknown"]
    
    st.write("### Monthly Energy Profiles")
    cols = st.columns(4)  # Arrange charts in a 3x4 grid for compact display
    
    for month in range(1, 13):
        with cols[(month - 1) % 4]:  # Ensure charts are displayed in a 3x4 grid
            fig, ax = plt.subplots(figsize=(4, 3))  # Small chart for each month
            monthly_data = df[df['month'] == month].groupby('hour')[activity_columns].mean()
            monthly_data.plot(kind='bar', stacked=True, ax=ax)
            ax.set_title(f'Month {month}')
            ax.set_xlabel('Hour')
            ax.set_ylabel('Avg Consumption (W)')
            ax.legend().set_visible(False)
            st.pyplot(fig)

def main():
    
    st.set_page_config(page_title='Domestic Consumption Dashboard', layout="wide")
    st.title('Domestic Consumption Data Dashboard')

    load_data()
    create_daily_distribution_chart()
    create_hourly_distribution_for_month()
    create_detailed_daily_breakdown()
    add_pie_chart()  # Remove parameters here
    add_monthly_distribution_charts()




if __name__ == '__main__':
    main()
