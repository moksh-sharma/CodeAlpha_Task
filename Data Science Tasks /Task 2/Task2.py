import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from calendar import month_name

df1 = pd.read_csv("Task 2/Unemployment in India.csv")
df2 = pd.read_csv("Task 2/Unemployment_Rate_upto_11_2020.csv")

df1.columns = df1.columns.str.strip()
df2.columns = df2.columns.str.strip()

df1['Date'] = pd.to_datetime(df1['Date'], dayfirst=True, errors='coerce')
df2['Date'] = pd.to_datetime(df2['Date'], dayfirst=True, errors='coerce')

common_columns = [
    'Region', 'Date', 'Estimated Unemployment Rate (%)',
    'Estimated Employed', 'Estimated Labour Participation Rate (%)'
]
df1_clean = df1[common_columns].copy()
df2_clean = df2[common_columns].copy()

combined_df = pd.concat([df1_clean, df2_clean], ignore_index=True)
combined_df.dropna(inplace=True)
combined_df.sort_values(by='Date', inplace=True)

combined_df['Month'] = combined_df['Date'].dt.month_name()
combined_df['Covid_Phase'] = combined_df['Date'].apply(
    lambda x: 'Pre-COVID' if x < pd.to_datetime(
        '2020-03-01') else 'During-COVID'
)

sns.set(style="whitegrid", rc={"figure.figsize": (14, 6)})

plt.figure()
sns.lineplot(data=combined_df, x='Date',
             y='Estimated Unemployment Rate (%)', label='Unemployment Rate')
plt.axvline(pd.to_datetime("2020-03-01"), color='red',
            linestyle='--', label='COVID-19 Start (Mar 2020)')
plt.axvspan(pd.to_datetime("2020-03-01"), pd.to_datetime("2020-06-30"),
            color='red', alpha=0.1, label='Lockdown Period')
plt.title("Impact of COVID-19 on Unemployment Rate in India")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.legend()
plt.grid(True)
plt.show()

monthly_avg = combined_df.groupby(
    'Month')['Estimated Unemployment Rate (%)'].mean()
ordered_months = list(month_name)[1:]
monthly_avg = monthly_avg.reindex(ordered_months)

plt.figure()
sns.lineplot(x=monthly_avg.index, y=monthly_avg.values,
             marker='o', color='green')
plt.title("Average Monthly Unemployment Rate in India (Seasonal Trends)")
plt.xticks(rotation=45)
plt.ylabel("Unemployment Rate (%)")
plt.xlabel("Month")
plt.grid(True)
plt.show()

covid_comparison = combined_df.groupby(['Covid_Phase', 'Region'])[
    'Estimated Unemployment Rate (%)'].mean().reset_index()

plt.figure(figsize=(14, 8))
sns.barplot(data=covid_comparison,
            x='Estimated Unemployment Rate (%)', y='Region', hue='Covid_Phase')
plt.title("Unemployment Rate: Pre-COVID vs During COVID by Region")
plt.xlabel("Average Unemployment Rate (%)")
plt.ylabel("Region")
plt.legend(title='Phase')
plt.tight_layout()
plt.show()
