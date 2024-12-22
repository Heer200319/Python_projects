import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import requests

Crypto= pd.read_csv("Crypto.csv") #if that is how we want the file name to be
print(Crypto)
#!pip install fuzzywuzzy #data cleaning
#!pip install rapidfuzz
from rapidfuzz import process #make sure to work with the original file, not the already processed one prior
from concurrent.futures import ThreadPoolExecutor

Crypto['name'] = Crypto['name'].str.lower().str.strip()
unique_names = Crypto['name'].unique()

def cluster_names_parallel(names, threshold=80):
    clusters = {}
    def process_name(name):
        match = process.extractOne(name, clusters.keys(), score_cutoff=threshold)
        if match:
            clusters[match[0]].append(name)
        else:
            clusters[name] = [name]

    with ThreadPoolExecutor() as executor:
        executor.map(process_name, names)

    return clusters

clusters = cluster_names_parallel(unique_names, threshold=80)

name_mapping = {name: cluster for cluster, names in clusters.items() for name in names}

Crypto['cleaned_name'] = Crypto['name'].map(name_mapping)

Crypto.to_csv('Cleaned_Crypto.csv', index=False)
print(Crypto[['name', 'cleaned_name']].head())

#!pip install requests #additional cleaning

#comparing names if they actually exist and creating a file
# import requests

# Fetch the list of cryptocurrencies from CoinGecko
url = 'https://api.coingecko.com/api/v3/coins/list'
response = requests.get(url)

# Check for successful response
if response.status_code == 200:
    data = response.json()
    # Extract cryptocurrency names and convert them to lowercase for comparison
    crypto_names = [coin['name'].lower() for coin in data]

    valid_names_df = pd.DataFrame(crypto_names, columns=['name'])

    Crypto_Data = pd.read_csv('Crypto.csv')

    Crypto['name_lower'] = Crypto['name'].str.lower()

    # Filter out rows where the name does not exist in the valid names list
    cleaned_crypto = Crypto[Crypto['name_lower'].isin(valid_names_df['name'])]

    # Drop the temporary lowercase column
    cleaned_crypto = cleaned_crypto.drop(columns=['name_lower'])

    # Save the cleaned dataset
    cleaned_crypto.to_csv('Cleaned_Crypto.csv', index=False) # this is the cleaned file we have created

    print(cleaned_crypto.head())

else:
    print(f"Error fetching data: {response.status_code}")


Crypto2= pd.read_csv("Cleaned_Crypto.csv") # this will display only 6581 rows during first run the results might vary
print(Crypto2)


#changing date column to the same format #RUN THIS TO CHANDEG DATE
# Define a function to handle various date formats
def try_parse_dates(date_str):
    try:
        # parsing in dd-mm-yyyy format first
        return pd.to_datetime(date_str, format='%d-%m-%Y', errors='raise')
    except:
        try:
            # parsing in mm/dd/yyyy format if the first fails
            return pd.to_datetime(date_str, format='%m/%d/%Y', errors='raise')
        except:
            # general parsing if both formats fail
            return pd.to_datetime(date_str, errors='raise')

# Apply the function to the 'date_taken' column
Crypto2['date_taken_converted'] = Crypto2['date_taken'].apply(try_parse_dates)

# Display the result to check the conversion
print(Crypto2[['name', 'date_taken', 'date_taken_converted']].head(20))  # Preview the data

Crypto2['date_taken_converted'] = Crypto2['date_taken_converted'].dt.strftime('%m/%d/%Y')# mm/dd/YYYY #RUN THIS TO CHANDEG DATE SECOND

Crypto2.info() # provides information about columns, any null values, and data types
Crypto2.isnull().sum()
# checks for null values in each column

#converting 'marketcap' column to a string type to remove commas.
Crypto2['marketcap'] = Crypto2['marketcap'].astype(str).str.replace(',','')
#converting string type 'marketcap' column to a numeric type (float), error = 'coerce' will set non-numeric to NaN.
Crypto2['marketcap'] = pd.to_numeric (Crypto2 ['marketcap'].str.replace(',',''), errors= 'coerce')
#shows that datatype of 'marketcap' column is now a float.
print (Crypto2['marketcap'].dtypes)

Crypto2.head(20) #preview of dataset. You can see commas has been removed in 'marketcap' column

#check for duplicates
duplicates = Crypto2.duplicated()
print(duplicates)
print (Crypto2.duplicated().sum()) # no duplicates in dataset

top_10_crypto = Crypto2.nlargest(10, 'price') #general info to get the idea
print(top_10_crypto)

# Filtered by the abbreviation (abbr) of Bitcoin, Solana, and Avalanche
btc_sol_avax = Crypto2.loc [Crypto2['abbr'].isin (['BTC', 'SOL', 'AVAX'])]

# Created a bar chart to visualize comparison
plt.figure (figsize = (10, 6))
sns.barplot (x = 'abbr', y = 'marketcap', data = btc_sol_avax)
plt.title ('Comparison of MarketCap for Three Cryptocurrencies: Bitcoin, Solana, Avalanche')
plt.xlabel ('Cryptocurrency')
plt.ylabel ('Market Cap (million USD)')
plt.show ()
plt.savefig("Plot1.png")

print (btc_sol_avax[['name', 'abbr','marketcap','date_taken_converted']])

# Reload the dataset
crypto_df = pd.read_csv('Cleaned_Crypto.csv')

# Convert and reformat the `date_taken` column
crypto_df['date_taken'] = pd.to_datetime(crypto_df['date_taken'], format='%d-%m-%Y', errors='coerce')
crypto_df['date_taken'] = crypto_df['date_taken'].dt.strftime('%m/%d/%Y')

# Save the updated dataset back to the CSV file
crypto_df.to_csv('Cleaned_Crypto.csv', index=False)

# Function to get the price on a specific date
def get_price_on_date(df, crypto_abbr, date_str):
    # Ensure the input date matches the dataset's format
    date_str = pd.to_datetime(date_str, format='%m/%d/%Y').strftime('%m/%d/%Y')
    
    # Filter rows where abbreviation and date match
    price_series = df[(df['abbr'] == crypto_abbr) & (df['date_taken'] == date_str)]['price']
    
    # Return a single value if it exists, otherwise report no data
    return price_series.item() if not price_series.empty else "No data available"

# Prompt the user for input
crypto_abbr = input("Enter the abbreviation of the cryptocurrency (e.g., ETH): ")
date_input = input("Enter the date (mm/dd/yyyy, e.g., 12/13/2007): ")

# Retrieve the price
price = get_price_on_date(crypto_df, crypto_abbr, date_input)

# Print the result
print(f"Price of {crypto_abbr} on {date_input}: {price}")

# Convert Crypto2 into a DataFrame if it's not already one
df = pd.DataFrame(Crypto2)

# Data cleaning and type conversion
df['marketcap'] = pd.to_numeric(df['marketcap'], errors='coerce')
df['volume24hrs'] = pd.to_numeric(df['volume24hrs'], errors='coerce')
df['circulatingsupply'] = pd.to_numeric(df['circulatingsupply'], errors='coerce')
df['maxsupply'] = pd.to_numeric(df['maxsupply'], errors='coerce')

# Drop rows with missing values in key columns
df.dropna(subset=['marketcap', 'volume24hrs', 'circulatingsupply', 'maxsupply'], inplace=True)

# Function to remove outliers using the IQR method
def remove_outliers(dataframe, column):
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return dataframe[(dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound)]

# Remove outliers from relevant columns
df = remove_outliers(df, 'marketcap')
df = remove_outliers(df, 'volume24hrs')
df = remove_outliers(df, 'circulatingsupply')

# Calculate circulating ratio
df['circulating_ratio'] = df['circulatingsupply'] / df['maxsupply']

# Descriptive statistics
print("Descriptive Statistics:")
print(df[['marketcap', 'volume24hrs', 'circulatingsupply', 'maxsupply', 'circulating_ratio']].describe())

# Correlation analysis
correlation_matrix = df[['marketcap', 'volume24hrs', 'circulatingsupply', 'maxsupply']].corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Heatmap for the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()
plt.savefig("Plot2.png")

# Top 10 Cryptocurrencies by Market Cap
top_10_marketcap = df.nlargest(10, 'marketcap')[['name', 'marketcap']]
print("\nTop 10 Cryptocurrencies by Market Cap:")
print(top_10_marketcap)

# Bar plot for Top 10 by Market Cap
plt.figure(figsize=(10, 6))
sns.barplot(x='marketcap', y='name', data=top_10_marketcap, palette='viridis')
plt.title("Top 10 Cryptocurrencies by Market Cap")
plt.xlabel("Market Cap (USD)")
plt.ylabel("Cryptocurrency")
plt.show()
plt.savefig("Plot3.png")

# Top 10 Cryptocurrencies by 24hr Trading Volume
top_10_volume = df.nlargest(10, 'volume24hrs')[['name', 'volume24hrs']]
print("\nTop 10 Cryptocurrencies by 24hr Trading Volume:")
print(top_10_volume)

# Bar plot for Top 10 by 24hr Volume
plt.figure(figsize=(10, 6))
sns.barplot(x='volume24hrs', y='name', data=top_10_volume, palette='viridis')
plt.title("Top 10 Cryptocurrencies by 24hr Trading Volume")
plt.xlabel("Volume (24 hrs) in USD")
plt.ylabel("Cryptocurrency")
plt.show()
plt.savefig("Plot4.png")

# Pie Chart - Market Cap Distribution (Top 5)
top_5_marketcap = df.nlargest(5, 'marketcap')[['name', 'marketcap']]
top_5_marketcap.set_index('name', inplace=True)

plt.figure(figsize=(8, 8))
top_5_marketcap['marketcap'].plot(kind='pie', autopct='%1.1f%%', startangle=90, cmap='coolwarm')
plt.title("Market Cap Distribution of Top 5 Cryptocurrencies")
plt.ylabel('')
plt.show()
plt.savefig("Plot5.png")

# Box plot for Circulating vs Max Supply
plt.figure(figsize=(10, 6))
sns.boxplot(data=[df['circulatingsupply'], df['maxsupply']], orient='h', palette='Set2')
plt.title("Circulating Supply vs Max Supply Distribution")
plt.xlabel("Supply Amount")
plt.yticks([0, 1], ['Circulating Supply', 'Max Supply'])
plt.show()
plt.savefig("Plot6.png")

# Trend Analysis - Volume vs Date
df['date_taken_converted'] = pd.to_datetime(df['date_taken_converted'], errors='coerce')
df.dropna(subset=['date_taken_converted'], inplace=True)

daily_volume = df.groupby('date_taken_converted')['volume24hrs'].sum()

# Plotting the trend of total volume over time
plt.figure(figsize=(12, 6))
daily_volume.plot(kind='line', color='green', linewidth=2)
plt.title("Total 24-hour Trading Volume Trend")
plt.xlabel("Date")
plt.ylabel("Total Volume (USD)")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plt.savefig("Plot7.png")

