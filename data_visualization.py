import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def hello_seaborn():
    fifa_filepath = 'fifa.csv'
    fifa_data = pd.read_csv(fifa_filepath, index_col='Date', parse_dates=True)
    plt.figure(figsize=(16,6))
    sns.lineplot(data=fifa_data)
    plt.show()

def spotify():
    spotify_filepath = 'spotify.csv'
    spotify_data = pd.read_csv(spotify_filepath, index_col='Date', parse_dates=True)
    print(spotify_data.head())
    print(spotify_data.tail())
    print(spotify_data.columns)

    # 14 inches wide by 6 inches high.
    plt.figure(figsize=(14,6))
    plt.title('Daily Global Streams of Popular Songs in 2017-2018')
    sns.lineplot(data=spotify_data["Shape of You"], label="Shape of You")
    sns.lineplot(data=spotify_data["Despacito"], label="Despacito")
    plt.xlabel("Date")
    plt.show()

def line_charts():
    museum_filepath = 'museum_visitors.csv'
    museum_data = pd.read_csv(museum_filepath, index_col='Date', parse_dates=True)
    print(museum_data.tail())
    print(museum_data.columns)
    sns.lineplot(data=museum_data['Avila Adobe'])
    plt.show()

def flight_delays():
    flight_filepath = 'flight_delays.csv'
    flight_data = pd.read_csv(flight_filepath, index_col='Month')
    print(flight_data)

    # Bar plot
    plt.figure(figsize=(10,6))
    plt.title('Average Arrival Delay for Spirit Airplines Flights, by Month')
    sns.barplot(x=flight_data.index, y=flight_data['NK'])
    plt.ylabel('Arrival delay (in minutes)')
    plt.show()

    # Heat map
    plt.figure(figsize=(14,7))
    plt.title('Average Arrival Delay for Each Airpline, by Month')
    sns.heatmap(data=flight_data, annot=True)
    plt.xlabel('Airline')
    plt.show()

def bar_charts_and_heatmaps():
    ign_filepath = 'ign_scores.csv'
    ign_data = pd.read_csv(ign_filepath, index_col='Platform')
    # What is the highest average score received by PC games, for any platform?
    print(ign_data.loc['PC'].max())
    # On the Playstation Vita platform, which genre has the
    # lowest average score?
    print(ign_data.idxmin(axis=1))

    # Bar plot
    plt.figure(figsize=(10,6))
    plt.title('average score for racing games, for each platform')
    sns.barplot(x=ign_data.index, y=ign_data['Racing'])
    #plt.ylabel('Arrival delay (in minutes)')
    plt.show()

    # Heat map
    plt.figure(figsize=(14,7))
    sns.heatmap(data=ign_data, annot=True)
    plt.show()

def insurance():
    insurance_filepath = 'insurance.csv'
    insurance_data = pd.read_csv(insurance_filepath)
    print(insurance_data.head())
    #sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'])
    #sns.regplot(x=insurance_data['bmi'], y=insurance_data['charges'])
    #sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'], hue=insurance_data['smoker'])
    #sns.lmplot(x='bmi', y='charges', hue='smoker', data=insurance_data)
    sns.swarmplot(x='smoker', y='charges', data=insurance_data)
    plt.show()

def candy():
    candy_filepath = 'candy.csv'
    candy_data = pd.read_csv(candy_filepath, index_col='id')
    plt.figure()
    #sns.scatterplot(x='sugarpercent', y='winpercent', data=candy_data)
    #sns.regplot(x='sugarpercent', y='winpercent', data=candy_data)
    #sns.scatterplot(x='pricepercent', y='winpercent', hue='chocolate', data=candy_data)
    #sns.lmplot(x='pricepercent', y='winpercent', hue='chocolate', data=candy_data)
    sns.swarmplot(x='chocolate', y='winpercent', data=candy_data)
    plt.show()

def iris():
    iris_filepath = 'iris.csv'
    iris_data = pd.read_csv(iris_filepath, index_col='Id')
    print(iris_data.head())

    #sns.distplot(a=iris_data['Petal Length (cm)'], kde=False)
    #sns.kdeplot(data=iris_data['Petal Length (cm)'], shade=True)
    #sns.jointplot(x=iris_data['Petal Length (cm)'], y=iris_data['Sepal Width (cm)'], kind='kde')

    iris_set_filepath = 'iris_setosa.csv'
    iris_ver_filepath = 'iris_versicolor.csv'
    iris_vir_filepath = 'iris_virginica.csv'

    iris_set_data = pd.read_csv(iris_set_filepath, index_col='Id')
    iris_ver_data = pd.read_csv(iris_ver_filepath, index_col='Id')
    iris_vir_data = pd.read_csv(iris_vir_filepath, index_col='Id')

    print(iris_ver_data.head())

    # Histograms for each species
    #sns.distplot(a=iris_set_data['Petal Length (cm)'], label='Iris-setosa', kde=False)
    #sns.distplot(a=iris_ver_data['Petal Length (cm)'], label='Iris-versicolor', kde=False)
    #sns.distplot(a=iris_vir_data['Petal Length (cm)'], label='Iris-virginica', kde=False)
    #plt.title('Histogram of Petal Lengths, by Species')
    #plt.legend()

    # KDE plots for each species
    sns.kdeplot(data=iris_set_data['Petal Length (cm)'], label='Iris-setosa', shade=True)
    sns.kdeplot(data=iris_ver_data['Petal Length (cm)'], label='Iris-versicolor', shade=True)
    sns.kdeplot(data=iris_vir_data['Petal Length (cm)'], label='Iris-virginica', shade=True)
    plt.title('Distribution of Petal Lengths by Species')

    plt.show()

def distributions():
    cancer_b_filepath = 'cancer_b.csv'
    cancer_m_filepath = 'cancer_m.csv'

    cancer_b_data = pd.read_csv(cancer_b_filepath, index_col='Id')
    cancer_m_data = pd.read_csv(cancer_m_filepath, index_col='Id')

    # Histograms
    #sns.distplot(a=cancer_b_data['Area (mean)'], label='Benign', kde=False)
    #sns.distplot(a=cancer_m_data['Area (mean)'], label='Malignant', kde=False)
    #plt.legend()

    # KDE plots
    sns.kdeplot(data=cancer_b_data['Radius (worst)'], label='Benign')
    sns.kdeplot(data=cancer_m_data['Radius (worst)'], label='Malignant')

    plt.show()

def customize():
    spotify_filepath = 'spotify.csv'
    spotify_data = pd.read_csv(spotify_filepath, index_col='Date', parse_dates=True)

    sns.set_style('whitegrid')
    plt.figure(figsize=(12,6))
    sns.lineplot(data=spotify_data)
    plt.show()

def final_project():
    parks_filepath = 'parks.csv'
    species_filepath = 'species.csv'

    parks_data = pd.read_csv(parks_filepath, index_col='Park Name')
    species_data = pd.read_csv(species_filepath, index_col='Park Name')

    #sns.scatterplot(x="Category", y="Order", data=species_data)
    species_data = species_data[species_data.Occurrence == 'Present']
    species_data_cat = species_data['Category'].value_counts()
    #.groupby(['Category']).agg(['count'])
    print(species_data_cat)
    #print(species_data_cat.columns)
    print(species_data_cat.index)
    #print(species_data_category['Species ID'])
    plt.figure(figsize=(6,6))
    cat_plot = sns.barplot(x=species_data_cat.index, y=species_data_cat.values)
    cat_plot.set_xticklabels(cat_plot.get_xticklabels(), rotation=90)
    #sns.barplot(data=species_data_cat)
    plt.title('NPS unique species count by category')
    plt.tight_layout()
    plt.show()


#hello_seaborn()
#spotify()
#line_charts()
#flight_delays()
#bar_charts_and_heatmaps()
#insurance()
#candy()
#iris()
#distributions()
#customize()
final_project()

'''
Trends - A trend is defined as a pattern of change.
sns.lineplot - Line charts are best to show trends over a period of time, and multiple lines can be used to show trends in more than one group.

Relationship - There are many different chart types that you can use to understand relationships between variables in your data.
sns.barplot - Bar charts are useful for comparing quantities corresponding to different groups.
sns.heatmap - Heatmaps can be used to find color-coded patterns in tables of numbers.
sns.scatterplot - Scatter plots show the relationship between two continuous variables; if color-coded, we can also show the relationship with a third categorical variable.
sns.regplot - Including a regression line in the scatter plot makes it easier to see any linear relationship between two variables.
sns.lmplot - This command is useful for drawing multiple regression lines, if the scatter plot contains multiple, color-coded groups.
sns.swarmplot - Categorical scatter plots show the relationship between a continuous variable and a categorical variable.

Distribution - We visualize distributions to show the possible values that we can expect to see in a variable, along with how likely they are.
sns.distplot - Histograms show the distribution of a single numerical variable.
sns.kdeplot - KDE plots (or 2D KDE plots) show an estimated, smooth distribution of a single numerical variable (or two numerical variables).
sns.jointplot - This command is useful for simultaneously displaying a 2D KDE plot with the corresponding KDE plots for each individual variable.
'''
