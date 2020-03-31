import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from IPython.core.pylabtools import figsize
from matplotlib.lines import Line2D

figsize(10, 6)

plt.style.use("fivethirtyeight")
plt.rcParams["font.size"] = 12
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"


def model_mae_comparison_graph(model):
    model.sort_values('MAE', ascending=False).plot(x='Model', y='MAE', kind='barh', color='red', legend=False)
    plt.ylabel("")
    plt.yticks(size=8)
    plt.xlabel("MAE")
    plt.xticks(size=8)
    plt.title('Model comparison based on MAE')
    plt.show()


def model_rmse_comparison_graph(model):
    model.sort_values('RMSE', ascending=False).plot(x='Model', y='RMSE', kind='barh', color='red', legend=False)
    plt.ylabel("")
    plt.yticks(size=8)
    plt.xlabel("RMSE")
    plt.xticks(size=8)
    plt.title('Model comparison based on RMSE')
    plt.show()


# Histogram plot of Year of release
def year_of_rel(data):
    num_years = data["Year"].max() - data["Year"].min() + 1
    plt.hist(data["Year"], bins=num_years, color="lightskyblue", edgecolor="black")
    plt.title("Distribution of year of release")
    plt.xlabel("Year")
    plt.ylabel("Number of games")
    plt.show()


class has_score:

    def distribution(self, data):
        plt.hist(data[data["Has_Score"] == True]["Global"], color="limegreen", alpha=0.5,
                 edgecolor="black")
        plt.hist(data[data["Has_Score"] == False]["Global"], color="indianred", alpha=0.5,
                 edgecolor="black")
        plt.title("Distribution of global sales")
        plt.xlabel("Global sales, $M")
        plt.ylabel("Number of games")
        plt.legend(handles=[Line2D([0], [0], color="limegreen", lw=20, label="True", alpha=0.5),
                            Line2D([0], [0], color="indianred", lw=20, label="False", alpha=0.5)],
                   title="Has_Score", loc=7)
        plt.show()

    def compare_plot(self, data):
        data["Country"] = data[["NA", "EU", "JP", "Other"]].idxmax(1, skipna=True)
        palette = {True: "limegreen", False: "indianred"}
        sns.factorplot(y="Country", hue="Has_Score", data=data, size=8, kind="count", palette=palette)
        plt.show()

    def compare_plot_median(self, data):
        data["Country"] = data[["NA", "EU", "JP", "Other"]].idxmax(1, skipna=True)
        palette = {True: "limegreen", False: "indianred"}
        sns.factorplot(y="Country", x="Global", hue="Has_Score", data=data, size=8, kind="bar", palette=palette,
                       estimator=lambda x: np.median(x))
        plt.show()


def error_graph(results):
    plt.plot(results["param_n_estimators"], -1 * results["mean_test_score"], label="Testing Error")
    plt.plot(results["param_n_estimators"], -1 * results["mean_train_score"], label="Training Error")
    plt.xlabel("Number of Trees")
    plt.ylabel("Mean Abosolute Error")
    plt.legend()
    plt.title("Performance vs Number of Trees")
    plt.show()


def test_train_pred(target_test, target_train, prediction):
    sns.kdeplot(prediction, label='Predictions')
    sns.kdeplot(target_test, label='Test')
    sns.kdeplot(target_train, label='Test')

    plt.xlabel("Global Sales")
    plt.ylabel('Density')
    plt.title('Train, Test val and predictions')
    plt.show()
