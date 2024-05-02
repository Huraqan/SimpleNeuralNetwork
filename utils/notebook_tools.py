import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display_html



def side_by_side(df_dict:dict):
    """Generic function that can display dataframes side by side, usage:
    
    side_by_side({
        "df title": df,
        "df title2": df,
        ...
    })
    """
    
    html = ""
    for name, df in df_dict.items():
        html += df.style.set_table_attributes("style='display:inline; vertical-align:top;'").set_caption(name)._repr_html_()
    display_html(html, raw=True)


def get_win_fails(df):
    ndf = pd.DataFrame(df["Win or Fail"].value_counts())
    percent = ndf["count"].iloc[0] / (ndf["count"].iloc[0] + ndf["count"].iloc[1]) * 100
    new_name = f"{np.round(percent, 2)}%"
    ndf = ndf.rename(columns={"count": new_name})
    return ndf


def get_win_fails_by_digit(df):
    return pd.DataFrame(df.groupby(['Digit']).value_counts(['Win or Fail']).to_frame())


def plot_win_fails(df):
    ndf = df.copy()
    ndf["Digit"] = ndf["Digit"].astype("str")
    ndf["Digit"] = pd.Categorical(ndf["Digit"], ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "noise"])

    sns.histplot(data=ndf, x="Digit", hue="Win or Fail", multiple="stack")
    plt.show()
    
    
    