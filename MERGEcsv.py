import pandas as pd

# =========================
# FUNCTION TO LOAD GOOGLE
# =========================
def load_google(file, trend_name):
    df = pd.read_csv(file)

    # adjust if columns different
    df.columns = ["date", "google_trend"]
    df["date"] = pd.to_datetime(df["date"])
    df["trend"] = trend_name.lower()

    return df


# =========================
# FUNCTION TO LOAD WIKI
# =========================
def load_wiki(file, trend_name):
    df = pd.read_csv(file)
    df.columns = ["date", "wiki_views"]
    df["date"] = pd.to_datetime(df["date"])
    df["trend"] = trend_name.lower()
    return df


# =========================
# LOAD GOOGLE DATA
# =========================
g_barbie = load_google(""C:/Users/alici/PycharmProjects/MDM3fashion-/data/Barbie_coretimeseries.csv"", "barbie")
g_cottage = load_google("data/cottage_coretimeseries.csv", "cottagecore")
g_grunge = load_google("data/Grunge.csv", "grunge")
g_y2k = load_google("data/Y2Krevival.csv", "y2k")

google = pd.concat([g_barbie, g_cottage, g_grunge, g_y2k])


# =========================
# LOAD WIKI DATA
# =========================
w_barbie = load_wiki("data/BARBIECOREWIK.csv", "barbie")
w_cottage = load_wiki("data/COTTAGECOREWIK.csv", "cottagecore")
w_grunge = load_wiki("data/GRUNGECOREWIK.csv", "grunge")
w_y2k = load_wiki("data/Y2KCOREWIK.csv", "y2k")

wiki = pd.concat([w_barbie, w_cottage, w_grunge, w_y2k])


# =========================
# MERGE
# =========================
df = pd.merge(google, wiki, on=["date", "trend"], how="left")

# fill missing wiki
df["wiki_views"] = df["wiki_views"].fillna(0)


# =========================
# SORT
# =========================
df = df.sort_values(["trend", "date"])


# =========================
# SAVE
# =========================
df.to_csv("FINAL_DATASET.csv", index=False)

print("DONE — FINAL_DATASET.csv created")