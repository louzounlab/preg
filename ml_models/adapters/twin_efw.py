"""Twin fetal weight estimation adapter.

Mirrors the behavior of the original ``ml_models/twin_efw/app/app.py`` so that
the website's twin-efw page renders the same result view: percentile/z-score
labels, a trend-line image, per-week gaussian plots, CSV downloads, and the
"Adjust Range" form.
"""
import gc
import os
import pickle
import shutil
import time
from functools import lru_cache
from os.path import join

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

import numpy as np
import pandas as pd
from scipy.stats import norm

rcParams["font.family"] = "Times New Roman"
rcParams["font.size"] = 25


@lru_cache(maxsize=1)
def _load_data(data_path: str) -> pd.DataFrame:
    return pd.read_csv(data_path)


def _get_values(df: pd.DataFrame, mcda: int, week: float) -> pd.DataFrame:
    sub = df[df["MCDA"] == mcda]
    weeks = list(sub["Week"])
    if week in weeks:
        return sub[sub["Week"] == week]
    week_below = max(w for w in weeks if w < week)
    week_above = min(w for w in weeks if w > week)
    a = (week_above - week) / (week_above - week_below)
    b = (week - week_below) / (week_above - week_below)
    arr = a * sub[sub["Week"] == week_below].to_numpy() + b * sub[sub["Week"] == week_above].to_numpy()
    return pd.DataFrame(arr, columns=df.columns)


def _percentage_below_x(x, mean, std):
    z = (x - mean) / std
    pct = norm.cdf(z) * 100
    return pct, z


def _gaussian(x, mean, std):
    return 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-((x - mean) ** 2) / (2 * std ** 2))


def _plot_gaussian(df_full, mcda, week, weight1, weight2, save_path, title=""):
    df = _get_values(df_full, mcda, week)
    mean = float(df["50"].iloc[0])
    std = float(df["Std"].iloc[0])

    x = np.linspace(mean - 3 * std, mean + 3 * std, 100)
    y = [_gaussian(xi, mean, std) for xi in x]

    fig, ax = plt.subplots(figsize=(15, 12), facecolor="none")
    ax.plot(x, y, color="darkturquoise", linewidth=4)
    ax.set_title(title)
    ax.set_xlabel("Weight")
    ax.set_ylabel("Probability Density")

    x_values = [float(val) for val in list(df.iloc[0][["5", "10", "50", "90", "95"]])]
    y_values = [float(_gaussian(xi, mean, std)) for xi in x_values]

    width = 3 * std / 100
    ax.bar(x_values[2], y_values[2], color="black", width=width)
    ax.bar([x_values[1], x_values[3]], [y_values[1], y_values[3]], color="grey", width=width)
    ax.bar([x_values[0], x_values[4]], [y_values[0], y_values[4]], color="lightgrey", width=width)

    if weight1:
        ax.bar(weight1, _gaussian(weight1, mean, std), color="dodgerblue", width=width)
        ax.scatter(weight1, _gaussian(weight1, mean, std), color="dodgerblue", label="EFW1", s=400)
    if weight2:
        ax.bar(weight2, _gaussian(weight2, mean, std), color="hotpink", width=width)
        ax.scatter(weight2, _gaussian(weight2, mean, std), color="hotpink", label="EFW2", s=400)
    ax.legend()

    plt.savefig(save_path)
    plt.close(fig)

    pct1 = z1 = pct2 = z2 = None
    if weight1:
        pct1, z1 = _percentage_below_x(weight1, mean, std)
    if weight2:
        pct2, z2 = _percentage_below_x(weight2, mean, std)
    return pct1, pct2, z1, z2


def _plot_trend(df_full, mcda, week_df, week1, week2, weight1_df, weight2_df,
                save_path, title="Trend Line", extend_by=1):
    weeks = list(week_df["week"])
    weights1 = list(weight1_df["weight"]) if hasattr(weight1_df, "weight") else list(weight1_df)
    weights2 = list(weight2_df["weight"]) if hasattr(weight2_df, "weight") else list(weight2_df)

    extended = [max(12, min(weeks) - extend_by)] + weeks + [min(36.5, max(weeks) + extend_by)]
    dfs = [_get_values(df_full, mcda, w) for w in extended]
    df = pd.concat(dfs)

    colors = ["lightgrey", "grey", "black", "grey", "lightgrey"]
    pers = ["5", "10", "50", "90", "95"]

    fig, ax = plt.subplots(figsize=(15, 12), facecolor="none")
    for i, c in enumerate(colors):
        ax.plot(extended, df[pers[i]], color=c, label=f"{pers[i]}%", linewidth=2)
    ax.fill_between(extended, df["5"], df["95"], color="lightgrey", alpha=0.5)
    ax.fill_between(extended, df["10"], df["90"], color="grey", alpha=0.5)

    ax.plot(week1, weights1, color="dodgerblue", label="EFW1", linewidth=4)
    ax.plot(week2, weights2, color="hotpink", label="EFW2", linewidth=4)
    ax.scatter(week1, weights1, color="dodgerblue", s=6 * rcParams["lines.markersize"] ** 2)
    ax.scatter(week2, weights2, color="hotpink", s=6 * rcParams["lines.markersize"] ** 2)

    ax.set_title(title)
    ax.set_xlabel("Week")
    ax.set_ylabel("Weight")
    ax.legend()

    plt.savefig(save_path)
    plt.close(fig)


def clean_old_files(static_root: str) -> None:
    """Delete result folders older than one hour."""
    gc.collect()
    if not os.path.isdir(static_root):
        return
    for entry in os.listdir(static_root):
        full = join(static_root, entry)
        if not os.path.isdir(full):
            continue
        try:
            ts = float(entry)
        except ValueError:
            continue
        if time.time() - ts > 3600:
            shutil.rmtree(full, ignore_errors=True)


def predict(payload: dict, submodule_root: str, static_root: str | None = None) -> dict:
    """Run the full twin-efw pipeline.

    Returns a dict with all values needed to render the result template:
    percentage_dict, zscore_dict, discordance_index, highlight_index,
    trend_line url, gaussian image urls, csv urls, pickle path, last_row.
    """
    data_path = join(submodule_root, "app", "static", "data.csv")
    df_full = _load_data(data_path)

    if static_root is None:
        static_root = join(os.getcwd(), "static")
    os.makedirs(static_root, exist_ok=True)
    clean_old_files(static_root)

    last_row = int(payload.get("last_row", 4) or 4)

    cda_type = payload.get("cda_type", "None")
    if cda_type == "None" or not cda_type:
        raise ValueError("Please select a MCDA/DCDA")
    mcda = 1 if cda_type == "MCDA" else 0

    request_time = str(time.time())
    folder_path = join(static_root, request_time)
    os.mkdir(folder_path)
    # url path served by Flask (always forward slashes)
    folder_url = f"/static/{request_time}"

    weeks_list = []
    for i in range(1, 11):
        wi = payload.get(f"week{i}", "")
        if not wi:
            continue
        wi = int(wi)
        di = payload.get(f"Day{i}", "")
        if di:
            di = int(di)
            if wi == 36:
                di = min(di, 3)
            wi += di / 7
        weeks_list.append(wi)

    week_df = pd.DataFrame({"week": weeks_list})
    if week_df.shape[0] == 0:
        raise ValueError("Please input at least one week.")

    weights_list = [[], []]
    for j in range(2):
        for i in range(1, 11):
            v = payload.get(f"EFW{j + 1}_{i}", "")
            if v:
                weights_list[j].append(float(v))
            else:
                weights_list[j].append(np.nan)
    # trim to as many entries as we have weeks (the form has up to 10 rows)
    weights_list[0] = weights_list[0][: len(weeks_list)]
    weights_list[1] = weights_list[1][: len(weeks_list)]

    weight_df_1 = pd.DataFrame({"weight": weights_list[0]})
    weight_df_2 = pd.DataFrame({"weight": weights_list[1]})

    discordance_index, highlight_index = {}, {}
    na1, na2 = weight_df_1.isna(), weight_df_2.isna()
    for i in range(len(weeks_list)):
        w1 = None if na1.iloc[i]["weight"] else weight_df_1.iloc[i]["weight"]
        w2 = None if na2.iloc[i]["weight"] else weight_df_2.iloc[i]["weight"]
        if w1 and w2:
            disc = abs(w1 - w2) / max(w1, w2)
            discordance_index[i + 1] = f"{100 * disc:.2f}%"
            highlight_index[i + 1] = 1 if disc > 0.2 else 0

    if weight_df_1.dropna().shape[0] == 0:
        raise ValueError("Please input at least one weight for twin 1.")
    if weight_df_2.dropna().shape[0] == 0:
        raise ValueError("Please input at least one weight for twin 2.")

    week_twin_1 = week_df[~weight_df_1["weight"].isna()].reset_index(drop=True)
    week_twin_2 = week_df[~weight_df_2["weight"].isna()].reset_index(drop=True)
    weight_df_1 = weight_df_1.dropna().reset_index(drop=True)
    weight_df_2 = weight_df_2.dropna().reset_index(drop=True)

    twin1_pct, twin2_pct, twin1_z, twin2_z = [], [], [], []
    for i in range(week_df.shape[0]):
        week = week_df.iloc[i].week
        w1 = w2 = None
        if week in list(week_twin_2["week"]):
            w2 = float(weight_df_2[week_twin_2["week"] == week].iloc[0].weight)
        if week in list(week_twin_1["week"]):
            w1 = float(weight_df_1[week_twin_1["week"] == week].iloc[0].weight)
        p1, p2, z1, z2 = _plot_gaussian(
            df_full, mcda, week, w1, w2,
            save_path=join(folder_path, f"gaussian_{week}.png"),
            title=f"Week {week}",
        )
        if p1 is not None:
            twin1_pct.append(p1)
            twin1_z.append(z1)
        if p2 is not None:
            twin2_pct.append(p2)
            twin2_z.append(z2)

    trend_path = join(folder_path, "trend_line.png")
    _plot_trend(
        df_full, mcda, week_df,
        list(week_twin_1["week"]), list(week_twin_2["week"]),
        weight_df_1, weight_df_2,
        save_path=trend_path,
    )

    gaussian_files = sorted(
        f"{folder_url}/{f}" for f in os.listdir(folder_path) if f.startswith("gaussian_")
    )

    week1, week2 = list(week_twin_1["week"]), list(week_twin_2["week"])
    percentage_df = pd.DataFrame({"Week": [], "Twin 1": [], "Twin 2": []})
    zscore_df = pd.DataFrame({"Week": [], "Twin 1": [], "Twin 2": []})
    i = j = 0
    for k, week in enumerate(week_df["week"]):
        per1 = per2 = z1v = z2v = np.nan
        if i < len(week1) and week == week1[i]:
            per1 = twin1_pct[i]
            z1v = twin1_z[i]
            i += 1
        if j < len(week2) and week == week2[j]:
            per2 = twin2_pct[j]
            z2v = twin2_z[j]
            j += 1
        percentage_df.loc[k] = [week, per1, per2]
        zscore_df.loc[k] = [week, z1v, z2v]

    pct_csv = join(folder_path, "percentages.csv")
    z_csv = join(folder_path, "zscores.csv")
    percentage_df.to_csv(pct_csv, index=False)
    zscore_df.to_csv(z_csv, index=False)

    percentage_dict, zscore_dict = {}, {}
    for j in range(1, 3):
        for i in range(1, 11):
            try:
                v = str(percentage_df[f"Twin {j}"].iloc[i - 1])
                v = "" if v == "nan" else f"{float(v):.2f}%"
            except Exception:
                v = ""
            percentage_dict[f"per{j}_{i}"] = v
            try:
                v = str(zscore_df[f"Twin {j}"].iloc[i - 1])
                v = "" if v == "nan" else f"{float(v):.3f}"
            except Exception:
                v = ""
            zscore_dict[f"z{j}_{i}"] = v

    trend_pkl = join(folder_path, "trend_data.pkl")
    trend_data = {
        "mcda": mcda,
        "week": week_df,
        "week1": list(week_twin_1["week"]),
        "week2": list(week_twin_2["week"]),
        "weight1": list(weight_df_1["weight"]),
        "weight2": list(weight_df_2["weight"]),
        "save_path": trend_path,
        "data_path": data_path,
        "data": dict(payload),
        "trend_line": f"{folder_url}/trend_line.png",
        "gaussians": gaussian_files,
        "percentages_df": f"{folder_url}/percentages.csv",
        "zscores_df": f"{folder_url}/zscores.csv",
        "trend_data_path": trend_pkl,
        "percentage_dict": percentage_dict,
        "zscore_dict": zscore_dict,
        "discordance_index": discordance_index,
        "highlight_index": highlight_index,
        "last_row": last_row,
    }
    with open(trend_pkl, "wb") as fh:
        pickle.dump(trend_data, fh)

    return {
        "data": dict(payload),
        "results": True,
        "trend_line": f"{folder_url}/trend_line.png",
        "gaussians": gaussian_files,
        "percentages_df": f"{folder_url}/percentages.csv",
        "zscores_df": f"{folder_url}/zscores.csv",
        "trend_data": trend_pkl,
        "extended_by": 1,
        "percentage_dict": percentage_dict,
        "zscore_dict": zscore_dict,
        "discordance_index": discordance_index,
        "highlight_index": highlight_index,
        "last_row": last_row,
    }


def adjust_trend(trend_data_path: str, extended_by: int = 1, static_root: str | None = None) -> dict:
    """Re-render the trend-line plot with a new range and return template kwargs.

    ``trend_data_path`` arrives from a client-controlled hidden form field, so it
    must be validated before it is opened: an attacker could otherwise point it at
    any file and trigger ``pickle.load`` on it (arbitrary code execution). We only
    accept a ``trend_data.pkl`` that resolves to inside ``static_root`` — i.e. a
    pickle this app wrote itself for a previous request.
    """
    if static_root is None:
        raise ValueError("static_root is required to validate the trend data path")

    base = os.path.realpath(static_root)
    target = os.path.realpath(trend_data_path or "")
    if (
        os.path.basename(target) != "trend_data.pkl"
        or os.path.commonpath([base, target]) != base
        or not os.path.isfile(target)
    ):
        raise ValueError("Invalid trend data reference")

    with open(target, "rb") as fh:
        td = pickle.load(fh)
    df_full = _load_data(td["data_path"])
    _plot_trend(
        df_full, td["mcda"], td["week"], td["week1"], td["week2"],
        pd.DataFrame({"weight": td["weight1"]}),
        pd.DataFrame({"weight": td["weight2"]}),
        save_path=td["save_path"], extend_by=extended_by,
    )
    return {
        "data": td["data"],
        "results": True,
        "trend_line": td["trend_line"],
        "gaussians": td["gaussians"],
        "percentages_df": td["percentages_df"],
        "zscores_df": td["zscores_df"],
        "trend_data": td["trend_data_path"],
        "extended_by": extended_by,
        "percentage_dict": td["percentage_dict"],
        "zscore_dict": td["zscore_dict"],
        "discordance_index": td["discordance_index"],
        "highlight_index": td["highlight_index"],
        "last_row": td.get("last_row", 4),
    }
