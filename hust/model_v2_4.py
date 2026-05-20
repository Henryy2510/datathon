
# %% [0] Config
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error

try:
    import lightgbm as lgb
except Exception as e:
    raise RuntimeError("lightgbm is required for this script") from e

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

VERSION = "v2_4"
TRAIN_PATH = "old/train_outlier_fixed_median.csv"
SUB_PATH = "sample_submission.csv"
OUT_PATH = "submission_v2_4.csv"
METRICS_PATH = "metrics_v2_4.json"

SEED = 42
HORIZON = 56
VAL_DAYS = 28
TOP_WEIGHT_N = 400
USE_XGB = HAS_XGB
USE_NEURAL = False and HAS_TORCH

PROFILE = {"fill": "median", "outlier": True}
np.random.seed(SEED)
print("version", VERSION, "profile", PROFILE, "HAS_XGB", HAS_XGB, "HAS_TORCH", HAS_TORCH)

# %% [1] Load data
float_cols = ["Quantity", "UnitPrice", "SalesAmount", "Unit Cost", "Cost Amount"]
usecols = ["Date", "ItemCode"] + float_cols

df_raw = pd.read_csv(TRAIN_PATH, usecols=usecols, dtype={"ItemCode": "category"}, parse_dates=["Date"])
for col in float_cols:
    df_raw[col] = (
        df_raw[col].astype(str).str.replace(",", ".", regex=False).astype("float32")
    )

print("raw", df_raw.shape, df_raw["Date"].min().date(), "->", df_raw["Date"].max().date())
print(df_raw.dtypes)

# %% [2] Data audit
q = df_raw["Quantity"].astype("float64")
up = df_raw["UnitPrice"].astype("float64")
sa = df_raw["SalesAmount"].astype("float64")
uc = df_raw["Unit Cost"].astype("float64")
ca = df_raw["Cost Amount"].astype("float64")

sales_err = (sa - q * up).abs()
cost_err = (ca - q * uc).abs()
price_q999 = float(up.quantile(0.999))
cost_q999 = float(uc.quantile(0.999))

print("nulls", df_raw.isna().sum().to_dict())
print("sales_err", sales_err.quantile([0.5, 0.9, 0.95, 0.99]).to_dict())
print("cost_err", cost_err.quantile([0.5, 0.9, 0.95, 0.99]).to_dict())
print("price<=0", int((up <= 0).sum()), "cost<=0", int((uc <= 0).sum()))
print("price_q999", price_q999, "cost_q999", cost_q999)

# %% [3] Daily aggregation and panel
df_raw["sales_err"] = sales_err.astype("float32")
df_raw["cost_err"] = cost_err.astype("float32")
df_raw["is_price_bad"] = ((df_raw["UnitPrice"] <= 0) | (df_raw["UnitPrice"] > price_q999)).astype("int8")
df_raw["is_cost_bad"] = ((df_raw["Unit Cost"] <= 0) | (df_raw["Unit Cost"] > cost_q999)).astype("int8")

daily = (
    df_raw.groupby(["Date", "ItemCode"], observed=True, as_index=False)
    .agg(
        Quantity=("Quantity", "sum"),
        SalesAmount=("SalesAmount", "sum"),
        CostAmount=("Cost Amount", "sum"),
        UnitPrice=("UnitPrice", "mean"),
        UnitCost=("Unit Cost", "mean"),
        sales_err=("sales_err", "mean"),
        cost_err=("cost_err", "mean"),
        is_price_bad=("is_price_bad", "max"),
        is_cost_bad=("is_cost_bad", "max"),
    )
    .sort_values(["ItemCode", "Date"])
)

daily["y"] = daily["Quantity"].clip(lower=0).astype("float32")
daily["profit"] = (daily["SalesAmount"] - daily["CostAmount"]).astype("float32")
daily["margin"] = (daily["UnitPrice"] - daily["UnitCost"]).astype("float32")
daily["margin_pct"] = (daily["margin"] / daily["UnitPrice"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")

all_dates = pd.date_range(daily["Date"].min(), daily["Date"].max(), freq="D")
all_skus = daily["ItemCode"].cat.categories if str(daily["ItemCode"].dtype) == "category" else pd.Index(daily["ItemCode"].unique())
panel = pd.MultiIndex.from_product([all_dates, all_skus], names=["Date", "ItemCode"]).to_frame(index=False)
panel["ItemCode"] = panel["ItemCode"].astype("category")
panel = panel.merge(daily, on=["Date", "ItemCode"], how="left").sort_values(["ItemCode", "Date"]).reset_index(drop=True)

for c in ["Quantity", "SalesAmount", "CostAmount", "y", "profit", "sales_err", "cost_err", "is_price_bad", "is_cost_bad"]:
    panel[c] = panel[c].fillna(0).astype("float32")

# Variant-specific value filling. Future inference uses the same profile.
def fill_value_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby("ItemCode", observed=True, sort=False)
    if PROFILE["fill"] == "ffill":
        for c in ["UnitPrice", "UnitCost"]:
            out[c] = out[c].replace(0, np.nan)
            out[c] = out.groupby("ItemCode", observed=True, sort=False)[c].ffill()
            out[c] = out.groupby("ItemCode", observed=True, sort=False)[c].bfill()
            med = out[c].median()
            out[c] = out[c].fillna(med).astype("float32")
    else:
        sku_price = out.groupby("ItemCode", observed=True)["UnitPrice"].transform(lambda s: s.replace(0, np.nan).median())
        sku_cost = out.groupby("ItemCode", observed=True)["UnitCost"].transform(lambda s: s.replace(0, np.nan).median())
        out["UnitPrice"] = out["UnitPrice"].replace(0, np.nan).fillna(sku_price).fillna(out["UnitPrice"].replace(0, np.nan).median()).astype("float32")
        out["UnitCost"] = out["UnitCost"].replace(0, np.nan).fillna(sku_cost).fillna(out["UnitCost"].replace(0, np.nan).median()).astype("float32")

    if PROFILE["outlier"]:
        for c in ["UnitPrice", "UnitCost"]:
            lo = out[c].quantile(0.001)
            hi = out[c].quantile(0.999)
            out[f"{c}_capped"] = out[c].clip(lo, hi).astype("float32")
        out["margin_capped"] = (out["UnitPrice_capped"] - out["UnitCost_capped"]).astype("float32")
    else:
        out["UnitPrice_capped"] = out["UnitPrice"].astype("float32")
        out["UnitCost_capped"] = out["UnitCost"].astype("float32")
        out["margin_capped"] = (out["UnitPrice"] - out["UnitCost"]).astype("float32")

    out["margin"] = (out["UnitPrice"] - out["UnitCost"]).astype("float32")
    out["margin_pct"] = (out["margin"] / out["UnitPrice"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
    return out

panel = fill_value_columns(panel)
print("panel", panel.shape, "daily rows", daily.shape, "negative daily qty", int((daily["Quantity"] < 0).sum()))

# %% [4] Feature engineering
def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dow"] = out["Date"].dt.dayofweek.astype("int8")
    out["dom"] = out["Date"].dt.day.astype("int8")
    out["month"] = out["Date"].dt.month.astype("int8")
    out["quarter"] = out["Date"].dt.quarter.astype("int8")
    out["woy"] = out["Date"].dt.isocalendar().week.astype("int16")
    out["is_weekend"] = (out["dow"] >= 5).astype("int8")
    out["is_month_start"] = out["Date"].dt.is_month_start.astype("int8")
    out["is_month_end"] = out["Date"].dt.is_month_end.astype("int8")
    return out


def add_demand_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby("ItemCode", observed=True, sort=False)["y"]
    lags = [1, 2, 3, 7, 14, 21, 28, 56, 84]
    windows = [7, 14, 28, 56, 84]

    for lag in lags:
        out[f"lag_{lag}"] = g.shift(lag).astype("float32")

    shifted = g.shift(1)
    for w in windows:
        roll = shifted.groupby(out["ItemCode"], observed=True).rolling(w)
        out[f"rmean_{w}"] = roll.mean().reset_index(level=0, drop=True).astype("float32")
        out[f"rstd_{w}"] = roll.std().reset_index(level=0, drop=True).astype("float32")
        out[f"rsum_{w}"] = roll.sum().reset_index(level=0, drop=True).astype("float32")
        out[f"rmax_{w}"] = roll.max().reset_index(level=0, drop=True).astype("float32")
        out[f"rnz_{w}"] = roll.apply(lambda x: np.count_nonzero(x > 0), raw=True).reset_index(level=0, drop=True).astype("float32")
        out[f"zero_ratio_{w}"] = (1.0 - out[f"rnz_{w}"] / float(w)).astype("float32")
        out[f"avg_nonzero_{w}"] = (out[f"rsum_{w}"] / out[f"rnz_{w}"].replace(0, np.nan)).fillna(0).astype("float32")

    sale = out["y"] > 0
    last_sale_idx = out.groupby("ItemCode", observed=True, sort=False).cumcount().where(sale)
    last_sale_idx = last_sale_idx.groupby(out["ItemCode"], observed=True).ffill()
    last_sale_idx = last_sale_idx.groupby(out["ItemCode"], observed=True).shift(1)
    cur_idx = out.groupby("ItemCode", observed=True, sort=False).cumcount()
    out["days_since_last_sale"] = (cur_idx - last_sale_idx).fillna(9999).clip(0, 9999).astype("int16")
    last_nonzero = out["y"].where(sale).groupby(out["ItemCode"], observed=True).ffill()
    out["last_nonzero_qty"] = last_nonzero.groupby(out["ItemCode"], observed=True).shift(1).fillna(0).astype("float32")
    return out


def add_value_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby("ItemCode", observed=True, sort=False)
    base_cols = ["UnitPrice", "UnitCost", "margin", "margin_pct", "UnitPrice_capped", "UnitCost_capped", "margin_capped"]

    if PROFILE["fill"] == "ffill":
        for c in base_cols:
            for lag in [1, 7, 28]:
                out[f"{c}_lag_{lag}"] = g[c].shift(lag).astype("float32")
            out[f"{c}_chg_7"] = (out[c] - g[c].shift(7)).astype("float32")
            shifted_c = g[c].shift(1)
            out[f"{c}_rmean_28"] = shifted_c.groupby(out["ItemCode"], observed=True).rolling(28).mean().reset_index(level=0, drop=True).astype("float32")
    else:
        med_cols = ["UnitPrice", "UnitCost", "margin", "margin_pct"]
        for c in med_cols:
            med = out.groupby("ItemCode", observed=True)[c].transform("median").astype("float32")
            out[f"sku_{c}_median"] = med
            out[f"{c}_vs_sku_median"] = (out[c] - med).astype("float32")

    if PROFILE["outlier"]:
        out["sales_err_ratio"] = (out["sales_err"] / out["SalesAmount"].abs().replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 10).astype("float32")
        out["cost_err_ratio"] = (out["cost_err"] / out["CostAmount"].abs().replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 10).astype("float32")
        out["is_value_outlier"] = ((out["sales_err_ratio"] > 0.25) | (out["cost_err_ratio"] > 0.25) | (out["is_price_bad"] > 0) | (out["is_cost_bad"] > 0)).astype("int8")

    return out


def add_sku_stats(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    sku = out.groupby("ItemCode", observed=True)["y"]
    stats = sku.agg(["mean", "std", "sum"])
    stats.columns = ["sku_y_mean", "sku_y_std", "sku_y_sum"]
    active = sku.apply(lambda s: np.mean(s > 0)).rename("sku_active_ratio")
    stats = stats.join(active)
    profit = out.groupby("ItemCode", observed=True)["profit"].sum().clip(lower=0)
    if float(profit.sum()) > 0:
        stats["profit_weight"] = profit / profit.sum()
    else:
        stats["profit_weight"] = 1.0 / len(profit)
    stats["profit_weight_bucket"] = pd.qcut(stats["profit_weight"].rank(method="first"), 10, labels=False).astype("int8")
    return out.merge(stats.reset_index(), on="ItemCode", how="left")


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    out = add_calendar_features(df)
    out = add_demand_features(out)
    out = add_value_features(out)
    out = add_sku_stats(out)
    return out

# %% [5] WRMSSE
class ModelWrapper:
    def __init__(self, name, model, log_target=False):
        self.name = name
        self.model = model
        self.log_target = log_target

    def fit(self, X, y):
        yy = np.log1p(y) if self.log_target else y
        self.model.fit(X, yy)
        return self

    def predict(self, X):
        p = self.model.predict(X)
        if self.log_target:
            p = np.expm1(p)
        return np.clip(p, 0, None)


def build_wrmsse_artifacts(train_panel: pd.DataFrame):
    diff = train_panel.groupby("ItemCode", observed=True)["y"].diff()
    scale = diff.pow(2).groupby(train_panel["ItemCode"]).mean().replace(0, np.nan)
    med = scale.median()
    scale = scale.fillna(float(med) if np.isfinite(med) and med > 0 else 1.0)
    profit = train_panel.groupby("ItemCode", observed=True)["profit"].sum().clip(lower=0)
    weights = profit / profit.sum() if float(profit.sum()) > 0 else pd.Series(1.0 / len(profit), index=profit.index)
    return scale.astype("float64"), weights.astype("float64")


def wrmsse_score(y_true, y_pred, sku, scale, weights):
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    sku = np.asarray(sku)
    df = pd.DataFrame({"sku": sku, "err2": (y_true - y_pred) ** 2})
    mse = df.groupby("sku", observed=True)["err2"].mean()
    common = mse.index.intersection(scale.index).intersection(weights.index)
    if len(common) == 0:
        return float("nan")
    rmsse = np.sqrt((mse.loc[common] / scale.loc[common]).clip(lower=0))
    return float((rmsse * weights.loc[common]).sum())


def optimize_blend(preds, y_true, sku, scale, weights):
    names = list(preds)
    P = np.column_stack([preds[n] for n in names])
    y_true_arr = np.asarray(y_true, dtype="float64")

    def obj(w):
        p = np.clip(P @ w, 0, None)
        return wrmsse_score(y_true_arr, p, sku, scale, weights)

    x0 = np.ones(len(names)) / len(names)
    bounds = [(0.0, 1.0)] * len(names)
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 200})
    w = res.x if res.success else x0
    w = np.clip(w, 0, None)
    w = w / w.sum() if w.sum() > 0 else x0
    return names, w, obj(w)

# %% [6] Train validation candidates
feat_df = make_features(panel).sort_values(["ItemCode", "Date"]).reset_index(drop=True)
last_date = feat_df["Date"].max()
valid_start = last_date - pd.Timedelta(days=VAL_DAYS - 1)

exclude = {"Date", "ItemCode", "Quantity", "SalesAmount", "CostAmount", "UnitPrice", "UnitCost", "profit", "y"}
feature_cols = [c for c in feat_df.columns if c not in exclude]

train_df = feat_df[feat_df["Date"] < valid_start].copy()
valid_df = feat_df[feat_df["Date"] >= valid_start].copy()
X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
y_train = train_df["y"].values.astype("float32")
X_valid = valid_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
y_valid = valid_df["y"].values.astype("float32")
scale, weights = build_wrmsse_artifacts(train_df[["ItemCode", "y", "profit"]])

def candidate_models():
    models = []
    models.append(ModelWrapper("rf", RandomForestRegressor(n_estimators=500, max_depth=18, min_samples_leaf=2, max_features="sqrt", bootstrap=True, n_jobs=-1, random_state=SEED)))
    models.append(ModelWrapper("extra_trees", ExtraTreesRegressor(n_estimators=500, max_depth=22, min_samples_leaf=2, max_features="sqrt", n_jobs=-1, random_state=SEED)))
    models.append(ModelWrapper("lgb_tweedie", lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.2, n_estimators=900, learning_rate=0.035, num_leaves=255, min_child_samples=60, subsample=0.85, colsample_bytree=0.85, reg_alpha=0.03, reg_lambda=0.8, random_state=SEED, n_jobs=-1, verbosity=-1)))
    models.append(ModelWrapper("lgb_poisson", lgb.LGBMRegressor(objective="poisson", n_estimators=900, learning_rate=0.035, num_leaves=255, min_child_samples=60, subsample=0.85, colsample_bytree=0.85, reg_alpha=0.03, reg_lambda=0.8, random_state=SEED + 1, n_jobs=-1, verbosity=-1)))
    models.append(ModelWrapper("lgb_log1p", lgb.LGBMRegressor(objective="regression", n_estimators=900, learning_rate=0.035, num_leaves=255, min_child_samples=60, subsample=0.85, colsample_bytree=0.85, reg_alpha=0.03, reg_lambda=0.8, random_state=SEED + 2, n_jobs=-1, verbosity=-1), log_target=True))
    models.append(ModelWrapper("lgb_rf", lgb.LGBMRegressor(boosting_type="rf", objective="regression", n_estimators=700, learning_rate=0.05, num_leaves=255, min_child_samples=40, bagging_fraction=0.75, bagging_freq=1, feature_fraction=0.75, random_state=SEED + 3, n_jobs=-1, verbosity=-1)))
    if USE_XGB:
        models.append(ModelWrapper("xgb", xgb.XGBRegressor(n_estimators=900, learning_rate=0.035, max_depth=10, subsample=0.85, colsample_bytree=0.85, reg_alpha=0.03, reg_lambda=0.8, objective="reg:squarederror", tree_method="hist", random_state=SEED, n_jobs=-1)))
    return models

valid_preds = {}
valid_scores = []
trained = []
print("train rows", len(train_df), "valid rows", len(valid_df), "features", len(feature_cols))
for wrapper in candidate_models():
    print("training", wrapper.name)
    wrapper.fit(X_train, y_train)
    pred = wrapper.predict(X_valid)
    valid_preds[wrapper.name] = pred
    rmse = float(np.sqrt(mean_squared_error(y_valid, pred)))
    wr = wrmsse_score(y_valid, pred, valid_df["ItemCode"], scale, weights)
    valid_scores.append({"model": wrapper.name, "rmse": rmse, "wrmsse": wr})
    trained.append(wrapper)
    print(wrapper.name, "rmse", rmse, "wrmsse", wr)

score_df = pd.DataFrame(valid_scores).sort_values(["wrmsse", "rmse"])
print(score_df)

# %% [7] Blend optimization
names, global_w, global_wr = optimize_blend(valid_preds, y_valid, valid_df["ItemCode"], scale, weights)
global_pred = np.clip(np.column_stack([valid_preds[n] for n in names]) @ global_w, 0, None)
global_rmse = float(np.sqrt(mean_squared_error(y_valid, global_pred)))
print("global weights", dict(zip(names, map(float, global_w))), "rmse", global_rmse, "wrmsse", global_wr)

top_skus = weights.sort_values(ascending=False).head(TOP_WEIGHT_N).index
top_mask = valid_df["ItemCode"].isin(top_skus).values
if top_mask.sum() > 0:
    top_preds = {n: p[top_mask] for n, p in valid_preds.items()}
    top_scale = scale.loc[scale.index.intersection(top_skus)]
    top_weights = weights.loc[weights.index.intersection(top_skus)]
    top_weights = top_weights / top_weights.sum() if float(top_weights.sum()) > 0 else top_weights
    top_names, top_w, top_wr = optimize_blend(top_preds, y_valid[top_mask], valid_df.loc[top_mask, "ItemCode"], top_scale, top_weights)
else:
    top_names, top_w, top_wr = names, global_w, global_wr
print("top weights", dict(zip(top_names, map(float, top_w))), "top_wrmsse", top_wr)

# %% [8] Refit full history
full_feat = make_features(panel).sort_values(["ItemCode", "Date"]).reset_index(drop=True)
X_full = full_feat[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
y_full = full_feat["y"].values.astype("float32")
refit = []
for wrapper in candidate_models():
    print("refit", wrapper.name)
    wrapper.fit(X_full, y_full)
    refit.append(wrapper)
refit_by_name = {m.name: m for m in refit}

# %% [9] Recursive forecast 56 days
state_cols = ["Date", "ItemCode", "Quantity", "SalesAmount", "CostAmount", "UnitPrice", "UnitCost", "sales_err", "cost_err", "is_price_bad", "is_cost_bad", "UnitPrice_capped", "UnitCost_capped", "margin_capped", "margin", "margin_pct", "profit", "y"]
state = panel[state_cols].copy().sort_values(["ItemCode", "Date"]).reset_index(drop=True)
last_date = state["Date"].max()
future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=HORIZON, freq="D")
skus = state["ItemCode"].cat.categories if str(state["ItemCode"].dtype) == "category" else pd.Index(state["ItemCode"].unique())

pred_rows = []
for d in future_dates:
    last_vals = state.groupby("ItemCode", observed=True).tail(1).set_index("ItemCode")
    step = pd.DataFrame({"Date": d, "ItemCode": skus})
    step["ItemCode"] = step["ItemCode"].astype("category")
    step = step.join(last_vals[["UnitPrice", "UnitCost", "UnitPrice_capped", "UnitCost_capped", "margin_capped", "margin", "margin_pct"]], on="ItemCode")
    for c in ["Quantity", "SalesAmount", "CostAmount", "sales_err", "cost_err", "is_price_bad", "is_cost_bad", "profit"]:
        step[c] = 0.0
    step["y"] = np.nan
    state = pd.concat([state, step[state_cols]], ignore_index=True).sort_values(["ItemCode", "Date"]).reset_index(drop=True)

    feat_state = make_features(state)
    cur = feat_state[feat_state["Date"] == d].copy()
    X_cur = cur[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")

    pred_mat = np.column_stack([refit_by_name[n].predict(X_cur) for n in names])
    pred = np.clip(pred_mat @ global_w, 0, None)

    cur_sku = cur["ItemCode"].astype(str).values
    top_future_mask = np.isin(cur_sku, top_skus.astype(str))
    if top_future_mask.any():
        top_mat = np.column_stack([refit_by_name[n].predict(X_cur.loc[top_future_mask]) for n in top_names])
        pred[top_future_mask] = np.clip(top_mat @ top_w, 0, None)

    state.loc[state["Date"] == d, "y"] = pred.astype("float32")
    pred_rows.append(pd.DataFrame({"Date": d, "ItemCode": cur["ItemCode"].values, "pred": pred}))
    print("forecast", d.date(), "q50", float(np.quantile(pred, 0.5)), "q99", float(np.quantile(pred, 0.99)), "near0", float(np.mean(pred < 1e-6)))

pred56 = pd.concat(pred_rows, ignore_index=True)
print("pred56", pred56.shape, pred56["pred"].quantile([0, 0.5, 0.9, 0.99]).to_dict())

# %% [10] Submission export
sample = pd.read_csv(SUB_PATH)
fcols = [f"F{i}" for i in range(1, 29)]
sku_to_vals = pred56.sort_values(["ItemCode", "Date"]).groupby("ItemCode", observed=True)["pred"].apply(list).to_dict()
rows = []
for rid in sample["id"]:
    sku, suffix = rid.rsplit("_", 1)
    vals = sku_to_vals.get(sku, [0.0] * HORIZON)
    block = vals[:28] if suffix == "validation" else vals[28:56]
    rows.append([rid] + [float(max(0.0, v)) for v in block])
sub = pd.DataFrame(rows, columns=["id"] + fcols)
assert sub.shape == sample.shape
assert sub["id"].is_unique
assert set(sub["id"]) == set(sample["id"])
assert np.isfinite(sub[fcols].to_numpy()).all()
assert (sub[fcols].to_numpy() >= 0).all()
sub.to_csv(OUT_PATH, index=False)

metrics = {
    "version": VERSION,
    "train_path": TRAIN_PATH,
    "profile": PROFILE,
    "scores": valid_scores,
    "global_weights": dict(zip(names, map(float, global_w))),
    "global_rmse": global_rmse,
    "global_wrmsse": global_wr,
    "top_weights": dict(zip(top_names, map(float, top_w))),
    "top_wrmsse": top_wr,
    "pred_quantiles": {str(k): float(v) for k, v in pred56["pred"].quantile([0, 0.5, 0.9, 0.99]).to_dict().items()},
    "pred_near_zero_share": float((pred56["pred"] < 1e-6).mean()),
    "feature_count": len(feature_cols),
}
Path(METRICS_PATH).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print("saved", OUT_PATH, METRICS_PATH)
