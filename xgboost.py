# ─── 1) IMPORTS & STYLE ─────────────────────────────────────────────────
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt

from glob                                import glob
from sklearn.linear_model                import LogisticRegression
import statsmodels.api                   as sm
from statsmodels.genmod.families         import Gamma
from statsmodels.genmod.families.links   import Log as loglink

plt.style.use("ggplot")


# ─── 2) FIND & LOAD YOUR FILES ───────────────────────────────────────────
orig_files = sorted(glob("sample_orig_*.txt"))
svc_files  = sorted(glob("sample_svcg_*.txt"))
if not orig_files or not svc_files:
    raise FileNotFoundError("Make sure sample_orig_*.txt and sample_svcg_*.txt are in your cwd")

orig_cols = [
    "CREDIT_SCORE","FIRST_PAYMENT_DATE","FIRST_TIME_HOMEBUYER_FLAG","MATURITY_DATE",
    "MSA","MI_PERCENT","NUM_UNITS","OCCUPANCY_STATUS","ORIG_CLTV","ORIG_DTI","ORIG_UPB",
    "ORIG_LTV","ORIG_INTEREST_RATE","CHANNEL","PPM_FLAG","AMORT_TYPE","PROP_STATE",
    "PROP_TYPE","POSTAL_CODE","LOAN_SEQ","LOAN_PURPOSE","ORIG_TERM","NUM_BORROWERS",
    "SELLER_NAME","SERVICER_NAME","SUPER_CONFORM_FLAG","PRE_RELIEF_LSN",
    "PROGRAM_IND","RELIEF_REFI_IND","PROP_VAL_METHOD","IO_IND","MI_CANCEL_IND"
]
svc_cols = [
    "LOAN_SEQ","PERIOD","CUR_ACTUAL_UPB","CUR_DELINQ_STATUS","LOAN_AGE",
    "REMAIN_MONTHS","DEFECT_SETTLE_DATE","MOD_FLAG","ZERO_BALANCE_CODE",
    "ZERO_BALANCE_DATE","CUR_INTEREST_RATE","CUR_NON_INT_UPB","DDLPI","MI_RECOV",
    "NET_SALE_PROCS","NON_MI_RECOV","TOTAL_EXPENSES","LEGAL_COSTS",
    "MAINT_PRES_COSTS","TAXES_INS","MISC_EXPENSES","ACTUAL_LOSS","CUM_MOD_COST",
    "STEP_MOD_FLAG","PAYMENT_DEFERRAL","ELTV","ZB_REMOVAL_UPB",
    "DELINQ_ACCRUED_INT","DISASTER_FLAG","BORROWER_ASSIST_CODE",
    "CUR_MONTH_MOD_COST","INT_BEARING_UPB"
]

orig_df = pd.concat(
    [pd.read_csv(f, sep="|", header=None, names=orig_cols, dtype=str) for f in orig_files],
    ignore_index=True
)
svc_df  = pd.concat(
    [pd.read_csv(f, sep="|", header=None, names=svc_cols, dtype=str) for f in svc_files],
    ignore_index=True
)


# ─── 3) PARSE DATES & CAST NUMERIC FIELDS ───────────────────────────────
orig_df["FIRST_PAYMENT_DATE"] = pd.to_datetime(orig_df["FIRST_PAYMENT_DATE"], format="%Y%m", errors="coerce")
orig_df["MATURITY_DATE"]      = pd.to_datetime(orig_df["MATURITY_DATE"],      format="%Y%m", errors="coerce")
svc_df ["PERIOD"]             = pd.to_datetime(svc_df ["PERIOD"],             format="%Y%m", errors="coerce")

num_fields_svc  = ["CUR_ACTUAL_UPB","ZB_REMOVAL_UPB","DELINQ_ACCRUED_INT",
                   "NET_SALE_PROCS","MI_RECOV","NON_MI_RECOV","TOTAL_EXPENSES",
                   "LOAN_AGE","ELTV","CUR_INTEREST_RATE"]
num_fields_orig = ["ORIG_UPB","ORIG_LTV","ORIG_DTI","ORIG_INTEREST_RATE","ORIG_TERM"]

svc_df [num_fields_svc ] = svc_df [num_fields_svc ].apply(pd.to_numeric, errors="coerce")
orig_df[num_fields_orig] = orig_df[num_fields_orig].apply(pd.to_numeric, errors="coerce")


# ─── 4) BUILD PANEL & DTV ────────────────────────────────────────────────
orig_sub = orig_df[["LOAN_SEQ","ORIG_UPB","ORIG_LTV","ORIG_DTI"]]
panel    = svc_df.merge(orig_sub, on="LOAN_SEQ", how="left")
panel["DTV"] = panel["CUR_ACTUAL_UPB"] / ((panel["ELTV"] / 100) * panel["ORIG_UPB"])


# ─── 5) TERMINATION RECORDS & LOSS AMOUNT ───────────────────────────────
term_codes = {"02","03","09","16","96"}        # paid‐off, foreclosure, REO, etc.
disp = panel[panel["ZERO_BALANCE_CODE"].isin(term_codes)].copy()

disp["LOSS_AMT"] = (
      disp["ZB_REMOVAL_UPB"].fillna(0)
    + disp["DELINQ_ACCRUED_INT"].fillna(0)
    - disp["NET_SALE_PROCS"].fillna(0)
    - disp["MI_RECOV"].fillna(0)
    - disp["NON_MI_RECOV"].fillna(0)
    - disp["TOTAL_EXPENSES"].fillna(0)
)
disp["LOSS_ZERO"] = (disp["LOSS_AMT"] == 0)


# ─── 6) LGD FEATURE SET ─────────────────────────────────────────────────
lgd_features = ["ORIG_LTV","DTV","LOAN_AGE","CUR_INTEREST_RATE","ORIG_DTI"]
lgd_df = disp[["LOAN_SEQ","PERIOD"] + lgd_features + ["LOSS_ZERO","LOSS_AMT","CUR_ACTUAL_UPB","ZB_REMOVAL_UPB"]].dropna()


# ─── 7) STAGE 1 — ZERO-LOSS CLASSIFIER ──────────────────────────────────
X0 = lgd_df[lgd_features].astype(float)
y0 = lgd_df["LOSS_ZERO"].astype(int)

if y0.nunique() > 1:
    clf0 = LogisticRegression(solver="liblinear").fit(X0, y0)
    lgd_df["P_zero"] = clf0.predict_proba(X0)[:, 1]
else:
    lgd_df["P_zero"] = 0.0
lgd_df["P_pos"] = 1.0 - lgd_df["P_zero"]


# ─── 8) STAGE 2 — GAMMA GLM ON POSITIVE LOSSES ──────────────────────────
pos = lgd_df[~lgd_df["LOSS_ZERO"]].copy().assign(Intercept=1.0)

glm = sm.GLM(
    endog = pos["LOSS_AMT"],
    exog  = pos[["Intercept"] + lgd_features],
    family = Gamma(link=loglink())
).fit()

print(glm.summary())

preds = glm.predict(pos[["Intercept"] + lgd_features])
plt.figure(figsize=(6, 5))
plt.scatter(preds, pos["LOSS_AMT"], s=8, alpha=0.3)
plt.plot([0, preds.max()], [0, preds.max()], "r--")
plt.title("Gamma GLM: Predicted vs Actual Loss")
plt.xlabel("Predicted Loss");  plt.ylabel("Actual Loss")
plt.tight_layout()
plt.show()

pos["mu"] = preds


# ─── 9) MERGE LGD COMPONENTS & COMPUTE LGD_pred ─────────────────────────
disp = (
    disp
    .merge(lgd_df[["LOAN_SEQ","PERIOD","P_pos"]], on=["LOAN_SEQ","PERIOD"], how="left")
    .merge(pos   [["LOAN_SEQ","PERIOD","mu"   ]], on=["LOAN_SEQ","PERIOD"], how="left")
)
disp["mu"] = disp["mu"].fillna(0.0)
disp["ELG"] = disp["P_pos"] * disp["mu"]

# ► FIX: divide by UPB AT REMOVAL, not by zero current balance
base_upb = disp["ZB_REMOVAL_UPB"].replace({0: np.nan})
disp["LGD_pred"] = (disp["ELG"] / base_upb).fillna(0.0)


# ───10) PD TARGET (12-MONTH LOOK-AHEAD) ─────────────────────────────────
pan2 = panel.copy()
pan2["DQ_NUM"] = pd.to_numeric(pan2["CUR_DELINQ_STATUS"], errors="coerce").fillna(0)
pan2 = pan2.sort_values(["LOAN_SEQ","PERIOD"])

pan2["FWD_MAX_DQ"] = (
    pan2.groupby("LOAN_SEQ")["DQ_NUM"]
        .apply(lambda x: x[::-1].rolling(12, min_periods=1).max()[::-1].shift(-1).fillna(0))
        .reset_index(level=0, drop=True)
)
pan2["PD_TARGET"] = (pan2["FWD_MAX_DQ"] >= 2).astype(int)

pd_features = ["ORIG_LTV","DTV","LOAN_AGE","CUR_INTEREST_RATE","ORIG_DTI","CUR_ACTUAL_UPB"]
pd_df = pan2[["LOAN_SEQ","PERIOD"] + pd_features + ["PD_TARGET"]].dropna()


# ───11) PD MODEL (LOGIT) ────────────────────────────────────────────────
X1 = pd_df[pd_features].astype(float)
y1 = pd_df["PD_TARGET"].astype(int)

if y1.nunique() > 1:
    clf1 = LogisticRegression(max_iter=1000).fit(X1, y1)
    pd_df["PD12"] = clf1.predict_proba(X1)[:, 1]
else:
    pd_df["PD12"] = 0.0


# ───12) COMBINE PD × LGD → 12-MONTH EXPECTED LOSS ───────────────────────
el_df = (
    disp[["LOAN_SEQ","PERIOD","LGD_pred"]]
    .merge(pd_df[["LOAN_SEQ","PERIOD","PD12"]], on=["LOAN_SEQ","PERIOD"], how="inner")
)
el_df["EL_12M"] = el_df["PD12"] * el_df["LGD_pred"]

print(el_df.head())


# ─── XGBOOST ───────────────────────────────────────────
# ────────────────────────────────────────────────────────────
# XGBoost 回归（reg:tweedie）+ RandomizedSearchCV + xgb.cv Early Stopping
# ────────────────────────────────────────────────────────────
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor, plot_importance
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.metrics        import mean_squared_error
import matplotlib.pyplot as plt

X = mdl_df[features].astype(float)
# log1p 变换 + 异常值截断示例
y = mdl_df["LOSS_AMT"].clip(upper=mdl_df["LOSS_AMT"].quantile(0.99))
y_log = np.log1p(y)

# 1) train/test split
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)

# 2) sklearn-wrapper
xgb_base = XGBRegressor(
    objective="reg:tweedie",
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)
param_dist = {
    "n_estimators":     [100, 200, 500],
    "learning_rate":    [0.01, 0.03, 0.05, 0.1],
    "max_depth":        [3, 4, 5, 6],
    "subsample":        [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_alpha":        [0.0, 0.1, 1.0],
    "reg_lambda":       [1.0, 5.0, 10.0],
    "tweedie_variance_power": [1.2, 1.5, 1.8]
}
search = RandomizedSearchCV(
    xgb_base,
    param_distributions=param_dist,
    n_iter=30,
    scoring="neg_root_mean_squared_error",
    cv=KFold(5, shuffle=True, random_state=42),
    random_state=42,
    n_jobs=-1,
    verbose=1,
    refit=True
)
search.fit(X_tr, y_tr)
best_params = search.best_params_
print("▶︎ 随机搜索 最佳超参：", best_params)

dtrain = xgb.DMatrix(X_tr, label=y_tr)
dvalid = xgb.DMatrix(X_te, label=y_te)

params = {
    "objective":             "reg:tweedie",
    "tree_method":           "hist",
    "eval_metric":           "rmse",
    "seed":                  42,
    "eta":                   best_params["learning_rate"],
    "max_depth":             best_params["max_depth"],
    "subsample":             best_params["subsample"],
    "colsample_bytree":      best_params["colsample_bytree"],
    "reg_alpha":             best_params["reg_alpha"],
    "reg_lambda":            best_params["reg_lambda"],
    "tweedie_variance_power":best_params["tweedie_variance_power"]
}
cvres = xgb.cv(
    params,
    dtrain,
    num_boost_round=best_params["n_estimators"],
    nfold=5,
    early_stopping_rounds=20,
    verbose_eval=50,
    seed=42
)
best_round = len(cvres)
print(f"▶︎ CV 推荐最佳迭代次数：{best_round}")

bst = xgb.train(
    params,
    dtrain,
    num_boost_round=best_round,
    evals=[(dvalid, "valid")],
    early_stopping_rounds=20,
    verbose_eval=False
)

y_pred = bst.predict(dvalid)
rmse = mean_squared_error(y_te, y_pred, squared=False)
print(f"▶︎ 最终 Test RMSE = {rmse:.2f}")

plt.figure(figsize=(6,6))
plt.scatter(y_pred, y_te, alpha=0.5, s=20)
lims = [0, max(y_pred.max(), y_te.max())]
plt.plot(lims, lims, "r--")
plt.xlabel("Predicted Loss")
plt.ylabel("Actual Loss")
plt.title("XGBoost (Tweedie) Pred vs Actual")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(8,5))
plot_importance(
    bst,
    ax=ax,
    max_num_features=15,
    importance_type="gain",
    height=0.6
)
ax.set_title("Feature Importance (Gain)")
plt.tight_layout()
plt.show()
