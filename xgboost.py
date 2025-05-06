import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from concurrent.futures import ThreadPoolExecutor

from sklearn.model_selection import train_test_split, ParameterSampler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from xgboost import plot_importance

plt.style.use("ggplot")


def read_and_merge(orig_pattern="sample_orig_*.txt", svc_pattern="sample_svcg_*.txt"):
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

    orig_files = sorted(glob(orig_pattern))
    svc_files  = sorted(glob(svc_pattern))
    if not orig_files or not svc_files:
        raise FileNotFoundError("Place sample_orig_*.txt and sample_svcg_*.txt here.")

    def _read(path, cols):
        return pd.read_csv(path, sep="|", header=None, names=cols, dtype=str)

    with ThreadPoolExecutor() as exe:
        orig_df = pd.concat(exe.map(lambda f: _read(f, orig_cols), orig_files), ignore_index=True)
        svc_df  = pd.concat(exe.map(lambda f: _read(f, svc_cols), svc_files),  ignore_index=True)

    # parse dates
    orig_df["FIRST_PAYMENT_DATE"] = pd.to_datetime(orig_df["FIRST_PAYMENT_DATE"], format="%Y%m", errors="coerce")
    orig_df["MATURITY_DATE"]      = pd.to_datetime(orig_df["MATURITY_DATE"],      format="%Y%m", errors="coerce")
    svc_df["PERIOD"]              = pd.to_datetime(svc_df["PERIOD"],             format="%Y%m", errors="coerce")
    svc_df["ZERO_BALANCE_DATE"]   = pd.to_datetime(svc_df["ZERO_BALANCE_DATE"],   format="%Y%m%d", errors="coerce")

    # numeric casts
    orig_num = ["ORIG_UPB","ORIG_LTV","ORIG_DTI","ORIG_INTEREST_RATE","ORIG_TERM"]
    svc_num  = ["CUR_ACTUAL_UPB","ZB_REMOVAL_UPB","DELINQ_ACCRUED_INT",
                "NET_SALE_PROCS","MI_RECOV","NON_MI_RECOV","TOTAL_EXPENSES",
                "LOAN_AGE","ELTV","CUR_INTEREST_RATE"]
    orig_df[orig_num] = orig_df[orig_num].apply(pd.to_numeric, errors="coerce")
    svc_df[svc_num]   = svc_df[svc_num].apply(pd.to_numeric, errors="coerce")

    # merge + compute DTV
    orig_sub = orig_df[["LOAN_SEQ","ORIG_UPB","ORIG_LTV","ORIG_DTI","FIRST_PAYMENT_DATE"]]
    panel   = svc_df.merge(orig_sub, on="LOAN_SEQ", how="left")
    panel["DTV"] = panel["CUR_ACTUAL_UPB"] / ((panel["ELTV"]/100) * panel["ORIG_UPB"])

    # filter terminations + loss
    term_codes = {"02","03","09","15","16","96"}
    disp = panel[panel["ZERO_BALANCE_CODE"].isin(term_codes)].copy()
    disp["LOSS_AMT"] = (
          disp["ZB_REMOVAL_UPB"].fillna(0)
        + disp["DELINQ_ACCRUED_INT"].fillna(0)
        - disp["NET_SALE_PROCS"].fillna(0)
        - disp["MI_RECOV"].fillna(0)
        - disp["NON_MI_RECOV"].fillna(0)
        - disp["TOTAL_EXPENSES"].fillna(0)
    )
    return disp


def add_features(disp):
    # time & interactions
    disp["ORIG_YEAR"] = disp["FIRST_PAYMENT_DATE"].dt.year
    disp["SEASONING"] = ((disp["ZERO_BALANCE_DATE"] - disp["FIRST_PAYMENT_DATE"])
                          .dt.days.clip(lower=0) / 30.0)
    disp["LTV_DIFF"]  = disp["ORIG_LTV"] - disp["DTV"] * 100
    disp["DTI_X_IR"]  = disp["ORIG_DTI"] * disp["CUR_INTEREST_RATE"]
    disp["LTV_HIGH"]  = (disp["ORIG_LTV"] > 60).astype(int)

    # one-hot 4 categories
    cats = ["CHANNEL","PROP_STATE","OCCUPANCY_STATUS","LOAN_PURPOSE"]
    for c in cats:
        if c in disp: disp[c] = disp[c].astype(str)
    disp = pd.get_dummies(disp, columns=[c for c in cats if c in disp], drop_first=True)
    return disp


def prepare_matrix(disp):
    # base numeric features
    base_feats = [
        "ORIG_LTV","DTV","LOAN_AGE","CUR_INTEREST_RATE","ORIG_DTI",
        "ORIG_YEAR","SEASONING","LTV_DIFF","DTI_X_IR","LTV_HIGH"
    ]
    # dummy prefixes only
    prefixes = ["CHANNEL_","PROP_STATE_","OCCUPANCY_STATUS_","LOAN_PURPOSE_"]
    dummy_feats = [c for c in disp.columns if any(c.startswith(p) for p in prefixes)]

    features = base_feats + dummy_feats

    mdl = disp[features + ["LOSS_AMT"]].dropna(subset=["LOSS_AMT"]).copy()
    # all these are now numeric
    mdl[features] = mdl[features].fillna(mdl[features].median())

    X = mdl[features].astype(float)
    y_raw = mdl["LOSS_AMT"].clip(upper=mdl["LOSS_AMT"].quantile(0.99))
    y_log = np.log1p(y_raw)

    return X, y_raw, y_log


def tune_xgb(X_tr, y_log_tr):
    dtrain = xgb.DMatrix(X_tr, label=y_log_tr)
    fixed = {
        "objective":"reg:squarederror","eval_metric":"rmsle",
        "tree_method":"hist","seed":42
    }
    grid = {
      "eta":[0.01,0.03,0.05,0.1],
      "max_depth":[4,6,8],
      "subsample":[0.6,0.8,1.0],
      "colsample_bytree":[0.6,0.8,1.0],
      "reg_alpha":[0,0.1,1.0],
      "reg_lambda":[1.0,5.0,20.0]
    }

    best_score,best_round,best_params = np.inf,0,None
    print("▶ Random search + 5-fold CV (RMSLE)…")
    for trial in ParameterSampler(grid, n_iter=30, random_state=42):
        params = {**fixed,**trial}
        cvres = xgb.cv(params, dtrain, num_boost_round=500, nfold=5,
                       early_stopping_rounds=20, metrics="rmsle",
                       seed=42, as_pandas=True, verbose_eval=False)
        score = cvres["test-rmsle-mean"].min()
        rounds= int(cvres["test-rmsle-mean"].idxmin())+1
        if score<best_score:
            best_score,best_round,best_params = score,rounds,params

    print(f"▶ Best RMSLE={best_score:.4f} @ round={best_round}")
    bst = xgb.train(best_params, dtrain, num_boost_round=best_round)
    return bst


def evaluate(bst, X_te, y_log_te, y_raw_te):
    dtest = xgb.DMatrix(X_te)
    y_log_pred = bst.predict(dtest)
    y_pred     = np.expm1(y_log_pred)
    y_true     = np.expm1(y_log_te)
    mask = np.isfinite(y_pred)&np.isfinite(y_true)
    y_pred, y_true = y_pred[mask], y_true[mask]

    rmse = mean_squared_error(y_true,y_pred,squared=False)
    r2   = r2_score(y_true,y_pred)
    print(f"▶ Test RMSE = {rmse:.0f}, R² = {r2:.3f}")

    plt.figure(figsize=(6,6))
    plt.scatter(y_pred,y_true,alpha=0.5,s=20)
    lims=[0, max(y_pred.max(),y_true.max())]
    plt.plot(lims,lims,"r--")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title(f"Pred vs Actual — RMSE={rmse:.0f}, R²={r2:.3f}")
    plt.tight_layout(); plt.show()

    fig,ax=plt.subplots(figsize=(8,5))
    plot_importance(bst,ax=ax,importance_type="gain",max_num_features=15)
    ax.set_title("Feature Importance (Gain)")
    plt.tight_layout(); plt.show()


if __name__=="__main__":
    disp = read_and_merge()
    disp = add_features(disp)
    X, y_raw, y_log = prepare_matrix(disp)

    # train-test split
    X_tr,X_te,y_log_tr,y_log_te,y_raw_tr,y_raw_te = train_test_split(
        X,y_log,y_raw,test_size=0.20,random_state=42
    )

    bst = tune_xgb(X_tr, y_log_tr)
    evaluate(bst, X_te, y_log_te, y_raw_te)





    dtest  = xgb.DMatrix(X_te)
y_pred = bst.predict(dtest)

y_true_arr = y_true.reset_index(drop=True).values  

n = min(len(y_true_arr), len(y_pred))
y_true_arr = y_true_arr[:n]
y_pred     = y_pred[:n]
residuals = y_true_arr - y_pred

import matplotlib.pyplot as plt
plt.figure(figsize=(5,3))
plt.hist(residuals, bins=30, edgecolor="k")
plt.title("Residuals Distribution")
plt.xlabel("y_true - y_pred")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

plt.figure(figsize=(5,3))
plt.scatter(y_pred, residuals, alpha=0.3, s=10)
plt.axhline(0, color="r", linestyle="--")
plt.title("Pred vs Residuals")
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.tight_layout()
plt.show()



y_true_arr = y_true.reset_index(drop=True).values
y_true_arr = y_true_arr[:n]
y_pred_arr = y_pred[:n]

import pandas as pd, numpy as np
df = pd.DataFrame({"pred": y_pred_arr, "true": y_true_arr})

df["decile"] = pd.qcut(df["pred"], 10, labels=False)
decile_err = df.groupby("decile").apply(
    lambda g: np.sqrt(((g.true - g.pred)**2).mean())
)
print(decile_err)







import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

dtest = xgb.DMatrix(X_te)
shap_vals = bst.predict(dtest, pred_contribs=True)
feature_names = list(X_te.columns)
shap_contribs = shap_vals[:, :len(feature_names)]
shap_df = pd.DataFrame(shap_contribs, columns=feature_names)

mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False).head(15)
plt.figure(figsize=(8,4))
mean_abs_shap.plot.barh()
plt.gca().invert_yaxis()
plt.title("Approximate SHAP Feature Importance")
plt.xlabel("Mean |SHAP value|")
plt.tight_layout()
plt.show()

feat = "ORIG_LTV"
plt.figure(figsize=(5,4))
plt.scatter(X_te[feat], shap_df[feat], alpha=0.3, s=20)
plt.xlabel(feat)
plt.ylabel("SHAP value")
plt.title(f"SHAP Scatter for {feat}")
plt.tight_layout()
plt.show()

grid = np.linspace(X_te[feat].min(), X_te[feat].max(), 20)
pdp_vals = []
for val in grid:
    X_tmp = X_te.copy()
    X_tmp[feat] = val
    pdp_vals.append(bst.predict(xgb.DMatrix(X_tmp)).mean())

plt.figure(figsize=(5,4))
plt.plot(grid, pdp_vals, marker="o")
plt.xlabel(feat)
plt.ylabel("Average predicted loss")
plt.title(f"PDP for {feat}")
plt.tight_layout()
plt.show()