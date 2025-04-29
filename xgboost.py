import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from glob                    import glob
from sklearn.model_selection import train_test_split, ParameterSampler
from sklearn.metrics        import mean_squared_error, r2_score
import xgboost as xgb
from xgboost                 import plot_importance

plt.style.use("ggplot")

# ────────────────────────────────────────────────────────────
# 1) 读取 & 合并 orig 与 svc 数据
# ────────────────────────────────────────────────────────────
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

orig_files = sorted(glob("sample_orig_*.txt"))
svc_files  = sorted(glob("sample_svcg_*.txt"))
if not orig_files or not svc_files:
    raise FileNotFoundError("请确认 sample_orig_*.txt 和 sample_svcg_*.txt 在当前目录下")

orig_df = pd.concat(
    [pd.read_csv(f, sep="|", header=None, names=orig_cols, dtype=str) for f in orig_files],
    ignore_index=True
)
svc_df  = pd.concat(
    [pd.read_csv(f, sep="|", header=None, names=svc_cols, dtype=str) for f in svc_files],
    ignore_index=True
)

# ────────────────────────────────────────────────────────────
# 2) 解析日期 & 强制数值
# ────────────────────────────────────────────────────────────
orig_df["FIRST_PAYMENT_DATE"] = pd.to_datetime(orig_df["FIRST_PAYMENT_DATE"], format="%Y%m", errors="coerce")
orig_df["MATURITY_DATE"]      = pd.to_datetime(orig_df["MATURITY_DATE"],      format="%Y%m", errors="coerce")
svc_df["PERIOD"]              = pd.to_datetime(svc_df["PERIOD"],              format="%Y%m", errors="coerce")

num_svc  = ["CUR_ACTUAL_UPB","ZB_REMOVAL_UPB","DELINQ_ACCRUED_INT",
            "NET_SALE_PROCS","MI_RECOV","NON_MI_RECOV","TOTAL_EXPENSES",
            "LOAN_AGE","ELTV","CUR_INTEREST_RATE"]
num_orig = ["ORIG_UPB","ORIG_LTV","ORIG_DTI","ORIG_INTEREST_RATE","ORIG_TERM"]

svc_df[num_svc]   = svc_df[num_svc].apply(pd.to_numeric, errors="coerce")
orig_df[num_orig] = orig_df[num_orig].apply(pd.to_numeric, errors="coerce")

# ────────────────────────────────────────────────────────────
# 3) 合并成面板 & 计算 DTV
# ────────────────────────────────────────────────────────────
orig_sub = orig_df[["LOAN_SEQ","ORIG_UPB","ORIG_LTV","ORIG_DTI","FIRST_PAYMENT_DATE"]]
panel   = svc_df.merge(orig_sub, on="LOAN_SEQ", how="left")
panel["DTV"] = panel["CUR_ACTUAL_UPB"] / ((panel["ELTV"]/100) * panel["ORIG_UPB"])

# ────────────────────────────────────────────────────────────
# 4) 过滤“终止”记录 & 计算 LOSS_AMT
# ────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────
# 5) 特征工程
# ────────────────────────────────────────────────────────────
# 5a) 新增时间衍生 & 交互
disp["ZERO_BALANCE_DATE"] = pd.to_datetime(disp["ZERO_BALANCE_DATE"], format="%Y%m%d", errors="coerce")
disp["ORIG_YEAR"]     = disp["FIRST_PAYMENT_DATE"].dt.year
disp["SEASONING"]     = (disp["ZERO_BALANCE_DATE"].sub(disp["FIRST_PAYMENT_DATE"])
                           .dt.days.clip(lower=0) / 30.0)
disp["LTV_DIFF"]      = disp["ORIG_LTV"] - disp["DTV"]*100
disp["DTI_X_IR"]      = disp["ORIG_DTI"] * disp["CUR_INTEREST_RATE"]

# 5b) One‐Hot 编码
cat_cols = ["CHANNEL","PROP_STATE","OCCUPANCY_STATUS","LOAN_PURPOSE"]
present  = [c for c in cat_cols if c in disp.columns]
disp = pd.get_dummies(disp, columns=present, drop_first=True)

# 5c) 最终特征列表
features = [
    "ORIG_LTV","DTV","LOAN_AGE","CUR_INTEREST_RATE","ORIG_DTI",
    "ORIG_YEAR","SEASONING","LTV_DIFF","DTI_X_IR"
]
features += [c for c in disp.columns
             if any(c.startswith(f + "_") for f in present)]

# ────────────────────────────────────────────────────────────
# 6) 构建训练表；对特征填 NA（中位数），仅丢弃 LOSS_AMT 缺失
# ────────────────────────────────────────────────────────────
mdl = disp[features + ["LOSS_AMT"]].copy()
mdl = mdl[~mdl["LOSS_AMT"].isna()]
mdl[features] = mdl[features].fillna(mdl[features].median())

X_full = mdl[features].astype(float)
y_raw  = mdl["LOSS_AMT"].clip(upper=mdl["LOSS_AMT"].quantile(0.99))
y_log  = np.log1p(y_raw)

print(f"总样本数：{len(X_full)}，特征数：{len(features)}")

# ────────────────────────────────────────────────────────────
# 7) 80/20 划分 + 用 xgb.cv 做早停 + 随机搜索优化 RMSLE
# ────────────────────────────────────────────────────────────
X_tr, X_te, y_tr, y_te = train_test_split(
    X_full, y_log,
    test_size=0.20,
    random_state=42
)
dtrain = xgb.DMatrix(X_tr, label=y_tr)

param_grid = {
    "eta":              [0.01, 0.03, 0.05, 0.1],
    "max_depth":        [4, 6, 8],
    "subsample":        [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_alpha":        [0, 0.1, 1.0],
    "reg_lambda":       [1.0,5.0,20.0],
}
fixed = {
    "objective":   "reg:squarederror",
    "eval_metric": "rmsle",
    "tree_method": "hist",
    "seed":        42,
}

best_score, best_rounds, best_params = np.inf, 0, None
print("▶︎ 随机搜索 + CV（优化 RMSLE）…")
t0 = time.time()
for trial in ParameterSampler(param_grid, n_iter=30, random_state=42):
    p = {**fixed, **trial}
    cvres = xgb.cv(
        p, dtrain,
        num_boost_round=500,
        nfold=5,
        early_stopping_rounds=20,
        as_pandas=True,
        verbose_eval=False,
        seed=42
    )
    rmsle = cvres["test-rmsle-mean"].min()
    rnd   = int(cvres["test-rmsle-mean"].idxmin()) + 1
    if rmsle < best_score:
        best_score, best_rounds, best_params = rmsle, rnd, p.copy()
        print(f"  ↳ 新最优 RMSLE={best_score:.4f} @ round={best_rounds}")

print(f"▶︎ 搜索完成，耗时 {time.time()-t0:.1f}s")
print("▶︎ 最优 Params:", best_params)
print("▶︎ 最优迭代轮数:", best_rounds)

# ────────────────────────────────────────────────────────────
# 8) 用最优配置训练 & 在测试集上评估
# ────────────────────────────────────────────────────────────
bst = xgb.train(best_params, dtrain, num_boost_round=best_rounds)

y_pred_log = bst.predict(xgb.DMatrix(X_te))
y_pred     = np.expm1(y_pred_log)
y_true     = np.expm1(y_te)

rmse = mean_squared_error(y_true, y_pred, squared=False)
r2   = r2_score(y_true, y_pred)
print(f"▶︎ Test  RMSE = {rmse:.2f}")
print(f"▶︎ Test  R²   = {r2:.3f}")

# ────────────────────────────────────────────────────────────
# 9) 可视化
# ────────────────────────────────────────────────────────────
plt.figure(figsize=(6,6))
plt.scatter(y_pred, y_true, alpha=0.5, s=20)
lims = [0, max(y_pred.max(), y_true.max())]
plt.plot(lims, lims, 'r--')
plt.xlabel("Predicted Loss")
plt.ylabel("Actual Loss")
plt.title(f"XGB Pred vs Actual — RMSE={rmse:.0f}, R²={r2:.3f}")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(8,5))
plot_importance(bst, ax=ax, importance_type="gain", max_num_features=15)
ax.set_title("Feature Importance (Gain)")
plt.tight_layout()
plt.show()