import numpy as np
import pandas as pd

pth = r"F:\Je respire\トレーニング後\統合データ_標準化\merged_nagashio.xlsx"
INPUT_PATH = f"{pth}"
OUTPUT_PATH = f"{pth.replace('.xlsx', '_standardized.xlsx')}"

# このExcelは「1行目がタイトル、2行目がヘッダ」になっていたので header=1 を使う
HEADER_ROW = 1

# 標準化から除外したい列名があればここに入れる（例: ID列など）
EXCLUDE_COLS = ["Control Box No.32(Oxy)", "Mark", "Time", "PreScan", "HbOffset",
                "Control Box No.32(Deoxy)", "Mark.1", "Time.1", "PreScan.1", "HbOffset.1",
                "Control Box No.32(Total)", "Mark.2", "Time.2", "PreScan.2", "HbOffset.2",
                "TIME",
                "device_time_stamp", "system_time_stamp", "pc_time_stamp", "left_validity", "right_validity", "blink"]

def compute_global_stats(xls: pd.ExcelFile, header_row: int, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = []

    global_sum = {}
    global_sumsq = {}
    global_count = {}

    # 1st pass: 全シートを走査して sum / sumsq / count を集計
    for sh in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sh, header=header_row)

        # 全部NaNの列があれば落とす（末尾の空列対策）
        df = df.dropna(axis=1, how="all")

        # 数値列だけ対象（文字列列はそのまま保持）
        num_cols = df.select_dtypes(include="number").columns.tolist()
        num_cols = [c for c in num_cols if c not in exclude_cols]

        for c in num_cols:
            s = df[c]
            s_non_na = s.dropna()
            if s_non_na.empty:
                continue

            v = s_non_na.to_numpy(dtype=float)

            global_sum[c] = global_sum.get(c, 0.0) + float(np.sum(v))
            global_sumsq[c] = global_sumsq.get(c, 0.0) + float(np.sum(v * v))
            global_count[c] = global_count.get(c, 0) + int(v.size)

    # mean / std(ddof=0) を作る（全体を母集団として扱う）
    global_mean = {}
    global_std = {}

    for c, cnt in global_count.items():
        mu = global_sum[c] / cnt
        var = (global_sumsq[c] / cnt) - (mu * mu)
        var = max(var, 0.0)  # 数値誤差対策
        sd = float(np.sqrt(var))

        global_mean[c] = float(mu)
        global_std[c] = sd

    return global_mean, global_std

def standardize_workbook(input_path: str, output_path: str, header_row: int, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = []

    xls = pd.ExcelFile(input_path)

    global_mean, global_std = compute_global_stats(xls, header_row, exclude_cols=exclude_cols)

    # 2nd pass: 各シートを標準化して書き出し
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sh in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sh, header=header_row)
            df = df.dropna(axis=1, how="all")

            num_cols = df.select_dtypes(include="number").columns.tolist()
            num_cols = [c for c in num_cols if c not in exclude_cols]

            for c in num_cols:
                mu = global_mean.get(c, None)
                sd = global_std.get(c, None)

                # そもそも統計が作れなかった列はスキップ
                if mu is None or sd is None:
                    continue

                # 標準偏差0（全シート通して一定値）の列は 0 にする（NaN回避）
                if sd == 0.0:
                    df[c] = 0.0
                else:
                    df[c] = (df[c] - mu) / sd

            df.to_excel(writer, sheet_name=sh, index=False)

    return output_path

if __name__ == "__main__":
    out = standardize_workbook(
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
        header_row=HEADER_ROW,
        exclude_cols=EXCLUDE_COLS,
    )
    print(f"Saved: {out}")
