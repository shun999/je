import shlex
import re
from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd


# -----------------------------
# Time parsing -> Timedelta (relative)
# -----------------------------
def parse_time_series(s: pd.Series) -> pd.Series:
    """
    Robust time parser:
      - numeric seconds
      - HH:MM:SS(.f)
      - MM:SS(.f)  <-- CSVがこれ
      - handles weird spaces / fullwidth colon
    Returns Timedelta since first valid timestamp.
    """
    s0 = s.astype(str)

    # normalize weird spaces / fullwidth colon
    s0 = (s0
          .str.replace("\u00a0", " ", regex=False)  # NBSP
          .str.replace("：", ":", regex=False)
          .str.strip())

    # numeric seconds?
    numeric = pd.to_numeric(s0, errors="coerce")
    if numeric.notna().mean() > 0.8:
        return pd.to_timedelta(numeric, unit="s")

    # --- MM:SS(.f) detection (e.g., 26:26.0) ---
    mmss = s0.str.match(r"^\d{1,3}:\d{2}(\.\d+)?$")
    if mmss.mean() > 0.8:
        def _mmss_to_seconds(x: str):
            try:
                m, sec = x.split(":")
                return float(m) * 60.0 + float(sec)
            except Exception:
                return None
        secs = s0.map(_mmss_to_seconds)
        td = pd.to_timedelta(secs, unit="s")
        # relative to first valid
        first = td.dropna().iloc[0] if td.notna().any() else pd.Timedelta(0)
        return td - first

    # --- HH:MM:SS(.f) ---
    dt = pd.to_datetime(s0, errors="coerce", format="%H:%M:%S.%f")
    if dt.notna().mean() < 0.5:
        dt = pd.to_datetime(s0, errors="coerce", format="%H:%M:%S")

    # fallback general parser
    if dt.notna().mean() < 0.5:
        dt = pd.to_datetime(s0, errors="coerce")

    if not dt.notna().any():
        return pd.to_timedelta(pd.Series([pd.NA] * len(s0)))

    first = dt.dropna().iloc[0]
    return pd.to_timedelta(dt - first)



# -----------------------------
# CSV reader that can skip metadata header
# -----------------------------
def read_csv_with_dynamic_header(path: Path) -> pd.DataFrame:
    header_i = None
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if re.search(r"\bMark\b", line) and re.search(r"\bTime\b", line) and line.count(",") >= 3:
                header_i = i
                break

    if header_i is None:
        return pd.read_csv(path)
    return pd.read_csv(path, skiprows=header_i)


# -----------------------------
# Marker handling
# -----------------------------
def add_sequential_marker_on_mark_change(df: pd.DataFrame, mark_col="Mark", out_col="Marker") -> pd.DataFrame:
    """
    Markの値が変化した行ごとに、出現順で Marker:0, Marker:1, ... を付与する
    （Mark値そのものはMarker番号に使わない）
    """
    if mark_col not in df.columns:
        raise ValueError(f"'{mark_col}'列が見つかりません: columns={list(df.columns)}")

    out = df.copy()
    mark = pd.to_numeric(out[mark_col], errors="coerce")
    prev = mark.shift(1)
    is_change = (mark.notna()) & (prev.notna()) & (mark != prev)

    out[out_col] = pd.NA
    change_idx = out.index[is_change].to_list()
    for k, idx in enumerate(change_idx):
        out.at[idx, out_col] = f"Marker:{k}"
    return out


def detect_marker_column(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if df[c].astype(str).str.contains(r"Marker:\s*\d+", na=False).any():
            return c
    return None


def reindex_marker_by_appearance(df: pd.DataFrame, marker_col: str, start: int = 0) -> pd.DataFrame:
    """
    Markerがある行を出現順に Marker:start, start+1, ... に再採番
    """
    out = df.copy()
    s = out[marker_col].astype(str)
    mask = s.str.contains(r"Marker:\s*\d+", na=False)
    idxs = out.index[mask].to_list()
    for k, idx in enumerate(idxs):
        out.at[idx, marker_col] = f"Marker:{start + k}"
    return out


def ensure_marker_csv(df: pd.DataFrame) -> pd.DataFrame:
    # CSVはMark変化点に sequential marker
    df = add_sequential_marker_on_mark_change(df, mark_col="Mark", out_col="Marker")
    return df


def ensure_marker_txt(df: pd.DataFrame) -> pd.DataFrame:
    """
    TXTは Events 列に Marker: が入っている想定。
    Events -> Marker にコピーし、出現順で Marker:0,1,2... に再採番。
    """
    if "Events" not in df.columns:
        raise ValueError(f"TXTに 'Events' 列が見つかりません: columns={list(df.columns)}")

    out = df.copy()
    out["Marker"] = out["Events"]

    # 出現順で Marker:0,1,2...
    out = reindex_marker_by_appearance(out, "Marker", start=0)
    return out



def extract_marker_rows(df: pd.DataFrame, marker_col: str = "Marker") -> pd.DataFrame:
    m = df[marker_col].astype(str).str.extract(r"Marker:\s*(\d+)")
    mask = m[0].notna()
    ev = df.loc[mask, [marker_col]].copy()
    ev["marker_num"] = m.loc[mask, 0].astype(int).values
    ev["row_index"] = ev.index.values
    return ev.sort_values("row_index")


def compute_marker_intervals(df: pd.DataFrame, time_col: str, marker_col: str = "Marker") -> pd.DataFrame:
    if time_col not in df.columns:
        raise ValueError(f"'{time_col}'列が見つかりません: columns={list(df.columns)}")

    t = parse_time_series(df[time_col])
    tmp = df.copy()
    tmp["_t"] = t

    ev = extract_marker_rows(tmp, marker_col)
    if ev.empty:
        return pd.DataFrame(columns=["segment", "from", "to", "duration_s"])

    ev["t"] = tmp.loc[ev["row_index"], "_t"].values

    segments = []
    start_t = tmp["_t"].dropna().iloc[0] if tmp["_t"].notna().any() else pd.Timedelta(0)

    prev_t = start_t
    prev_label = "START"

    for _, row in ev.iterrows():
        cur_label = f"Marker:{int(row['marker_num'])}"
        cur_t = row["t"]
        dur = (cur_t - prev_t).total_seconds() if pd.notna(cur_t) and pd.notna(prev_t) else pd.NA
        segments.append({
            "segment": len(segments),   # 0,1,2,... (sheet name と一致)
            "from": prev_label,
            "to": cur_label,
            "duration_s": dur
        })
        prev_t = cur_t
        prev_label = cur_label

    return pd.DataFrame(segments)


def delete_markers_and_reindex(df: pd.DataFrame, delete_nums: List[int], marker_col: str = "Marker") -> pd.DataFrame:
    out = df.copy()
    s = out[marker_col].astype(str)
    m = s.str.extract(r"Marker:\s*(\d+)")
    num = pd.to_numeric(m[0], errors="coerce")
    out.loc[num.isin(delete_nums), marker_col] = pd.NA
    out = reindex_marker_by_appearance(out, marker_col, start=0)
    return out


def split_into_segments(df: pd.DataFrame, marker_col: str = "Marker") -> List[pd.DataFrame]:
    """
    segment 0: START〜Marker:0直前
    segment 1: Marker:0〜Marker:1直前
    segment 2: Marker:1〜Marker:2直前 ...
    """
    ev = extract_marker_rows(df, marker_col)
    if ev.empty:
        return [df.copy()]

    idx = df.index.to_list()
    marker_rows = ev["row_index"].to_list()

    segs = []
    # seg0
    first = marker_rows[0]
    pos_first = idx.index(first)
    segs.append(df.iloc[:pos_first].copy())

    # seg1..K
    for i, start in enumerate(marker_rows):
        end = marker_rows[i + 1] if i + 1 < len(marker_rows) else None
        start_pos = idx.index(start)
        if end is None:
            segs.append(df.iloc[start_pos:].copy())
        else:
            end_pos = idx.index(end)
            segs.append(df.iloc[start_pos:end_pos].copy())

    return segs


# -----------------------------
# TXT reading
# -----------------------------
def read_txt_table(path: Path) -> pd.DataFrame:
    """
    あなたのTXT形式（冒頭にメタ情報、途中に 'TIME ... Events ...' ヘッダ、以降タブ区切り）専用。
    - 'TIME' を含むヘッダ行を探して、そこから tab 区切りで読む
    """
    header_i = None
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            # ヘッダ行: "TIME" と "Events" が含まれる
            if "\t" in line and re.search(r"\bTIME\b", line) and re.search(r"\bEvents\b", line):
                header_i = i
                break

    if header_i is None:
        raise ValueError("TXT内でヘッダ行（TIME と Events を含む行）が見つかりませんでした。")

    df = pd.read_csv(path, sep="\t", skiprows=header_i, engine="python")
    return df


def parse_txt_as_log(path: Path) -> pd.DataFrame:
    """
    行ごとに列数がバラバラなTXT（ログ）から TIME と Marker を抽出する。
    - TIME は 'TIME=...' や 'TIME: ...' や 先頭の時刻っぽい表記にも対応するように頑張る
    - Marker は 'Marker:数字' を抽出
    """
    time_patterns = [
        re.compile(r"\bTIME\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\b", re.IGNORECASE),   # TIME=12.34
        re.compile(r"\bTIME\s*[:=]\s*([0-9]{1,2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?)\b", re.IGNORECASE),  # TIME=HH:MM:SS(.ms)
        re.compile(r"^([0-9]{1,2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?)\b"),          # 行頭 HH:MM:SS(.ms)
        re.compile(r"^([0-9]+(?:\.[0-9]+)?)\b"),                                # 行頭 秒
    ]
    marker_pat = re.compile(r"(Marker:\s*\d+)", re.IGNORECASE)

    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            marker_m = marker_pat.search(line)
            if not marker_m:
                continue  # Markerが無い行は基本スキップ（必要なら残してもOK）

            marker = marker_m.group(1).replace(" ", "")  # "Marker: 3" -> "Marker:3"

            time_val = None
            for tp in time_patterns:
                tm = tp.search(line)
                if tm:
                    time_val = tm.group(1)
                    break

            # TIMEが取れない行も一応残す（後で前方埋めする）
            rows.append({"LINE": i, "RAW": line, "TIME": time_val, "Marker": marker})

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("TXTから Marker 行が抽出できませんでした。Marker表記がある行を確認してください。")

    # TIMEを前方埋め（Marker行にTIMEが無いログ形式のため）
    df["TIME"] = df["TIME"].ffill()

    return df


# -----------------------------
# Excel writer: each segment -> sheet "0","1",...
# put each dataset side-by-side
# -----------------------------
def write_segments_to_excel(out_path: Path, blocks: List[Tuple[str, List[pd.DataFrame]]]):
    max_seg = max(len(segs) for _, segs in blocks)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for seg_i in range(max_seg):
            sheet = str(seg_i)
            col = 0
            for name, segs in blocks:
                df_seg = segs[seg_i] if seg_i < len(segs) else pd.DataFrame()
                pd.DataFrame([[name]]).to_excel(writer, sheet_name=sheet, index=False, header=False, startrow=0, startcol=col)
                df_seg.to_excel(writer, sheet_name=sheet, index=False, startrow=1, startcol=col)
                col += (df_seg.shape[1] if df_seg.shape[1] > 0 else 1) + 2


# -----------------------------
# Interactive main
# -----------------------------
def main():
    print("=== Interactive Marker preprocessing & merge ===")

    csv_paths_str = input("CSV paths (space separated): ").strip()
    txt_path_str = input("TXT path: ").strip()
    out_xlsx_str = input("Output Excel path (.xlsx): ").strip()

    csv_paths = [Path(p) for p in shlex.split(csv_paths_str)]
    txt_path = Path(txt_path_str.strip().strip('"'))
    out_path = Path(out_xlsx_str.strip().strip('"'))

    # load + marker creation
    csv_dfs = []
    for p in csv_paths:
        df = read_csv_with_dynamic_header(p)
        df = ensure_marker_csv(df)
        csv_dfs.append((p, df))

    txt_df = read_txt_table(txt_path)
    txt_df = ensure_marker_txt(txt_df)

    # show intervals first
    print("\n--- Marker intervals (seconds) BEFORE deletion ---")
    for p, df in csv_dfs:
        itv = compute_marker_intervals(df, time_col="Time", marker_col="Marker")
        print(f"\n[CSV] {p.name}")
        print(itv.to_string(index=False))

    if "TIME" not in txt_df.columns:
        raise ValueError(f"TXTに 'TIME' 列が見つかりません: columns={list(txt_df.columns)}")
    itv_txt = compute_marker_intervals(txt_df, time_col="TIME", marker_col="Marker")
    print(f"\n[TXT] {txt_path.name}")
    print(itv_txt.to_string(index=False))

    # user chooses markers to delete
    # user chooses markers to delete (CSV / TXT / BOTH)
mode = input("\nDelete markers from which source? [1]=CSV [2]=TXT [3]=BOTH  (default=3): ").strip()
if mode not in {"1", "2", "3", ""}:
    mode = "3"
if mode == "":
    mode = "3"

s_csv = input("Delete marker numbers for CSV (e.g. 2 5) | empty = none : ").strip()
del_csv = [int(x) for x in s_csv.split()] if s_csv else []

s_txt = input("Delete marker numbers for TXT (e.g. 2 5) | empty = none : ").strip()
del_txt = [int(x) for x in s_txt.split()] if s_txt else []

# apply deletion + reindex separately
if mode in {"1", "3"} and del_csv:
    csv_dfs2 = []
    for p, df in csv_dfs:
        df2 = delete_markers_and_reindex(df, del_csv, marker_col="Marker")
        csv_dfs2.append((p, df2))
    csv_dfs = csv_dfs2

if mode in {"2", "3"} and del_txt:
    txt_df = delete_markers_and_reindex(txt_df, del_txt, marker_col="Marker")

# show intervals again after deletion
if (mode in {"1", "3"} and del_csv) or (mode in {"2", "3"} and del_txt):
    print("\n--- Marker intervals (seconds) AFTER deletion/reindex ---")
    for p, df in csv_dfs:
        itv = compute_marker_intervals(df, time_col="Time", marker_col="Marker")
        print(f"\n[CSV] {p.name}")
        print(itv.to_string(index=False))

    itv_txt2 = compute_marker_intervals(txt_df, time_col="TIME", marker_col="Marker")
    print(f"\n[TXT] {txt_path.name}")
    print(itv_txt2.to_string(index=False))

    # warn if counts differ
    def _count_events(d):
        return int(d["Marker"].astype(str).str.contains(r"Marker:\s*\d+", na=False).sum())

    csv_counts = [(_count_events(df), p.name) for p, df in csv_dfs]
    txt_count = _count_events(txt_df)

    print("\n--- Marker event counts AFTER deletion ---")
    for c, name in csv_counts:
        print(f"[CSV] {name}: {c}")
    print(f"[TXT] {txt_path.name}: {txt_count}")

    if any(c != txt_count for c, _ in csv_counts):
        print("\n[WARN] CSVとTXTでMarker数が一致していません。Excel統合時に区間がズレて空白が出る可能性があります。")

        csv_dfs2 = []
        for p, df in csv_dfs:
            df2 = delete_markers_and_reindex(df, delete_nums, marker_col="Marker")
            csv_dfs2.append((p, df2))
        csv_dfs = csv_dfs2
        txt_df = delete_markers_and_reindex(txt_df, delete_nums, marker_col="Marker")

        print("\n--- Marker intervals (seconds) AFTER deletion/reindex ---")
        for p, df in csv_dfs:
            itv = compute_marker_intervals(df, time_col="Time", marker_col="Marker")
            print(f"\n[CSV] {p.name}")
            print(itv.to_string(index=False))

        itv_txt2 = compute_marker_intervals(txt_df, time_col="TIME", marker_col="Marker")
        print(f"\n[TXT] {txt_path.name}")
        print(itv_txt2.to_string(index=False))

    # split into segments and write excel
    blocks = []
    for p, df in csv_dfs:
        blocks.append((f"CSV:{p.stem}", split_into_segments(df, marker_col="Marker")))
    blocks.append(("TXT", split_into_segments(txt_df, marker_col="Marker")))

    write_segments_to_excel(out_path, blocks)
    print(f"\n[OK] Wrote Excel: {out_path.resolve()}")


if __name__ == "__main__":
    main()
