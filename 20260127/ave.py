import pandas as pd

def compute_column_means_all_sheets_one_table(
    input_excel_path: str,
    output_excel_path: str,
    output_sheet_name: str = "sheetwise_means"
):
    # Excel全シートを読み込み
    xls = pd.ExcelFile(input_excel_path)

    mean_rows = []

    for sheet_name in xls.sheet_names:
        df = xls.parse(sheet_name)

        # 数値カラムのみ
        numeric_df = df.select_dtypes(include="number")

        # 平均（Series: index=column）
        mean_series = numeric_df.mean()

        # シート名を行ラベルとして付与
        mean_series.name = sheet_name

        mean_rows.append(mean_series)

    # シート × カラム の表を作成
    mean_table = pd.DataFrame(mean_rows)

    # Excelに出力（1シートのみ）
    with pd.ExcelWriter(output_excel_path, engine="openpyxl") as writer:
        mean_table.to_excel(writer, sheet_name=output_sheet_name)

    print("✅ すべてのシートの平均値を1枚のシートにまとめました")
    print(f"出力ファイル: {output_excel_path}")


if __name__ == "__main__":
    input_excel = r"F:\Je respire\トレーニング後\統合データ_標準化\merged_nagashio_standardized.xlsx"
    output_excel = r"F:\Je respire\トレーニング後\統合データ_標準化\merged_nagashio_standardized_sheetwise_column_means.xlsx"

    compute_column_means_all_sheets_one_table(
        input_excel_path=input_excel,
        output_excel_path=output_excel
    )
