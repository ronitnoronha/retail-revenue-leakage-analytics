"""
Step 1 - Data cleaning.

Takes the raw Online Retail II (2010-2011) export and splits it into clean,
purpose-specific tables. Nothing is silently deleted: every row ends up either
in an output table or in the cleaning log with the reason it was set aside.

Outputs (data/processed/):
  sales.parquet          real product sales lines (positive qty, positive price)
  cancellations.parquet  product cancellation / return lines (invoice starts with "C")
  non_product.parquet    postage, fees, manual adjustments, bad-debt entries
  stock_writeoffs.parquet  internal stock adjustments (damaged, lost, etc.)
  cleaning_log.csv       row counts at each rule
"""
import pandas as pd

RAW = "data/raw/online_retail_II_2010_2011.csv"
OUT = "data/processed/"

# Codes that are not physical products: postage, fees, manual corrections, vouchers.
NON_PRODUCT_CODES = {
    "POST", "DOT", "C2", "M", "m", "D", "S", "BANK CHARGES", "AMAZONFEE",
    "CRUK", "B", "PADS",
}

log = []
def record(step, rows, note):
    log.append({"step": step, "rows": rows, "note": note})
    print(f"{step:<38} {rows:>8,}  {note}")


# ---------------------------------------------------------------- 1. Load
# Invoice and StockCode are read as text: they contain letters ("C536379", "85123A")
# and leading zeros must not be lost. latin-1 because the file has £ signs in that encoding.
df = pd.read_csv(RAW, encoding="latin-1", dtype={"Invoice": str, "StockCode": str})
record("1 raw rows", len(df), "as exported")

# ---------------------------------------------------------------- 2. Types and text
df = df.rename(columns={"Customer ID": "CustomerID"})
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="%m/%d/%y %H:%M")
# Customer IDs were stored as floats (17850.0) because of missing values; make them clean text.
df["CustomerID"] = df["CustomerID"].astype("Int64").astype(str).replace("<NA>", pd.NA)
df["Description"] = (df["Description"].str.strip()
                     .str.replace("Â£", "£", regex=False))  # fix mis-encoded pound sign
df["StockCode"] = df["StockCode"].str.strip().str.upper()
df["Revenue"] = df["Quantity"] * df["Price"]

# ---------------------------------------------------------------- 3. Exact duplicates
# Same invoice, product, quantity, price, timestamp and customer on two rows:
# the till/system logged the line twice. Keep the first copy.
n_before = len(df)
df = df.drop_duplicates(subset=["Invoice", "StockCode", "Description", "Quantity",
                                "InvoiceDate", "Price", "CustomerID", "Country"])
record("3 drop exact duplicates", len(df), f"removed {n_before - len(df):,}")

# ---------------------------------------------------------------- 4. Split by line type
is_cancel = df["Invoice"].str.startswith("C")
is_adjust_invoice = df["Invoice"].str.startswith("A")          # "Adjust bad debt" entries
is_non_product = df["StockCode"].isin(NON_PRODUCT_CODES) | df["StockCode"].str.startswith("GIFT_") | is_adjust_invoice
# Negative quantity, zero price, not a "C" invoice, no customer: warehouse write-offs
# ("damaged", "thrown away", "check"). These are stock movements, not customer activity.
is_writeoff = ~is_cancel & ~is_non_product & (df["Quantity"] < 0)

non_product = df[is_non_product].copy()
writeoffs = df[is_writeoff].copy()
cancels = df[is_cancel & ~is_non_product].copy()
rest = df[~is_cancel & ~is_non_product & ~is_writeoff].copy()
record("4a non-product lines set aside", len(non_product), "postage, fees, manual, bad debt, vouchers")
record("4b stock write-offs set aside", len(writeoffs), "negative qty, price 0, internal")
record("4c product cancellation lines", len(cancels), "invoice starts with C")
record("4d remaining sale lines", len(rest), "before price checks")

# ---------------------------------------------------------------- 5. Zero-price sale lines
# Price 0 lines are almost all missing a customer and a description: free samples,
# stock checks or system placeholders. They carry no revenue, so drop them.
zero = rest["Price"] <= 0
record("5 drop zero-price sale lines", int((~zero).sum()), f"removed {int(zero.sum()):,}")
sales = rest[~zero].copy()

# ---------------------------------------------------------------- 6. Keying errors
# A purchase immediately reversed by a cancellation of the identical quantity for the
# same customer and product (e.g. 80,995 paper birdies, cancelled 12 minutes later)
# is a data-entry mistake, not a real return. Left in, two such rows alone would
# add ~£245k of fake "cancellations". Flag pairs reversed within 24 hours with qty >= 1,000.
big_c = cancels[cancels["Quantity"] <= -1000]
pairs = big_c.merge(sales, on=["CustomerID", "StockCode", "Price"], suffixes=("_c", "_s"))
pairs = pairs[(pairs["Quantity_s"] == -pairs["Quantity_c"])
              & (pairs["InvoiceDate_c"] >= pairs["InvoiceDate_s"])
              & (pairs["InvoiceDate_c"] - pairs["InvoiceDate_s"] <= pd.Timedelta("24h"))]
err_sale_idx = sales.index[sales.set_index(["Invoice", "StockCode"]).index.isin(
    list(zip(pairs["Invoice_s"], pairs["StockCode"])))]
err_cancel_idx = cancels.index[cancels.set_index(["Invoice", "StockCode"]).index.isin(
    list(zip(pairs["Invoice_c"], pairs["StockCode"])))]
sales = sales.drop(err_sale_idx)
cancels = cancels.drop(err_cancel_idx)
record("6 drop keying-error pairs (sales)", len(sales), f"removed {len(err_sale_idx)} sale lines")
record("6 drop keying-error pairs (cancels)", len(cancels), f"removed {len(err_cancel_idx)} cancel lines")

# ---------------------------------------------------------------- 7. One description per product
# 650 stock codes have several spellings ("DAMAGED", "damages", case changes). Use each
# code's most common description so product-level counts aren't split.
canon = (pd.concat([sales, cancels]).dropna(subset=["Description"])
         .groupby("StockCode")["Description"].agg(lambda s: s.value_counts().index[0]))
for t in (sales, cancels):
    t["Description"] = t["StockCode"].map(canon).fillna(t["Description"])

# ---------------------------------------------------------------- 8. Helper columns
for t in (sales, cancels):
    t["HasCustomer"] = t["CustomerID"].notna()
    t["Month"] = t["InvoiceDate"].dt.to_period("M").astype(str)
# December 2011 only runs to the 9th; analyses comparing months must treat it as partial.
sales["PartialMonth"] = sales["Month"] == "2011-12"

# ---------------------------------------------------------------- 9. Save
sales.to_parquet(OUT + "sales.parquet", index=False)
cancels.to_parquet(OUT + "cancellations.parquet", index=False)
non_product.to_parquet(OUT + "non_product.parquet", index=False)
writeoffs.to_parquet(OUT + "stock_writeoffs.parquet", index=False)
pd.DataFrame(log).to_csv(OUT + "cleaning_log.csv", index=False)

print("\nSummary")
print(f"  product sales      £{sales['Revenue'].sum():>12,.0f}  ({len(sales):,} lines, "
      f"{sales['Invoice'].nunique():,} invoices)")
print(f"  product cancels    £{-cancels['Revenue'].sum():>12,.0f}  ({len(cancels):,} lines, "
      f"{cancels['Invoice'].nunique():,} invoices)  = "
      f"{-cancels['Revenue'].sum() / sales['Revenue'].sum():.1%} of sales")
print(f"  sales w/o customer {(~sales['HasCustomer']).mean():.1%} of lines, "
      f"{sales.loc[~sales['HasCustomer'], 'Revenue'].sum() / sales['Revenue'].sum():.1%} of revenue")
print(f"  customers          {sales['CustomerID'].nunique():,}   countries {sales['Country'].nunique()}")
print(f"  date range         {sales['InvoiceDate'].min()}  ->  {sales['InvoiceDate'].max()}")
