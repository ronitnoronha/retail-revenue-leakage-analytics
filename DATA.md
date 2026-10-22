# About the data

## Source
| | |
|---|---|
| **Dataset** | Online Retail II |
| **Published by** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii) (University of California, Irvine) |
| **Creator / donor** | Daqing Chen · donated 20 September 2019 |
| **DOI** | [10.24432/C5CG6D](https://doi.org/10.24432/C5CG6D) |
| **Licence** | [Creative Commons Attribution 4.0 (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) – free to share and adapt with credit |
| **Citation** | Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D |

## The business
Every transaction of a **UK-based, registered, non-store online retailer** (it sells only online). The
company mainly sells **unique all-occasion giftware**, and **many of its customers are wholesalers** –
which is why some orders run to thousands of units. The full dataset covers 1 Dec 2009 – 9 Dec 2011
(1,067,371 rows) in an Excel file with one sheet per year.

## What this project uses
* **The "Year 2010-2011" sheet only:** 1 Dec 2010 – 9 Dec 2011, **541,910 rows**, saved as CSV.
* The raw file is **not stored in this repository** (44 MB). To reproduce the analysis:
  1. Download `online_retail_II.xlsx` from the UCI page above.
  2. Open the **Year 2010-2011** sheet and save it as CSV (Excel: *File → Save As → CSV*).
  3. Save it as `data/raw/online_retail_II_2010_2011.csv`, then run the scripts listed in the README.
* Read with `encoding="latin-1"` (the file contains £ signs in that encoding).

## Columns
| Column in our CSV | UCI name | Meaning |
|---|---|---|
| `Invoice` | InvoiceNo | Invoice number, 6 digits. Starts with **"C"** for a cancellation; "A" marks a bad-debt adjustment |
| `StockCode` | StockCode | Product code, usually 5 digits plus an optional letter (e.g. `85123A`). Non-product codes exist: `POST` postage, `M` manual, `AMAZONFEE`, `BANK CHARGES`, `D` discount, … |
| `Description` | Description | Product name (free text; some codes have several spellings) |
| `Quantity` | Quantity | Units on the line; negative on cancellations and stock write-offs |
| `InvoiceDate` | InvoiceDate | Date and time of the invoice (`M/D/YY H:MM` in our CSV) |
| `Price` | UnitPrice | Price per unit in **pounds sterling (£)** |
| `Customer ID` | CustomerID | Customer number, 5 digits. **Missing on about 25% of rows.** Anonymous – no names or contact details |
| `Country` | Country | Customer's country of residence (38 countries; ~90% of customers are in the UK) |

## Data quality issues found (handled in `src/01_clean.py`)
| Issue | Rows | How it was handled |
|---|---|---|
| Exact duplicate rows | 5,268 | Removed |
| Missing Customer ID | 135,080 | Kept in revenue totals, excluded from customer models; reported as a finding |
| Cancellation invoices ("C") | 9,288 | Kept in their own table (cancellation analysis) |
| Non-product lines (postage, fees, manual, bad debt, vouchers) | 2,945 | Separated from product sales |
| Stock write-offs (negative qty, price 0, notes like "damaged") | 1,336 | Separated (write-off analysis) |
| Zero-price sale lines | 1,156 | Removed |
| Keying errors (e.g. 80,995 units ordered and reversed 12 min later) | 6 | Removed both sides |
| Last month partial (data ends 9 Dec 2011) | – | Flagged; excluded from monthly trends |

Full details: [`docs/step1_cleaning.md`](docs/step1_cleaning.md).

## Privacy
Customers appear only as anonymous 5-digit IDs; the data contains no names, addresses or contact
details. Customer IDs shown in this project's app and dashboard are those anonymous IDs.
