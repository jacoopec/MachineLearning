| Issue                                  | Description                                                   |
| -------------------------------------- | ------------------------------------------------------------- |
| **Missing values**                     | In `brand`, `efficiency_rating`, `price_usd`                  |
| **Duplicate rows**                     | 2 intentional duplicates added                                |
| **Inconsistent labels**                | Variations in casing and spacing (e.g., `"a++"`, `" A + "`)   |
| **Wrong data types**                   | `price_usd` has string values (e.g., `"$99.40"`, `"9999.99"`) |
| **Outliers**                           | Extreme values in `power_rating_watts`, `price_usd`           |
| **Messy text**                         | Emojis and emoticons in `product_name`                        |
| **Category casing issues**             | Values like `" appliance "` and `"MOTOR"`                     |
| **NaNs in numeric/categorical fields** | Several fields randomly blanked                               |




Imputation / Deletion for missing values

Removing duplicates

Standardizing categorical values

Fixing price format and converting to float

Detecting & removing outliers (e.g., IQR or Z-score)

Removing emojis & emoticons from text fields

Scaling numeric fields (e.g., MinMaxScaler)

Encoding categorical fields

Creating new features (like price_per_kg, usage_score, etc.)