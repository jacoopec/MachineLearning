from scipy.stats import zscore
import numpy as np

z_scores = np.abs(zscore(df['column']))
df_clean = df[z_scores < 3]