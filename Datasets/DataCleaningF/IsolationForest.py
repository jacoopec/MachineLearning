from sklearn.ensemble import IsolationForest

iso = IsolationForest(contamination=0.05)
outliers = iso.fit_predict(df[['column']])
df_clean = df[outliers == 1]  # -1 are outliers
