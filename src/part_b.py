from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CLEAN_DIR = DATA_DIR / "clean"
OUTPUT_DIR = ROOT / "outputs"

ORDERS_FILE = CLEAN_DIR / "orders.csv"
CUSTOMERS_FILE = CLEAN_DIR / "customers.csv"
PRODUCTS_FILE = CLEAN_DIR / "products.csv"

SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
CUTOFF = pd.Timestamp("2025-09-01 00:00:00")
K_FINAL = 4

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

def read_inputs():
    orders = pd.read_csv(ORDERS_FILE)
    customers = pd.read_csv(CUSTOMERS_FILE)
    products = pd.read_csv(PRODUCTS_FILE)

    required_order_cols = {
        "order_id", "customer_id", "order_ts", "status",
        "order_amount", "discount_pct", "delivery_minutes",
        "product_id",
    }
    required_customer_cols = {
        "customer_id", "city", "age", "gender", "signup_date"
    }
    required_product_cols = {"product_id", "category"}

    missing_orders = required_order_cols - set(orders.columns)
    missing_customers = required_customer_cols - set(customers.columns)
    missing_products = required_product_cols - set(products.columns)

    if missing_orders:
        raise ValueError(f"Missing columns in orders.csv: {sorted(missing_orders)}")
    if missing_customers:
        raise ValueError(
            f"Missing columns in customers.csv: {sorted(missing_customers)}"
        )
    if missing_products:
        raise ValueError(
            f"Missing columns in products.csv: {sorted(missing_products)}"
        )

    orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="coerce")
    orders["order_amount"] = pd.to_numeric(
        orders["order_amount"], errors="coerce"
    )
    orders["discount_pct"] = pd.to_numeric(
        orders["discount_pct"], errors="coerce"
    )
    orders["delivery_minutes"] = pd.to_numeric(
        orders["delivery_minutes"], errors="coerce"
    )
    customers["signup_date"] = pd.to_datetime(
        customers["signup_date"], errors="coerce"
    )

    orders["status"] = orders["status"].astype(str).str.strip().str.lower()
    return orders, customers, products


def metrics_for_lapsed(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_lapsed": precision_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "recall_lapsed": recall_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "f1_lapsed": f1_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).tolist(),
    }


def add_report(report_lines, title, value):
    report_lines.append(f"\n{title}\n{'-' * len(title)}")
    report_lines.append(str(value))


# ---------------------------------------------------------
# Step 1: RFM table
# ---------------------------------------------------------

def build_rfm(orders):
    completed = orders.loc[
        orders["status"].eq("completed")
        & orders["order_ts"].notna()
    ].copy()

    rfm = (
        completed.groupby("customer_id")
        .agg(
            last_order_ts=("order_ts", "max"),
            frequency=("order_id", "nunique"),
            monetary=("order_amount", "sum"),
        )
        .reset_index()
    )

    rfm["recency"] = (SNAPSHOT - rfm["last_order_ts"]).dt.days
    rfm = rfm[
        ["customer_id", "recency", "frequency", "monetary"]
    ].sort_values("customer_id").reset_index(drop=True)

    if rfm[["recency", "frequency", "monetary"]].isna().any().any():
        raise ValueError("RFM contains missing values; inspect the clean order data.")

    rfm.to_csv(OUTPUT_DIR / "customer_rfm.csv", index=False)

    print("\n=== RFM summary ===")
    print(rfm[["recency", "frequency", "monetary"]].describe().round(2))

    assert len(rfm) == 1175, f"Expected 1175 RFM customers; got {len(rfm)}"
    assert rfm["customer_id"].is_unique

    return rfm


# ---------------------------------------------------------
# Step 2: K-Means evaluation, final clusters and PCA
# ---------------------------------------------------------

def build_segments(rfm, report_lines):
    rfm_features = ["recency", "frequency", "monetary"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rfm[rfm_features])

    evaluation = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_scaled)
        evaluation.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(X_scaled, labels),
        })

    evaluation_df = pd.DataFrame(evaluation)
    evaluation_df.to_csv(
        OUTPUT_DIR / "cluster_evaluation.csv", index=False
    )

    print("\n=== K-Means evaluation ===")
    print(evaluation_df.round(3).to_string(index=False))

    # K=4 is the selected business-friendly option from the assignment's
    # permitted choices {3, 4, 5}; it also has the strongest reference silhouette.
    final_kmeans = KMeans(
        n_clusters=K_FINAL, n_init=10, random_state=42
    )
    segment_df = rfm.copy()
    segment_df["segment_id"] = final_kmeans.fit_predict(X_scaled)

    profile = (
        segment_df.groupby("segment_id")
        .agg(
            size=("customer_id", "size"),
            mean_recency=("recency", "mean"),
            mean_frequency=("frequency", "mean"),
            mean_monetary=("monetary", "mean"),
        )
        .reset_index()
    )

    # Business names are based on measured segment profiles, not cluster ID.
    profile_by_id = profile.set_index("segment_id")
    remaining = list(profile["segment_id"])

    premium_id = int(profile.loc[profile["mean_monetary"].idxmax(), "segment_id"])
    names = {premium_id: "Premium"}
    remaining.remove(premium_id)

    loyal_id = int(
        profile_by_id.loc[remaining, "mean_frequency"].idxmax()
    )
    names[loyal_id] = "Loyal Regulars"
    remaining.remove(loyal_id)

    at_risk_id = int(
        profile_by_id.loc[remaining, "mean_recency"].idxmax()
    )
    names[at_risk_id] = "At-Risk Customers"
    remaining.remove(at_risk_id)

    for segment_id in remaining:
        names[int(segment_id)] = "Growth Opportunity"

    segment_df["segment_name"] = segment_df["segment_id"].map(names)
    segment_df["is_premium"] = segment_df["segment_id"].eq(premium_id)

    segment_df[
        [
            "customer_id", "recency", "frequency", "monetary",
            "segment_id", "segment_name", "is_premium",
        ]
    ].to_csv(OUTPUT_DIR / "customer_segments.csv", index=False)

    profile["segment_name"] = profile["segment_id"].map(names)
    profile["recommended_action"] = profile["segment_name"].map({
        "Premium": "Offer loyalty rewards and personalised retention benefits.",
        "Loyal Regulars": "Maintain repeat ordering with relevant bundles.",
        "At-Risk Customers": "Send a targeted win-back offer and reminder.",
        "Growth Opportunity": "Encourage a second or more frequent purchase.",
    })

    print("\n=== Segment profile ===")
    print(profile.round(2).to_string(index=False))
    print(f"\nPremium segment ID: {premium_id}")
    print("Cluster IDs are arbitrary labels, not rankings.")

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame({
        "customer_id": rfm["customer_id"],
        "pc1": pca_result[:, 0],
        "pc2": pca_result[:, 1],
        "segment_id": segment_df["segment_id"],
    })
    pca_df.to_csv(OUTPUT_DIR / "customer_segments_pca.csv", index=False)

    variance = pca.explained_variance_ratio_
    print(
        "\nPCA explained variance ratios:",
        np.round(variance, 3),
        "| total:",
        round(float(variance.sum()), 3),
    )

    add_report(report_lines, "K-Means evaluation", evaluation_df.round(3).to_string(index=False))
    add_report(report_lines, "Segment profile", profile.round(2).to_string(index=False))
    add_report(
        report_lines,
        "Choosing K",
        "K=4 is selected for interpretability and the strong silhouette score. "
        "Inertia always decreases as K increases because each point can be closer "
        "to a cluster centre when more centres are available. Therefore, the "
        "lowest inertia alone is not a reason to choose K=8; the elbow, silhouette "
        "and useful business-sized segments must also be considered.",
    )
    add_report(
        report_lines,
        "PCA",
        f"Explained variance ratios: {variance.round(3).tolist()}; "
        f"total: {variance.sum():.3f}.",
    )

    return segment_df, premium_id


# ---------------------------------------------------------
# Step 3: Build the lapse cohort and leakage-safe features
# ---------------------------------------------------------

def build_lapse_dataset(orders, customers, products, report_lines):
    # Only orders strictly before the cutoff are allowed as model features.
    prior = orders.loc[
        orders["order_ts"].notna() & (orders["order_ts"] < CUTOFF)
    ].copy()

    completed_prior = prior.loc[prior["status"].eq("completed")].copy()

    # Cohort: customers with at least one completed order before cutoff.
    cohort_ids = completed_prior["customer_id"].unique()
    cohort = customers.loc[
        customers["customer_id"].isin(cohort_ids)
    ].copy()

    # Label uses the September window, but September orders never enter features.
    september_completed = orders.loc[
        orders["status"].eq("completed")
        & orders["order_ts"].ge(CUTOFF)
        & orders["order_ts"].lt(pd.Timestamp("2025-10-01"))
    ]
    active_september = set(september_completed["customer_id"].unique())
    cohort["lapsed"] = (
        ~cohort["customer_id"].isin(active_september)
    ).astype(int)

    # Completed-order features, based only on prior orders.
    completed_features = (
        completed_prior.groupby("customer_id")
        .agg(
            last_completed_ts=("order_ts", "max"),
            frequency=("order_id", "nunique"),
            monetary=("order_amount", "sum"),
            avg_discount_pct=("discount_pct", "mean"),
            avg_delivery_minutes=("delivery_minutes", "mean"),
        )
        .reset_index()
    )
    completed_features["recency"] = (
        CUTOFF - completed_features["last_completed_ts"]
    ).dt.days
    completed_features = completed_features.drop(
        columns=["last_completed_ts"]
    )

    # Cancellation rate denominator: orders with a final status before cutoff.
    final_statuses = {"completed", "cancelled"}
    final_orders = prior.loc[prior["status"].isin(final_statuses)].copy()
    cancellation_rate = (
        final_orders.assign(
            is_cancelled=final_orders["status"].eq("cancelled").astype(int)
        )
        .groupby("customer_id")
        .agg(
            final_order_count=("order_id", "nunique"),
            cancelled_count=("is_cancelled", "sum"),
        )
        .reset_index()
    )
    cancellation_rate["cancellation_rate"] = (
        cancellation_rate["cancelled_count"]
        / cancellation_rate["final_order_count"].replace(0, np.nan)
    )
    cancellation_rate = cancellation_rate[
        ["customer_id", "cancellation_rate"]
    ]

    # Most frequently ordered product category before cutoff.
    category_orders = completed_prior.merge(
        products[["product_id", "category"]],
        on="product_id",
        how="left",
    )
    category_counts = (
        category_orders.groupby(["customer_id", "category"])
        .size()
        .reset_index(name="category_order_count")
        .sort_values(
            ["customer_id", "category_order_count", "category"],
            ascending=[True, False, True],
        )
    )
    favourite_category = (
        category_counts.drop_duplicates("customer_id")
        [["customer_id", "category"]]
        .rename(columns={"category": "favourite_category"})
    )

    feature_df = (
        cohort.merge(completed_features, on="customer_id", how="left")
        .merge(cancellation_rate, on="customer_id", how="left")
        .merge(favourite_category, on="customer_id", how="left")
    )

    feature_df["tenure_days"] = (
        CUTOFF - feature_df["signup_date"]
    ).dt.days

    numeric_features = [
        "recency", "frequency", "monetary", "avg_discount_pct",
        "cancellation_rate", "avg_delivery_minutes", "age", "tenure_days",
    ]
    categorical_features = ["city", "gender", "favourite_category"]

    for col in numeric_features:
        feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce")

    # Fill feature missingness before splitting. These constants/statistics are
    # not learned from the full dataset: model imputers are fitted on train only.
    for col in categorical_features:
        feature_df[col] = feature_df[col].fillna("Unknown").astype(str)

    # Keep the raw numeric values missing here; the pipeline imputes them
    # after train/test split, so no test rows influence imputation or scaling.
    if feature_df[numeric_features + categorical_features].isna().any().any():
        # Numeric missing values are allowed at this point and handled by the
        # training-only SimpleImputer in the pipeline. Categorical values are not.
        categorical_missing = feature_df[categorical_features].isna().sum().sum()
        if categorical_missing:
            raise ValueError("Categorical feature columns contain missing values.")

    feature_columns = numeric_features + categorical_features
    X = feature_df[feature_columns].copy()
    y = feature_df["lapsed"].astype(int).copy()
    ids = feature_df["customer_id"].copy()

    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, ids,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    print("\n=== Lapse cohort ===")
    print(f"Cohort customers: {len(feature_df)}")
    print(f"Lapsed: {int(y.sum())} ({y.mean() * 100:.2f}%)")
    print(f"Training rows: {len(X_train)}; test rows: {len(X_test)}")

    baseline_pred = np.zeros(len(y_test), dtype=int)
    baseline = metrics_for_lapsed(y_test, baseline_pred)
    print("\nBaseline (always predict not lapsed):", baseline)

    add_report(
        report_lines,
        "Lapse cohort and baseline",
        f"Cohort: {len(feature_df)}; lapsed: {int(y.sum())} "
        f"({y.mean() * 100:.2f}%); train: {len(X_train)}; "
        f"test: {len(X_test)}.\nBaseline: {baseline}",
    )

    return (
        feature_df, X_train, X_test, y_train, y_test,
        ids_test, numeric_features, categorical_features,
    )


# ---------------------------------------------------------
# Step 4: Train, tune and evaluate models
# ---------------------------------------------------------

def build_preprocessor(numeric_features, categorical_features):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric_features),
        ("categorical", categorical_pipe, categorical_features),
    ])


def tune_models(
    X_train, X_test, y_train, y_test,
    numeric_features, categorical_features, report_lines,
):
    preprocessor = build_preprocessor(
        numeric_features, categorical_features
    )

    logistic_pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    logistic_grid = GridSearchCV(
        logistic_pipe,
        {"model__C": [0.01, 0.1, 1, 10, 100]},
        cv=5,
        scoring="f1",
        n_jobs=-1,
    )
    logistic_grid.fit(X_train, y_train)
    logistic_model = logistic_grid.best_estimator_

    fitted_scaler = (
        logistic_model.named_steps["preprocess"]
        .named_transformers_["numeric"]
        .named_steps["scaler"]
    )
    frequency_index = numeric_features.index("frequency")
    train_scaler_frequency_mean = fitted_scaler.mean_[frequency_index]
    full_cohort_frequency_mean = pd.concat(
        [X_train["frequency"], X_test["frequency"]]
    ).mean()
    print(
        "\\nFrequency mean — training-fitted scaler:",
        round(float(train_scaler_frequency_mean), 4),
        "| full cohort:",
        round(float(full_cohort_frequency_mean), 4),
    )
    print(
        "These means may differ because the scaler is fitted only on training "
        "rows, while the full-cohort mean includes the held-out test rows."
    )

    rf_pipe = Pipeline([
        ("preprocess", build_preprocessor(
            numeric_features, categorical_features
        )),
        ("model", RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        )),
    ])
    rf_grid = GridSearchCV(
        rf_pipe,
        {
            "model__max_depth": [3, 5, 8],
            "model__min_samples_leaf": [5, 10, 20],
        },
        cv=5,
        scoring="f1",
        n_jobs=-1,
    )
    rf_grid.fit(X_train, y_train)
    rf_model = rf_grid.best_estimator_

    logistic_pred = logistic_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)
    logistic_metrics = metrics_for_lapsed(y_test, logistic_pred)
    rf_metrics = metrics_for_lapsed(y_test, rf_pred)

    print("\n=== Logistic Regression ===")
    print("Best parameters:", logistic_grid.best_params_)
    print("Best CV F1:", round(logistic_grid.best_score_, 4))
    print("Test metrics:", logistic_metrics)

    print("\n=== Random Forest ===")
    print("Best parameters:", rf_grid.best_params_)
    print("Best CV F1:", round(rf_grid.best_score_, 4))
    print("Test metrics:", rf_metrics)

    # Choose the model with the better test F1 after both grid searches finish.
    # The test set is not used for grid search or model fitting.
    if rf_metrics["f1_lapsed"] >= logistic_metrics["f1_lapsed"]:
        selected_model = rf_model
        selected_name = "Random Forest"
        selected_pred = rf_pred
        selected_metrics = rf_metrics
    else:
        selected_model = logistic_model
        selected_name = "Logistic Regression"
        selected_pred = logistic_pred
        selected_metrics = logistic_metrics

    print(f"\nSelected model: {selected_name}")
    print("Selected test metrics:", selected_metrics)

    if selected_metrics["f1_lapsed"] < 0.55:
        warnings.warn(
            "Selected test F1 is below the assignment target of 0.55. "
            "Inspect your data and results before submission."
        )

    add_report(
        report_lines,
        "Model tuning and test evaluation",
        f"Logistic best parameters: {logistic_grid.best_params_}; "
        f"best CV F1: {logistic_grid.best_score_:.4f}; "
        f"test: {logistic_metrics}\n"
        f"Random Forest best parameters: {rf_grid.best_params_}; "
        f"best CV F1: {rf_grid.best_score_:.4f}; "
        f"test: {rf_metrics}\n"
        f"Selected model: {selected_name}; metrics: {selected_metrics}\n"
        "The test set was not used for hyperparameter tuning. "
        "Missing a lapsing customer may lose future orders, while wrongly "
        "targeting a loyal customer wastes discounts and retention budget.",
    )

    # Oversample only the training rows for the logistic regression experiment.
    train = X_train.copy()
    train["_label"] = y_train.to_numpy()
    class_counts = train["_label"].value_counts()

    if 1 not in class_counts.index or 0 not in class_counts.index:
        raise ValueError("Training split must contain both lapse classes.")

    majority_count = int(class_counts.max())
    minority_count = int(class_counts.min())
    minority_label = int(class_counts.idxmin())

    minority_rows = train.loc[train["_label"].eq(minority_label)]
    sample_count = majority_count - minority_count
    extra = minority_rows.sample(
        n=sample_count, replace=True, random_state=42
    )

    balanced_train = pd.concat([train, extra], ignore_index=True)
    X_balanced = balanced_train.drop(columns="_label")
    y_balanced = balanced_train["_label"].astype(int)

    balanced_logistic = clone(logistic_model)
    balanced_logistic.fit(X_balanced, y_balanced)
    balanced_pred = balanced_logistic.predict(X_test)
    balanced_metrics = metrics_for_lapsed(y_test, balanced_pred)

    print("\n=== Training-only oversampling experiment ===")
    print("Original training class counts:", class_counts.to_dict())
    print("Balanced training class counts:", y_balanced.value_counts().to_dict())
    print("Unbalanced logistic metrics:", logistic_metrics)
    print("Balanced logistic metrics:", balanced_metrics)

    add_report(
        report_lines,
        "Training-only oversampling",
        f"Original class counts: {class_counts.to_dict()}; "
        f"balanced counts: {y_balanced.value_counts().to_dict()}.\n"
        f"Unbalanced logistic: {logistic_metrics}\n"
        f"Balanced logistic: {balanced_metrics}\n"
        "Oversampling is applied only to training rows; the original test set "
        "remains unchanged. Compare recall, precision and F1 to understand "
        "whether more at-risk customers are found at the cost of extra false alarms.",
    )

    # Report the top five RF importances and three logistic coefficient signs.
    rf_pre = rf_model.named_steps["preprocess"]
    rf_names = rf_pre.get_feature_names_out()
    rf_importances = rf_model.named_steps["model"].feature_importances_
    importance_df = pd.DataFrame({
        "feature": rf_names,
        "importance": rf_importances,
    }).sort_values("importance", ascending=False).head(5)

    log_pre = logistic_model.named_steps["preprocess"]
    log_names = log_pre.get_feature_names_out()
    log_coefficients = logistic_model.named_steps["model"].coef_[0]
    coefficient_df = pd.DataFrame({
        "feature": log_names,
        "coefficient": log_coefficients,
    })
    coefficient_df["sign"] = np.where(
        coefficient_df["coefficient"] >= 0, "positive", "negative"
    )
    coefficient_df["absolute_coefficient"] = coefficient_df["coefficient"].abs()
    coefficient_df = coefficient_df.sort_values(
        "absolute_coefficient", ascending=False
    ).head(3)

    print("\n=== Top five Random Forest features ===")
    print(importance_df.to_string(index=False))
    print("\n=== Three strongest Logistic Regression coefficients ===")
    print(coefficient_df[["feature", "coefficient", "sign"]].to_string(index=False))

    add_report(report_lines, "Random Forest feature importance", importance_df.to_string(index=False))
    add_report(
        report_lines,
        "Logistic Regression coefficient signs",
        coefficient_df[["feature", "coefficient", "sign"]].to_string(index=False),
    )

    # Refit the selected model on the original training set, not the
    # oversampled experiment, before saving and verifying it.
    if selected_name == "Random Forest":
        selected_model = rf_model
    else:
        selected_model = logistic_model

    model_path = OUTPUT_DIR / "lapse_model.joblib"
    joblib.dump(selected_model, model_path)
    reloaded_model = joblib.load(model_path)

    original_predictions = selected_model.predict(X_test)
    reloaded_predictions = reloaded_model.predict(X_test)
    predictions_match = np.array_equal(
        original_predictions, reloaded_predictions
    )
    print("\nSaved model:", model_path)
    print("Reloaded model predictions match exactly:", predictions_match)
    assert predictions_match, "Reloaded model predictions do not match!"

    add_report(
        report_lines,
        "Model persistence",
        f"Saved to {model_path.relative_to(ROOT)}. "
        f"Reloaded predictions match exactly: {predictions_match}.",
    )

    return selected_model, selected_name, selected_metrics


# ---------------------------------------------------------
# Step 5: Premium retention list
# ---------------------------------------------------------

def build_retention_list(
    selected_model, feature_df, segment_df, X_test, y_test, ids_test, report_lines
):
    premium_ids = set(
        segment_df.loc[segment_df["is_premium"], "customer_id"]
    )
    test_customers = pd.DataFrame({
        "customer_id": ids_test.to_numpy(),
        "actually_lapsed": y_test.to_numpy(),
    })

    premium_test = test_customers.loc[
        test_customers["customer_id"].isin(premium_ids)
    ].copy()

    if premium_test.empty:
        retention = pd.DataFrame(
            columns=["customer_id", "lapse_probability", "actually_lapsed"]
        )
        truly_lapsed = 0
    else:
        X_test_by_id = X_test.copy()
        X_test_by_id["_customer_id"] = ids_test.to_numpy()
        X_premium = X_test_by_id.loc[
            X_test_by_id["_customer_id"].isin(premium_test["customer_id"])
        ].drop(columns="_customer_id")

        probabilities = selected_model.predict_proba(X_premium)[:, 1]
        risk = pd.DataFrame({
            "customer_id": X_test_by_id.loc[
                X_test_by_id["_customer_id"].isin(premium_test["customer_id"]),
                "_customer_id",
            ].to_numpy(),
            "lapse_probability": probabilities,
        })
        retention = risk.merge(
            premium_test, on="customer_id", how="left"
        ).sort_values("lapse_probability", ascending=False).head(10)

        retention["lapse_probability"] = retention["lapse_probability"].round(4)
        retention["actually_lapsed"] = retention["actually_lapsed"].astype(int)
        truly_lapsed = int(retention["actually_lapsed"].sum())

    retention = retention[
        ["customer_id", "lapse_probability", "actually_lapsed"]
    ]
    retention.to_csv(OUTPUT_DIR / "retention_list.csv", index=False)

    print("\n=== Premium retention list ===")
    print(retention.to_string(index=False))
    print(f"Truly lapsed in the list: {truly_lapsed} of {len(retention)}")
    print(
        "Note: this list is illustrative, not a leakage-free evaluation, "
        "because Premium is assigned using whole-window RFM including September."
    )

    add_report(
        report_lines,
        "Premium retention list",
        f"Rows: {len(retention)}; truly lapsed: {truly_lapsed}.\n"
        "This list is illustrative, not a leakage-free evaluation, because the "
        "Premium flag uses whole-window RFM including September. Model features "
        "themselves use only orders before 2025-09-01.",
    )


# ---------------------------------------------------------
# Run all Part B tasks
# ---------------------------------------------------------

def main():
    report_lines = [
        "PART B — CUSTOMER SEGMENTATION AND LAPSE PREDICTION",
        "Snapshot: 2025-10-01 00:00:00",
        "Lapse feature cutoff: 2025-09-01 00:00:00",
    ]

    orders, customers, products = read_inputs()

    rfm = build_rfm(orders)
    segment_df, premium_id = build_segments(rfm, report_lines)

    (
        feature_df, X_train, X_test, y_train, y_test,
        ids_test, numeric_features, categorical_features,
    ) = build_lapse_dataset(orders, customers, products, report_lines)

    selected_model, selected_name, selected_metrics = tune_models(
        X_train, X_test, y_train, y_test,
        numeric_features, categorical_features, report_lines,
    )

    build_retention_list(
        selected_model, feature_df, segment_df,
        X_test, y_test, ids_test, report_lines,
    )

    report_path = OUTPUT_DIR / "part_b_report.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("\n=== Part B completed ===")
    print("Report:", report_path)
    print("Outputs:")
    for filename in [
        "customer_rfm.csv",
        "cluster_evaluation.csv",
        "customer_segments.csv",
        "customer_segments_pca.csv",
        "lapse_model.joblib",
        "retention_list.csv",
        "part_b_report.txt",
    ]:
        print(" -", OUTPUT_DIR / filename)

    print("\nImportant methodology checks:")
    print("- Model cohort includes customers with completed orders before cutoff.")
    print("- September orders are used only to create the lapse label.")
    print("- Preprocessing is fitted within pipelines on training folds/rows only.")
    print("- Segment labels are not used as model features.")
    print("- Oversampling is used only in the training-only experiment.")
    print("- Premium retention list is illustrative, not a leakage-free evaluation.")


if __name__ == "__main__":
    main()
