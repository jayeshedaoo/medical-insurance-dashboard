import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Medical Insurance Statistical Dashboard",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# LOAD DATA
# =========================================================

DATA_URL = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL)


df = load_data()


# =========================================================
# TITLE
# =========================================================

st.title("Medical Insurance Statistical Dashboard")

st.write(
    "Applied Statistical Modeling & Interactive Web Dashboard — Lab 4"
)


# =========================================================
# SIDEBAR FILTERS
# =========================================================

st.sidebar.header("Data Filters")


# Age filter
age_range = st.sidebar.slider(
    "Select Age Range",
    min_value=int(df["age"].min()),
    max_value=int(df["age"].max()),
    value=(
        int(df["age"].min()),
        int(df["age"].max())
    ),
    key="age_filter"
)


# Smoker filter
smoker_options = sorted(
    df["smoker"].unique().tolist()
)

selected_smoker = st.sidebar.multiselect(
    "Smoker Status",
    options=smoker_options,
    default=smoker_options,
    key="smoker_filter"
)


# Region filter
region_options = sorted(
    df["region"].unique().tolist()
)

selected_region = st.sidebar.multiselect(
    "Region",
    options=region_options,
    default=region_options,
    key="region_filter"
)


# Sex filter
sex_options = sorted(
    df["sex"].unique().tolist()
)

selected_sex = st.sidebar.multiselect(
    "Sex",
    options=sex_options,
    default=sex_options,
    key="sex_filter"
)


# =========================================================
# APPLY SIDEBAR FILTERS
# =========================================================

filtered_df = df[
    (df["age"] >= age_range[0])
    & (df["age"] <= age_range[1])
    & (df["smoker"].isin(selected_smoker))
    & (df["region"].isin(selected_region))
    & (df["sex"].isin(selected_sex))
]


# =========================================================
# CREATE THREE TABS
# =========================================================

tab1, tab2, tab3 = st.tabs(
    [
        "1. Data Exploration",
        "2. Hypothesis Testing Lab",
        "3. Live Prediction & Diagnostics"
    ]
)


# =========================================================
# TAB 1 — DATA EXPLORATION
# =========================================================

with tab1:

    st.header("Data Exploration")

    # -----------------------------------------------------
    # DATASET OVERVIEW
    # -----------------------------------------------------

    st.subheader("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Number of Records",
        filtered_df.shape[0]
    )

    col2.metric(
        "Number of Variables",
        filtered_df.shape[1]
    )

    if len(filtered_df) > 0:

        col3.metric(
            "Average Charges",
            f"${filtered_df['charges'].mean():,.2f}"
        )

        col4.metric(
            "Average BMI",
            f"{filtered_df['bmi'].mean():.2f}"
        )

    else:

        col3.metric(
            "Average Charges",
            "N/A"
        )

        col4.metric(
            "Average BMI",
            "N/A"
        )


    # -----------------------------------------------------
    # DATA PREVIEW
    # -----------------------------------------------------

    st.subheader("Filtered Dataset")

    st.dataframe(
        filtered_df,
        use_container_width=True
    )


    # -----------------------------------------------------
    # DESCRIPTIVE STATISTICS
    # -----------------------------------------------------

    st.subheader("Descriptive Statistics")

    numerical_columns = [
        "age",
        "bmi",
        "children",
        "charges"
    ]

    if len(filtered_df) > 0:

        descriptive_stats = pd.DataFrame({

            "Mean":
                filtered_df[numerical_columns].mean(),

            "Median":
                filtered_df[numerical_columns].median(),

            "Standard Deviation":
                filtered_df[numerical_columns].std(),

            "IQR":
                filtered_df[numerical_columns].quantile(0.75)
                -
                filtered_df[numerical_columns].quantile(0.25),

            "Skewness":
                filtered_df[numerical_columns].skew(),

            "Kurtosis":
                filtered_df[numerical_columns].kurtosis()

        })

        st.dataframe(
            descriptive_stats.round(4),
            use_container_width=True
        )

    else:

        st.warning(
            "No records match the selected filters."
        )


    # -----------------------------------------------------
    # DISTRIBUTION PLOT
    # -----------------------------------------------------

    st.subheader("Distribution Analysis")

    distribution_variable = st.selectbox(
        "Select Numerical Variable",
        numerical_columns,
        key="distribution_variable"
    )

    if len(filtered_df) > 0:

        fig_distribution = px.histogram(
            filtered_df,
            x=distribution_variable,
            nbins=30,
            marginal="box",
            title=f"Distribution of {distribution_variable}"
        )

        st.plotly_chart(
            fig_distribution,
            use_container_width=True
        )


    # -----------------------------------------------------
    # BOX PLOT
    # -----------------------------------------------------

    st.subheader("Box Plot")

    box_col1, box_col2 = st.columns(2)

    with box_col1:

        box_numeric = st.selectbox(
            "Numerical Variable",
            numerical_columns,
            key="box_numeric_variable"
        )

    with box_col2:

        box_category = st.selectbox(
            "Categorical Variable",
            [
                "sex",
                "smoker",
                "region"
            ],
            key="box_category_variable"
        )


    if len(filtered_df) > 0:

        fig_box = px.box(
            filtered_df,
            x=box_category,
            y=box_numeric,
            color=box_category,
            title=f"{box_numeric} by {box_category}"
        )

        st.plotly_chart(
            fig_box,
            use_container_width=True
        )


    # -----------------------------------------------------
    # SCATTER PLOT
    # -----------------------------------------------------

    st.subheader("Bivariate Analysis")

    scatter_col1, scatter_col2 = st.columns(2)

    with scatter_col1:

        x_variable = st.selectbox(
            "X-axis",
            [
                "age",
                "bmi",
                "children"
            ],
            key="scatter_x_variable"
        )

    with scatter_col2:

        y_variable = st.selectbox(
            "Y-axis",
            [
                "charges",
                "age",
                "bmi"
            ],
            key="scatter_y_variable"
        )


    if len(filtered_df) > 0:

        fig_scatter = px.scatter(
            filtered_df,
            x=x_variable,
            y=y_variable,
            color="smoker",
            hover_data=[
                "sex",
                "region"
            ],
            title=f"{y_variable} vs {x_variable}"
        )

        st.plotly_chart(
            fig_scatter,
            use_container_width=True
        )


    # -----------------------------------------------------
    # CORRELATION MATRIX
    # -----------------------------------------------------

    st.subheader("Correlation Matrix")

    if len(filtered_df) > 1:

        correlation_matrix = filtered_df[
            numerical_columns
        ].corr()

        fig_correlation = px.imshow(
            correlation_matrix,
            text_auto=".2f",
            aspect="auto",
            title="Correlation Matrix"
        )

        st.plotly_chart(
            fig_correlation,
            use_container_width=True
        )


# =========================================================
# TAB 2 — HYPOTHESIS TESTING LAB
# =========================================================

with tab2:

    st.header("Hypothesis Testing Lab")

    st.write(
        "All hypothesis tests use a significance level "
        "of α = 0.05."
    )

    alpha = 0.05


    # =====================================================
    # HYPOTHESIS TEST 1
    # =====================================================

    st.subheader(
        "Hypothesis Test 1 — Two-Group Comparison"
    )

    st.write(
        "Compare a continuous numerical variable between "
        "two independent groups."
    )


    # -----------------------------------------------------
    # GROUPING VARIABLE
    # -----------------------------------------------------

    group_variable = st.selectbox(
        "Select Grouping Variable",
        [
            "smoker",
            "sex",
            "region"
        ],
        key="test1_group_variable"
    )


    # -----------------------------------------------------
    # NUMERICAL VARIABLE
    # -----------------------------------------------------

    test1_numeric = st.selectbox(
        "Select Continuous Variable",
        [
            "charges",
            "age",
            "bmi",
            "children"
        ],
        key="test1_numeric_variable"
    )


    # -----------------------------------------------------
    # AVAILABLE GROUPS
    # -----------------------------------------------------

    available_groups = sorted(
        df[group_variable].dropna().unique().tolist()
    )


    if len(available_groups) >= 2:

        col1, col2 = st.columns(2)

        with col1:

            group1 = st.selectbox(
                "Select Group 1",
                available_groups,
                index=0,
                key="test1_group1"
            )

        with col2:

            group2 = st.selectbox(
                "Select Group 2",
                available_groups,
                index=1,
                key="test1_group2"
            )


        if group1 == group2:

            st.warning(
                "Please select two different groups."
            )

        else:

            # ---------------------------------------------
            # EXTRACT GROUP DATA
            # ---------------------------------------------

            data1 = df[
                df[group_variable] == group1
            ][test1_numeric].dropna()

            data2 = df[
                df[group_variable] == group2
            ][test1_numeric].dropna()


            # ---------------------------------------------
            # HYPOTHESES
            # ---------------------------------------------

            st.markdown("### Hypotheses")

            st.write(
                "**H₀:** There is no significant difference "
                "between the two groups."
            )

            st.write(
                "**H₁:** There is a significant difference "
                "between the two groups."
            )


            # ---------------------------------------------
            # SHAPIRO-WILK TEST
            # ---------------------------------------------

            shapiro1 = stats.shapiro(data1)

            shapiro2 = stats.shapiro(data2)


            # ---------------------------------------------
            # LEVENE'S TEST
            # ---------------------------------------------

            levene_result = stats.levene(
                data1,
                data2
            )


            # ---------------------------------------------
            # NORMALITY DECISION
            # ---------------------------------------------

            normal1 = (
                shapiro1.pvalue > alpha
            )

            normal2 = (
                shapiro2.pvalue > alpha
            )


            # ---------------------------------------------
            # SELECT FINAL TEST
            # ---------------------------------------------

            if normal1 and normal2:

                equal_variance = (
                    levene_result.pvalue > alpha
                )

                final_result = stats.ttest_ind(
                    data1,
                    data2,
                    equal_var=equal_variance
                )

                test_name = (
                    "Independent Two-Sample t-test"
                )

            else:

                final_result = stats.mannwhitneyu(
                    data1,
                    data2,
                    alternative="two-sided"
                )

                test_name = (
                    "Mann-Whitney U Test"
                )


            # ---------------------------------------------
            # ASSUMPTION TEST RESULTS
            # ---------------------------------------------

            st.markdown(
                "### Assumption Tests"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    f"Shapiro p-value ({group1})",
                    f"{shapiro1.pvalue:.6g}"
                )

            with col2:

                st.metric(
                    f"Shapiro p-value ({group2})",
                    f"{shapiro2.pvalue:.6g}"
                )

            with col3:

                st.metric(
                    "Levene p-value",
                    f"{levene_result.pvalue:.6g}"
                )


            # ---------------------------------------------
            # ASSUMPTION INTERPRETATION
            # ---------------------------------------------

            if normal1:

                st.write(
                    f"Normality for **{group1}**: "
                    "Satisfied because p > 0.05."
                )

            else:

                st.write(
                    f"Normality for **{group1}**: "
                    "Not satisfied because p ≤ 0.05."
                )


            if normal2:

                st.write(
                    f"Normality for **{group2}**: "
                    "Satisfied because p > 0.05."
                )

            else:

                st.write(
                    f"Normality for **{group2}**: "
                    "Not satisfied because p ≤ 0.05."
                )


            if levene_result.pvalue > alpha:

                st.write(
                    "Equal variance assumption: "
                    "Satisfied because p > 0.05."
                )

            else:

                st.write(
                    "Equal variance assumption: "
                    "Not satisfied because p ≤ 0.05."
                )


            # ---------------------------------------------
            # FINAL TEST
            # ---------------------------------------------

            st.markdown(
                "### Final Statistical Test"
            )

            st.write(
                f"**Selected Test:** {test_name}"
            )

            st.write(
                f"**Test Statistic:** "
                f"{final_result.statistic:.6f}"
            )

            st.write(
                f"**p-value:** "
                f"{final_result.pvalue:.6g}"
            )


            # ---------------------------------------------
            # DECISION
            # ---------------------------------------------

            if final_result.pvalue < alpha:

                st.error(
                    "Decision: Reject H₀"
                )

                st.success(
                    f"Conclusion: There is a statistically "
                    f"significant difference in "
                    f"**{test1_numeric}** between "
                    f"**{group1}** and **{group2}** "
                    f"at α = 0.05."
                )

            else:

                st.success(
                    "Decision: Fail to Reject H₀"
                )

                st.info(
                    f"Conclusion: There is insufficient "
                    f"evidence of a statistically significant "
                    f"difference in **{test1_numeric}** between "
                    f"**{group1}** and **{group2}** "
                    f"at α = 0.05."
                )


            # ---------------------------------------------
            # GROUP SUMMARY
            # ---------------------------------------------

            st.markdown(
                "### Group Descriptive Statistics"
            )

            group_summary = pd.DataFrame({

                "Group": [
                    group1,
                    group2
                ],

                "Sample Size": [
                    len(data1),
                    len(data2)
                ],

                "Mean": [
                    data1.mean(),
                    data2.mean()
                ],

                "Median": [
                    data1.median(),
                    data2.median()
                ],

                "Standard Deviation": [
                    data1.std(),
                    data2.std()
                ]
            })


            st.dataframe(
                group_summary.round(4),
                use_container_width=True
            )


    # =====================================================
    # HYPOTHESIS TEST 2 — ONE-WAY ANOVA
    # =====================================================

    st.divider()

    st.subheader(
        "Hypothesis Test 2 — One-Way ANOVA"
    )

    st.write(
        "Test whether a numerical variable differs "
        "across three or more groups."
    )


    # -----------------------------------------------------
    # ANOVA GROUP
    # -----------------------------------------------------

    anova_group = st.selectbox(
        "Select ANOVA Grouping Variable",
        [
            "region",
            "smoker",
            "sex"
        ],
        index=0,
        key="anova_group_variable"
    )


    # -----------------------------------------------------
    # ANOVA NUMERICAL VARIABLE
    # -----------------------------------------------------

    anova_numeric = st.selectbox(
        "Select ANOVA Numerical Variable",
        [
            "charges",
            "age",
            "bmi",
            "children"
        ],
        index=0,
        key="anova_numeric_variable"
    )


    # -----------------------------------------------------
    # CREATE ANOVA GROUPS
    # -----------------------------------------------------

    anova_groups = []

    for group_name, group_data in df.groupby(
        anova_group
    ):

        values = group_data[
            anova_numeric
        ].dropna()

        if len(values) > 1:

            anova_groups.append(
                values
            )


    # -----------------------------------------------------
    # RUN ANOVA
    # -----------------------------------------------------

    if len(anova_groups) >= 3:

        # ---------------------------------------------
        # HYPOTHESES
        # ---------------------------------------------

        st.markdown(
            "### Hypotheses"
        )

        st.write(
            f"**H₀:** The mean **{anova_numeric}** "
            "is equal across all groups."
        )

        st.write(
            f"**H₁:** At least one group has a different "
            f"mean **{anova_numeric}**."
        )


        # ---------------------------------------------
        # ANOVA
        # ---------------------------------------------

        anova_result = stats.f_oneway(
            *anova_groups
        )


        # ---------------------------------------------
        # RESULTS
        # ---------------------------------------------

        st.markdown(
            "### ANOVA Results"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "F-statistic",
                f"{anova_result.statistic:.6f}"
            )

        with col2:

            st.metric(
                "p-value",
                f"{anova_result.pvalue:.6g}"
            )


        # ---------------------------------------------
        # DECISION
        # ---------------------------------------------

        if anova_result.pvalue < alpha:

            st.error(
                "Decision: Reject H₀"
            )

            st.success(
                f"Conclusion: The mean "
                f"**{anova_numeric}** differs "
                f"significantly across at least "
                f"one of the groups."
            )

        else:

            st.success(
                "Decision: Fail to Reject H₀"
            )

            st.info(
                f"Conclusion: There is insufficient "
                f"evidence that the mean "
                f"**{anova_numeric}** differs "
                f"significantly across the groups."
            )


        # ---------------------------------------------
        # GROUP STATISTICS
        # ---------------------------------------------

        st.markdown(
            "### Group Statistics"
        )

        anova_summary = df.groupby(
            anova_group
        )[anova_numeric].agg(
            [
                "count",
                "mean",
                "median",
                "std"
            ]
        ).reset_index()


        st.dataframe(
            anova_summary.round(4),
            use_container_width=True
        )


        # ---------------------------------------------
        # ANOVA BOXPLOT
        # ---------------------------------------------

        st.markdown(
            "### Group Comparison"
        )

        fig_anova = px.box(
            df,
            x=anova_group,
            y=anova_numeric,
            color=anova_group,
            title=(
                f"{anova_numeric} Across "
                f"{anova_group} Groups"
            )
        )


        st.plotly_chart(
            fig_anova,
            use_container_width=True
        )

    else:

        st.warning(
            "One-Way ANOVA requires at least "
            "three groups."
        )


# =========================================================
# TAB 3 — TEMPORARY PLACEHOLDER
# =========================================================

# =========================================================
# TAB 3 — LIVE PREDICTION & DIAGNOSTICS
# =========================================================

with tab3:

    st.header("Live Prediction & Diagnostics")

    st.write(
        "Multiple Linear Regression using Ordinary Least Squares (OLS)"
    )

    st.write(
        "Response variable: Medical Insurance Charges"
    )

    # =====================================================
    # 1. MODEL FORMULATION
    # =====================================================

    st.subheader("1. Multiple Linear Regression Model")

    st.markdown(
        """
        **Model:**

        Medical Charges = β₀ + β₁(Age) + β₂(BMI)
        + β₃(Children) + categorical effects + ε

        Categorical variables are represented using dummy variables.
        """
    )

    # =====================================================
    # 2. PREPARE DATA FOR OLS
    # =====================================================

    model_data = df[
        [
            "age",
            "bmi",
            "children",
            "sex",
            "smoker",
            "region",
            "charges"
        ]
    ].dropna().copy()

    # Predictor variables
    X_raw = model_data[
        [
            "age",
            "bmi",
            "children",
            "sex",
            "smoker",
            "region"
        ]
    ].copy()

    # Response variable
    y = model_data["charges"].copy()

    # -----------------------------------------------------
    # Convert categorical variables into dummy variables
    # -----------------------------------------------------

    X = pd.get_dummies(
        X_raw,
        columns=[
            "sex",
            "smoker",
            "region"
        ],
        drop_first=True
    )

    # Make sure all columns are numeric
    X = X.astype(float)

    # Add intercept
    X = sm.add_constant(
        X,
        has_constant="add"
    )

    # =====================================================
    # 3. FIT OLS MODEL
    # =====================================================

    model = sm.OLS(
        y,
        X
    ).fit()

    # =====================================================
    # 4. MODEL PERFORMANCE
    # =====================================================

    st.subheader("2. Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "R²",
            f"{model.rsquared:.4f}"
        )

    with col2:

        st.metric(
            "Adjusted R²",
            f"{model.rsquared_adj:.4f}"
        )

    with col3:

        st.metric(
            "F-statistic",
            f"{model.fvalue:.4f}"
        )

    with col4:

        if model.f_pvalue < 0.0001:
            model_p_display = "< 0.0001"
        else:
            model_p_display = f"{model.f_pvalue:.4f}"

        st.metric(
            "Model p-value",
            model_p_display
        )

    if model.f_pvalue < 0.05:

        st.success(
            "The overall regression model is statistically "
            "significant at α = 0.05."
        )

    else:

        st.info(
            "The overall regression model is not statistically "
            "significant at α = 0.05."
        )

    # =====================================================
    # 5. REGRESSION COEFFICIENTS
    # =====================================================

    st.subheader("3. Regression Coefficients")

    confidence_intervals = model.conf_int(alpha=0.05)

    coefficient_table = pd.DataFrame(
        {
            "Coefficient": model.params,
            "Std. Error": model.bse,
            "t-statistic": model.tvalues,
            "p-value": model.pvalues,
            "CI Lower (95%)": confidence_intervals[0],
            "CI Upper (95%)": confidence_intervals[1]
        }
    )

    coefficient_table.index.name = "Variable"

    st.dataframe(
        coefficient_table.style.format(
            {
                "Coefficient": "{:.4f}",
                "Std. Error": "{:.4f}",
                "t-statistic": "{:.4f}",
                "p-value": "{:.6g}",
                "CI Lower (95%)": "{:.4f}",
                "CI Upper (95%)": "{:.4f}"
            }
        ),
        use_container_width=True
    )

    # =====================================================
    # 6. COEFFICIENT INTERPRETATION
    # =====================================================

    st.markdown("### Parameter Interpretation")

    st.write(
        """
        For numerical predictors such as age, BMI and number of
        children, the coefficient represents the expected change
        in medical charges associated with a one-unit increase in
        that predictor, while holding the other variables constant.
        """
    )

    st.write(
        """
        For categorical dummy variables, the coefficient represents
        the expected difference in medical charges relative to the
        corresponding reference category, holding all other
        predictors constant.
        """
    )

    # =====================================================
    # 7. SIGNIFICANT PREDICTORS
    # =====================================================

    st.markdown("### Statistically Significant Predictors")

    significant_predictors = coefficient_table[
        coefficient_table["p-value"] < 0.05
    ].copy()

    if len(significant_predictors) > 0:

        st.dataframe(
            significant_predictors.style.format(
                {
                    "Coefficient": "{:.4f}",
                    "Std. Error": "{:.4f}",
                    "t-statistic": "{:.4f}",
                    "p-value": "{:.6g}",
                    "CI Lower (95%)": "{:.4f}",
                    "CI Upper (95%)": "{:.4f}"
                }
            ),
            use_container_width=True
        )

    else:

        st.info(
            "No predictors are statistically significant at α = 0.05."
        )

    # =====================================================
    # 8. ESTIMATED REGRESSION EQUATION
    # =====================================================

    st.subheader("4. Estimated Regression Equation")

    equation = (
        f"charges = {model.params['const']:.2f}"
    )

    for variable in model.params.index:

        if variable == "const":
            continue

        coefficient = model.params[variable]

        if coefficient >= 0:

            equation += (
                f" + {coefficient:.2f} × {variable}"
            )

        else:

            equation += (
                f" - {abs(coefficient):.2f} × {variable}"
            )

    st.code(
        equation,
        language="text"
    )

    # =====================================================
    # 9. RESIDUALS
    # =====================================================

    fitted_values = model.fittedvalues

    residuals = model.resid

    # =====================================================
    # 10. RESIDUALS VS FITTED
    # =====================================================

    st.subheader("5. Residual Diagnostics")

    st.markdown(
        "### Residuals vs Fitted Values"
    )

    residual_df = pd.DataFrame(
        {
            "Fitted Values": fitted_values,
            "Residuals": residuals
        }
    )

    fig_residuals = px.scatter(
        residual_df,
        x="Fitted Values",
        y="Residuals",
        title="Residuals vs Fitted Values"
    )

    fig_residuals.add_hline(
        y=0,
        line_dash="dash"
    )

    fig_residuals.update_layout(
        height=500
    )

    st.plotly_chart(
        fig_residuals,
        use_container_width=True
    )

    st.write(
        """
        Ideally, residuals should be randomly scattered around
        zero with approximately constant spread. A visible pattern
        or funnel-shaped spread may indicate violations of linearity
        or homoscedasticity assumptions.
        """
    )

    # =====================================================
    # 11. NORMAL Q-Q PLOT
    # =====================================================

    st.markdown(
        "### Normal Q-Q Plot"
    )

    fig_qq = sm.qqplot(
        residuals,
        line="45",
        fit=True
    )

    fig_qq.figure.set_size_inches(
        9,
        5
    )

    st.pyplot(
        fig_qq.figure,
        clear_figure=True
    )

    plt.close(
        fig_qq.figure
    )

    st.write(
        """
        If the residuals are approximately normally distributed,
        the points should lie close to the 45-degree reference
        line. Strong deviations, especially in the tails, indicate
        departure from normality.
        """
    )

    # =====================================================
    # 12. JARQUE-BERA TEST
    # =====================================================

    st.markdown(
        "### Jarque-Bera Test for Residual Normality"
    )

    jb_stat, jb_pvalue, skew, kurtosis = (
        sm.stats.stattools.jarque_bera(residuals)
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Jarque-Bera Statistic",
            f"{jb_stat:.4f}"
        )

    with col2:

        st.metric(
            "JB p-value",
            f"{jb_pvalue:.6g}"
        )

    with col3:

        st.metric(
            "Residual Skewness",
            f"{skew:.4f}"
        )

    with col4:

        st.metric(
            "Residual Kurtosis",
            f"{kurtosis:.4f}"
        )

    st.write(
        "H₀: The residuals are normally distributed."
    )

    st.write(
        "H₁: The residuals are not normally distributed."
    )

    if jb_pvalue < 0.05:

        st.warning(
            "Decision: Reject H₀. The Jarque-Bera test provides "
            "evidence that the residuals are not normally distributed "
            "at α = 0.05."
        )

    else:

        st.success(
            "Decision: Fail to Reject H₀. There is insufficient "
            "evidence that the residuals depart from normality "
            "at α = 0.05."
        )

    # =====================================================
    # 13. MULTICOLLINEARITY — VIF
    # =====================================================

    st.subheader(
        "6. Multicollinearity Diagnostic — VIF"
    )

    st.write(
        """
        Variance Inflation Factor (VIF) is calculated for the
        continuous and dummy-variable predictors. The intercept
        is excluded from the VIF calculation.
        """
    )

    # Remove intercept before calculating VIF
    X_vif = X.drop(
        columns=["const"]
    ).copy()

    vif_records = []

    for i, column in enumerate(X_vif.columns):

        try:

            vif_value = variance_inflation_factor(
                X_vif.values,
                i
            )

        except Exception:

            vif_value = np.nan

        vif_records.append(
            {
                "Variable": column,
                "VIF": vif_value
            }
        )

    vif_table = pd.DataFrame(
        vif_records
    )

    st.dataframe(
        vif_table.style.format(
            {
                "VIF": "{:.4f}"
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    valid_vif = vif_table[
        np.isfinite(vif_table["VIF"])
    ]

    if len(valid_vif) > 0:

        max_vif = valid_vif["VIF"].max()

        if max_vif < 5:

            st.success(
                "All predictors have VIF < 5, suggesting no "
                "serious multicollinearity."
            )

        elif max_vif < 10:

            st.warning(
                "At least one predictor has VIF between 5 and 10, "
                "indicating possible moderate multicollinearity."
            )

        else:

            st.error(
                "At least one predictor has VIF ≥ 10, indicating "
                "potentially serious multicollinearity."
            )

    # =====================================================
    # 14. COMPLETE OLS SUMMARY
    # =====================================================

    st.subheader(
        "7. Complete OLS Model Summary"
    )

    with st.expander(
        "Show Full Statsmodels OLS Summary"
    ):

        st.text(
            model.summary().as_text()
        )

    # =====================================================
    # 15. LIVE PREDICTION
    # =====================================================

    st.subheader(
        "8. Live Medical Charge Prediction"
    )

    st.write(
        """
        Enter patient characteristics to generate a real-time
        predicted insurance charge together with a 95% confidence
        interval and a 95% prediction interval.
        """
    )

    # -----------------------------------------------------
    # INPUT CONTROLS
    # -----------------------------------------------------

    input_col1, input_col2, input_col3 = st.columns(3)

    with input_col1:

        prediction_age = st.slider(
            "Age",
            min_value=int(df["age"].min()),
            max_value=int(df["age"].max()),
            value=30,
            key="prediction_age"
        )

        prediction_bmi = st.number_input(
            "BMI",
            min_value=float(df["bmi"].min()),
            max_value=float(df["bmi"].max()),
            value=30.0,
            step=0.1,
            format="%.1f",
            key="prediction_bmi"
        )

    with input_col2:

        prediction_children = st.number_input(
            "Number of Children",
            min_value=int(df["children"].min()),
            max_value=int(df["children"].max()),
            value=0,
            step=1,
            key="prediction_children"
        )

        prediction_sex = st.selectbox(
            "Sex",
            sorted(df["sex"].unique()),
            key="prediction_sex"
        )

    with input_col3:

        prediction_smoker = st.selectbox(
            "Smoker",
            sorted(df["smoker"].unique()),
            key="prediction_smoker"
        )

        prediction_region = st.selectbox(
            "Region",
            sorted(df["region"].unique()),
            key="prediction_region"
        )

    # =====================================================
    # 16. CREATE NEW PATIENT
    # =====================================================

    new_patient = pd.DataFrame(
        {
            "age": [prediction_age],
            "bmi": [prediction_bmi],
            "children": [prediction_children],
            "sex": [prediction_sex],
            "smoker": [prediction_smoker],
            "region": [prediction_region]
        }
    )

    # -----------------------------------------------------
    # Encode categorical variables exactly like training data
    # -----------------------------------------------------

    new_patient_encoded = pd.get_dummies(
        new_patient,
        columns=[
            "sex",
            "smoker",
            "region"
        ],
        drop_first=True
    )

    new_patient_encoded = new_patient_encoded.astype(float)

    # -----------------------------------------------------
    # Make prediction data have exactly the same columns
    # as the training design matrix
    # -----------------------------------------------------

    new_patient_encoded = new_patient_encoded.reindex(
        columns=X_vif.columns,
        fill_value=0
    )

    # Add intercept
    new_patient_X = sm.add_constant(
        new_patient_encoded,
        has_constant="add"
    )

    # Ensure exact same column order as training X
    new_patient_X = new_patient_X.reindex(
        columns=X.columns,
        fill_value=0
    )

    # =====================================================
    # 17. GENERATE PREDICTION
    # =====================================================

    prediction_result = model.get_prediction(
        new_patient_X
    )

    prediction_summary = prediction_result.summary_frame(
        alpha=0.05
    )

    predicted_charge = float(
        prediction_summary["mean"].iloc[0]
    )

    mean_ci_lower = float(
        prediction_summary["mean_ci_lower"].iloc[0]
    )

    mean_ci_upper = float(
        prediction_summary["mean_ci_upper"].iloc[0]
    )

    prediction_lower = float(
        prediction_summary["obs_ci_lower"].iloc[0]
    )

    prediction_upper = float(
        prediction_summary["obs_ci_upper"].iloc[0]
    )

    # =====================================================
    # 18. DISPLAY PREDICTION
    # =====================================================

    st.markdown(
        "### Prediction Result"
    )

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:

        st.metric(
            "Predicted Medical Charges",
            f"${predicted_charge:,.2f}"
        )

    with result_col2:

        st.metric(
            "95% Mean Confidence Interval",
            f"${mean_ci_lower:,.2f} – "
            f"${mean_ci_upper:,.2f}"
        )

    with result_col3:

        st.metric(
            "95% Prediction Interval",
            f"${prediction_lower:,.2f} – "
            f"${prediction_upper:,.2f}"
        )

    # =====================================================
    # 19. PREDICTION DETAILS
    # =====================================================

    st.markdown(
        "### Prediction Details"
    )

    prediction_table = pd.DataFrame(
        {
            "Measure": [
                "Predicted Charge",
                "95% Mean CI Lower",
                "95% Mean CI Upper",
                "95% Prediction Interval Lower",
                "95% Prediction Interval Upper"
            ],
            "Value": [
                predicted_charge,
                mean_ci_lower,
                mean_ci_upper,
                prediction_lower,
                prediction_upper
            ]
        }
    )

    st.dataframe(
        prediction_table.style.format(
            {
                "Value": "${:,.2f}"
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    # =====================================================
    # 20. INTERPRETATION OF PREDICTION INTERVAL
    # =====================================================

    st.info(
        """
        The 95% prediction interval represents the range in which
        the medical charge for an individual patient is expected
        to fall, based on the fitted regression model.
        """
    )

    if prediction_lower < 0:

        st.warning(
            """
            The lower bound of the raw statistical prediction
            interval is below zero. Since actual medical charges
            cannot be negative, this is a limitation of the
            unconstrained linear regression model. The raw interval
            is displayed rather than artificially changing the
            statistical result.
            """
        )