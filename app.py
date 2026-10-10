"""
Employee Attrition Analysis - Streamlit web application

A final-year project that explores the IBM employee attrition dataset.

Run locally with:   streamlit run app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# 1. CONSTANTS AND PAGE SETUP

BASE_DIR = Path(__file__).parent
DATA_FILE_NAME = "IBM Attrition Data.csv"
# Look for the CSV inside the 'data' folder first, then next to app.py
DATA_PATH = next(
    (path for path in [BASE_DIR / "data" / DATA_FILE_NAME, BASE_DIR / DATA_FILE_NAME] if path.exists()),
    BASE_DIR / "data" / DATA_FILE_NAME,
)
CSS_PATH = BASE_DIR / "style.css"

CHART_FONT = "Times New Roman, Times, Liberation Serif, serif"
COLOR_STAY = "#2f5d8a"   # employees who stayed
COLOR_LEAVE = "#b5472f"  # employees who left
COLOR_MAP = {"No": COLOR_STAY, "Yes": COLOR_LEAVE}

# Columns the application needs in order to work
REQUIRED_COLUMNS = [
    "Age", "Attrition", "Department", "DistanceFromHome", "Education",
    "EducationField", "EnvironmentSatisfaction", "JobSatisfaction",
    "MaritalStatus", "MonthlyIncome", "NumCompaniesWorked",
    "WorkLifeBalance", "YearsAtCompany",
]
NUMERIC_COLUMNS = [
    "Age", "DistanceFromHome", "Education", "EnvironmentSatisfaction",
    "JobSatisfaction", "MonthlyIncome", "NumCompaniesWorked",
    "WorkLifeBalance", "YearsAtCompany",
]
TEXT_COLUMNS = ["Attrition", "Department", "EducationField", "MaritalStatus"]

# Continuous numeric variables used for histograms, scatter plots and correlation
CONTINUOUS_COLUMNS = ["Age", "MonthlyIncome", "DistanceFromHome", "YearsAtCompany", "NumCompaniesWorked"]

# Variables that can be used to group employees (category -> attrition rate)
GROUPING_COLUMNS = [
    "Department", "EducationField", "MaritalStatus", "Education",
    "EnvironmentSatisfaction", "JobSatisfaction", "WorkLifeBalance",
    "NumCompaniesWorked", "AgeGroup", "TenureBand", "IncomeBand", "DistanceBand",
]

# Short descriptions shown in the Dataset Overview page
DATA_DICTIONARY = {
    "Age": "Employee age in years",
    "Attrition": "Whether the employee left the company (Yes / No)",
    "Department": "Department where the employee works",
    "DistanceFromHome": "Distance between home and workplace (as recorded in the file)",
    "Education": "Education level, coded 1 to 5",
    "EducationField": "Field of study",
    "EnvironmentSatisfaction": "Satisfaction with work environment, coded 1 to 4",
    "JobSatisfaction": "Satisfaction with the job, coded 1 to 4",
    "MaritalStatus": "Marital status (Single / Married / Divorced)",
    "MonthlyIncome": "Monthly income (currency units as recorded in the file)",
    "NumCompaniesWorked": "Number of companies the employee has worked for",
    "WorkLifeBalance": "Work-life balance rating, coded 1 to 4",
    "YearsAtCompany": "Years the employee has spent at the company",
}

st.set_page_config(
    page_title="Employee Attrition Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css() -> None:
    """Read style.css and inject it into the page."""
    if CSS_PATH.exists():
        st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    else:
        st.warning("style.css was not found, so the default Streamlit styling is being used.")



# 2. DATA LOADING AND CLEANING

@st.cache_data(show_spinner=False)
def read_csv_file(path_or_buffer) -> pd.DataFrame:
    """Read the CSV file into a DataFrame (cached so it only loads once)."""
    return pd.read_csv(path_or_buffer)


def clean_data(raw: pd.DataFrame):
    """
    Clean the raw dataset and document every step.

    Returns
    -------
    clean : the cleaned DataFrame with extra helper columns
    log   : a DataFrame listing each cleaning check and what was done
    """
    df = raw.copy()
    log = []

    # Step 1 - remove accidental spaces in text columns
    for col in TEXT_COLUMNS:
        df[col] = df[col].astype(str).str.strip()
    log.append(("Trim spaces in text columns", "Applied to 4 text columns", "Prevents 'Sales' and 'Sales ' being counted as different groups."))

    # Step 2 - make sure numeric columns really are numbers
    coerced_total = 0
    for col in NUMERIC_COLUMNS:
        before_missing = df[col].isna().sum()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        coerced_total += int(df[col].isna().sum() - before_missing)
    log.append(("Validate numeric data types", f"{coerced_total} values could not be converted", "Non-numeric entries would become missing values."))

    # Step 3 - missing values (numbers -> median, text -> 'Unknown')
    missing_before = int(df.isna().sum().sum())
    for col in NUMERIC_COLUMNS:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    for col in TEXT_COLUMNS:
        df[col] = df[col].replace({"nan": np.nan}).fillna("Unknown")
    log.append(("Handle missing values", f"{missing_before} missing values found", "Median is used for numbers and 'Unknown' for text, so no rows are lost."))

    # Step 4 - duplicate rows
    duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    log.append(("Remove duplicate records", f"{duplicates} duplicate rows removed", "Duplicates would count the same employee twice."))

    # Step 5 - the Attrition column must contain only Yes / No
    invalid_attrition = int((~df["Attrition"].isin(["Yes", "No"])).sum())
    df = df[df["Attrition"].isin(["Yes", "No"])].reset_index(drop=True)
    log.append(("Validate Attrition values", f"{invalid_attrition} rows with values other than Yes/No removed", "The analysis needs a clear Yes/No outcome."))

    # Step 6 - range checks (reported only; rows are not removed)
    bad_age = int(((df["Age"] < 16) | (df["Age"] > 70)).sum())
    bad_income = int((df["MonthlyIncome"] <= 0).sum())
    bad_rating = int(sum(((df[c] < 1) | (df[c] > 5)).sum() for c in
                         ["Education", "EnvironmentSatisfaction", "JobSatisfaction", "WorkLifeBalance"]))
    bad_negative = int(sum((df[c] < 0).sum() for c in ["DistanceFromHome", "YearsAtCompany", "NumCompaniesWorked"]))
    log.append(("Range validation", f"{bad_age + bad_income + bad_rating + bad_negative} impossible values found",
                "Checked age (16-70), income (> 0), ratings (1-5) and non-negative counts."))

    # Step 7 - helper columns used by the charts
    df["AttritionFlag"] = (df["Attrition"] == "Yes").astype(int)
    df["AgeGroup"] = pd.cut(df["Age"], bins=[0, 25, 35, 45, 55, 120],
                            labels=["25 and under", "26-35", "36-45", "46-55", "56 and over"])
    df["TenureBand"] = pd.cut(df["YearsAtCompany"], bins=[-1, 1, 3, 5, 10, 1000],
                              labels=["0-1 yrs", "2-3 yrs", "4-5 yrs", "6-10 yrs", "11+ yrs"])
    df["IncomeBand"] = pd.cut(df["MonthlyIncome"], bins=[0, 3000, 5000, 8000, 12000, np.inf],
                              labels=["Under 3,000", "3,000-4,999", "5,000-7,999", "8,000-11,999", "12,000+"])
    df["DistanceBand"] = pd.cut(df["DistanceFromHome"], bins=[0, 5, 10, 20, np.inf],
                                labels=["1-5", "6-10", "11-20", "21+"])
    log.append(("Create helper columns", "AttritionFlag, AgeGroup, TenureBand, IncomeBand, DistanceBand",
                "Grouping continuous variables makes patterns easier to read. The bands are fixed cut-offs chosen for readability."))

    log_table = pd.DataFrame(log, columns=["Step", "Result", "Why it matters"])
    return df, log_table


def find_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Count outliers with the IQR rule (values beyond 1.5 x IQR). They are reported, not deleted."""
    rows = []
    for col in ["Age", "MonthlyIncome", "DistanceFromHome", "YearsAtCompany", "NumCompaniesWorked"]:
        q1, q3 = np.percentile(df[col], [25, 75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((df[col] < low) | (df[col] > high)).sum())
        rows.append({"Variable": col, "Lower limit": round(low, 1), "Upper limit": round(high, 1),
                     "Outliers": count, "Share of rows (%)": round(count / len(df) * 100, 1)})
    return pd.DataFrame(rows)



# 3. SMALL HELPER FUNCTIONS (cards, charts, tables)

def metric_card(label: str, value: str, note: str = "", accent: bool = False) -> None:
    """Draw one styled metric card."""
    css_class = "metric-card accent" if accent else "metric-card"
    st.markdown(
        f"""<div class="{css_class}">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-note">{note}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def info_box(text: str, warning: bool = False) -> None:
    """Show a highlighted paragraph."""
    css_class = "warn-box" if warning else "info-box"
    st.markdown(f'<div class="{css_class}">{text}</div>', unsafe_allow_html=True)


def style_fig(fig, height: int = 400):
    """Apply the same font, colours and margins to every chart."""
    fig.update_layout(
        font=dict(family=CHART_FONT, size=14, color="#1b1b1b"),
        title_font=dict(family=CHART_FONT, size=17, color="#1f3a5f"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=10, r=10, t=55, b=10),
        height=height,
        legend=dict(font=dict(family=CHART_FONT)),
    )
    fig.update_xaxes(showgrid=False, linecolor="#999999")
    fig.update_yaxes(gridcolor="#e6e6e6", linecolor="#999999")
    return fig


def show_chart(fig, height: int = 400) -> None:
    """Style a Plotly figure and display it at full width."""
    st.plotly_chart(style_fig(fig, height), width="stretch")


def attrition_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Number of employees, leavers and attrition rate (%) for each group of `column`."""
    table = (df.groupby(column, observed=True)["AttritionFlag"]
               .agg(Employees="count", Leavers="sum").reset_index())
    table["Attrition Rate (%)"] = (table["Leavers"] / table["Employees"] * 100).round(1)
    return table


def rate_bar(df: pd.DataFrame, column: str, title: str = "", height: int = 380):
    """Bar chart of attrition rate for each group, with a dashed line for the overall rate."""
    table = attrition_by(df, column)
    table[column] = table[column].astype(str)
    overall = df["AttritionFlag"].mean() * 100
    fig = px.bar(table, x=column, y="Attrition Rate (%)", text="Attrition Rate (%)",
                 hover_data={"Employees": True, "Leavers": True},
                 title=title or f"Attrition rate by {column}", color_discrete_sequence=[COLOR_STAY])
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.add_hline(y=overall, line_dash="dash", line_color=COLOR_LEAVE,
                  annotation_text=f"Overall {overall:.1f}%", annotation_position="top right")
    fig.update_xaxes(type="category", title=column)
    fig.update_yaxes(title="Attrition rate (%)", range=[0, max(table["Attrition Rate (%)"].max() * 1.25, overall * 1.3)])
    return style_fig(fig, height)


def show_table(df: pd.DataFrame) -> None:
    """Static HTML table (uses Times New Roman, unlike the interactive grid)."""
    st.table(df)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Convert a DataFrame to CSV bytes for the download buttons."""
    return df.to_csv(index=False).encode("utf-8")



# 4. PAGES

def page_home(raw: pd.DataFrame, df: pd.DataFrame) -> None:
    st.markdown(
        """<div class="hero">
              <div class="tag"> &middot;</div>
              <h1>Employee Attrition Analysis</h1>
              <p>An interactive study of who leaves a company and which workplace and personal
              factors are associated with leaving, based on the IBM attrition dataset.</p>
           </div>""",
        unsafe_allow_html=True,
    )

    st.header("Dataset at a Glance")
    leavers = int(df["AttritionFlag"].sum())
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Employees", f"{len(df):,}", "Records analysed")
    with c2:
        metric_card("Variables", f"{raw.shape[1]}", "Columns in the file")
    with c3:
        metric_card("Employees who left", f"{leavers:,}", "Attrition = Yes", accent=True)
    with c4:
        metric_card("Attrition rate", f"{leavers / len(df) * 100:.1f}%", "Share of all employees", accent=True)

    st.header("Background")
    st.write(
        "Employee attrition happens when employees leave an organisation and are not immediately replaced. "
        "It is costly: organisations lose experience, must recruit and train new staff, and may see "
        "team morale fall. Understanding which groups of employees are more likely to leave helps "
        "managers decide where retention efforts are most needed."
    )
    st.write(
        "This project analyses an HR dataset in which each row is one employee. For each employee the data "
        "records age, department, education, income, satisfaction ratings, work-life balance, "
        "years at the company and whether the employee left."
    )

    st.header("Objectives")
    st.markdown(
        """
1. Examine the structure and quality of the dataset and document every cleaning decision.
2. Describe the employees using summary statistics and visualisations.
3. Measure the overall attrition rate and compare it across departments, age groups, income levels, tenure and satisfaction ratings.
4. Build an interactive dashboard so that any group of employees can be explored with filters.
5. Report evidence-based findings and state clearly what the data can and cannot show.
        """
    )

    st.header("How to Use This Application")
    st.markdown(
        """
- Use the **navigation menu** in the sidebar to move between sections.
- The **Filters** panel in the sidebar controls the *Interactive Dashboard* and *Data Download* pages.
- Hover over any chart to see exact values.
        """
    )
    info_box("<b>Note:</b> The analysis describes <i>associations</i> in the data. It does not prove that any single factor <i>causes</i> employees to leave.")


def page_overview(raw: pd.DataFrame, df: pd.DataFrame) -> None:
    st.title("Dataset Overview")
    st.write("This section shows the size, structure and basic statistics of the original (raw) dataset.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Rows", f"{raw.shape[0]:,}", "Employees")
    with c2:
        metric_card("Columns", f"{raw.shape[1]}", "Variables")
    with c3:
        metric_card("Numeric columns", f"{raw.select_dtypes(include='number').shape[1]}", "Integer values")
    with c4:
        metric_card("Text columns", f"{raw.shape[1] - raw.select_dtypes(include='number').shape[1]}", "Categories")

    st.header("Sample Records")
    rows_to_show = st.slider("Number of rows to display", 5, 25, 10)
    show_table(raw.head(rows_to_show))

    st.header("Columns and Data Types")
    structure = pd.DataFrame({
        "Column": raw.columns,
        "Data type": [str(t) for t in raw.dtypes],
        "Non-null values": raw.notna().sum().values,
        "Unique values": raw.nunique().values,
        "Description": [DATA_DICTIONARY.get(c, "") for c in raw.columns],
    })
    show_table(structure)
    st.caption("Descriptions of the coded columns (1 to 4 or 1 to 5) assume that higher numbers mean a higher rating, "
               "which is the usual coding for this dataset. The CSV file itself does not contain a data dictionary.")

    st.header("Summary Statistics (Numeric Variables)")
    summary = raw[NUMERIC_COLUMNS].describe().T
    # NumPy is used for the inter-quartile range and the coefficient of variation
    summary["IQR"] = [np.percentile(raw[c], 75) - np.percentile(raw[c], 25) for c in summary.index]
    summary["CV (std / mean)"] = np.round(summary["std"] / summary["mean"], 2)
    summary["skewness"] = raw[NUMERIC_COLUMNS].skew().round(2)
    summary = summary.round(2).rename(columns={"50%": "median", "25%": "Q1", "75%": "Q3"})
    show_table(summary.reset_index().rename(columns={"index": "Variable"}))

    st.header("Categorical Variables")
    chosen = st.selectbox("Choose a variable", TEXT_COLUMNS)
    counts = raw[chosen].value_counts().reset_index()
    counts.columns = [chosen, "Employees"]
    counts["Share (%)"] = (counts["Employees"] / counts["Employees"].sum() * 100).round(1)
    left, right = st.columns([1, 1.2])
    with left:
        show_table(counts)
    with right:
        fig = px.bar(counts, x=chosen, y="Employees", text="Employees",
                     title=f"Number of employees by {chosen}", color_discrete_sequence=[COLOR_STAY])
        fig.update_traces(textposition="outside")
        show_chart(fig, 340)


def page_cleaning(raw: pd.DataFrame, df: pd.DataFrame, log_table: pd.DataFrame) -> None:
    st.title("Data Cleaning")
    st.write("Every check below was run on the raw file before any analysis. "
             "The results are calculated live from the data, not typed in by hand.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Rows before", f"{len(raw):,}")
    with c2:
        metric_card("Rows after", f"{len(df):,}")
    with c3:
        metric_card("Missing values", f"{int(raw.isna().sum().sum())}", "In the raw file")
    with c4:
        metric_card("Duplicate rows", f"{int(raw.duplicated().sum())}", "In the raw file")

    st.header("Cleaning Log")
    show_table(log_table)

    st.header("Missing Values by Column")
    missing = pd.DataFrame({"Column": raw.columns, "Missing values": raw.isna().sum().values,
                            "Missing (%)": (raw.isna().mean() * 100).round(2).values})
    show_table(missing)

    st.header("Outlier Check (IQR Rule)")
    outliers = find_outliers(df)
    show_table(outliers)
    info_box("<b>Decision:</b> Outliers were <b>kept</b>. High incomes, long careers and many previous employers "
             "are realistic values for some employees, and nothing suggests they are data-entry errors. "
             "Removing them would hide real employees, especially long-serving or highly paid staff. "
             "Because of this, <b>medians</b> are shown next to means in the analysis, as they are less affected by extreme values.")

    st.header("Effect of the Cleaning Decisions")
    st.markdown(
        """
- **No rows were removed** if the checks above show zero duplicates, zero invalid Attrition values and zero missing values, so the cleaned data contains the same employees as the raw file.
- **Banding continuous variables** (age, income, tenure, distance) makes patterns easier to see, but the choice of cut-off points is a judgement. Different cut-offs could change how the bars look, although the overall pattern is unlikely to reverse.
- **Rating columns** (Education, satisfaction, work-life balance) are treated as ordered categories, not exact measurements, because the difference between a rating of 2 and 3 is not necessarily the same as between 3 and 4.
        """
    )

    st.header("Cleaned Data Preview")
    show_table(df.head(8))


def page_eda(df: pd.DataFrame) -> None:
    st.title("Exploratory Analysis")
    st.write("This section explores the full cleaned dataset. Use the tabs to move between topics.")

    tab_overall, tab_numeric, tab_category, tab_bands, tab_corr = st.tabs(
        ["Attrition Overview", "Numeric Variables", "Categorical Variables", "Grouped Patterns", "Correlation"]
    )

    # ---- Tab 1: overall attrition
    with tab_overall:
        left, right = st.columns([1, 1.4])
        with left:
            split = df["Attrition"].value_counts().reset_index()
            split.columns = ["Attrition", "Employees"]
            fig = px.pie(split, names="Attrition", values="Employees", hole=0.5, color="Attrition",
                         color_discrete_map=COLOR_MAP, title="Employees who stayed (No) and left (Yes)")
            fig.update_traces(textinfo="label+percent", textfont_size=15)
            show_chart(fig, 380)
        with right:
            show_chart(rate_bar(df, "Department"), 380)
        st.caption("About one in six employees left, so the data is imbalanced: far more employees stayed than left. "
                   "Attrition rates for small groups should be read with care.")

    # ---- Tab 2: numeric variables
    with tab_numeric:
        variable = st.selectbox("Choose a numeric variable", CONTINUOUS_COLUMNS)
        left, right = st.columns(2)
        with left:
            fig = px.histogram(df, x=variable, color="Attrition", barmode="overlay", opacity=0.65,
                               nbins=30, color_discrete_map=COLOR_MAP, title=f"Distribution of {variable}")
            show_chart(fig, 380)
        with right:
            fig = px.box(df, x="Attrition", y=variable, color="Attrition", color_discrete_map=COLOR_MAP,
                         title=f"{variable} for employees who stayed and left")
            fig.update_layout(showlegend=False)
            show_chart(fig, 380)
        by_group = df.groupby("Attrition")[variable].agg(["count", "mean", "median", "std", "min", "max"]).round(2).reset_index()
        show_table(by_group)

    # ---- Tab 3: categorical variables
    with tab_category:
        variable = st.selectbox("Choose a grouping variable", GROUPING_COLUMNS[:9], key="eda_cat")
        table = attrition_by(df, variable)
        reliable = table[table["Employees"] >= 30].copy()
        reliable[variable] = reliable[variable].astype(str)  # avoids "1.0" for rating columns
        left, right = st.columns([1.6, 1])
        with left:
            show_chart(rate_bar(df, variable), 400)
        with right:
            show_table(table)
        if not reliable.empty:
            top = reliable.loc[reliable["Attrition Rate (%)"].idxmax()]
            bottom = reliable.loc[reliable["Attrition Rate (%)"].idxmin()]
            st.markdown(
                f"Among groups with at least 30 employees, the highest attrition rate is **{top[variable]}** "
                f"({top['Attrition Rate (%)']}%, {int(top['Employees'])} employees) and the lowest is **{bottom[variable]}** "
                f"({bottom['Attrition Rate (%)']}%, {int(bottom['Employees'])} employees)."
            )

    # ---- Tab 4: grouped continuous variables
    with tab_bands:
        st.write("Continuous variables were grouped into bands so that the pattern of attrition is easier to read.")
        c1, c2 = st.columns(2)
        with c1:
            show_chart(rate_bar(df, "AgeGroup", "Attrition rate by age group"), 360)
        with c2:
            show_chart(rate_bar(df, "TenureBand", "Attrition rate by years at company"), 360)
        c3, c4 = st.columns(2)
        with c3:
            show_chart(rate_bar(df, "IncomeBand", "Attrition rate by monthly income"), 360)
        with c4:
            show_chart(rate_bar(df, "DistanceBand", "Attrition rate by distance from home"), 360)

    # ---- Tab 5: correlation
    with tab_corr:
        columns = CONTINUOUS_COLUMNS + ["JobSatisfaction", "EnvironmentSatisfaction", "WorkLifeBalance", "AttritionFlag"]
        corr = df[columns].corr().round(2)
        fig = px.imshow(corr, text_auto=True, zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                        aspect="auto", title="Correlation matrix (Pearson)")
        show_chart(fig, 560)
        st.caption("AttritionFlag is 1 for employees who left and 0 for those who stayed. "
                   "A negative value means the variable tends to be lower among leavers. Correlation does not prove cause and effect.")

        st.subheader("Relationship between two variables")
        c1, c2 = st.columns(2)
        x_var = c1.selectbox("X axis", CONTINUOUS_COLUMNS, index=0, key="eda_x")
        y_var = c2.selectbox("Y axis", CONTINUOUS_COLUMNS, index=1, key="eda_y")
        fig = px.scatter(df, x=x_var, y=y_var, color="Attrition", opacity=0.6,
                         color_discrete_map=COLOR_MAP, title=f"{y_var} against {x_var}")
        show_chart(fig, 430)


def page_dashboard(df: pd.DataFrame, filtered: pd.DataFrame) -> None:
    st.title("Interactive Dashboard")
    st.write("Use the **Filters** panel in the sidebar to choose a group of employees. "
             "Every number and chart on this page updates to match your selection.")

    if filtered.empty:
        st.error("No employees match the selected filters. Please widen the filters in the sidebar.")
        return

    overall_rate = df["AttritionFlag"].mean() * 100
    selected_rate = filtered["AttritionFlag"].mean() * 100
    difference = selected_rate - overall_rate

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Employees selected", f"{len(filtered):,}", f"{len(filtered) / len(df) * 100:.1f}% of all employees")
    with c2:
        metric_card("Attrition rate", f"{selected_rate:.1f}%", f"{difference:+.1f} points vs overall ({overall_rate:.1f}%)", accent=True)
    with c3:
        metric_card("Employees who left", f"{int(filtered['AttritionFlag'].sum()):,}", "In the selected group", accent=True)
    c4, c5, c6 = st.columns(3)
    with c4:
        metric_card("Average age", f"{filtered['Age'].mean():.1f}", "Years")
    with c5:
        metric_card("Average monthly income", f"{filtered['MonthlyIncome'].mean():,.0f}", f"Median {filtered['MonthlyIncome'].median():,.0f}")
    with c6:
        metric_card("Average years at company", f"{filtered['YearsAtCompany'].mean():.1f}", f"Median {filtered['YearsAtCompany'].median():.0f}")

    if len(filtered) < 50:
        info_box("Fewer than 50 employees are selected, so the percentages below may change a lot with just a few people.", warning=True)

    st.header("Attrition by Group")
    group_col = st.selectbox("Compare attrition rate by", GROUPING_COLUMNS, key="dash_group")
    left, right = st.columns([1.6, 1])
    with left:
        show_chart(rate_bar(filtered, group_col), 400)
    with right:
        split = filtered["Attrition"].value_counts().reset_index()
        split.columns = ["Attrition", "Employees"]
        fig = px.pie(split, names="Attrition", values="Employees", hole=0.5, color="Attrition",
                     color_discrete_map=COLOR_MAP, title="Stayed (No) and left (Yes)")
        fig.update_traces(textinfo="label+percent")
        show_chart(fig, 400)

    st.header("Explore Variables")
    c1, c2 = st.columns(2)
    with c1:
        box_var = st.selectbox("Variable for box plot", CONTINUOUS_COLUMNS, index=1, key="dash_box")
        fig = px.box(filtered, x="Department", y=box_var, color="Attrition", color_discrete_map=COLOR_MAP,
                     title=f"{box_var} by department and attrition")
        show_chart(fig, 400)
    with c2:
        sc1, sc2 = st.columns(2)
        x_var = sc1.selectbox("Scatter X", CONTINUOUS_COLUMNS, index=0, key="dash_x")
        y_var = sc2.selectbox("Scatter Y", CONTINUOUS_COLUMNS, index=1, key="dash_y")
        fig = px.scatter(filtered, x=x_var, y=y_var, color="Attrition", opacity=0.6,
                         color_discrete_map=COLOR_MAP, title=f"{y_var} against {x_var}")
        show_chart(fig, 400)

    st.header("Summary Table for the Selection")
    summary = (filtered.groupby("Department")
               .agg(Employees=("AttritionFlag", "count"), Leavers=("AttritionFlag", "sum"),
                    AvgAge=("Age", "mean"), AvgIncome=("MonthlyIncome", "mean"), AvgTenure=("YearsAtCompany", "mean"))
               .reset_index())
    summary["Attrition Rate (%)"] = (summary["Leavers"] / summary["Employees"] * 100).round(1)
    summary[["AvgAge", "AvgIncome", "AvgTenure"]] = summary[["AvgAge", "AvgIncome", "AvgTenure"]].round(1)
    show_table(summary)


def page_insights(df: pd.DataFrame) -> None:
    st.title("Insights and Findings")
    st.write("All figures on this page are calculated from the cleaned dataset when the page loads.")

    overall = df["AttritionFlag"].mean() * 100
    leavers = int(df["AttritionFlag"].sum())

    st.header("1. Overall Attrition")
    st.markdown(f"**{leavers} of {len(df):,} employees ({overall:.1f}%)** left the company. "
                f"The remaining {len(df) - leavers:,} stayed, so the outcome is imbalanced.")

    # Highest and lowest attrition group for every grouping variable
    st.header("2. Which Groups Have the Highest and Lowest Attrition?")
    rows = []
    for column in GROUPING_COLUMNS:
        table = attrition_by(df, column)
        table[column] = table[column].astype(str)  # keeps ratings as "1" instead of "1.0"
        table = table[table["Employees"] >= 30]  # ignore very small groups
        top = table.loc[table["Attrition Rate (%)"].idxmax()]
        bottom = table.loc[table["Attrition Rate (%)"].idxmin()]
        rows.append({
            "Factor": column,
            "Highest-attrition group": f"{top[column]} (n={int(top['Employees'])})",
            "Highest rate (%)": top["Attrition Rate (%)"],
            "Lowest-attrition group": f"{bottom[column]} (n={int(bottom['Employees'])})",
            "Lowest rate (%)": bottom["Attrition Rate (%)"],
            "Gap (points)": round(top["Attrition Rate (%)"] - bottom["Attrition Rate (%)"], 1),
        })
    comparison = pd.DataFrame(rows).sort_values("Gap (points)", ascending=False).reset_index(drop=True)
    show_table(comparison)
    st.caption("Only groups with at least 30 employees are compared. The larger the gap, the more attrition differs across that factor.")

    st.header("3. Employees Who Left Compared With Those Who Stayed")
    compare = df.groupby("Attrition")[["Age", "MonthlyIncome", "YearsAtCompany", "DistanceFromHome", "NumCompaniesWorked"]].agg(["mean", "median"]).round(1)
    compare.columns = [f"{a} ({b})" for a, b in compare.columns]
    show_table(compare.reset_index())

    st.header("4. Main Findings")
    age_table = attrition_by(df, "AgeGroup")
    tenure_table = attrition_by(df, "TenureBand")
    income_table = attrition_by(df, "IncomeBand")
    marital_table = attrition_by(df, "MaritalStatus").set_index("MaritalStatus")
    wlb_table = attrition_by(df, "WorkLifeBalance").set_index("WorkLifeBalance")
    env_table = attrition_by(df, "EnvironmentSatisfaction").set_index("EnvironmentSatisfaction")
    job_table = attrition_by(df, "JobSatisfaction").set_index("JobSatisfaction")
    dept_table = attrition_by(df, "Department").set_index("Department")
    dist_table = attrition_by(df, "DistanceBand").set_index("DistanceBand")

    def rate(table, label):
        return table.loc[label, "Attrition Rate (%)"]

    st.markdown(
        f"""
- **Age and tenure:** Attrition is highest among the youngest and newest employees: {age_table.iloc[0]['Attrition Rate (%)']}% for the {age_table.iloc[0]['AgeGroup']} group
  and {tenure_table.iloc[0]['Attrition Rate (%)']}% for employees with {tenure_table.iloc[0]['TenureBand']} at the company, compared with {overall:.1f}% overall.
- **Income:** Employees earning under 3,000 per month had an attrition rate of {income_table.iloc[0]['Attrition Rate (%)']}%, against
  {income_table.iloc[-1]['Attrition Rate (%)']}% for those earning 12,000 or more.
- **Marital status:** Single employees left at {rate(marital_table, 'Single')}%, compared with {rate(marital_table, 'Married')}% for married and {rate(marital_table, 'Divorced')}% for divorced employees.
- **Satisfaction and work-life balance:** Employees who gave the lowest rating (1) had higher attrition than the overall rate:
  job satisfaction {rate(job_table, 1)}%, environment satisfaction {rate(env_table, 1)}% and work-life balance {rate(wlb_table, 1)}%
  (work-life balance rating 1 has only {int(wlb_table.loc[1, 'Employees'])} employees, so treat that figure with care).
- **Distance from home:** Attrition rises from {rate(dist_table, '1-5')}% for those living 1-5 units away to {rate(dist_table, '21+')}% for those 21 or more units away.
- **Department:** Sales had the highest rate of the two large departments ({rate(dept_table, 'Sales')}%), against {rate(dept_table, 'Research & Development')}% in Research & Development.
        """
    )

    st.header("5. Limitations and Cautions")
    corr_income_age = df["MonthlyIncome"].corr(df["Age"])
    corr_income_tenure = df["MonthlyIncome"].corr(df["YearsAtCompany"])
    info_box(
        f"<b>Association is not causation.</b> This analysis shows which groups leave more often; it does not show why. "
        f"<b>Factors overlap:</b> monthly income is correlated with age (r = {corr_income_age:.2f}) and with years at the company "
        f"(r = {corr_income_tenure:.2f}), so part of the 'income effect' may really be an age or tenure effect. "
        f"<b>Limited variables:</b> the file contains only 13 columns, so important factors such as job role, overtime, promotion history or "
        f"manager quality are not available. <b>Imbalanced outcome:</b> only {overall:.1f}% of employees left, and some groups are small. "
        f"<b>No prediction model</b> was built; the findings are descriptive.",
        warning=True,
    )


def page_download(df: pd.DataFrame, filtered: pd.DataFrame) -> None:
    st.title("Data Download")
    st.write("Download the employees selected with the sidebar filters, or the full cleaned dataset. "
             "Files are in CSV format and open in Excel.")

    if filtered.empty:
        st.error("No employees match the selected filters, so there is nothing to download. Please widen the filters in the sidebar.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Rows in selection", f"{len(filtered):,}", f"of {len(df):,} total")
    with c2:
        metric_card("Columns", f"{filtered.shape[1]}", "Including helper columns")
    with c3:
        metric_card("Attrition rate", f"{filtered['AttritionFlag'].mean() * 100:.1f}%", "In the selection", accent=True)

    st.header("Download Files")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button("Filtered data (CSV)", to_csv_bytes(filtered), "filtered_attrition_data.csv", "text/csv")
    with d2:
        st.download_button("Full cleaned data (CSV)", to_csv_bytes(df), "cleaned_attrition_data.csv", "text/csv")
    with d3:
        summary = attrition_by(filtered, "Department")
        st.download_button("Department summary (CSV)", to_csv_bytes(summary), "department_summary.csv", "text/csv")

    st.header("Preview of the Filtered Data")
    st.caption("The interactive grid below is drawn by Streamlit and may not use Times New Roman. The downloaded file is unaffected.")
    st.dataframe(filtered.drop(columns=["AttritionFlag"]), width="stretch", hide_index=True)



# 5. SIDEBAR FILTERS

def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Draw the filter widgets and return the filtered DataFrame."""
    with st.sidebar.expander("Filters (Dashboard and Download)", expanded=False):
        departments = sorted(df["Department"].unique())
        marital = sorted(df["MaritalStatus"].unique())
        selected_dept = st.multiselect("Department", departments, default=departments, key="f_dept")
        selected_marital = st.multiselect("Marital status", marital, default=marital, key="f_marital")
        selected_status = st.multiselect("Attrition", ["No", "Yes"], default=["No", "Yes"], key="f_status")
        age_range = st.slider("Age", int(df["Age"].min()), int(df["Age"].max()),
                              (int(df["Age"].min()), int(df["Age"].max())), key="f_age")
        income_range = st.slider("Monthly income", int(df["MonthlyIncome"].min()), int(df["MonthlyIncome"].max()),
                                 (int(df["MonthlyIncome"].min()), int(df["MonthlyIncome"].max())), step=100, key="f_income")
        tenure_range = st.slider("Years at company", int(df["YearsAtCompany"].min()), int(df["YearsAtCompany"].max()),
                                 (int(df["YearsAtCompany"].min()), int(df["YearsAtCompany"].max())), key="f_tenure")
        job_ratings = sorted(df["JobSatisfaction"].unique())
        selected_job = st.multiselect("Job satisfaction", job_ratings, default=job_ratings, key="f_job")

    mask = (
        df["Department"].isin(selected_dept)
        & df["MaritalStatus"].isin(selected_marital)
        & df["Attrition"].isin(selected_status)
        & df["Age"].between(*age_range)
        & df["MonthlyIncome"].between(*income_range)
        & df["YearsAtCompany"].between(*tenure_range)
        & df["JobSatisfaction"].isin(selected_job)
    )
    return df[mask]



# 6. MAIN PROGRAM

def load_dataset():
    """Load the CSV from the data folder, or from an upload if the file is missing."""
    try:
        if DATA_PATH.exists():
            raw = read_csv_file(DATA_PATH)
        else:
            st.sidebar.warning("The dataset file was not found in the data folder.")
            uploaded = st.sidebar.file_uploader("Upload the IBM Attrition CSV file", type="csv")
            if uploaded is None:
                st.info("Please upload the dataset in the sidebar, or place 'IBM Attrition Data.csv' inside the 'data' folder.")
                return None
            raw = read_csv_file(uploaded)
    except Exception as error:  # unreadable or corrupt file
        st.error(f"The dataset could not be read. Details: {error}")
        return None

    missing_columns = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing_columns:
        st.error(f"The dataset is missing required columns: {', '.join(missing_columns)}")
        return None
    return raw


def main() -> None:
    load_css()

    raw = load_dataset()
    if raw is None:
        st.stop()

    df, log_table = clean_data(raw)

    st.sidebar.markdown("## Employee Attrition")
    st.sidebar.caption("Final Year Project")
    page = st.sidebar.radio(
        "Navigation",
        ["Home", "Dataset Overview", "Data Cleaning", "Exploratory Analysis",
         "Interactive Dashboard", "Insights and Findings", "Data Download"],
        label_visibility="collapsed",
    )
    filtered = sidebar_filters(df)

    if page == "Home":
        page_home(raw, df)
    elif page == "Dataset Overview":
        page_overview(raw, df)
    elif page == "Data Cleaning":
        page_cleaning(raw, df, log_table)
    elif page == "Exploratory Analysis":
        page_eda(df)
    elif page == "Interactive Dashboard":
        page_dashboard(df, filtered)
    elif page == "Insights and Findings":
        page_insights(df)
    else:
        page_download(df, filtered)

    st.markdown('<div class="footer">Employee Attrition Analysis &middot; Built with Python, Pandas, NumPy, Plotly and Streamlit</div>',
                unsafe_allow_html=True)


main()