import os
import io
import base64
import asyncio
import warnings
import threading
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import graphviz
from openai import OpenAI
from agents import Agent, Runner, function_tool
from llama_cloud import LlamaCloud
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings('ignore')


# CONFIG

OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY")
LLAMAINDEX_API_KEY  = os.getenv("LLAMAINDEX_API_KEY")
LLAMACLOUD_PIPELINE_ID = os.getenv("LLAMACLOUD_PIPELINE_ID", "")
MODEL_FLAGSHIP      = "gpt-5.4"
MODEL_MINI          = "gpt-5.4-mini"

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Global summary cache that function_tools read from at runtime
_summary_cache: dict = {}

# DATA LOADING


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1")

    # Rename columns to match what the app expects
    df = df.rename(columns={
        "Incident Number": "INCIDENT_NUMBER",
        "From Date": "OCCURRED_ON_DATE",
        "BPD District": "DISTRICT",
        #"Crime Category": "OFFENSE_CODE_GROUP",
        "Offense Description": "OFFENSE_CODE_GROUP",
        "Crime Part": "UCR_PART",
        "Hour of Day": "HOUR",
        "Day of Week": "DAY_OF_WEEK",
        "Block Address": "STREET",
        "Year": "YEAR",
        "Month": "MONTH",
    })

    df["OCCURRED_ON_DATE"] = pd.to_datetime(df["OCCURRED_ON_DATE"], errors="coerce")

    df = df[df["YEAR"].between(2020, 2026)]

    # SHOOTING column doesn't exist in this dataset, add a placeholder
    if "SHOOTING" not in df.columns:
        df["SHOOTING"] = 0

    return df

def build_summary_cache(df: pd.DataFrame):
    """Populate global stats cache so agent tools can read it without holding df."""
    global _summary_cache
    _summary_cache = {
        "overview": (
            f"Dataset contains {len(df):,} incidents across {df['YEAR'].nunique()} years "
            f"({df['YEAR'].min()} to {df['YEAR'].max()}), "
            f"{df['DISTRICT'].nunique()} districts, and "
            f"{df['OFFENSE_CODE_GROUP'].nunique()} offense categories."
        ),
        "hourly": df.groupby("HOUR")["INCIDENT_NUMBER"].count().to_dict(),
        "daily": df.groupby("DAY_OF_WEEK")["INCIDENT_NUMBER"].count().to_dict(),
        "district": df.groupby("DISTRICT")["INCIDENT_NUMBER"].count().sort_values(ascending=False).head(10).to_dict(),
        "offense": df.groupby("OFFENSE_CODE_GROUP")["INCIDENT_NUMBER"].count().sort_values(ascending=False).head(10).to_dict(),
        "ucr": df["UCR_PART"].value_counts().to_dict(),
        "streets": df.groupby("STREET")["INCIDENT_NUMBER"].count().sort_values(ascending=False).head(10).to_dict(),
        "shootings": int(df["SHOOTING"].sum()),
    }

# CHART GENERATION

def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded

@st.cache_data
def generate_charts(df_hash, _df) -> dict:
    """Generate all dashboard charts. Cached by dataframe hash."""
    plt.style.use("seaborn-v0_8")
    charts = {}

    # 1: Incidents by hour
    fig, ax = plt.subplots(figsize=(10, 3.8))
    hourly = _df.groupby("HOUR")["INCIDENT_NUMBER"].count()
    ax.bar(hourly.index, hourly.values, color="steelblue", alpha=0.82)
    ax.axvspan(9.5, 22.5, alpha=0.07, color="red", label="Peak window (10am to 10pm)")
    ax.set_title("Incidents by Hour of Day", fontweight="bold", fontsize=13)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Incident Count")
    ax.legend(fontsize=9)
    charts["hourly"] = {"b64": fig_to_base64(fig), "title": "Incidents by Hour of Day"}

    # Map integer days to names
    day_map = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
               4: "Friday", 5: "Saturday", 6: "Sunday"}
    _df["DAY_OF_WEEK"] = _df["DAY_OF_WEEK"].map(day_map)

    # 2: Incidents by day of week
    fig, ax = plt.subplots(figsize=(10, 3.8))
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    daily = _df.groupby("DAY_OF_WEEK")["INCIDENT_NUMBER"].count().reindex(day_order)
    bar_colors = ["#e74c3c" if d in ["Friday", "Saturday"] else "steelblue" for d in day_order]
    ax.bar(day_order, daily.values, color=bar_colors, alpha=0.85)
    ax.set_title("Incidents by Day of Week", fontweight="bold", fontsize=13)
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Incident Count")
    ax.tick_params(axis="x", rotation=20)
    charts["daily"] = {"b64": fig_to_base64(fig), "title": "Incidents by Day of Week"}

    # 3: Top districts
    fig, ax = plt.subplots(figsize=(10, 3.8))
    district = _df.groupby("DISTRICT")["INCIDENT_NUMBER"].count().sort_values(ascending=False).head(10)
    ax.bar(district.index, district.values, color="darkorange", alpha=0.85)
    ax.set_title("Top Districts by Incident Volume", fontweight="bold", fontsize=13)
    ax.set_xlabel("District")
    ax.set_ylabel("Incident Count")
    charts["district"] = {"b64": fig_to_base64(fig), "title": "Top Districts by Incident Volume"}

    # 4: Top offense descriptions
    fig, ax = plt.subplots(figsize=(11, 5))
    offenses = _df.groupby("OFFENSE_CODE_GROUP")["INCIDENT_NUMBER"].count().sort_values(ascending=False).head(10)
    offenses.sort_values().plot(kind="barh", ax=ax, color="mediumseagreen", alpha=0.85)
    ax.set_title("Top 10 Offense Descriptions", fontweight="bold", fontsize=13)
    ax.set_xlabel("Incident Count")
    ax.set_ylabel("Offense Type")
    plt.tight_layout()
    charts["offense"] = {"b64": fig_to_base64(fig), "title": "Top 10 Offense Descriptions"}

    # 5: Crime Part distribution (Boston PD classification, extends standard UCR)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ucr_counts = _df["UCR_PART"].value_counts()

    # Canonical order rather than frequency order
    part_order = ["Part One", "Part Two", "Part Three", "Other"]

    part_descriptions = {
        "Part One": "Part One: Major index crimes",
        "Part Two": "Part Two: Less serious offenses",
        "Part Three": "Part Three: Calls for service and non-criminal incidents",
        "Other": "Other: Miscellaneous",
    }

    colors_map = {
        "Part One": "#e74c3c",
        "Part Two": "#f39c12",
        "Part Three": "#3498db",
        "Other": "#95a5a6",
    }

    labels = [p for p in part_order if p in ucr_counts.index]
    values = [ucr_counts[p] for p in labels]
    colors = [colors_map[p] for p in labels]
    legend_labels = [part_descriptions[p] for p in labels]

    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        pctdistance=0.78,
        textprops={'fontsize': 9},
    )

    ax.set_title("Crime Part Distribution", fontweight="bold", fontsize=12, pad=10)

    ax.legend(
        wedges,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(1, -0.02),
        fontsize=5,
        frameon=False,
        handlelength=1.2,
    )

    plt.tight_layout()
    charts["ucr"] = {"b64": fig_to_base64(fig), "title": "Crime Part Distribution"}

    # 6: Month x Hour heatmap
    fig, ax = plt.subplots(figsize=(12, 4.5))
    pivot = _df.pivot_table(
        index="MONTH", columns="HOUR",
        values="INCIDENT_NUMBER", aggfunc="count"
    )
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax, linewidths=0.2, linecolor="white", cbar_kws={"shrink": 0.8})
    ax.set_title("Incident Heatmap: Month vs Hour", fontweight="bold", fontsize=13)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Month")
    charts["heatmap"] = {"b64": fig_to_base64(fig), "title": "Month x Hour Heatmap"}

    # 7: Top streets
    fig, ax = plt.subplots(figsize=(10, 4.5))
    streets = _df.groupby("STREET")["INCIDENT_NUMBER"].count().sort_values(ascending=False).head(10)
    streets.sort_values().plot(kind="barh", ax=ax, color="purple", alpha=0.78)
    ax.set_title("Top 10 Streets by Incident Count", fontweight="bold", fontsize=13)
    ax.set_xlabel("Incident Count")
    charts["streets"] = {"b64": fig_to_base64(fig), "title": "Top 10 Streets by Incident Count"}

    # 8: Shootings by District (2024 vs 2025 comparison)
    fig, ax = plt.subplots(figsize=(11, 4.5))

    # Filter to shootings only, and only 2024-2025
    shootings_df = _df[(_df["SHOOTING"] == 1) & (_df["YEAR"].isin([2024, 2025]))]

    if len(shootings_df) == 0:
        # Graceful fallback if no shooting data is available
        ax.text(0.5, 0.5,
                "Shooting data not available in this dataset.\n"
                "Upload the preprocessed CSV with the SHOOTING column to enable this chart.",
                ha="center", va="center", fontsize=11, color="gray",
                transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        # Top 10 districts by total shooting volume across 2024-2025
        top_districts = (
            shootings_df.groupby("DISTRICT")["INCIDENT_NUMBER"]
            .count()
            .sort_values(ascending=False)
            .head(10)
            .index.tolist()
        )

        # Counts per district for each year
        counts_2024 = (
            shootings_df[shootings_df["YEAR"] == 2024]
            .groupby("DISTRICT")["INCIDENT_NUMBER"]
            .count()
            .reindex(top_districts, fill_value=0)
        )
        counts_2025 = (
            shootings_df[shootings_df["YEAR"] == 2025]
            .groupby("DISTRICT")["INCIDENT_NUMBER"]
            .count()
            .reindex(top_districts, fill_value=0)
        )

        # Side-by-side bars
        import numpy as np
        x = np.arange(len(top_districts))
        width = 0.4

        ax.bar(x - width / 2, counts_2024.values, width,
               color="#95a5a6", alpha=0.85, label="2024")
        ax.bar(x + width / 2, counts_2025.values, width,
               color="#c0392b", alpha=0.9, label="2025")

        ax.set_xticks(x)
        ax.set_xticklabels(top_districts)
        ax.set_title("Shootings by District (2024 vs 2025)", fontweight="bold", fontsize=13)
        ax.set_xlabel("District")
        ax.set_ylabel("Shooting Incidents")
        ax.legend(fontsize=10)

        # Citywide totals annotation
        total_2024 = int(counts_2024.sum())
        total_2025 = int(counts_2025.sum())
        change = total_2025 - total_2024
        pct = (change / total_2024 * 100) if total_2024 > 0 else 0
        arrow = "▲" if change > 0 else ("▼" if change < 0 else "→")
        ax.text(0.99, 0.97,
                f"2024: {total_2024:,} shootings\n"
                f"2025: {total_2025:,} shootings\n"
                f"{arrow} {abs(change):,} ({pct:+.1f}%)",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, family="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="#888", alpha=0.9))

    plt.tight_layout()
    charts["shootings"] = {"b64": fig_to_base64(fig), "title": "Shootings by District (2024 vs 2025)"}

    return charts

# LLAMACLOUD RETRIEVAL TOOL

@function_tool
def retrieve_crime_analysis_context(query: str) -> str:
    "Retrieve relevant crime analysis context from the indexed knowledge base."
    "When calling retrieve_crime_analysis_context, formulate queries that reference "
    "specific crime categories, districts, or time periods when possible. "
    "The knowledge base contains Boston BRIC weekly reports and FIO stop data."
    try:
        llama_client = LlamaCloud(api_key=LLAMAINDEX_API_KEY)
        results = llama_client.pipelines.retrieve(
            pipeline_id=LLAMACLOUD_PIPELINE_ID,
            query=query,
            dense_similarity_top_k=3,
        )
        return str(results)
    except Exception as e:
        return f"LlamaCloud retrieval unavailable: {str(e)}. Falling back to cached stats."

# CRIME STATS TOOL (reads from global cache)

@function_tool
def get_crime_dataset_stats(metric: str) -> str:
    """
    Get precomputed summary statistics from the loaded crime dataset.
    metric must be one of: overview, hourly, daily, district, offense, ucr, streets, shootings
    """
    if not _summary_cache:
        return "No dataset loaded. Please upload a CSV file first."
    val = _summary_cache.get(metric)
    if val is None:
        keys = list(_summary_cache.keys())
        return f"Unknown metric '{metric}'. Available: {keys}"
    return str(val)

# AGENT DEFINITIONS

data_retrieval_agent = Agent(
    name="Crime Data Retrieval Agent",
    instructions=(
        "You are a crime data retrieval specialist. "
        "For every user question about crime patterns or data, you MUST call "
        "retrieve_crime_analysis_context first to fetch relevant context from the knowledge base, "
        "then call get_crime_dataset_stats with the most relevant metric (overview, hourly, daily, "
        "district, offense, ucr, streets, or shootings) to pull actual numbers. "
        "Combine both sources and respond with factual, specific, data-driven summaries."
    ),
    tools=[retrieve_crime_analysis_context, get_crime_dataset_stats],
    model=MODEL_MINI,
)

strategy_recommendation_agent = Agent(
    name="Police Strategy Recommendation Agent",
    instructions=(
        "You are a police strategy analyst producing briefing materials for command staff. "
        "Your job is to turn crime data patterns into clear, numbered, actionable recommendations. "
        "Every recommendation must cite the specific pattern or statistic that justifies it. "
        "Use direct language. Format output as: numbered strategic priorities, each with a "
        "one-sentence rationale. Keep the tone professional and concise."
    ),
    model=MODEL_FLAGSHIP,
)

supervisor_agent = Agent(
    name="Crime Analysis Supervisor",
    instructions=(
        "You are the supervisor for a police crime analysis AI tool. "
        "Route requests as follows: "
        "For questions about what the data shows, trends, patterns, or specific statistics, "
        "hand off to the Crime Data Retrieval Agent. "
        "For requests about what actions to take, recommendations, deployment, or strategy, "
        "hand off to the Police Strategy Recommendation Agent. "
        "For chart image analysis, use your own vision capabilities to describe what you see "
        "and then hand off to the Strategy agent for recommendations. "
        "Always state which agent handled the request at the end of your response."
    ),
    handoffs=[data_retrieval_agent, strategy_recommendation_agent],
    model=MODEL_FLAGSHIP,
)

# VISION ANALYSIS (direct OpenAI client for image inputs)

def analyze_chart_with_vision(b64_image: str, chart_title: str, model: str) -> str:
    """Send a chart to GPT vision and return a strategic analysis."""
    prompt = (
        f"You are assisting a police department crime analyst preparing a weekly leadership briefing. "
        f"The attached chart is titled: '{chart_title}'. "
        f"Analyze the chart and provide: "
        f"(1) Key patterns or anomalies visible in the data. "
        f"(2) Peak periods or high-volume categories that warrant attention. "
        f"(3) Two to three specific, actionable patrol or resource allocation recommendations "
        f"that a police commander could act on this week based on what you see. "
        f"Keep the tone direct and suitable for a command staff audience."
    )
    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                    },
                ],
            }
        ],
        max_completion_tokens=600,
    )
    return response.choices[0].message.content

# AGENT RUNNER (thread-safe for Streamlit)

def run_agent_sync(message: str) -> str:
    """Run the supervisor agent in a background thread to avoid event loop conflicts."""
    result_holder = {}

    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                Runner.run(supervisor_agent, message)
            )
            result_holder["output"] = result.final_output
        except Exception as e:
            result_holder["output"] = f"Agent error: {str(e)}"
        finally:
            loop.close()

    t = threading.Thread(target=_run)
    t.start()
    t.join(timeout=60)
    return result_holder.get("output", "Agent timed out. Please try again.")

# GRAPHVIZ AGENT NETWORK

def build_agent_graph() -> graphviz.Digraph:
    dot = graphviz.Digraph(comment="Crime Analysis Agent Network")
    dot.attr(rankdir="LR", bgcolor="transparent")
    dot.attr(
        "node",
        shape="box",
        style="filled,rounded",
        fontname="Arial",
        fontsize="10",
    )
    dot.attr("edge", fontname="Arial", fontsize="9", color="#4a9eff")

    dot.node("user",       "Analyst / User",                    fillcolor="#2d2d2d", fontcolor="white", shape="oval")
    dot.node("supervisor", f"Supervisor Agent\n{MODEL_FLAGSHIP}", fillcolor="#1f3864", fontcolor="white")
    dot.node("retrieval",  f"Data Retrieval Agent\n{MODEL_MINI}", fillcolor="#1a5c1a", fontcolor="white")
    dot.node("strategy",   f"Strategy Agent\n{MODEL_FLAGSHIP}",  fillcolor="#5c1a1a", fontcolor="white")
    dot.node("llama",      "LlamaCloud\nKnowledge Base",         fillcolor="#4a3000", fontcolor="white", shape="cylinder")
    dot.node("vision",     "OpenAI Vision API",                  fillcolor="#1a3a5c", fontcolor="white", shape="cylinder")
    dot.node("cache",      "Dataset Stats Cache",                fillcolor="#2d2d2d", fontcolor="white", shape="cylinder")

    dot.edge("user",       "supervisor", label="query")
    dot.edge("supervisor", "retrieval",  label="data/trends")
    dot.edge("supervisor", "strategy",   label="recommendations")
    dot.edge("supervisor", "vision",     label="chart images")
    dot.edge("retrieval",  "llama",      label="retrieve()")
    dot.edge("retrieval",  "cache",      label="get_stats()")
    dot.edge("llama",      "retrieval",  label="nodes")
    dot.edge("cache",      "retrieval",  label="stats")
    dot.edge("retrieval",  "supervisor", label="context")
    dot.edge("strategy",   "supervisor", label="briefing")
    dot.edge("vision",     "supervisor", label="insights")
    dot.edge("supervisor", "user",       label="response")

    return dot

# STREAMLIT UI

def main():
    st.set_page_config(
        page_title="Crime Analyst AI Tool",
        page_icon="🚔",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ---- HEADER ----
    st.markdown(
        """
        <h1 style='text-align:center; color:#1f3864;'>🚔 Crime Analyst AI Tool</h1>
        <p style='text-align:center; color:#555; font-size:1.05em;'>
        ChatGPT-powered weekly crime briefing assistant &nbsp;|&nbsp;
        ISM 647 Group Project &nbsp;|&nbsp; Powered by OpenAI Agents + LlamaCloud
        </p>
        <hr/>
        """,
        unsafe_allow_html=True,
    )

    # ---- SIDEBAR ----
    with st.sidebar:
        st.header("Configuration")

        uploaded_file = st.file_uploader(
            "Upload Crime Dataset (CSV)",
            type=["csv"],
            help="Upload crimeDATA.csv or any dataset with the same column schema.",
        )

        st.divider()
        selected_model = st.selectbox(
            "Vision Model",
            options=[MODEL_FLAGSHIP, MODEL_MINI],
            index=0,
            help="Model used for chart image analysis.",
        )

        llamacloud_ok = bool(LLAMAINDEX_API_KEY and LLAMACLOUD_PIPELINE_ID)
        st.divider()
        st.markdown("**LlamaCloud Status**")
        if llamacloud_ok:
            st.success("Connected")
            st.caption(f"Pipeline: `{LLAMACLOUD_PIPELINE_ID[:12]}...`")
        else:
            st.warning("Not configured - add LLAMAINDEX_API_KEY and LLAMACLOUD_PIPELINE_ID to .env")

        st.divider()
        st.markdown("**Agent Network**")
        try:
            dot = build_agent_graph()
            st.graphviz_chart(dot)
        except Exception as e:
            st.caption(f"Graph unavailable: {e}")

    # ---- LOAD DATA ----
    df = None
    charts = None

    if uploaded_file:
        try:
            df = load_data(uploaded_file)
            build_summary_cache(df)
            df_hash = len(df)
            charts = generate_charts(df_hash, df)
            st.success(f"Loaded {len(df):,} incidents across {df['YEAR'].nunique()} years.")
        except Exception as e:
            st.error(f"Error loading data: {e}")
    else:
        st.info("Upload crimeDATA.csv in the sidebar to get started.")

    # ---- TABS ----
    tab1, tab2, tab3 = st.tabs(["📊 Crime Dashboard", "💬 AI Analyst Chat", "📋 Strategy Briefing"])


    # TAB 1: DASHBOARD

    with tab1:
        if charts is None:
            st.warning("Upload a dataset to generate charts.")
        else:
            st.subheader("Crime Trend Visualizations")
            st.caption("Click 'Analyze with AI' beneath any chart to get GPT-powered insights and patrol recommendations.")

            chart_keys = list(charts.keys())

            for i in range(0, len(chart_keys), 2):
                col1, col2 = st.columns(2)
                for col, key in zip([col1, col2], chart_keys[i : i + 2]):
                    c = charts[key]
                    with col:
                        st.markdown(f"**{c['title']}**")
                        st.image(
                            base64.b64decode(c["b64"]),
                            use_container_width=True,
                        )
                        btn_key = f"analyze_{key}"
                        if st.button(f"Analyze with AI", key=btn_key):
                            with st.spinner(f"Analyzing {c['title']}..."):
                                analysis = analyze_chart_with_vision(
                                    c["b64"], c["title"], selected_model
                                )
                            with st.expander(f"AI Analysis: {c['title']}", expanded=True):
                                st.markdown(analysis)


    # TAB 2: AI ANALYST CHAT

    with tab2:
        st.subheader("Ask the Crime Analysis AI")
        st.caption(
            "The supervisor agent routes your question to the Data Retrieval Agent "
            "(LlamaCloud + dataset stats) or the Strategy Agent depending on what you ask."
        )

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask about crime patterns, trends, or strategy..."):
            if df is None:
                st.warning("Please upload a dataset first.")
            else:
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Routing to agent..."):
                        response = run_agent_sync(prompt)
                    st.markdown(response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )

        if st.session_state.messages:
            if st.button("Clear Chat"):
                st.session_state.messages = []
                st.rerun()


    # TAB 3: STRATEGY BRIEFING

    with tab3:
        st.subheader("Generate Full Weekly Strategy Briefing")
        st.caption(
            "Generates a command-ready briefing summary combining insights from all charts. "
            "This mirrors the workflow from the ISM 647 use case: "
            "charts in, strategic priorities out."
        )

        if df is None or charts is None:
            st.warning("Upload a dataset to generate a briefing.")
        else:
            col_left, col_right = st.columns([1, 2])

            with col_left:
                st.markdown("**Select charts to include**")
                selected_charts = {}
                for key, c in charts.items():
                    if st.checkbox(c["title"], value=True, key=f"chk_{key}"):
                        selected_charts[key] = c

                include_stats = st.checkbox("Include dataset summary stats", value=True)
                audience = st.selectbox(
                    "Briefing audience",
                    ["Command Staff", "City Council", "Detective Bureau"],
                )

            with col_right:
                if st.button("Generate Briefing", type="primary"):
                    if not selected_charts:
                        st.warning("Select at least one chart.")
                    else:
                        with st.spinner("Analyzing charts and generating briefing..."):
                            chart_insights = []
                            prog = st.progress(0)
                            for idx, (key, c) in enumerate(selected_charts.items()):
                                analysis = analyze_chart_with_vision(
                                    c["b64"], c["title"], selected_model
                                )
                                chart_insights.append(
                                    f"Chart: {c['title']}\n{analysis}"
                                )
                                prog.progress((idx + 1) / len(selected_charts))

                            stats_block = ""
                            if include_stats and _summary_cache:
                                stats_block = (
                                    f"\n\nDataset Overview: {_summary_cache.get('overview', '')}"
                                    f"\nTop Districts: {_summary_cache.get('district', '')}"
                                    f"\nTop Offenses: {_summary_cache.get('offense', '')}"
                                )

                            synthesis_prompt = (
                                f"You are a police crime analyst preparing a weekly briefing for {audience}. "
                                f"Below are AI-generated insights from this week's crime trend charts."
                                f"{stats_block}\n\n"
                                + "\n\n---\n\n".join(chart_insights)
                                + "\n\n---\n\n"
                                f"Based on all of the above, write a concise weekly strategy briefing for {audience}. "
                                f"Include: (1) a two-sentence executive summary of the week's crime landscape, "
                                f"(2) three numbered strategic priorities with specific justification, "
                                f"(3) one district that should receive focused patrol attention and why. "
                                f"Keep it under 400 words. Use direct, clear language."
                            )

                            final_response = openai_client.chat.completions.create(
                                model=MODEL_FLAGSHIP,
                                messages=[{"role": "user", "content": synthesis_prompt}],
                                max_completion_tokens=800,
                            )
                            briefing_text = final_response.choices[0].message.content

                        st.markdown("---")
                        st.markdown(f"### Weekly Strategy Briefing for {audience}")
                        st.markdown(briefing_text)

                        st.download_button(
                            label="Download Briefing (.txt)",
                            data=briefing_text,
                            file_name="weekly_crime_briefing.txt",
                            mime="text/plain",
                        )


if __name__ == "__main__":
    main()
