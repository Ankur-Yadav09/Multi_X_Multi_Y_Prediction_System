"""
app.py — Soft Sensor Prediction System
=======================================
Thin Streamlit entry point with integrated Enterprise What-If Dashboard.

Responsibilities
----------------
1. Configure the Streamlit page (title, layout, CSS theme)
2. Bootstrap the SQLite database schema
3. Initialise Streamlit session state
4. Render the sidebar navigation
5. Dispatch to the selected page module dynamically
"""

import io
import numpy as np
import pandas as pd
import streamlit as st

from src.data.database import init_db
from src.ui.layout import configure_page, render_sidebar
from src.ui.session import init_session_state
from whatif_runner import load_process_data, whatif_analysis
import base64

# ---- Page config & CSS theme (must be the first Streamlit call) ----
configure_page()

# ---- Bootstrap database schema ----
init_db()

# ---- Initialise session-state keys ----
init_session_state()

# ---- Sidebar navigation ----
selected = render_sidebar()


# ---- Integrated What-If Dashboard Core Module ----
@st.cache_data(show_spinner=False)
def get_cached_process_data():
    """
    Loads historical data from historian database and pre-computes 
    datetime parsing. Cached globally across all user sessions.
    """
    loaded_df = load_process_data()
    loaded_df.index = pd.to_datetime(loaded_df.index)
    return loaded_df


def inject_premium_styles():
    """Injects custom enterprise CSS themes into the application layout."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    .stApp {
        background: radial-gradient(circle at top right, #1e293b, #0f172a);
        color: #f8fafc;
        font-family: 'Inter', -apple-system, sans-serif;
    }

    h1 {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 34px !important;
        font-weight: 800 !important;
        text-align: left !important;
        padding-bottom: 2px;
        margin-bottom: 24px;
    }

    h2, h3, h4 {
        color: #f1f5f9 !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }

    label {
        color: #e2e8f0 !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        margin-bottom: 6px !important;
    }

    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

    section[data-testid="stSidebar"] * {
        color: #f1f5f9 !important;
    }

    section[data-testid="stSidebar"] h3 {
        color: #38bdf8 !important;
        font-size: 13px !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 15px;
    }

    button[data-baseweb="tab"] {
        font-size: 15px !important;
        font-weight: 600 !important;
        color: #94a3b8 !important;
        transition: all 0.2s ease;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom-color: #38bdf8 !important;
    }

    div[data-baseweb="select"] > div {
        background-color: #1e293b !important;
        color: #f8fafc !important;
        border-radius: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        min-height: 42px !important;
    }

    div[data-baseweb="select"] span {
        color: #f8fafc !important;
        font-weight: 500 !important;
    }

    .stTextInput input, .stNumberInput input {
        background-color: #1e293b !important;
        color: #f8fafc !important;
        border-radius: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        font-size: 14px !important;
        padding: 8px 12px !important;
    }

    .stTextInput input:focus, .stNumberInput input:focus {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 0 1px #38bdf8 !important;
    }

    [data-testid="stDataFrame"], .stDataEditor {
        border-radius: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
        background-color: #1e293b !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #1d4ed8) !important;
        color: #ffffff !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 10px 24px !important;
        width: 100%;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.25);
        transition: all 0.2s ease-in-out;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(59, 130, 246, 0.35);
        background: linear-gradient(135deg, #2563eb, #1e40af) !important;
    }

    .executive-card {
        background-color: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)


def init_dashboard_schemas():
    """Ensures fallback container matrix schemas are allocated into Streamlit state."""
    if "user_inputs_df" not in st.session_state:
        st.session_state["user_inputs_df"] = pd.DataFrame(
            columns=["Parameter", "Value", "Lower Limit", "Upper Limit", "Remark"]
        )
    if "model_details_df" not in st.session_state:
        st.session_state["model_details_df"] = pd.DataFrame(
            columns=["Predicted parameter"] + [f"Input parameter_{i}" for i in range(1, 9)]
        )
    if "constraints_df" not in st.session_state:
        st.session_state["constraints_df"] = pd.DataFrame(
            columns=["Parameter", "user input value", "Max value", "UOM", "Remark"]
        )
    if "display_order_df" not in st.session_state:
        st.session_state["display_order_df"] = pd.DataFrame(
            columns=["Sr.no", "Preferred columns"]
        )
    if "pi_names_df" not in st.session_state:
        st.session_state["pi_names_df"] = pd.DataFrame(
            columns=["Pi_tags", "Yanpet_OLF1_description", "Generalized Description", "Section"]
        )
    if "lbt_input_df" not in st.session_state:
        st.session_state["lbt_input_df"] = pd.DataFrame(
            columns=["Iteration No", "Match_tags", "Tolerance_minimum", "Tolerance_maximum", "Match_Tag_Decimal", "performance_tag", "direction"]
        )


def callback_sync_user_inputs():
    """Calculates and matches live boundaries for modified rows inside data editor contexts."""
    if "ui_editor_user_inputs" in st.session_state and st.session_state["ui_editor_user_inputs"]:
        delta = st.session_state["ui_editor_user_inputs"]
        df_base = st.session_state["user_inputs_df"].copy()
        df_source = get_cached_process_data()
        
        if delta.get("deleted_rows"):
            df_base = df_base.drop(delta["deleted_rows"]).reset_index(drop=True)
            
        if delta.get("added_rows"):
            for row in delta.get("added_rows"):
                new_row = {
                    "Parameter": row.get("Parameter"), 
                    "Value": row.get("Value"), 
                    "Lower Limit": None, "Upper Limit": None, "Remark": ""
                }
                df_base = pd.concat([df_base, pd.DataFrame([new_row])], ignore_index=True)
                
        if delta.get("edited_rows"):
            for idx, changes in delta.get("edited_rows").items():
                idx = int(idx)
                for col, val in changes.items():
                    df_base.at[idx, col] = val
        
        for idx, row in df_base.iterrows():
            tag = row.get("Parameter")
            if pd.notna(tag) and tag in df_source.columns:
                if pd.api.types.is_numeric_dtype(df_source[tag]):
                    p01 = round(float(df_source[tag].quantile(0.01)), 2)
                    p99 = round(float(df_source[tag].quantile(0.99)), 2)
                else:
                    p01, p99 = 0.0, 1000000.0
                
                df_base.at[idx, "Lower Limit"] = p01
                df_base.at[idx, "Upper Limit"] = p99
                df_base.at[idx, "Remark"] = "⚡ Live 1%-99% Pctl"
            else:
                df_base.at[idx, "Lower Limit"] = None
                df_base.at[idx, "Upper Limit"] = None
                df_base.at[idx, "Remark"] = ""
        
        st.session_state["user_inputs_df"] = df_base


def render_dashboard_workflow(focus_tab="📊 What-if Dashboard"):
    """
    Renders either the what if Configuration Hub structure or the analytics interface
    depending on sidebar navigation routing parameters.
    """
    inject_premium_styles()
    init_dashboard_schemas()
    
    df = get_cached_process_data()
    tag_options = sorted(df.columns.tolist())

    # =============================================================================
    # ROUTE: what if Configuration Hub
    # =============================================================================
    if focus_tab == "⚙️ what if Configuration Hub":
        st.title("YP OLF1 Configuration Framework")
        
        st.markdown('<div class="executive-card">', unsafe_allow_html=True)
        st.subheader("🎛️ Master Configuration Framework")
        st.markdown(
            """
            Fine-tune engineering boundaries, tag dictionaries, and model rules directly within the interactive matrices below. 
            You can also upload a completed **Config_file.xlsx** workbook to bulk-load all 6 sheets instantly.
            """
        )
        
        uploaded_config = st.file_uploader("📂 Import config matrix (Config_file.xlsx)", type=["xlsx"])
        if uploaded_config is not None:
            try:
                excel_file = pd.ExcelFile(uploaded_config)
                sheet_mappings = {
                    "user inputs": "user_inputs_df",
                    "Model details": "model_details_df",
                    "Constraints": "constraints_df",
                    "display_column_order": "display_order_df",
                    "PI_generalised_Name": "pi_names_df",
                    "lbt_input": "lbt_input_df"
                }
                for sheet, state_key in sheet_mappings.items():
                    if sheet in excel_file.sheet_names:
                        st.session_state[state_key] = pd.read_excel(uploaded_config, sheet_name=sheet)
                st.success("✅ Architecture matrix successfully compiled and parsed into system memory.")
            except Exception as e:
                st.error(f"⚠️ Initialization Error: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

        c_tab1, c_tab2, c_tab3, c_tab4, c_tab5, c_tab6 = st.tabs([
            "📋 1. User Input Table", "🧠 2. Model Mapping", "⚠️ 3. Constraints", 
            "📊 4. Reporting Order", "🏷️ 5. PI Tag Mapping", "🔍 6. LBT Input"
        ])
        
        with c_tab1:
            st.subheader("Scenario Inputs & Statistical Percentiles")
            input_col_config = {
                "Parameter": st.column_config.SelectboxColumn("Parameter", options=tag_options, required=True),
                "Value": st.column_config.NumberColumn("Value", default=None),
                "Lower Limit": st.column_config.NumberColumn("Lower Limit", disabled=True, format="%.2f"),
                "Upper Limit": st.column_config.NumberColumn("Upper Limit", disabled=True, format="%.2f"),
                "Remark": st.column_config.TextColumn("Remark", disabled=True)
            }
            st.session_state["user_inputs_df"] = st.data_editor(
                st.session_state["user_inputs_df"], column_config=input_col_config,
                use_container_width=True, num_rows="dynamic", key="ui_editor_user_inputs", on_change=callback_sync_user_inputs
            )

        with c_tab2:
            model_config = {col: st.column_config.SelectboxColumn(col, options=tag_options) for col in st.session_state["model_details_df"].columns}
            st.session_state["model_details_df"] = st.data_editor(st.session_state["model_details_df"], column_config=model_config, use_container_width=True, num_rows="dynamic")

        with c_tab3:
            bounds_config = {"Parameter": st.column_config.SelectboxColumn("Parameter", options=tag_options, required=True)} if "Parameter" in st.session_state["constraints_df"].columns else {}
            st.session_state["constraints_df"] = st.data_editor(st.session_state["constraints_df"], column_config=bounds_config, use_container_width=True, num_rows="dynamic")

        with c_tab4:
            order_config = {"Preferred columns": st.column_config.SelectboxColumn("Preferred columns", options=tag_options, required=True)} if "Preferred columns" in st.session_state["display_order_df"].columns else {}
            st.session_state["display_order_df"] = st.data_editor(st.session_state["display_order_df"], column_config=order_config, use_container_width=True, num_rows="dynamic")

        with c_tab5:
            st.session_state["pi_names_df"] = st.data_editor(st.session_state["pi_names_df"], use_container_width=True, num_rows="dynamic")

        with c_tab6:
            lbt_config = {col: st.column_config.SelectboxColumn(col, options=tag_options) for col in ["Match_tags", "performance_tag"] if col in st.session_state["lbt_input_df"].columns}
            st.session_state["lbt_input_df"] = st.data_editor(st.session_state["lbt_input_df"], column_config=lbt_config, use_container_width=True, num_rows="dynamic")
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Sync & Apply Changes to What-if Engine", use_container_width=True):
            st.session_state["applied_user_inputs_df"] = st.session_state["user_inputs_df"].copy()
            st.session_state["applied_display_order_df"] = st.session_state["display_order_df"].copy()
            st.toast("🎯 Configuration successfully locked to Simulation Dashboard!", icon="🎯")
            st.rerun()
        
        config_buffer = io.BytesIO()
        try:
            with pd.ExcelWriter(config_buffer, engine='openpyxl') as writer:
                st.session_state["user_inputs_df"].to_excel(writer, sheet_name="user inputs", index=False)
                st.session_state["model_details_df"].to_excel(writer, sheet_name="Model details", index=False)
                st.session_state["constraints_df"].to_excel(writer, sheet_name="Constraints", index=False)
                st.session_state["display_order_df"].to_excel(writer, sheet_name="display_column_order", index=False)
                st.session_state["pi_names_df"].to_excel(writer, sheet_name="PI_generalised_Name", index=False)
                st.session_state["lbt_input_df"].to_excel(writer, sheet_name="lbt_input", index=False)
            
            st.download_button(
                label="📥 Export config file", data=config_buffer.getvalue(), file_name="Config_file.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True
            )
        except Exception as export_error:
            st.error(f"Failed to generate compiled configuration artifact: {export_error}")

    # =============================================================================
    # ROUTE: WHAT-IF ANALYTICS DASHBOARD
    # =============================================================================
    elif focus_tab == "📊 What-if Dashboard":
        st.title("YP OLF1 What-if Analytics Dashboard")
        
        # user_inputs_df = st.session_state["applied_user_inputs_df"]
        # display_order_df = st.session_state["applied_display_order_df"]
        
        user_inputs_df = st.session_state.get("applied_user_inputs_df", pd.DataFrame())
        display_order_df = st.session_state.get("applied_display_order_df", pd.DataFrame())
        
        # -------------------------------------------------
        # PARAMETER DETERMINATION LAYER (CONFIG VS DROPDOWN FALLBACK)
        # -------------------------------------------------
        if not user_inputs_df.empty:
            # Scenario A: Configuration workbook uploaded successfully or custom items added manually via Tab 1
            df_limits = user_inputs_df.copy()
            if "Parameter" in df_limits.columns and "Lower Limit" in df_limits.columns and "Upper Limit" in df_limits.columns:
                df_limits["Parameter"] = df_limits["Parameter"].astype(str).str.strip()
                limits_dict = df_limits.set_index("Parameter")[["Lower Limit", "Upper Limit"]].to_dict("index")
                user_defined_input_tags = df_limits["Parameter"].dropna().tolist()
            else:
                st.error("⚠️ Invalid Matrix Format: Ensure column keys 'Parameter', 'Lower Limit', and 'Upper Limit' are active in Tab 1.")
                st.stop()
        else:
            # Scenario B: No configuration file uploaded - Activate full tag list dropdown fallback
            st.sidebar.markdown("### 🏷️ Dynamic Tag Selection")
            all_historian_tags = tag_options
            
            user_defined_input_tags = st.sidebar.multiselect(
                "User defined inputs",
                options=all_historian_tags,
                default=None,
                help="No configuration workbook uploaded. Select process tags from this complete database dropdown to simulate scenarios manually."
            )
            
            # Calculate limits on-the-fly using historical min/max values
            limits_dict = {}
            for tag in user_defined_input_tags:
                if pd.api.types.is_numeric_dtype(df[tag]):
                    limits_dict[tag] = {
                        "Lower Limit": float(df[tag].min()),
                        "Upper Limit": float(df[tag].max())
                    }
                else:
                    limits_dict[tag] = {"Lower Limit": 0.0, "Upper Limit": 1000000.0}

            if not user_defined_input_tags:
                st.info("💡 **Quick Start Guide:** Choose one or more operational metrics from the **'User defined inputs'** dropdown menu in the left sidebar to begin adjusting What-if conditions.")

        # Horizontal Controller Bar
        #st.markdown('<div class="executive-card">', unsafe_allow_html=True)
        sel_col1, sel_col2 = st.columns(2)
        with sel_col1:
            available_dates = sorted(pd.Series(df.index.date).unique(), reverse=False)
            selected_date = st.selectbox("Historical Target Date", available_dates)
        with sel_col2:
            filtered_timestamps = df.index[df.index.date == selected_date]
            filtered_timestamps = sorted(filtered_timestamps, reverse=False)
            timestamp_options = [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in filtered_timestamps]
            selected_time_str = st.selectbox("Process Snapshot Timestamp", timestamp_options, key="timestamp_selector")
        
        selected_time = pd.Timestamp(selected_time_str)
        st.markdown('</div>', unsafe_allow_html=True)

        # Process Baseline Display
        with st.expander("🔎 Baseline Process Values at Selected Timestamp", expanded=False):
            try:
                active_tags = [tag for tag in user_defined_input_tags if tag in df.columns] if user_defined_input_tags else df.columns.tolist()
                if active_tags:
                    selected_row = df.loc[[selected_time], active_tags].T
                    selected_row.columns = ["Current Value"]
                    selected_row = selected_row.round(2)
                    st.dataframe(selected_row, use_container_width=True, height=280)
                else:
                    st.caption("No metrics selected. Select parameters from the sidebar to isolate rows.")
            except KeyError:
                st.error("⚠️ Selected snapshot index mismatch.")
                
        # Sidebar Input Overrides Rendering Block
        # st.sidebar.markdown("### 🔧 Simulation Overrides")
        user_inputs = []

        for param in user_defined_input_tags:
            limits = limits_dict.get(param)
            if limits is None:
                continue

            lower = limits.get("Lower Limit", 0.0)
            upper = limits.get("Upper Limit", 1000000.0)

            st.sidebar.markdown(f"**{param}**")
            st.sidebar.caption(f"Historical Boundary Range: {lower:.2f} to {upper:.2f}")
            
            val = st.sidebar.text_input(
                "Override target value",
                value="",
                key=param,
                placeholder=""
            )

            val_float = np.nan
            if val:
                try:
                    val_float = float(val)
                    if not (lower <= val_float <= upper):
                        st.sidebar.error(f"Value must fall between {lower:.2f} and {upper:.2f}")
                        val_float = np.nan
                except ValueError:
                    st.sidebar.error("Numeric input required")
                    val_float = np.nan

            user_inputs.append({
                "Parameter": param,
                "Value": val_float
            })

        user_input_df = pd.DataFrame(user_inputs, columns=["Parameter", "Value"])

        if st.button("Compute What-If Scenario Analysis"):
            with st.spinner("Processing operational scenario model rules..."):
                st.session_state["result"] = whatif_analysis(df, selected_time, user_input_df)
                st.session_state["selected_time"] = selected_time

        # Analytical Output Section
        if "result" in st.session_state:
            result = st.session_state["result"]
            selected_time_download = st.session_state["selected_time"]

            
            if isinstance(result, pd.io.formats.style.Styler):
                result_df = result.data.copy()
            else:
                result_df = result.copy()

            result_df.drop(columns=["Timestamp"], errors="ignore", inplace=True)

            if "actual" in result_df.index and "estimated" in result_df.index:
                result_df.loc["Change"] = (
                    pd.to_numeric(result_df.loc["estimated"], errors='coerce') - 
                    pd.to_numeric(result_df.loc["actual"], errors='coerce')
                )
             
            # def dynamic_format(val):
            #     try:
            #         val_float = float(val)
            #         if abs(val_float) >= 1000:
            #             return f"{val_float:.0f}"  # Zero decimals for >= 1000
            #         else:
            #             return f"{val_float:.1f}"  # One decimal for < 1000
            #     except (ValueError, TypeError):
            #         return val  # Return unchanged if it's a string/tag name

            # # Apply formatting directly to display targets
            # result_df['actual'] = result_df['actual'].apply(dynamic_format)
            # result_df['estimated'] = result_df['estimated'].apply(dynamic_format)
            # result_df['Change'] = result_df['Change'].apply(dynamic_format)
                    
            transpose_df = result_df.T
            transpose_df = transpose_df[~transpose_df.index.duplicated(keep='first')]

            # Sorting logic configuration handling fallback safely if ordering rule dataframe is empty
            preferred_order = display_order_df["Preferred columns"].dropna().tolist() if ("Preferred columns" in display_order_df.columns and not display_order_df.empty) else []
            preferred_existing = [col for col in preferred_order if col in transpose_df.index]
            remaining_cols = [col for col in transpose_df.index if col not in preferred_existing]
            final_order = preferred_existing + remaining_cols

            transpose_df = transpose_df.loc[final_order]
            
            
            # -----------------------------------------------------------------
            # 🌟 HIGH-VISIBILITY EXECUTIVE KPI TILE BANNER
            # -----------------------------------------------------------------
            kpi_tags_to_track = [
                'DMCTF_feed', 'Quench_tower_overhead_temp', 'CGC_TURBINE_1_SPEED_(RPM)', 
                'CGC_STAGE_1_SUCTION_PRESSURE', 'CGC_Power_KW', 'PRC_turbine_RPM', 'PRC_1ST_STAGE_Suction_PRESSURE',
                'PRC_Total_estimated_power_MW', 'ERC_power', 'ERC_1ST_STAGE_Suction_PRESSURE', 'ERC_turbine_Speed', 
                'Total_Power_(KW)', 'Total_required_steam_flow_(TPH)','Ethylene_product_flow'
            ]   
            active_kpis = [tag for tag in kpi_tags_to_track if tag in transpose_df.index]
            
            if active_kpis:
                st.markdown("#### 📊 Key Performance Indicators")
                
                # Dynamically break elements into dense rows of 4 columns each
                for chunk in [active_kpis[i:i + 4] for i in range(0, len(active_kpis), 4)]:
                    tile_cols = st.columns(len(chunk))
                    for col_idx, kpi_tag in enumerate(chunk):
                        with tile_cols[col_idx]:
                            try:
                                act_val = float(transpose_df.at[kpi_tag, "actual"])
                                est_val = float(transpose_df.at[kpi_tag, "estimated"])
                                chg_val = float(transpose_df.at[kpi_tag, "Change"])
                                
                                # 🎯 DYNAMIC DECIMAL ROUNDING RULE (UI DISPLAY STRINGS)
                                act_str = f"{act_val:.0f}" if abs(act_val) >= 1000 else f"{act_val:.1f}"
                                est_str = f"{est_val:.0f}" if abs(est_val) >= 1000 else f"{est_val:.1f}"
                                chg_str = f"{chg_val:+.0f}" if abs(chg_val) >= 1000 else f"{chg_val:+.1f}"

                                # 🎯 BUSINESS RULE COLOR DETERMINATION
                                if est_val < act_val:
                                    border_color = "#ef4444"      # Red border
                                    delta_bg = "rgba(239, 68, 68, 0.15)"
                                    delta_text = "#f87171"
                                elif est_val > act_val:
                                    border_color = "#10b981"      # Green border
                                    delta_bg = "rgba(16, 185, 129, 0.15)"
                                    delta_text = "#34d399"
                                else:
                                    border_color = "rgba(255,255,255,0.1)" # Gray border
                                    delta_bg = "rgba(148, 163, 184, 0.1)"
                                    delta_text = "#94a3b8"

                                clean_label = kpi_tag.replace('_', ' ')

                                st.markdown(f"""
                                <div style="
                                    background: linear-gradient(135deg, #1e293b, #0f172a);
                                    border-left: 4px solid {border_color};
                                    border-top: 1px solid rgba(255,255,255,0.05);
                                    border-right: 1px solid rgba(255,255,255,0.05);
                                    border-bottom: 1px solid rgba(255,255,255,0.05);
                                    border-radius: 8px;
                                    padding: 16px;
                                    margin-bottom: 14px;
                                    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                                ">
                                    <div style="color: #94a3b8; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.03em; margin-bottom: 6px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="{kpi_tag}">
                                        🏷️ {clean_label}
                                    </div>
                                    <div style="color: #f8fafc; font-size: 26px; font-weight: 800; line-height: 1.2; margin-bottom: 8px;">
                                        {est_str}
                                    </div>
                                    <div style="
                                        display: inline-block;
                                        padding: 3px 8px;
                                        border-radius: 4px;
                                        font-size: 16px;
                                        font-weight: 700;
                                        background-color: {delta_bg};
                                        color: {delta_text};
                                    ">
                                        {chg_str} vs Act ({act_str})
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                            except Exception:
                                pass
                # st.markdown("<br>", unsafe_allow_html=True)
            #-----------------------------------------------------------------
            #<br><h3>
            
            st.markdown('<br><h3>📈 Evaluated Actual vs Estimated Scenario Output</h3>', unsafe_allow_html=True)
            
            transpose_df = transpose_df.map(
                lambda x: f"{x:.2f}".rstrip('0').rstrip('.') if isinstance(x, (int, float, np.number)) else x
            )
            
            # Premium Muted Highlighting Rules for Clean Enterprise Look
            def highlight_change_only(row):
                styles = []
                for col in row.index:
                    if col.lower() == "change":
                        try:
                            v = float(row[col])
                        except:
                            styles.append("")
                            continue
                        if v < 0:
                            styles.append("color: #ef4444; font-weight: 600; background-color: rgba(239, 68, 68, 0.08);")
                        elif v > 0:
                            styles.append("color: #10b981; font-weight: 600; background-color: rgba(16, 185, 129, 0.08);")
                        else:
                            styles.append("color: #94a3b8; font-weight: 500;")
                    else:
                        styles.append("")
                return styles
                
            styled_df = transpose_df.style.apply(highlight_change_only, axis=1)
            st.dataframe(styled_df, use_container_width=True, height=480)

            # Elegant Downloader Block
            download_df = transpose_df.copy().reset_index()
            download_df.rename(columns={"index": "Parameter"}, inplace=True)
            download_df.insert(0, "Selected Timestamp", str(selected_time_download))
            
            csv_data = download_df.to_csv(index=True)
            b64 = base64.b64encode(csv_data.encode()).decode()
            
            download_button_html = f"""
            <a href="data:file/csv;base64,{b64}"
            download="WhatIf_Result.csv"
            style="
            display:inline-block;
            width:100%;
            padding:12px 20px;
            font-size:15px;
            font-weight:600;
            text-align:center;
            color:white;
            background:linear-gradient(135deg, #10b981, #059669);
            border-radius:8px;
            text-decoration:none;
            margin-top:12px;
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.2);
            transition: 0.2s ease;
            ">
            📥 Export Baseline vs Simulation Matrix (.CSV)
            </a>
            """
            st.markdown(download_button_html, unsafe_allow_html=True)

            # -------------------------------------------------
            # COMPONENT VALIDATION LAYER
            # -------------------------------------------------
            st.sidebar.markdown("### 🔍 Validation Filters")
            
            target_tags = [
                'DMCTF_feed', 'Quench_tower_overhead_temp', 'Fresh_ethane_feed', 
                'fresh_feed_ethane_content', 'CGC_STAGE_1_SUCTION_PRESSURE'
            ]
            available_tags = [col for col in target_tags if col in df.columns]
            
            if not available_tags:
                st.error("Target analysis validation tags are missing from the primary database index keys.")
            else:
                filter_values = {}
                for tag in available_tags:
                    if pd.api.types.is_numeric_dtype(df[tag]):
                        global_min = float(df[tag].min())
                        global_max = float(df[tag].max())
                        
                        st.sidebar.markdown(f"**Filter {tag}**")
                        col1, col2 = st.sidebar.columns(2)
                        with col1:
                            min_val = st.number_input(f"Min", value=global_min, key=f"{tag}_min")
                        with col2:
                            max_val = st.number_input(f"Max", value=global_max, key=f"{tag}_max")
                        filter_values[tag] = (min_val, max_val)
                    else:
                        unique_vals = df[tag].unique()
                        selected_vals = st.sidebar.multiselect(f"Select {tag}", options=unique_vals, default=unique_vals)
                        filter_values[tag] = selected_vals
            
                df_filtered = df.copy()
                for tag, criteria in filter_values.items():
                    if pd.api.types.is_numeric_dtype(df_filtered[tag]):
                        min_v, max_v = criteria
                        df_filtered = df_filtered[(df_filtered[tag] >= min_v) & (df_filtered[tag] <= max_v)]
                    else:
                        df_filtered = df_filtered[df_filtered[tag].isin(criteria)]
               
                preferred_existing = [col for col in preferred_order if col in df_filtered.columns]
                remaining_cols = [col for col in df_filtered.columns if col not in preferred_existing]
                final_order = preferred_existing + remaining_cols
                df_filtered = df_filtered[final_order]
            
                df_filtered = df_filtered.map(
                    lambda x: f"{x:.2f}".rstrip('0').rstrip('.') if isinstance(x, (int, float, np.number)) else x
                )
                
                df_filtered = df_filtered.T
                df_filtered.index.name = "Parameter"
                
                st.markdown('<br><h3>🔍 Correlated Historical Validation Sets</h3>', unsafe_allow_html=True)
                st.dataframe(df_filtered, use_container_width=True)
                
                combined_df = pd.merge(download_df, df_filtered, on="Parameter", how="inner")
                csv = combined_df.to_csv().encode('utf-8')
                
                st.download_button(
                    label="📥 Export Unified Comparison & Historical Validation Data (.CSV)",
                    data=csv,
                    file_name='filtered_validation_data.csv',
                    mime='text/csv',
                )


# ---- Page dispatch System ----
from src.ui.pages import (  # noqa: E402
    feature_selection,
    predict,
    preprocess,
    train,
    upload,
)

# Wrapper dispatches pointing to the unified workflow block
def render_configuration_hub():
    render_dashboard_workflow(focus_tab="⚙️ what if Configuration Hub")

def render_what_if_dashboard_only():
    render_dashboard_workflow(focus_tab="📊 What-if Dashboard")


_PAGE_MAP = {
    "Upload Data":       upload.render,
    "Preprocessing":     preprocess.render,
    "Feature Selection": feature_selection.render,
    "Train Model":       train.render,
    "Predict":           predict.render,
    "what if Configuration Hub": render_configuration_hub,        # Bound smoothly to your sidebar choice
    "What-If Dashboard": render_what_if_dashboard_only,   # Bound smoothly to your sidebar choice
}

# Run the selected page function
_PAGE_MAP[selected]()