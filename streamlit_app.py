"""
Mia Training Tracker - Streamlit Web App
Soccer training analytics and tracking application
Accessible from any browser including mobile devices
"""

import streamlit as st

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Mia Training Tracker",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

import pandas as pd
import numpy as np
from datetime import datetime
from PIL import Image
import io
import re
import os

# Import matplotlib with proper backend for Streamlit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import pytesseract and configure for Streamlit Cloud
try:
    import pytesseract

    # Configure Tesseract path for different environments
    # Streamlit Cloud uses /usr/bin/tesseract
    if os.path.exists('/usr/bin/tesseract'):
        pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
    elif os.path.exists('/opt/homebrew/bin/tesseract'):
        pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

    # Test if tesseract is working
    try:
        pytesseract.get_tesseract_version()
        OCR_AVAILABLE = True
        OCR_ERROR = None
    except Exception as e:
        OCR_AVAILABLE = False
        OCR_ERROR = "Tesseract not found"
except ImportError:
    OCR_AVAILABLE = False
    OCR_ERROR = "pytesseract module not installed"

# Google Sheets Integration
try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSHEETS_AVAILABLE = True
except ImportError:
    GSHEETS_AVAILABLE = False

# Google Sheets functions
def connect_to_google_sheets():
    """Connect to Google Sheets using service account credentials from Streamlit secrets"""
    try:
        if not GSHEETS_AVAILABLE:
            return None, "Google Sheets libraries not installed"

        # Get credentials from Streamlit secrets
        if "gcp_service_account" not in st.secrets:
            return None, "Google Sheets credentials not configured. See setup guide."

        # Define the required scopes
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]

        # Create credentials from the secrets
        credentials = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=scopes
        )

        # Authorize and return the client
        client = gspread.authorize(credentials)
        return client, None

    except Exception as e:
        return None, f"Error connecting to Google Sheets: {str(e)}"

def load_data_from_google_sheets():
    """Load training data from Google Sheets"""
    try:
        # Check if Google Sheets is configured
        if "google_sheets_url" not in st.secrets:
            return None, "Google Sheets URL not configured"

        # Connect to Google Sheets
        client, error = connect_to_google_sheets()
        if error:
            return None, error

        # Open the spreadsheet by URL
        sheet_url = st.secrets["google_sheets_url"]
        spreadsheet = client.open_by_url(sheet_url)

        # Get the first worksheet (or specify by name)
        worksheet = spreadsheet.sheet1

        # Get all values and convert to DataFrame
        data = worksheet.get_all_values()

        if len(data) < 2:
            return None, "No data found in Google Sheet"

        # First row is headers
        headers = data[0]
        rows = data[1:]

        # Create DataFrame
        df = pd.DataFrame(rows, columns=headers)

        # Normalize column names to lowercase with underscores for consistency
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

        # Apply the same column mapping as Excel load (for internal consistency)
        # This maps your Google Sheet column names to the internal names the app uses
        column_mapping = {
            'top_speed_mph': 'top_speed',
            'sprint_distance_yd': 'sprint_distance',
            'total_distance_mi': 'total_distance',
            'duration_min': 'duration',
            'kicking_power_mph': 'kicking_power_mph',
            'avg_turn_entry_speed_mph': 'avg_turn_entry',
            'avg_turn_exit_speed_mph': 'avg_turn_exit',
            'sprints': 'num_sprints',
            'accl_decl': 'accelerations',
        }
        df.rename(columns=column_mapping, inplace=True)

        # Convert date column to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Convert numeric columns (using internal mapped names)
        numeric_columns = [
            'duration', 'ball_touches', 'total_distance', 'sprint_distance',
            'accelerations', 'kicking_power_mph', 'top_speed', 'num_sprints',
            'left_touches', 'right_touches', 'left_foot_pct',
            'left_releases', 'right_releases',
            'left_kicking_power_mph', 'right_kicking_power_mph',
            'left_turns', 'back_turns', 'right_turns', 'intense_turns',
            'avg_turn_entry', 'avg_turn_exit', 'total_turns', 'work_rate',
            'goals', 'assists', 'ball_possessions'
        ]

        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df, None

    except Exception as e:
        return None, f"Error loading from Google Sheets: {str(e)}"

def save_data_to_google_sheets(df):
    """Save training data to Google Sheets"""
    try:
        # Connect to Google Sheets
        client, error = connect_to_google_sheets()
        if error:
            return False, error

        # Open the spreadsheet
        sheet_url = st.secrets["google_sheets_url"]
        spreadsheet = client.open_by_url(sheet_url)
        worksheet = spreadsheet.sheet1

        # Clear existing data
        worksheet.clear()

        # Prepare data for upload
        # Convert DataFrame to list of lists
        data = [df.columns.tolist()] + df.astype(str).values.tolist()

        # Update the sheet
        worksheet.update('A1', data)

        return True, None

    except Exception as e:
        return False, f"Error saving to Google Sheets: {str(e)}"

def append_row_to_google_sheets(row_data):
    """Append a single row to Google Sheets"""
    try:
        # Connect to Google Sheets
        client, error = connect_to_google_sheets()
        if error:
            return False, error

        # Open the spreadsheet
        sheet_url = st.secrets["google_sheets_url"]
        spreadsheet = client.open_by_url(sheet_url)
        worksheet = spreadsheet.sheet1

        # Append the row
        worksheet.append_row(row_data)

        return True, None

    except Exception as e:
        return False, f"Error appending to Google Sheets: {str(e)}"

# Custom CSS for mobile responsiveness and hide sidebar
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    /* Hide sidebar completely */
    [data-testid="stSidebar"] {
        display: none;
    }
    [data-testid="collapsedControl"] {
        display: none;
    }
    /* Adjust main content to use full width */
    .main .block-container {
        max-width: 100%;
        padding-left: 2rem;
        padding-right: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'personal_records' not in st.session_state:
    st.session_state.personal_records = {}
if 'pr_dates' not in st.session_state:
    st.session_state.pr_dates = {}
if 'uploaded_images' not in st.session_state:
    st.session_state.uploaded_images = []
if 'initial_load_done' not in st.session_state:
    st.session_state.initial_load_done = False

# Column mapping for Excel data
COLUMN_MAPPING = {
    # Session Info
    'date': 'date',
    'session_name': 'session_name',
    'coach': 'coach',
    'location': 'location',
    'surface': 'surface',
    'with_ball': 'with_ball',
    'training_type': 'training_type',
    'duration_min': 'duration',
    'intensity': 'intensity',
    # Movement Metrics
    'total_distance_mi': 'total_distance',
    'sprint_distance_yd': 'sprint_distance',
    'accl_decl': 'accelerations',
    'top_speed_mph': 'top_speed',
    'sprints': 'num_sprints',
    # Agility
    'left_turns': 'left_turns',
    'back_turns': 'back_turns',
    'right_turns': 'right_turns',
    'intense_turns': 'intense_turns',
    'total_turns': 'total_turns',
    'avg_turn_entry_speed_mph': 'avg_turn_entry',
    'avg_turn_exit_speed_mph': 'avg_turn_exit',
    # Ball Work
    'ball_touches': 'ball_touches',
    'left_touches': 'left_touches',
    'right_touches': 'right_touches',
    'left_foot_pct': 'left_foot_pct',
    'left_releases': 'left_releases',
    'right_releases': 'right_releases',
    'kicking_power_mph': 'kicking_power_mph',
    'left_kicking_power_mph': 'left_kicking_power_mph',
    'right_kicking_power_mph': 'right_kicking_power_mph',
    # Match Stats
    'position': 'position',
    'goals': 'goals',
    'assists': 'assists',
    'work_rate': 'work_rate',
    'ball_possessions': 'ball_possessions',
}

def load_excel_file(uploaded_file):
    """Load and process Excel file"""
    try:
        df = pd.read_excel(uploaded_file)

        # Normalize column names
        df.columns = df.columns.str.strip().str.lower()

        # Apply column mapping
        df.rename(columns=COLUMN_MAPPING, inplace=True)

        # Convert date column to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

        st.session_state.df = df
        calculate_personal_records()

        return True, f"✅ Loaded {len(df)} sessions successfully!"
    except Exception as e:
        return False, f"❌ Error loading file: {str(e)}"

def calculate_personal_records():
    """Calculate all personal records from loaded data"""
    if st.session_state.df is None or len(st.session_state.df) == 0:
        return

    df = st.session_state.df

    pr_columns = {
        'top_speed': 'Top Speed',
        'sprint_distance': 'Sprint Distance',
        'ball_touches': 'Ball Touches',
        'kicking_power': 'Kicking Power',
        'total_distance': 'Total Distance',
        'intense_turns': 'Intense Turns'
    }

    personal_records = {}
    pr_dates = {}

    for col, name in pr_columns.items():
        if col in df.columns:
            numeric_col = pd.to_numeric(df[col], errors='coerce')
            max_val = numeric_col.max()
            if pd.notna(max_val):
                personal_records[col] = max_val
                max_idx = numeric_col.idxmax()
                if 'date' in df.columns and pd.notna(df.loc[max_idx, 'date']):
                    pr_dates[col] = pd.to_datetime(df.loc[max_idx, 'date'])
                else:
                    pr_dates[col] = None

    # L/R Touch Ratio
    if 'left_touches' in df.columns and 'right_touches' in df.columns:
        best_ratio = None
        best_distance = float('inf')
        best_date = None

        for idx, row in df.iterrows():
            left = pd.to_numeric(row.get('left_touches'), errors='coerce')
            right = pd.to_numeric(row.get('right_touches'), errors='coerce')

            if pd.notna(left) and pd.notna(right) and left > 0 and right > 0:
                ratio = left / right

                if ratio >= 0.5:
                    distance = abs(ratio - 0.5)
                else:
                    distance = 0.5 - ratio

                if distance < best_distance:
                    best_distance = distance
                    best_ratio = ratio
                    if 'date' in row and pd.notna(row['date']):
                        best_date = pd.to_datetime(row['date'])

        personal_records['left_right_ratio'] = best_ratio if best_ratio else 0.0
        pr_dates['left_right_ratio'] = best_date

    # L/R Ratio Average
    if 'left_touches' in df.columns and 'right_touches' in df.columns:
        left = pd.to_numeric(df['left_touches'], errors='coerce')
        right = pd.to_numeric(df['right_touches'], errors='coerce')
        valid_mask = (left > 0) & (right > 0)

        if valid_mask.any():
            ratios = left[valid_mask] / right[valid_mask]
            personal_records['left_right_ratio_avg'] = ratios.mean()
        else:
            personal_records['left_right_ratio_avg'] = 0.0

        pr_dates['left_right_ratio_avg'] = None

    st.session_state.personal_records = personal_records
    st.session_state.pr_dates = pr_dates

def parse_ocr_text(text):
    """Parse OCR text to extract metrics - enhanced version from v4.2"""
    extracted = {}

    # Common patterns - more flexible to handle OCR variations
    patterns = {
        # Session info
        'duration': r'(\d+)\s*min',
        'training_type': r'(technical|speed|agility|conditioning)',
        'intensity': r'(moderate|light|intense|hard)',

        # Movement metrics
        'total_distance': r'(?:total\s+)?distance[:\s]*(\d+\.?\d*)\s*mi',
        'sprint_distance': r'sprint\s+distance[:\s]*(\d+\.?\d*)\s*(?:yd|yards)',
        'top_speed': r'top\s+speed[:\s]*(\d+\.?\d*)\s*mph',
        'num_sprints': r'sprints?\s*[:\s#]*(\d+)',
        'accelerations': r'(?:accl?|accel(?:eration)?s?)\s*[:\s/]+\s*(?:decl?)?\s*(\d+)',

        # Ball work
        'ball_touches': r'(?:ball\s+)?touches\s*[:\s#]*(\d+)',

        # Kicking power
        'kicking_power': r'(?:kicking\s+)?power\s*[:\s]*(\d+\.?\d*)\s*mph',

        # Agility
        'left_turns': r'left\s+turns?\s*[:\s]*(\d+)',
        'right_turns': r'right\s+turns?\s*[:\s]*(\d+)',
        'back_turns': r'back\s+turns?\s*[:\s]*(\d+)',
        'intense_turns': r'intense\s+turns?\s*[:\s#]*(\d+)',
        'avg_turn_entry': r'(?:average\s+)?turn\s+entry\s+speed\s*[:\s]*(\d+\.?\d*)\s*mph',
        'avg_turn_exit': r'(?:average\s+)?turn\s+exit\s+speed\s*[:\s]*(\d+\.?\d*)\s*mph',
    }

    text_lower = text.lower()

    for key, pattern in patterns.items():
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            extracted[key] = match.group(1)

    # Special handling for left/right kicking power
    lines = text_lower.split('\n')
    for i, line in enumerate(lines):
        if 'kicking power' in line and i > 0:
            context = '\n'.join(lines[max(0, i-3):min(len(lines), i+4)])
            power_matches = re.findall(r'(\d+\.?\d*)\s*mph', context)
            if len(power_matches) >= 2:
                extracted['left_kicking_power_mph'] = power_matches[0]
                extracted['right_kicking_power_mph'] = power_matches[1]
                break

    # Touch percentages and counts
    touch_pattern = re.search(r'(\d+)\s*\((\d+)%\)[^\d]*touch[^\d]*(\d+)\s*\((\d+)%\)', text_lower, re.IGNORECASE)
    if touch_pattern:
        extracted['left_touches'] = touch_pattern.group(1)
        extracted['left_pct'] = touch_pattern.group(2)
        extracted['right_touches'] = touch_pattern.group(3)
        extracted['right_pct'] = touch_pattern.group(4)

    # Aggressive turn detection
    if 'left_turns' not in extracted:
        for pattern in [r'(\d+)\s*[^\w\d]*left\s+turns?', r'left\s+turns?\s*[^\w\d]*(\d+)']:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                extracted['left_turns'] = match.group(1)
                break

    if 'right_turns' not in extracted:
        for pattern in [r'(\d+)\s*[^\w\d]*right\s+turns?', r'right\s+turns?\s*[^\w\d]*(\d+)']:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                extracted['right_turns'] = match.group(1)
                break

    if 'back_turns' not in extracted:
        for pattern in [r'(\d+)\s*[^\w\d]*back\s+turns?', r'back\s+turns?\s*[^\w\d]*(\d+)']:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                extracted['back_turns'] = match.group(1)
                break

    # Ultimate fallback for back turns - process of elimination
    if 'back_turns' not in extracted and 'left_turns' in extracted and 'right_turns' in extracted:
        agility_match = re.search(r'agility', text_lower, re.IGNORECASE)
        if agility_match:
            agility_section = text_lower[agility_match.end():agility_match.end()+500]
            numbers = re.findall(r'\b(\d{2,3})\b', agility_section)
            if len(numbers) >= 3:
                for num in numbers:
                    if num != extracted.get('left_turns') and num != extracted.get('right_turns'):
                        num_val = int(num)
                        if 20 <= num_val <= 150:
                            extracted['back_turns'] = num
                            break

    return extracted

def generate_executive_summary(df):
    """Generate a 500-word executive summary of Mia's current status"""
    summary = "📋 EXECUTIVE SUMMARY\n" + "-" * 80 + "\n"

    # Collect detailed metrics for comprehensive summary
    sessions = len(df)

    # Date range
    date_info = ""
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        days_span = (df['date'].max() - df['date'].min()).days
        date_info = f" over {days_span} days"

    # AGILITY ANALYSIS (Most Important)
    agility_status = ""
    intense_current = 0
    intense_best = 0
    intense_trend = ""
    if 'intense_turns' in df.columns:
        intense_vals = pd.to_numeric(df['intense_turns'], errors='coerce').dropna()
        if len(intense_vals) > 0:
            intense_current = intense_vals.tail(3).mean()
            intense_best = intense_vals.max()
            overall_avg = intense_vals.mean()

            if len(intense_vals) > 3:
                early_avg = intense_vals.head(3).mean()
                recent_avg = intense_vals.tail(3).mean()
                if recent_avg > early_avg * 1.2:
                    intense_trend = "significant improvement"
                elif recent_avg > early_avg:
                    intense_trend = "steady growth"
                else:
                    intense_trend = "stable performance"

            if intense_current >= 10:
                agility_status = "elite-level agility with consistent high-speed direction changes"
            elif intense_current >= 7:
                agility_status = "strong agility development, approaching elite performance"
            elif intense_current >= 5:
                agility_status = "solid agility foundation with room to reach elite levels"
            else:
                agility_status = "early-stage agility development with significant growth opportunity"

    # SPEED & POWER ANALYSIS
    speed_status = ""
    speed_current = 0
    speed_best = 0
    if 'top_speed' in df.columns:
        speed_vals = pd.to_numeric(df['top_speed'], errors='coerce').dropna()
        if len(speed_vals) > 0:
            speed_current = speed_vals.tail(3).mean()
            speed_best = speed_vals.max()

            if speed_current >= speed_best * 0.95:
                speed_status = "performing at peak speed"
            elif speed_current >= speed_vals.mean() * 1.05:
                speed_status = "showing recent speed gains"
            else:
                speed_status = "maintaining speed development"

    # TURN SPEED DYNAMICS
    acceleration_status = ""
    if 'avg_turn_entry' in df.columns and 'avg_turn_exit' in df.columns:
        entry = pd.to_numeric(df['avg_turn_entry'], errors='coerce').dropna()
        exit_speed = pd.to_numeric(df['avg_turn_exit'], errors='coerce').dropna()

        if len(entry) > 0 and len(exit_speed) > 0:
            avg_entry = entry.mean()
            avg_exit = exit_speed.mean()
            speed_diff = avg_exit - avg_entry

            if speed_diff > 0.5:
                acceleration_status = "She shows explosive acceleration out of cuts, a critical game-speed skill"
            elif speed_diff > 0:
                acceleration_status = "She maintains speed through turns with slight acceleration"
            else:
                acceleration_status = "Her turn exit speed presents an opportunity for explosive power development"

    # BUILD COMPREHENSIVE NARRATIVE
    summary += f"Based on analysis of {sessions} training sessions{date_info}, "

    # Overall status
    if agility_status:
        summary += f"Mia demonstrates {agility_status}. "
        if intense_trend:
            summary += f"Her intense turns show {intense_trend}, currently averaging {intense_current:.1f} per session with a best of {intense_best:.0f}. "

    # Speed
    if speed_status:
        summary += f"In terms of speed development, she is {speed_status}, with recent sessions averaging {speed_current:.1f} mph against a personal best of {speed_best:.1f} mph. "

    # Turn dynamics
    if acceleration_status:
        summary += f"{acceleration_status}. "

    # PRIMARY FOCUS AREAS
    summary += "\n\nPRIMARY FOCUS AREAS: "
    focus_items = []

    if intense_current < 10:
        focus_items.append(f"Increase intense turns from {intense_current:.1f} to 10+ per session through high-speed cutting drills and small-sided games")

    if acceleration_status and "opportunity" in acceleration_status:
        focus_items.append("Develop explosive acceleration out of turns with plyometrics and first-step quickness drills")

    if not focus_items:
        focus_items.append("Continue balanced development while maintaining current momentum across all metrics")

    for i, item in enumerate(focus_items[:2], 1):
        summary += f"\n  {i}. {item}"

    summary += "\n\n"
    return summary

def generate_30day_change_summary(df):
    """Generate a 500-word narrative summary analyzing trends over the past 30 days"""
    summary = "📅 30-DAY CHANGE SUMMARY\n" + "-" * 80 + "\n"

    # Ensure date column exists and is datetime
    if 'date' not in df.columns:
        summary += "Date information not available for 30-day analysis.\n\n"
        return summary

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df = df.sort_values('date')

    if len(df) == 0:
        summary += "No date data available for analysis.\n\n"
        return summary

    # Get the most recent date
    latest_date = df['date'].max()

    # Calculate 30 days ago
    thirty_days_ago = latest_date - pd.Timedelta(days=30)

    # Get all sessions in the last 30 days
    last_30_days = df[df['date'] > thirty_days_ago].copy()

    # Get sessions from before 30 days ago for comparison
    before_30_days = df[df['date'] <= thirty_days_ago].copy()

    if len(last_30_days) == 0:
        summary += f"No sessions recorded in the past 30 days from {latest_date.strftime('%b %d, %Y')}.\n\n"
        return summary

    sessions_count = len(last_30_days)
    date_range_start = last_30_days['date'].min()
    date_range_end = last_30_days['date'].max()
    actual_days = (date_range_end - date_range_start).days + 1

    # Start narrative
    summary += f"Over the past 30 days ({date_range_start.strftime('%b %d')} - {date_range_end.strftime('%b %d, %Y')}), "
    summary += f"Mia completed {sessions_count} training session{'s' if sessions_count != 1 else ''} "
    summary += f"spanning {actual_days} day{'s' if actual_days != 1 else ''}. "

    # AGILITY TRENDS
    agility_narrative = ""
    if 'intense_turns' in df.columns:
        recent_intense = pd.to_numeric(last_30_days['intense_turns'], errors='coerce').dropna()
        if len(recent_intense) > 0:
            recent_avg = recent_intense.mean()
            recent_max = recent_intense.max()
            recent_trend = ""

            # Check trend within the 30 days
            if len(recent_intense) > 2:
                first_half_avg = recent_intense.iloc[:len(recent_intense)//2].mean()
                second_half_avg = recent_intense.iloc[len(recent_intense)//2:].mean()

                if second_half_avg > first_half_avg * 1.15:
                    recent_trend = "surging upward"
                elif second_half_avg > first_half_avg * 1.05:
                    recent_trend = "trending positively"
                elif second_half_avg < first_half_avg * 0.85:
                    recent_trend = "declining"
                else:
                    recent_trend = "holding steady"

            # Compare to historical baseline
            comparison = ""
            if len(before_30_days) > 0 and 'intense_turns' in before_30_days.columns:
                baseline_intense = pd.to_numeric(before_30_days['intense_turns'], errors='coerce').dropna()
                if len(baseline_intense) > 0:
                    baseline_avg = baseline_intense.mean()
                    pct_change = ((recent_avg - baseline_avg) / baseline_avg * 100) if baseline_avg > 0 else 0

                    if pct_change > 20:
                        comparison = f", representing a {pct_change:.0f}% improvement from her earlier baseline"
                    elif pct_change > 5:
                        comparison = f", showing {pct_change:.0f}% growth from previous performance"
                    elif pct_change < -10:
                        comparison = f", down {abs(pct_change):.0f}% from her earlier average"

            agility_narrative = f"Her agility development shows intense turns averaging {recent_avg:.1f} per session with a peak of {recent_max:.0f}, {recent_trend}{comparison}. "

    # SPEED & POWER TRENDS
    speed_narrative = ""
    if 'top_speed' in df.columns:
        recent_speed = pd.to_numeric(last_30_days['top_speed'], errors='coerce').dropna()
        if len(recent_speed) > 0:
            recent_avg_speed = recent_speed.mean()
            recent_max_speed = recent_speed.max()

            speed_consistency = ""
            if recent_speed.std() < 0.5:
                speed_consistency = "with excellent consistency"
            elif recent_speed.std() < 1.0:
                speed_consistency = "showing reliable performance"
            else:
                speed_consistency = "with some variability"

            speed_narrative = f"Speed metrics reveal an average top speed of {recent_avg_speed:.1f} mph (best: {recent_max_speed:.1f} mph) {speed_consistency}. "

    # TURN SPEED DYNAMICS
    turn_narrative = ""
    if 'avg_turn_entry' in df.columns and 'avg_turn_exit' in df.columns:
        recent_entry = pd.to_numeric(last_30_days['avg_turn_entry'], errors='coerce').dropna()
        recent_exit = pd.to_numeric(last_30_days['avg_turn_exit'], errors='coerce').dropna()

        if len(recent_entry) > 0 and len(recent_exit) > 0:
            avg_entry = recent_entry.mean()
            avg_exit = recent_exit.mean()
            speed_diff = avg_exit - avg_entry

            if speed_diff > 0.5:
                turn_narrative = f"Turn dynamics are exceptional, with exit speeds averaging {speed_diff:.1f} mph faster than entry speeds, demonstrating explosive acceleration out of cuts. "
            elif speed_diff > 0:
                turn_narrative = f"Turn performance shows positive exit acceleration ({speed_diff:+.1f} mph), indicating developing explosive power. "
            else:
                turn_narrative = f"Turn exit speeds are currently {abs(speed_diff):.1f} mph slower than entry speeds on average, presenting a clear opportunity for plyometric and explosive training focus. "

    # TRAINING VOLUME
    volume_narrative = ""
    if 'duration' in df.columns:
        recent_duration = pd.to_numeric(last_30_days['duration'], errors='coerce').dropna()
        if len(recent_duration) > 0:
            total_mins = recent_duration.sum()
            avg_session = recent_duration.mean()
            volume_narrative = f"Training volume totaled {total_mins:.0f} minutes ({total_mins/60:.1f} hours) with sessions averaging {avg_session:.0f} minutes each. "

    # Assemble complete narrative
    summary += agility_narrative
    summary += speed_narrative
    summary += turn_narrative
    summary += volume_narrative

    # Add closing insight
    summary += "These trends provide a comprehensive view of recent development patterns and identify specific areas where focused training can accelerate progress toward elite performance.\n\n"

    return summary

def analyze_training_data(df):
    """Analyze training data and generate comprehensive insights"""
    insights = "🤖 COMPREHENSIVE TRAINING INSIGHTS REPORT\n"
    insights += "=" * 80 + "\n"
    insights += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    insights += f"Total Sessions Analyzed: {len(df)}\n"

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        date_range = (df['date'].max() - df['date'].min()).days
        insights += f"Date Range: {df['date'].min().strftime('%b %d, %Y')} to {df['date'].max().strftime('%b %d, %Y')} ({date_range} days)\n"

    insights += "\n"

    # Generate executive summary
    insights += generate_executive_summary(df)
    insights += "\n"

    # Generate 30-day change summary
    insights += generate_30day_change_summary(df)
    insights += "\n"

    # 1. Training Volume Analysis
    insights += "📊 TRAINING VOLUME ANALYSIS\n" + "-" * 80 + "\n"
    if 'duration' in df.columns:
        total_mins = pd.to_numeric(df['duration'], errors='coerce').sum()
        avg_session = pd.to_numeric(df['duration'], errors='coerce').mean()
        insights += f"  • Total Training Time: {total_mins:.0f} minutes ({total_mins/60:.1f} hours)\n"
        insights += f"  • Average Session Length: {avg_session:.1f} minutes\n"

    if 'training_type' in df.columns:
        type_dist = df['training_type'].value_counts()
        insights += f"\n  Training Distribution:\n"
        for ttype, count in type_dist.items():
            insights += f"    - {ttype}: {count} sessions ({count/len(df)*100:.1f}%)\n"

    # 2. Performance Trends
    insights += "\n📈 PERFORMANCE TRENDS\n" + "-" * 80 + "\n"

    metrics_to_analyze = [
        ('top_speed', 'Top Speed', 'mph'),
        ('ball_touches', 'Ball Touches', 'touches'),
        ('sprint_distance', 'Sprint Distance', 'yards')
    ]

    for metric, name, unit in metrics_to_analyze:
        if metric in df.columns:
            values = pd.to_numeric(df[metric], errors='coerce').dropna()
            if len(values) > 1:
                trend = "IMPROVING" if values.iloc[-1] > values.iloc[0] else "DECLINING"
                change = ((values.iloc[-1] - values.iloc[0]) / values.iloc[0] * 100)
                insights += f"  • {name}: {trend} ({change:+.1f}% from first to last session)\n"
                insights += f"    Current: {values.iloc[-1]:.1f} {unit} | Best: {values.max():.1f} {unit} | Avg: {values.mean():.1f} {unit}\n"

    # 3. Two-Footed Development
    insights += "\n⚽ TWO-FOOTED DEVELOPMENT\n" + "-" * 80 + "\n"
    if 'left_pct' in df.columns and 'right_pct' in df.columns:
        left_avg = pd.to_numeric(df['left_pct'], errors='coerce').mean()
        right_avg = pd.to_numeric(df['right_pct'], errors='coerce').mean()
        insights += f"  • Left Foot Usage: {left_avg:.1f}%\n"
        insights += f"  • Right Foot Usage: {right_avg:.1f}%\n"

        if left_avg < 30:
            insights += "  ⚠️ RECOMMENDATION: Increase left foot training to improve balance\n"
        else:
            insights += "  ✅ Good two-footed development - keep it up!\n"

    # 4. Coach Analysis
    insights += "\n👨‍🏫 COACH PERFORMANCE ANALYSIS\n" + "-" * 80 + "\n"
    if 'coach' in df.columns:
        for coach in df['coach'].dropna().unique():
            coach_df = df[df['coach'] == coach]
            # Display "No Coach" instead of "Solo"
            coach_display = "No Coach" if str(coach).lower() == "solo" else coach
            insights += f"  {coach_display} ({len(coach_df)} sessions):\n"

            if 'top_speed' in df.columns:
                avg_speed = pd.to_numeric(coach_df['top_speed'], errors='coerce').mean()
                insights += f"    - Avg Top Speed: {avg_speed:.2f} mph\n"

            if 'ball_touches' in df.columns:
                avg_touches = pd.to_numeric(coach_df['ball_touches'], errors='coerce').mean()
                if not pd.isna(avg_touches):
                    insights += f"    - Avg Ball Touches: {avg_touches:.0f}\n"

    # 5. Agility Analysis (THE CORE FOCUS)
    insights += "\n🔄 AGILITY DEVELOPMENT ANALYSIS\n" + "-" * 80 + "\n"

    # Intense turns - MOST IMPORTANT
    if 'intense_turns' in df.columns:
        intense = pd.to_numeric(df['intense_turns'], errors='coerce')
        avg_intense = intense.mean()
        max_intense = intense.max()
        insights += f"  ⚡ INTENSE TURNS (Game-Speed Agility at 9+ mph):\n"
        insights += f"    Average: {avg_intense:.1f} | Best: {max_intense:.0f}\n"

    # Turn speed relationship (EXIT vs ENTRY)
    if 'avg_turn_entry' in df.columns and 'avg_turn_exit' in df.columns:
        entry = pd.to_numeric(df['avg_turn_entry'], errors='coerce')
        exit_s = pd.to_numeric(df['avg_turn_exit'], errors='coerce')
        avg_entry = entry.mean()
        avg_exit = exit_s.mean()
        speed_diff = avg_exit - avg_entry

        insights += f"\n  💨 TURN SPEED DYNAMICS:\n"
        insights += f"    Entry Speed: {avg_entry:.1f} mph | Exit Speed: {avg_exit:.1f} mph\n"
        insights += f"    Speed Change: {speed_diff:+.1f} mph\n"

        if speed_diff > 0.5:
            insights += f"    ✅ EXPLOSIVE: Accelerating out of cuts - excellent power!\n"
        elif speed_diff < 0:
            insights += f"    ⚠️ FOCUS: Exit slower than entry - work on explosive acceleration\n"
            insights += f"       → Plyometric training, first-step drills\n"

    # Turn directional balance
    if all(col in df.columns for col in ['left_turns', 'right_turns', 'back_turns']):
        left_avg = pd.to_numeric(df['left_turns'], errors='coerce').mean()
        right_avg = pd.to_numeric(df['right_turns'], errors='coerce').mean()
        back_avg = pd.to_numeric(df['back_turns'], errors='coerce').mean()

        insights += f"\n  ↔️ DIRECTIONAL AGILITY:\n"
        insights += f"    Left Turns: {left_avg:.1f} | Right Turns: {right_avg:.1f} | Back Turns: {back_avg:.1f}\n"

        if left_avg > 0 and right_avg > 0:
            ratio = left_avg / right_avg
            if 0.7 <= ratio <= 1.3:
                insights += f"    ✅ Balanced left/right development\n"

    # 6. Speed & Power Analysis
    insights += "\n🚀 SPEED & POWER DEVELOPMENT\n" + "-" * 80 + "\n"

    if 'top_speed' in df.columns:
        speed = pd.to_numeric(df['top_speed'], errors='coerce')
        insights += f"  Top Speed: Current Max {speed.max():.1f} mph | Avg {speed.mean():.1f} mph\n"

        # Trend analysis
        if len(speed) > 3:
            recent_trend = speed.tail(3).mean() - speed.head(3).mean()
            if recent_trend > 0.5:
                insights += f"  ✅ Improving: +{recent_trend:.1f} mph in recent sessions\n"
            elif recent_trend < -0.5:
                insights += f"  ⚠️ Declining: {recent_trend:.1f} mph - may need recovery\n"

    if 'kicking_power' in df.columns:
        kick = pd.to_numeric(df['kicking_power'], errors='coerce')
        insights += f"\n  Kicking Power: Max {kick.max():.1f} mph | Avg {kick.mean():.1f} mph\n"

        # Left vs Right kicking power
        if 'left_kicking_power_mph' in df.columns and 'right_kicking_power_mph' in df.columns:
            left_kick = pd.to_numeric(df['left_kicking_power_mph'], errors='coerce').mean()
            right_kick = pd.to_numeric(df['right_kicking_power_mph'], errors='coerce').mean()
            if pd.notna(left_kick) and pd.notna(right_kick):
                insights += f"  Left Foot: {left_kick:.1f} mph | Right Foot: {right_kick:.1f} mph\n"
                diff = abs(left_kick - right_kick)
                if diff < 3:
                    insights += f"  ✅ Balanced kicking power between feet\n"
                else:
                    weaker_foot = "left" if left_kick < right_kick else "right"
                    insights += f"  ⚠️ {weaker_foot.capitalize()} foot needs power development\n"

    if 'sprint_distance' in df.columns and 'num_sprints' in df.columns:
        sprint_dist = pd.to_numeric(df['sprint_distance'], errors='coerce')
        num_sprints = pd.to_numeric(df['num_sprints'], errors='coerce')
        insights += f"\n  Sprint Volume: Avg {sprint_dist.mean():.0f} yards over {num_sprints.mean():.1f} sprints\n"

    # 7. Technical Skill - Two-Footed & Ball Work
    insights += "\n⚽ TECHNICAL DEVELOPMENT\n" + "-" * 80 + "\n"

    if 'ball_touches' in df.columns:
        touches = pd.to_numeric(df['ball_touches'], errors='coerce')
        ball_sessions = touches.notna().sum()
        insights += f"  Ball Work Frequency: {ball_sessions}/{len(df)} sessions ({ball_sessions/len(df)*100:.0f}%)\n"
        if ball_sessions > 0:
            insights += f"  Avg Touches: {touches.mean():.0f} | Max: {touches.max():.0f}\n"

    # Left/Right touch balance and ratio
    if 'left_touches' in df.columns and 'right_touches' in df.columns:
        left_touch = pd.to_numeric(df['left_touches'], errors='coerce')
        right_touch = pd.to_numeric(df['right_touches'], errors='coerce')

        # Calculate ratios for sessions where both exist
        ratios = []
        for idx, row in df.iterrows():
            l = pd.to_numeric(row.get('left_touches'), errors='coerce')
            r = pd.to_numeric(row.get('right_touches'), errors='coerce')
            if pd.notna(l) and pd.notna(r) and l > 0 and r > 0:
                ratios.append(l / r)

        if ratios:
            avg_ratio = sum(ratios) / len(ratios)
            best_ratio = min(ratios, key=lambda x: abs(x - 0.5))
            insights += f"\n  Left/Right Touch Ratio:\n"
            insights += f"    Average: {avg_ratio:.2f} | Best: {best_ratio:.2f} | Goal: 0.50\n"

            if avg_ratio >= 0.5:
                insights += f"    ✅ GOAL MET: Excellent left foot development!\n"
            elif avg_ratio >= 0.4:
                insights += f"    📈 CLOSE: Almost at goal - keep working left foot!\n"
            else:
                insights += f"    ⚠️ FOCUS: Need more left foot touches (currently {avg_ratio:.0%} of right)\n"

    # 8. Workload & Recovery
    insights += "\n📊 TRAINING LOAD MANAGEMENT\n" + "-" * 80 + "\n"

    if 'duration' in df.columns:
        duration = pd.to_numeric(df['duration'], errors='coerce')
        total_mins = duration.sum()
        insights += f"  Total Volume: {total_mins:.0f} minutes ({total_mins/60:.1f} hours)\n"
        insights += f"  Avg Session: {duration.mean():.1f} minutes\n"

    if 'date' in df.columns:
        date_range = (df['date'].max() - df['date'].min()).days
        sessions_per_week = len(df) / (date_range / 7) if date_range > 0 else 0
        insights += f"  Training Frequency: {sessions_per_week:.1f} sessions/week\n"

        if sessions_per_week < 3:
            insights += f"  ⚠️ Consider increasing to 3-4 sessions/week for optimal development\n"

    if 'intensity' in df.columns:
        intensity_dist = df['intensity'].value_counts(normalize=True)
        insights += f"\n  Intensity Distribution:\n"
        for intensity, pct in intensity_dist.items():
            insights += f"    {intensity}: {pct*100:.0f}%\n"

    # 9. Metric Relationships & Correlations
    insights += "\n🔗 PERFORMANCE RELATIONSHIPS\n" + "-" * 80 + "\n"

    # Agility vs Speed
    if 'intense_turns' in df.columns and 'top_speed' in df.columns:
        intense_vals = pd.to_numeric(df['intense_turns'], errors='coerce')
        speed_vals = pd.to_numeric(df['top_speed'], errors='coerce')
        valid_mask = intense_vals.notna() & speed_vals.notna()

        if valid_mask.sum() > 3:
            correlation = intense_vals[valid_mask].corr(speed_vals[valid_mask])
            insights += f"  Agility-Speed Correlation: {correlation:.2f}\n"

    # Ball work vs Distance
    if 'ball_touches' in df.columns and 'total_distance' in df.columns:
        touches_vals = pd.to_numeric(df['ball_touches'], errors='coerce')
        dist_vals = pd.to_numeric(df['total_distance'], errors='coerce')

        if touches_vals.notna().any() and dist_vals.notna().any():
            insights += f"\n  Ball Work Intensity:\n"
            high_touch_sessions = df[touches_vals > touches_vals.median()]
            if len(high_touch_sessions) > 0 and 'total_distance' in high_touch_sessions.columns:
                avg_dist_high_touch = pd.to_numeric(high_touch_sessions['total_distance'], errors='coerce').mean()
                insights += f"    High-touch sessions avg distance: {avg_dist_high_touch:.2f} miles\n"

    # 10. Training Environment Analysis
    insights += "\n🌍 TRAINING ENVIRONMENT ANALYSIS\n" + "-" * 80 + "\n"

    # Location analysis
    if 'location' in df.columns:
        location_dist = df['location'].value_counts()
        insights += f"  Training Locations:\n"
        for location, count in location_dist.items():
            insights += f"    • {location}: {count} sessions ({count/len(df)*100:.1f}%)\n"

        # Performance by location
        if 'top_speed' in df.columns and len(location_dist) > 1:
            insights += f"\n  Performance by Location:\n"
            for location in location_dist.index:
                loc_df = df[df['location'] == location]
                avg_speed = pd.to_numeric(loc_df['top_speed'], errors='coerce').mean()
                if pd.notna(avg_speed):
                    insights += f"    {location}: Avg Top Speed {avg_speed:.1f} mph\n"

    # Surface analysis
    if 'surface' in df.columns:
        surface_dist = df['surface'].value_counts()
        insights += f"\n  Surface Distribution:\n"
        for surface, count in surface_dist.items():
            insights += f"    • {surface}: {count} sessions ({count/len(df)*100:.1f}%)\n"

        # Performance by surface
        if len(surface_dist) > 1:
            insights += f"\n  Surface Performance Comparison:\n"
            for surface in surface_dist.index:
                surf_df = df[df['surface'] == surface]

                if 'top_speed' in df.columns:
                    avg_speed = pd.to_numeric(surf_df['top_speed'], errors='coerce').mean()
                    if pd.notna(avg_speed):
                        insights += f"    {surface}:\n"
                        insights += f"      - Top Speed: {avg_speed:.1f} mph\n"

                if 'intense_turns' in df.columns:
                    avg_turns = pd.to_numeric(surf_df['intense_turns'], errors='coerce').mean()
                    if pd.notna(avg_turns):
                        insights += f"      - Intense Turns: {avg_turns:.1f}\n"

            # Find best surface
            if 'top_speed' in df.columns:
                best_surface = None
                best_speed = 0
                for surface in surface_dist.index:
                    surf_df = df[df['surface'] == surface]
                    avg_speed = pd.to_numeric(surf_df['top_speed'], errors='coerce').mean()
                    if pd.notna(avg_speed) and avg_speed > best_speed:
                        best_speed = avg_speed
                        best_surface = surface
                if best_surface:
                    insights += f"\n    ✅ Best surface for speed: {best_surface} ({best_speed:.1f} mph avg)\n"

    # With Ball vs Without Ball analysis
    if 'with_ball' in df.columns:
        ball_dist = df['with_ball'].value_counts()
        insights += f"\n  Ball Work Distribution:\n"
        for status, count in ball_dist.items():
            insights += f"    • {status} Ball: {count} sessions ({count/len(df)*100:.1f}%)\n"

        # Compare ball vs non-ball sessions
        if len(ball_dist) > 1:
            ball_sessions = df[df['with_ball'].str.lower() == 'yes']
            no_ball_sessions = df[df['with_ball'].str.lower() == 'no']

            insights += f"\n  Ball vs Non-Ball Performance:\n"

            if 'top_speed' in df.columns:
                ball_speed = pd.to_numeric(ball_sessions['top_speed'], errors='coerce').mean()
                no_ball_speed = pd.to_numeric(no_ball_sessions['top_speed'], errors='coerce').mean()
                if pd.notna(ball_speed) and pd.notna(no_ball_speed):
                    insights += f"    Top Speed: With Ball {ball_speed:.1f} mph | Without Ball {no_ball_speed:.1f} mph\n"
                    diff = no_ball_speed - ball_speed
                    if diff > 1:
                        insights += f"    💡 {diff:.1f} mph faster without ball - expected for pure speed work\n"

            if 'intense_turns' in df.columns:
                ball_turns = pd.to_numeric(ball_sessions['intense_turns'], errors='coerce').mean()
                no_ball_turns = pd.to_numeric(no_ball_sessions['intense_turns'], errors='coerce').mean()
                if pd.notna(ball_turns) and pd.notna(no_ball_turns):
                    insights += f"    Intense Turns: With Ball {ball_turns:.1f} | Without Ball {no_ball_turns:.1f}\n"

            if 'duration' in df.columns:
                ball_duration = pd.to_numeric(ball_sessions['duration'], errors='coerce').mean()
                no_ball_duration = pd.to_numeric(no_ball_sessions['duration'], errors='coerce').mean()
                if pd.notna(ball_duration) and pd.notna(no_ball_duration):
                    insights += f"    Avg Duration: With Ball {ball_duration:.0f} min | Without Ball {no_ball_duration:.0f} min\n"

    # Session type analysis
    if 'training_type' in df.columns:
        training_dist = df['training_type'].value_counts()
        insights += f"\n  Training Types:\n"
        top_trainings = training_dist.head(5)
        for training, count in top_trainings.items():
            insights += f"    • {training}: {count} sessions\n"

        if len(training_dist) > 5:
            insights += f"    • ({len(training_dist) - 5} other training types)\n"

    # 11. Actionable Recommendations
    insights += "\n💡 PERSONALIZED ACTION PLAN\n" + "-" * 80 + "\n"

    recommendations = []

    # Agility recommendations
    if 'intense_turns' in df.columns:
        intense_avg = pd.to_numeric(df['intense_turns'], errors='coerce').mean()
        if intense_avg < 5:
            recommendations.append("🎯 PRIORITY: Increase intense turns to 10+/session\n   → High-speed cutting drills, small-sided games")

    # Exit speed recommendations
    if 'avg_turn_entry' in df.columns and 'avg_turn_exit' in df.columns:
        entry_avg = pd.to_numeric(df['avg_turn_entry'], errors='coerce').mean()
        exit_avg = pd.to_numeric(df['avg_turn_exit'], errors='coerce').mean()
        if exit_avg <= entry_avg:
            recommendations.append("⚡ Improve explosive power out of cuts\n   → Plyometrics, resistance training, first-step drills")

    # Left foot development
    if 'left_touches' in df.columns and 'right_touches' in df.columns:
        left_avg = pd.to_numeric(df['left_touches'], errors='coerce').mean()
        right_avg = pd.to_numeric(df['right_touches'], errors='coerce').mean()
        if left_avg > 0 and right_avg > 0:
            ratio = left_avg / right_avg
            if ratio < 0.4:
                recommendations.append("👣 Increase left foot training for better balance\n   → Dedicated left-foot drills, force left-only touches")

    # Speed development
    if 'top_speed' in df.columns:
        speed = pd.to_numeric(df['top_speed'], errors='coerce')
        if len(speed) > 3:
            recent = speed.tail(3).mean()
            if recent < speed.max() * 0.9:
                recommendations.append("🚀 Focus on maintaining top speed\n   → Sprint intervals, resistance training, proper recovery")

    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            insights += f"  {i}. {rec}\n\n"
    else:
        insights += "  ✅ All metrics show balanced, effective development!\n"
        insights += "     Continue current training approach.\n"

    # 12. Next Milestones
    insights += "\n🎯 NEXT MILESTONE TARGETS\n" + "-" * 80 + "\n"

    if 'top_speed' in df.columns:
        current_speed = pd.to_numeric(df['top_speed'], errors='coerce').max()
        next_target = ((current_speed // 0.5) + 1) * 0.5
        insights += f"  • Top Speed: Current {current_speed:.1f} mph → Target {next_target:.1f} mph\n"

    if 'ball_touches' in df.columns:
        current_touches = pd.to_numeric(df['ball_touches'], errors='coerce').max()
        next_target = ((current_touches // 50) + 1) * 50
        insights += f"  • Ball Touches: Current {current_touches:.0f} → Target {next_target:.0f}\n"

    insights += "\n" + "=" * 80 + "\n"
    insights += "Keep pushing boundaries and tracking progress! 🌟⚽\n"

    return insights

# Open Graph meta tags for link previews
st.markdown("""
<meta property="og:title" content="Mia Training Tracker" />
<meta property="og:description" content="Soccer Performance Analytics - Track speed, agility, ball work, and comprehensive training metrics" />
<meta property="og:image" content="https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/preview.png" />
<meta property="og:type" content="website" />
<meta name="twitter:card" content="summary_large_image" />
<meta name="twitter:title" content="Mia Training Tracker" />
<meta name="twitter:description" content="Soccer Performance Analytics" />
<meta name="twitter:image" content="https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/preview.png" />
""", unsafe_allow_html=True)

# Main App Header
st.markdown('<div class="main-header">⚽ Mia Training Tracker</div>', unsafe_allow_html=True)

# Display OCR warning if not available
if not OCR_AVAILABLE:
    st.warning(f"⚠️ OCR not available: {OCR_ERROR}. Manual data entry only.")

# Auto-load data from Google Sheets on app start
if GSHEETS_AVAILABLE and "google_sheets_url" in st.secrets:
    if 'auto_loaded' not in st.session_state:
        st.session_state.auto_loaded = False

    if not st.session_state.auto_loaded:
        with st.spinner("Loading data from Google Sheets..."):
            df, error = load_data_from_google_sheets()
            if not error and df is not None:
                st.session_state.df = df
                st.session_state.auto_loaded = True
                calculate_personal_records()

# Refresh Data button (visible in main area)
if GSHEETS_AVAILABLE and "google_sheets_url" in st.secrets:
    col_refresh1, col_refresh2, col_refresh3 = st.columns([1, 1, 1])
    with col_refresh2:
        if st.button("🔄 Refresh Data", type="primary", use_container_width=True):
            with st.spinner("Refreshing data from Google Sheets..."):
                df, error = load_data_from_google_sheets()
                if error:
                    st.error(f"❌ Failed to refresh: {error}")
                else:
                    st.session_state.df = df
                    calculate_personal_records()

                    # Clear cached AI insights so user regenerates with refreshed data
                    if 'ai_insights_generated' in st.session_state:
                        st.session_state.ai_insights_generated = False
                    if 'ai_insights_report' in st.session_state:
                        st.session_state.ai_insights_report = ""

                    st.success("✅ Data refreshed!")
                    st.rerun()

st.markdown("---")

# Main tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "📊 Dashboard",
    "📸 Upload & Extract",
    "📈 Analytics",
    "⚡ Speed",
    "🔄 Agility",
    "⚽ Ball Work",
    "⚽ Match Play",
    "🏆 Personal Records",
    "🤖 AI Insights"
])

# Tab 1: Dashboard
with tab1:
    st.header("📊 Training Dashboard")

    if st.session_state.df is None or len(st.session_state.df) == 0:
        st.warning("📊 No data loaded. Please upload your Excel file in the sidebar or sync from Google Sheets.")
    else:
        df = st.session_state.df.copy()

        # Key Metrics Row 1
        st.subheader("🎯 Key Performance Indicators")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if 'top_speed' in df.columns:
                values = pd.to_numeric(df['top_speed'], errors='coerce').dropna()
                if len(values) > 0:
                    avg_speed = values.mean()
                    best_speed = values.max()
                    st.metric("Top Speed (mph) (avg)", f"{avg_speed:.1f}",
                             delta=f"Best: {best_speed:.1f}", delta_color="normal")

        with col2:
            if 'intense_turns' in df.columns:
                values = pd.to_numeric(df['intense_turns'], errors='coerce').dropna()
                if len(values) > 0:
                    avg_turns = values.mean()
                    best_turns = values.max()
                    st.metric("Intense Turns (avg)", f"{avg_turns:.1f}",
                             delta=f"Best: {best_turns:.1f}", delta_color="normal")

        with col3:
            if 'left_kicking_power_mph' in df.columns and 'right_kicking_power_mph' in df.columns:
                left_power = pd.to_numeric(df['left_kicking_power_mph'], errors='coerce').dropna()
                right_power = pd.to_numeric(df['right_kicking_power_mph'], errors='coerce').dropna()

                if len(left_power) > 0 or len(right_power) > 0:
                    avg_left = left_power.mean() if len(left_power) > 0 else 0
                    avg_right = right_power.mean() if len(right_power) > 0 else 0
                    best_left = left_power.max() if len(left_power) > 0 else 0
                    best_right = right_power.max() if len(right_power) > 0 else 0

                    if avg_left >= avg_right:
                        top_power_avg = avg_left
                        top_power_best = best_left
                        foot = "L"
                    else:
                        top_power_avg = avg_right
                        top_power_best = best_right
                        foot = "R"

                    st.metric("Top Foot Power (mph) (avg)", f"{top_power_avg:.1f} {foot}",
                             delta=f"Best: {top_power_best:.1f} {foot}", delta_color="normal")

        with col4:
            if 'left_touches' in df.columns and 'right_touches' in df.columns:
                left = pd.to_numeric(df['left_touches'], errors='coerce')
                right = pd.to_numeric(df['right_touches'], errors='coerce')
                valid_mask = (left > 0) & (right > 0)

                if valid_mask.any():
                    ratios = left[valid_mask] / right[valid_mask]
                    avg_ratio = ratios.mean()
                    best_ratio = ratios.max()

                    st.metric("L/R Touch Ratio (avg)", f"{avg_ratio:.2f}",
                             delta=f"Best: {best_ratio:.2f}", delta_color="normal")
                else:
                    st.metric("L/R Touch Ratio (avg)", "N/A",
                             delta="No data", delta_color="off")

        st.markdown("---")

        # Training Summary Row
        st.subheader("📅 Training Summary")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Sessions", len(df))

        with col2:
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                date_range = (df['date'].max() - df['date'].min()).days
                st.metric("Training Period (days)", date_range)

        with col3:
            if 'duration' in df.columns:
                total_mins = pd.to_numeric(df['duration'], errors='coerce').sum()
                st.metric("Total Training Time", f"{total_mins/60:.1f} hrs")

        with col4:
            if 'with_ball' in df.columns:
                ball_sessions = (df['with_ball'].str.lower() == 'yes').sum()
                st.metric("Ball Work Sessions", f"{ball_sessions}/{len(df)}")

        st.markdown("---")

        # Recent Performance Trends
        st.subheader("📈 Recent Trends (Last 5 Sessions)")

        if len(df) >= 5:
            recent_df = df.tail(5)

            col1, col2 = st.columns(2)

            with col1:
                # Speed trend
                if 'top_speed' in recent_df.columns and 'date' in recent_df.columns:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    speed_data = recent_df[['date', 'top_speed']].dropna()
                    if len(speed_data) > 0:
                        ax.plot(speed_data['date'], speed_data['top_speed'],
                               marker='o', linewidth=2, markersize=8, color='#1E88E5')
                        ax.set_xlabel('Date', fontsize=10)
                        ax.set_ylabel('Top Speed (mph)', fontsize=10)
                        ax.set_title('Top Speed Trend', fontsize=12, fontweight='bold')
                        ax.grid(True, alpha=0.3)
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)

            with col2:
                # Agility trend
                if 'intense_turns' in recent_df.columns and 'date' in recent_df.columns:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    agility_data = recent_df[['date', 'intense_turns']].dropna()
                    if len(agility_data) > 0:
                        ax.plot(agility_data['date'], agility_data['intense_turns'],
                               marker='s', linewidth=2, markersize=8, color='#E53935')
                        ax.set_xlabel('Date', fontsize=10)
                        ax.set_ylabel('Intense Turns', fontsize=10)
                        ax.set_title('Agility Trend', fontsize=12, fontweight='bold')
                        ax.grid(True, alpha=0.3)
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
        else:
            st.info("Need at least 5 sessions to show trend charts")

        st.markdown("---")

        # Quick Insights
        st.subheader("💡 Quick Insights")
        col1, col2 = st.columns(2)

        with col1:
            # Two-footed balance
            if 'left_touches' in df.columns and 'right_touches' in df.columns:
                left_total = pd.to_numeric(df['left_touches'], errors='coerce').sum()
                right_total = pd.to_numeric(df['right_touches'], errors='coerce').sum()
                if right_total > 0:
                    ratio = left_total / right_total
                    st.info(f"**⚖️ Two-Footed Balance**\n\nLeft/Right Ratio: {ratio:.2f}\n\n" +
                           ("✅ Excellent balance!" if ratio >= 0.5 else
                            "📈 Good progress - keep working left foot!" if ratio >= 0.4 else
                            "⚠️ Focus on left foot development"))

        with col2:
            # Training intensity
            if 'intensity' in df.columns:
                intensity_counts = df['intensity'].value_counts()
                if len(intensity_counts) > 0:
                    most_common = intensity_counts.index[0]
                    count = intensity_counts.iloc[0]
                    st.info(f"**🔥 Training Intensity**\n\nMost Common: {most_common}\n\n" +
                           f"Used in {count}/{len(df)} sessions ({count/len(df)*100:.0f}%)")

# Tab 2: Upload & Extract
with tab2:
    st.header("Upload CityPlay Screenshots")

    uploaded_files = st.file_uploader(
        "Upload one or more screenshots",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        key='screenshot_upload'
    )

    if uploaded_files:
        st.write(f"📸 {len(uploaded_files)} image(s) uploaded")

        # Show thumbnails
        cols = st.columns(min(4, len(uploaded_files)))
        for idx, file in enumerate(uploaded_files):
            with cols[idx % 4]:
                image = Image.open(file)
                st.image(image, caption=f"Image {idx+1}", use_container_width=True)

        if st.button("🔍 Extract Data from All Images", disabled=not OCR_AVAILABLE):
            if not OCR_AVAILABLE:
                st.error("❌ OCR is not available. Please enter data manually below.")
            else:
                with st.spinner("Processing images with OCR..."):
                    all_extracted = {}

                    for file in uploaded_files:
                        try:
                            image = Image.open(file)
                            text = pytesseract.image_to_string(image)
                            extracted = parse_ocr_text(text)

                            for key, value in extracted.items():
                                if value and str(value).strip():
                                    all_extracted[key] = value
                        except Exception as e:
                            st.error(f"Error processing {file.name}: {e}")

                    st.success(f"✅ Extracted {len(all_extracted)} fields!")
                    st.session_state['extracted_data'] = all_extracted

        if not OCR_AVAILABLE and uploaded_files:
            st.info("💡 OCR unavailable. Please manually enter data from your screenshots below.")

    # Data entry form
    st.subheader("Session Data Entry")

    # Initialize extracted_data in session state if not present
    if 'extracted_data' not in st.session_state:
        st.session_state['extracted_data'] = {}

    # Initialize form counter for reset functionality
    if 'form_counter' not in st.session_state:
        st.session_state['form_counter'] = 0

    # Initialize confirmation state
    if 'pending_submission' not in st.session_state:
        st.session_state['pending_submission'] = None

    extracted_data = st.session_state.get('extracted_data', {})

    # Training type selection (outside form to control field visibility)
    training_types = ["Speed and Agility", "Ball Work", "Match-Grass", "Match-Turf", "Match-Hard"]
    selected_training_type = st.selectbox(
        "Select Training Type",
        training_types,
        key=f"training_type_selector_{st.session_state['form_counter']}"
    )

    with st.form(f"session_form_{st.session_state['form_counter']}"):
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Session Info**")
            date = st.date_input("Date", value=datetime.now())

            # Session Name - dropdown with existing values + custom entry
            existing_sessions = []
            if st.session_state.df is not None and 'session_name' in st.session_state.df.columns:
                existing_sessions = st.session_state.df['session_name'].dropna().unique().tolist()
            session_options = ["-- Enter New --"] + sorted(existing_sessions)
            session_choice = st.selectbox("Session Name", session_options, key="session_name_select")
            if session_choice == "-- Enter New --":
                session_name = st.text_input("↳ Enter New Session Name", value=extracted_data.get('session_name', ''), key="session_name_input")
            else:
                session_name = session_choice

            # Coach - dropdown with existing values + custom entry
            existing_coaches = []
            if st.session_state.df is not None and 'coach' in st.session_state.df.columns:
                existing_coaches = st.session_state.df['coach'].dropna().unique().tolist()
            coach_options = ["-- Enter New --"] + sorted(existing_coaches)
            coach_choice = st.selectbox("Coach", coach_options, key="coach_select")
            if coach_choice == "-- Enter New --":
                coach = st.text_input("↳ Enter New Coach Name", value=extracted_data.get('coach', ''), key="coach_input")
            else:
                coach = coach_choice

            # Location - dropdown with existing values + custom entry
            existing_locations = []
            if st.session_state.df is not None and 'location' in st.session_state.df.columns:
                existing_locations = st.session_state.df['location'].dropna().unique().tolist()
            location_options = ["-- Enter New --"] + sorted(existing_locations)
            location_choice = st.selectbox("Location", location_options, key="location_select")
            if location_choice == "-- Enter New --":
                location = st.text_input("↳ Enter New Location", value=extracted_data.get('location', ''), key="location_input")
            else:
                location = location_choice

            # Surface - fixed dropdown
            surface = st.selectbox("Surface", ["Grass", "Turf", "Hard"])

            # Auto-set With Ball based on training type
            # "Yes" for Ball Work and any Match types (Match-Grass, Match-Turf, Match-Hard)
            with_ball = "Yes" if (selected_training_type == "Ball Work" or "Match" in selected_training_type) else "No"

            duration = st.number_input("Duration (min)", value=int(extracted_data.get('duration', 0)) if extracted_data.get('duration') else 0)

            # Intensity - fixed dropdown with 10 levels
            intensity_options = ["Minimal", "Extremely Easy", "Very Easy", "Easy", "Moderate",
                               "Somewhat Hard", "Hard", "Very Hard", "Extremely Hard", "Maximal"]
            intensity = st.selectbox("Intensity", intensity_options)

            st.write("**Movement Metrics (All Types)**")
            total_distance = st.number_input("Total Distance (mi)", value=float(extracted_data.get('total_distance', 0)) if extracted_data.get('total_distance') else 0.0, format="%.2f")
            sprint_distance = st.number_input("Sprint Distance (yd)", value=float(extracted_data.get('sprint_distance', 0)) if extracted_data.get('sprint_distance') else 0.0, format="%.1f")
            top_speed = st.number_input("Top Speed (mph)", value=float(extracted_data.get('top_speed', 0)) if extracted_data.get('top_speed') else 0.0, format="%.2f")
            num_sprints = st.number_input("Number of Sprints", value=int(extracted_data.get('num_sprints', 0)) if extracted_data.get('num_sprints') else 0)
            accelerations = st.number_input("Accl/Decl", value=int(extracted_data.get('accelerations', 0)) if extracted_data.get('accelerations') else 0)

            st.write("**Agility (All Types)**")
            left_turns = st.number_input("Left Turns", value=int(extracted_data.get('left_turns', 0)) if extracted_data.get('left_turns') else 0)
            right_turns = st.number_input("Right Turns", value=int(extracted_data.get('right_turns', 0)) if extracted_data.get('right_turns') else 0)
            back_turns = st.number_input("Back Turns", value=int(extracted_data.get('back_turns', 0)) if extracted_data.get('back_turns') else 0)
            intense_turns = st.number_input("Intense Turns", value=int(extracted_data.get('intense_turns', 0)) if extracted_data.get('intense_turns') else 0)
            # Total Turns - calculated field (removed from user input)
            avg_turn_entry = st.number_input("Avg Turn Entry Speed (mph)", value=float(extracted_data.get('avg_turn_entry', 0)) if extracted_data.get('avg_turn_entry') else 0.0, format="%.1f")
            avg_turn_exit = st.number_input("Avg Turn Exit Speed (mph)", value=float(extracted_data.get('avg_turn_exit', 0)) if extracted_data.get('avg_turn_exit') else 0.0, format="%.1f")

        with col2:
            # Ball Work fields - shown for Ball Work and Match types
            if selected_training_type in ["Ball Work", "Match-Grass", "Match-Turf", "Match-Hard"]:
                st.write("**Ball Work**")
                # Ball Touches - calculated field (removed from user input)
                left_touches = st.number_input("Left Foot Touches", value=int(extracted_data.get('left_touches', 0)) if extracted_data.get('left_touches') else 0)
                right_touches = st.number_input("Right Foot Touches", value=int(extracted_data.get('right_touches', 0)) if extracted_data.get('right_touches') else 0)
                # Left Foot % - calculated field (removed from user input)
                left_releases = st.number_input("Left Releases", value=int(extracted_data.get('left_releases', 0)) if extracted_data.get('left_releases') else 0)
                right_releases = st.number_input("Right Releases", value=int(extracted_data.get('right_releases', 0)) if extracted_data.get('right_releases') else 0)
                # Kicking Power - calculated field (removed from user input)
                left_kicking_power = st.number_input("Left Kicking Power (mph)", value=float(extracted_data.get('left_kicking_power_mph', 0)) if extracted_data.get('left_kicking_power_mph') else 0.0, format="%.2f")
                right_kicking_power = st.number_input("Right Kicking Power (mph)", value=float(extracted_data.get('right_kicking_power_mph', 0)) if extracted_data.get('right_kicking_power_mph') else 0.0, format="%.2f")
            else:
                # Set defaults for Speed and Agility
                left_touches = 0
                right_touches = 0
                left_releases = 0
                right_releases = 0
                left_kicking_power = 0
                right_kicking_power = 0

            # Match-specific fields - shown only for Match types
            if selected_training_type in ["Match-Grass", "Match-Turf", "Match-Hard"]:
                st.write("**Match Stats**")
                position = st.text_input("Position", value=extracted_data.get('position', ''))
                goals = st.number_input("Goals", value=int(extracted_data.get('goals', 0)) if extracted_data.get('goals') else 0)
                assists = st.number_input("Assists", value=int(extracted_data.get('assists', 0)) if extracted_data.get('assists') else 0)
                # Work Rate - calculated field (removed from user input)
                ball_possessions = st.number_input("Ball Possessions", value=int(extracted_data.get('ball_possessions', 0)) if extracted_data.get('ball_possessions') else 0)
            else:
                # Set defaults for non-match types
                position = ''
                goals = 0
                assists = 0
                ball_possessions = 0

        col_submit1, col_submit2 = st.columns([1, 1])
        with col_submit1:
            submitted = st.form_submit_button("💾 Add to Data File", use_container_width=True)
        with col_submit2:
            reset = st.form_submit_button("🔄 Reset Form", use_container_width=True)

        if reset:
            # Clear extracted data and increment form counter to reset form
            st.session_state['extracted_data'] = {}
            st.session_state['form_counter'] += 1
            st.session_state['pending_submission'] = None
            st.rerun()

        if submitted:
            # Calculate derived fields
            # Total Turns = Left Turns + Right Turns + Back Turns
            total_turns = left_turns + right_turns + back_turns

            # Ball Touches = Left Foot Touches + Right Foot Touches
            ball_touches = left_touches + right_touches

            # Left Foot % = Left Foot Touches / Ball Touches (if Ball Touches > 0)
            left_foot_pct = (left_touches / ball_touches * 100) if ball_touches > 0 else 0

            # Kicking Power = Max of Left Kicking Power or Right Kicking Power
            kicking_power = max(left_kicking_power, right_kicking_power)

            # Work Rate = (Total Distance in miles × 1760 yards/mile) / Duration in minutes
            # Result is in yards per minute, rounded to 2 decimal places
            work_rate = round(((total_distance * 1760) / duration), 2) if duration > 0 else 0

            # Store submission data for confirmation
            new_row = {
                'date': date,
                'session_name': session_name,
                'coach': coach,
                'location': location,
                'surface': surface,
                'with_ball': with_ball,
                'training_type': selected_training_type,
                'duration': duration,
                'intensity': intensity,
                'total_distance': total_distance,
                'sprint_distance': sprint_distance,
                'accelerations': accelerations,
                'top_speed': top_speed,
                'num_sprints': num_sprints,
                'left_turns': left_turns,
                'back_turns': back_turns,
                'right_turns': right_turns,
                'intense_turns': intense_turns,
                'total_turns': total_turns,
                'avg_turn_entry': avg_turn_entry,
                'avg_turn_exit': avg_turn_exit,
                'ball_touches': ball_touches,
                'left_touches': left_touches,
                'right_touches': right_touches,
                'left_foot_pct': left_foot_pct,
                'left_releases': left_releases,
                'right_releases': right_releases,
                'kicking_power_mph': kicking_power,
                'left_kicking_power_mph': left_kicking_power,
                'right_kicking_power_mph': right_kicking_power,
                'position': position,
                'goals': goals,
                'assists': assists,
                'work_rate': work_rate,
                'ball_possessions': ball_possessions,
            }

            # Store the row for confirmation
            st.session_state['pending_submission'] = new_row

    # Show confirmation dialog if there's a pending submission
    if st.session_state.get('pending_submission') is not None:
        st.warning("⚠️ **Confirm you want to update the data file**")

        col_confirm1, col_confirm2, col_confirm3 = st.columns([1, 1, 2])

        with col_confirm1:
            if st.button("✅ Yes", type="primary", use_container_width=True):
                new_row = st.session_state['pending_submission']

                # Add to dataframe
                if st.session_state.df is None:
                    st.session_state.df = pd.DataFrame([new_row])
                else:
                    st.session_state.df = pd.concat([st.session_state.df, pd.DataFrame([new_row])], ignore_index=True)

                # Ensure data types are preserved after concat
                numeric_columns = [
                    'duration', 'ball_touches', 'total_distance', 'sprint_distance',
                    'accelerations', 'kicking_power_mph', 'top_speed', 'num_sprints',
                    'left_touches', 'right_touches', 'left_foot_pct',
                    'left_releases', 'right_releases',
                    'left_kicking_power_mph', 'right_kicking_power_mph',
                    'left_turns', 'back_turns', 'right_turns', 'intense_turns',
                    'avg_turn_entry', 'avg_turn_exit', 'total_turns', 'work_rate',
                    'goals', 'assists', 'ball_possessions'
                ]

                for col in numeric_columns:
                    if col in st.session_state.df.columns:
                        st.session_state.df[col] = pd.to_numeric(st.session_state.df[col], errors='coerce')

                # Ensure date column is datetime
                if 'date' in st.session_state.df.columns:
                    st.session_state.df['date'] = pd.to_datetime(st.session_state.df['date'], errors='coerce')

                calculate_personal_records()

                # Auto-save to Google Sheets if configured
                if GSHEETS_AVAILABLE and "google_sheets_url" in st.secrets:
                    with st.spinner("Saving to Google Sheets..."):
                        success, error = save_data_to_google_sheets(st.session_state.df)
                        if error:
                            st.warning(f"⚠️ Session added locally but cloud save failed: {error}")
                            st.info("💡 Use 'Save to Cloud' button in sidebar to sync manually")
                        else:
                            st.success("✅ Session added and saved to cloud!")
                else:
                    st.success("✅ Session added successfully!")

                st.session_state['extracted_data'] = {}  # Clear extracted data
                st.session_state['form_counter'] += 1  # Increment to reset form
                st.session_state['pending_submission'] = None  # Clear pending submission

                # Clear cached AI insights so user regenerates with new data
                if 'ai_insights_generated' in st.session_state:
                    st.session_state.ai_insights_generated = False
                if 'ai_insights_report' in st.session_state:
                    st.session_state.ai_insights_report = ""

                st.rerun()

        with col_confirm2:
            if st.button("❌ No", use_container_width=True):
                st.session_state['pending_submission'] = None
                st.rerun()

# Tab 3: Analytics
with tab3:
    st.header("📊 Training Analytics")

    if st.session_state.df is None or len(st.session_state.df) == 0:
        st.warning("📊 No data loaded. Please upload your Excel file in the sidebar.")
    else:
        df = st.session_state.df.copy()

        # Ensure date column is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.sort_values('date')

        # Coach filter
        if 'coach' in df.columns:
            coaches = df['coach'].dropna().unique().tolist()
            # Replace "Solo" with "No Coach" in the display
            coaches_display = ["No Coach" if str(c).lower() == "solo" else c for c in coaches]
            coaches_display_map = dict(zip(coaches_display, coaches))

            selected_coach_display = st.selectbox(
                "Filter by Coach",
                ["All Coaches"] + coaches_display,
                key="analytics_coach_filter"
            )

            if selected_coach_display != "All Coaches":
                selected_coach = coaches_display_map[selected_coach_display]
                df = df[df['coach'] == selected_coach]

        # Chart selection
        chart_option = st.selectbox(
            "Select Chart",
            ["-- Select a Chart --", "Top Speed Progress", "Ball Touches Progress", "Sprint Distance Progress",
             "Kicking Power Progress", "Agility Performance", "Turn Speed Analysis"]
        )

        if chart_option == "-- Select a Chart --":
            st.info("📊 **Select a Chart from the List Above**")
        else:
            fig, ax = plt.subplots(figsize=(10, 6))

            if chart_option == "Top Speed Progress":
                if 'top_speed' in df.columns and 'date' in df.columns:
                    speed_data = df[['date', 'top_speed']].dropna()
                    ax.plot(speed_data['date'], speed_data['top_speed'], marker='o', linewidth=2, markersize=8)
                    ax.set_xlabel('Date', fontsize=12)
                    ax.set_ylabel('Top Speed (mph)', fontsize=12)
                    ax.set_title('Top Speed Progress Over Time', fontsize=14, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    plt.xticks(rotation=45)

            elif chart_option == "Ball Touches Progress":
                if 'ball_touches' in df.columns and 'date' in df.columns:
                    touches_data = df[df['ball_touches'] > 0][['date', 'ball_touches']].dropna()
                    ax.plot(touches_data['date'], touches_data['ball_touches'], marker='s', linewidth=2, markersize=8, color='green')
                    ax.set_xlabel('Date', fontsize=12)
                    ax.set_ylabel('Ball Touches', fontsize=12)
                    ax.set_title('Ball Touches Progress Over Time', fontsize=14, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    plt.xticks(rotation=45)

            elif chart_option == "Sprint Distance Progress":
                if 'sprint_distance' in df.columns and 'date' in df.columns:
                    sprint_data = df[['date', 'sprint_distance']].dropna()
                    ax.plot(sprint_data['date'], sprint_data['sprint_distance'], marker='^', linewidth=2, markersize=8, color='orange')
                    ax.set_xlabel('Date', fontsize=12)
                    ax.set_ylabel('Sprint Distance (yards)', fontsize=12)
                    ax.set_title('Sprint Distance Progress Over Time', fontsize=14, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    plt.xticks(rotation=45)

            elif chart_option == "Kicking Power Progress":
                if 'left_kicking_power_mph' in df.columns or 'right_kicking_power_mph' in df.columns:
                    left_vals = pd.to_numeric(df['left_kicking_power_mph'], errors='coerce') if 'left_kicking_power_mph' in df.columns else pd.Series([None]*len(df))
                    right_vals = pd.to_numeric(df['right_kicking_power_mph'], errors='coerce') if 'right_kicking_power_mph' in df.columns else pd.Series([None]*len(df))

                    mask = left_vals.notna() | right_vals.notna()
                    df_filtered = df[mask].copy()

                    if 'left_kicking_power_mph' in df.columns:
                        left_filtered = left_vals[mask]
                        ax.plot(df_filtered['date'], left_filtered, marker='s', linewidth=2, markersize=7, label='Left Foot', color='#3498db')

                    if 'right_kicking_power_mph' in df.columns:
                        right_filtered = right_vals[mask]
                        ax.plot(df_filtered['date'], right_filtered, marker='^', linewidth=2, markersize=7, label='Right Foot', color='#e74c3c')

                    ax.set_xlabel('Date', fontsize=12)
                    ax.set_ylabel('Kicking Power (mph)', fontsize=12)
                    ax.set_title('Kicking Power Progress (Left vs Right)', fontsize=14, fontweight='bold')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.xticks(rotation=45)

            elif chart_option == "Agility Performance":
                if 'intense_turns' in df.columns and 'date' in df.columns:
                    agility_data = df[['date', 'intense_turns']].dropna()
                    ax.plot(agility_data['date'], agility_data['intense_turns'], marker='D', linewidth=2, markersize=8, color='purple')
                    ax.axhline(y=10, color='r', linestyle='--', label='Elite Target (10+)')
                    ax.set_xlabel('Date', fontsize=12)
                    ax.set_ylabel('Intense Turns', fontsize=12)
                    ax.set_title('Intense Turns Over Time', fontsize=14, fontweight='bold')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.xticks(rotation=45)

            elif chart_option == "Turn Speed Analysis":
                if 'avg_turn_entry' in df.columns and 'avg_turn_exit' in df.columns:
                    entry_data = pd.to_numeric(df['avg_turn_entry'], errors='coerce')
                    exit_data = pd.to_numeric(df['avg_turn_exit'], errors='coerce')

                    mask = entry_data.notna() | exit_data.notna()
                    df_filtered = df[mask].copy()

                    ax.plot(df_filtered['date'], entry_data[mask], marker='o', linewidth=2, markersize=7, label='Entry Speed', color='#3498db')
                    ax.plot(df_filtered['date'], exit_data[mask], marker='s', linewidth=2, markersize=7, label='Exit Speed', color='#e74c3c')
                    ax.set_xlabel('Date', fontsize=12)
                    ax.set_ylabel('Speed (mph)', fontsize=12)
                    ax.set_title('Turn Entry vs Exit Speed', fontsize=14, fontweight='bold')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.xticks(rotation=45)

            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

# Tab 4: Speed
with tab4:
    st.header("⚡ Speed Analysis")

    if st.session_state.df is None or len(st.session_state.df) == 0:
        st.warning("📊 No data loaded. Please upload your Excel file in the sidebar.")
    else:
        df = st.session_state.df.copy()

        st.info("**What is Speed?**\n\nSpeed measures explosive power, acceleration, and top-end velocity in training sessions.")

        # Coach filter
        if 'coach' in df.columns:
            coaches = df['coach'].dropna().unique().tolist()
            # Replace "Solo" with "No Coach" in the display
            coaches_display = ["No Coach" if str(c).lower() == "solo" else c for c in coaches]
            coaches_display_map = dict(zip(coaches_display, coaches))

            selected_coach_display = st.selectbox(
                "Filter by Coach",
                ["All Coaches"] + coaches_display,
                key="speed_coach_filter"
            )

            if selected_coach_display != "All Coaches":
                selected_coach = coaches_display_map[selected_coach_display]
                df = df[df['coach'] == selected_coach]

        # Time filter with date range display
        col1, col2 = st.columns([1, 2])
        with col1:
            speed_time_filter = st.radio(
                "Time Period",
                ["All Time", "Last 30 Days"],
                horizontal=True,
                key="speed_time_filter"
            )

        # Filter data based on selection and show date range
        total_sessions = len(df)
        if speed_time_filter == "Last 30 Days" and 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            cutoff_date = datetime.now() - pd.Timedelta(days=30)
            df = df[df['date'] >= cutoff_date]

            if len(df) > 0 and df['date'].notna().any():
                date_min = df['date'].min().strftime('%b %d, %Y')
                date_max = df['date'].max().strftime('%b %d, %Y')
                with col2:
                    st.markdown(f"**📅 {date_min} - {date_max}** ({len(df)} of {total_sessions} sessions)")
            else:
                with col2:
                    st.markdown(f"**📊 {len(df)} of {total_sessions} sessions**")
        else:
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                if df['date'].notna().any():
                    date_min = df['date'].min().strftime('%b %d, %Y')
                    date_max = df['date'].max().strftime('%b %d, %Y')
                    with col2:
                        st.markdown(f"**📅 {date_min} - {date_max}** ({total_sessions} sessions)")
                else:
                    with col2:
                        st.markdown(f"**📊 {total_sessions} sessions**")
            else:
                with col2:
                    st.markdown(f"**📊 {total_sessions} sessions**")

        # Calculate speed statistics
        speed_metrics = [
            ('top_speed', 'Top Speed (mph)', '🚀 Maximum velocity achieved'),
            ('num_sprints', 'Sprints', '💨 Number of sprint efforts'),
            ('sprint_distance', 'Sprint Distance (yards)', '🏃 Total distance at sprint speed'),
            ('accelerations', 'Accl/Decl', '⚡ Explosive movements and changes of pace'),
        ]

        cols = st.columns(3)
        for idx, (col_name, label, description) in enumerate(speed_metrics):
            with cols[idx % 3]:
                if col_name in df.columns:
                    values = pd.to_numeric(df[col_name], errors='coerce').dropna()
                    if len(values) > 0:
                        avg_val = values.mean()
                        best_val = values.max()

                        st.metric(f"{label} (avg)", f"{avg_val:.1f}", delta=f"Best: {best_val:.1f}")
                        st.caption(description)

# Tab 5: Agility
with tab5:
    st.header("🔄 Agility Analysis")

    if st.session_state.df is None or len(st.session_state.df) == 0:
        st.warning("📊 No data loaded. Please upload your Excel file in the sidebar.")
    else:
        df = st.session_state.df.copy()

        st.info("**What is Agility?**\n\nAgility is the ability to respond to game actions fast through quick turns or changes in pace.")

        # Coach filter
        if 'coach' in df.columns:
            coaches = df['coach'].dropna().unique().tolist()
            # Replace "Solo" with "No Coach" in the display
            coaches_display = ["No Coach" if str(c).lower() == "solo" else c for c in coaches]
            coaches_display_map = dict(zip(coaches_display, coaches))

            selected_coach_display = st.selectbox(
                "Filter by Coach",
                ["All Coaches"] + coaches_display,
                key="agility_coach_filter"
            )

            if selected_coach_display != "All Coaches":
                selected_coach = coaches_display_map[selected_coach_display]
                df = df[df['coach'] == selected_coach]

        # Time filter with date range display
        col1, col2 = st.columns([1, 2])
        with col1:
            agility_time_filter = st.radio(
                "Time Period",
                ["All Time", "Last 30 Days"],
                horizontal=True,
                key="agility_time_filter"
            )

        # Filter data based on selection and show date range
        total_sessions = len(df)
        if agility_time_filter == "Last 30 Days" and 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            cutoff_date = datetime.now() - pd.Timedelta(days=30)
            df = df[df['date'] >= cutoff_date]

            if len(df) > 0 and df['date'].notna().any():
                date_min = df['date'].min().strftime('%b %d, %Y')
                date_max = df['date'].max().strftime('%b %d, %Y')
                with col2:
                    st.markdown(f"**📅 {date_min} - {date_max}** ({len(df)} of {total_sessions} sessions)")
            else:
                with col2:
                    st.markdown(f"**📊 {len(df)} of {total_sessions} sessions**")
        else:
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                if df['date'].notna().any():
                    date_min = df['date'].min().strftime('%b %d, %Y')
                    date_max = df['date'].max().strftime('%b %d, %Y')
                    with col2:
                        st.markdown(f"**📅 {date_min} - {date_max}** ({total_sessions} sessions)")
                else:
                    with col2:
                        st.markdown(f"**📊 {total_sessions} sessions**")
            else:
                with col2:
                    st.markdown(f"**📊 {total_sessions} sessions**")

        # Calculate agility statistics
        agility_metrics = [
            ('intense_turns', 'Intense Turns', '⚡ MOST IMPORTANT - Direction changes at 9+ mph'),
            ('avg_turn_exit', 'Turn Exit Speed (mph)', '🎯 Speed coming out of turns - Higher exit than entry = explosive power'),
            ('avg_turn_entry', 'Turn Entry Speed (mph)', '💨 Speed going into turns - Higher = attacking at speed'),
            ('right_turns', 'Right Turns', '➡️ Cuts to right (30-150°)'),
            ('left_turns', 'Left Turns', '⬅️ Cuts to left (30-150°)'),
            ('back_turns', 'Back Turns', '↩️ Direction reversals (150°+)'),
        ]

        cols = st.columns(3)
        for idx, (col_name, label, description) in enumerate(agility_metrics):
            with cols[idx % 3]:
                if col_name in df.columns:
                    values = pd.to_numeric(df[col_name], errors='coerce').dropna()
                    if len(values) > 0:
                        avg_val = values.mean()
                        best_val = values.max()

                        st.metric(f"{label} (avg)", f"{avg_val:.1f}", delta=f"Best: {best_val:.1f}")
                        st.caption(description)

# Tab 6: Ball Work
with tab6:
    st.header("⚽ Ball Work Analysis")

    if st.session_state.df is None or len(st.session_state.df) == 0:
        st.warning("📊 No data loaded. Please upload your Excel file in the sidebar.")
    else:
        df = st.session_state.df.copy()

        st.info("**What is Ball Work?**\n\nBall work measures technical skill development through foot touches, two-footed ability, and kicking power.")

        # Coach filter
        if 'coach' in df.columns:
            coaches = df['coach'].dropna().unique().tolist()
            # Replace "Solo" with "No Coach" in the display
            coaches_display = ["No Coach" if str(c).lower() == "solo" else c for c in coaches]
            coaches_display_map = dict(zip(coaches_display, coaches))

            selected_coach_display = st.selectbox(
                "Filter by Coach",
                ["All Coaches"] + coaches_display,
                key="ball_coach_filter"
            )

            if selected_coach_display != "All Coaches":
                selected_coach = coaches_display_map[selected_coach_display]
                df = df[df['coach'] == selected_coach]

        # Time filter with date range display
        col1, col2 = st.columns([1, 2])
        with col1:
            ball_time_filter = st.radio(
                "Time Period",
                ["All Time", "Last 30 Days"],
                horizontal=True,
                key="ball_time_filter"
            )

        # Filter data based on selection and show date range
        total_sessions = len(df)
        if ball_time_filter == "Last 30 Days" and 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            cutoff_date = datetime.now() - pd.Timedelta(days=30)
            df = df[df['date'] >= cutoff_date]

            if len(df) > 0 and df['date'].notna().any():
                date_min = df['date'].min().strftime('%b %d, %Y')
                date_max = df['date'].max().strftime('%b %d, %Y')
                with col2:
                    st.markdown(f"**📅 {date_min} - {date_max}** ({len(df)} of {total_sessions} sessions)")
            else:
                with col2:
                    st.markdown(f"**📊 {len(df)} of {total_sessions} sessions**")
        else:
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                if df['date'].notna().any():
                    date_min = df['date'].min().strftime('%b %d, %Y')
                    date_max = df['date'].max().strftime('%b %d, %Y')
                    with col2:
                        st.markdown(f"**📅 {date_min} - {date_max}** ({total_sessions} sessions)")
                else:
                    with col2:
                        st.markdown(f"**📊 {total_sessions} sessions**")
            else:
                with col2:
                    st.markdown(f"**📊 {total_sessions} sessions**")

        ball_metrics = [
            ('ball_touches', 'Total Ball Touches', '📊 Overall volume per session'),
            ('left_touches', 'Left Foot Touches', '⬅️ Weak foot development'),
            ('right_touches', 'Right Foot Touches', '➡️ Dominant foot'),
            ('left_releases', 'Left Foot Releases', '🎯 Weak foot passes/shots'),
            ('right_releases', 'Right Foot Releases', '🎯 Dominant passes/shots'),
            ('left_kicking_power_mph', 'Left Foot Power (mph)', '💪 Weak foot striking'),
            ('right_kicking_power_mph', 'Right Foot Power (mph)', '💪 Dominant striking'),
        ]

        cols = st.columns(3)
        for idx, (col_name, label, description) in enumerate(ball_metrics):
            with cols[idx % 3]:
                if col_name in df.columns:
                    values = pd.to_numeric(df[col_name], errors='coerce').dropna()
                    if len(values) > 0:
                        avg_val = values.mean()
                        best_val = values.max()

                        st.metric(f"{label} (avg)", f"{avg_val:.1f}", delta=f"Best: {best_val:.1f}")
                        st.caption(description)

        # L/R Ratio
        if 'left_touches' in df.columns and 'right_touches' in df.columns:
            left = pd.to_numeric(df['left_touches'], errors='coerce')
            right = pd.to_numeric(df['right_touches'], errors='coerce')
            valid_mask = (left > 0) & (right > 0)

            if valid_mask.any():
                ratios = left[valid_mask] / right[valid_mask]
                avg_ratio = ratios.mean()
                best_ratio = ratios.max()

                st.metric("L/R Touch Ratio (avg)", f"{avg_ratio:.2f}", delta=f"Best: {best_ratio:.2f}")
                st.caption("⚖️ Target: ≥ 0.5 for balance")

# Tab 7: Match Play
with tab7:
    st.header("⚽ Match Play Analysis")

    if st.session_state.df is None or len(st.session_state.df) == 0:
        st.warning("📊 No data loaded. Please upload your Excel file in the sidebar.")
    else:
        df = st.session_state.df.copy()

        # Filter to only Match training types
        if 'training_type' in df.columns:
            match_df = df[df['training_type'].str.contains('Match', na=False, case=False)]

            if len(match_df) == 0:
                st.info("⚽ No match data found. Match data is recorded when training type contains 'Match' (Match-Grass, Match-Turf, Match-Hard).")
            else:
                df = match_df

                st.info("**What is Match Play?**\n\nMatch play tracks performance during actual game situations, including position played, goals, assists, work rate, and ball possessions.")

                # Coach filter
                if 'coach' in df.columns:
                    coaches = df['coach'].dropna().unique().tolist()
                    # Replace "Solo" with "No Coach" in the display
                    coaches_display = ["No Coach" if str(c).lower() == "solo" else c for c in coaches]
                    coaches_display_map = dict(zip(coaches_display, coaches))

                    selected_coach_display = st.selectbox(
                        "Filter by Coach",
                        ["All Coaches"] + coaches_display,
                        key="match_coach_filter"
                    )

                    if selected_coach_display != "All Coaches":
                        selected_coach = coaches_display_map[selected_coach_display]
                        df = df[df['coach'] == selected_coach]

                # Time filter with date range display
                col1, col2 = st.columns([1, 2])
                with col1:
                    match_time_filter = st.radio(
                        "Time Period",
                        ["All Time", "Last 30 Days"],
                        horizontal=True,
                        key="match_time_filter"
                    )

                # Filter data based on selection and show date range
                total_sessions = len(df)
                if match_time_filter == "Last 30 Days" and 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    cutoff_date = datetime.now() - pd.Timedelta(days=30)
                    df = df[df['date'] >= cutoff_date]

                    if len(df) > 0 and df['date'].notna().any():
                        date_min = df['date'].min().strftime('%b %d, %Y')
                        date_max = df['date'].max().strftime('%b %d, %Y')
                        with col2:
                            st.markdown(f"**📅 {date_min} - {date_max}** ({len(df)} of {total_sessions} sessions)")
                    else:
                        with col2:
                            st.markdown(f"**📊 {len(df)} of {total_sessions} sessions**")
                else:
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'], errors='coerce')
                        if df['date'].notna().any():
                            date_min = df['date'].min().strftime('%b %d, %Y')
                            date_max = df['date'].max().strftime('%b %d, %Y')
                            with col2:
                                st.markdown(f"**📅 {date_min} - {date_max}** ({total_sessions} sessions)")
                        else:
                            with col2:
                                st.markdown(f"**📊 {total_sessions} sessions**")
                    else:
                        with col2:
                            st.markdown(f"**📊 {total_sessions} sessions**")

                # Specific session selector
                if 'date' in df.columns and 'session_name' in df.columns:
                    # Create session labels with date and session name
                    df_temp = df.copy()
                    df_temp['date'] = pd.to_datetime(df_temp['date'], errors='coerce')
                    df_temp = df_temp.sort_values('date', ascending=False)

                    session_options = []
                    session_indices = []
                    for idx, row in df_temp.iterrows():
                        if pd.notna(row['date']) and pd.notna(row['session_name']):
                            date_str = row['date'].strftime('%b %d, %Y')
                            session_label = f"{date_str} - {row['session_name']}"
                            session_options.append(session_label)
                            session_indices.append(idx)

                    if len(session_options) > 0:
                        selected_session = st.selectbox(
                            "Select a Specific Match",
                            ["All Matches"] + session_options,
                            key="match_session_filter"
                        )

                        if selected_session != "All Matches":
                            # Find the selected session index
                            selected_idx = session_indices[session_options.index(selected_session)]
                            df = df[df.index == selected_idx]

                # Match-Specific KPIs
                st.subheader("🎯 Match Performance Indicators")

                match_metrics = [
                    ('position', 'Position', '📍 Most played position'),
                    ('goals', 'Goals', '⚽ Goals scored'),
                    ('assists', 'Assists', '🎯 Assists made'),
                    ('work_rate', 'Work Rate (yd/min)', '💪 Effort level'),
                    ('ball_possessions', 'Ball Possessions', '🏃 Time on ball'),
                ]

                cols = st.columns(5)
                for idx, (col_name, label, description) in enumerate(match_metrics):
                    with cols[idx]:
                        if col_name in df.columns:
                            if col_name == 'position':
                                # For position, show most common (filter out empty strings)
                                values = df[col_name].dropna()
                                values = values[values.str.strip() != ''] if len(values) > 0 else values
                                if len(values) > 0:
                                    most_common = values.mode()
                                    if len(most_common) > 0:
                                        st.metric(label, str(most_common[0]))
                                        st.caption(description)
                                    else:
                                        st.metric(label, "N/A")
                                        st.caption(description)
                                else:
                                    st.metric(label, "N/A")
                                    st.caption(description)
                            elif col_name == 'work_rate':
                                # For work rate, show average (numeric field)
                                values = pd.to_numeric(df[col_name], errors='coerce').dropna()
                                if len(values) > 0:
                                    avg_val = values.mean()
                                    st.metric(label, f"{avg_val:.2f}")
                                    st.caption(description)
                            else:
                                # For numeric fields, show total and average
                                values = pd.to_numeric(df[col_name], errors='coerce').dropna()
                                if len(values) > 0:
                                    total_val = values.sum()
                                    avg_val = values.mean()
                                    st.metric(label, f"{total_val:.0f}", delta=f"Avg: {avg_val:.1f}")
                                    st.caption(description)

                # Additional Performance Metrics
                st.subheader("📊 Overall Match Performance")

                performance_metrics = [
                    ('top_speed', 'Top Speed (mph)', '🚀 Maximum velocity'),
                    ('intense_turns', 'Intense Turns', '🔄 High-speed changes'),
                    ('ball_touches', 'Ball Touches', '⚽ Total touches'),
                    ('sprints', 'Sprints', '💨 Sprint count'),
                    ('sprint_distance', 'Sprint Distance (yd)', '🏃 Sprint yardage'),
                    ('total_distance', 'Total Distance (mi)', '📏 Ground covered'),
                    ('left_kicking_power_mph', 'Left Foot Power (mph)', '💪 Left foot striking'),
                    ('right_kicking_power_mph', 'Right Foot Power (mph)', '💪 Right foot striking'),
                    ('duration', 'Duration (min)', '⏱️ Match time'),
                ]

                cols = st.columns(3)
                for idx, (col_name, label, description) in enumerate(performance_metrics):
                    with cols[idx % 3]:
                        if col_name in df.columns:
                            values = pd.to_numeric(df[col_name], errors='coerce').dropna()
                            if len(values) > 0:
                                avg_val = values.mean()
                                best_val = values.max()
                                st.metric(f"{label} (avg)", f"{avg_val:.1f}", delta=f"Best: {best_val:.1f}")
                                st.caption(description)

                # Match Surface Breakdown
                if 'surface' in df.columns:
                    st.subheader("🌱 Surface Breakdown")
                    surface_counts = df['surface'].value_counts()
                    cols = st.columns(len(surface_counts))
                    for idx, (surface, count) in enumerate(surface_counts.items()):
                        with cols[idx]:
                            percentage = (count / len(df)) * 100
                            st.metric(f"{surface}", f"{count}", delta=f"{percentage:.0f}%")
        else:
            st.info("⚽ No training type data available. Upload data with match training types to see match analysis.")

# Tab 8: Personal Records
with tab8:
    st.header("🏆 Personal Records")

    if not st.session_state.personal_records:
        st.warning("📊 No personal records calculated. Please upload your Excel file in the sidebar.")
    else:
        records = [
            ("Top Speed", 'top_speed', "mph", "🚀"),
            ("Sprint Distance", 'sprint_distance', "yards", "🏃"),
            ("Ball Touches", 'ball_touches', "touches", "⚽"),
            ("Total Distance", 'total_distance', "miles", "📏"),
            ("Intense Turns", 'intense_turns', "turns", "🔄"),
            ("L/R Touch Ratio", 'left_right_ratio', "(best, goal: ≥0.5)", "🎯"),
            ("L/R Ratio Average", 'left_right_ratio_avg', "(all time avg)", "📊"),
        ]

        cols = st.columns(3)
        for idx, (name, key, unit, emoji) in enumerate(records):
            with cols[idx % 3]:
                value = st.session_state.personal_records.get(key, 0)
                pr_date = st.session_state.pr_dates.get(key)

                if 'ratio' in key.lower():
                    display_value = f"{value:.2f}" if value > 0 else "N/A"
                else:
                    display_value = f"{value:.1f}" if value > 0 else "N/A"

                st.metric(f"{emoji} {name}", f"{display_value} {unit}")

                if pr_date:
                    st.caption(f"📅 {pr_date.strftime('%b %d, %Y')}")

# Tab 9: AI Insights - Comprehensive Analysis
with tab9:
    st.header("🤖 AI Insights")

    if st.session_state.df is None or len(st.session_state.df) == 0:
        st.warning("📊 No data loaded. Please upload your Excel file in the sidebar.")
    else:
        st.markdown("""
        Click the button below to generate a comprehensive training insights report analyzing:
        - Executive summary of current performance
        - 30-day trend analysis
        - Performance metrics and trends
        - Coach performance analysis
        - Agility and speed development
        - Technical skills assessment
        - Personalized action plan
        - Next milestone targets
        """)

        st.markdown("---")

        # Initialize session state for insights
        if 'ai_insights_generated' not in st.session_state:
            st.session_state.ai_insights_generated = False
            st.session_state.ai_insights_report = ""

        # Button to generate insights
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔍 Generate Comprehensive AI Insights", type="primary", use_container_width=True):
                with st.spinner("🤖 Analyzing training data and generating insights..."):
                    df = st.session_state.df.copy()
                    st.session_state.ai_insights_report = analyze_training_data(df)
                    st.session_state.ai_insights_generated = True

        # Display the report if generated
        if st.session_state.ai_insights_generated and st.session_state.ai_insights_report:
            st.markdown("---")

            # Add a download button for the report
            st.download_button(
                label="📥 Download Report as Text File",
                data=st.session_state.ai_insights_report,
                file_name=f"mia_training_insights_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain"
            )

            # Display the report with word wrap
            st.text_area(
                label="AI Insights Report",
                value=st.session_state.ai_insights_report,
                height=600,
                label_visibility="collapsed"
            )

            # Add option to clear/regenerate
            if st.button("🔄 Clear Report"):
                st.session_state.ai_insights_generated = False
                st.session_state.ai_insights_report = ""
                st.rerun()

if __name__ == "__main__":
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Mia Training Tracker v1.0**")
    st.sidebar.markdown("Built with Streamlit")
