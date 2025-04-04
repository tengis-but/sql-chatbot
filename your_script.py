#!/home/ubuntu/sql_chatbot/venv/bin/python3

import tkinter as tk
from tkinter import scrolledtext
import threading
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from sqlalchemy import text
from tabulate import tabulate
import csv
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import logging
import locale
import google.generativeai as genai
from tkinter import ttk

locale.setlocale(locale.LC_ALL, '')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GOOGLE_API_KEY = "AIzaSyBPwhPFnUJ1YzshqGRL4Lc_REuMdh-8XzM"
DB_PASSWORD = "Admin123$"  # Updated password
db_uri = f"postgresql+psycopg2://tengis.t:{DB_PASSWORD}@18.204.2.2:5432/data-db"  # Updated db_uri

db = SQLDatabase.from_uri(db_uri)
try:
    with db._engine.connect() as conn:
        result = conn.execute(text("SELECT 1")).fetchone()
    logger.info("Database connection established successfully. Result: %s", result[0])
except Exception as e:
    logger.error("Failed to connect to database: %s", str(e))
    exit(1)

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", google_api_key=GOOGLE_API_KEY, temperature=1)

genai.configure(api_key=GOOGLE_API_KEY)

chat_history = []
last_table_sql = None
answered = 0
current_table_data = None
current_graph_fig = None

def sql_query_checker(sql_query, question, history_str):
    """Check and correct SQL query for valid tables, columns, and data types."""
    segment_vals, category_vals, subcategory_vals, brand_vals, item_company_vals, flavor_vals, company_name_vals, shortname_vals, subchannel_vals, financialchannelcode_vals, name_vals = get_metadata_values()
    
    # Define the schema as in simple_answer
    schema = {
        "sales": ["DimDeliveryDateID", "DimCompanyID", "Channel", "DimShipToCustomerID", "DimItemID", "DocumentType", "PaymentTerm", "HLAMOUNT", "Quantity", "Amount"],
        "customers": ["DimCustomerID", "UniqueCustomerCode", "CustomerCode", "TaxRegistrationNo", "UniqueParentCustomerCode", "ParentCustomerCode", "Name", "VATName", "CompanyCode", "CompanyName", "SalesRepCode", "RegionChannelCode", "RegionChannel", "ChannelCode", "Channel", "SubChannelCode", "SubChannel", "FinancialChannelCode", "FinancialChannel", "FinancialSubChannelCode", "FinancialSubChannel", "SRChannelCode", "SRChannel", "SegmentCode", "Segment", "RegionCode", "Region", "LoyaltyCode", "Loyalty", "ZoneCode", "Zone", "StateCode", "State", "AddressLine1", "AddressLine2", "AddressLine3", "AddressLine4", "City", "CustomerLimit", "CustomerLimitType", "RowActiveIndicator", "INSERT_DATETIME", "UPDATE_DATETIME"],
        "companies": ["DimCompanyID", "CompanyCode", "ShortCode", "ShortName", "Name", "TaxRegistrationNo", "Address"],
        "items": ["DimItemID", "ItemCode", "ItemShortCode", "Name", "Description", "Group", "SegmentCode", "SegmentShortName", "Segment", "SegmentOrder", "CategoryCode", "Category", "CategoryOrder", "SubCategoryCode", "SubCategory", "BrandCode", "Brand", "BrandOrder", "SubBrandCode", "SubBrand", "CompanyCode", "Company", "PacketTypeCode", "PacketType", "ABV", "Flavor", "PackSize", "Weight", "HL", "UOM", "CaseType", "CaseWeight", "Cases", "ItemURL", "BrandURL"],
        "calendar": ["DateKey", "Date", "JulianDate", "Year", "YearWeekNo", "Quarter", "Month", "MonthNo", "DaysInMonth", "MonthWeekNo", "WeekDay", "WeekDayNo", "Day", "YearDay", "Holiday", "DaySuffix", "DayOfWeekUSA", "RunningCountOfDayInYear", "RunningCountOfDayInQuarter", "WeekOfQuarter", "MonthOfQuarter", "QuarterName", "QuarterNameExt", "YearName", "YYYYMM", "MMYYYY", "FirstDayOfWeekInYear", "LastDayOfWeekInYear", "FirstDayOfWeek", "LastDayOfWeek", "FirstDayOfMonth", "LastDayOfMonth", "FirstDayOfQuarter", "LastDayOfQuarter", "FirstDayOfYear", "LastDayOfYear", "IsWeekday"]
    }
    
    # Simplified data type assumptions (for validation)
    numeric_columns = {"HLAMOUNT", "Quantity", "Amount", "CustomerLimit", "SegmentOrder", "CategoryOrder", "BrandOrder", "PackSize", "Weight", "HL", "CaseWeight", "Cases"}
    
    prompt = (
        f"Previous conversation:\n{history_str}\n\nValidate and correct this SQL query: '{sql_query}' for the question: '{question}'. "
        "Use ONLY the following tables and their specified columns in the company_sales_db database:\n"
        f"- sales (columns: {', '.join(schema['sales'])})\n"
        f"- customers (columns: {', '.join(schema['customers'])})\n"
        f"- companies (columns: {', '.join(schema['companies'])})\n"
        f"- items (columns: {', '.join(schema['items'])})\n"
        f"- calendar (columns: {', '.join(schema['calendar'])})\n"
        f"Metadata for filtering:\n- Item Segment values: {', '.join(segment_vals)}\n- Item Category values: {', '.join(category_vals)}\n- Item SubCategory values: {', '.join(subcategory_vals)}\n- Item Brand values: {', '.join(brand_vals)}\n- Item Company values: {', '.join(item_company_vals)}\n- Item Flavor values: {', '.join(flavor_vals)}\n- Company Name values: {', '.join(company_name_vals)}\n- Company ShortName values: {', '.join(shortname_vals)}\n- Customer SubChannel values: {', '.join(subchannel_vals)}\n- Customer FinancialChannelCode values: {', '.join(financialchannelcode_vals)}\n- Items Name values: {', '.join(name_vals)}\n"
        "Check:\n1. If tables exist in the schema.\n2. If columns exist in their respective tables.\n3. If column data types align with the query (e.g., no string operations on numeric columns like HLAMOUNT).\n"
        "If invalid:\n- Replace incorrect table names with the closest match from the schema.\n- Replace invalid column names with valid ones from the intended table, prioritizing context from the question.\n- Adjust operations to match data types (e.g., use numeric comparisons for HLAMOUNT).\n"
        "Return the corrected SQL query as plain text if changes are made, or the original query if valid. If unfixable, return 'INVALID_SQL'. No formatting or code blocks."
        "in the sql never use ALTER or UPDATE or anything that might change the date form the postgresql"
    )
    
    try:
        corrected_sql = llm.invoke(prompt).content.strip()
        corrected_sql = corrected_sql.replace("```sql", "").replace("```", "").strip()
        logger.info(f"Original SQL: {sql_query}\nCorrected SQL: {corrected_sql}")
        return corrected_sql
    except Exception as e:
        logger.error("Error in sql_query_checker: %s", str(e))
        return "INVALID_SQL"
    
def get_metadata_values():
    try:
        with db._engine.connect() as conn:
            segment_values = [row[0] for row in conn.execute(text("SELECT DISTINCT segment FROM items WHERE segment IS NOT NULL;")).fetchall()]
            category_values = [row[0] for row in conn.execute(text("SELECT DISTINCT category FROM items WHERE category IS NOT NULL;")).fetchall()]
            subcategory_values = [row[0] for row in conn.execute(text("SELECT DISTINCT subcategory FROM items WHERE subcategory IS NOT NULL;")).fetchall()]
            brand_values = [row[0] for row in conn.execute(text("SELECT DISTINCT brand FROM items WHERE brand IS NOT NULL;")).fetchall()]
            item_company_values = [row[0] for row in conn.execute(text("SELECT DISTINCT company FROM items WHERE company IS NOT NULL;")).fetchall()]
            flavor_values = [row[0] for row in conn.execute(text("SELECT DISTINCT flavor FROM items WHERE flavor IS NOT NULL;")).fetchall()]
            company_name_values = [row[0] for row in conn.execute(text("SELECT DISTINCT name FROM companies WHERE name IS NOT NULL;")).fetchall()]
            shortname_values = [row[0] for row in conn.execute(text("SELECT DISTINCT shortname FROM companies WHERE shortname IS NOT NULL;")).fetchall()]
            subchannel_values = [row[0] for row in conn.execute(text("SELECT DISTINCT subchannel FROM customers WHERE subchannel IS NOT NULL;")).fetchall()]
            financialchannelcode_values = [row[0] for row in conn.execute(text("SELECT DISTINCT financialchannelcode FROM customers WHERE financialchannelcode IS NOT NULL;")).fetchall()]
            name_values = [row[0] for row in conn.execute(text("SELECT DISTINCT name FROM items WHERE name IS NOT NULL;")).fetchall()]

        return segment_values, category_values, subcategory_values, brand_values, item_company_values, flavor_values, company_name_values, shortname_values, subchannel_values, financialchannelcode_values, name_values
    except Exception as e:
        logger.error("Error retrieving metadata values: %s", str(e))
        return [], [], [], [], [], [], [], [], [], [], []

def vis_expo_check(question):
    prompt = f"Check if the sentence '{question}' is asking for: 1. A simple answer (no graph or table), 2. To be graphed, 3. To retrieve data ('yes' only if it’s related to showing a table), 4. To export data. Return four values: 'yes' or 'no' for simple answer, graphing, retrieving data (table), export, e.g., 'yes no no no' or 'no no yes yes'. No formatting or code blocks."
    try:
        response = llm.invoke(prompt).content.strip()
        simple, vis, table, expo = response.split()
        print(simple,vis,table,expo)
        if vis=="yes":
            table="yes"
        if simple == "no" and vis == "no" and table == "no" and expo == "no":
            print("can you rephrase the question")
        return simple == "yes", vis == "yes", table == "yes", expo == "yes"
    except Exception:
        return False, False, False, False

def text_to_sql(question, prompt):
    try:
        response = llm.invoke(prompt).content.strip()
        logger.info("Raw LLM response: %s", response)  # Log the raw response for debugging
        sql = response.replace("```sql", "").replace("```", "").strip()
        logger.info("Cleaned SQL: %s", sql)  # Log the cleaned SQL
        if not sql:  # Check if the SQL is empty after cleaning
            logger.error("Generated SQL is empty after cleaning")
            return None
        if not (sql.upper().startswith("SELECT") or sql.upper().startswith("WITH")):
            logger.error("Generated SQL does not start with SELECT or WITH: %s", sql)
            return None
        return sql
    except Exception as e:
        logger.error("Error in text_to_sql: %s", str(e))
        return None

def simple_answer(question, retry_count=0):
    global answered
    history_str = "\n".join([f"Q: {q}\nSQL: {sql}" for q, sql in chat_history]) if chat_history else "No previous conversation."
    is_follow_up = any("explain more" in q.lower() or "tell me more" in q.lower() for q, _ in chat_history[-1:]) and len(chat_history) > 1
    segment_vals, category_vals, subcategory_vals, brand_vals, item_company_vals, flavor_vals, company_name_vals, shortname_vals, subchannel_vals, financialchannelcode_vals, name_vals = get_metadata_values()
    
    sql_prompt = (
        f"Previous conversation:\n{history_str}\n\nGenerate a PostgreSQL query for this question: '{question}'. "
        "Use ONLY the following tables and their specified columns in the company_sales_db database:\n"
        "- sales (columns: DimDeliveryDateID, DimCompanyID, Channel, DimShipToCustomerID, DimItemID, DocumentType, PaymentTerm, HLAMOUNT, Quantity, Amount)\n"
        "- customers (columns: DimCustomerID, UniqueCustomerCode, CustomerCode, TaxRegistrationNo, UniqueParentCustomerCode, ParentCustomerCode, Name, VATName, CompanyCode, CompanyName, SalesRepCode, RegionChannelCode, RegionChannel, ChannelCode, Channel, SubChannelCode, SubChannel, FinancialChannelCode, FinancialChannel, FinancialSubChannelCode, FinancialSubChannel, SRChannelCode, SRChannel, SegmentCode, Segment, RegionCode, Region, LoyaltyCode, Loyalty, ZoneCode, Zone, StateCode, State, AddressLine1, AddressLine2, AddressLine3, AddressLine4, City, CustomerLimit, CustomerLimitType, RowActiveIndicator, INSERT_DATETIME, UPDATE_DATETIME)\n"
        "- companies (columns: DimCompanyID, CompanyCode, ShortCode, ShortName, Name, TaxRegistrationNo, Address)\n"
        "- items (columns: DimItemID, ItemCode, ItemShortCode, Name, Description, Group, SegmentCode, SegmentShortName, Segment, SegmentOrder, CategoryCode, Category, CategoryOrder, SubCategoryCode, SubCategory, BrandCode, Brand, BrandOrder, SubBrandCode, SubBrand, CompanyCode, Company, PacketTypeCode, PacketType, ABV, Flavor, PackSize, Weight, HL, UOM, CaseType, CaseWeight, Cases, ItemURL, BrandURL)\n"
        "- calendar (columns: DateKey, Date, JulianDate, Year, YearWeekNo, Quarter, Month, MonthNo, DaysInMonth, MonthWeekNo, WeekDay, WeekDayNo, Day, YearDay, Holiday, DaySuffix, DayOfWeekUSA, RunningCountOfDayInYear, RunningCountOfDayInQuarter, WeekOfQuarter, MonthOfQuarter, QuarterName, QuarterNameExt, YearName, YYYYMM, MMYYYY, FirstDayOfWeekInYear, LastDayOfWeekInYear, FirstDayOfWeek, LastDayOfWeek, FirstDayOfMonth, LastDayOfMonth, FirstDayOfQuarter, LastDayOfQuarter, FirstDayOfYear, LastDayOfYear, IsWeekday)\n"
        f"Metadata for filtering:\n- Item Segment values: {', '.join(segment_vals)}\n- Item Category values: {', '.join(category_vals)}\n- Item SubCategory values: {', '.join(subcategory_vals)}\n- Item Brand values: {', '.join(brand_vals)}\n- Item Company values: {', '.join(item_company_vals)}\n- Item Flavor values: {', '.join(flavor_vals)}\n- Company Name values: {', '.join(company_name_vals)}\n- Company ShortName values: {', '.join(shortname_vals)}\n- Customer SubChannel values: {', '.join(subchannel_vals)}\n- Customer FinancialChannelCode values: {', '.join(financialchannelcode_vals)}\n- items name{', '.join(name_vals)}"
        "Use these metadata values to filter data accurately (e.g., Category = 'Vodka' for vodka items). Do not reference any tables or columns outside of those listed above. "
        "For product names, use items.Name; for quantities, use sales.Quantity; for dates, use sales.DimDeliveryDateID joined with calendar.Date. "
        "Carefully consider the chat history to maintain context:\n- If the previous question mentions a specific company, customer, or item attribute, apply it unless overridden.\n- If an aggregation (e.g., SUM, AVG) was used previously, assume it unless specified otherwise.\n- Focus on the intent of the current question while respecting conversation flow.\n"
        "If the question asks for a list (e.g., 'list all vodka items'), use DISTINCT.\nJoin tables as needed (e.g., sales with calendar on DimDeliveryDateID = DateKey).\nUse WITH clauses (CTEs) when the query involves complex subqueries or reusable intermediate results for clarity and efficiency; otherwise, use SELECT.\nEnsure the query starts with SELECT or WITH; if it doesn’t, return 'INVALID_SQL' instead.\n"
        "After generating the SQL query, provide two explanations:\n- A short explanation (20-30 words) of what the query does, including the main table and column(s) used and the calculation method.\n- A detailed explanation (60 words) for if the user asks for more details later, covering the table(s), column(s), joins, and calculation in depth.\n"
        "Return the SQL query followed by 'SHORT_EXPLANATION:' and the short explanation, then 'DETAILED_EXPLANATION:' and the detailed explanation, separated by newlines, no formatting or code blocks."
        "in the sql never use ALTER or UPDATE or anything that might change the date form the postgresql"
    )
    sql_response = text_to_sql(question, prompt)

    try:
        parts = sql_response.split("SHORT_EXPLANATION:")
        sql_query = parts[0].strip()
        short_and_detailed = parts[1].split("DETAILED_EXPLANATION:")
        short_explanation = short_and_detailed[0].strip()
        detailed_explanation = short_and_detailed[1].strip() if len(short_and_detailed) > 1 else "No detailed explanation available."
    except (IndexError, ValueError):
        sql_query = sql_response.strip()
        short_explanation = "I pulled this for the table, but couldn’t clarify the details."
        detailed_explanation = "Sorry, I couldn’t generate a detailed breakdown for this table."

    sql_response = sql_query_checker(sql_query, question, history_str)
    
    if not sql_response or sql_response.strip() == "INVALID_SQL":
        return "Oops, couldn’t get the query right for the table!"

    if not (sql_query.upper().startswith("SELECT") or sql_query.upper().startswith("WITH")):
        return "Oops, I messed up the query. Can you try asking that another way?"

    try:
        with db._engine.connect() as conn:
            result = conn.execute(text(sql_query))
            rows = result.fetchall()
            if not rows:
                if retry_count < 1:
                    return table_answer(question, retry_count + 1)
                chat_history.append((question, "No data returned"))
                return "I checked the database twice, but there’s nothing to show here."
            headers = list(result.keys())
            unique_headers = []
            header_count = {}
            for header in headers:
                if header in header_count:
                    header_count[header] += 1
                    unique_header = f"{header}_{header_count[header]}"
                else:
                    header_count[header] = 1
                    unique_header = header
                unique_headers.append(unique_header)
            cleaned_rows = [[str(col).strip() if col is not None else "Null" for col in row] for row in rows]
            current_table_data = (unique_headers, cleaned_rows)
            explanation = detailed_explanation if is_follow_up and question.lower() in ["explain more", "tell me more"] else short_explanation
            
            # Check row count and adjust response
            total_rows = len(cleaned_rows)
            display_limit = 10
            if total_rows > display_limit:
                response = (
                    f"Here’s your table (see left panel, showing {display_limit} of {total_rows} rows):\n"
                    f"{explanation}\n"
                    f"Want to see the full data? Export it using the 'CSV Экспорт' button!"
                )
            else:
                response = f"Here’s your table (see left panel):\n{explanation}\n{'Need more details?' if not is_follow_up else 'Hope that clears it up!'}"
            
            chat_history.append((question, sql_query))
            return response
    except Exception as e:
        logger.error("Error executing SQL in table_answer: %s", str(e))
        return "Sorry, something went wrong with the table!"

def graph_answer(question, retry_count=0):
    global last_table_sql, current_graph_fig, current_table_data
    history_str = "\n".join([f"Q: {q}\nSQL: {sql}" for q, sql in chat_history]) if chat_history else "No previous conversation."
    is_follow_up = any("explain more" in q.lower() or "tell me more" in q.lower() for q, _ in chat_history[-1:]) and len(chat_history) > 1
    
    # Check if table data exists
    if not current_table_data:
        return "No table data available to graph. Please generate a table first!"

    headers, rows = current_table_data
    available_columns = ", ".join(headers)

    # Prompt to determine x_title, y_title, and chart type
    prompt = (
        f"Previous conversation:\n{history_str}\n\nAnalyze the following sentence: '{question}'. "
        f"The available table data has these columns: {available_columns}. "
        "Check if the question specifies 'graph x,y' with defined x and y variables matching these columns. "
        "If specified, use those as x_title and y_title. "
        "If not specified, recommend the best x-axis title and y-axis title from the available columns based on the question’s intent. "
        "For time-based data, prefer a date column (e.g., 'Date') for x-axis; for values, prefer a numeric column (e.g., 'Quantity', 'Amount'). "
        "Identify the chart type: 'line_chart', 'pie_chart', or 'graph_chart' (default to 'graph_chart' if unspecified). "
        "Output: 'x_title y_title type_of_chart' as plain text."
        "dont change any of the data show the data as it is"
        "in the sql never use ALTER or UPDATE or anything that might change the date form the postgresql"
    )
    try:
        response = llm.invoke(prompt).content.strip().lower()
        x_title, y_title, type_of_chart = response.split()
        print(x_title, y_title)
        # Validate that x_title and y_title exist in headers
        if x_title not in headers or y_title not in headers:
            return f"Oops, '{x_title}' or '{y_title}' isn’t in the table data ({available_columns}). Try specifying valid columns!"
    except Exception as e:
        logger.error("Error determining graph titles: %s", str(e))
        return "Hmm, I couldn’t sort out the graph details."

    # Limit to maximum 100 rows
    max_graph_rows = 100
    if len(rows) > max_graph_rows:
        limited_rows = rows[:max_graph_rows]
    else:
        limited_rows = rows

    try:
        if not limited_rows:
            if retry_count < 1:
                return graph_answer(question, retry_count + 1)
            chat_history.append((question, "No data returned"))
            return "Looks like there’s no data to graph, even after a second try."

        # Extract x_data and y_data from limited_rows
        x_idx = headers.index(x_title)
        y_idx = headers.index(y_title)
        x_data = [row[x_idx] for row in limited_rows]
        y_data = [row[y_idx] for row in limited_rows]

        # Convert y_data to numeric if possible, handle non-numeric gracefully
        try:
            y_data = [float(y) if y != "Null" else 0 for y in y_data]
        except ValueError:
            return f"Error: '{y_title}' data ({', '.join(y_data[:5])}...) isn’t numeric for graphing!"

        # Create the graph with a wider figure size
        fig = plt.Figure(figsize=(10, 4))  # Increased width from 6 to 10 for more horizontal space
        ax = fig.add_subplot(111)

        # Plot the data
        if type_of_chart == "line_chart":
            ax.plot(x_data, y_data, marker='o', linestyle='-', color='b', label=f"{y_title} vs {x_title}")
        elif type_of_chart == "pie_chart":
            ax.pie(y_data, labels=x_data, autopct='%1.1f%%', startangle=90)
        else:
            ax.scatter(x_data, y_data, color='g', label=f"{y_title} vs {x_title}")

        # Set labels and title
        ax.set_xlabel(x_title)
        ax.set_ylabel(y_title)
        ax.set_title(f"{y_title} vs {x_title}")
        ax.legend()
        ax.grid(True)

        # Dynamically adjust x-axis labels to prevent overlap
        num_data_points = len(x_data)
        if num_data_points > 10:  # If more than 10 points, reduce label frequency
            step = max(1, num_data_points // 10)  # Show approximately 10 labels
            ax.set_xticks(range(0, num_data_points, step))
            ax.set_xticklabels([x_data[i] for i in range(0, num_data_points, step)], rotation=45, ha='right')
        else:
            ax.set_xticks(range(num_data_points))
            ax.set_xticklabels(x_data, rotation=45, ha='right')

        # Adjust layout to prevent label cutoff
        fig.tight_layout()

        current_graph_fig = fig

        # Generate explanations based on table data usage
        short_explanation = f"This graphs {y_title} against {x_title} using the current table data (up to {max_graph_rows} rows)."
        detailed_explanation = (
            f"This uses the existing table data with columns {available_columns}. "
            f"It plots {y_title} (y-axis) against {x_title} (x-axis) from up to {max_graph_rows} rows retrieved previously, "
            f"assuming the data reflects the question’s intent."
            "in the sql never use ALTER or UPDATE or anything that might change the date form the postgresql"
    )
        explanation = detailed_explanation if is_follow_up and question.lower() in ["explain more", "tell me more"] else short_explanation
        chat_history.append((question, f"Graph using table data: {x_title} vs {y_title} (limited to {max_graph_rows} rows)"))
        return f"Here’s the graph (see left panel):\n{explanation}\n{'Need more details?' if not is_follow_up else 'Hope that clears it up!'}"
    except Exception as e:
        logger.error("Error in graph_answer: %s", str(e))
        return "Sorry, the graph didn’t work out!"
    

def table_answer(question, retry_count=0):
    global last_table_sql, current_table_data
    history_str = "\n".join([f"Q: {q}\nSQL: {sql}" for q, sql in chat_history]) if chat_history else "No previous conversation."
    is_follow_up = any("explain more" in q.lower() or "tell me more" in q.lower() for q, _ in chat_history[-1:]) and len(chat_history) > 1
    segment_vals, category_vals, subcategory_vals, brand_vals, item_company_vals, flavor_vals, company_name_vals, shortname_vals, subchannel_vals, financialchannelcode_vals, name_vals = get_metadata_values()
    
    # Prompt logic for generating SQL (unchanged)
    if "from that table" in question.lower() and chat_history:
        base_sql = chat_history[-1][1]
        prompt = (
            f"Previous conversation:\n{history_str}\n\nTake this base SQL query from the previous question: '{base_sql}'. Modify it to answer this question: '{question}'. "
            "Use ONLY the following tables and their specified columns in the company_sales_db database:\n"
            "- sales (columns: DimDeliveryDateID, DimCompanyID, Channel, DimShipToCustomerID, DimItemID, DocumentType, PaymentTerm, HLAMOUNT, Quantity, Amount)\n"
            "- customers (columns: DimCustomerID, UniqueCustomerCode, CustomerCode, TaxRegistrationNo, UniqueParentCustomerCode, ParentCustomerCode, Name, VATName, CompanyCode, CompanyName, SalesRepCode, RegionChannelCode, RegionChannel, ChannelCode, Channel, SubChannelCode, SubChannel, FinancialChannelCode, FinancialChannel, FinancialSubChannelCode, FinancialSubChannel, SRChannelCode, SRChannel, SegmentCode, Segment, RegionCode, Region, LoyaltyCode, Loyalty, ZoneCode, Zone, StateCode, State, AddressLine1, AddressLine2, AddressLine3, AddressLine4, City, CustomerLimit, CustomerLimitType, RowActiveIndicator, INSERT_DATETIME, UPDATE_DATETIME)\n"
            "- companies (columns: DimCompanyID, CompanyCode, ShortCode, ShortName, Name, TaxRegistrationNo, Address)\n"
            "- items (columns: DimItemID, ItemCode, ItemShortCode, Name, Description, Group, SegmentCode, SegmentShortName, Segment, SegmentOrder, CategoryCode, Category, CategoryOrder, SubCategoryCode, SubCategory, BrandCode, Brand, BrandOrder, SubBrandCode, SubBrand, CompanyCode, Company, PacketTypeCode, PacketType, ABV, Flavor, PackSize, Weight, HL, UOM, CaseType, CaseWeight, Cases, ItemURL, BrandURL)\n"
            "- calendar (columns: DateKey, Date, JulianDate, Year, YearWeekNo, Quarter, Month, MonthNo, DaysInMonth, MonthWeekNo, WeekDay, WeekDayNo, Day, YearDay, Holiday, DaySuffix, DayOfWeekUSA, RunningCountOfDayInYear, RunningCountOfDayInQuarter, WeekOfQuarter, MonthOfQuarter, QuarterName, QuarterNameExt, YearName, YYYYMM, MMYYYY, FirstDayOfWeekInYear, LastDayOfWeekInYear, FirstDayOfWeek, LastDayOfWeek, FirstDayOfMonth, LastDayOfMonth, FirstDayOfQuarter, LastDayOfQuarter, FirstDayOfYear, LastDayOfYear, IsWeekday)\n"
            f"Metadata for filtering:\n- Item Segment values: {', '.join(segment_vals)}\n- Item Category values: {', '.join(category_vals)}\n- Item SubCategory values: {', '.join(subcategory_vals)}\n- Item Brand values: {', '.join(brand_vals)}\n- Item Company values: {', '.join(item_company_vals)}\n- Item Flavor values: {', '.join(flavor_vals)}\n- Company Name values: {', '.join(company_name_vals)}\n- Company ShortName values: {', '.join(shortname_vals)}\n- Customer SubChannel values: {', '.join(subchannel_vals)}\n- Customer FinancialChannelCode values: {', '.join(financialchannelcode_vals)}\n- Items Name values: {', '.join(name_vals)}\n"
            "Return the SQL query followed by 'SHORT_EXPLANATION:' and the short explanation, then 'DETAILED_EXPLANATION:' and the detailed explanation, separated by newlines, no formatting or code blocks."
            "in the sql never use ALTER or UPDATE or anything that might change the date form the postgresql"
    )
    else:
        prompt = (
            f"Previous conversation:\n{history_str}\n\nGenerate a PostgreSQL query for this question: '{question}'. "
            "Use ONLY the following tables and their specified columns in the company_sales_db database:\n"
            "- sales (columns: DimDeliveryDateID, DimCompanyID, Channel, DimShipToCustomerID, DimItemID, DocumentType, PaymentTerm, HLAMOUNT, Quantity, Amount)\n"
            "- customers (columns: DimCustomerID, UniqueCustomerCode, CustomerCode, TaxRegistrationNo, UniqueParentCustomerCode, ParentCustomerCode, Name, VATName, CompanyCode, CompanyName, SalesRepCode, RegionChannelCode, RegionChannel, ChannelCode, Channel, SubChannelCode, SubChannel, FinancialChannelCode, FinancialChannel, FinancialSubChannelCode, FinancialSubChannel, SRChannelCode, SRChannel, SegmentCode, Segment, RegionCode, Region, LoyaltyCode, Loyalty, ZoneCode, Zone, StateCode, State, AddressLine1, AddressLine2, AddressLine3, AddressLine4, City, CustomerLimit, CustomerLimitType, RowActiveIndicator, INSERT_DATETIME, UPDATE_DATETIME)\n"
            "- companies (columns: DimCompanyID, CompanyCode, ShortCode, ShortName, Name, TaxRegistrationNo, Address)\n"
            "- items (columns: DimItemID, ItemCode, ItemShortCode, Name, Description, Group, SegmentCode, SegmentShortName, Segment, SegmentOrder, CategoryCode, Category, CategoryOrder, SubCategoryCode, SubCategory, BrandCode, Brand, BrandOrder, SubBrandCode, SubBrand, CompanyCode, Company, PacketTypeCode, PacketType, ABV, Flavor, PackSize, Weight, HL, UOM, CaseType, CaseWeight, Cases, ItemURL, BrandURL)\n"
            "- calendar (columns: DateKey, Date, JulianDate, Year, YearWeekNo, Quarter, Month, MonthNo, DaysInMonth, MonthWeekNo, WeekDay, WeekDayNo, Day, YearDay, Holiday, DaySuffix, DayOfWeekUSA, RunningCountOfDayInYear, RunningCountOfDayInQuarter, WeekOfQuarter, MonthOfQuarter, QuarterName, QuarterNameExt, YearName, YYYYMM, MMYYYY, FirstDayOfWeekInYear, LastDayOfWeekInYear, FirstDayOfWeek, LastDayOfWeek, FirstDayOfMonth, LastDayOfMonth, FirstDayOfQuarter, LastDayOfQuarter, FirstDayOfYear, LastDayOfYear, IsWeekday)\n"
            f"Metadata for filtering:\n- Item Segment values: {', '.join(segment_vals)}\n- Item Category values: {', '.join(category_vals)}\n- Item SubCategory values: {', '.join(subcategory_vals)}\n- Item Brand values: {', '.join(brand_vals)}\n- Item Company values: {', '.join(item_company_vals)}\n- Item Flavor values: {', '.join(flavor_vals)}\n- Company Name values: {', '.join(company_name_vals)}\n- Company ShortName values: {', '.join(shortname_vals)}\n- Customer SubChannel values: {', '.join(subchannel_vals)}\n- Customer FinancialChannelCode values: {', '.join(financialchannelcode_vals)}\n- Items Name values: {', '.join(name_vals)}\n"
            "Return the SQL query followed by 'SHORT_EXPLANATION:' and the short explanation, then 'DETAILED_EXPLANATION:' and the detailed explanation, separated by newlines, no formatting or code blocks."
            "in the sql never use ALTER or UPDATE or anything that might change the date form the postgresql"
    )
    
    sql_response = text_to_sql(question, prompt)

    # Check if sql_response is None
    if sql_response is None:
        logger.error("SQL response is None, likely due to invalid SQL generation")
        return "Oops, I couldn’t generate a valid SQL query. Can you try rephrasing the question?"

    try:
        parts = sql_response.split("SHORT_EXPLANATION:")
        sql_query = parts[0].strip()
        short_and_detailed = parts[1].split("DETAILED_EXPLANATION:")
        short_explanation = short_and_detailed[0].strip()
        detailed_explanation = short_and_detailed[1].strip() if len(short_and_detailed) > 1 else "No detailed explanation available."
    except (IndexError, ValueError):
        sql_query = sql_response.strip()
        short_explanation = "I pulled this for the table, but couldn’t clarify the details."
        detailed_explanation = "Sorry, I couldn’t generate a detailed breakdown for this table."

    sql_response = sql_query_checker(sql_query, question, history_str)
    
    if not sql_response or sql_response.strip() == "INVALID_SQL":
        return "Oops, couldn’t get the query right for the table!"

    if not (sql_query.upper().startswith("SELECT") or sql_query.upper().startswith("WITH")):
        return "Oops, I messed up the query. Can you try asking that another way?"

    try:
        with db._engine.connect() as conn:
            result = conn.execute(text(sql_query))
            rows = result.fetchall()
            if not rows:
                if retry_count < 1:
                    return table_answer(question, retry_count + 1)
                chat_history.append((question, "No data returned"))
                return "I checked the database twice, but there’s nothing to show here."
            headers = list(result.keys())
            unique_headers = []
            header_count = {}
            for header in headers:
                if header in header_count:
                    header_count[header] += 1
                    unique_header = f"{header}_{header_count[header]}"
                else:
                    header_count[header] = 1
                    unique_header = header
                unique_headers.append(unique_header)
            cleaned_rows = [[str(col).strip() if col is not None else "Null" for col in row] for row in rows]
            current_table_data = (unique_headers, cleaned_rows)
            explanation = detailed_explanation if is_follow_up and question.lower() in ["explain more", "tell me more"] else short_explanation
            
            # Check row count and adjust response
            total_rows = len(cleaned_rows)
            display_limit = 20
            if total_rows > display_limit:
                response = (
                    f"Here’s your table (see left panel, showing {display_limit} of {total_rows} rows):\n"
                    f"{explanation}\n"
                    f"Want to see the full data? Export it using the 'CSV Экспорт' button!"
                )
            else:
                response = f"Here’s your table (see left panel):\n{explanation}\n{'Need more details?' if not is_follow_up else 'Hope that clears it up!'}"
            
            chat_history.append((question, sql_query))
            return response
    except Exception as e:
        logger.error("Error executing SQL in table_answer: %s", str(e))
        return "Sorry, something went wrong with the table!"

# Updated refresh_left_panel to use Treeview
def refresh_left_panel():
    for widget in left_frame.winfo_children():
        widget.destroy()

    # Display the table if current_table_data exists
    if current_table_data:
        # Create a subframe for the table and button
        table_frame = tk.Frame(left_frame, bg="#ffffff")
        table_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Export button
        export_button = tk.Button(table_frame, text="CSV Export", command=export_current_data, bg="#006400", fg="white", 
                                font=("Segoe UI", 10, "bold"), relief="flat", bd=0)
        export_button.pack(side=tk.TOP, anchor="nw", pady=(0, 5))

        headers, rows = current_table_data
        # Reduce the height to show half the table (from 20 rows to 10 rows)
        tree = ttk.Treeview(table_frame, columns=headers, show="headings", height=min(len(rows), 10))  # Changed from 20 to 10
        tree.pack(fill=tk.BOTH, expand=True)

        for col in headers:
            tree.heading(col, text=col, command=lambda c=col: sort_column(tree, c))
            tree.column(col, width=150, minwidth=100, stretch=False)

        for row in rows[:20]:  # Still insert up to 20 rows, but only 10 will be visible without scrolling
            tree.insert("", "end", values=row)

        # Vertical scrollbar for the table
        v_scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.configure(yscrollcommand=v_scrollbar.set)

        # Horizontal scrollbar for the table
        h_scrollbar = ttk.Scrollbar(table_frame, orient="horizontal", command=tree.xview)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        tree.configure(xscrollcommand=h_scrollbar.set)

        # Add vertical scroll wheel support for the Treeview
        def on_table_mouse_wheel(event):
            # Determine scroll direction (positive for up, negative for down)
            if event.delta > 0:
                tree.yview_scroll(-1, "units")  # Scroll up
            elif event.delta < 0:
                tree.yview_scroll(1, "units")   # Scroll down

        # Bind mouse wheel events (Windows uses <MouseWheel>, Linux/Mac may use <Button-4> and <Button-5>)
        tree.bind("<MouseWheel>", on_table_mouse_wheel)  # For Windows
        tree.bind("<Button-4>", lambda event: tree.yview_scroll(-1, "units"))  # For Linux (scroll up)
        tree.bind("<Button-5>", lambda event: tree.yview_scroll(1, "units"))   # For Linux (scroll down)

        # Ensure the Treeview can receive focus for mouse wheel events
        tree.bind("<Enter>", lambda event: tree.focus_set())

        # Hover event handlers for export button
        def on_enter_export(e):
            animate_color_transition(export_button, "#006400", "#abc32f", duration=150)
        def on_leave_export(e):
            animate_color_transition(export_button, "#abc32f", "#006400", duration=150)
        export_button.bind("<Enter>", on_enter_export)
        export_button.bind("<Leave>", on_leave_export)

    # Display the graph if current_graph_fig exists (below the table)
    if current_graph_fig:
        # Create a subframe for the graph and scrollbars
        graph_frame = tk.Frame(left_frame, bg="#ffffff")
        graph_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create a canvas for the graph with scrollable region
        canvas = FigureCanvasTkAgg(current_graph_fig, master=graph_frame)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        
        # Pack the canvas inside the frame
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Add horizontal scrollbar
        h_scrollbar = ttk.Scrollbar(graph_frame, orient="horizontal", command=canvas_widget.xview)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas_widget.configure(xscrollcommand=h_scrollbar.set)

        # Add vertical scrollbar (optional)
        v_scrollbar = ttk.Scrollbar(graph_frame, orient="vertical", command=canvas_widget.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas_widget.configure(yscrollcommand=v_scrollbar.set)
        
        # Configure canvas scrolling region (based on figure size)
        canvas_widget.configure(scrollregion=(0, 0, current_graph_fig.get_size_inches()[0] * 100, current_graph_fig.get_size_inches()[1] * 100))

        # Add horizontal scroll wheel support for the graph
        def on_graph_mouse_wheel(event):
            # Determine scroll direction (positive for right, negative for left)
            if event.delta > 0:
                canvas_widget.xview_scroll(-1, "units")  # Scroll left
            elif event.delta < 0:
                canvas_widget.xview_scroll(1, "units")   # Scroll right

        # Bind mouse wheel events for the graph
        canvas_widget.bind("<MouseWheel>", on_graph_mouse_wheel)  # For Windows
        canvas_widget.bind("<Button-4>", lambda event: canvas_widget.xview_scroll(-1, "units"))  # For Linux (scroll up/left)
        canvas_widget.bind("<Button-5>", lambda event: canvas_widget.xview_scroll(1, "units"))   # For Linux (scroll down/right)

        # Ensure the canvas can receive focus for mouse wheel events
        canvas_widget.bind("<Enter>", lambda event: canvas_widget.focus_set())

# Sorting functions
def sort_column(tree, col):
    items = [(tree.set(k, col), k) for k in tree.get_children()]
    # Try to sort numerically if possible, otherwise as strings
    try:
        items.sort(key=lambda x: float(x[0]) if x[0] != "Null" else float('inf'))
    except ValueError:
        items.sort()
    for index, (val, k) in enumerate(items):
        tree.move(k, "", index)
    tree.heading(col, command=lambda c=col: sort_column_reverse(tree, c))
    

def export_data(question, retry_count=0):
    global last_table_sql
    history_str = "\n".join([f"Q: {q}\nSQL: {sql}" for q, sql in chat_history]) if chat_history else "No previous conversation."
    is_follow_up = any("explain more" in q.lower() or "tell me more" in q.lower() for q, _ in chat_history[-1:]) and len(chat_history) > 1
    
    if "that table" in question.lower() and last_table_sql:
        sql_query = last_table_sql
        short_explanation = "This exports the previous table’s data from the sales database."
        detailed_explanation = "This takes the last table’s SQL query, runs it again, and exports the results from the sales database, including any joins or filters, to a CSV file."
    else:
        segment_vals, category_vals, subcategory_vals, brand_vals, item_company_vals, flavor_vals, company_name_vals, shortname_vals, subchannel_vals, financialchannelcode_vals,name_vals = get_metadata_values()
        prompt = (
            f"Previous conversation:\n{history_str}\n\nGenerate a PostgreSQL query for this question: '{question}'. "
            "Use ONLY the following tables and their specified columns in the company_sales_db database:\n"
            "- sales (columns: DimDeliveryDateID, DimCompanyID, Channel, DimShipToCustomerID, DimItemID, DocumentType, PaymentTerm, HLAMOUNT, Quantity, Amount)\n"
            "- customers (columns: DimCustomerID, UniqueCustomerCode, CustomerCode, TaxRegistrationNo, UniqueParentCustomerCode, ParentCustomerCode, Name, VATName, CompanyCode, CompanyName, SalesRepCode, RegionChannelCode, RegionChannel, ChannelCode, Channel, SubChannelCode, SubChannel, FinancialChannelCode, FinancialChannel, FinancialSubChannelCode, FinancialSubChannel, SRChannelCode, SRChannel, SegmentCode, Segment, RegionCode, Region, LoyaltyCode, Loyalty, ZoneCode, Zone, StateCode, State, AddressLine1, AddressLine2, AddressLine3, AddressLine4, City, CustomerLimit, CustomerLimitType, RowActiveIndicator, INSERT_DATETIME, UPDATE_DATETIME)\n"
            "- companies (columns: DimCompanyID, CompanyCode, ShortCode, ShortName, Name, TaxRegistrationNo, Address)\n"
            "- items (columns: DimItemID, ItemCode, ItemShortCode, Name, Description, Group, SegmentCode, SegmentShortName, Segment, SegmentOrder, CategoryCode, Category, CategoryOrder, SubCategoryCode, SubCategory, BrandCode, Brand, BrandOrder, SubBrandCode, SubBrand, CompanyCode, Company, PacketTypeCode, PacketType, ABV, Flavor, PackSize, Weight, HL, UOM, CaseType, CaseWeight, Cases, ItemURL, BrandURL)\n"
            "- calendar (columns: DateKey, Date, JulianDate, Year, YearWeekNo, Quarter, Month, MonthNo, DaysInMonth, MonthWeekNo, WeekDay, WeekDayNo, Day, YearDay, Holiday, DaySuffix, DayOfWeekUSA, RunningCountOfDayInYear, RunningCountOfDayInQuarter, WeekOfQuarter, MonthOfQuarter, QuarterName, QuarterNameExt, YearName, YYYYMM, MMYYYY, FirstDayOfWeekInYear, LastDayOfWeekInYear, FirstDayOfWeek, LastDayOfWeek, FirstDayOfMonth, LastDayOfMonth, FirstDayOfQuarter, LastDayOfQuarter, FirstDayOfYear, LastDayOfYear, IsWeekday)\n"
            f"Metadata for filtering:\n- Item Segment values: {', '.join(segment_vals)}\n- Item Category values: {', '.join(category_vals)}\n- Item SubCategory values: {', '.join(subcategory_vals)}\n- Item Brand values: {', '.join(brand_vals)}\n- Item Company values: {', '.join(item_company_vals)}\n- Item Flavor values: {', '.join(flavor_vals)}\n- Company Name values: {', '.join(company_name_vals)}\n- Company ShortName values: {', '.join(shortname_vals)}\n- Customer SubChannel values: {', '.join(subchannel_vals)}\n- Customer FinancialChannelCode values: {', '.join(financialchannelcode_vals)}\n"
            "Consider chat history context:\n- Apply previous filters unless overridden.\n- Use previous aggregations unless specified otherwise.\n- Focus on the intent of the current question.\nJoin tables as needed (e.g., sales with calendar on DimDeliveryDateID = DateKey) using only the listed tables and columns.\n"
            "For product names, use items.Name; for quantities, use sales.Quantity; for dates, use sales.DimDeliveryDateID joined with calendar.Date.\n"
            "Use WITH clauses (CTEs) when the query involves complex subqueries or reusable intermediate results for clarity and efficiency; otherwise, use SELECT.\nEnsure the query starts with SELECT or WITH; if it doesn’t, return 'INVALID_SQL'.\nReturn only the SQL query as plain text, no formatting or code blocks.\n"
            "After generating the SQL query, provide two explanations:\n- A short explanation (20-30 words) of what the query does, including the main table and column(s) used.\n- A detailed explanation (60 words) for if the user asks for more details later, covering the table(s), column(s), joins, and calculation in depth.\n"
            "Return the SQL query followed by 'SHORT_EXPLANATION:' and the short explanation, then 'DETAILED_EXPLANATION:' and the detailed explanation, separated by newlines, no formatting or code blocks."
            "in the sql never use ALTER or UPDATE or anything that might change the date form the postgresql"
    )
        sql_response = text_to_sql(question, prompt)
        
        sql_response=sql_query_checker(sql_response,question,history_str)
        if not sql_response or sql_response.strip() == "INVALID_SQL":
            return "Oops, couldn’t set up the export query!"
        
        try:
            parts = sql_response.split("SHORT_EXPLANATION:")
            sql_query = parts[0].strip()
            short_and_detailed = parts[1].split("DETAILED_EXPLANATION:")
            short_explanation = short_and_detailed[0].strip()
            detailed_explanation = short_and_detailed[1].strip() if len(short_and_detailed) > 1 else "No detailed explanation available."
        except (IndexError, ValueError):
            sql_query = sql_response.strip()
            short_explanation = "I pulled this for export, but couldn’t clarify the details."
            detailed_explanation = "Sorry, I couldn’t generate a detailed breakdown for this export."

    if not (sql_query.upper().startswith("SELECT") or sql_query.upper().startswith("WITH")):
        return "Oops, I messed up the query. Can you try asking that another way?"

    try:
        with db._engine.connect() as conn:
            result = conn.execute(text(sql_query))
            rows = result.fetchall()
            if not rows:
                if retry_count < 1:
                    return export_data(question, retry_count + 1)
                chat_history.append((question, "No data returned"))
                return "There’s no data to export, even after a second attempt."
            headers = list(result.keys())
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"exported_data_{timestamp}.csv"
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(headers)
                writer.writerows(rows)
            explanation = detailed_explanation if is_follow_up and question.lower() in ["explain more", "tell me more"] else short_explanation
            chat_history.append((question, sql_query))
            return f"Here’s your export:\nSaved to {filename}\n{explanation}\n{'Need more details?' if not is_follow_up else 'Hope that clears it up!'}"
    except Exception as e:
        logger.error("Error in export_data: %s", str(e))
        return "Sorry, the export hit a snag!"

def execute_and_answer(question):
    global answered, current_table_data, current_graph_fig
    simple, vis, table, expo = vis_expo_check(question)
    response = ""
    sql_queries = []
    current_table_data = None
    current_graph_fig = None

    if simple:
        resp = simple_answer(question)
        response += resp
        history_str = "\n".join([f"Q: {q}\nSQL: {sql}" for q, sql in chat_history]) if chat_history else "No previous conversation."
        prompt = (
            f"Previous conversation:\n{history_str}\n\nGenerate a PostgreSQL query for this question: '{question}'. "
            "Use ONLY the following tables and their specified columns in the company_sales_db database:\n"
            "- sales (columns: DimDeliveryDateID, DimCompanyID, Channel, DimShipToCustomerID, DimItemID, DocumentType, PaymentTerm, HLAMOUNT, Quantity, Amount)\n"
            "- customers (columns: DimCustomerID, UniqueCustomerCode, CustomerCode, TaxRegistrationNo, UniqueParentCustomerCode, ParentCustomerCode, Name, VATName, CompanyCode, CompanyName, SalesRepCode, RegionChannelCode, RegionChannel, ChannelCode, Channel, SubChannelCode, SubChannel, FinancialChannelCode, FinancialChannel, FinancialSubChannelCode, FinancialSubChannel, SRChannelCode, SRChannel, SegmentCode, Segment, RegionCode, Region, LoyaltyCode, Loyalty, ZoneCode, Zone, StateCode, State, AddressLine1, AddressLine2, AddressLine3, AddressLine4, City, CustomerLimit, CustomerLimitType, RowActiveIndicator, INSERT_DATETIME, UPDATE_DATETIME)\n"
            "- companies (columns: DimCompanyID, CompanyCode, ShortCode, ShortName, Name, TaxRegistrationNo, Address)\n"
            "- items (columns: DimItemID, ItemCode, ItemShortCode, Name, Description, Group, SegmentCode, SegmentShortName, Segment, SegmentOrder, CategoryCode, Category, CategoryOrder, SubCategoryCode, SubCategory, BrandCode, Brand, BrandOrder, SubBrandCode, SubBrand, CompanyCode, Company, PacketTypeCode, PacketType, ABV, Flavor, PackSize, Weight, HL, UOM, CaseType, CaseWeight, Cases, ItemURL, BrandURL)\n"
            "- calendar (columns: DateKey, Date, JulianDate, Year, YearWeekNo, Quarter, Month, MonthNo, DaysInMonth, MonthWeekNo, WeekDay, WeekDayNo, Day, YearDay, Holiday, DaySuffix, DayOfWeekUSA, RunningCountOfDayInYear, RunningCountOfDayInQuarter, WeekOfQuarter, MonthOfQuarter, QuarterName, QuarterNameExt, YearName, YYYYMM, MMYYYY, FirstDayOfWeekInYear, LastDayOfWeekInYear, FirstDayOfWeek, LastDayOfWeek, FirstDayOfMonth, LastDayOfMonth, FirstDayOfQuarter, LastDayOfQuarter, FirstDayOfYear, LastDayOfYear, IsWeekday)\n"
            "Return only the SQL query as plain text, no formatting or code blocks."
            "in the sql never use ALTER or UPDATE or anything that might change the date form the postgresql"
    )
        sql_query = text_to_sql(question, prompt)
        if sql_query:
            sql_queries.append(sql_query)
    if table and not answered:
        resp = table_answer(question)
        response += resp
        if last_table_sql:
            sql_queries.append(last_table_sql)
    if vis:
        resp = graph_answer(question)
        response += f"\n{resp}"
        if last_table_sql:
            sql_queries.append(last_table_sql)
    if expo:
        resp = export_data(question)
        response += f"\n{resp}"
        if last_table_sql:
            sql_queries.append(last_table_sql)

    answered = 0
    return response.strip(), sql_queries

def rewrite(question):
    history_str = "\n".join([f"Q: {q}\nSQL: {sql}" for q, sql in chat_history]) if chat_history else "No previous conversation."
    prompt = f"Previous conversation:\n{history_str}\n\nIf needed, rewrite the question '{question}' with correct grammar without changing any key words and ensuring it has the exact same meaning, because it will be converted into SQL later. Make sure the rewritten question has the exact same meaning as the original and respects the context from the chat history. If the question is already grammatically correct and clear in context, return it unchanged."
    return llm.invoke(prompt).content.strip()

def export_current_data():
    global current_table_data
    if current_table_data:
        headers, rows = current_table_data  # Export all rows, not limited to 20
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"exported_data_{timestamp}.csv"
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            writer.writerows(rows)
        chat_display.config(state=tk.NORMAL)
        chat_display.insert(tk.END, f"Here is your export:\nSaved to {filename}\n\n", "ai")
        chat_display.config(state=tk.DISABLED)
        chat_display.see(tk.END)
    else:
        chat_display.config(state=tk.NORMAL)
        chat_display.insert(tk.END, "No data to export!\n\n", "ai")
        chat_display.config(state=tk.DISABLED)
        chat_display.see(tk.END)


def refresh_left_panel():
    for widget in left_frame.winfo_children():
        widget.destroy()

    # Display the table if current_table_data exists
    if current_table_data:
        # Create a subframe for the table and button
        table_frame = tk.Frame(left_frame, bg="#ffffff")
        table_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Export button
        export_button = tk.Button(table_frame, text="CSV Экспорт", command=export_current_data, bg="#006400", fg="white", 
                                font=("Segoe UI", 10, "bold"), relief="flat", bd=0)
        export_button.pack(side=tk.TOP, anchor="nw", pady=(0, 5))

        headers, rows = current_table_data
        # Set the height to show up to 10 rows
        tree = ttk.Treeview(table_frame, columns=headers, show="headings", height=min(len(rows), 10))
        tree.pack(fill=tk.BOTH, expand=True)

        for col in headers:
            tree.heading(col, text=col, command=lambda c=col: sort_column(tree, c))
            tree.column(col, width=150, minwidth=100, stretch=False)

        # Insert only 10 rows into the Treeview
        for row in rows[:10]:  # Changed from rows[:20] to rows[:10]
            tree.insert("", "end", values=row)

        # Vertical scrollbar for the table
        v_scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.configure(yscrollcommand=v_scrollbar.set)

        # Horizontal scrollbar for the table
        h_scrollbar = ttk.Scrollbar(table_frame, orient="horizontal", command=tree.xview)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        tree.configure(xscrollcommand=h_scrollbar.set)

        # Add vertical scroll wheel support for the Treeview
        def on_table_mouse_wheel(event):
            # Determine scroll direction (positive for up, negative for down)
            if event.delta > 0:
                tree.yview_scroll(-1, "units")  # Scroll up
            elif event.delta < 0:
                tree.yview_scroll(1, "units")   # Scroll down

        # Bind mouse wheel events (Windows uses <MouseWheel>, Linux/Mac may use <Button-4> and <Button-5>)
        tree.bind("<MouseWheel>", on_table_mouse_wheel)  # For Windows
        tree.bind("<Button-4>", lambda event: tree.yview_scroll(-1, "units"))  # For Linux (scroll up)
        tree.bind("<Button-5>", lambda event: tree.yview_scroll(1, "units"))   # For Linux (scroll down)

        # Ensure the Treeview can receive focus for mouse wheel events
        tree.bind("<Enter>", lambda event: tree.focus_set())

        # Hover event handlers for export button
        def on_enter_export(e):
            animate_color_transition(export_button, "#006400", "#abc32f", duration=150)
        def on_leave_export(e):
            animate_color_transition(export_button, "#abc32f", "#006400", duration=150)
        export_button.bind("<Enter>", on_enter_export)
        export_button.bind("<Leave>", on_leave_export)

    # Display the graph if current_graph_fig exists (below the table)
    if current_graph_fig:
        # Create a subframe for the graph and scrollbars
        graph_frame = tk.Frame(left_frame, bg="#ffffff")
        graph_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create a canvas for the graph with scrollable region
        canvas = FigureCanvasTkAgg(current_graph_fig, master=graph_frame)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        
        # Pack the canvas inside the frame
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Add horizontal scrollbar
        h_scrollbar = ttk.Scrollbar(graph_frame, orient="horizontal", command=canvas_widget.xview)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas_widget.configure(xscrollcommand=h_scrollbar.set)

        # Add vertical scrollbar (optional)
        v_scrollbar = ttk.Scrollbar(graph_frame, orient="vertical", command=canvas_widget.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas_widget.configure(yscrollcommand=v_scrollbar.set)
        
        # Configure canvas scrolling region (based on figure size)
        canvas_widget.configure(scrollregion=(0, 0, current_graph_fig.get_size_inches()[0] * 100, current_graph_fig.get_size_inches()[1] * 100))

        # Add horizontal scroll wheel support for the graph
        def on_graph_mouse_wheel(event):
            # Determine scroll direction (positive for right, negative for left)
            if event.delta > 0:
                canvas_widget.xview_scroll(-1, "units")  # Scroll left
            elif event.delta < 0:
                canvas_widget.xview_scroll(1, "units")   # Scroll right

        # Bind mouse wheel events for the graph
        canvas_widget.bind("<MouseWheel>", on_graph_mouse_wheel)  # For Windows
        canvas_widget.bind("<Button-4>", lambda event: canvas_widget.xview_scroll(-1, "units"))  # For Linux (scroll up/left)
        canvas_widget.bind("<Button-5>", lambda event: canvas_widget.xview_scroll(1, "units"))   # For Linux (scroll down/right)

        # Ensure the canvas can receive focus for mouse wheel events
        canvas_widget.bind("<Enter>", lambda event: canvas_widget.focus_set())
# [Other functions like sort_column, export_current_data, etc., remain unchanged]

def send_message():
    question = entry.get().strip()
    if not question:
        return
    chat_display.config(state=tk.NORMAL)
    chat_display.insert(tk.END, f"{question}\n\n", "user")
    chat_display.tag_add("user", "end-3l", "end-2l")
    chat_display.config(state=tk.DISABLED)
    entry.delete(0, tk.END)

    def process_question():
        rewritten_question = rewrite(question)
        print(f"Rewritten Question: {rewritten_question}")
        response, sql_queries = execute_and_answer(rewritten_question)
        if response:
            response_in_mongolian = response
            print(f"Translated Response (to Mongolian): {response_in_mongolian}")
            chat_display.config(state=tk.NORMAL)
            chat_display.insert(tk.END, f"{response_in_mongolian}\n\n", "ai")
            chat_display.tag_add("ai", "end-3l", "end-2l")
            chat_display.config(state=tk.DISABLED)
            chat_display.see(tk.END)
            refresh_left_panel()

    threading.Thread(target=process_question, daemon=True).start()

root = tk.Tk()
root.title("Coretech SQL Chatbot")
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
chat_width = int(screen_width * 0.7)  # 30% smaller than full width
chat_height = int(screen_height*0.8)  # 30% smaller than full height
root.geometry(f"{chat_width}x{chat_height}+{int((screen_width-chat_width)/2)}+{int((screen_height-chat_height)/2)}")
root.configure(bg="#f5f7fa")

main_frame = tk.Frame(root, bg="#f5f7fa")
main_frame.pack(fill=tk.BOTH, expand=True)

# Left Panel (fixed width)
left_frame = tk.Frame(main_frame, bg="#ffffff", width=chat_width//2)
left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
left_frame.pack_propagate(False)  # Prevent resizing based on content

# Right Panel (50% width)
right_frame = tk.Frame(main_frame, bg="#ffffff", width=chat_width//2)
right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

# Chat Display in Right Panel (with internal padding)
chat_display = scrolledtext.ScrolledText(right_frame, wrap=tk.WORD, bg="#ffffff", fg="#2d3436", font=("Segoe UI", 10), 
                                        height=25, padx=15, pady=15)  # Added internal padding
chat_display.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 8))  # External padding unchanged
chat_display.config(state=tk.DISABLED)
chat_display.tag_configure("user", foreground="#000000", justify="right", font=("Segoe UI", 10))
chat_display.tag_configure("ai", foreground="#4A4A4A", justify="left", font=("Segoe UI", 10))

# Input Frame in Right Panel (8px padding between right_frame and entry_frame)
entry_frame = tk.Frame(right_frame, bg="#ffffff", bd=1, relief="solid", highlightbackground="#d3d3d3", highlightcolor="#d3d3d3", highlightthickness=1)
entry_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=8, pady=(0, 8))  # 8px padding on left, right, and bottom
entry = tk.Entry(entry_frame, bg="#ffffff", fg="#2d3436", font=("Segoe UI", 10), relief="flat", borderwidth=0)
entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0), pady=0)  # No vertical padding, minimal horizontal
send_button = tk.Button(entry_frame, text="send", command=send_message, bg="#006400", fg="#ffffff", 
                        font=("Segoe UI", 10, "bold"), relief="flat", bd=0)
send_button.pack(side=tk.RIGHT, padx=0, pady=0)  # No padding to touch edges

# Color transition functions (used for both buttons)
def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb):
    """Convert RGB tuple to hex color."""
    return '#{:02x}{:02x}{:02x}'.format(*rgb)

def interpolate_color(start_color, end_color, fraction):
    """Calculate intermediate color based on fraction (0 to 1)."""
    start_rgb = hex_to_rgb(start_color)
    end_rgb = hex_to_rgb(end_color)
    r = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * fraction)
    g = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * fraction)
    b = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * fraction)
    return rgb_to_hex((r, g, b))

def animate_color_transition(button, start_color, end_color, duration=1500, steps=30):
    """Animate button color change over duration (in ms)."""
    if hasattr(button, '_after_ids'):
        for after_id in button._after_ids:
            button.after_cancel(after_id)
    button._after_ids = []

    step_time = duration // steps  # Time per step in ms
    for step in range(steps + 1):
        fraction = step / steps
        color = interpolate_color(start_color, end_color, fraction)
        after_id = button.after(step_time * step, lambda c=color: button.config(bg=c))
        button._after_ids.append(after_id)

# Hover event handlers for Send button
def on_enter(e):
    animate_color_transition(send_button, "#006400", "#abc32f", duration=150)

def on_leave(e):
    animate_color_transition(send_button, "#abc32f", "#006400", duration=150)

# Bind hover events for Send button
send_button.bind("<Enter>", on_enter)
send_button.bind("<Leave>", on_leave)

root.bind('<Return>', lambda event: send_message())
root.bind('<Escape>', lambda event: root.quit())

root.mainloop()
