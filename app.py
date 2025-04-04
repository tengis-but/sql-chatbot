from flask import Flask, render_template, request, jsonify
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from sqlalchemy import text
import matplotlib.pyplot as plt
import io
import base64
import logging
import google.generativeai as genai
import csv
from datetime import datetime

app = Flask(__name__)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
DB_PASSWORD = "Admin123$"
db_uri = f"postgresql+psycopg2://tengis.t:{DB_PASSWORD}@18.204.2.2:5432/data-db"
db = SQLDatabase.from_uri(db_uri)
try:
    with db._engine.connect() as conn:
        result = conn.execute(text("SELECT 1")).fetchone()
    logger.info("Database connection established successfully. Result: %s", result[0])
except Exception as e:
    logger.error("Failed to connect to database: %s", str(e))
    exit(1)

# LLM setup
GOOGLE_API_KEY = "AIzaSyBPwhPFnUJ1YzshqGRL4Lc_REuMdh-8XzM"
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", google_api_key=GOOGLE_API_KEY, temperature=1)
genai.configure(api_key=GOOGLE_API_KEY)

# Global variables
chat_history = []
current_table_data = None
current_graph_fig = None

# Placeholder functions
def sql_query_checker(sql_query, question, history_str):
    """Check and correct SQL query for valid tables, columns, and data types."""
    segment_vals, category_vals, subcategory_vals, brand_vals, item_company_vals, flavor_vals, company_name_vals, shortname_vals, subchannel_vals, financialchannelcode_vals, name_vals, cat_vals = get_metadata_values()
    
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
            cat_values = [row[0] for row in conn.execute(text("SELECT DISTINCT category FROM items WHERE category IS NOT NULL;")).fetchall()]
            
        return segment_values, category_values, subcategory_values, brand_values, item_company_values, flavor_values, company_name_values, shortname_values, subchannel_values, financialchannelcode_values, name_values, cat_values
    except Exception as e:
        logger.error("Error retrieving metadata values: %s", str(e))
        return [], [], [], [], [], [], [], [], [], [], []

def vis_expo_check(question):
    prompt = f"Check if the sentence '{question}' is asking for: 1. A simple answer (no graph or table), 2. To be graphed, 3. To retrieve data ('yes' only if it’s related to showing a table), 4. To export data. Return four values: 'yes' or 'no' for simple answer, graphing, retrieving data (table), export, e.g., 'yes no no no' or 'no no yes yes'. No formatting or code blocks."
    response = llm.invoke(prompt).content.strip()
    simple, vis, table, expo = response.split()
    if vis == "yes":
        table = "yes"
    return simple == "yes", vis == "yes", table == "yes", expo == "yes"

def text_to_sql(question, prompt):
    try:
        response = llm.invoke(prompt).content.strip()
        sql = response.replace("```sql", "").replace("```", "").strip()
        if not sql or not (sql.upper().startswith("SELECT") or sql.upper().startswith("WITH")):
            return None
        logger.info("Generated SQL: %s", sql)  # Log the SQL here
        return sql
    except Exception as e:
        logger.error("Error in text_to_sql: %s", str(e))
        return None


    
def table_answer(question, retry_count=0):
    global current_table_data
    history_str = "\n".join([f"Q: {q}\nSQL: {sql}" for q, sql in chat_history]) if chat_history else "No previous conversation."
    is_follow_up = any("explain more" in q.lower() or "tell me more" in q.lower() for q, _ in chat_history[-1:]) and len(chat_history) > 1
    segment_vals, category_vals, subcategory_vals, brand_vals, item_company_vals, flavor_vals, company_name_vals, shortname_vals, subchannel_vals, financialchannelcode_vals, name_vals, cat_vals = get_metadata_values()

    prompt = (
            f"Previous conversation:\n{history_str}\n\nGenerate a PostgreSQL query for this question: '{question}'. "
        "Use ONLY the following tables and their specified columns in the company_sales_db database:\n"
        "- sales (columns: DimDeliveryDateID, DimCompanyID, Channel, DimShipToCustomerID, DimItemID, DocumentType, PaymentTerm, HLAMOUNT, Quantity, Amount)\n"
        "- customers (columns: DimCustomerID, UniqueCustomerCode, CustomerCode, TaxRegistrationNo, UniqueParentCustomerCode, ParentCustomerCode, Name, VATName, CompanyCode, CompanyName, SalesRepCode, RegionChannelCode, RegionChannel, ChannelCode, Channel, SubChannelCode, SubChannel, FinancialChannelCode, FinancialChannel, FinancialSubChannelCode, FinancialSubChannel, SRChannelCode, SRChannel, SegmentCode, Segment, RegionCode, Region, LoyaltyCode, Loyalty, ZoneCode, Zone, StateCode, State, AddressLine1, AddressLine2, AddressLine3, AddressLine4, City, CustomerLimit, CustomerLimitType, RowActiveIndicator, INSERT_DATETIME, UPDATE_DATETIME)\n"
        "- companies (columns: DimCompanyID, CompanyCode, ShortCode, ShortName, Name, TaxRegistrationNo, Address)\n"
        "- items (columns: DimItemID, ItemCode, ItemShortCode, Name, Description, Group, SegmentCode, SegmentShortName, Segment, SegmentOrder, CategoryCode, Category, CategoryOrder, SubCategoryCode, SubCategory, BrandCode, Brand, BrandOrder, SubBrandCode, SubBrand, CompanyCode, Company, PacketTypeCode, PacketType, ABV, Flavor, PackSize, Weight, HL, UOM, CaseType, CaseWeight, Cases, ItemURL, BrandURL)\n"
        "- calendar (columns: DateKey, Date, JulianDate, Year, YearWeekNo, Quarter, Month, MonthNo, DaysInMonth, MonthWeekNo, WeekDay, WeekDayNo, Day, YearDay, Holiday, DaySuffix, DayOfWeekUSA, RunningCountOfDayInYear, RunningCountOfDayInQuarter, WeekOfQuarter, MonthOfQuarter, QuarterName, QuarterNameExt, YearName, YYYYMM, MMYYYY, FirstDayOfWeekInYear, LastDayOfWeekInYear, FirstDayOfWeek, LastDayOfWeek, FirstDayOfMonth, LastDayOfMonth, FirstDayOfQuarter, LastDayOfQuarter, FirstDayOfYear, LastDayOfYear, IsWeekday)\n"
        f"Metadata for filtering:\n- Item Segment values: {', '.join(segment_vals)}\n- Item Category values: {', '.join(category_vals)}\n- Item SubCategory values: {', '.join(subcategory_vals)}\n- Item Brand values: {', '.join(brand_vals)}\n- Items Category values: {cat_vals}\n - Item Company values: {', '.join(item_company_vals)}\n- Item Flavor values: {', '.join(flavor_vals)}\n- Company Name values: {', '.join(company_name_vals)}\n- Company ShortName values: {', '.join(shortname_vals)}\n- Customer SubChannel values: {', '.join(subchannel_vals)}\n- Customer FinancialChannelCode values: {', '.join(financialchannelcode_vals)}\n- items name{', '.join(name_vals)}"
        "Use these metadata values to filter data accurately (e.g., Category = 'Vodka' for vodka items). Do not reference any tables or columns outside of those listed above. "
        "For product names, use items.Name; for quantities, use sales.Quantity; for dates, use sales.DimDeliveryDateID joined with calendar.Date. "
        "Carefully consider the chat history to maintain context:\n- If the previous question mentions a specific company, customer, or item attribute, apply it unless overridden.\n- If an aggregation (e.g., SUM, AVG) was used previously, assume it unless specified otherwise.\n- Focus on the intent of the current question while respecting conversation flow.\n"
        "If the question asks for a list (e.g., 'list all vodka items'), use DISTINCT.\nJoin tables as needed (e.g., sales with calendar on DimDeliveryDateID = DateKey).\nUse WITH clauses (CTEs) when the query involves complex subqueries or reusable intermediate results for clarity and efficiency; otherwise, use SELECT.\nEnsure the query starts with SELECT or WITH; if it doesn’t, return 'INVALID_SQL' instead.\n"
        "After generating the SQL query, provide two explanations:\n- A short explanation (20-30 words) of what the query does, including the main table and column(s) used and the calculation method.\n- A detailed explanation (60 words) for if the user asks for more details later, covering the table(s), column(s), joins, and calculation in depth.\n"
        "Return the SQL query followed by 'SHORT_EXPLANATION:' and the short explanation, then 'DETAILED_EXPLANATION:' and the detailed explanation, separated by newlines, no formatting or code blocks."
        "in the sql never use ALTER or UPDATE or anything that might change the date form the postgresql"
   )
    
    sql_response = text_to_sql(question, prompt)
    if not sql_response:
        return "Oops, couldn’t generate a valid SQL query!"

    try:
        parts = sql_response.split("SHORT_EXPLANATION:")
        sql_query = parts[0].strip()
        short_and_detailed = parts[1].split("DETAILED_EXPLANATION:")
        short_explanation = short_and_detailed[0].strip()
        detailed_explanation = short_and_detailed[1].strip() if len(short_and_detailed) > 1 else "No detailed explanation available."
    except (IndexError, ValueError):
        sql_query = sql_response.strip()
        short_explanation = "I pulled this for the table, but couldn’t clarify the details."
        detailed_explanation = "Sorry, I couldn’t generate a detailed breakdown."

    sql_query = sql_query_checker(sql_query, question, history_str)
    if sql_query == "INVALID_SQL":
        return "Oops, couldn’t get the query right!"

    logger.info("Executing SQL for table: %s", sql_query)  # Log SQL before execution
    try:
        with db._engine.connect() as conn:
            result = conn.execute(text(sql_query))
            rows = result.fetchall()
            if not rows:
                if retry_count < 1:
                    return table_answer(question, retry_count + 1)
                chat_history.append((question, "No data returned"))
                return "No data found after two attempts."
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
            total_rows = len(cleaned_rows)
            display_limit = 10
            response = f"Table generated ({display_limit} of {total_rows} rows shown):\n{explanation}"
            chat_history.append((question, sql_query))
            return response
    except Exception as e:
        logger.error("Error executing SQL: %s", str(e))
        return "Sorry, something went wrong with the table!"

def graph_answer(question, retry_count=0):
    global current_graph_fig, current_table_data
    if not current_table_data:
        return "No table data available to graph. Please generate a table first!"
    
    headers, rows = current_table_data
    available_columns = ", ".join(headers)
    history_str = "\n".join([f"Q: {q}\nSQL: {sql}" for q, sql in chat_history]) if chat_history else "No previous conversation."
    
    prompt = (
        f"Previous conversation:\n{history_str}\n\nAnalyze the following sentence: '{question}'. "
        f"The available table data has these columns: {available_columns}. "
        "Check if the question specifies 'graph x,y' with defined x and y variables matching these columns. "
        "If specified, use those as x_title and y_title. If not specified, recommend the best x-axis title and y-axis title from the available columns based on the question’s intent. "
        "For time-based data, prefer a date column (e.g., 'Date') for x-axis; for values, prefer a numeric column (e.g., 'Quantity', 'Amount'). "
        "Identify the chart type: 'line_chart', 'pie_chart', or 'graph_chart' (default to 'graph_chart' if unspecified). "
        "Output: 'x_title y_title type_of_chart' as plain text."
    )
    try:
        response = llm.invoke(prompt).content.strip().lower()
        x_title, y_title, type_of_chart = response.split()
        if x_title not in headers or y_title not in headers:
            return f"Oops, '{x_title}' or '{y_title}' isn’t in the table data ({available_columns})."
    except Exception as e:
        logger.error("Error determining graph titles: %s", str(e))
        return "Hmm, I couldn’t sort out the graph details."

    max_graph_rows = 100
    limited_rows = rows[:max_graph_rows] if len(rows) > max_graph_rows else rows

    try:
        if not limited_rows:
            if retry_count < 1:
                return graph_answer(question, retry_count + 1)
            return "No data to graph after retry."
        
        x_idx = headers.index(x_title)
        y_idx = headers.index(y_title)
        x_data = [row[x_idx] for row in limited_rows]
        y_data = [float(row[y_idx]) if row[y_idx] != "Null" else 0 for row in limited_rows]

        fig = plt.Figure(figsize=(10, 4))
        ax = fig.add_subplot(111)
        if type_of_chart == "line_chart":
            ax.plot(x_data, y_data, marker='o', linestyle='-', color='b')
        elif type_of_chart == "pie_chart":
            ax.pie(y_data, labels=x_data, autopct='%1.1f%%', startangle=90)
        else:
            ax.scatter(x_data, y_data, color='g')
        ax.set_xlabel(x_title)
        ax.set_ylabel(y_title)
        ax.set_title(f"{y_title} vs {x_title}")
        ax.grid(True)
        num_data_points = len(x_data)
        if num_data_points > 10:
            step = max(1, num_data_points // 10)
            ax.set_xticks(range(0, num_data_points, step))
            ax.set_xticklabels([x_data[i] for i in range(0, num_data_points, step)], rotation=45, ha='right')
        else:
            ax.set_xticklabels(x_data, rotation=45, ha='right')
        fig.tight_layout()
        current_graph_fig = fig
        chat_history.append((question, f"Graph: {x_title} vs {y_title}"))
        return f"Graph generated:\nThis plots {y_title} against {x_title} (up to {max_graph_rows} rows)."
    except Exception as e:
        logger.error("Error in graph_answer: %s", str(e))
        return "Sorry, the graph didn’t work out!"

def export_data(question):
    global current_table_data
    if not current_table_data:
        return "No data to export!"
    headers, rows = current_table_data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"exported_data_{timestamp}.csv"
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(rows)
    chat_history.append((question, "Export completed"))
    return f"Data exported to {filename}."

def execute_and_answer(question):
    global current_table_data, current_graph_fig
    simple, vis, table, expo = vis_expo_check(question)
    response = ""
    data = {}
    sql_used = None

    logger.info("Processing question: %s (simple=%s, vis=%s, table=%s, expo=%s)", question, simple, vis, table, expo)

    if table or simple:
        response = table_answer(question)
        if chat_history and chat_history[-1][1] not in ["No data returned", "Export completed"]:
            sql_used = chat_history[-1][1]
        if current_table_data:
            headers, rows = current_table_data
            data["table"] = {"headers": headers, "rows": rows[:10]}
            logger.info("Table data prepared: %s rows", len(rows))

    if vis:
        graph_response = graph_answer(question)
        response += f"\n{graph_response}"
        if current_graph_fig:
            img = io.BytesIO()
            current_graph_fig.savefig(img, format='png', bbox_inches='tight')
            img.seek(0)
            graph_url = base64.b64encode(img.getvalue()).decode('utf-8')
            data["graph"] = graph_url  # Ensure graph data is included
            logger.info("Graph data prepared: %s", graph_url[:50] + "...")  # Log part of base64 for verification

    if expo:
        export_response = export_data(question)
        response += f"\n{export_response}"

    if sql_used:
        logger.info("SQL used for question '%s': %s", question, sql_used)
    data["sql"] = sql_used
    return response, data


@app.route('/')
def index():
    return render_template('index.html', chat_history=chat_history)

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    chat_history.append((question, ""))
    response, data = execute_and_answer(question)
    chat_history[-1] = (question, response)
    return jsonify({"response": response, **data})

@app.route('/export', methods=['GET'])
def export():
    global current_table_data
    if not current_table_data:
        return "No data to export!", 400
    headers, rows = current_table_data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"exported_data_{timestamp}.csv"
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(rows)
    return jsonify({"message": f"Data exported to {filename}"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5004)
