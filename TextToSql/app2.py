from flask import Flask, request, render_template, redirect, url_for, jsonify
import os
import pandas as pd
import sqlite3
import json
from dotenv import load_dotenv
import plotly.express as px
from langchain_community.utilities import SQLDatabase
from langchain_openai import AzureChatOpenAI
import ast
from typing import *
load_dotenv()

# ==========================
# Environment Variables
# ==========================
# ==========================
# Environment Variables
# ==========================


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
# ==========================
# LLM and Toolkit Setup
# ==========================
llm = AzureChatOpenAI(
    azure_deployment="Alfred-gpt-4-o-mini",
    api_version=OPENAI_API_VERSION,
    temperature=0.1,
    max_tokens=None,
)

class Formatter(TypedDict):
    graph_type: Annotated[str | None, ..., "The appropriate graph type from Bar, Line, or Scatterplot, or None if not applicable"]
    axis: Annotated[List[str] | None, ..., "Exactly two column names from `column_names`, used as X and Y axes"]
    column_names: Annotated[List[str] | None, ..., "All column names selected in the SQL query"]
    sql_query: Annotated[str | None, ..., "The SQL query based on the user's request"]
    reason: Annotated[str | None, ..., "Use **Markdown** to explain errors, invalid queries, or non-SQL questions"]

struct_llm = llm.with_structured_output(Formatter)

def AskLLM(question, table_name, top5_rows):
    prompt = f"""
You are an SQL Agent.

Your task is to convert a natural language user query into a valid SQL query and return structured metadata for data retrieval and visualization — ensuring consistency between the query output and graph configuration.

📥 Input Context:
Table Name: {table_name}

Top 5 Rows of Table: {top5_rows}
SQL SOFTWARE or Dialect: SQLite
🧠 Instructions:
Analyze the user’s query and generate a valid SQL query using only the columns present in the input table.

If your query includes aggregation or aliasing, the column_names list must use the alias, not the original column name.

✅ AVG(median_house_value) AS average_median_house_value → Use average_median_house_value in column_names

❌ Do not include median_house_value unless it is directly selected

Recommend a chart type from: Bar, Line, or Scatterplot.

Define the axis as [X-axis, Y-axis]:

Must exactly match column names in column_names

Must be valid data types (X = categorical/time, Y = numeric)

If the query:

References a non-existent column

Has mismatched axis and column_names

Has any other issue
➤ Return None for all other fields, and give a clear Markdown-formatted message in the reason field.

For general or meta questions (e.g., “What can I ask from this table?”):

Set all fields to None except for reason

Use Markdown to explain

If the SQL query is valid and matches all criteria, reason must be None.

Output must strictly follow the format below — no extra text.
"""
    response = struct_llm.invoke(prompt + question)
    return response

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

database_path = 'database.db'

def RunSQLQuery(sql_query, column_names):
    db = SQLDatabase.from_uri(f"sqlite:///{database_path}")
    result = db.run(sql_query)
    data_list = ast.literal_eval(result)
    result_df = pd.DataFrame(data_list, columns=column_names)
    return result_df

def Plot(graph_type, result_df, axis):
    if graph_type.lower() == 'bar':
        fig = px.bar(data_frame=result_df, x=axis[0], y=axis[1])
    elif graph_type.lower() == 'scatterplot':
        fig = px.scatter(data_frame=result_df, x=axis[0], y=axis[1])
    elif graph_type.lower() == 'line':
        fig = px.line(data_frame=result_df, x=axis[0], y=axis[1])
    return fig

@app.route('/')
def home():
    with open('data.json', 'r') as f:
        data = json.load(f)
    tables = list(data.keys())
    return render_template('chatGPTInterface.html', tables=tables)

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        table_name = os.path.splitext(file.filename)[0].replace(" ", "_").replace('.csv', '')
        df = pd.read_csv(filepath)
        conn = sqlite3.connect(database_path)
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        conn.commit()
        conn.close()
        with open('data.json', 'r') as f:
            data = json.load(f)
        data[table_name] = df.head().to_dict(orient='records')
        with open('data.json', 'w') as f:
            json.dump(data, f)
        os.remove(filepath)
        return jsonify({'success': True, 'table_name': table_name})
    return jsonify({'success': False, 'error': 'No file uploaded'})

@app.route('/query', methods=['POST'])
def query():
    user_query = request.form['query']
    table_name = request.form['table_name']
    print(table_name)
    with open('data.json', 'r') as f:
        data = json.load(f)
    top5_rows = data.get(table_name, [])
    response = AskLLM(user_query, table_name, top5_rows)
    if response['reason']:
        # Send the reason as a Markdown-formatted chat message
        return jsonify({'success': False, 'error': response['reason']})
    else:
        sql_query = response['sql_query']
        graph_type = response['graph_type']
        axis = response['axis']
        column_names = response['column_names']
        result_df = RunSQLQuery(sql_query, column_names)
        fig = Plot(graph_type, result_df, axis)
        return jsonify({'success': True, 'plot': fig.to_html(full_html=False, include_plotlyjs='cdn')})

if __name__ == '__main__':
    app.run(debug=True)