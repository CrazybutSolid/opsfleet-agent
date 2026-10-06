# AI Technical Assignment - Retail Company

## The Challenge: Design and Build a Data Analysis Chat Assistant

You are building data analysis agent for a Retail Company's non-technical executive teams (OpsFleet is the client for this project and the Team Lead + Harry are the Project managers). These executives need to ask complex questions about sales, and performance (e.g., "Why are the users for sate X underspending? And how does it compare to users in state Y?", " Why did our churn rate spike last month? ", "create a report for Q1 including insights and actions items for Q2")

The agent will have access to:
* The Database: A read-only connection to a SQL database containing raw transaction logs.
* The "Golden Knowledge" Bucket: A data lake containing historical "Trios" (Question → SQL Query → Analyst Report) created by human experts (Theoretical).

The Task: Design a production-ready full High-Level Design (HLD), accompanied by a detailed technical explanation, and build a data analysis chat-agent prototype that allows executives to ask question about this data naturally and discuss about it. The system should be easily extendable for new capabilities (generating graphs, sending reports via mail, searching the web for trends, etc).

Note: We expect to see what services you aim to use, what the communication is between the components, how and where data is stored and handled. The "detailed technical explanation" needs to be detailed enough for us to understand how this system will function in production.

Feel free to ask about any assumptions you need to make, or follow ups on requirements, we (OpsFleet team) are the client for this project.

## Requirements

1. Hybrid Intelligence: The agent cannot rely on BQ data alone. It must use the "Golden Bucket" to understand how analysts previously interpreted questions and apply similar logic to new queries. Explain how you will update the golden bucket over time and how you will provide relevant data at user query time.
2. Safety & PII Masking: The agent is only allowed to answer analysis questions and needs to be safeguarded against malicious users. The raw transaction logs contain sensitive data. The agent is strictly forbidden from displaying these PIIs in the final output. Each user should only be able to analyze data on products related to him.
3. High-Stakes Oversight (Destructive Ops): While the DB is read-only, the agent manages a "Saved Reports" library. The agent must support inputs like "Delete all reports mentioning Client X" / "Delete all the reports we made in this conversation" This is a destructive action and requires a strict confirmation flow before execution, without breaking UX - the users are allowed to delete their own reports.
4. Continuous Improvement (The Learning Loop):
    1. User Level: The agent should remember that "User A" prefers tables while "User B" prefers bullet points, as well as learn user preferences over time for example: how deep they like the analysis, do they prefer charts or text, etc.
    2. System Level: The agent should be able to learn from past interactions and improve itself.
5. Resilience & Graceful Error Handling: The system must detect syntax errors or empty returns and attempt to self-correct before giving up, without crashing the user interface and without inflating costs. The system should be resilient to API/3rd party services failures/downtime
6. Quality Assurance: How do you evaluate the agent before deployment? How do you verify that the generated reports answer users intent correctly? How do you evaluate UX?
7. Observability: We need to know when the agent is failing and why. Define the metrics you would track at the agent level and how you would support deep dive analysis/debugging (understanding what the message correspondence is and what went wrong).
8. Agility (Persona Management): The CEO wants to change the "tone" of the reports weekly. The system must allow non-developers to update the agent's instructions without redeployment.

## Deliverables

1. Architecture Diagram (Mermaid etc): Highlighting the building blocks, services, compute and flow of the system based on the requirements above. In case you are planning to use a infrastructure/framework/service/data store for one of the building blocks, specify which one and include an explanation.
2. Detailed technical explanation covering:
    1. Reasoning for the chosen Cloud services / LLM models/frameworks used.
    2. Data flow between components (if needs to elaborate on HLD).
    3. Error handling and fallback strategies.
    4. Setup Instructions and example run
    5. Make sure to include a detailed explanation of how you handle/solve each of the requirements
3. Working Code/Prototype: Build a chat agent that allows users to ask data related questions naturally, discuss about it and create a report with action items when asked to. The prototype needs to support all of these following requirements (defined above, rest of the requirements are for design only):
    1. Safety & PII Masking
    2. High-Stakes Oversight
    3. Resilience & Graceful Error Handling
    4. Observability
4. CLI-based interface for chat interactions.
5. Your solution must be runnable on another machine (Docker is not a must, just proper setup instructions).
6. Use a framework of your choice, explain why you choose it and your level of experience with it.

Our assessment will focus mainly on system design, the technical explanation, and an elegant Prototype that fits the Prototype requirements

## Dataset Specification

* Dataset: bigquery-public-data.thelook_ecommerce
* Required Tables:
    * orders - Customer order information
    * order_items - Individual items within orders
    * products - Product catalog and details
    * users - Customer demographics and information

## Expected Agent Capabilities

Your prototype chat agent should be able to perform data analysis and generate insightful reports such as:
* Customer behavior (e.g., top customers, total spend)
* Product performance (e.g compare performance of product X and Y, and why do they perform differently)
* Time-based metrics (e.g., monthly revenue, up-to-date revenue by product)
* Answer questions about the general structure of the database (what data is available, what can we do with it)
* Multi step queries (as described above)

1. Use BigQuery integration to query and analyze the specified tables. Your agent should be able to construct and execute SQL queries dynamically based on the analysis requirements.
2. You should preferably use one of the newer Google Gemini models. You can get a free API key from Google AI Studio. Please be mindful of the rate limits.

## Setup Instructions

### Environment Setup
1. Install Python dependencies:
   pip install -r requirements.txt

### BigQuery Setup
1. Set up BigQuery access by following the BigQuery Client Libraries documentation if you don't already have BigQuery access configured.
2. Free Tier: Google Cloud provides 1TB of free BigQuery compute per month, which is more than sufficient for this challenge.
3. Authentication: Ensure your environment is authenticated with Google Cloud to access the public datasets.

## Submission

Share with us a your public GitHub repository with:
* Documentation
* Source code
* Architecture diagram

## Resources (provided starter code)

```python
import logging
from typing import Optional, List, Dict, Any
import pandas as pd
from google.cloud import bigquery


class BigQueryRunner:
    """A lean BigQuery client for executing SQL queries and returning DataFrame results."""

    def __init__(self, project_id: Optional[str] = None, dataset_id: Optional[str] = "bigquery-public-data.thelook_ecommerce") -> None:
        """Initialize BigQuery client.

        Args:
            project_id: Google Cloud project ID. If None, uses default credentials.
            dataset_id: BigQuery dataset ID. If None, uses default dataset.
        """
        logging.info("Initializing BigQuery client")
        try:
            self.client = bigquery.Client(project=project_id)
            self.dataset_id = dataset_id
            logging.info(f"BigQuery client initialized for dataset: {self.dataset_id}")
        except Exception as e:
            logging.error(f"Failed to initialize BigQuery client: {str(e)}")
            raise

    def execute_query(self, sql_query: str) -> pd.DataFrame:
        """Execute a SQL query and return results as a DataFrame.

        Args:
            sql_query: The SQL query to execute.

        Returns:
            DataFrame containing the query results.

        Raises:
            Exception: If query execution fails.
        """
        try:
            logging.info(f"Executing BigQuery query")
            query_job = self.client.query(sql_query)
            df = query_job.result().to_dataframe()
            logging.info(f"Query completed successfully, returned {len(df)} rows")
            return df
        except Exception as e:
            logging.error(f"BigQuery execution failed: {str(e)}")
            raise

    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """Get schema information for a specific table.

        Args:
            table_name: Name of the table (orders, order_items, products, users).

        Returns:
            List of dictionaries containing column information.
        """
        try:
            table_ref = f"{self.dataset_id}.{table_name}"
            table = self.client.get_table(table_ref)
            schema_info = []
            for field in table.schema:
                schema_info.append({
                    "name": field.name,
                    "type": field.field_type,
                    "mode": field.mode,
                    "description": field.description or ""
                })
            logging.info(f"Retrieved schema for table {table_name}")
            return schema_info
        except Exception as e:
            logging.error(f"Failed to get schema for table {table_name}: {str(e)}")
            raise
```
