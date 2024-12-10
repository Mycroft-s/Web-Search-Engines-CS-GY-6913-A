from langchain_openai import ChatOpenAI  # 新的模块路径
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.messages import HumanMessage
import sys
import json
import os
api_key = os.getenv("OPENAI_API_KEY")

def expand_query(query, api_key):
    """
    Expands a given search query by adding semantically related terms generated using OpenAI's GPT model.
    Combines the original query (repeated five times) with the expanded query for improved search relevance.

    Args:
        query (str): The original search query.
        api_key (str): The OpenAI API key for authentication.

    Returns:
        str: A combined query containing the repeated original query and expanded terms.
    """

    # Initialize the OpenAI GPT model with the specified API key
    openai_model = ChatOpenAI(model="gpt-4o", openai_api_key=api_key)
    #change to gpt-4o-mini or gpt-o

    # Create a prompt message instructing the model to expand the query
    message = HumanMessage(
    content = (
    f"You are a search optimization assistant. Your task is to generate a corrected and expanded search query in a single step. "
    f"First, correct any spelling or typographical errors in the given query: {query}. "
    f"Next, expand the corrected query by adding exactly 3 semantically related terms that are highly relevant to the original query. "
    f"Finally, create the output query as follows: "
    f"include the corrected query repeated exactly 5 times, followed by the 3 expanded terms, all terms separated by a single space and without any punctuation or special characters. "
    f"The output should be a single query string ready for use in search engines."
)
)
    try:
        # Invoke the GPT model to handle the entire query correction and expansion process
        response = openai_model.invoke([message])

        # Extract the content of the response and clean up formatting
        combined_query = response.content.strip()  # Clean up leading/trailing whitespace
        print(f"Show Combined Query: {combined_query}")

        # Return the final combined query directly
        return combined_query

    except Exception as e:
        # Handle errors during the expansion process
        print(f"Error occurred: {e}")
        return None



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python query_expansion.py <query>")
        sys.exit(1)

    query = sys.argv[1]

    expanded_query = expand_query(query, api_key)

    if expanded_query:
        print(json.dumps({"expanded_query": expanded_query}))
    else:
        print(json.dumps({"error": "Failed to expand query"}))

