from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate

if __name__ == "__main__":
  load_dotenv()

  print("Hello Langchain")

  summary_template = """
    Given the information about the person I want:
    1. A short summary.
    2. Two interesting facts about them.
  """