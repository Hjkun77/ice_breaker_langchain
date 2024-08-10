from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate
from langchain_openai.chat_models import AzureChatOpenAI
from azure.identity import DefaultAzureCredential
from langchain_core.output_parsers import StrOutputParser
information = """
    Elon Reeve Musk FRS (/ˈiːlɒn/; born June 28, 1971) is a businessman and investor known for his key roles in space company SpaceX and automotive company Tesla, Inc. Other involvements include ownership of X Corp., the company that operates the social media platform X (formerly known as Twitter), and his role in the founding of The Boring Company, xAI, Neuralink and OpenAI. He is one of the wealthiest people in the world; as of August 2024, Forbes estimates his net worth to be US$241 billion.[3]

Musk was born in Pretoria to model Maye and businessman and engineer Errol Musk, and briefly attended the University of Pretoria before immigrating to Canada at age 18, acquiring citizenship through his Canadian-born mother. Two years later, he matriculated at Queen's University at Kingston in Canada. Musk later transferred to the University of Pennsylvania and received bachelor's degrees in economics and physics. He moved to California in 1995 to attend Stanford University, but dropped out after two days and, with his brother Kimbal, co-founded online city guide software company Zip2. The startup was acquired by Compaq for $307 million in 1999. That same year, Musk co-founded X.com, a direct bank. X.com merged with Confinity in 2000 to form PayPal. In October 2002, eBay acquired PayPal for $1.5 billion. Using $100 million of the money he made from the sale of PayPal, Musk founded SpaceX, a spaceflight services company, in 2002.
"""

import os

if __name__ == '__main__':
    load_dotenv()
    print("Hello Langchain!")
    # print(os.environ['COOL_API_KEY'])

    # Main Setup
    HEADERS = {
        "userid": "sison.hm@pg.com",
        "project-name": "GBS_RAG_TRAIN",
    }

    OPEN_API_VERSION="2023-12-01-preview"

    # Request credential - you can use "az login" for this to work
    credential = DefaultAzureCredential(exclude_interactive_browser_credential=True)
    token = credential.get_token(os.environ['COGNITIVE_SERVICES']).token

    model = AzureChatOpenAI(
        azure_endpoint=os.environ['GENAI_PROXY'],
        azure_deployment="GPT-4O-2024-05-13",
        api_version=OPEN_API_VERSION,
        api_key=token,
        temperature=0.0,
        default_headers=HEADERS,
        verbose=True
    )

    summary_template = """"
            given the information {information} about a person from I want you to create:
            1. a short summary
            2. two interesting facts about them   
        """

    summary_prompt_template = PromptTemplate(input_variables=["information"], template=summary_template)

    chain = summary_prompt_template | model | StrOutputParser()

    res = chain.invoke(input={"information": information})

    print(res)

