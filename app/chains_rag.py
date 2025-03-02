from langchain.prompts import ChatPromptTemplate 
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from app.consts import l_llm
from typing import Literal, List
from pydantic import BaseModel, Field
from app.consts import create_prompt

LLM_s = l_llm['gpt-4o-mini']
LLM_l = l_llm['gpt-4o']

def get_relevance_chain():
    class Relevance(BaseModel):
        """ Whether a page is relevant or not """ 
        rel : Literal["RELEVANT",  "IRRELEVANT"] = Field(description = "Whether the page-content is relevant or not")
        
    relevance_prompt = """
    You are a equity research analyst. 
    You will be provided a passage from a company's filing report. 
    Your task is to check the passage is relevant in better understanding its business or not.
    If the passage is relevant if it talks about anything regarding the company's product portfolio, operational metrics or the macro factors, sectoral phenomena, competition, or company's plan for expansion, mergers, etc.
    It is irrelevant if the passage seems is filled with legal jargon or are highly technical details of numbers or numerical calculations, or seems completely unrelated to the business.
    Your response has to a single word - "RELEVANT" or "IRRELEVANT"
    """
    chain = create_prompt(relevance_prompt) | LLM_s.with_structured_output(Relevance) | RunnableLambda(lambda x: x.rel == 'RELEVANT')
    return chain





def get_excerpt_chain():
    class Excerpt(BaseModel):
        """ 
        An excerpt with it's tag
        """ 
        excerpt  : str = Field(description = "A small passage from a company's document")
        tag      : Literal["Operational Metrics",  "Financial Metrics",  "Key Personell",  "Product Strategy",  "Expansion" , "Competition",  "Sectoral Phenomenon",  "Regulatory", "Macro Factors",  "Organizational Details",  "MnA", "Alliances"] = Field(description = "Tag of the passage")

    class ListExcerpts(BaseModel):
        """List of excerpt and  tags""" 
        l_excerpt : List[Excerpt] = Field(description="List of excerpt and tags")
         

    excerpt_prompt = """
    You are a business analyst extracting relevant information from a passage of a company's annual report.
    Note that given the parsing process of the document, some unneccessary newline or whitespaces may have been added in the passage.
    You have been provided with a list of important tags below along with their description. 
    Go through the contents of the input provided to you, and  find excerpts which talks about these one of these tags: ["Operational Metrics",  "Financial Metrics",  "Key Personell",  "Product Strategy",  "Expansion" , "Competition",  "Sectoral Phenomenon",  "Regulatory", "Macro Factors",  "Organizational Details",  "MnA", "Alliances"] 
    Your goal is to to extract all relevant excerpts. Each excerpt can range in length from a phrase to a sentence or at max 2-3 sentences, and should be as short as possible while making semantic sense.
    For each excerpt identify the most relevant tag. Only use the tags from the below list.
    Your response should the different excerpts with their related tag, formatted as a list of tuples
    Sample response:
    [
        {{"excerpt": "The Company has purchased additional 4.7% equity interest in org X from org Y", "tag": "MnA"}},
        {{"excerpt": "In partnership with org X, the company has conducted India's first 5G trial in the 700 MHz band.", "tag": "Alliances"}},
        {{"excerpt": "Company's X division launched feature Y, a digital platform for customers to be able to do Z", "tag": "Product Strategy"}},
        {{"excerpt": "The Company had 208.4 million data customers at the end of March 31, 2022, of which 200.8 million were segment X customers.", "tag": "Operational Metrics"}},
        {{"excerpt": "Consequently, EBIT for the year was 248,531 Mn, increasing by 49.6% and resulting in a margin of 21.3% vis-a-vis 16.5% in the previous year.", "tag": "Financial Metrics"}}
    ]
    If the page talks about none of these tags, respond with an empty list [].

    List of Tags to choose from
    - Operational Metrics: Any excerpt that talks about the companies operational metrics for the business, eg. Average Revenue for User, Churn Rate, Net Promoter Score, Resoliton Time etc. Basically anything that indicates how well the company is serving its customers.
    - Financial Metrics: Any excerpt talking about the companies financia; metrics for the business, eg.any Revenue, Costs, Profits, Debts, Valuation, Share Price etc.
    - Key Personell: Any excerpt talking about any executive, management or key personell like a CxO,MD, board of directors etc. - any announcement regarding a change in the personell, modification in their role. or their achievement. 
    - Product Strategy: If the excerpt talks about it's companies portfolio of products or offerings. If any particular product is doing very well or poorly, or if they have changed the product portfolio - adding or removing some items, or changing their focus areas.
    - Expansion: If the excerpt is talking about expansion plans of the organization. This could be by capital expenditure, vertical growth by entering into some other part of the value chain, horizontal growth by targetting a different product line or customer segment, or entering a new business line. This could be an organic growth with internal team working on the new initiative, or inorganic growth via by Mergers or Acquisitions.
    - Competition: If the excerpt talks about it's competitor or the compeitive landscape for its industry.
    - Sectoral Phenomenon: If the excerpt talks about any wholesale changes in consumer preferences; price changes in key ingredients like commodities, energy, labour ; technological advancements 
    - Regulatory: If the excerpt talks about the regulatory landscape, or if the government or any other governing body's policy has impacted the company's business in any way.
    - Macro Factors: If the excerpt talks about any macro phenomena that impacts the business - like global politics, trade treaties, war etc. Also if the excerpt talks about country's economics and populations well being 
    - Organizational Details: If the excerpt talks about the organization structure, or about any parent or subsidiary organization.
    - MnA: If the excerpt talks about any mergers or acquisisions that the company has undertaken.
    - Alliances: If the company has entered into a strategic alliance or partnership with any other organization
    Ensure that the each of tag is one of these: ["Operational Metrics",  "Financial Metrics",  "Key Personell",  "Product Strategy",  "Expansion" , "Competition",  "Sectoral Phenomenon",  "Regulatory", "Macro Factors",  "Organizational Details",  "MnA", "Alliances"] = Field(description = "Tag of the passage")
    Do NOT use any other tag
    """.strip()
    get_l_excerpts =  RunnableLambda(lambda x: x.model_dump()['l_excerpt'])
    llm_l_st = LLM_l.with_structured_output(ListExcerpts)
    chain = create_prompt(excerpt_prompt) | llm_l_st | get_l_excerpts

    return chain


