import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

# Optional dotenv support (local only)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


class Chain:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Set it in Streamlit Secrets or environment variables."
            )

        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0,
            groq_api_key=api_key
        )

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### Scraped text from website:
            {page_data}

            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Extract job postings and return JSON with keys:
            'role', 'experience', 'skills', 'description'.
            Skills must be comma-separated.
            Only return valid JSON.
            """
        )

        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke({"page_data": cleaned_text})

        try:
            parser = JsonOutputParser()
            res = parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too large to parse")

        return res if isinstance(res, list) else [res]

    def write_mail(self, job, links):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION
            {job_description}

            ### INSTRUCTION:
            You are Manideep, Business Development Executive at Anarch.
            Write a cold email explaining how Anarch can fulfill this role.
            Highlight relevant portfolio links: {link_list}
            No preamble. Return only the email.
            """
        )

        chain_email = prompt_email | self.llm
        res = chain_email.invoke({
            "job_description": str(job),
            "link_list": links
        })

        return res.content
