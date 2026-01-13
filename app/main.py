import streamlit as st

# Must be first Streamlit command
st.set_page_config(
    page_title="Cold Email Generator",
    page_icon="🚀",
    layout="wide"
)

import pandas as pd
from langchain_community.document_loaders import WebBaseLoader
from chains import Chain
from portfolio import Portfolio


st.markdown("""
<style>
    .main { background-color: #f7f7f7; }
    h1 { color: #ff4b4b; }
    .stButton > button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


def create_stream_app(llm, portfolio):
    st.title("🚀 Cold Email Generator")
    st.markdown("Enter a job listing URL and generate tailored cold emails.")

    url = st.text_input(
        "Job Listing URL",
        value="https://jobdetails.nestle.com/job/Esplugues-Llobregat-Technology-Expert-R&D-Information-Technology-B-08950/1204832601/"
    )

    if st.button("Generate Emails"):
        with st.spinner("Processing..."):
            try:
                loader = WebBaseLoader([url])
                content = loader.load()[0].page_content

                portfolio.load_portfolio()
                jobs = llm.extract_jobs(content)

                records = []

                for job in jobs:
                    title = job.get("role", "Unknown Role")
                    skills = job.get("skills", "")

                    if isinstance(skills, str):
                        skills = [s.strip() for s in skills.split(",") if s.strip()]

                    links = portfolio.query_links(skills) if skills else []
                    email = llm.write_mail(job, links)

                    st.subheader(title)
                    st.markdown(f"**Skills:** {', '.join(skills) if skills else 'N/A'}")
                    st.code(email)

                    records.append({
                        "Role": title,
                        "Skills": ", ".join(skills),
                        "Email": email
                    })

                if records:
                    df = pd.DataFrame(records)
                    st.download_button(
                        "Download CSV",
                        df.to_csv(index=False),
                        "emails.csv",
                        "text/csv"
                    )

            except Exception as e:
                st.error(f"Error: {e}")


try:
    chain = Chain()
    portfolio = Portfolio()
    create_stream_app(chain, portfolio)
except Exception as e:
    st.error(str(e))
