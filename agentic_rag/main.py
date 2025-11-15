from dotenv import load_dotenv

load_dotenv()

from graph.graph import app


if __name__ == "__main__":
    question = "What is React?"
    print(app.invoke(input={"question": question}))
