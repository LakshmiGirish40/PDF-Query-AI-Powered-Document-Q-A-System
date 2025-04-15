import uvicorn
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import os
import tempfile

from app.utils import extract_text_from_pdf, create_vector_database, retrieve_relevant_chunks, query_groq_llm

from fastapi import FastAPI, File, UploadFile, Form, Header
app = FastAPI()

@app.post("/ask")
async def ask_question(
    pdf: UploadFile = File(...),
    question: str = Form(...),
    groq_api_key: str = Header(..., alias="GROQ-API-Key")
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await pdf.read())
        tmp_path = tmp.name

    try:
        text = extract_text_from_pdf(tmp_path)
        text_chunks = text.split("\n")

        index, _ = create_vector_database(text_chunks)
        relevant = retrieve_relevant_chunks(question, index, text_chunks)
        context = "\n".join(relevant)

        # Pass API key to the query function
        answer = query_groq_llm(question, context, api_key=groq_api_key)

        return JSONResponse(content={"answer": answer})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        os.remove(tmp_path)
