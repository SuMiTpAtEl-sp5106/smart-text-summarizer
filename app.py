from fastapi import FastAPI,Request
import re
import torch
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration,T5Tokenizer
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(title="Text summarizer app", description="text summerizer using t5",version="1.0")

model=T5ForConditionalGeneration.from_pretrained("./saved_model")
Tokenizer=T5Tokenizer.from_pretrained("./saved_model")

if torch.backends.mps.is_available():
    device =torch.device("mps")
elif torch.cuda.is_available():
    device=torch.device("cuda")
else:
    device = torch.device("cpu")
model.to(device)

templates = Jinja2Templates(directory=".")

class DialogueInput(BaseModel):
    dialogue: str

def clean_data(text):
    text = re.sub(r"\r\n"," ",text )
    text = re.sub(r"\s+", " " ,text)
    text = re.sub(r"<.*?>"," ",text)
    text = text.strip().lower()
    return text


def summary_generate(data:str):
  data = clean_data(data)

  #convert inot token
  inputs = Tokenizer(
    data,             
    padding="max_length",
    max_length=512,
    truncation=True,
    return_tensors="pt"
  ).to(device)
   
  #genrate output as token id
  model.to(device)
  output = model.generate(
    input_ids = inputs["input_ids"],
    attention_mask = inputs["attention_mask"],
    max_length = 150,
    num_beams = 4,
    early_stopping=True
  )

  #token_id to summary
  summary = Tokenizer.decode(output[0],max_length = 120,skip_special_tokens=True)

  return summary


@app.get("/")
async def home():
    return FileResponse("index.html")

@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    summary = summary_generate(dialogue_input.dialogue)
    return {"summary": summary}