from fastapi import FastAPI
from pydantic import BaseModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class Item(BaseModel):
    data: str

@app.post("/process")
async def process(item: Item):
    # Log the data
    logger.info(f"Received dataa: {item.data}")
    return {"status": "success", "processed_data": item.data}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000)