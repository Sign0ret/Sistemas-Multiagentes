from fastapi import FastAPI
from pydantic import BaseModel
import logging
from robotsController import robotsModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class Item(BaseModel):
    data: str

@app.get("/step")
async def step():
    logger.info(f"Step: 1")
    result = robotsModel()  # Call the function from logic.py
    return {result}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000)