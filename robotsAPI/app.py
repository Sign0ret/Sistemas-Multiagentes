from fastapi import FastAPI
from pydantic import BaseModel
import logging
from typing import List, Tuple
from robotsController import robotsModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Define the input model using Pydantic
class StepInput(BaseModel):
    m: int
    n: int
    steps: int
    robots: List[Tuple[int, int]]
    boxes: List[Tuple[int, int]]
    containers: List[Tuple[int, int]]
    begin: bool

@app.post("/step")
async def step(input_data: StepInput):
    #logger.info(f"Step: {input_data.steps}")
    #logger.info(f"Step: {input_data.begin}")
    # Access the data passed in the POST request
    parameters = {
        'M': input_data.m,
        'N': input_data.n,
        'robots': input_data.robots,
        'boxes': input_data.boxes,  
        'containers': input_data.containers,
        'steps': input_data.steps,
        'begin': input_data.begin
    }
    result = robotsModel(parameters)  # Pass the data to your function
    
    return result

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000)