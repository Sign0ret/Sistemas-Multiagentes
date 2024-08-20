## SISTEMAS MULTIAGENTES

STRATEGY => {
    Use a RESTFUL API:
    1-. First a POST which indicates the matrix size and the initial positions.
    	- Receives an array with the following shape:
		{{0,1,B,W}, {0,1,B,W}, {0,1,B,W}, {0,1,B,W}} 
	    		// 0 means nothing, 
			// 1 or number means how much on that space, 
			// B means Bot,
			// W means Wall.
	- Generate a local database for each bot state (or keep the image of the previous state? like screenshot?)
    2-. Second a UPDATE to update during each step each bot configuration.
    3-. Third     	- 
}

Wright an API
https://www.linode.com/docs/guides/create-restful-api-using-python-and-flask/

- Compile

  export FLASK_APP=prog_lang_app.py
  flask run

- Run the get
  curl http://127.0.0.1:5000/programming_languages

