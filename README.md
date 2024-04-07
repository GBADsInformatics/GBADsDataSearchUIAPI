# Virtual Environment

### To create a virtual environment:
`pip install virtualenv`
`python3.9 -m venv env`

### To activate, use the following command:

Linux:
`source env/bin/activate`

Windows:
`cd /d env\Scripts && activate && cd ../..`

### To deactivate, use the following command:
`deactivate`

### Installation
`pip install -r requirements.txt`

## More PREREQ's

The following glove zip file will also be needed. Within this zip file is a file called:

`glove.6B.50d.txt`

you can download the zip with the following command:

`curl -L -o glove.6B.zip https://nlp.stanford.edu/data/glove.6B.zip`

place the `glove.6B.50d.txt` file in the root of this project

## Running the server
`uvicorn main:app --reload`

## Testing & Linting
Testing and linting can be applied with the following command:
`nox`