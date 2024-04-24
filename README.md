# Lucky Passbook

A book recommender system.


## Installation

First, you need to install [pypoetry](https://python-poetry.org/docs/) for dependencies management. And run the following commands at the project's root dir for installation.

```
# if you don't need poetry to manage your venv
poetry config virtualenvs.create false

# install the project
poetry install
```

## Test Data
You can download the
[cleaned database](https://drive.google.com/file/d/1k-IeJkeRWoQQ1VtD71pfqNm9ewxGTCxS/view?usp=drive_link)
and put it into the `./instance` dir for developing or testing.

## Server

Once you installed the project, you can run the command `flask --app recom_system.server run` for a dev server. And access http://localhost:5000 for the demo page.
