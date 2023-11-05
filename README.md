# PhotoHolmes

Acá va a haber un proyecto de fin de grado. Todavía no esta, pero va a estarlo.

## Development setup

To start developing, clone this repository with:
```
git clone git@gitlab.fing.edu.uy:photoholmes/photoholmes.git photoholmes
```

Create a virtual enviroment, either with conda or with pip. Activate
the enviroment and install the library and required packages with:
```
pip install -r requirements.txt
pip intsall -r requirements-dev.txt
pip install -e .
```

### Pre-commit hooks install

Pre-commit runs check before a commit to ensure the code quality is being preserved. To 
install the git hooks, run:
```bash
pre-commit install
```


### TODO
[ ] add the requirements.txt to the setup so we can skip the first pip install.

