# Testing the Common Framework

The Common Framework is tested using the [pytest](https://docs.pytest.org/en/latest/) framework. The tests are located in the `commonframework/tests` directory. The tests are run using the `pytest` command. E.g.

```bash
pytest test_gnn.py
```

or simply

```bash
pytest
```

to run all tests.

Note that some tests rely on example data, which can be downloaded with

```bash
wget https://portal.nersc.gov/project/m3443/dtmurnane/GNN4ITk/TestData/test_files.zip
unzip test_files.zip
```