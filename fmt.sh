#!/bin/bash
ruff check . --fix
isort .
ruff format --line-length 150 .