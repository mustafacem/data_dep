# Insight Engine - ClueBox

AI backend for the Insight Engine:

- Business development buddy in your pocket
- Sparring partner for client engagement
- Customised to your needs
- Insights available instantly

## About The Project

RAG

### Built With

## Getting Started

### Prerequisites

### Installation

## Usage

### ENV variables
To configure the application, you'll need to set up several environment variables. You can find a template of the required variables in the `.env.example` file. The code is looking for this vars in `.env`.

- `DB_COLLECTION`: name of the vector db collection the Agents will use

## Development guidelines

### Pre-commit tool

This project uses the [pre-commit](https://pre-commit.com) tool to run some code checks before every commit. For all the hooks look at `.pre-commit-config.yaml`.

To install the Git hooks so that they are run before every commit, [install the pre-commit tool](https://pre-commit.com/#installation) and execute the following command:

```
pre-commit install
```
