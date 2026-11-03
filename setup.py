from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

## edit below variables as per your requirements -
REPO_NAME = "Book-Recommender-MLOps"
AUTHOR_USER_NAME = "Tushar Nile"
SRC_REPO = "books_recommender"
LIST_OF_REQUIREMENTS = []


setup(
    name=SRC_REPO,
    version="0.0.1",
    author="Tushar Nile",
    description="An End-to-End Book Recommendation System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tusharaitechie/Book-Recommender-MLOps",
    author_email="tusharaitechie@gmail.com",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=LIST_OF_REQUIREMENTS
)
