from setuptools import setup, find_packages

setup(
    name="src",
    version="1.0.0",
    author="Anonymous Authors",
    author_email="mail@domain.com",
    description=r'Detect hallucinations in LLMs',
    packages=find_packages(),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    python_requires='>=3.11',
)
