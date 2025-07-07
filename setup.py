from setuptools import setup, find_packages

setup(
    name="src",
    version="1.0.0",
    author="Lucrezia Laraspata",
    author_email="l.laraspata3@phd.uniba.it",
    description=r'Detect hallucinations in LLMs',
    packages=find_packages(),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/llaraspata/HallucinationDetection.git",
    python_requires='>=3.11',
)
