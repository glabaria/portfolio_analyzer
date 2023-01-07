from setuptools import setup

setup(name='portfolio_analyzer_tool',
      version='0.1.0',
      description='Portfolio Analyzer',
      author='George Labaria',
      packages=["portfolio_analyzer_tool"],
      entry_points={"console_scripts": ["portfolio_analyzer_tool=portfolio_analyzer_tool.cli:cli"]}
     )