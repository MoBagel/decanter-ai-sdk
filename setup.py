import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()
print(long_description)

requires = [
    'requests>=2.25.1',
    'requests-toolbelt>=0.9.1',
    'pandas>=1.3.0',
    'matplotlib>=3.4.2',
    'tqdm>=4.61.2',
    'numpy>=1.21.0'
]

dev_requirements = [
    'twine',
    'tox',
    'pytest',
    'responses',
    'flake8',
    'pylint-quotes'
]

setuptools.setup(
    name='decanter-ai-core-sdk',
    author='Mobagel',
    author_email='us@mobagel.com',
    version='1.1.8',
    license='MIT',
    description='Decanter AI Core SDK for the easy use of Decanter Core API.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/MoBagel/decanter-ai-core-sdk',
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requires,
    extras_require={
        'dev': dev_requirements
    },
    test_suite='tests',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'
    ],
    python_requires='>=3.7'
)
