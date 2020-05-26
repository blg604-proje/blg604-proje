import setuptools

setuptools.setup(
    name="simstar",
    version="1.5.2",
    author="Eatron Technologies",
    author_email="info@eatron.com",
    description="ADAS Simulator based on Unreal Engine",
   packages=setuptools.find_packages(),
    license='Proprietary',
    classifiers=(
        "Programming Language :: Python :: 3",
    ),
    install_requires=[
          'msgpack-rpc-python', 'numpy','utm'
    ]
)