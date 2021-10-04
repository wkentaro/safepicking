from setuptools import find_packages
from setuptools import setup


def main():
    setup(
        name="mercury",
        version="0.1",
        packages=find_packages(),
        install_requires=[],  # XXX: see requirements.txt
        author="Kentaro Wada",
        author_email="www.kentaro.wada@gmail.com",
        license="MIT",
        url="https://github.com/wkentaro/mercury",
        entry_points={
            "console_scripts": ["urdf_view=mercury.cli.urdf_view:main"]
        },
    )


if __name__ == "__main__":
    main()
