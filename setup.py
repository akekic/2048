from setuptools import setup


def get_install_requirements(path):
    with open(path) as f:
        requires = f.read().splitlines()
    return requires


setup(
    name='gym_env_2048',
    version='0.0.1',
    install_requires=get_install_requirements('requirements.txt'),  # And any other dependencies gym_env_2048 needs
)
