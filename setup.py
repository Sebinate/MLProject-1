from setuptools import find_packages, setup
from typing import List


HYPHEN_E_DOT = '-e .'
def get_requirements(file_path: str) -> List[str]:
    """
    This function creates a list of the needed requirements
    """    
    requirements_list = []
    
    with open(file_path, 'r') as file:
        requirements = file.readlines()
        
        for req in requirements:
            req = req.strip()
            if req != HYPHEN_E_DOT:
                requirements_list.append(req)
    
    return requirements_list   
    
setup(
    name = 'MLProj-1',
    version = '0.0.1',
    author = 'Sebastian',
    author_email = 'sebastiansampao@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)