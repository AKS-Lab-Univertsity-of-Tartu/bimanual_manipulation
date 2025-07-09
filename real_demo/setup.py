from setuptools import find_packages, setup
from glob import glob
import os


package_name = 'real_demo'

data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ]

folders = ['ur5e_hande_mjx', 'mj_planner', 'data', 'ik_based_planner', 'collision_free_ik', 'sampling_based_planner']

for folder in folders:
    for dirpath, dirnames, filenames in os.walk(folder):
        if filenames:
            files = [os.path.join(dirpath, f) for f in filenames]
            install_path = os.path.join('share', package_name, dirpath)
            data_files.append((install_path, files))

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='patsyuk',
    maintainer_email='patsyuk@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'dual_arm_demo = real_demo.dual_arm_demo:main',
            'visualizer = real_demo.visualizer:main'
        ],
    },
)
