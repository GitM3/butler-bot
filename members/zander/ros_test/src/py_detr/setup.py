from setuptools import find_packages, setup

package_name = 'py_detr'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/'+package_name+'/launch', [f"launch/{package_name}.launch.py"]),
    ],
    install_requires=[
        "setuptools",
        "supervision",
        "opencv-python",
        "torch",
    ],
    zip_safe=True,
    maintainer='zander',
    maintainer_email='zander@polsons.info',
    description='TODO: Package description bla bla',
    license='TODO: bla bla declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            f"video_node = {package_name}.video_node:main",
            f"rfdetr_node = {package_name}.rf_detr_node:main",
            f"tracker_node = {package_name}.tracker_node:main",
        ],
    },
)
