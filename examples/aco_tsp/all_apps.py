"""
Configure visualization elements and instantiate a server
"""

import solara
from app_aco import build_page as build_aco_page
from app_as import build_page as build_as_page
from app_evo import build_page as build_evo_page

# read README.md into a string
with open("README.md") as f:
    readme_text = f.read()
readme_page = solara.Markdown(
    md_text=readme_text,
)


routes = [
    solara.Route(path="/", component=readme_page, label="README"),
    solara.Route(path="ant-system", component=build_as_page, label="Ant System"),
    solara.Route(
        path="ant-colony-optimization",
        component=build_aco_page,
        label="Ant Colony Optimization",
    ),
    solara.Route(path="evo", component=build_evo_page, label="Evo"),
]
