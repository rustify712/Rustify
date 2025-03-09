from argparse import ArgumentParser

from core.config import Config
# from core.db.session import init_db
from core.agents.project_manager import ProjectManager

parser = ArgumentParser(description="TransFactor")
parser.add_argument("-p", "--project", help="project directory", required=True)
args = parser.parse_args()

if args.project:
    # 初始化数据库
    # init_db()
    # 启动转译
    project_dir = args.project
    project_manager = ProjectManager(llm_config=Config.LLM_CONFIG)
    project_manager.start(project_dir=args.project)
else:
    print("Please specify the project directory")

# init_db()
# INPUT_DIR = "../../Input"
# project_dir = INPUT_DIR + "/research/quadtree"
# project_manager = ProjectManager(llm_config=Config.LLM_CONFIG)
# project_manager.start(project_dir=project_dir)
