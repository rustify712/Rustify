from argparse import ArgumentParser
import os
from core.config_back import Config
from core.translator import Translator
from core.agents.project_manager import ProjectManager

parser = ArgumentParser(description="TransFactor")
parser.add_argument("-p", "--project", help="project directory", required=True)
args = parser.parse_args()

if args.project:
    # 启动转译
    source_project_dir = args.project
    target_project_dir = args.project + "_rust"
    translator = Translator(
        prompt_folders=Config.PROMPT_PATHS,
        rustc_bin=Config.RUSTC_BIN,
        cargo_bin=Config.CARGO_BIN,
        llm_config=Config.LLM_CONFIGS[0],
        rag_config=Config.RAG_CONFIG,
        db_config={
            "url": Config.DB_URL,
            "debug_sql": False
        },
        reasoner_config=Config.LLM_CONFIGS[-1],
        state_file=os.path.join(target_project_dir, "states.json")
    )
    project_manager = ProjectManager(translator)
    project_manager.start(source_project_dir, target_project_dir)
else:
    print("Please specify the project directory")