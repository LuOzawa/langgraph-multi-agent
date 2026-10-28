"""HTTP API 层（可选依赖）。

只安装基础包时导入本模块不会引入 FastAPI；需要先在仓库根目录
`pip install -e ".[api]"`，再 `from multi_agent.api.app import create_app` 或运行 `multi-agent-api`。
"""
