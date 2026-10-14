"""支持 python -m videolens；实际参数解析放在 cli.py 中。"""

from .cli import main

raise SystemExit(main())
