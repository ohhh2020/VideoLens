"""版本化提示词从配置读取，已接受任务使用提交时保存的原文。"""
def get_prompt(key, ledger=None):
    if ledger:
        saved = (ledger.get('policy') or {}).get('_prompts', {})
        if key in saved:
            return saved[key]
    from backend.prompts import get_prompt_registry
    return get_prompt_registry().get(key)
