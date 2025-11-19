# src/eval/metrics.py
def compute_primary(results: dict) -> dict:
    """
    Primary metric: mission success rate = fraction of agents with score>0
    """
    leaderboard = results.get("leaderboard", [])
    if not leaderboard:
        return {"mission_success_rate": 0.0}
    success_count = sum(1 for r in leaderboard if r.get("score", 0) > 0)
    return {"mission_success_rate": success_count / len(leaderboard)}
