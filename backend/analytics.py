from typing import Dict, List, Optional
import json
from datetime import datetime, timedelta
import redis
from dataclasses import dataclass, asdict
import random

@dataclass
class UserPerformance:
    user_id: str
    total_attempts: int = 0
    correct_attempts: int = 0
    problem_history: List[Dict] = None
    difficulty_levels: Dict[str, int] = None
    
    def __post_init__(self):
        if self.problem_history is None:
            self.problem_history = []
        if self.difficulty_levels is None:
            self.difficulty_levels = {"easy": 0, "medium": 0, "hard": 0}

class PerformanceAnalytics:
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        
    def record_attempt(self, user_id: str, problem_id: str, 
                      difficulty: str, correct: bool, time_taken: float) -> None:
        """Record a user's attempt at solving a problem"""
        key = f"user_performance:{user_id}"
        performance = self._get_or_create_performance(user_id)
        
        # Update statistics
        performance.total_attempts += 1
        if correct:
            performance.correct_attempts += 1
            performance.difficulty_levels[difficulty] += 1
            
        # Add to history
        performance.problem_history.append({
            "problem_id": problem_id,
            "difficulty": difficulty,
            "correct": correct,
            "time_taken": time_taken,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 100 attempts in history
        if len(performance.problem_history) > 100:
            performance.problem_history = performance.problem_history[-100:]
            
        # Save to Redis
        self.redis_client.set(key, json.dumps(asdict(performance)))
        
    def get_user_stats(self, user_id: str) -> Dict:
        """Get comprehensive statistics for a user"""
        performance = self._get_or_create_performance(user_id)
        
        success_rate = (performance.correct_attempts / performance.total_attempts * 100 
                       if performance.total_attempts > 0 else 0)
                       
        recent_performance = self._analyze_recent_performance(performance.problem_history)
        
        return {
            "overall_success_rate": round(success_rate, 2),
            "total_problems_solved": performance.correct_attempts,
            "difficulty_breakdown": performance.difficulty_levels,
            "recent_performance": recent_performance,
            "suggested_difficulty": self._suggest_next_difficulty(performance)
        }
        
    def _get_or_create_performance(self, user_id: str) -> UserPerformance:
        """Get or create a new user performance record"""
        key = f"user_performance:{user_id}"
        data = self.redis_client.get(key)
        
        if data:
            data_dict = json.loads(data)
            return UserPerformance(**data_dict)
        return UserPerformance(user_id=user_id)
        
    def _analyze_recent_performance(self, history: List[Dict]) -> Dict:
        """Analyze user's recent performance trends"""
        if not history:
            return {"trend": "neutral", "success_rate": 0}
            
        recent_attempts = history[-10:]  # Look at last 10 attempts
        recent_success_rate = sum(1 for x in recent_attempts if x["correct"]) / len(recent_attempts) * 100
        
        return {
            "trend": "improving" if recent_success_rate > 70 else "needs_practice",
            "success_rate": round(recent_success_rate, 2)
        }
        
    def _suggest_next_difficulty(self, performance: UserPerformance) -> str:
        """Suggest next problem difficulty based on performance"""
        if performance.total_attempts < 5:
            return "easy"
            
        success_rate = performance.correct_attempts / performance.total_attempts * 100
        
        if success_rate > 80 and performance.difficulty_levels["easy"] > 5:
            return "medium"
        elif success_rate > 85 and performance.difficulty_levels["medium"] > 5:
            return "hard"
        elif success_rate < 40:
            return "easy"
        return "medium"

class ProblemGenerator:
    def __init__(self):
        self.templates = {
            "easy": [
                {
                    "type": "DFA",
                    "description": "Create a DFA that accepts all strings over {a, b} that end with 'ab'",
                    "solution_hint": "You'll need a state to track when you've seen 'a' and another for when you've seen 'ab'"
                },
                {
                    "type": "DFA",
                    "description": "Create a DFA that accepts all strings containing an even number of 0s",
                    "solution_hint": "Use two states to keep track of even/odd count"
                }
            ],
            "medium": [
                {
                    "type": "NFA",
                    "description": "Create an NFA that accepts strings that contain either 'abc' or 'bac'",
                    "solution_hint": "Use epsilon transitions to handle the alternative patterns"
                }
            ],
            "hard": [
                {
                    "type": "PDA",
                    "description": "Create a PDA that accepts the language {a^n b^n | n ≥ 0}",
                    "solution_hint": "Use the stack to count 'a's and match them with 'b's"
                }
            ]
        }
        
    def generate_problem(self, difficulty: str, user_performance: Optional[Dict] = None) -> Dict:
        """Generate a problem based on difficulty and user performance"""
        templates = self.templates.get(difficulty, self.templates["easy"])
        
        # Select template based on what the user hasn't seen recently
        if user_performance and user_performance.get("problem_history"):
            recent_problems = set(p["problem_id"] for p in user_performance["problem_history"][-5:])
            available_templates = [t for t in templates if hash(t["description"]) not in recent_problems]
            if available_templates:
                templates = available_templates
                
        template = random.choice(templates)
        
        return {
            "id": hash(template["description"]),
            "type": template["type"],
            "difficulty": difficulty,
            "description": template["description"],
            "solution_hint": template["solution_hint"],
            "test_cases": self._generate_test_cases(template["type"])
        }
        
    def _generate_test_cases(self, problem_type: str) -> List[Dict]:
        """Generate test cases based on problem type"""
        # Implementation would generate appropriate test cases
        # based on the problem type and complexity
        return [
            {"input": "example_input", "expected": True},
            {"input": "counter_example", "expected": False}
        ]