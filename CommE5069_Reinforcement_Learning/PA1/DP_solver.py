import numpy as np
from gridworld import GridWorld

#### implementation of the three async methods from class + novel method start ####
# import numpy as np
# from gridworld import GridWorld

# np.set_printoptions(threshold=np.inf, linewidth=np.inf)

# STEP_REWARD = -1.0
# GOAL_REWARD = 1.0
# TRAP_REWARD = -1.0
# DISCOUNT_FACTOR = 0.9


# class DynamicProgramming:
#     """Base class for dynamic programming algorithms"""

#     def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
#         self.grid_world = grid_world
#         self.discount_factor = discount_factor
#         self.threshold = 1e-4
#         self.values = np.zeros(grid_world.get_state_space())
#         self.policy = np.zeros(grid_world.get_state_space(), dtype=int)

#     def get_q_value(self, state: int, action: int) -> float:
#         next_state, reward, done = self.grid_world.step(state, action)
#         q_value = reward + self.discount_factor * self.values[next_state] * (1 - done)
#         return q_value


# class InPlaceValueIteration(DynamicProgramming):
#     """Class Method 1: In-Place Dynamic Programming"""
    
#     def run(self) -> None:
#         """In-place value iteration - single copy of value function"""
#         while True:
#             delta = 0
#             # In-place updates - update immediately within same iteration
#             for state in range(self.grid_world.get_state_space()):
#                 old_value = self.values[state]
                
#                 # Compute max Q-value
#                 max_q = float('-inf')
#                 for action in range(self.grid_world.get_action_space()):
#                     q_value = self.get_q_value(state, action)
#                     max_q = max(max_q, q_value)
                
#                 self.values[state] = max_q
#                 delta = max(delta, abs(old_value - self.values[state]))
            
#             if delta < self.threshold:
#                 break
        
#         # Extract policy
#         for state in range(self.grid_world.get_state_space()):
#             q_values = []
#             for action in range(self.grid_world.get_action_space()):
#                 q_values.append(self.get_q_value(state, action))
#             self.policy[state] = np.argmax(q_values)


# class PrioritizedSweeping(DynamicProgramming):
#     """Class Method 2: Prioritized Sweeping (standard implementation)"""
    
#     def run(self) -> None:
#         """Prioritized sweeping - backup states by Bellman error magnitude"""
#         import heapq
        
#         # Build predecessors by querying all transitions once
#         predecessors = [set() for _ in range(self.grid_world.get_state_space())]
#         for s in range(self.grid_world.get_state_space()):
#             for a in range(self.grid_world.get_action_space()):
#                 ns, _, _ = self.grid_world.step(s, a)
#                 predecessors[ns].add(s)
        
#         # Initialize priority queue
#         pq = []
#         in_queue = set()
        
#         for s in range(self.grid_world.get_state_space()):
#             max_q = float('-inf')
#             for a in range(self.grid_world.get_action_space()):
#                 q = self.get_q_value(s, a)
#                 max_q = max(max_q, q)
            
#             priority = abs(max_q - self.values[s])
#             if priority > self.threshold:
#                 heapq.heappush(pq, (-priority, s))
#                 in_queue.add(s)
        
#         # Main loop - backup highest priority states
#         while pq:
#             _, s = heapq.heappop(pq)
#             in_queue.remove(s)
            
#             # Update state value (queries environment)
#             max_q = float('-inf')
#             for a in range(self.grid_world.get_action_space()):
#                 q = self.get_q_value(s, a)
#                 max_q = max(max_q, q)
            
#             self.values[s] = max_q
            
#             # Update predecessors
#             for pred in predecessors[s]:
#                 max_q_pred = float('-inf')
#                 for a in range(self.grid_world.get_action_space()):
#                     q = self.get_q_value(pred, a)
#                     max_q_pred = max(max_q_pred, q)
                
#                 priority = abs(max_q_pred - self.values[pred])
                
#                 if priority > self.threshold and pred not in in_queue:
#                     heapq.heappush(pq, (-priority, pred))
#                     in_queue.add(pred)
        
#         # Extract policy
#         for s in range(self.grid_world.get_state_space()):
#             q_values = []
#             for a in range(self.grid_world.get_action_space()):
#                 q_values.append(self.get_q_value(s, a))
#             self.policy[s] = np.argmax(q_values)


# class RealTimeDynamicProgramming(DynamicProgramming):
#     """Class Method 3: Real-Time Dynamic Programming"""
    
#     def run(self) -> None:
#         """RTDP - backup states along agent trajectories"""
#         # Run multiple episodes to ensure convergence
#         max_episodes = 1000
#         max_steps_per_episode = 100
        
#         for episode in range(max_episodes):
#             # Start from state 0 (or random state)
#             state = 0
#             episode_states = set()
            
#             for step in range(max_steps_per_episode):
#                 # Track visited states
#                 episode_states.add(state)
                
#                 # Backup current state
#                 old_value = self.values[state]
#                 max_q = float('-inf')
#                 best_action = 0
                
#                 for action in range(self.grid_world.get_action_space()):
#                     q = self.get_q_value(state, action)
#                     if q > max_q:
#                         max_q = q
#                         best_action = action
                
#                 self.values[state] = max_q
                
#                 # Take action (greedy with respect to current values)
#                 next_state, reward, done = self.grid_world.step(state, best_action)
                
#                 if done:
#                     break
                
#                 state = next_state
            
#             # Check convergence: if values stable across multiple episodes
#             if episode > 10 and episode % 10 == 0:
#                 # Run test episode to check stability
#                 test_state = 0
#                 all_stable = True
#                 visited = set()
                
#                 for _ in range(max_steps_per_episode):
#                     if test_state in visited:
#                         break
#                     visited.add(test_state)
                    
#                     old_val = self.values[test_state]
#                     max_q = float('-inf')
#                     best_a = 0
                    
#                     for a in range(self.grid_world.get_action_space()):
#                         ns, r, d = self.grid_world.step(test_state, a)
#                         q = r + self.discount_factor * self.values[ns] * (1 - d)
#                         if q > max_q:
#                             max_q = q
#                             best_a = a
                    
#                     if abs(max_q - old_val) > self.threshold:
#                         all_stable = False
#                         break
                    
#                     test_state, _, done = self.grid_world.step(test_state, best_a)
#                     if done:
#                         break
                
#                 if all_stable:
#                     break
        
#         # Extract policy
#         for s in range(self.grid_world.get_state_space()):
#             q_values = []
#             for a in range(self.grid_world.get_action_space()):
#                 q_values.append(self.get_q_value(s, a))
#             self.policy[s] = np.argmax(q_values)


# class ModelBasedPrioritizedSweeping(DynamicProgramming):
#     """Novel Method: Model-Based Prioritized Sweeping"""
    
#     def run(self) -> None:
#         """Model-based prioritized sweeping - separate model learning from planning"""
#         import heapq
        
#         # Phase 1: Build model (only phase that calls step)
#         model = {}
#         for s in range(self.grid_world.get_state_space()):
#             for a in range(self.grid_world.get_action_space()):
#                 ns, r, d = self.grid_world.step(s, a)
#                 model[(s, a)] = (ns, r, d)
        
#         # Build predecessors from model
#         predecessors = [set() for _ in range(self.grid_world.get_state_space())]
#         for s in range(self.grid_world.get_state_space()):
#             for a in range(self.grid_world.get_action_space()):
#                 ns, _, _ = model[(s, a)]
#                 predecessors[ns].add(s)
        
#         # Phase 2: Planning (uses cached model, no step calls)
#         pq = []
#         in_queue = set()
        
#         for s in range(self.grid_world.get_state_space()):
#             max_q = float('-inf')
#             for a in range(self.grid_world.get_action_space()):
#                 ns, r, d = model[(s, a)]
#                 q = r + self.discount_factor * self.values[ns] * (1 - d)
#                 max_q = max(max_q, q)
            
#             priority = abs(max_q - self.values[s])
#             if priority > self.threshold:
#                 heapq.heappush(pq, (-priority, s))
#                 in_queue.add(s)
        
#         while pq:
#             _, s = heapq.heappop(pq)
#             in_queue.remove(s)
            
#             max_q = float('-inf')
#             for a in range(self.grid_world.get_action_space()):
#                 ns, r, d = model[(s, a)]
#                 q = r + self.discount_factor * self.values[ns] * (1 - d)
#                 max_q = max(max_q, q)
            
#             self.values[s] = max_q
            
#             for pred in predecessors[s]:
#                 max_q_pred = float('-inf')
#                 for a in range(self.grid_world.get_action_space()):
#                     ns, r, d = model[(pred, a)]
#                     q = r + self.discount_factor * self.values[ns] * (1 - d)
#                     max_q_pred = max(max_q_pred, q)
                
#                 priority = abs(max_q_pred - self.values[pred])
                
#                 if priority > self.threshold and pred not in in_queue:
#                     heapq.heappush(pq, (-priority, pred))
#                     in_queue.add(pred)
        
#         # Phase 3: Extract policy from model
#         for s in range(self.grid_world.get_state_space()):
#             best_a = 0
#             max_q = float('-inf')
#             for a in range(self.grid_world.get_action_space()):
#                 ns, r, d = model[(s, a)]
#                 q = r + self.discount_factor * self.values[ns] * (1 - d)
#                 if q > max_q:
#                     max_q = q
#                     best_a = a
#             self.policy[s] = best_a


# def test_method(method_class, method_name, grid_world):
#     print(f"\n{'='*60}")
#     print(f"{method_name}")
#     print(f"{'='*60}")
    
#     grid_world.reset()
#     method = method_class(grid_world, discount_factor=DISCOUNT_FACTOR)
#     method.run()
    
#     steps = grid_world.get_step_count()
#     print(f"Steps: {steps}")
    
#     # Verify policy reaches goal
#     grid_world.reset()
#     history = grid_world.run_policy(method.policy, 0)
#     print(f"Start state: {history[0][0]}, End state: {history[-1][0]}")
    
#     return steps


# if __name__ == "__main__":
#     print("Testing All Async DP Methods")
#     print("="*60)
    
#     grid_world = GridWorld(
#         "maze.txt",
#         step_reward=STEP_REWARD,
#         goal_reward=GOAL_REWARD,
#         trap_reward=TRAP_REWARD,
#     )
    
#     grid_world.print_maze()
#     print()
    
#     # Test all four methods (3 from class + 1 novel)
#     steps1 = test_method(InPlaceValueIteration, 
#                          "Class Method 1: In-Place Value Iteration", 
#                          grid_world)
    
#     steps2 = test_method(PrioritizedSweeping, 
#                          "Class Method 2: Prioritized Sweeping", 
#                          grid_world)
    
#     steps3 = test_method(RealTimeDynamicProgramming,
#                          "Class Method 3: Real-Time Dynamic Programming",
#                          grid_world)
    
#     steps4 = test_method(ModelBasedPrioritizedSweeping, 
#                          "Novel Method: Model-Based Prioritized Sweeping", 
#                          grid_world)
    
#     # Summary
#     print(f"\n{'='*60}")
#     print("SUMMARY")
#     print(f"{'='*60}")
#     print(f"Class Method 1 (In-Place VI):        {steps1:4d} steps")
#     print(f"Class Method 2 (Prioritized Sweep):  {steps2:4d} steps")
#     print(f"Class Method 3 (Real-Time DP):       {steps3:4d} steps")
#     print(f"Novel Method (Model-Based PS):       {steps4:4d} steps")
#     print(f"\nNovel method speedup vs Class Method 1: {steps1/steps4:.2f}x")
#     print(f"Novel method speedup vs Class Method 2: {steps2/steps4:.2f}x")
#     print(f"Novel method speedup vs Class Method 3: {steps3/steps4:.2f}x")
#     print(f"{'='*60}")

#### implementation of the three async methods from class + novel method end ####

class DynamicProgramming:
    """Base class for dynamic programming algorithms"""

    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for DynamicProgramming

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        self.grid_world = grid_world
        self.discount_factor = discount_factor
        self.threshold = 1e-4  # default threshold for convergence
        self.values = np.zeros(grid_world.get_state_space())  # V(s)
        self.policy = np.zeros(grid_world.get_state_space(), dtype=int)  # pi(s)

    def set_threshold(self, threshold: float) -> None:
        """Set the threshold for convergence

        Args:
            threshold (float): threshold for convergence
        """
        self.threshold = threshold

    def get_policy(self) -> np.ndarray:
        """Return the policy

        Returns:
            np.ndarray: policy
        """
        return self.policy

    def get_values(self) -> np.ndarray:
        """Return the values

        Returns:
            np.ndarray: values
        """
        return self.values

    def get_q_value(self, state: int, action: int) -> float:
        """Get the q-value for a state and action
        
        Bellman Equation with Done Flag:
        Q(s,a) = R(s,a) + γ * V(s') * (1 - done)
        
        where (s', R, done) = step(s, a)

        Args:
            state (int): Current state
            action (int): Action to take

        Returns:
            float: Q-value for the state-action pair
        """
        next_state, reward, done = self.grid_world.step(state, action)
        q_value = reward + self.discount_factor * self.values[next_state] * (1 - done)
        return q_value


class IterativePolicyEvaluation(DynamicProgramming):
    """
    Task 1: Iterative Policy Evaluation
    
    Evaluates a given stochastic policy π using the Bellman expectation equation.
    
    Bellman Expectation Equation:
    V_π(s) = Σ_a π(a|s) Σ_{s',r} p(s',r|s,a)[r + γ*V_π(s')]
    
    Simplified for our implementation:
    V_π(s) = Σ_a π(a|s) * Q(s,a)
    
    Algorithm:
    1. Initialize V(s) = 0 for all s ∈ S
    2. Loop:
        Δ ← 0
        For each s ∈ S:
            v ← V(s)
            V(s) ← Σ_a π(a|s) * [R(s,a) + γ*V(s')]  # Synchronous update
            Δ ← max(Δ, |v - V(s)|)
        Until Δ < θ
    """
    
    def __init__(
        self, grid_world: GridWorld, policy: np.ndarray, discount_factor: float
    ):
        """Constructor for IterativePolicyEvaluation

        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): policy (probability distribution state_space x 4)
            discount (float): discount factor gamma
        """
        super().__init__(grid_world, discount_factor)
        self.policy = policy

    def get_state_value(self, state: int) -> float:
        """Get the value for a state under the given policy
        
        Formula: V_π(s) = Σ_a π(a|s) * Q(s,a)

        Args:
            state (int): State index

        Returns:
            float: Expected value of state under policy π
        """
        # Sum over all actions weighted by policy probability
        value = 0.0
        for action in range(self.grid_world.get_action_space()):
            q_value = self.get_q_value(state, action)
            value += self.policy[state, action] * q_value
        return value

    def evaluate(self):
        """Evaluate the policy and update the values for one iteration
        
        Performs synchronous update: compute all new values first, then update all at once
        """
        new_values = np.zeros_like(self.values)
        for state in range(self.grid_world.get_state_space()):
            new_values[state] = self.get_state_value(state)
        self.values = new_values

    def run(self) -> None:
        """Run the algorithm until convergence
        
        Convergence criterion: max_s |V_new(s) - V_old(s)| < θ
        """
        while True:
            old_values = self.values.copy()
            self.evaluate()
            # Calculate maximum change across all states
            delta = np.max(np.abs(old_values - self.values))
            if delta < self.threshold:
                break


class PolicyIteration(DynamicProgramming):
    """
    Task 2: Policy Iteration
    
    Finds the optimal deterministic policy by alternating between:
    1. Policy Evaluation: Compute V_π(s) for current policy π
    2. Policy Improvement: Update π to be greedy w.r.t. V
    
    Policy Evaluation (for deterministic policy):
    V_π(s) = Σ_{s',r} p(s',r|s,π(s))[r + γ*V_π(s')]
    Simplified: V_π(s) = Q(s, π(s))
    
    Policy Improvement:
    π'(s) = argmax_a Σ_{s',r} p(s',r|s,a)[r + γ*V_π(s')]
    Simplified: π'(s) = argmax_a Q(s,a)
    
    Algorithm:
    1. Initialization: V(s) ∈ ℝ and π(s) ∈ A(s) arbitrarily for all s ∈ S
    2. Policy Evaluation:
        Loop until Δ < θ:
            For each s ∈ S:
                v ← V(s)
                V(s) ← Q(s, π(s))
                Δ ← max(Δ, |v - V(s)|)
    3. Policy Improvement:
        policy_stable ← true
        For each s ∈ S:
            old_action ← π(s)
            π(s) ← argmax_a Q(s,a)
            if old_action ≠ π(s): policy_stable ← false
        if policy_stable: return π, V
        else: go to step 2
    """
    
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for PolicyIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)

    def get_state_value(self, state: int) -> float:
        """Get the value for a state under current deterministic policy
        
        Formula: V_π(s) = Q(s, π(s))
        
        For deterministic policy, we only evaluate the action selected by π

        Args:
            state (int): State index

        Returns:
            float: Value of state under current policy
        """
        action = self.policy[state]
        return self.get_q_value(state, action)

    def policy_evaluation(self):
        """Evaluate the current policy and update values
        
        Iteratively compute V_π(s) until convergence using:
        V(s) ← Q(s, π(s))
        
        Uses synchronous updates for stability
        """
        while True:
            old_values = self.values.copy()
            new_values = np.zeros_like(self.values)
            for state in range(self.grid_world.get_state_space()):
                new_values[state] = self.get_state_value(state)
            self.values = new_values
            delta = np.max(np.abs(old_values - self.values))
            if delta < self.threshold:
                break

    def policy_improvement(self):
        """Improve the policy based on current values (greedy improvement)
        
        Formula: π'(s) = argmax_a Q(s,a)
        
        Returns:
            bool: True if policy is stable (no changes made), False otherwise
        """
        policy_stable = True
        for state in range(self.grid_world.get_state_space()):
            old_action = self.policy[state]
            # Find best action: π(s) ← argmax_a Q(s,a)
            q_values = []
            for action in range(self.grid_world.get_action_space()):
                q_values.append(self.get_q_value(state, action))
            self.policy[state] = np.argmax(q_values)
            if old_action != self.policy[state]:
                policy_stable = False
        return policy_stable

    def run(self) -> None:
        """Run policy iteration algorithm until convergence
        
        Alternates between policy evaluation and improvement until policy is stable
        """
        while True:
            # Step 1: Policy Evaluation - compute V_π
            self.policy_evaluation()
            # Step 2: Policy Improvement - update π greedily
            policy_stable = self.policy_improvement()
            # Step 3: Check convergence
            if policy_stable:
                break


class ValueIteration(DynamicProgramming):
    """
    Task 3: Value Iteration
    
    Combines policy evaluation and improvement in one step using Bellman optimality equation.
    
    Bellman Optimality Equation:
    V*(s) = max_a Σ_{s',r} p(s',r|s,a)[r + γ*V*(s')]
    Simplified: V*(s) = max_a Q(s,a)
    
    Optimal Policy Extraction:
    π*(s) = argmax_a Σ_{s',r} p(s',r|s,a)[r + γ*V*(s')]
    Simplified: π*(s) = argmax_a Q(s,a)
    
    Algorithm:
    1. Initialize V(s) = 0 for all s ∈ S^+, except V(terminal) = 0
    2. Loop:
        Δ ← 0
        For each s ∈ S:
            v ← V(s)
            V(s) ← max_a [R(s,a) + γ*V(s')]  # Synchronous update
            Δ ← max(Δ, |v - V(s)|)
        Until Δ < θ
    3. Output deterministic policy:
        π(s) = argmax_a Q(s,a)
    """
    
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for ValueIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)

    def get_state_value(self, state: int) -> float:
        """Get the optimal value for a state
        
        Formula: V*(s) = max_a Q(s,a)

        Args:
            state (int): State index

        Returns:
            float: Optimal value of the state
        """
        q_values = []
        for action in range(self.grid_world.get_action_space()):
            q_values.append(self.get_q_value(state, action))
        return max(q_values)

    def policy_evaluation(self):
        """Perform one iteration of Bellman optimality backup for all states
        
        Updates: V(s) ← max_a Q(s,a) for all s
        Uses synchronous updates
        """
        new_values = np.zeros_like(self.values)
        for state in range(self.grid_world.get_state_space()):
            new_values[state] = self.get_state_value(state)
        self.values = new_values

    def policy_improvement(self):
        """Extract the greedy policy from converged values
        
        Formula: π*(s) = argmax_a Q(s,a)
        """
        for state in range(self.grid_world.get_state_space()):
            q_values = []
            for action in range(self.grid_world.get_action_space()):
                q_values.append(self.get_q_value(state, action))
            self.policy[state] = np.argmax(q_values)

    def run(self) -> None:
        """Run value iteration until convergence
        
        Iteratively apply Bellman optimality operator until V converges,
        then extract optimal policy
        """
        while True:
            old_values = self.values.copy()
            self.policy_evaluation()
            delta = np.max(np.abs(old_values - self.values))
            if delta < self.threshold:
                break
        # Extract optimal policy after convergence
        self.policy_improvement()


class AsyncDynamicProgramming(DynamicProgramming):
    """
    Task 4: Asynchronous Dynamic Programming
    
    IMPLEMENTATION NOTE: This class implements the NOVEL METHOD - Model-Based Prioritized Sweeping.
    I have also implemented all three async methods from class (see documentation below).
    The novel method is chosen as the final submission because it significantly outperforms
    all standard async methods.
    
    ============================================================================
    THREE ASYNC METHODS FROM CLASS (implemented and tested):
    ============================================================================
    
    METHOD 1: In-Place Dynamic Programming
    ----------------------------------------
    Unlike synchronous VI (which uses V_old and V_new), in-place DP uses single array:
    
    For all s ∈ S:
        V(s) ← max_a [R_s^a + γ * Σ_{s'} P_{ss'}^a * V(s')]
    
    Advantages: Reduces memory, faster convergence (later states use updated values)
    Performance: 1,056 steps
    
    
    METHOD 2: Prioritized Sweeping
    --------------------------------
    Backs up states by urgency (Bellman error magnitude):
    
    Priority(s) = |max_a Q(s,a) - V(s)|
    
    Algorithm:
    1. Initialize priority queue with all states
    2. While queue not empty:
        - Pop state s with highest priority
        - Update V(s) ← max_a Q(s,a)
        - For each predecessor p: update priority and add to queue if > θ
    
    Requires reverse dynamics (predecessor states)
    Performance: 2,000 steps
    
    
    METHOD 3: Real-Time Dynamic Programming
    -----------------------------------------
    Uses agent experience to guide state selection:
    
    After each transition (s_t, a_t, r_{t+1}, s_{t+1}):
        V(s_t) ← max_a [R_{s_t}^a + γ * Σ_{s'} P_{s_t,s'}^a * V(s')]
    
    Algorithm:
    1. Start from initial state
    2. Follow greedy policy w.r.t. current values
    3. Backup current state after each step
    4. Run multiple episodes until convergence
    
    Focuses on relevant states along trajectories
    Performance: 1,628 steps
    
    ============================================================================
    NOVEL METHOD: Model-Based Prioritized Sweeping (IMPLEMENTED BELOW)
    ============================================================================
    
    Key Innovation: Separates model learning from planning
    
    Three Phases:
    
    Phase 1 - Model Learning:
    Build complete model M(s,a) = (s', r, done) by querying each (s,a) exactly once
    Cost: |S| × |A| environment steps
    
    Phase 2 - Planning (uses cached model, 0 additional steps):
    Priority(s) = |max_a [r(s,a) + γ*V(s'(s,a))*(1-done)] - V(s)|
    Update: V(s) ← max_a [r(s,a) + γ*V(s'(s,a))*(1-done(s,a))]
    All quantities retrieved from cached model M (no environment queries)
    
    Phase 3 - Policy Extraction:
    π*(s) = argmax_a [r(s,a) + γ*V(s'(s,a))*(1-done(s,a))]
    Uses cached model
    
    Difference from class methods:
    - In-Place DP: queries environment in every sweep through states
    - Prioritized Sweeping: queries during priority computation and value updates
    - Real-Time DP: queries during trajectory execution (multiple episodes)
    - NOVEL METHOD: queries ONLY during model building, then plans offline with 0 queries
    
    Performance Analysis:
    - Novel Method: 88 steps (|S|=22, |A|=4 → 22×4=88)
    - In-Place DP: 1,056 steps (12.00× slower)
    - Prioritized Sweeping: 2,000 steps (22.73× slower)
    - Real-Time DP: 1,628 steps (18.50× slower)
    - Value Iteration: 1,144 steps (13.00× slower)
    - Policy Iteration: 3,256 steps (37.00× slower)
    
    Theoretical justification: Achieves minimum possible steps for model-based methods
    in deterministic environments: exactly |S| × |A|
    """
    
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for AsyncDynamicProgramming

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)

    def run(self) -> None:
        """Run the asynchronous DP algorithm until convergence
        
        Implementation: Model-Based Prioritized Sweeping
        
        This method achieves optimal policy with minimal environment interaction
        by caching the complete environment model and using it for all planning.
        """
        import heapq
        
        # =====================================================================
        # PHASE 1: Model Learning
        # =====================================================================
        # Build complete deterministic model: M(s,a) = (s', r, done)
        # Query each state-action pair exactly once
        # Cost: |S| × |A| environment steps (only step() calls that count)
        # =====================================================================
        
        model = {}  # Cache: model[(s,a)] = (next_state, reward, done)
        
        for s in range(self.grid_world.get_state_space()):
            for a in range(self.grid_world.get_action_space()):
                # Query environment once per (s,a) pair
                next_state, reward, done = self.grid_world.step(s, a)
                model[(s, a)] = (next_state, reward, done)
        
        # Build predecessor graph: predecessors[s'] = {s : ∃a, s→s' under a}
        # Used for efficient backward propagation of value updates
        predecessors = [set() for _ in range(self.grid_world.get_state_space())]
        for s in range(self.grid_world.get_state_space()):
            for a in range(self.grid_world.get_action_space()):
                ns, _, _ = model[(s, a)]  # Use cached model (no step() call)
                predecessors[ns].add(s)
        
        # =====================================================================
        # PHASE 2: Planning (Zero additional environment steps)
        # =====================================================================
        # All operations below use the cached model - no step() calls
        # Priority queue implementation of asynchronous value iteration
        # =====================================================================
        
        # Initialize priority queue: (-priority, state) for max-heap behavior
        pq = []
        in_queue = set()  # Track which states are currently in queue
        
        # Compute initial priorities for all states
        for s in range(self.grid_world.get_state_space()):
            # Compute max_a Q(s,a) using cached model
            max_q = float('-inf')
            for a in range(self.grid_world.get_action_space()):
                ns, r, d = model[(s, a)]  # Cached lookup (no step() call)
                q = r + self.discount_factor * self.values[ns] * (1 - d)
                max_q = max(max_q, q)
            
            # Bellman error: |max_a Q(s,a) - V(s)|
            priority = abs(max_q - self.values[s])
            
            # Add to queue if priority exceeds threshold
            if priority > self.threshold:
                heapq.heappush(pq, (-priority, s))  # Negative for max-heap
                in_queue.add(s)
        
        # Main planning loop: iteratively update highest-priority states
        while pq:
            # Pop state with highest priority (largest Bellman error)
            _, s = heapq.heappop(pq)
            in_queue.remove(s)
            
            # Update state value using Bellman optimality operator
            # V(s) ← max_a [r(s,a) + γ*V(s'(s,a))*(1-done(s,a))]
            max_q = float('-inf')
            for a in range(self.grid_world.get_action_space()):
                ns, r, d = model[(s, a)]  # Cached lookup (no step() call)
                q = r + self.discount_factor * self.values[ns] * (1 - d)
                max_q = max(max_q, q)
            
            self.values[s] = max_q  # In-place update
            
            # Backward propagation: update priorities of predecessor states
            for pred in predecessors[s]:
                # Recompute max_a Q(pred, a) using cached model
                max_q_pred = float('-inf')
                for a in range(self.grid_world.get_action_space()):
                    ns, r, d = model[(pred, a)]  # Cached lookup
                    q = r + self.discount_factor * self.values[ns] * (1 - d)
                    max_q_pred = max(max_q_pred, q)
                
                # Compute new priority (Bellman error)
                priority = abs(max_q_pred - self.values[pred])
                
                # Add to queue if priority is significant and not already queued
                if priority > self.threshold and pred not in in_queue:
                    heapq.heappush(pq, (-priority, pred))
                    in_queue.add(pred)
        
        # =====================================================================
        # PHASE 3: Policy Extraction
        # =====================================================================
        # Extract greedy policy: π*(s) = argmax_a Q(s,a)
        # Uses cached model (no step() calls)
        # =====================================================================
        
        for s in range(self.grid_world.get_state_space()):
            best_action = 0
            max_q = float('-inf')
            
            # Find action with highest Q-value
            for a in range(self.grid_world.get_action_space()):
                ns, r, d = model[(s, a)]  # Cached lookup
                q = r + self.discount_factor * self.values[ns] * (1 - d)
                if q > max_q:
                    max_q = q
                    best_action = a
            
            self.policy[s] = best_action
        
        # Total environment steps: |S| × |A| (from Phase 1 only)
        # Planning steps (Phase 2): 0
        # Policy extraction steps (Phase 3): 0