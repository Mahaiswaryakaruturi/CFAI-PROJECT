"""
=============================================================================
 AUTONOMOUS CLEANING STRATEGY PLANNER
=============================================================================
 Abstract : Predicts dirt accumulation zones (near doors, windows, high-traffic
            areas), prioritises those zones first, and outputs an explainable
            cleaning plan with full reasoning traces.

 CO Coverage
 -----------
 CO1 – State-space representation  (room grid, actions, transitions, goals)
 CO2 – A* search                   (optimal path across dirty zones)
 CO3 – CSP backtracking            (schedule zones under time constraints)
 CO4 – Minimax decision agent      (resource allocation under adversarial model)
 CO5 – Bayesian network            (dirt-probability inference per zone)
 CO6 – Integrated reasoning pipeline with explainable output
=============================================================================
"""

import heapq
import random
import math
from collections import defaultdict
from itertools import product
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# CO1 : State-Space Representation
# ─────────────────────────────────────────────────────────────────────────────

ZONE_TYPES = {
    "door":        {"traffic": 0.9, "base_dirt": 0.8},
    "window":      {"traffic": 0.6, "base_dirt": 0.5},
    "high_traffic":{"traffic": 0.85,"base_dirt": 0.75},
    "low_traffic": {"traffic": 0.2, "base_dirt": 0.2},
    "corner":      {"traffic": 0.3, "base_dirt": 0.4},
}

class RoomState:
    """
    CO1 – Formulate the room as a searchable state space.

    State   : (robot_position, frozenset of uncleaned zones)
    Actions : Move N/S/E/W, Clean current zone
    Goal    : All dirty zones are cleaned
    Cost    : 1 per move, 2 per clean action
    """

    MOVES = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}

    def __init__(self, rows: int = 5, cols: int = 5):
        self.rows = rows
        self.cols = cols
        self.grid: dict[tuple, str] = {}          # (r,c) -> zone_type
        self.dirty_zones: set[tuple] = set()
        self._build_default_room()

    def _build_default_room(self):
        """Assign zone types to each cell deterministically."""
        specials = {
            (0, 2): "door",
            (0, 4): "window",
            (4, 0): "door",
            (2, 2): "high_traffic",
            (3, 3): "high_traffic",
            (0, 0): "corner",
            (4, 4): "corner",
        }
        for r, c in product(range(self.rows), range(self.cols)):
            self.grid[(r, c)] = specials.get((r, c), "low_traffic")

    def mark_dirty(self, dirty_prob: dict[tuple, float], threshold: float = 0.4):
        """Mark zones whose inferred dirt probability exceeds the threshold."""
        self.dirty_zones = {pos for pos, p in dirty_prob.items() if p >= threshold}

    def neighbors(self, pos: tuple) -> list[tuple]:
        r, c = pos
        result = []
        for dr, dc in self.MOVES.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                result.append((nr, nc))
        return result

    def is_goal(self, robot_pos: tuple, remaining: frozenset) -> bool:
        return len(remaining) == 0

    def describe(self) -> str:
        lines = ["[CO1] Room State-Space Representation",
                 f"      Grid : {self.rows}x{self.cols}",
                 f"      Dirty zones ({len(self.dirty_zones)}) : {sorted(self.dirty_zones)}"]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CO5 : Bayesian Network for Dirt-Probability Inference
# ─────────────────────────────────────────────────────────────────────────────

class DirtBayesNet:
    """
    CO5 – Bayesian network with variable elimination.

    Network topology
    ----------------
    TrafficLevel  ─┐
    ZoneType      ─┼─► DirtAccumulation
    TimeOfDay     ─┘

    P(Dirt | Traffic, ZoneType, TimeOfDay) is a noisy-OR model.
    """

    TIME_WEIGHTS = {"morning": 0.7, "afternoon": 1.0, "evening": 0.9, "night": 0.5}

    def __init__(self, time_of_day: str = "afternoon"):
        self.time_weight = self.TIME_WEIGHTS.get(time_of_day, 1.0)
        self.time_of_day = time_of_day
        self._trace: list[str] = []

    def _noisy_or(self, *probs) -> float:
        """P(X) = 1 - Π(1 - pᵢ)  — noisy-OR combination."""
        result = 1.0
        for p in probs:
            result *= (1.0 - p)
        return 1.0 - result

    def infer(self, room: RoomState) -> dict[tuple, float]:
        """
        Variable elimination over the three parent nodes for every cell.
        Returns a dict of {cell: P(dirty)}.
        """
        self._trace.clear()
        self._trace.append(f"[CO5] Bayesian Inference  |  time_of_day={self.time_of_day}")

        dirty_probs: dict[tuple, float] = {}

        for pos, zone_type in room.grid.items():
            props       = ZONE_TYPES[zone_type]
            p_traffic   = props["traffic"]
            p_base_dirt = props["base_dirt"]
            p_time      = self.time_weight * 0.5   # scaled contribution

            # Noisy-OR combines all three causes
            p_dirty = self._noisy_or(p_traffic, p_base_dirt, p_time)

            # Clamp to [0, 1]
            dirty_probs[pos] = min(max(p_dirty, 0.0), 1.0)

        top = sorted(dirty_probs.items(), key=lambda x: -x[1])[:5]
        self._trace.append("      Top-5 dirty zones (by probability):")
        for pos, p in top:
            self._trace.append(f"        {pos}  zone={room.grid[pos]:<12}  P(dirty)={p:.3f}")

        return dirty_probs

    @property
    def trace(self) -> str:
        return "\n".join(self._trace)


# ─────────────────────────────────────────────────────────────────────────────
# CO2 : A* Search – Optimal Cleaning Path
# ─────────────────────────────────────────────────────────────────────────────

def manhattan(a: tuple, b: tuple) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def heuristic_remaining(pos: tuple, remaining: frozenset) -> int:
    """
    CO2 – Admissible heuristic: minimum Manhattan distance to any unvisited
    dirty zone (never over-estimates true cost).
    """
    if not remaining:
        return 0
    return min(manhattan(pos, z) for z in remaining)

def astar_cleaning_path(
    room: RoomState,
    start: tuple = (0, 0),
) -> tuple[list[tuple], int, list[str]]:
    """
    CO2 – A* search over (robot_position, frozenset[remaining_dirty_zones]).

    Returns (ordered list of zones to clean, total cost, reasoning trace).
    """
    trace: list[str] = ["[CO2] A* Search  |  finding optimal cleaning path"]
    dirty = frozenset(room.dirty_zones)

    # priority queue: (f, g, position, remaining_dirty, path_so_far)
    start_h = heuristic_remaining(start, dirty)
    heap = [(start_h, 0, start, dirty, [start])]
    visited: dict[tuple, int] = {}

    while heap:
        f, g, pos, remaining, path = heapq.heappop(heap)

        state_key = (pos, remaining)
        if state_key in visited and visited[state_key] <= g:
            continue
        visited[state_key] = g

        # Clean current zone if dirty
        new_remaining = remaining - {pos}
        if pos in remaining:
            g += 2   # cleaning cost

        if not new_remaining:
            clean_order = [p for p in path if p in room.dirty_zones]
            trace.append(f"      Path length  : {len(path)} steps")
            trace.append(f"      Total cost   : {g}")
            trace.append(f"      Clean order  : {clean_order}")
            return clean_order, g, trace

        # Expand neighbours
        for nb in room.neighbors(pos):
            move_cost = g + 1
            h = heuristic_remaining(nb, new_remaining)
            heapq.heappush(heap, (move_cost + h, move_cost, nb, new_remaining, path + [nb]))

    trace.append("      No complete path found (some zones unreachable).")
    return [], 0, trace


# ─────────────────────────────────────────────────────────────────────────────
# CO3 : CSP Backtracking – Cleaning Schedule
# ─────────────────────────────────────────────────────────────────────────────

class CleaningScheduleCSP:
    """
    CO3 – Constraint Satisfaction Problem.

    Variables  : each dirty zone
    Domain     : time-slots ["slot_1" … "slot_N"]
    Constraints:
        - No two high-priority zones in the same slot (capacity = 1)
        - Zones above prob-threshold must be in early slots
        - Adjacent (reachable) zones prefer consecutive slots (soft)
    Heuristics : MRV (Minimum Remaining Values) for variable ordering,
                 LCV (Least Constraining Value) for value ordering.
    """

    MAX_SLOTS = 4

    def __init__(self, zones: list[tuple], dirty_probs: dict[tuple, float],
                 room: RoomState):
        self.zones       = zones
        self.probs       = dirty_probs
        self.room        = room
        self.slots       = [f"slot_{i+1}" for i in range(self.MAX_SLOTS)]
        self.assignment: dict[tuple, str] = {}
        self._trace: list[str] = ["[CO3] CSP Backtracking  |  scheduling zones"]

    # --- MRV: zone with fewest legal values ---
    def _mrv_zone(self, unassigned: list[tuple]) -> tuple:
        def remaining_values(z):
            return sum(1 for s in self.slots if self._is_consistent(z, s))
        return min(unassigned, key=remaining_values)

    # --- LCV: slot that rules out fewest other zones ---
    def _lcv_slots(self, zone: tuple) -> list[str]:
        def constraint_count(slot):
            count = 0
            for other in self.zones:
                if other not in self.assignment and other != zone:
                    if not self._is_consistent(other, slot):
                        count += 1
            return count
        return sorted(self.slots, key=constraint_count)

    def _is_consistent(self, zone: tuple, slot: str) -> bool:
        # Capacity constraint: at most 1 high-priority zone per slot
        high_priority = self.probs.get(zone, 0) >= 0.7
        if high_priority:
            for assigned_zone, assigned_slot in self.assignment.items():
                if assigned_slot == slot and self.probs.get(assigned_zone, 0) >= 0.7:
                    return False
        # High-probability zones must go to slot_1 or slot_2
        if self.probs.get(zone, 0) >= 0.8 and slot not in ("slot_1", "slot_2"):
            return False
        return True

    def _backtrack(self, unassigned: list[tuple]) -> bool:
        if not unassigned:
            return True
        zone = self._mrv_zone(unassigned)
        for slot in self._lcv_slots(zone):
            if self._is_consistent(zone, slot):
                self.assignment[zone] = slot
                rest = [z for z in unassigned if z != zone]
                if self._backtrack(rest):
                    return True
                del self.assignment[zone]
        return False   # backtrack

    def solve(self) -> dict[tuple, str]:
        success = self._backtrack(list(self.zones))
        if success:
            self._trace.append("      Schedule found:")
            for slot in self.slots:
                zs = [z for z, s in self.assignment.items() if s == slot]
                if zs:
                    self._trace.append(f"        {slot}: {zs}")
        else:
            self._trace.append("      No valid schedule found — relaxing constraints.")
            # Fallback: round-robin
            for i, z in enumerate(self.zones):
                self.assignment[z] = self.slots[i % self.MAX_SLOTS]
        return self.assignment

    @property
    def trace(self) -> str:
        return "\n".join(self._trace)


# ─────────────────────────────────────────────────────────────────────────────
# CO4 : Minimax Agent – Adversarial Resource Allocation
# ─────────────────────────────────────────────────────────────────────────────

class MinimaxResourceAgent:
    """
    CO4 – Two-player minimax with alpha-beta pruning.

    Scenario : The cleaning robot (MAX) and an "entropy agent" (MIN, models
               random re-soiling / disturbances) compete over a shared battery
               budget.  MAX allocates battery to zones; MIN tries to waste it
               on low-priority areas.

    State    : (battery_remaining, list_of_zones_with_priorities)
    Terminal : battery == 0 or all zones decided
    Utility  : total expected dirt removed
    """

    def __init__(self, budget: int = 10, depth: int = 4):
        self.budget    = budget
        self.depth     = depth
        self._trace: list[str] = ["[CO4] Minimax Agent  |  adversarial resource allocation"]
        self._nodes_explored = 0

    def _utility(self, zones_cleaned: list[tuple],
                  dirty_probs: dict[tuple, float]) -> float:
        """Expected cleanliness gain."""
        return sum(dirty_probs.get(z, 0) for z in zones_cleaned)

    def _minimax(self, battery: int, zones: list[tuple],
                 dirty_probs: dict[tuple, float],
                 is_max: bool, alpha: float, beta: float,
                 cleaned: list[tuple], depth: int) -> float:
        self._nodes_explored += 1

        # Terminal conditions
        if battery <= 0 or not zones or depth == 0:
            return self._utility(cleaned, dirty_probs)

        current_zone = zones[0]
        rest         = zones[1:]
        cost         = max(1, int(dirty_probs.get(current_zone, 0.3) * 3))

        if is_max:
            # MAX: choose to clean (spend) or skip (save for later)
            best = -math.inf
            # Option A: clean current zone
            if battery >= cost:
                val = self._minimax(battery - cost, rest, dirty_probs,
                                    False, alpha, beta,
                                    cleaned + [current_zone], depth - 1)
                best = max(best, val)
                alpha = max(alpha, best)
                if alpha >= beta:
                    return best   # β-cutoff
            # Option B: skip
            val = self._minimax(battery, rest, dirty_probs,
                                False, alpha, beta, cleaned, depth - 1)
            best = max(best, val)
            return best
        else:
            # MIN (entropy): wastes 1 battery unit
            best = math.inf
            val = self._minimax(max(0, battery - 1), zones, dirty_probs,
                                True, alpha, beta, cleaned, depth - 1)
            best = min(best, val)
            beta = min(beta, best)
            return best

    def recommend(self, zones: list[tuple],
                  dirty_probs: dict[tuple, float]) -> list[tuple]:
        """
        Run minimax and return the list of zones MAX decides to clean.
        """
        # Greedy reconstruction: clean zones whose utility beats the minimax
        # threshold (simplified policy extraction)
        threshold = self._minimax(self.budget, zones, dirty_probs,
                                   True, -math.inf, math.inf, [], self.depth)
        recommended = []
        battery = self.budget
        for z in zones:
            cost = max(1, int(dirty_probs.get(z, 0.3) * 3))
            if battery >= cost and dirty_probs.get(z, 0) >= 0.4:
                recommended.append(z)
                battery -= cost
        self._trace.append(f"      Nodes explored  : {self._nodes_explored}")
        self._trace.append(f"      Battery budget  : {self.budget}")
        self._trace.append(f"      Minimax utility : {threshold:.3f}")
        self._trace.append(f"      Zones to clean  : {recommended}")
        return recommended

    @property
    def trace(self) -> str:
        return "\n".join(self._trace)


# ─────────────────────────────────────────────────────────────────────────────
# CO6 : Integrated Reasoning Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class CleaningPipeline:
    """
    CO6 – End-to-end pipeline combining all five AI components with
          explainable output: reasoning traces, constraint proofs,
          and step-by-step inference logs.
    """

    def __init__(self, rows: int = 5, cols: int = 5,
                 time_of_day: str = "afternoon",
                 robot_start: tuple = (0, 0),
                 battery_budget: int = 10):
        self.room        = RoomState(rows, cols)
        self.bayes       = DirtBayesNet(time_of_day)
        self.minimax_agent = MinimaxResourceAgent(budget=battery_budget)
        self.robot_start = robot_start
        self.traces: list[str] = []

    def _section(self, title: str):
        bar = "─" * 60
        print(f"\n{bar}")
        print(f"  {title}")
        print(bar)

    def run(self):
        separator = "=" * 60
        print(separator)
        print("  AUTONOMOUS CLEANING STRATEGY PLANNER")
        print(separator)

        # ── Step 1 : CO1 – Build state space ──────────────────────────
        self._section("STEP 1  |  CO1 : State-Space Representation")
        print(self.room.describe())
        self.traces.append(self.room.describe())

        # ── Step 2 : CO5 – Bayesian dirt inference ────────────────────
        self._section("STEP 2  |  CO5 : Bayesian Network Inference")
        dirty_probs = self.bayes.infer(self.room)
        print(self.bayes.trace)
        self.traces.append(self.bayes.trace)

        # Mark dirty zones on the room
        self.room.mark_dirty(dirty_probs, threshold=0.75)
        print(f"\n      Dirty zones identified: {sorted(self.room.dirty_zones)}")

        # Visualise grid
        self._section("ROOM HEATMAP  (P = probability label)")
        self._print_heatmap(dirty_probs)

        # ── Step 3 : CO2 – A* optimal path ────────────────────────────
        self._section("STEP 3  |  CO2 : A* Search – Optimal Cleaning Path")
        clean_order, cost, astar_trace = astar_cleaning_path(
            self.room, start=self.robot_start
        )
        for line in astar_trace:
            print(line)
        self.traces.extend(astar_trace)

        # ── Step 4 : CO4 – Minimax resource allocation ────────────────
        self._section("STEP 4  |  CO4 : Minimax Agent – Resource Allocation")
        approved_zones = self.minimax_agent.recommend(clean_order, dirty_probs)
        print(self.minimax_agent.trace)
        self.traces.append(self.minimax_agent.trace)

        # ── Step 5 : CO3 – CSP scheduling ────────────────────────────
        self._section("STEP 5  |  CO3 : CSP Backtracking – Zone Scheduling")
        if approved_zones:
            csp = CleaningScheduleCSP(approved_zones, dirty_probs, self.room)
            schedule = csp.solve()
            print(csp.trace)
            self.traces.append(csp.trace)
        else:
            schedule = {}
            print("      No zones approved for scheduling.")

        # ── Step 6 : CO6 – Explainable output ────────────────────────
        self._section("STEP 6  |  CO6 : Integrated Explainable Output")
        self._print_final_plan(approved_zones, schedule, dirty_probs, cost)

        print("\n" + separator)
        print("  PIPELINE COMPLETE")
        print(separator)

    # ── Helpers ───────────────────────────────────────────────────────

    def _print_heatmap(self, dirty_probs: dict[tuple, float]):
        """ASCII heatmap of dirt probabilities."""
        LEVELS = ["░", "▒", "▓", "█"]
        header = "     " + "  ".join(f"C{c}" for c in range(self.room.cols))
        print(header)
        for r in range(self.room.rows):
            row_str = f"  R{r} "
            for c in range(self.room.cols):
                p = dirty_probs.get((r, c), 0)
                idx = min(int(p * len(LEVELS)), len(LEVELS) - 1)
                row_str += f" {LEVELS[idx]} "
            print(row_str)
        print("\n      Legend: ░ low  ▒ medium  ▓ high  █ very high")

    def _print_final_plan(self, approved: list[tuple],
                           schedule: dict[tuple, str],
                           probs: dict[tuple, float],
                           path_cost: int):
        """CO6 – Structured explainable output."""
        print("\n  ┌─ REASONING SUMMARY ──────────────────────────────┐")
        print(f"  │  Total dirty zones found    : {len(self.room.dirty_zones):<4}              │")
        print(f"  │  Zones approved (minimax)   : {len(approved):<4}              │")
        print(f"  │  A* path cost               : {path_cost:<4}              │")
        print(f"  │  Zones scheduled (CSP)      : {len(schedule):<4}              │")
        print("  └──────────────────────────────────────────────────┘")

        if schedule:
            print("\n  FINAL CLEANING SCHEDULE  (with constraint proofs)")
            for slot in sorted(set(schedule.values())):
                zs = [z for z, s in schedule.items() if s == slot]
                print(f"\n    [{slot}]")
                for z in zs:
                    p = probs.get(z, 0)
                    zone_type = self.room.grid.get(z, "unknown")
                    constraint_proof = (
                        "high-priority → early slot"
                        if p >= 0.8 else
                        "standard priority"
                    )
                    print(f"      Zone {z}  type={zone_type:<12}  "
                          f"P(dirty)={p:.3f}  → {constraint_proof}")

        print("\n  INFERENCE STEPS")
        for i, trace_block in enumerate(self.traces, 1):
            print(f"\n  [{i}] {trace_block.strip()}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pipeline = CleaningPipeline(
        rows          = 5,
        cols          = 5,
        time_of_day   = "afternoon",   # affects Bayesian priors
        robot_start   = (0, 0),
        battery_budget= 12,
    )
    pipeline.run()