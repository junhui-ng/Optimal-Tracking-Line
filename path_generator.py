import numpy as np
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from path import Path, Point
from utils import random_points_in_polygon, relative_position_in_radians


@dataclass
class State:
    """
    Hidden state of points, used in step functions.
    """
    t: int  # time
    x: int  # current x location
    y: int  # current y location
    u: int  # previous horizontal movement
    v: int  # previous vertical movement
    d: float    # previous displacement
    r: float    # previous moving angle (degree)

@dataclass
class RandomStepRange:
    """
    Range of random values, required in StepFunction init method
    """
    r_min: int  # lower bound of rotation angle (in degree)
    r_max: int  # upper bound of rotation angle (in degree)
    d_min: int  # lower bound of displacement (in pixels)
    d_max: int  # upper bound of displacement (in pixels)

class StepFunction(ABC):
    """
    Abstract class for step functions.
    Implement the step() abstract method in child classes.
    """
    def __init__(self, randomness: RandomStepRange):
        self.randomness: RandomStepRange = randomness

    @abstractmethod
    def step(self, point: Point, state: State, t: int=None) -> Point:
        """
        Abstract method.
        Takes in Point and State from previous timestep and returns a simulated Point instance.
        State should be updated inplace.
        """
        raise NotImplementedError("Abstract method StepFunction.step() not implemented in child class.")

def generate_paths(sample_size: int, 
                   iterations: int,
                   step_fn: StepFunction,
                   initial_state: Dict[str, Any]=None,
                   starting_roi: np.ndarray=None,
                   destination_roi: np.ndarray=None,
                   frame_width: int=1920, 
                   frame_height: int=1080) -> List[Path]:
    """
    Path generator

    Args:
    """
    # initialize paths
    paths: List[Path] = [Path(id=id) for id in range(sample_size)]

    # if starting roi is not specified, use the bottom 20% and center 50% of the frame as the starting point
    if starting_roi is None:
        min_x: int = int(0.25 * frame_width)
        max_x: int = int(0.75 * frame_width)
        min_y: int = int(0.8 * frame_height)
        max_y: int = int(0.99 * frame_height)
        starting_roi = np.array([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]])
    # if destination_roi is not specified, use the top 30% of the frame as the destination roi
    if destination_roi is None:
        min_x: int = 0
        max_x: int = frame_width
        min_y: int = 0
        max_y: int = int(0.3 * frame_height)
        destination_roi = np.array([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]])

    # initialize starting positions
    initial_positions: List[Tuple[int, int]] = random_points_in_polygon(starting_roi, k=sample_size)
    # compute general moving direction based on starting and destination ROIs
    initial_rotation: float = relative_position_in_radians(anchor=starting_roi, target=destination_roi)

    # initialize states and add initial positions to path
    states: List[State] = list()
    for path, (x, y) in zip(paths, initial_positions):
        state = State(t=0, x=x, y=y, u=0, v=0, d=0, r=initial_rotation)

        # update states if provided
        if initial_state is not None:
            for attr, value in initial_state.items():
                if hasattr(state, attr):
                    setattr(state, attr, value)

        path.add(location=(state.x, state.y), t=state.t)
        states.append(state)

    # simulation
    for _ in range(1, iterations):
        for path, state in zip(paths, states):
            new_point: Point = step_fn.step(point=path.last_point, state=state)
            path.add(point=new_point)

    return paths


"""
Step functions:
- Random: Each object moves from starting point to target point with small random deviation along the path
- Dispersion
- Convergence
"""

class Random(StepFunction):
    def step(self, point: Point, state: State, t: int=None) -> Point:
        # auto increment timestep if not specified
        if t is None:
            t = point.t + 1

        # random rotation
        angle_deviation = np.radians(np.random.uniform(self.randomness.r_min, self.randomness.r_max))
        current_angle = state.r + angle_deviation

        # random distance
        delta_d = random.randint(self.randomness.d_min, self.randomness.d_max)
        dx = delta_d * np.cos(current_angle)
        dy = delta_d * np.sin(current_angle)

        # current position
        current_x = point.x + dx
        current_y = point.y + dy

        # update state
        state.t = t
        state.x = current_x
        state.y = current_y
        state.u = dx
        state.v = dy
        state.d = delta_d
        state.r = current_angle

        return Point(track_id=point.track_id, x=current_x, y=current_y, t=t)


"""
def sharp_turn(i: int, 
                previous_location: Tuple[int, int], 
                previous_movement: Tuple[int, int]) -> Tuple[int, int]:
    previous_angle = np.arctan2(previous_movement[1], previous_movement[0])
    if i == 50:
        angle_deviation = np.radians(np.random.uniform(85, 95))
    else:
        angle_deviation = np.radians(np.random.uniform(-1, 1))
    new_angle = previous_angle + angle_deviation

    delta_d = random.randint(2, 10)
    dx = delta_d * np.cos(new_angle)
    dy = delta_d * np.sin(new_angle)

    return dx, dy


def sharp_turn(state: State, randomness: RandomStepRange) -> State:
    previous_angle = state.r
    if state.t == 50:
        angle_deviation = np.radians(np.random.uniform(85, 95))
    else:
        angle_deviation = np.radians(np.random.uniform(-1, 1))
    new_angle = previous_angle + angle_deviation

    delta_d = random.randint(2, 10)
    dx = delta_d * np.cos(new_angle)
    dy = delta_d * np.sin(new_angle)

    return dx, dy


def dispersion(i: int, 
               previous_location: Tuple[int, int], 
               previous_movement: Tuple[int, int]) -> Tuple[int, int]:
    previous_angle = np.arctan2(previous_movement[1], previous_movement[0])
    angle_deviation = np.radians(np.random.uniform(-0.3*i, 0.3*i))
    new_angle = previous_angle + angle_deviation

    delta_d = random.randint(2, 10)
    dx = delta_d * np.cos(new_angle)
    dy = delta_d * np.sin(new_angle)

    return dx, dy
"""
