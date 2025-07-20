from typing import Any
from abc import ABC, abstractmethod
from functools import cached_property


class AutoController(ABC):
    """
    Abstract base class for all auto controllers.

    Attributes:
        show_camera (bool): If True, the camera view will be displayed to visualize
                            the actions of the auto controller within the environment.
    """

    # Class-level registry dictionary
    registry = {}

    def __init__(self, show_camera: bool = False):
        """
        Initialize the AutoController.

        Args:
            show_camera (bool): Whether to display the camera view showing the
                                actions of the auto controller inside the environment.
                                Defaults to False.
        """
        super().__init__()
        self.show_camera = show_camera  # Display camera view of the controller's actions

    @abstractmethod
    def get_cameras(self) -> dict[str, Any]:
        """
        Retrieve information or handles to the cameras used by the controller.

        This method should return a dictionary mapping camera names to either
        image-producing objects, stream URLs, or relevant metadata.

        Returns:
            dict[str, Any]: A dictionary containing information about available cameras.

        Raises:
            NotImplementedError: If the subclass does not support camera access.
        """
        raise NotImplementedError("Subclasses must implement the 'get_cameras' method.")

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """
        A dictionary describing the structure and types of the observations produced by the robot.
        Its structure (keys) should match the structure of what is returned by :meth:`get_observation`.
        Values for the dict should either be:
            - The type of the value if it's a simple value, e.g. `float` for single proprioceptive value (a joint's position/velocity)
            - A tuple representing the shape if it's an array-type value, e.g. `(height, width, channel)` for images

        Note: this property should be able to be called regardless of whether the robot is connected or not.
        """
        raise NotImplementedError(
            "Subclasses must implement the 'observation_features' cached property."
        )

    @cached_property
    def action_features(self) -> dict[str, type]:
        """
        A dictionary describing the structure and types of the actions expected by the robot.
        Its structure (keys) should match the structure of what is passed to :meth:`send_action`.
        Values for the dict should be the type of the value if it's a simple value,
        e.g. `float` for single proprioceptive value (a joint's goal position/velocity)

        Note: this property should be able to be called regardless of whether the robot is connected or not.
        """
        raise NotImplementedError(
            "Subclasses must implement the 'action_features' cached property."
        )

    @abstractmethod
    def reset(self):
        """
        Reset the auto controller to its initial state.
        """
        pass

    @abstractmethod
    def get_observation(self) -> dict[str, Any]:
        """
        Retrieve the current observation from the environment.

        Returns:
            dict[str, Any]: A dictionary representing the current observation,
                            typically including sensor readings, camera images,
                            or internal state.
        """
        pass

    @abstractmethod
    def get_action(self):
        """
        Compute and return the next action.

        Returns:
            The computed action.
        """
        pass

    @classmethod
    def register_subclass(cls, name):
        """
        Class decorator to register a subclass with a given name.

        Usage:
            @AutoController.register_subclass("my_controller")
            class MyController(AutoController):
                ...

        Args:
            name (str): Name to register the subclass under.

        Returns:
            decorator function.
        """
        def decorator(subclass):
            cls.registry[name] = subclass
            return subclass
        return decorator

    @classmethod
    def create(cls, name, *args, **kwargs):
        """
        Factory method to create an instance of a registered subclass by name.

        Args:
            name (str): The registered subclass name.
            *args, **kwargs: Arguments to pass to the subclass constructor.

        Returns:
            An instance of the requested subclass.

        Raises:
            ValueError: If no subclass with the given name is registered.
        """
        if name not in cls.registry:
            raise ValueError(f"AutoController subclass '{name}' is not registered.")
        return cls.registry[name](*args, **kwargs)

    def close(self):
        """
        Perform any cleanup or resource release needed by the controller.

        This might include closing camera windows, freeing hardware resources,
        or shutting down background threads.

        The base implementation does nothing and should be overridden by subclasses
        that require specific cleanup.
        """
        pass
