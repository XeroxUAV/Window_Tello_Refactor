"""Generic PID controller — pure, framework-free."""

import time

import numpy as np


class PID:
    def __init__(self, Kp, Ki, Kd, integral_limit=100.0, output_smoothing=0.3):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral_limit = integral_limit
        self.output_smoothing = output_smoothing

        self.prev_error = 0.0
        self.integral = 0.0
        self.last_output = 0.0
        self.last_time = time.perf_counter()

    def update(self, error: float) -> float:
        now = time.perf_counter()
        dt = max(now - self.last_time, 0.001)
        self.last_time = now

        self.integral += error * dt
        self.integral = float(
            np.clip(self.integral, -self.integral_limit, self.integral_limit)
        )

        derivative = (error - self.prev_error) / dt
        raw = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        alpha = self.output_smoothing
        output = (1.0 - alpha) * self.last_output + alpha * raw
        self.prev_error = error
        self.last_output = output
        return output

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_output = 0.0
        self.last_time = time.perf_counter()