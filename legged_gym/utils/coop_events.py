"""Physics-clock contact debouncing and guarded contact mismatch detection."""

import numpy as np


class ContactEvents:
    def __init__(self, n):
        self.contact = np.zeros((n, 4), dtype=bool)
        self.candidate = self.contact.copy()
        self.stable = np.zeros((n, 4))
        self.since_transition = np.zeros((n, 4))
        self.early_time = np.zeros((n, 4))
        self.late_time = np.zeros((n, 4))
        self.early = np.zeros((n, 4))
        self.late = np.zeros((n, 4))
        self.guard = np.full(n, 0.20)

    def reset(self, ids, guard=0.20):
        for name, value in vars(self).items():
            value[ids] = guard if name == "guard" else 0

    def update(self, raw, phase, dt, active):
        ids = np.flatnonzero(active)
        old = self.contact[ids].copy()
        same = raw[ids] == self.candidate[ids]
        self.stable[ids] = np.where(same, self.stable[ids] + dt, dt)
        self.candidate[ids] = raw[ids]
        accepted = self.stable[ids] >= np.where(raw[ids], 0.003, 0.005) - 1e-12
        self.contact[ids] = np.where(accepted, raw[ids], old)
        self.since_transition[ids] = np.where(self.contact[ids] != old, 0, self.since_transition[ids] + dt)
        self.guard[ids] = np.maximum(0, self.guard[ids] - dt)
        sine = np.sin(phase[ids])
        valid = (np.abs(sine) >= np.sin(2 * np.pi * 0.05)) & (self.guard[ids, None] <= 1e-12)
        early = valid & (sine > 0) & self.contact[ids]
        late = valid & (sine <= 0) & ~self.contact[ids]
        self.early_time[ids] = np.where(early, self.early_time[ids] + dt, 0)
        self.late_time[ids] = np.where(late, self.late_time[ids] + dt, 0)
        self.early[ids] = self.early_time[ids] >= 0.005 - 1e-12
        self.late[ids] = self.late_time[ids] >= 0.015 - 1e-12
