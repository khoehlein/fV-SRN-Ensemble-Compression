from enum import Enum


class FlowDirection(Enum):
    FORWARD = 'forward'
    REVERSE = 'reverse'

    def opposite(self):
        return FlowDirection.REVERSE if self == FlowDirection.FORWARD else FlowDirection.FORWARD
