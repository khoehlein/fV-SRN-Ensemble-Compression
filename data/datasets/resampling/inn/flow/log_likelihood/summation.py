
class HierarchicalSum(object):

    def __init__(self):
        self._values = []

    def reset(self):
        self._values = []

    def _set_value(self, index, value):
        while index >= len(self._values):
            self._values.append(None)
        self._values[index] = value

    def add(self, item):
        index = 0
        while index < len(self._values) and self._values[index] is not None:
            item = self._values[index] + item
            self._values[index] = None
            index = index + 1
        if index < len(self._values):
            self._values[index] = item
        else:
            self._values.append(item)

    def merge(self, other: 'HierarchicalSum') -> 'HierarchicalSum':
        out = HierarchicalSum()
        out.add(self.value())
        out.add(other.value())
        return out

    def value(self):
        sum = None
        for x in self._values:
            if x is not None:
                sum = x if sum is None else sum + x
        return sum


class _HierarchicalSum(object):

    def __init__(self):
        self._values = None

    def reset(self):
        self._values = None

    def add(self, item):
        if self._values is None:
            self._values = item
        else:
            self._values = self._values + item

    def merge(self, other: 'HierarchicalSum') -> 'HierarchicalSum':
        out = HierarchicalSum()
        out.add(self.value())
        out.add(other.value())
        return out

    def value(self):
        return self._values