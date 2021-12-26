class Timeout(object):
    def __init__(self):
        pass

    def __str__(self):
        return 'Timeout'


class ReachGoal(object):
    def __init__(self):
        pass

    def __str__(self):
        return 'Reaching Goal'


class Discomfort(object):
    def __init__(self):
        pass

    def __str__(self):
        return 'Too Close'

class Lost(object):

    def __init__(self):
        pass

    def __str__(self):
        return 'Too Far'

class Collision(object):
    def __init__(self):
        pass

    def __str__(self):
        return 'Collision'


class Nothing(object):
    def __init__(self):
        pass

    def __str__(self):
        return ''