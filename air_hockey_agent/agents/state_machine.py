
class StateMachine:
    def __init__(self):
        self.states = ['home', 'hit', 'defend', 'repel', 'prepare']


    # TODO integrate the pick task

    def select_task(self, previous_state, desired_next_state):
        """
        Select the next task based on the current one.

        Transitions allowed are in the form 'cur_state --> next_state' where next_state is either
        cur_state or home_state.

        ex:
        hit --> home, allowed
        hit --> hit, allowed
        hit --> prepare, not allowed will be forced to hit

        previous_state: the previous task
        desired_next_state: the task picked from the select_task function, it won't necessary
                            be allowed
        """

        next_state = desired_next_state

        '''
        # Self-rings
        if desired_next_state == previous_state:
            next_state = desired_next_state

        # Allowed transactions
        else:
            # if not going home keep doing the same task
            if desired_next_state != 'home' and previous_state != 'home':
                next_state = previous_state
        '''

        if previous_state != 'home':
            if desired_next_state == previous_state or desired_next_state == 'home':
                next_state = desired_next_state
            elif previous_state == "defend" and desired_next_state != "home": # FIXME problems with defend instead of hit?
                next_state = "home"
            else:
                next_state = previous_state
                # next_state = 'home' # going home increases a lot the violated constraints

        return next_state

    def explicit_select_state(self, previous_state, desired_next_state):
        # explicitly select the transitions
        if previous_state == "hit":
            if desired_next_state == "hit" or desired_next_state == "home":
                next_state = desired_next_state
            else:
                next_state = previous_state
        elif previous_state == "defend":
            if desired_next_state == "defend" or desired_next_state == "home":
                next_state = desired_next_state
            else:
                next_state = previous_state
        elif previous_state == "repel":
            if desired_next_state == "repel" or desired_next_state == "home":
                next_state = desired_next_state
            else:
                next_state = previous_state
        elif previous_state == "prepare":
            if desired_next_state == "prepare" or desired_next_state == "home":
                next_state = desired_next_state
            else:
                next_state = previous_state
        elif previous_state == "home":
            if desired_next_state != "home":
                next_state = desired_next_state
            else:
                next_state = previous_state
        return next_state
