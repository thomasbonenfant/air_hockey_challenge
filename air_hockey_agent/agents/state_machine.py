
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
            else:
                next_state = previous_state  # todo should it go home?
                # next_state = 'home'

        return next_state
