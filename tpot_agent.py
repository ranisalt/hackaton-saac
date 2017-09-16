import numpy as np
from tpot import TPOTRegressor
from tpot.config import regressor_config_dict_light

from agent import SillyWalker, Action


if __name__ == '__main__':
    walker = SillyWalker()

    for _ in range(10):
        while not walker.done:
            walker.step()
        walker.reset()

    def scoring(y_real, y_predicted):
        return sum(y_predicted)[-1] / (len(y_predicted) - 1)

    for i in range(10):
        print('#' * 80)
        print(f'# GENERATION {i + 1}')
        print('#' * 80)
        x = np.array(walker.state_history[:-1])
        y = np.array([list(a) + [r] for a, r in zip(walker.action_history,
                                                    walker.reward_history)])

        walker.save_history(f'sillywalker{i+1}')
        model = TPOTRegressor(generations=5, population_size=20,
                              scoring=scoring, verbosity=2,
                              config_dict=regressor_config_dict_light)
        model.fit(x, y)
        for _ in range(10):
            while not walker.done:
                s = walker.state
                prediction = model.predict(np.array([s]))[0]
                print(prediction)

                action = Action(*prediction[:-1])
                walker.step(action)

            walker.reset()
