import matplotlib.pyplot as plt

from utils.agents import QSheep, QWolves
from utils.board import BaghChal


game_env = BaghChal(QSheep, QWolves)

history = {
    "rounds": [],
    "wolves": 0,
    "sheep": 0,
    "captured": [],
}
epochs = 1000

for epoch in range(epochs):

    _round = 1
    while True:
        # Sheep turn:
        new_state = game_env.sheep.make_turn(game_env.get_state())
        if new_state is None:
            game_env.sheep.update_q_from_trajectory(-1)
            game_env.wolves.update_q_from_trajectory(1)
            history["wolves"] += 1
            break
        game_env.step(new_state)

        # Wolves turn
        new_state = game_env.wolves.make_turn(game_env.get_state())
        if new_state is None:
            game_env.sheep.update_q_from_trajectory(1)
            game_env.wolves.update_q_from_trajectory(-1)
            history["sheep"] += 1
            break
        game_env.step(new_state)

        _round += 1
    history["captured"].append(game_env.wolves.captured_sheep)
    history["rounds"].append(_round)
    game_env.restart()


fig, axs = plt.subplots(1, 2, figsize=(15, 7))
fig.suptitle(f"W: {history['wolves']}, S: {history['sheep']}")

axs[0].plot(history["rounds"], label='rounds/match')
axs[0].legend()
axs[1].plot(history["captured"], label="captured/match")
axs[1].legend()
# fig.legend()
plt.show()
