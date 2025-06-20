#%%
# Code to generate payment options and save to excel
import pandas as pd
import numpy as np
from collections import defaultdict

def generate_payment_options():
    # Generate payment options
    game_definitions = pd.read_excel("game_definitions.xlsx", sheet_name=None)
    _route_cost = game_definitions["route_cost"]
    col_cols = ["black", "blue", "green", "orange", "purple", "red", "yellow", "white" , "gay"]

    payments = []
    route_ids = []
    for index, route_cost in _route_cost.iterrows():
        length, color, gay_cards = route_cost["length"], route_cost["color"], route_cost["gay_cards"]

        if gay_cards > 0:
            # Grey route with gay cards needed, first the case of paying for full with gay cards
            temp = np.zeros((1, 9))
            temp[0, 8] = length
            
            payments.append(temp)
            route_ids.extend([index] * len(temp))

            # Case swap additional normal cards for gay cards
            for i in range(gay_cards, length):
                temp = np.identity(8) * (length - i)
                temp = np.concat([temp, np.full(shape=(8, 1), fill_value=i)], axis=1)

                if i == gay_cards:
                    # Case swap a gay card for two of the same color
                    # Swap 1 gay card for two of the same
                    for row in temp:
                        temptemp = np.identity(8) * 2
                        temptemp = np.concat([temptemp, np.full(shape=(8, 1), fill_value=gay_cards - 1)], axis=1)
                        temptemp[:, :-1] += row[:-1] # Exclude gay card count

                        # Swap another gay card for two of the same
                        if gay_cards > 1:
                            for row2 in temptemp:
                                temptemptemp = np.identity(8) * 2
                                temptemptemp = np.concat([temptemptemp, np.full(shape=(8, 1), fill_value=gay_cards - 2)], axis=1)
                                temptemptemp[:, :-1] += row2[:-1]
                                
                                payments.append(temptemptemp)
                                route_ids.extend([index] * len(temp))

                        payments.append(temptemp)
                        route_ids.extend([index] * len(temp))

                payments.append(temp)
                route_ids.extend([index] * len(temp))

        else:
            # Normal route
            # First the option where the route is fully paid with the right color cards
            temp = np.zeros((1, 9))
            temp[0, color] = length
            
            payments.append(temp)
            route_ids.extend([index] * len(temp))

            for i in range(1, length + 1):
                # Add a row where an increasing number of cards are swapped for gay cards
                temp = np.zeros((1, 9))
                temp[0, color] = length - i
                temp[0, 8] = i

                payments.append(temp)
                route_ids.extend([index] * len(temp))

    # Turn into dataframe and add route ids
    total_payments = pd.DataFrame(np.vstack(payments), columns=col_cols)
    total_payments["route_id"] = route_ids

    # Drop duplicates and group route ids into lists
    grouped = total_payments.groupby(col_cols, as_index=False).agg({"route_id": list})
    grouped["route_id"] = grouped["route_id"].apply(lambda x: sorted(set(x)))

    # Invert to add payment ids to route cost table
    route_to_payment = defaultdict(list)

    for payment_id, route_ids in zip(grouped.index, grouped["route_id"]):
        for route_id in route_ids:
            route_to_payment[route_id].append(payment_id)

    _route_cost["payment_ids"] = _route_cost.index.map(lambda idx: route_to_payment.get(idx, []))

    # Write to excel
    with pd.ExcelWriter("game_definitions.xlsx", mode="a", if_sheet_exists="replace") as writer:
        grouped.to_excel(writer, sheet_name="route_cost_payments", index=False)
        _route_cost.to_excel(writer, sheet_name="route_cost", index=False)

#%%
# Partial code of previous _can_route_be_claimed function
route_length = self._routes[route_id, LENGTH_INDEX]

# Not yet claimed, proceed to check
if self._routes[route_id, GAY_CARDS_INDEX] == 0:
    # No required gay card, route color is not gray
    route_color = self._routes[route_id, COLOR_INDEX]

    if player_hand[route_color] >= route_length:
        # Player has enough cards of the right color
        return True
    else:
        # Check if the player has enough gay cards to compensate
        n_missing = route_length - player_hand[route_color]

        if player_hand[-1] >= n_missing:
            return True # Can compensate with gay cards
        else:
            return False # Not enough cards
else:
    # Ferry route
    n_gay_cards = self._routes[route_id, GAY_CARDS_INDEX]
    n_non_gay = route_length - n_gay_cards

    bool_enough_cards = player_hand[:] >= n_non_gay

    if np.any(bool_enough_cards):
        # Enough non-gay cards
        if player_hand[-1] >= n_gay_cards:
            # Enough gay cards
            if not np.any(bool_enough_cards[:-1]) and (player_hand[-1] < route_length):
                # Fully dependent on gay cards, but don't have enough to cover whole length
                return False
            else:
                return True # Got enough!
        else:
            # Not enough gay cards, but may be compensated by swapping for two of the same color
            n_missing = n_gay_cards - player_hand[-1]

            # Loop over the options for paying the non-gay cards
            for non_gay_card_paid in np.where(bool_enough_cards)[0]:
                # Get hand after paying non-gay card cost
                temp_hand = player_hand[:].copy()
                temp_hand[non_gay_card_paid] -= n_non_gay

                if np.sum(temp_hand[:-1] // 2) >= n_missing:
                    # Still enough pairs of cards exist
                    return True
                
            return False # None of the options worked
    else:
        return False # Not enough non-gay cards