# Created by Hansi at 28/04/2023

import re


def get_indices(text, phrase):
    for match in re.finditer(phrase, text):
        print(match.start(), match.end())


if __name__ == '__main__':
    text = "In a mechanical ventilation system, the air handling units and chambers shall be able to withstand the loads caused by fan pressure while the shutoff dampers are closed."
    phrase = "mechanical"
    get_indices(text, phrase)
