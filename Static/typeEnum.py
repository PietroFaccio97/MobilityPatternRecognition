from enum import Enum

class Paths(Enum):
    bus = './Dataset/bus/*.csv'
    car = './Dataset/car/*.csv'
    pedestrian = './Dataset/pedestrian/*.csv'
    static = './Dataset/static/*.csv'
    train = './Dataset/train/*.csv'