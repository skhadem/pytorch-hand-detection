from data_structures import BBox

if __name__ == '__main__':
    reg = BBox()

    reg.add_region((0,0), 1)
    reg.add_region((0,1), 1)
    reg.add_region((1,0), 1)
    reg.add_region((1,1), 1)
    reg.add_region((-2,-2), 1)
    reg.add_region((4,4),2)
    reg.add_region((0,-4),1)


    print(reg)
